use itertools::Itertools;
use std::ops::Mul;

use ndarray::parallel::prelude::*;
use ndarray::s;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use ndarray::Axis;
use num_traits::One;
use num_traits::Zero;
use strum::EnumCount;
use strum_macros::Display;
use strum_macros::EnumCount as EnumCountMacro;
use strum_macros::EnumIter;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use triton_opcodes::instruction::Instruction;
use U32TableChallengeId::*;

use crate::table::challenges::TableChallenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::ConstraintCircuitMonad;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::DualRowIndicator::*;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::*;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::PermArg;
use crate::table::master_table::NUM_BASE_COLUMNS;
use crate::table::master_table::NUM_EXT_COLUMNS;
use crate::table::table_column::BaseTableColumn;
use crate::table::table_column::ExtTableColumn;
use crate::table::table_column::MasterBaseTableColumn;
use crate::table::table_column::MasterExtTableColumn;
use crate::table::table_column::U32BaseTableColumn;
use crate::table::table_column::U32BaseTableColumn::*;
use crate::table::table_column::U32ExtTableColumn;
use crate::table::table_column::U32ExtTableColumn::*;
use crate::vm::AlgebraicExecutionTrace;

pub const U32_TABLE_NUM_PERMUTATION_ARGUMENTS: usize = 2;
pub const U32_TABLE_NUM_EVALUATION_ARGUMENTS: usize = 0;
pub const U32_TABLE_NUM_EXTENSION_CHALLENGES: usize = U32TableChallengeId::COUNT;

pub const BASE_WIDTH: usize = U32BaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = U32ExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

pub struct U32Table {}

pub struct ExtU32Table {}

impl ExtU32Table {
    pub fn ext_initial_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            U32TableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let challenge = |c| circuit_builder.challenge(c);
        let one = circuit_builder.b_constant(1_u32.into());

        let copy_flag = circuit_builder.input(BaseRow(CopyFlag.master_base_table_index()));
        let lhs = circuit_builder.input(BaseRow(LHS.master_base_table_index()));
        let rhs = circuit_builder.input(BaseRow(RHS.master_base_table_index()));
        let ci = circuit_builder.input(BaseRow(CI.master_base_table_index()));

        let rp = circuit_builder.input(ExtRow(ProcessorPermArg.master_ext_table_index()));

        let normalized_instruction_deselector = |instruction_to_select: Instruction| {
            let instructions_to_deselect = [
                Instruction::Lt,
                Instruction::And,
                Instruction::Xor,
                Instruction::Log2Floor,
                Instruction::Pow,
                Instruction::Div,
            ]
            .into_iter()
            .filter(|&instruction| instruction != instruction_to_select)
            .collect_vec();

            let deselect_0 = ci.clone();
            let deselector = instructions_to_deselect
                .iter()
                .map(|&instruction| ci.clone() - circuit_builder.b_constant(instruction.opcode_b()))
                .fold(deselect_0, ConstraintCircuitMonad::mul);
            let normalize_deselected_0 = instruction_to_select.opcode_b();
            let normalizing_factor = instructions_to_deselect
                .iter()
                .fold(normalize_deselected_0, |acc, instr| {
                    acc * (instruction_to_select.opcode_b() - instr.opcode_b())
                })
                .inverse();
            circuit_builder.b_constant(normalizing_factor) * deselector
        };

        let result = normalized_instruction_deselector(Instruction::Lt)
            * circuit_builder.input(BaseRow(LT.master_base_table_index()))
            + normalized_instruction_deselector(Instruction::And)
                * circuit_builder.input(BaseRow(AND.master_base_table_index()))
            + normalized_instruction_deselector(Instruction::Xor)
                * circuit_builder.input(BaseRow(XOR.master_base_table_index()))
            + normalized_instruction_deselector(Instruction::Log2Floor)
                * circuit_builder.input(BaseRow(Log2Floor.master_base_table_index()))
            + normalized_instruction_deselector(Instruction::Pow)
                * circuit_builder.input(BaseRow(Pow.master_base_table_index()))
            + normalized_instruction_deselector(Instruction::Div)
                * circuit_builder.input(BaseRow(LT.master_base_table_index()));

        let initial_factor = challenge(ProcessorPermIndeterminate)
            - challenge(LhsWeight) * lhs
            - challenge(RhsWeight) * rhs
            - challenge(CIWeight) * ci
            - challenge(ResultWeight) * result;
        let if_copy_flag_is_1_then_rp_has_accumulated_first_row =
            copy_flag.clone() * (rp.clone() - initial_factor);

        let default_initial = circuit_builder.x_constant(PermArg::default_initial());
        let if_copy_flag_is_0_then_rp_is_default_initial =
            (one - copy_flag) * (default_initial - rp);

        [
            if_copy_flag_is_0_then_rp_is_default_initial,
            if_copy_flag_is_1_then_rp_has_accumulated_first_row,
        ]
        .map(|circuit| circuit.consume())
        .to_vec()
    }

    pub fn ext_consistency_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            U32TableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let one = circuit_builder.b_constant(1_u32.into());

        let copy_flag = circuit_builder.input(BaseRow(CopyFlag.master_base_table_index()));
        let bits = circuit_builder.input(BaseRow(Bits.master_base_table_index()));
        let bits_minus_33_inv =
            circuit_builder.input(BaseRow(BitsMinus33Inv.master_base_table_index()));
        let lhs = circuit_builder.input(BaseRow(LHS.master_base_table_index()));
        let rhs = circuit_builder.input(BaseRow(RHS.master_base_table_index()));
        let lt = circuit_builder.input(BaseRow(LT.master_base_table_index()));
        let and = circuit_builder.input(BaseRow(AND.master_base_table_index()));
        let xor = circuit_builder.input(BaseRow(XOR.master_base_table_index()));
        let log2floor = circuit_builder.input(BaseRow(Log2Floor.master_base_table_index()));
        let lhs_copy = circuit_builder.input(BaseRow(LhsCopy.master_base_table_index()));
        let pow = circuit_builder.input(BaseRow(Pow.master_base_table_index()));
        let lhs_inv = circuit_builder.input(BaseRow(LhsInv.master_base_table_index()));
        let rhs_inv = circuit_builder.input(BaseRow(RhsInv.master_base_table_index()));

        let copy_flag_is_bit = copy_flag.clone() * (one.clone() - copy_flag.clone());
        let copy_flag_is_0_or_bits_is_0 = copy_flag.clone() * bits.clone();
        let bits_minus_33_inv_is_inverse_of_bits_minus_33 = one.clone()
            - bits_minus_33_inv * (bits - circuit_builder.b_constant(BFieldElement::new(33)));
        let lhs_inv_is_0_or_the_inverse_of_lhs =
            lhs_inv.clone() * (one.clone() - lhs.clone() * lhs_inv.clone());
        let lhs_is_0_or_lhs_inverse_is_the_inverse_of_lhs =
            lhs.clone() * (one.clone() - lhs.clone() * lhs_inv.clone());
        let rhs_inv_is_0_or_the_inverse_of_rhs =
            rhs_inv.clone() * (one.clone() - rhs.clone() * rhs_inv.clone());
        let rhs_is_0_or_rhs_inverse_is_the_inverse_of_rhs =
            rhs.clone() * (one.clone() - rhs.clone() * rhs_inv.clone());
        let copy_flag_is_0_or_lhs_copy_is_lhs = copy_flag.clone() * (lhs_copy - lhs.clone());
        let padding_row = (one.clone() - copy_flag)
            * (one.clone() - lhs.clone() * lhs_inv.clone())
            * (one.clone() - rhs * rhs_inv);
        let padding_row_or_lt_is_2 =
            padding_row.clone() * (lt - circuit_builder.b_constant(BFieldElement::new(2)));
        let padding_row_or_and_is_0 = padding_row.clone() * and;
        let padding_row_or_xor_is_0 = padding_row.clone() * xor;
        let padding_row_or_pow_is_1 = padding_row * (one.clone() - pow);
        let lhs_is_not_zero_or_log2floor_is_negative_1 =
            (one.clone() - lhs * lhs_inv) * (log2floor + one);

        [
            copy_flag_is_bit,
            copy_flag_is_0_or_bits_is_0,
            bits_minus_33_inv_is_inverse_of_bits_minus_33,
            lhs_inv_is_0_or_the_inverse_of_lhs,
            lhs_is_0_or_lhs_inverse_is_the_inverse_of_lhs,
            rhs_inv_is_0_or_the_inverse_of_rhs,
            rhs_is_0_or_rhs_inverse_is_the_inverse_of_rhs,
            copy_flag_is_0_or_lhs_copy_is_lhs,
            padding_row_or_lt_is_2,
            padding_row_or_and_is_0,
            padding_row_or_xor_is_0,
            padding_row_or_pow_is_1,
            lhs_is_not_zero_or_log2floor_is_negative_1,
        ]
        .map(|circuit| circuit.consume())
        .to_vec()
    }

    pub fn ext_transition_constraints_as_circuits() -> Vec<
        ConstraintCircuit<U32TableChallenges, DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>>,
    > {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let challenge = |c| circuit_builder.challenge(c);
        let one = circuit_builder.b_constant(1_u32.into());
        let two = circuit_builder.b_constant(2_u32.into());

        let copy_flag = circuit_builder.input(CurrentBaseRow(CopyFlag.master_base_table_index()));
        let bits = circuit_builder.input(CurrentBaseRow(Bits.master_base_table_index()));
        let ci = circuit_builder.input(CurrentBaseRow(CI.master_base_table_index()));
        let lhs = circuit_builder.input(CurrentBaseRow(LHS.master_base_table_index()));
        let rhs = circuit_builder.input(CurrentBaseRow(RHS.master_base_table_index()));
        let lt = circuit_builder.input(CurrentBaseRow(LT.master_base_table_index()));
        let and = circuit_builder.input(CurrentBaseRow(AND.master_base_table_index()));
        let xor = circuit_builder.input(CurrentBaseRow(XOR.master_base_table_index()));
        let log2floor = circuit_builder.input(CurrentBaseRow(Log2Floor.master_base_table_index()));
        let lhs_copy = circuit_builder.input(CurrentBaseRow(LhsCopy.master_base_table_index()));
        let pow = circuit_builder.input(CurrentBaseRow(Pow.master_base_table_index()));
        let rp = circuit_builder.input(CurrentExtRow(ProcessorPermArg.master_ext_table_index()));

        let copy_flag_next = circuit_builder.input(NextBaseRow(CopyFlag.master_base_table_index()));
        let bits_next = circuit_builder.input(NextBaseRow(Bits.master_base_table_index()));
        let ci_next = circuit_builder.input(NextBaseRow(CI.master_base_table_index()));
        let lhs_next = circuit_builder.input(NextBaseRow(LHS.master_base_table_index()));
        let rhs_next = circuit_builder.input(NextBaseRow(RHS.master_base_table_index()));
        let lt_next = circuit_builder.input(NextBaseRow(LT.master_base_table_index()));
        let and_next = circuit_builder.input(NextBaseRow(AND.master_base_table_index()));
        let xor_next = circuit_builder.input(NextBaseRow(XOR.master_base_table_index()));
        let log2floor_next =
            circuit_builder.input(NextBaseRow(Log2Floor.master_base_table_index()));
        let lhs_copy_next = circuit_builder.input(NextBaseRow(LhsCopy.master_base_table_index()));
        let pow_next = circuit_builder.input(NextBaseRow(Pow.master_base_table_index()));
        let lhs_inv_next = circuit_builder.input(NextBaseRow(LhsInv.master_base_table_index()));
        let rp_next = circuit_builder.input(NextExtRow(ProcessorPermArg.master_ext_table_index()));

        let normalized_instruction_deselector = |instruction_to_select: Instruction| {
            let instructions_to_deselect = [
                Instruction::Lt,
                Instruction::And,
                Instruction::Xor,
                Instruction::Log2Floor,
                Instruction::Pow,
                Instruction::Div,
            ]
            .into_iter()
            .filter(|&instruction| instruction != instruction_to_select)
            .collect_vec();

            let deselect_0 = ci_next.clone();
            let deselector = instructions_to_deselect
                .iter()
                .map(|&instruction| {
                    ci_next.clone() - circuit_builder.b_constant(instruction.opcode_b())
                })
                .fold(deselect_0, ConstraintCircuitMonad::mul);
            let normalize_deselected_0 = instruction_to_select.opcode_b();
            let normalizing_factor = instructions_to_deselect
                .iter()
                .fold(normalize_deselected_0, |acc, instr| {
                    acc * (instruction_to_select.opcode_b() - instr.opcode_b())
                })
                .inverse();
            circuit_builder.b_constant(normalizing_factor) * deselector
        };

        let result_next = normalized_instruction_deselector(Instruction::Lt)
            * circuit_builder.input(NextBaseRow(LT.master_base_table_index()))
            + normalized_instruction_deselector(Instruction::Div)
                * circuit_builder.input(NextBaseRow(LT.master_base_table_index()))
            + normalized_instruction_deselector(Instruction::And)
                * circuit_builder.input(NextBaseRow(AND.master_base_table_index()))
            + normalized_instruction_deselector(Instruction::Xor)
                * circuit_builder.input(NextBaseRow(XOR.master_base_table_index()))
            + normalized_instruction_deselector(Instruction::Log2Floor)
                * circuit_builder.input(NextBaseRow(Log2Floor.master_base_table_index()))
            + normalized_instruction_deselector(Instruction::Pow)
                * circuit_builder.input(NextBaseRow(Pow.master_base_table_index()));

        let if_copy_flag_next_is_1_then_lhs_is_0 = copy_flag_next.clone() * lhs.clone();
        let if_copy_flag_next_is_1_then_rhs_is_0 = copy_flag_next.clone() * rhs.clone();
        let if_copy_flag_next_is_0_then_ci_stays =
            (copy_flag_next.clone() - one.clone()) * (ci_next.clone() - ci);
        let if_copy_flag_next_is_0_then_lhs_copy_stays =
            (copy_flag_next.clone() - one.clone()) * (lhs_copy_next - lhs_copy.clone());
        let if_copy_flag_next_is_0_and_lhs_next_is_nonzero_then_bits_increases_by_1 =
            (copy_flag_next.clone() - one.clone())
                * lhs.clone()
                * (bits_next.clone() - bits.clone() - one.clone());
        let if_copy_flag_next_is_0_and_rhs_next_is_nonzero_then_bits_increases_by_1 =
            (copy_flag_next.clone() - one.clone())
                * rhs.clone()
                * (bits_next - bits.clone() - one.clone());
        let lhs_lsb = lhs.clone() - two.clone() * lhs_next.clone();
        let rhs_lsb = rhs - two.clone() * rhs_next.clone();
        let if_copy_flag_next_is_0_then_lhs_lsb_is_a_bit = (copy_flag_next.clone() - one.clone())
            * lhs_lsb.clone()
            * (lhs_lsb.clone() - one.clone());
        let if_copy_flag_next_is_0_then_rhs_lsb_is_a_bit = (copy_flag_next.clone() - one.clone())
            * rhs_lsb.clone()
            * (rhs_lsb.clone() - one.clone());
        let if_copy_flag_next_is_0_and_lt_next_is_0_then_lt_is_0 = (copy_flag_next.clone()
            - one.clone())
            * (lt_next.clone() - one.clone())
            * (lt_next.clone() - two.clone())
            * lt.clone();
        let if_copy_flag_next_is_0_and_lt_next_is_1_then_lt_is_1 = (copy_flag_next.clone()
            - one.clone())
            * lt_next.clone()
            * (lt_next.clone() - two.clone())
            * (lt.clone() - one.clone());
        let if_copy_flag_next_is_0_and_lt_next_is_2_and_lt_known_then_lt_is_1 =
            (copy_flag_next.clone() - one.clone())
                * lt_next.clone()
                * (lt_next.clone() - one.clone())
                * (lhs_lsb.clone() - one.clone())
                * rhs_lsb.clone()
                * (lt.clone() - one.clone());
        let if_copy_flag_next_is_0_and_lt_next_is_2_and_gte_known_then_lt_is_0 =
            (copy_flag_next.clone() - one.clone())
                * lt_next.clone()
                * (lt_next.clone() - one.clone())
                * lhs_lsb.clone()
                * (rhs_lsb.clone() - one.clone())
                * lt.clone();
        let lt_result_unclear = (copy_flag_next.clone() - one.clone())
            * lt_next.clone()
            * (lt_next - one.clone())
            * (one.clone()
                - lhs_lsb.clone()
                - rhs_lsb.clone()
                - two.clone() * lhs_lsb.clone() * rhs_lsb.clone());
        let if_copy_flag_next_is_0_and_lt_next_is_2_and_lsbs_equal_then_lt_is_2 = lt_result_unclear
            .clone()
            * (copy_flag.clone() - one.clone())
            * (lt.clone() - two.clone());
        let if_copy_flag_next_is_0_and_lt_next_is_2_and_lsbs_equal_in_top_row_then_lt_is_0 =
            lt_result_unclear * copy_flag * lt;
        let if_copy_flag_next_is_0_then_and_updates_correctly = (copy_flag_next.clone()
            - one.clone())
            * (and - two.clone() * and_next - lhs_lsb.clone() * rhs_lsb.clone());
        let if_copy_flag_next_is_0_then_xor_updates_correctly = (copy_flag_next.clone()
            - one.clone())
            * (xor - two.clone() * xor_next - lhs_lsb.clone() - rhs_lsb.clone()
                + two * lhs_lsb * rhs_lsb.clone());
        let if_copy_flag_next_is_0_and_lhs_next_is_0_and_lhs_is_nonzero_then_log2floor_is_bits =
            (copy_flag_next.clone() - one.clone())
                * (one.clone() - lhs_next.clone() * lhs_inv_next)
                * lhs
                * (log2floor.clone() - bits);
        let if_copy_flag_next_is_0_and_lhs_next_is_nonzero_then_log2floor_stays =
            (copy_flag_next.clone() - one.clone())
                * lhs_next.clone()
                * (log2floor_next - log2floor);
        let if_copy_flag_next_is_0_and_rhs_lsb_is_0_then_pow_squares = (copy_flag_next.clone()
            - one.clone())
            * (rhs_lsb.clone() - one.clone())
            * (pow.clone() - pow_next.clone() * pow_next.clone());
        let if_copy_flag_next_is_0_and_rhs_lsb_is_0_then_pow_squares_times_lhs_copy =
            (copy_flag_next.clone() - one.clone())
                * rhs_lsb
                * (pow - pow_next.clone() * pow_next * lhs_copy);

        let if_copy_flag_next_is_0_then_running_product_stays =
            (copy_flag_next.clone() - one) * (rp_next.clone() - rp.clone());

        let compressed_row_next = challenge(CIWeight) * ci_next
            + challenge(LhsWeight) * lhs_next
            + challenge(RhsWeight) * rhs_next
            + challenge(ResultWeight) * result_next;
        let if_copy_flag_next_is_1_then_running_product_absorbs_next_row = copy_flag_next
            * (rp_next - rp * (challenge(ProcessorPermIndeterminate) - compressed_row_next));

        [
            if_copy_flag_next_is_1_then_lhs_is_0,
            if_copy_flag_next_is_1_then_rhs_is_0,
            if_copy_flag_next_is_0_then_ci_stays,
            if_copy_flag_next_is_0_then_lhs_copy_stays,
            if_copy_flag_next_is_0_and_lhs_next_is_nonzero_then_bits_increases_by_1,
            if_copy_flag_next_is_0_and_rhs_next_is_nonzero_then_bits_increases_by_1,
            if_copy_flag_next_is_0_then_lhs_lsb_is_a_bit,
            if_copy_flag_next_is_0_then_rhs_lsb_is_a_bit,
            if_copy_flag_next_is_0_and_lt_next_is_0_then_lt_is_0,
            if_copy_flag_next_is_0_and_lt_next_is_1_then_lt_is_1,
            if_copy_flag_next_is_0_and_lt_next_is_2_and_lt_known_then_lt_is_1,
            if_copy_flag_next_is_0_and_lt_next_is_2_and_gte_known_then_lt_is_0,
            if_copy_flag_next_is_0_and_lt_next_is_2_and_lsbs_equal_then_lt_is_2,
            if_copy_flag_next_is_0_and_lt_next_is_2_and_lsbs_equal_in_top_row_then_lt_is_0,
            if_copy_flag_next_is_0_then_and_updates_correctly,
            if_copy_flag_next_is_0_then_xor_updates_correctly,
            if_copy_flag_next_is_0_and_lhs_next_is_0_and_lhs_is_nonzero_then_log2floor_is_bits,
            if_copy_flag_next_is_0_and_lhs_next_is_nonzero_then_log2floor_stays,
            if_copy_flag_next_is_0_and_rhs_lsb_is_0_then_pow_squares,
            if_copy_flag_next_is_0_and_rhs_lsb_is_0_then_pow_squares_times_lhs_copy,
            if_copy_flag_next_is_0_then_running_product_stays,
            if_copy_flag_next_is_1_then_running_product_absorbs_next_row,
        ]
        .map(|circuit| circuit.consume())
        .to_vec()
    }

    pub fn ext_terminal_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            U32TableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let circuit_builder = ConstraintCircuitBuilder::new();

        let lhs = circuit_builder.input(BaseRow(LHS.master_base_table_index()));
        let rhs = circuit_builder.input(BaseRow(RHS.master_base_table_index()));

        [lhs, rhs].map(|circuit| circuit.consume()).to_vec()
    }
}

impl U32Table {
    pub fn fill_trace(u32_table: &mut ArrayViewMut2<BFieldElement>, aet: &AlgebraicExecutionTrace) {
        let mut next_section_start = 0;
        for &(instruction, lhs, rhs) in aet.u32_entries.iter() {
            let mut first_row = Array2::zeros([1, BASE_WIDTH]);
            first_row[[0, CopyFlag.base_table_index()]] = BFieldElement::one();
            first_row[[0, Bits.base_table_index()]] = BFieldElement::zero();
            first_row[[0, BitsMinus33Inv.base_table_index()]] = (-BFieldElement::new(33)).inverse();
            first_row[[0, CI.base_table_index()]] = instruction.opcode_b();
            first_row[[0, LHS.base_table_index()]] = lhs;
            first_row[[0, RHS.base_table_index()]] = rhs;
            first_row[[0, LhsCopy.base_table_index()]] = lhs;
            let u32_section = Self::u32_section_next_row(first_row);

            let next_section_end = next_section_start + u32_section.nrows();
            u32_table
                .slice_mut(s![next_section_start..next_section_end, ..])
                .assign(&u32_section);
            next_section_start = next_section_end;
        }
    }

    fn u32_section_next_row(mut section: Array2<BFieldElement>) -> Array2<BFieldElement> {
        let zero = BFieldElement::zero();
        let one = BFieldElement::one();
        let two = BFieldElement::new(2);
        let thirty_three = BFieldElement::new(33);

        let row_idx = section.nrows() - 1;
        if section[[row_idx, LHS.base_table_index()]].is_zero()
            && section[[row_idx, RHS.base_table_index()]].is_zero()
        {
            section[[row_idx, LT.base_table_index()]] = two;
            section[[row_idx, AND.base_table_index()]] = zero;
            section[[row_idx, XOR.base_table_index()]] = zero;
            section[[row_idx, Log2Floor.base_table_index()]] = -one;
            section[[row_idx, Pow.base_table_index()]] = one;
            return section;
        }

        let lhs_lsb = BFieldElement::new(section[[row_idx, LHS.base_table_index()]].value() % 2);
        let rhs_lsb = BFieldElement::new(section[[row_idx, RHS.base_table_index()]].value() % 2);
        let mut next_row = section.row(row_idx).to_owned();
        next_row[CopyFlag.base_table_index()] = zero;
        next_row[Bits.base_table_index()] += one;
        next_row[BitsMinus33Inv.base_table_index()] =
            (next_row[Bits.base_table_index()] - thirty_three).inverse();
        next_row[LHS.base_table_index()] =
            (section[[row_idx, LHS.base_table_index()]] - lhs_lsb) / two;
        next_row[RHS.base_table_index()] =
            (section[[row_idx, RHS.base_table_index()]] - rhs_lsb) / two;

        section.push_row(next_row.view()).unwrap();
        section = Self::u32_section_next_row(section);
        let (mut row, next_row) = section.multi_slice_mut((s![row_idx, ..], s![row_idx + 1, ..]));

        row[LhsInv.base_table_index()] = row[LHS.base_table_index()].inverse_or_zero();
        row[RhsInv.base_table_index()] = row[RHS.base_table_index()].inverse_or_zero();

        row[LT.base_table_index()] = if next_row[LT.base_table_index()].is_zero() {
            zero
        } else if next_row[LT.base_table_index()].is_one() {
            one
        } else {
            // LT == 2
            if lhs_lsb.is_zero() && rhs_lsb.is_one() {
                one
            } else if lhs_lsb.is_one() && rhs_lsb.is_zero() {
                zero
            } else {
                // lhs_lsb == rhs_lsb
                if row[CopyFlag.base_table_index()].is_zero() {
                    two
                } else {
                    zero
                }
            }
        };

        row[AND.base_table_index()] = two * next_row[AND.base_table_index()] + lhs_lsb * rhs_lsb;
        row[XOR.base_table_index()] =
            two * next_row[XOR.base_table_index()] + lhs_lsb + rhs_lsb - two * lhs_lsb * rhs_lsb;

        row[Log2Floor.base_table_index()] = if row[LHS.base_table_index()].is_zero() {
            -one
        } else if !next_row[LHS.base_table_index()].is_zero() {
            next_row[Log2Floor.base_table_index()]
        } else {
            // next_row[LHS.base_table_index()].is_zero() && !row[LHS.base_table_index()].is_zero()
            row[Bits.base_table_index()]
        };

        row[Pow.base_table_index()] = if row[RHS.base_table_index()].is_zero() {
            next_row[Pow.base_table_index()] * next_row[Pow.base_table_index()]
        } else {
            next_row[Pow.base_table_index()]
                * next_row[Pow.base_table_index()]
                * row[LhsCopy.base_table_index()]
        };

        section
    }

    pub fn pad_trace(u32_table: &mut ArrayViewMut2<BFieldElement>, u32_table_len: usize) {
        let mut padding_row = Array1::zeros([BASE_WIDTH]);
        padding_row[[BitsMinus33Inv.base_table_index()]] = (-BFieldElement::new(33)).inverse();
        padding_row[[LT.base_table_index()]] = BFieldElement::new(2);
        padding_row[[Log2Floor.base_table_index()]] = -BFieldElement::one();
        padding_row[[Pow.base_table_index()]] = BFieldElement::one();

        if u32_table_len > 0 {
            let last_row = u32_table.row(u32_table_len - 1);
            padding_row[[CI.base_table_index()]] = last_row[CI.base_table_index()];
            padding_row[[LhsCopy.base_table_index()]] = last_row[LhsCopy.base_table_index()];
        }

        u32_table
            .slice_mut(s![u32_table_len.., ..])
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| row.assign(&padding_row));
    }

    pub fn extend(
        base_table: ArrayView2<BFieldElement>,
        mut ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &U32TableChallenges,
    ) {
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());

        let mut running_product = PermArg::default_initial();
        for row_idx in 0..base_table.nrows() {
            let current_row = base_table.row(row_idx);
            if current_row[CopyFlag.base_table_index()].is_one() {
                let current_instruction = current_row[CI.base_table_index()]
                    .value()
                    .try_into()
                    .unwrap_or(Instruction::Split);
                let result = match current_instruction {
                    Instruction::Lt | Instruction::Div => current_row[LT.base_table_index()],
                    Instruction::And => current_row[AND.base_table_index()],
                    Instruction::Xor => current_row[XOR.base_table_index()],
                    Instruction::Log2Floor => current_row[Log2Floor.base_table_index()],
                    Instruction::Pow => current_row[Pow.base_table_index()],
                    _ => BFieldElement::zero(),
                };

                let compressed_row = challenges.ci_weight * current_row[CI.base_table_index()]
                    + challenges.lhs_weight * current_row[LHS.base_table_index()]
                    + challenges.rhs_weight * current_row[RHS.base_table_index()]
                    + challenges.result_weight * result;
                running_product *= challenges.processor_perm_indeterminate - compressed_row;
            }

            let mut extension_row = ext_table.row_mut(row_idx);
            extension_row[ProcessorPermArg.ext_table_index()] = running_product;
        }
    }
}

#[repr(usize)]
#[derive(Debug, Copy, Clone, Display, EnumCountMacro, EnumIter, PartialEq, Eq, Hash)]
pub enum U32TableChallengeId {
    LhsWeight,
    RhsWeight,
    CIWeight,
    ResultWeight,
    ProcessorPermIndeterminate,
}

impl From<U32TableChallengeId> for usize {
    fn from(val: U32TableChallengeId) -> Self {
        val as usize
    }
}

impl TableChallenges for U32TableChallenges {
    type Id = U32TableChallengeId;

    fn get_challenge(&self, id: Self::Id) -> XFieldElement {
        match id {
            LhsWeight => self.lhs_weight,
            RhsWeight => self.rhs_weight,
            CIWeight => self.ci_weight,
            ResultWeight => self.result_weight,
            ProcessorPermIndeterminate => self.processor_perm_indeterminate,
        }
    }
}

#[derive(Debug, Clone)]
pub struct U32TableChallenges {
    pub lhs_weight: XFieldElement,
    pub rhs_weight: XFieldElement,
    pub ci_weight: XFieldElement,
    pub result_weight: XFieldElement,
    pub processor_perm_indeterminate: XFieldElement,
}
