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

        let deselect_instructions = |instructions: &[Instruction]| {
            instructions
                .iter()
                .map(|&instr| ci.clone() - circuit_builder.b_constant(instr.opcode_b()))
                .fold(one.clone(), ConstraintCircuitMonad::mul)
                * ci.clone()
        };
        let lt_div_deselector = deselect_instructions(&[
            Instruction::And,
            Instruction::Xor,
            Instruction::Log2Floor,
            Instruction::Pow,
        ]);
        let and_deselector = deselect_instructions(&[
            Instruction::Lt,
            Instruction::Div,
            Instruction::Xor,
            Instruction::Log2Floor,
            Instruction::Pow,
        ]);
        let xor_deselector = deselect_instructions(&[
            Instruction::Lt,
            Instruction::Div,
            Instruction::And,
            Instruction::Log2Floor,
            Instruction::Pow,
        ]);
        let log2floor_deselector = deselect_instructions(&[
            Instruction::Lt,
            Instruction::Div,
            Instruction::And,
            Instruction::Xor,
            Instruction::Pow,
        ]);
        let pow_deselector = deselect_instructions(&[
            Instruction::Lt,
            Instruction::Div,
            Instruction::And,
            Instruction::Xor,
            Instruction::Log2Floor,
        ]);
        let result = lt_div_deselector
            * circuit_builder.input(BaseRow(LT.master_base_table_index()))
            + and_deselector * circuit_builder.input(BaseRow(AND.master_base_table_index()))
            + xor_deselector * circuit_builder.input(BaseRow(XOR.master_base_table_index()))
            + log2floor_deselector
                * circuit_builder.input(BaseRow(Log2Floor.master_base_table_index()))
            + pow_deselector * circuit_builder.input(BaseRow(Pow.master_base_table_index()));

        let initial_factor = challenge(ProcessorPermIndeterminate)
            - challenge(LhsWeight) * lhs
            - challenge(RhsWeight) * rhs
            - challenge(CIWeight) * ci
            - challenge(ResultWeight) * result;
        let copy_flag_is_0_or_rp_has_accumulated_first_row =
            copy_flag.clone() * (rp.clone() - initial_factor);

        let default_initial = circuit_builder.x_constant(PermArg::default_initial());
        let copy_flag_is_1_or_rp_is_default_initial = (one - copy_flag) * (default_initial - rp);

        [
            copy_flag_is_1_or_rp_is_default_initial,
            copy_flag_is_0_or_rp_has_accumulated_first_row,
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

        let deselect_instructions = |instructions: &[Instruction]| {
            instructions
                .iter()
                .map(|&instr| ci.clone() - circuit_builder.b_constant(instr.opcode_b()))
                .fold(one.clone(), ConstraintCircuitMonad::mul)
                * ci.clone()
        };
        let lt_div_deselector = deselect_instructions(&[
            Instruction::And,
            Instruction::Xor,
            Instruction::Log2Floor,
            Instruction::Pow,
        ]);
        let and_deselector = deselect_instructions(&[
            Instruction::Lt,
            Instruction::Div,
            Instruction::Xor,
            Instruction::Log2Floor,
            Instruction::Pow,
        ]);
        let xor_deselector = deselect_instructions(&[
            Instruction::Lt,
            Instruction::Div,
            Instruction::And,
            Instruction::Log2Floor,
            Instruction::Pow,
        ]);
        let log2floor_deselector = deselect_instructions(&[
            Instruction::Lt,
            Instruction::Div,
            Instruction::And,
            Instruction::Xor,
            Instruction::Pow,
        ]);
        let pow_deselector = deselect_instructions(&[
            Instruction::Lt,
            Instruction::Div,
            Instruction::And,
            Instruction::Xor,
            Instruction::Log2Floor,
        ]);
        let result = lt_div_deselector
            * circuit_builder.input(CurrentBaseRow(LT.master_base_table_index()))
            + and_deselector * circuit_builder.input(CurrentBaseRow(AND.master_base_table_index()))
            + xor_deselector * circuit_builder.input(CurrentBaseRow(XOR.master_base_table_index()))
            + log2floor_deselector
                * circuit_builder.input(CurrentBaseRow(Log2Floor.master_base_table_index()))
            + pow_deselector * circuit_builder.input(CurrentBaseRow(Pow.master_base_table_index()));

        let if_copy_flag_next_is_1_then_lhs_is_0 = copy_flag_next.clone() * lhs.clone();
        let if_copy_flag_next_is_1_then_rhs_is_0 = copy_flag_next.clone() * rhs.clone();
        let if_copy_flag_next_is_0_then_ci_stays =
            (copy_flag_next.clone() - one.clone()) * (ci_next - ci.clone());
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
        let lhs_lsb = two.clone() * lhs_next.clone() - lhs.clone();
        let rhs_lsb = two.clone() * rhs_next - rhs.clone();
        let lhs_lsb_is_a_bit = lhs_lsb.clone() * (lhs_lsb.clone() - one.clone());
        let rhs_lsb_is_a_bit = rhs_lsb.clone() * (rhs_lsb.clone() - one.clone());
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
            lt_result_unclear * copy_flag.clone() * lt;
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
                * lhs.clone()
                * (log2floor.clone() - bits);
        let if_copy_flag_next_is_0_and_lhs_next_is_nonzero_then_log2floor_stays =
            (copy_flag_next.clone() - one.clone()) * lhs_next * (log2floor_next - log2floor);
        let if_copy_flag_next_is_0_and_rhs_lsb_is_0_then_pow_squares = (copy_flag_next.clone()
            - one.clone())
            * (rhs_lsb.clone() - one.clone())
            * (pow.clone() - pow_next.clone() * pow_next.clone());
        let if_copy_flag_next_is_0_and_rhs_lsb_is_0_then_pow_squares_times_lhs_copy =
            (copy_flag_next.clone() - one.clone())
                * rhs_lsb
                * (pow - pow_next.clone() * pow_next * lhs_copy);

        let compressed_row = challenge(ProcessorPermIndeterminate)
            - challenge(LhsWeight) * lhs
            - challenge(RhsWeight) * rhs
            - challenge(CIWeight) * ci
            - challenge(ResultWeight) * result;
        let if_copy_flag_next_is_0_then_running_product_stays =
            (copy_flag_next - one.clone()) * (rp_next.clone() - rp.clone());
        let if_copy_flag_next_is_1_then_running_product_absorbs_row =
            copy_flag * (rp_next - rp * compressed_row);

        [
            if_copy_flag_next_is_1_then_lhs_is_0,
            if_copy_flag_next_is_1_then_rhs_is_0,
            if_copy_flag_next_is_0_then_ci_stays,
            if_copy_flag_next_is_0_then_lhs_copy_stays,
            if_copy_flag_next_is_0_and_lhs_next_is_nonzero_then_bits_increases_by_1,
            if_copy_flag_next_is_0_and_rhs_next_is_nonzero_then_bits_increases_by_1,
            lhs_lsb_is_a_bit,
            rhs_lsb_is_a_bit,
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
            if_copy_flag_next_is_1_then_running_product_absorbs_row,
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
                * next_row[LhsCopy.base_table_index()]
        };

        section
    }

    pub fn pad_trace(u32_table: &mut ArrayViewMut2<BFieldElement>, u32_table_len: usize) {
        let mut padding_row = Array1::zeros([BASE_WIDTH]);
        padding_row[[BitsMinus33Inv.base_table_index()]] = (-BFieldElement::new(33)).inverse();
        padding_row[[LT.base_table_index()]] = BFieldElement::new(2);
        padding_row[[Log2Floor.base_table_index()]] = -BFieldElement::one();
        padding_row[[Pow.base_table_index()]] = BFieldElement::one();

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
                let ci_opcode = current_row[CI.base_table_index()].value();
                let result = if ci_opcode == Instruction::Lt.opcode() as u64 {
                    current_row[LT.base_table_index()]
                } else if ci_opcode == Instruction::And.opcode() as u64 {
                    current_row[AND.base_table_index()]
                } else if ci_opcode == Instruction::Xor.opcode() as u64 {
                    current_row[XOR.base_table_index()]
                } else if ci_opcode == Instruction::Log2Floor.opcode() as u64 {
                    current_row[Log2Floor.base_table_index()]
                } else if ci_opcode == Instruction::Pow.opcode() as u64 {
                    current_row[Pow.base_table_index()]
                } else if ci_opcode == Instruction::Div.opcode() as u64 {
                    current_row[LT.base_table_index()]
                } else {
                    BFieldElement::zero()
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
