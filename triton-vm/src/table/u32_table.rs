use std::cmp::max;
use std::ops::Mul;

use arbitrary::Arbitrary;
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
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::instruction::Instruction;
use crate::profiler::profiler;
use crate::table::challenges::ChallengeId::*;
use crate::table::challenges::Challenges;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::ConstraintCircuitMonad;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::DualRowIndicator::*;
use crate::table::constraint_circuit::InputIndicator;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::*;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::LookupArg;
use crate::table::table_column::MasterBaseTableColumn;
use crate::table::table_column::MasterExtTableColumn;
use crate::table::table_column::U32BaseTableColumn;
use crate::table::table_column::U32BaseTableColumn::*;
use crate::table::table_column::U32ExtTableColumn;
use crate::table::table_column::U32ExtTableColumn::*;

pub const BASE_WIDTH: usize = U32BaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = U32ExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

/// An executed u32 instruction as well as its operands.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub struct U32TableEntry {
    pub instruction: Instruction,
    pub left_operand: BFieldElement,
    pub right_operand: BFieldElement,
}

impl U32TableEntry {
    pub fn new<L, R>(instruction: Instruction, left_operand: L, right_operand: R) -> Self
    where
        L: Into<BFieldElement>,
        R: Into<BFieldElement>,
    {
        Self {
            instruction,
            left_operand: left_operand.into(),
            right_operand: right_operand.into(),
        }
    }

    /// The number of rows this entry contributes to the U32 Table.
    pub(crate) fn table_height_contribution(&self) -> u32 {
        let lhs = self.left_operand.value();
        let rhs = self.right_operand.value();
        let dominant_operand = match self.instruction {
            Instruction::Pow => rhs, // left-hand side doesn't change between rows
            _ => max(lhs, rhs),
        };
        match dominant_operand {
            0 => 2 - 1,
            _ => 2 + dominant_operand.ilog2(),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct U32Table;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ExtU32Table;

impl ExtU32Table {
    fn instruction_deselector<II: InputIndicator>(
        instruction_to_select: Instruction,
        circuit_builder: &ConstraintCircuitBuilder<II>,
        current_instruction: &ConstraintCircuitMonad<II>,
    ) -> ConstraintCircuitMonad<II> {
        [
            Instruction::Split,
            Instruction::Lt,
            Instruction::And,
            Instruction::Log2Floor,
            Instruction::Pow,
            Instruction::PopCount,
        ]
        .into_iter()
        .filter(|&instruction| instruction != instruction_to_select)
        .map(|instruction| {
            current_instruction.clone() - circuit_builder.b_constant(instruction.opcode_b())
        })
        .fold(
            circuit_builder.b_constant(b_field_element::BFIELD_ONE),
            ConstraintCircuitMonad::mul,
        )
    }

    pub fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let one = circuit_builder.b_constant(1);

        let copy_flag = circuit_builder.input(BaseRow(CopyFlag.master_base_table_index()));
        let lhs = circuit_builder.input(BaseRow(LHS.master_base_table_index()));
        let rhs = circuit_builder.input(BaseRow(RHS.master_base_table_index()));
        let ci = circuit_builder.input(BaseRow(CI.master_base_table_index()));
        let result = circuit_builder.input(BaseRow(Result.master_base_table_index()));
        let lookup_multiplicity =
            circuit_builder.input(BaseRow(LookupMultiplicity.master_base_table_index()));

        let running_sum_log_derivative =
            circuit_builder.input(ExtRow(LookupServerLogDerivative.master_ext_table_index()));

        let compressed_row = challenge(U32LhsWeight) * lhs
            + challenge(U32RhsWeight) * rhs
            + challenge(U32CiWeight) * ci
            + challenge(U32ResultWeight) * result;
        let if_copy_flag_is_1_then_log_derivative_has_accumulated_first_row = copy_flag.clone()
            * (running_sum_log_derivative.clone() * (challenge(U32Indeterminate) - compressed_row)
                - lookup_multiplicity);

        let default_initial = circuit_builder.x_constant(LookupArg::default_initial());
        let if_copy_flag_is_0_then_log_derivative_is_default_initial =
            (copy_flag - one) * (running_sum_log_derivative - default_initial);

        let running_sum_log_derivative_starts_correctly =
            if_copy_flag_is_0_then_log_derivative_is_default_initial
                + if_copy_flag_is_1_then_log_derivative_has_accumulated_first_row;

        vec![running_sum_log_derivative_starts_correctly]
    }

    pub fn consistency_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let one = || circuit_builder.b_constant(1);
        let two = || circuit_builder.b_constant(2);

        let copy_flag = circuit_builder.input(BaseRow(CopyFlag.master_base_table_index()));
        let bits = circuit_builder.input(BaseRow(Bits.master_base_table_index()));
        let bits_minus_33_inv =
            circuit_builder.input(BaseRow(BitsMinus33Inv.master_base_table_index()));
        let ci = circuit_builder.input(BaseRow(CI.master_base_table_index()));
        let lhs = circuit_builder.input(BaseRow(LHS.master_base_table_index()));
        let lhs_inv = circuit_builder.input(BaseRow(LhsInv.master_base_table_index()));
        let rhs = circuit_builder.input(BaseRow(RHS.master_base_table_index()));
        let rhs_inv = circuit_builder.input(BaseRow(RhsInv.master_base_table_index()));
        let result = circuit_builder.input(BaseRow(Result.master_base_table_index()));
        let lookup_multiplicity =
            circuit_builder.input(BaseRow(LookupMultiplicity.master_base_table_index()));

        let instruction_deselector = |instruction_to_select| {
            Self::instruction_deselector(instruction_to_select, circuit_builder, &ci)
        };

        let copy_flag_is_bit = copy_flag.clone() * (one() - copy_flag.clone());
        let copy_flag_is_0_or_bits_is_0 = copy_flag.clone() * bits.clone();
        let bits_minus_33_inv_is_inverse_of_bits_minus_33 =
            one() - bits_minus_33_inv * (bits - circuit_builder.b_constant(33));
        let lhs_inv_is_0_or_the_inverse_of_lhs =
            lhs_inv.clone() * (one() - lhs.clone() * lhs_inv.clone());
        let lhs_is_0_or_lhs_inverse_is_the_inverse_of_lhs =
            lhs.clone() * (one() - lhs.clone() * lhs_inv.clone());
        let rhs_inv_is_0_or_the_inverse_of_rhs =
            rhs_inv.clone() * (one() - rhs.clone() * rhs_inv.clone());
        let rhs_is_0_or_rhs_inverse_is_the_inverse_of_rhs =
            rhs.clone() * (one() - rhs.clone() * rhs_inv.clone());
        let result_is_initialized_correctly_for_lt_with_copy_flag_0 =
            instruction_deselector(Instruction::Lt)
                * (copy_flag.clone() - one())
                * (one() - lhs.clone() * lhs_inv.clone())
                * (one() - rhs.clone() * rhs_inv.clone())
                * (result.clone() - two());
        let result_is_initialized_correctly_for_lt_with_copy_flag_1 =
            instruction_deselector(Instruction::Lt)
                * copy_flag.clone()
                * (one() - lhs.clone() * lhs_inv.clone())
                * (one() - rhs.clone() * rhs_inv.clone())
                * result.clone();
        let result_is_initialized_correctly_for_and = instruction_deselector(Instruction::And)
            * (one() - lhs.clone() * lhs_inv.clone())
            * (one() - rhs.clone() * rhs_inv.clone())
            * result.clone();
        let result_is_initialized_correctly_for_pow = instruction_deselector(Instruction::Pow)
            * (one() - rhs * rhs_inv)
            * (result.clone() - one());
        let result_is_initialized_correctly_for_log_2_floor =
            instruction_deselector(Instruction::Log2Floor)
                * (copy_flag.clone() - one())
                * (one() - lhs.clone() * lhs_inv.clone())
                * (result.clone() + one());
        let result_is_initialized_correctly_for_pop_count =
            instruction_deselector(Instruction::PopCount)
                * (one() - lhs.clone() * lhs_inv.clone())
                * result;
        let if_log_2_floor_on_0_then_vm_crashes = instruction_deselector(Instruction::Log2Floor)
            * copy_flag.clone()
            * (one() - lhs * lhs_inv);
        let if_copy_flag_is_0_then_lookup_multiplicity_is_0 =
            (copy_flag - one()) * lookup_multiplicity;

        vec![
            copy_flag_is_bit,
            copy_flag_is_0_or_bits_is_0,
            bits_minus_33_inv_is_inverse_of_bits_minus_33,
            lhs_inv_is_0_or_the_inverse_of_lhs,
            lhs_is_0_or_lhs_inverse_is_the_inverse_of_lhs,
            rhs_inv_is_0_or_the_inverse_of_rhs,
            rhs_is_0_or_rhs_inverse_is_the_inverse_of_rhs,
            result_is_initialized_correctly_for_lt_with_copy_flag_0,
            result_is_initialized_correctly_for_lt_with_copy_flag_1,
            result_is_initialized_correctly_for_and,
            result_is_initialized_correctly_for_pow,
            result_is_initialized_correctly_for_log_2_floor,
            result_is_initialized_correctly_for_pop_count,
            if_log_2_floor_on_0_then_vm_crashes,
            if_copy_flag_is_0_then_lookup_multiplicity_is_0,
        ]
    }

    pub fn transition_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let one = || circuit_builder.b_constant(1);
        let two = || circuit_builder.b_constant(2);

        let copy_flag = circuit_builder.input(CurrentBaseRow(CopyFlag.master_base_table_index()));
        let bits = circuit_builder.input(CurrentBaseRow(Bits.master_base_table_index()));
        let ci = circuit_builder.input(CurrentBaseRow(CI.master_base_table_index()));
        let lhs = circuit_builder.input(CurrentBaseRow(LHS.master_base_table_index()));
        let rhs = circuit_builder.input(CurrentBaseRow(RHS.master_base_table_index()));
        let result = circuit_builder.input(CurrentBaseRow(Result.master_base_table_index()));
        let running_sum_log_derivative = circuit_builder.input(CurrentExtRow(
            LookupServerLogDerivative.master_ext_table_index(),
        ));

        let copy_flag_next = circuit_builder.input(NextBaseRow(CopyFlag.master_base_table_index()));
        let bits_next = circuit_builder.input(NextBaseRow(Bits.master_base_table_index()));
        let ci_next = circuit_builder.input(NextBaseRow(CI.master_base_table_index()));
        let lhs_next = circuit_builder.input(NextBaseRow(LHS.master_base_table_index()));
        let rhs_next = circuit_builder.input(NextBaseRow(RHS.master_base_table_index()));
        let result_next = circuit_builder.input(NextBaseRow(Result.master_base_table_index()));
        let lhs_inv_next = circuit_builder.input(NextBaseRow(LhsInv.master_base_table_index()));
        let lookup_multiplicity_next =
            circuit_builder.input(NextBaseRow(LookupMultiplicity.master_base_table_index()));
        let running_sum_log_derivative_next = circuit_builder.input(NextExtRow(
            LookupServerLogDerivative.master_ext_table_index(),
        ));

        let instruction_deselector = |instruction_to_select: Instruction| {
            Self::instruction_deselector(instruction_to_select, circuit_builder, &ci_next)
        };

        // helpful aliases
        let ci_is_pow = ci.clone() - circuit_builder.b_constant(Instruction::Pow.opcode_b());
        let lhs_lsb = lhs.clone() - two() * lhs_next.clone();
        let rhs_lsb = rhs.clone() - two() * rhs_next.clone();

        // general constraints
        let if_copy_flag_next_is_1_then_lhs_is_0_or_ci_is_pow =
            copy_flag_next.clone() * lhs.clone() * ci_is_pow.clone();
        let if_copy_flag_next_is_1_then_rhs_is_0 = copy_flag_next.clone() * rhs.clone();
        let if_copy_flag_next_is_0_then_ci_stays =
            (copy_flag_next.clone() - one()) * (ci_next.clone() - ci);
        let if_copy_flag_next_is_0_and_lhs_next_is_nonzero_and_ci_not_pow_then_bits_increases_by_1 =
            (copy_flag_next.clone() - one())
                * lhs.clone()
                * ci_is_pow.clone()
                * (bits_next.clone() - bits.clone() - one());
        let if_copy_flag_next_is_0_and_rhs_next_is_nonzero_then_bits_increases_by_1 =
            (copy_flag_next.clone() - one()) * rhs * (bits_next - bits.clone() - one());
        let if_copy_flag_next_is_0_and_ci_not_pow_then_lhs_lsb_is_a_bit = (copy_flag_next.clone()
            - one())
            * ci_is_pow
            * lhs_lsb.clone()
            * (lhs_lsb.clone() - one());
        let if_copy_flag_next_is_0_then_rhs_lsb_is_a_bit =
            (copy_flag_next.clone() - one()) * rhs_lsb.clone() * (rhs_lsb.clone() - one());

        // instruction lt
        let if_copy_flag_next_is_0_and_ci_is_lt_and_result_next_is_0_then_result_is_0 =
            (copy_flag_next.clone() - one())
                * instruction_deselector(Instruction::Lt)
                * (result_next.clone() - one())
                * (result_next.clone() - two())
                * result.clone();
        let if_copy_flag_next_is_0_and_ci_is_lt_and_result_next_is_1_then_result_is_1 =
            (copy_flag_next.clone() - one())
                * instruction_deselector(Instruction::Lt)
                * result_next.clone()
                * (result_next.clone() - two())
                * (result.clone() - one());
        let if_copy_flag_next_is_0_and_ci_is_lt_and_result_next_is_2_and_lt_is_0_then_result_is_0 =
            (copy_flag_next.clone() - one())
                * instruction_deselector(Instruction::Lt)
                * result_next.clone()
                * (result_next.clone() - one())
                * (lhs_lsb.clone() - one())
                * rhs_lsb.clone()
                * (result.clone() - one());
        let if_copy_flag_next_is_0_and_ci_is_lt_and_result_next_is_2_and_lt_is_1_then_result_is_1 =
            (copy_flag_next.clone() - one())
                * instruction_deselector(Instruction::Lt)
                * result_next.clone()
                * (result_next.clone() - one())
                * lhs_lsb.clone()
                * (rhs_lsb.clone() - one())
                * result.clone();
        let if_copy_flag_next_is_0_and_ci_is_lt_and_result_still_not_known_then_result_is_2 =
            (copy_flag_next.clone() - one())
                * instruction_deselector(Instruction::Lt)
                * result_next.clone()
                * (result_next.clone() - one())
                * (one() - lhs_lsb.clone() - rhs_lsb.clone()
                    + two() * lhs_lsb.clone() * rhs_lsb.clone())
                * (copy_flag.clone() - one())
                * (result.clone() - two());
        let if_copy_flag_next_is_0_and_ci_is_lt_and_copy_flag_dictates_result_then_result_is_0 =
            (copy_flag_next.clone() - one())
                * instruction_deselector(Instruction::Lt)
                * result_next.clone()
                * (result_next.clone() - one())
                * (one() - lhs_lsb.clone() - rhs_lsb.clone()
                    + two() * lhs_lsb.clone() * rhs_lsb.clone())
                * copy_flag
                * result.clone();

        // instruction and
        let if_copy_flag_next_is_0_and_ci_is_and_then_results_updates_correctly = (copy_flag_next
            .clone()
            - one())
            * instruction_deselector(Instruction::And)
            * (result.clone() - two() * result_next.clone() - lhs_lsb.clone() * rhs_lsb.clone());

        // instruction log_2_floor
        let if_copy_flag_next_is_0_and_ci_is_log_2_floor_lhs_next_0_for_first_time_then_set_result =
            (copy_flag_next.clone() - one())
                * instruction_deselector(Instruction::Log2Floor)
                * (one() - lhs_next.clone() * lhs_inv_next)
                * lhs.clone()
                * (result.clone() - bits);
        let if_copy_flag_next_is_0_and_ci_is_log_2_floor_and_lhs_next_not_0_then_copy_result =
            (copy_flag_next.clone() - one())
                * instruction_deselector(Instruction::Log2Floor)
                * lhs_next.clone()
                * (result_next.clone() - result.clone());

        // instruction pow
        let if_copy_flag_next_is_0_and_ci_is_pow_then_lhs_remains_unchanged =
            (copy_flag_next.clone() - one())
                * instruction_deselector(Instruction::Pow)
                * (lhs_next.clone() - lhs.clone());

        let if_copy_flag_next_is_0_and_ci_is_pow_and_rhs_lsb_is_0_then_result_squares =
            (copy_flag_next.clone() - one())
                * instruction_deselector(Instruction::Pow)
                * (rhs_lsb.clone() - one())
                * (result.clone() - result_next.clone() * result_next.clone());

        let if_copy_flag_next_is_0_and_ci_is_pow_and_rhs_lsb_is_1_then_result_squares_and_mults =
            (copy_flag_next.clone() - one())
                * instruction_deselector(Instruction::Pow)
                * rhs_lsb
                * (result.clone() - result_next.clone() * result_next.clone() * lhs);

        let if_copy_flag_next_is_0_and_ci_is_pop_count_then_result_increases_by_lhs_lsb =
            (copy_flag_next.clone() - one())
                * instruction_deselector(Instruction::PopCount)
                * (result - result_next.clone() - lhs_lsb);

        // running sum for Lookup Argument with Processor Table
        let if_copy_flag_next_is_0_then_running_sum_log_derivative_stays = (copy_flag_next.clone()
            - one())
            * (running_sum_log_derivative_next.clone() - running_sum_log_derivative.clone());

        let compressed_row_next = challenge(U32CiWeight) * ci_next
            + challenge(U32LhsWeight) * lhs_next
            + challenge(U32RhsWeight) * rhs_next
            + challenge(U32ResultWeight) * result_next;
        let if_copy_flag_next_is_1_then_running_sum_log_derivative_accumulates_next_row =
            copy_flag_next
                * ((running_sum_log_derivative_next - running_sum_log_derivative)
                    * (challenge(U32Indeterminate) - compressed_row_next)
                    - lookup_multiplicity_next);

        vec![
            if_copy_flag_next_is_1_then_lhs_is_0_or_ci_is_pow,
            if_copy_flag_next_is_1_then_rhs_is_0,
            if_copy_flag_next_is_0_then_ci_stays,
            if_copy_flag_next_is_0_and_lhs_next_is_nonzero_and_ci_not_pow_then_bits_increases_by_1,
            if_copy_flag_next_is_0_and_rhs_next_is_nonzero_then_bits_increases_by_1,
            if_copy_flag_next_is_0_and_ci_not_pow_then_lhs_lsb_is_a_bit,
            if_copy_flag_next_is_0_then_rhs_lsb_is_a_bit,
            if_copy_flag_next_is_0_and_ci_is_lt_and_result_next_is_0_then_result_is_0,
            if_copy_flag_next_is_0_and_ci_is_lt_and_result_next_is_1_then_result_is_1,
            if_copy_flag_next_is_0_and_ci_is_lt_and_result_next_is_2_and_lt_is_0_then_result_is_0,
            if_copy_flag_next_is_0_and_ci_is_lt_and_result_next_is_2_and_lt_is_1_then_result_is_1,
            if_copy_flag_next_is_0_and_ci_is_lt_and_result_still_not_known_then_result_is_2,
            if_copy_flag_next_is_0_and_ci_is_lt_and_copy_flag_dictates_result_then_result_is_0,
            if_copy_flag_next_is_0_and_ci_is_and_then_results_updates_correctly,
            if_copy_flag_next_is_0_and_ci_is_log_2_floor_lhs_next_0_for_first_time_then_set_result,
            if_copy_flag_next_is_0_and_ci_is_log_2_floor_and_lhs_next_not_0_then_copy_result,
            if_copy_flag_next_is_0_and_ci_is_pow_then_lhs_remains_unchanged,
            if_copy_flag_next_is_0_and_ci_is_pow_and_rhs_lsb_is_0_then_result_squares,
            if_copy_flag_next_is_0_and_ci_is_pow_and_rhs_lsb_is_1_then_result_squares_and_mults,
            if_copy_flag_next_is_0_and_ci_is_pop_count_then_result_increases_by_lhs_lsb,
            if_copy_flag_next_is_0_then_running_sum_log_derivative_stays,
            if_copy_flag_next_is_1_then_running_sum_log_derivative_accumulates_next_row,
        ]
    }

    pub fn terminal_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let ci = circuit_builder.input(BaseRow(CI.master_base_table_index()));
        let lhs = circuit_builder.input(BaseRow(LHS.master_base_table_index()));
        let rhs = circuit_builder.input(BaseRow(RHS.master_base_table_index()));

        let lhs_is_0_or_ci_is_pow =
            lhs * (ci - circuit_builder.b_constant(Instruction::Pow.opcode_b()));
        let rhs_is_0 = rhs;

        vec![lhs_is_0_or_ci_is_pow, rhs_is_0]
    }
}

impl U32Table {
    pub fn fill_trace(u32_table: &mut ArrayViewMut2<BFieldElement>, aet: &AlgebraicExecutionTrace) {
        let mut next_section_start = 0;
        for (&u32_table_entry, &multiplicity) in &aet.u32_entries {
            let mut first_row = Array2::zeros([1, BASE_WIDTH]);
            first_row[[0, CopyFlag.base_table_index()]] = bfe!(1);
            first_row[[0, Bits.base_table_index()]] = bfe!(0);
            first_row[[0, BitsMinus33Inv.base_table_index()]] = bfe!(-33).inverse();
            first_row[[0, CI.base_table_index()]] = u32_table_entry.instruction.opcode_b();
            first_row[[0, LHS.base_table_index()]] = u32_table_entry.left_operand;
            first_row[[0, RHS.base_table_index()]] = u32_table_entry.right_operand;
            first_row[[0, LookupMultiplicity.base_table_index()]] = multiplicity.into();
            let u32_section = Self::u32_section_next_row(first_row);

            let next_section_end = next_section_start + u32_section.nrows();
            u32_table
                .slice_mut(s![next_section_start..next_section_end, ..])
                .assign(&u32_section);
            next_section_start = next_section_end;
        }
    }

    fn u32_section_next_row(mut section: Array2<BFieldElement>) -> Array2<BFieldElement> {
        let row_idx = section.nrows() - 1;
        let current_instruction: Instruction = section[[row_idx, CI.base_table_index()]]
            .value()
            .try_into()
            .expect("Unknown instruction");

        // Is the last row in this section reached?
        if (section[[row_idx, LHS.base_table_index()]].is_zero()
            || current_instruction == Instruction::Pow)
            && section[[row_idx, RHS.base_table_index()]].is_zero()
        {
            section[[row_idx, Result.base_table_index()]] = match current_instruction {
                Instruction::Split => bfe!(0),
                Instruction::Lt => bfe!(2),
                Instruction::And => bfe!(0),
                Instruction::Log2Floor => bfe!(-1),
                Instruction::Pow => bfe!(1),
                Instruction::PopCount => bfe!(0),
                _ => panic!("Must be u32 instruction, not {current_instruction}."),
            };

            // If instruction `lt` is executed on operands 0 and 0, the result is known to be 0.
            // The edge case can be reliably detected by checking whether column `Bits` is 0.
            let both_operands_are_0 = section[[row_idx, Bits.base_table_index()]].is_zero();
            if current_instruction == Instruction::Lt && both_operands_are_0 {
                section[[row_idx, Result.base_table_index()]] = bfe!(0);
            }

            // The right hand side is guaranteed to be 0. However, if the current instruction is
            // `pow`, then the left hand side might be non-zero.
            let lhs_inv_or_0 = section[[row_idx, LHS.base_table_index()]].inverse_or_zero();
            section[[row_idx, LhsInv.base_table_index()]] = lhs_inv_or_0;

            return section;
        }

        let lhs_lsb = bfe!(section[[row_idx, LHS.base_table_index()]].value() % 2);
        let rhs_lsb = bfe!(section[[row_idx, RHS.base_table_index()]].value() % 2);
        let mut next_row = section.row(row_idx).to_owned();
        next_row[CopyFlag.base_table_index()] = bfe!(0);
        next_row[Bits.base_table_index()] += bfe!(1);
        next_row[BitsMinus33Inv.base_table_index()] =
            (next_row[Bits.base_table_index()] - bfe!(33)).inverse();
        next_row[LHS.base_table_index()] = match current_instruction == Instruction::Pow {
            true => section[[row_idx, LHS.base_table_index()]],
            false => (section[[row_idx, LHS.base_table_index()]] - lhs_lsb) / bfe!(2),
        };
        next_row[RHS.base_table_index()] =
            (section[[row_idx, RHS.base_table_index()]] - rhs_lsb) / bfe!(2);
        next_row[LookupMultiplicity.base_table_index()] = bfe!(0);

        section.push_row(next_row.view()).unwrap();
        section = Self::u32_section_next_row(section);
        let (mut row, next_row) = section.multi_slice_mut((s![row_idx, ..], s![row_idx + 1, ..]));

        row[LhsInv.base_table_index()] = row[LHS.base_table_index()].inverse_or_zero();
        row[RhsInv.base_table_index()] = row[RHS.base_table_index()].inverse_or_zero();

        let next_row_result = next_row[Result.base_table_index()];
        row[Result.base_table_index()] = match current_instruction {
            Instruction::Split => next_row_result,
            Instruction::Lt => {
                match (
                    next_row_result.value(),
                    lhs_lsb.value(),
                    rhs_lsb.value(),
                    row[CopyFlag.base_table_index()].value(),
                ) {
                    (0 | 1, _, _, _) => next_row_result, // result already known
                    (2, 0, 1, _) => bfe!(1),             // LHS < RHS
                    (2, 1, 0, _) => bfe!(0),             // LHS > RHS
                    (2, _, _, 1) => bfe!(0),             // LHS == RHS
                    (2, _, _, 0) => bfe!(2),             // result still unknown
                    _ => panic!("Invalid state"),
                }
            }
            Instruction::And => bfe!(2) * next_row_result + lhs_lsb * rhs_lsb,
            Instruction::Log2Floor => {
                if row[LHS.base_table_index()].is_zero() {
                    bfe!(-1)
                } else if !next_row[LHS.base_table_index()].is_zero() {
                    next_row_result
                } else {
                    // LHS != 0 && LHS' == 0
                    row[Bits.base_table_index()]
                }
            }
            Instruction::Pow => match rhs_lsb.is_zero() {
                true => next_row_result * next_row_result,
                false => next_row_result * next_row_result * row[LHS.base_table_index()],
            },
            Instruction::PopCount => next_row_result + lhs_lsb,
            _ => panic!("Must be u32 instruction, not {current_instruction}."),
        };

        section
    }

    pub fn pad_trace(mut u32_table: ArrayViewMut2<BFieldElement>, u32_table_len: usize) {
        let mut padding_row = Array1::zeros([BASE_WIDTH]);
        padding_row[[CI.base_table_index()]] = Instruction::Split.opcode_b();
        padding_row[[BitsMinus33Inv.base_table_index()]] = bfe!(-33).inverse();

        if u32_table_len > 0 {
            let last_row = u32_table.row(u32_table_len - 1);
            padding_row[[CI.base_table_index()]] = last_row[CI.base_table_index()];
            padding_row[[LHS.base_table_index()]] = last_row[LHS.base_table_index()];
            padding_row[[LhsInv.base_table_index()]] = last_row[LhsInv.base_table_index()];
            padding_row[[Result.base_table_index()]] = last_row[Result.base_table_index()];

            // In the edge case that the last non-padding row comes from executing instruction
            // `lt` on operands 0 and 0, the `Result` column is 0. For the padding section,
            // where the `CopyFlag` is always 0, the `Result` needs to be set to 2 instead.
            if padding_row[[CI.base_table_index()]] == Instruction::Lt.opcode_b() {
                padding_row[[Result.base_table_index()]] = bfe!(2);
            }
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
        challenges: &Challenges,
    ) {
        profiler!(start "u32 table");
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());

        let ci_weight = challenges[U32CiWeight];
        let lhs_weight = challenges[U32LhsWeight];
        let rhs_weight = challenges[U32RhsWeight];
        let result_weight = challenges[U32ResultWeight];
        let lookup_indeterminate = challenges[U32Indeterminate];

        let mut running_sum_log_derivative = LookupArg::default_initial();
        for row_idx in 0..base_table.nrows() {
            let current_row = base_table.row(row_idx);
            if current_row[CopyFlag.base_table_index()].is_one() {
                let lookup_multiplicity = current_row[LookupMultiplicity.base_table_index()];
                let compressed_row = ci_weight * current_row[CI.base_table_index()]
                    + lhs_weight * current_row[LHS.base_table_index()]
                    + rhs_weight * current_row[RHS.base_table_index()]
                    + result_weight * current_row[Result.base_table_index()];
                running_sum_log_derivative +=
                    lookup_multiplicity * (lookup_indeterminate - compressed_row).inverse();
            }

            let mut extension_row = ext_table.row_mut(row_idx);
            extension_row[LookupServerLogDerivative.ext_table_index()] = running_sum_log_derivative;
        }
        profiler!(stop "u32 table");
    }
}
