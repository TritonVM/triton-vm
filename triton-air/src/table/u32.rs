use constraint_circuit::ConstraintCircuitBuilder;
use constraint_circuit::ConstraintCircuitMonad;
use constraint_circuit::DualRowIndicator;
use constraint_circuit::DualRowIndicator::CurrentAux;
use constraint_circuit::DualRowIndicator::CurrentMain;
use constraint_circuit::DualRowIndicator::NextAux;
use constraint_circuit::DualRowIndicator::NextMain;
use constraint_circuit::InputIndicator;
use constraint_circuit::SingleRowIndicator;
use constraint_circuit::SingleRowIndicator::Aux;
use constraint_circuit::SingleRowIndicator::Main;
use isa::instruction::Instruction;
use std::ops::Mul;

use crate::AIR;
use crate::challenge_id::ChallengeId;
use crate::cross_table_argument::CrossTableArg;
use crate::cross_table_argument::LookupArg;
use crate::table_column::MasterAuxColumn;
use crate::table_column::MasterMainColumn;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct U32Table;

impl crate::private::Seal for U32Table {}

impl AIR for U32Table {
    type MainColumn = crate::table_column::U32MainColumn;
    type AuxColumn = crate::table_column::U32AuxColumn;

    fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let main_row =
            |column: Self::MainColumn| circuit_builder.input(Main(column.master_main_index()));
        let aux_row =
            |column: Self::AuxColumn| circuit_builder.input(Aux(column.master_aux_index()));
        let challenge = |c| circuit_builder.challenge(c);
        let one = circuit_builder.b_constant(1);

        let copy_flag = main_row(Self::MainColumn::CopyFlag);
        let lhs = main_row(Self::MainColumn::LHS);
        let rhs = main_row(Self::MainColumn::RHS);
        let ci = main_row(Self::MainColumn::CI);
        let result = main_row(Self::MainColumn::Result);
        let lookup_multiplicity = main_row(Self::MainColumn::LookupMultiplicity);

        let running_sum_log_derivative = aux_row(Self::AuxColumn::LookupServerLogDerivative);

        let compressed_row = challenge(ChallengeId::U32LhsWeight) * lhs
            + challenge(ChallengeId::U32RhsWeight) * rhs
            + challenge(ChallengeId::U32CiWeight) * ci
            + challenge(ChallengeId::U32ResultWeight) * result;
        let if_copy_flag_is_1_then_log_derivative_has_accumulated_first_row = copy_flag.clone()
            * (running_sum_log_derivative.clone()
                * (challenge(ChallengeId::U32Indeterminate) - compressed_row)
                - lookup_multiplicity);

        let default_initial = circuit_builder.x_constant(LookupArg::default_initial());
        let if_copy_flag_is_0_then_log_derivative_is_default_initial =
            (copy_flag - one) * (running_sum_log_derivative - default_initial);

        let running_sum_log_derivative_starts_correctly =
            if_copy_flag_is_0_then_log_derivative_is_default_initial
                + if_copy_flag_is_1_then_log_derivative_has_accumulated_first_row;

        vec![running_sum_log_derivative_starts_correctly]
    }

    fn consistency_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let main_row =
            |column: Self::MainColumn| circuit_builder.input(Main(column.master_main_index()));
        let one = || circuit_builder.b_constant(1);
        let two = || circuit_builder.b_constant(2);

        let copy_flag = main_row(Self::MainColumn::CopyFlag);
        let bits = main_row(Self::MainColumn::Bits);
        let bits_minus_33_inv = main_row(Self::MainColumn::BitsMinus33Inv);
        let ci = main_row(Self::MainColumn::CI);
        let lhs = main_row(Self::MainColumn::LHS);
        let lhs_inv = main_row(Self::MainColumn::LhsInv);
        let rhs = main_row(Self::MainColumn::RHS);
        let rhs_inv = main_row(Self::MainColumn::RhsInv);
        let result = main_row(Self::MainColumn::Result);
        let lookup_multiplicity = main_row(Self::MainColumn::LookupMultiplicity);

        let instruction_deselector = |instruction_to_select| {
            instruction_deselector(instruction_to_select, circuit_builder, &ci)
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

    fn transition_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_main_row = |column: Self::MainColumn| {
            circuit_builder.input(CurrentMain(column.master_main_index()))
        };
        let next_main_row =
            |column: Self::MainColumn| circuit_builder.input(NextMain(column.master_main_index()));
        let curr_aux_row =
            |column: Self::AuxColumn| circuit_builder.input(CurrentAux(column.master_aux_index()));
        let next_aux_row =
            |column: Self::AuxColumn| circuit_builder.input(NextAux(column.master_aux_index()));
        let challenge = |c| circuit_builder.challenge(c);
        let one = || circuit_builder.b_constant(1);
        let two = || circuit_builder.b_constant(2);

        let copy_flag = curr_main_row(Self::MainColumn::CopyFlag);
        let bits = curr_main_row(Self::MainColumn::Bits);
        let ci = curr_main_row(Self::MainColumn::CI);
        let lhs = curr_main_row(Self::MainColumn::LHS);
        let rhs = curr_main_row(Self::MainColumn::RHS);
        let result = curr_main_row(Self::MainColumn::Result);
        let running_sum_log_derivative = curr_aux_row(Self::AuxColumn::LookupServerLogDerivative);

        let copy_flag_next = next_main_row(Self::MainColumn::CopyFlag);
        let bits_next = next_main_row(Self::MainColumn::Bits);
        let ci_next = next_main_row(Self::MainColumn::CI);
        let lhs_next = next_main_row(Self::MainColumn::LHS);
        let rhs_next = next_main_row(Self::MainColumn::RHS);
        let result_next = next_main_row(Self::MainColumn::Result);
        let lhs_inv_next = next_main_row(Self::MainColumn::LhsInv);
        let lookup_multiplicity_next = next_main_row(Self::MainColumn::LookupMultiplicity);
        let running_sum_log_derivative_next =
            next_aux_row(Self::AuxColumn::LookupServerLogDerivative);

        let instruction_deselector = |instruction_to_select: Instruction| {
            instruction_deselector(instruction_to_select, circuit_builder, &ci_next)
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

        let compressed_row_next = challenge(ChallengeId::U32CiWeight) * ci_next
            + challenge(ChallengeId::U32LhsWeight) * lhs_next
            + challenge(ChallengeId::U32RhsWeight) * rhs_next
            + challenge(ChallengeId::U32ResultWeight) * result_next;
        let if_copy_flag_next_is_1_then_running_sum_log_derivative_accumulates_next_row =
            copy_flag_next
                * ((running_sum_log_derivative_next - running_sum_log_derivative)
                    * (challenge(ChallengeId::U32Indeterminate) - compressed_row_next)
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

    fn terminal_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let main_row =
            |column: Self::MainColumn| circuit_builder.input(Main(column.master_main_index()));
        let constant = |c| circuit_builder.b_constant(c);

        let ci = main_row(Self::MainColumn::CI);
        let lhs = main_row(Self::MainColumn::LHS);
        let rhs = main_row(Self::MainColumn::RHS);

        let lhs_is_0_or_ci_is_pow = lhs * (ci - constant(Instruction::Pow.opcode_b()));
        let rhs_is_0 = rhs;

        vec![lhs_is_0_or_ci_is_pow, rhs_is_0]
    }
}

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
    .map(|instr| current_instruction.clone() - circuit_builder.b_constant(instr.opcode_b()))
    .fold(circuit_builder.b_constant(1), ConstraintCircuitMonad::mul)
}
