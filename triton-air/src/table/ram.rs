use constraint_circuit::ConstraintCircuitBuilder;
use constraint_circuit::ConstraintCircuitMonad;
use constraint_circuit::DualRowIndicator;
use constraint_circuit::DualRowIndicator::CurrentAux;
use constraint_circuit::DualRowIndicator::CurrentMain;
use constraint_circuit::DualRowIndicator::NextAux;
use constraint_circuit::DualRowIndicator::NextMain;
use constraint_circuit::SingleRowIndicator;
use constraint_circuit::SingleRowIndicator::Aux;
use constraint_circuit::SingleRowIndicator::Main;
use twenty_first::prelude::*;

use crate::AIR;
use crate::challenge_id::ChallengeId;
use crate::cross_table_argument::CrossTableArg;
use crate::cross_table_argument::LookupArg;
use crate::cross_table_argument::PermArg;
use crate::table_column::MasterAuxColumn;
use crate::table_column::MasterMainColumn;

pub const INSTRUCTION_TYPE_WRITE: BFieldElement = BFieldElement::new(0);
pub const INSTRUCTION_TYPE_READ: BFieldElement = BFieldElement::new(1);
pub const PADDING_INDICATOR: BFieldElement = BFieldElement::new(2);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct RamTable;

impl crate::private::Seal for RamTable {}

impl AIR for RamTable {
    type MainColumn = crate::table_column::RamMainColumn;
    type AuxColumn = crate::table_column::RamAuxColumn;

    fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let constant = |c| circuit_builder.b_constant(c);
        let x_constant = |c| circuit_builder.x_constant(c);
        let main_row =
            |column: Self::MainColumn| circuit_builder.input(Main(column.master_main_index()));
        let aux_row =
            |column: Self::AuxColumn| circuit_builder.input(Aux(column.master_aux_index()));

        let first_row_is_padding_row =
            main_row(Self::MainColumn::InstructionType) - constant(PADDING_INDICATOR);
        let first_row_is_not_padding_row = (main_row(Self::MainColumn::InstructionType)
            - constant(INSTRUCTION_TYPE_READ))
            * (main_row(Self::MainColumn::InstructionType) - constant(INSTRUCTION_TYPE_WRITE));

        let bezout_coefficient_polynomial_coefficient_0_is_0 =
            main_row(Self::MainColumn::BezoutCoefficientPolynomialCoefficient0);
        let bezout_coefficient_0_is_0 = aux_row(Self::AuxColumn::BezoutCoefficient0);
        let bezout_coefficient_1_is_bezout_coefficient_polynomial_coefficient_1 =
            aux_row(Self::AuxColumn::BezoutCoefficient1)
                - main_row(Self::MainColumn::BezoutCoefficientPolynomialCoefficient1);
        let formal_derivative_is_1 =
            aux_row(Self::AuxColumn::FormalDerivative) - constant(1_u32.into());
        let running_product_polynomial_is_initialized_correctly =
            aux_row(Self::AuxColumn::RunningProductOfRAMP)
                - challenge(ChallengeId::RamTableBezoutRelationIndeterminate)
                + main_row(Self::MainColumn::RamPointer);

        let clock_jump_diff_log_derivative_is_default_initial =
            aux_row(Self::AuxColumn::ClockJumpDifferenceLookupClientLogDerivative)
                - x_constant(LookupArg::default_initial());

        let compressed_row_for_permutation_argument = main_row(Self::MainColumn::CLK)
            * challenge(ChallengeId::RamClkWeight)
            + main_row(Self::MainColumn::InstructionType)
                * challenge(ChallengeId::RamInstructionTypeWeight)
            + main_row(Self::MainColumn::RamPointer) * challenge(ChallengeId::RamPointerWeight)
            + main_row(Self::MainColumn::RamValue) * challenge(ChallengeId::RamValueWeight);
        let running_product_permutation_argument_has_accumulated_first_row =
            aux_row(Self::AuxColumn::RunningProductPermArg)
                - challenge(ChallengeId::RamIndeterminate)
                + compressed_row_for_permutation_argument;
        let running_product_permutation_argument_is_default_initial =
            aux_row(Self::AuxColumn::RunningProductPermArg)
                - x_constant(PermArg::default_initial());

        let running_product_permutation_argument_starts_correctly =
            running_product_permutation_argument_has_accumulated_first_row
                * first_row_is_padding_row
                + running_product_permutation_argument_is_default_initial
                    * first_row_is_not_padding_row;

        vec![
            bezout_coefficient_polynomial_coefficient_0_is_0,
            bezout_coefficient_0_is_0,
            bezout_coefficient_1_is_bezout_coefficient_polynomial_coefficient_1,
            running_product_polynomial_is_initialized_correctly,
            formal_derivative_is_1,
            running_product_permutation_argument_starts_correctly,
            clock_jump_diff_log_derivative_is_default_initial,
        ]
    }

    fn consistency_constraints(
        _circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        // no further constraints
        vec![]
    }

    fn transition_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c| circuit_builder.b_constant(c);
        let challenge = |c| circuit_builder.challenge(c);
        let curr_main_row = |column: Self::MainColumn| {
            circuit_builder.input(CurrentMain(column.master_main_index()))
        };
        let curr_aux_row =
            |column: Self::AuxColumn| circuit_builder.input(CurrentAux(column.master_aux_index()));
        let next_main_row =
            |column: Self::MainColumn| circuit_builder.input(NextMain(column.master_main_index()));
        let next_aux_row =
            |column: Self::AuxColumn| circuit_builder.input(NextAux(column.master_aux_index()));

        let one = constant(1_u32.into());

        let bezout_challenge = challenge(ChallengeId::RamTableBezoutRelationIndeterminate);

        let clock = curr_main_row(Self::MainColumn::CLK);
        let ram_pointer = curr_main_row(Self::MainColumn::RamPointer);
        let ram_value = curr_main_row(Self::MainColumn::RamValue);
        let instruction_type = curr_main_row(Self::MainColumn::InstructionType);
        let inverse_of_ram_pointer_difference =
            curr_main_row(Self::MainColumn::InverseOfRampDifference);
        let bcpc0 = curr_main_row(Self::MainColumn::BezoutCoefficientPolynomialCoefficient0);
        let bcpc1 = curr_main_row(Self::MainColumn::BezoutCoefficientPolynomialCoefficient1);

        let running_product_ram_pointer = curr_aux_row(Self::AuxColumn::RunningProductOfRAMP);
        let fd = curr_aux_row(Self::AuxColumn::FormalDerivative);
        let bc0 = curr_aux_row(Self::AuxColumn::BezoutCoefficient0);
        let bc1 = curr_aux_row(Self::AuxColumn::BezoutCoefficient1);
        let rppa = curr_aux_row(Self::AuxColumn::RunningProductPermArg);
        let clock_jump_diff_log_derivative =
            curr_aux_row(Self::AuxColumn::ClockJumpDifferenceLookupClientLogDerivative);

        let clock_next = next_main_row(Self::MainColumn::CLK);
        let ram_pointer_next = next_main_row(Self::MainColumn::RamPointer);
        let ram_value_next = next_main_row(Self::MainColumn::RamValue);
        let instruction_type_next = next_main_row(Self::MainColumn::InstructionType);
        let bcpc0_next = next_main_row(Self::MainColumn::BezoutCoefficientPolynomialCoefficient0);
        let bcpc1_next = next_main_row(Self::MainColumn::BezoutCoefficientPolynomialCoefficient1);

        let running_product_ram_pointer_next = next_aux_row(Self::AuxColumn::RunningProductOfRAMP);
        let fd_next = next_aux_row(Self::AuxColumn::FormalDerivative);
        let bc0_next = next_aux_row(Self::AuxColumn::BezoutCoefficient0);
        let bc1_next = next_aux_row(Self::AuxColumn::BezoutCoefficient1);
        let rppa_next = next_aux_row(Self::AuxColumn::RunningProductPermArg);
        let clock_jump_diff_log_derivative_next =
            next_aux_row(Self::AuxColumn::ClockJumpDifferenceLookupClientLogDerivative);

        let next_row_is_padding_row =
            instruction_type_next.clone() - constant(PADDING_INDICATOR).clone();
        let if_current_row_is_padding_row_then_next_row_is_padding_row = (instruction_type.clone()
            - constant(INSTRUCTION_TYPE_READ))
            * (instruction_type - constant(INSTRUCTION_TYPE_WRITE))
            * next_row_is_padding_row.clone();

        let ram_pointer_difference = ram_pointer_next.clone() - ram_pointer;
        let ram_pointer_changes = one.clone()
            - ram_pointer_difference.clone() * inverse_of_ram_pointer_difference.clone();

        let iord_is_0_or_iord_is_inverse_of_ram_pointer_difference =
            inverse_of_ram_pointer_difference * ram_pointer_changes.clone();

        let ram_pointer_difference_is_0_or_iord_is_inverse_of_ram_pointer_difference =
            ram_pointer_difference.clone() * ram_pointer_changes.clone();

        let ram_pointer_changes_or_write_mem_or_ram_value_stays = ram_pointer_changes.clone()
            * (constant(INSTRUCTION_TYPE_WRITE) - instruction_type_next.clone())
            * (ram_value_next.clone() - ram_value);

        let bcbp0_only_changes_if_ram_pointer_changes =
            ram_pointer_changes.clone() * (bcpc0_next.clone() - bcpc0);

        let bcbp1_only_changes_if_ram_pointer_changes =
            ram_pointer_changes.clone() * (bcpc1_next.clone() - bcpc1);

        let running_product_ram_pointer_updates_correctly = ram_pointer_difference.clone()
            * (running_product_ram_pointer_next.clone()
                - running_product_ram_pointer.clone()
                    * (bezout_challenge.clone() - ram_pointer_next.clone()))
            + ram_pointer_changes.clone()
                * (running_product_ram_pointer_next - running_product_ram_pointer.clone());

        let formal_derivative_updates_correctly = ram_pointer_difference.clone()
            * (fd_next.clone()
                - running_product_ram_pointer
                - (bezout_challenge.clone() - ram_pointer_next.clone()) * fd.clone())
            + ram_pointer_changes.clone() * (fd_next - fd);

        let bezout_coefficient_0_is_constructed_correctly = ram_pointer_difference.clone()
            * (bc0_next.clone() - bezout_challenge.clone() * bc0.clone() - bcpc0_next)
            + ram_pointer_changes.clone() * (bc0_next - bc0);

        let bezout_coefficient_1_is_constructed_correctly = ram_pointer_difference.clone()
            * (bc1_next.clone() - bezout_challenge * bc1.clone() - bcpc1_next)
            + ram_pointer_changes.clone() * (bc1_next - bc1);

        let compressed_row = clock_next.clone() * challenge(ChallengeId::RamClkWeight)
            + ram_pointer_next * challenge(ChallengeId::RamPointerWeight)
            + ram_value_next * challenge(ChallengeId::RamValueWeight)
            + instruction_type_next.clone() * challenge(ChallengeId::RamInstructionTypeWeight);
        let rppa_accumulates_next_row = rppa_next.clone()
            - rppa.clone() * (challenge(ChallengeId::RamIndeterminate) - compressed_row);

        let next_row_is_not_padding_row = (instruction_type_next.clone()
            - constant(INSTRUCTION_TYPE_READ))
            * (instruction_type_next - constant(INSTRUCTION_TYPE_WRITE));
        let rppa_remains_unchanged = rppa_next - rppa;

        let rppa_updates_correctly = rppa_accumulates_next_row * next_row_is_padding_row.clone()
            + rppa_remains_unchanged * next_row_is_not_padding_row.clone();

        let clock_difference = clock_next - clock;
        let log_derivative_accumulates = (clock_jump_diff_log_derivative_next.clone()
            - clock_jump_diff_log_derivative.clone())
            * (challenge(ChallengeId::ClockJumpDifferenceLookupIndeterminate) - clock_difference)
            - one.clone();
        let log_derivative_remains =
            clock_jump_diff_log_derivative_next - clock_jump_diff_log_derivative.clone();

        let log_derivative_accumulates_or_ram_pointer_changes_or_next_row_is_padding_row =
            log_derivative_accumulates * ram_pointer_changes.clone() * next_row_is_padding_row;
        let log_derivative_remains_or_ram_pointer_doesnt_change =
            log_derivative_remains.clone() * ram_pointer_difference.clone();
        let log_derivative_remains_or_next_row_is_not_padding_row =
            log_derivative_remains * next_row_is_not_padding_row;

        let log_derivative_updates_correctly =
            log_derivative_accumulates_or_ram_pointer_changes_or_next_row_is_padding_row
                + log_derivative_remains_or_ram_pointer_doesnt_change
                + log_derivative_remains_or_next_row_is_not_padding_row;

        vec![
            if_current_row_is_padding_row_then_next_row_is_padding_row,
            iord_is_0_or_iord_is_inverse_of_ram_pointer_difference,
            ram_pointer_difference_is_0_or_iord_is_inverse_of_ram_pointer_difference,
            ram_pointer_changes_or_write_mem_or_ram_value_stays,
            bcbp0_only_changes_if_ram_pointer_changes,
            bcbp1_only_changes_if_ram_pointer_changes,
            running_product_ram_pointer_updates_correctly,
            formal_derivative_updates_correctly,
            bezout_coefficient_0_is_constructed_correctly,
            bezout_coefficient_1_is_constructed_correctly,
            rppa_updates_correctly,
            log_derivative_updates_correctly,
        ]
    }

    fn terminal_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let aux_row =
            |column: Self::AuxColumn| circuit_builder.input(Aux(column.master_aux_index()));

        let bezout_relation_holds = aux_row(Self::AuxColumn::BezoutCoefficient0)
            * aux_row(Self::AuxColumn::RunningProductOfRAMP)
            + aux_row(Self::AuxColumn::BezoutCoefficient1)
                * aux_row(Self::AuxColumn::FormalDerivative)
            - constant(1);

        vec![bezout_relation_holds]
    }
}
