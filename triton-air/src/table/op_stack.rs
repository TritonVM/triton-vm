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
use isa::op_stack::OpStackElement;
use strum::EnumCount;
use twenty_first::prelude::*;

use crate::AIR;
use crate::challenge_id::ChallengeId;
use crate::cross_table_argument::CrossTableArg;
use crate::cross_table_argument::LookupArg;
use crate::cross_table_argument::PermArg;
use crate::table_column::MasterAuxColumn;
use crate::table_column::MasterMainColumn;

/// The value indicating a padding row in the op stack table. Stored in the
/// `ib1_shrink_stack` column.
pub const PADDING_VALUE: BFieldElement = BFieldElement::new(2);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct OpStackTable;

impl crate::private::Seal for OpStackTable {}

impl AIR for OpStackTable {
    type MainColumn = crate::table_column::OpStackMainColumn;
    type AuxColumn = crate::table_column::OpStackAuxColumn;

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

        let initial_stack_length = u32::try_from(OpStackElement::COUNT).unwrap();
        let initial_stack_length = constant(initial_stack_length.into());
        let padding_indicator = constant(PADDING_VALUE);

        let stack_pointer_is_16 =
            main_row(Self::MainColumn::StackPointer) - initial_stack_length.clone();

        let compressed_row = challenge(ChallengeId::OpStackClkWeight)
            * main_row(Self::MainColumn::CLK)
            + challenge(ChallengeId::OpStackIb1Weight) * main_row(Self::MainColumn::IB1ShrinkStack)
            + challenge(ChallengeId::OpStackPointerWeight) * initial_stack_length
            + challenge(ChallengeId::OpStackFirstUnderflowElementWeight)
                * main_row(Self::MainColumn::FirstUnderflowElement);
        let rppa_initial = challenge(ChallengeId::OpStackIndeterminate) - compressed_row;
        let rppa_has_accumulated_first_row =
            aux_row(Self::AuxColumn::RunningProductPermArg) - rppa_initial;

        let rppa_is_default_initial = aux_row(Self::AuxColumn::RunningProductPermArg)
            - x_constant(PermArg::default_initial());

        let first_row_is_padding_row =
            main_row(Self::MainColumn::IB1ShrinkStack) - padding_indicator;
        let first_row_is_not_padding_row = main_row(Self::MainColumn::IB1ShrinkStack)
            * (main_row(Self::MainColumn::IB1ShrinkStack) - constant(bfe!(1)));

        let rppa_starts_correctly = rppa_has_accumulated_first_row * first_row_is_padding_row
            + rppa_is_default_initial * first_row_is_not_padding_row;

        let lookup_argument_initial = x_constant(LookupArg::default_initial());
        let clock_jump_diff_log_derivative_is_initialized_correctly =
            aux_row(Self::AuxColumn::ClockJumpDifferenceLookupClientLogDerivative)
                - lookup_argument_initial;

        vec![
            stack_pointer_is_16,
            rppa_starts_correctly,
            clock_jump_diff_log_derivative_is_initialized_correctly,
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
        let current_main_row = |column: Self::MainColumn| {
            circuit_builder.input(CurrentMain(column.master_main_index()))
        };
        let current_aux_row =
            |column: Self::AuxColumn| circuit_builder.input(CurrentAux(column.master_aux_index()));
        let next_main_row =
            |column: Self::MainColumn| circuit_builder.input(NextMain(column.master_main_index()));
        let next_aux_row =
            |column: Self::AuxColumn| circuit_builder.input(NextAux(column.master_aux_index()));

        let one = constant(1_u32.into());
        let padding_indicator = constant(PADDING_VALUE);

        let clk = current_main_row(Self::MainColumn::CLK);
        let ib1_shrink_stack = current_main_row(Self::MainColumn::IB1ShrinkStack);
        let stack_pointer = current_main_row(Self::MainColumn::StackPointer);
        let first_underflow_element = current_main_row(Self::MainColumn::FirstUnderflowElement);
        let rppa = current_aux_row(Self::AuxColumn::RunningProductPermArg);
        let clock_jump_diff_log_derivative =
            current_aux_row(Self::AuxColumn::ClockJumpDifferenceLookupClientLogDerivative);

        let clk_next = next_main_row(Self::MainColumn::CLK);
        let ib1_shrink_stack_next = next_main_row(Self::MainColumn::IB1ShrinkStack);
        let stack_pointer_next = next_main_row(Self::MainColumn::StackPointer);
        let first_underflow_element_next = next_main_row(Self::MainColumn::FirstUnderflowElement);
        let rppa_next = next_aux_row(Self::AuxColumn::RunningProductPermArg);
        let clock_jump_diff_log_derivative_next =
            next_aux_row(Self::AuxColumn::ClockJumpDifferenceLookupClientLogDerivative);

        let stack_pointer_increases_by_1_or_does_not_change =
            (stack_pointer_next.clone() - stack_pointer.clone() - one.clone())
                * (stack_pointer_next.clone() - stack_pointer.clone());

        let stack_pointer_inc_by_1_or_underflow_element_doesnt_change_or_next_ci_grows_stack =
            (stack_pointer_next.clone() - stack_pointer.clone() - one.clone())
                * (first_underflow_element_next.clone() - first_underflow_element.clone())
                * ib1_shrink_stack_next.clone();

        let next_row_is_padding_row = ib1_shrink_stack_next.clone() - padding_indicator.clone();
        let if_current_row_is_padding_row_then_next_row_is_padding_row = ib1_shrink_stack.clone()
            * (ib1_shrink_stack - one.clone())
            * next_row_is_padding_row.clone();

        // The running product for the permutation argument `rppa` is updated
        // correctly.
        let compressed_row = circuit_builder.challenge(ChallengeId::OpStackClkWeight)
            * clk_next.clone()
            + circuit_builder.challenge(ChallengeId::OpStackIb1Weight)
                * ib1_shrink_stack_next.clone()
            + circuit_builder.challenge(ChallengeId::OpStackPointerWeight)
                * stack_pointer_next.clone()
            + circuit_builder.challenge(ChallengeId::OpStackFirstUnderflowElementWeight)
                * first_underflow_element_next;

        let rppa_updates = rppa_next.clone()
            - rppa.clone() * (challenge(ChallengeId::OpStackIndeterminate) - compressed_row);

        let next_row_is_not_padding_row =
            ib1_shrink_stack_next.clone() * (ib1_shrink_stack_next.clone() - one.clone());
        let rppa_remains = rppa_next - rppa;

        let rppa_updates_correctly = rppa_updates * next_row_is_padding_row.clone()
            + rppa_remains * next_row_is_not_padding_row.clone();

        let clk_diff = clk_next - clk;
        let log_derivative_accumulates = (clock_jump_diff_log_derivative_next.clone()
            - clock_jump_diff_log_derivative.clone())
            * (challenge(ChallengeId::ClockJumpDifferenceLookupIndeterminate) - clk_diff)
            - one.clone();
        let log_derivative_remains =
            clock_jump_diff_log_derivative_next.clone() - clock_jump_diff_log_derivative.clone();

        let log_derivative_accumulates_or_stack_pointer_changes_or_next_row_is_padding_row =
            log_derivative_accumulates
                * (stack_pointer_next.clone() - stack_pointer.clone() - one.clone())
                * next_row_is_padding_row;
        let log_derivative_remains_or_stack_pointer_doesnt_change =
            log_derivative_remains.clone() * (stack_pointer_next.clone() - stack_pointer.clone());
        let log_derivatve_remains_or_next_row_is_not_padding_row =
            log_derivative_remains * next_row_is_not_padding_row;

        let log_derivative_updates_correctly =
            log_derivative_accumulates_or_stack_pointer_changes_or_next_row_is_padding_row
                + log_derivative_remains_or_stack_pointer_doesnt_change
                + log_derivatve_remains_or_next_row_is_not_padding_row;

        vec![
            stack_pointer_increases_by_1_or_does_not_change,
            stack_pointer_inc_by_1_or_underflow_element_doesnt_change_or_next_ci_grows_stack,
            if_current_row_is_padding_row_then_next_row_is_padding_row,
            rppa_updates_correctly,
            log_derivative_updates_correctly,
        ]
    }

    fn terminal_constraints(
        _circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        // no further constraints
        vec![]
    }
}
