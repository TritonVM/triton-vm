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

use crate::AIR;
use crate::challenge_id::ChallengeId;
use crate::challenge_id::ChallengeId::CascadeLookupIndeterminate;
use crate::challenge_id::ChallengeId::LookupTableInputWeight;
use crate::challenge_id::ChallengeId::LookupTableOutputWeight;
use crate::challenge_id::ChallengeId::LookupTablePublicIndeterminate;
use crate::challenge_id::ChallengeId::LookupTablePublicTerminal;
use crate::cross_table_argument::CrossTableArg;
use crate::cross_table_argument::EvalArg;
use crate::cross_table_argument::LookupArg;
use crate::table_column::LookupAuxColumn;
use crate::table_column::LookupAuxColumn::CascadeTableServerLogDerivative;
use crate::table_column::LookupAuxColumn::PublicEvaluationArgument;
use crate::table_column::LookupMainColumn;
use crate::table_column::LookupMainColumn::IsPadding;
use crate::table_column::LookupMainColumn::LookIn;
use crate::table_column::LookupMainColumn::LookOut;
use crate::table_column::LookupMainColumn::LookupMultiplicity;
use crate::table_column::MasterAuxColumn;
use crate::table_column::MasterMainColumn;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct LookupTable;

impl crate::private::Seal for LookupTable {}

impl AIR for LookupTable {
    type MainColumn = LookupMainColumn;
    type AuxColumn = LookupAuxColumn;

    fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let main_row =
            |col_id: Self::MainColumn| circuit_builder.input(Main(col_id.master_main_index()));
        let aux_row =
            |col_id: Self::AuxColumn| circuit_builder.input(Aux(col_id.master_aux_index()));
        let challenge = |challenge_id: ChallengeId| circuit_builder.challenge(challenge_id);

        let lookup_input = main_row(LookIn);
        let lookup_output = main_row(LookOut);
        let lookup_multiplicity = main_row(LookupMultiplicity);
        let cascade_table_server_log_derivative = aux_row(CascadeTableServerLogDerivative);
        let public_evaluation_argument = aux_row(PublicEvaluationArgument);

        let lookup_input_is_0 = lookup_input;

        // Lookup Argument with Cascade Table
        // note: `lookup_input` is known to be 0 and thus doesn't appear in the
        // compressed row
        let lookup_argument_default_initial =
            circuit_builder.x_constant(LookupArg::default_initial());
        let cascade_table_indeterminate = challenge(CascadeLookupIndeterminate);
        let compressed_row = lookup_output.clone() * challenge(LookupTableOutputWeight);
        let cascade_table_log_derivative_is_initialized_correctly =
            (cascade_table_server_log_derivative - lookup_argument_default_initial)
                * (cascade_table_indeterminate - compressed_row)
                - lookup_multiplicity;

        // public Evaluation Argument
        let eval_argument_default_initial = circuit_builder.x_constant(EvalArg::default_initial());
        let public_indeterminate = challenge(LookupTablePublicIndeterminate);
        let public_evaluation_argument_is_initialized_correctly = public_evaluation_argument
            - eval_argument_default_initial * public_indeterminate
            - lookup_output;

        vec![
            lookup_input_is_0,
            cascade_table_log_derivative_is_initialized_correctly,
            public_evaluation_argument_is_initialized_correctly,
        ]
    }

    fn consistency_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let constant = |c: u64| circuit_builder.b_constant(c);
        let main_row =
            |col_id: Self::MainColumn| circuit_builder.input(Main(col_id.master_main_index()));

        let padding_is_0_or_1 = main_row(IsPadding) * (constant(1) - main_row(IsPadding));

        vec![padding_is_0_or_1]
    }

    fn transition_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let one = || circuit_builder.b_constant(1);

        let current_main_row = |col_id: Self::MainColumn| {
            circuit_builder.input(CurrentMain(col_id.master_main_index()))
        };
        let next_main_row =
            |col_id: Self::MainColumn| circuit_builder.input(NextMain(col_id.master_main_index()));
        let current_aux_row =
            |col_id: Self::AuxColumn| circuit_builder.input(CurrentAux(col_id.master_aux_index()));
        let next_aux_row =
            |col_id: Self::AuxColumn| circuit_builder.input(NextAux(col_id.master_aux_index()));
        let challenge = |challenge_id: ChallengeId| circuit_builder.challenge(challenge_id);

        let lookup_input = current_main_row(LookIn);
        let is_padding = current_main_row(IsPadding);
        let cascade_table_server_log_derivative = current_aux_row(CascadeTableServerLogDerivative);
        let public_evaluation_argument = current_aux_row(PublicEvaluationArgument);

        let lookup_input_next = next_main_row(LookIn);
        let lookup_output_next = next_main_row(LookOut);
        let lookup_multiplicity_next = next_main_row(LookupMultiplicity);
        let is_padding_next = next_main_row(IsPadding);
        let cascade_table_server_log_derivative_next =
            next_aux_row(CascadeTableServerLogDerivative);
        let public_evaluation_argument_next = next_aux_row(PublicEvaluationArgument);

        // Padding section is contiguous: if the current row is a padding row,
        // then the next row is also a padding row.
        let if_current_row_is_padding_row_then_next_row_is_padding_row =
            is_padding * (one() - is_padding_next.clone());

        // Lookup Table's input increments by 1 if and only if the next row is
        // not a padding row
        let if_next_row_is_padding_row_then_lookup_input_next_is_0 =
            is_padding_next.clone() * lookup_input_next.clone();
        let if_next_row_is_not_padding_row_then_lookup_input_next_increments_by_1 =
            (one() - is_padding_next.clone()) * (lookup_input_next.clone() - lookup_input - one());
        let lookup_input_increments_if_and_only_if_next_row_is_not_padding_row =
            if_next_row_is_padding_row_then_lookup_input_next_is_0
                + if_next_row_is_not_padding_row_then_lookup_input_next_increments_by_1;

        // Lookup Argument with Cascade Table
        let cascade_table_indeterminate = challenge(CascadeLookupIndeterminate);
        let compressed_row = lookup_input_next * challenge(LookupTableInputWeight)
            + lookup_output_next.clone() * challenge(LookupTableOutputWeight);
        let cascade_table_log_derivative_remains = cascade_table_server_log_derivative_next.clone()
            - cascade_table_server_log_derivative.clone();
        let cascade_table_log_derivative_updates = (cascade_table_server_log_derivative_next
            - cascade_table_server_log_derivative)
            * (cascade_table_indeterminate - compressed_row)
            - lookup_multiplicity_next;
        let cascade_table_log_derivative_updates_if_and_only_if_next_row_is_not_padding_row =
            (one() - is_padding_next.clone()) * cascade_table_log_derivative_updates
                + is_padding_next.clone() * cascade_table_log_derivative_remains;

        // public Evaluation Argument
        let public_indeterminate = challenge(LookupTablePublicIndeterminate);
        let public_evaluation_argument_remains =
            public_evaluation_argument_next.clone() - public_evaluation_argument.clone();
        let public_evaluation_argument_updates = public_evaluation_argument_next
            - public_evaluation_argument * public_indeterminate
            - lookup_output_next;
        let public_evaluation_argument_updates_if_and_only_if_next_row_is_not_padding_row =
            (one() - is_padding_next.clone()) * public_evaluation_argument_updates
                + is_padding_next * public_evaluation_argument_remains;

        vec![
            if_current_row_is_padding_row_then_next_row_is_padding_row,
            lookup_input_increments_if_and_only_if_next_row_is_not_padding_row,
            cascade_table_log_derivative_updates_if_and_only_if_next_row_is_not_padding_row,
            public_evaluation_argument_updates_if_and_only_if_next_row_is_not_padding_row,
        ]
    }

    fn terminal_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let challenge = |challenge_id: ChallengeId| circuit_builder.challenge(challenge_id);
        let aux_row =
            |col_id: Self::AuxColumn| circuit_builder.input(Aux(col_id.master_aux_index()));

        let narrow_table_terminal_matches_user_supplied_terminal =
            aux_row(PublicEvaluationArgument) - challenge(LookupTablePublicTerminal);

        vec![narrow_table_terminal_matches_user_supplied_terminal]
    }
}
