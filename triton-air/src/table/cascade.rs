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
use crate::challenge_id::ChallengeId::HashCascadeLookInWeight;
use crate::challenge_id::ChallengeId::HashCascadeLookOutWeight;
use crate::challenge_id::ChallengeId::HashCascadeLookupIndeterminate;
use crate::challenge_id::ChallengeId::LookupTableInputWeight;
use crate::challenge_id::ChallengeId::LookupTableOutputWeight;
use crate::cross_table_argument::CrossTableArg;
use crate::cross_table_argument::LookupArg;
use crate::table_column::MasterAuxColumn;
use crate::table_column::MasterMainColumn;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct CascadeTable;

impl crate::private::Seal for CascadeTable {}

impl AIR for CascadeTable {
    type MainColumn = crate::table_column::CascadeMainColumn;
    type AuxColumn = crate::table_column::CascadeAuxColumn;

    fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let main_row =
            |col_id: Self::MainColumn| circuit_builder.input(Main(col_id.master_main_index()));
        let aux_row =
            |col_id: Self::AuxColumn| circuit_builder.input(Aux(col_id.master_aux_index()));
        let challenge = |challenge_id: ChallengeId| circuit_builder.challenge(challenge_id);

        let one = || circuit_builder.b_constant(1);
        let two = || circuit_builder.b_constant(2);
        let two_pow_8 = circuit_builder.b_constant(1 << 8);
        let lookup_arg_default_initial = circuit_builder.x_constant(LookupArg::default_initial());

        let is_padding = main_row(Self::MainColumn::IsPadding);
        let look_in_hi = main_row(Self::MainColumn::LookInHi);
        let look_in_lo = main_row(Self::MainColumn::LookInLo);
        let look_out_hi = main_row(Self::MainColumn::LookOutHi);
        let look_out_lo = main_row(Self::MainColumn::LookOutLo);
        let lookup_multiplicity = main_row(Self::MainColumn::LookupMultiplicity);
        let hash_table_server_log_derivative =
            aux_row(Self::AuxColumn::HashTableServerLogDerivative);
        let lookup_table_client_log_derivative =
            aux_row(Self::AuxColumn::LookupTableClientLogDerivative);

        let hash_indeterminate = challenge(HashCascadeLookupIndeterminate);
        let hash_input_weight = challenge(HashCascadeLookInWeight);
        let hash_output_weight = challenge(HashCascadeLookOutWeight);

        let lookup_indeterminate = challenge(CascadeLookupIndeterminate);
        let lookup_input_weight = challenge(LookupTableInputWeight);
        let lookup_output_weight = challenge(LookupTableOutputWeight);

        // Lookup Argument with Hash Table
        let compressed_row_hash = hash_input_weight
            * (two_pow_8.clone() * look_in_hi.clone() + look_in_lo.clone())
            + hash_output_weight * (two_pow_8 * look_out_hi.clone() + look_out_lo.clone());
        let hash_table_log_derivative_is_default_initial =
            hash_table_server_log_derivative.clone() - lookup_arg_default_initial.clone();
        let hash_table_log_derivative_has_accumulated_first_row = (hash_table_server_log_derivative
            - lookup_arg_default_initial.clone())
            * (hash_indeterminate - compressed_row_hash)
            - lookup_multiplicity;
        let hash_table_log_derivative_is_initialized_correctly = (one() - is_padding.clone())
            * hash_table_log_derivative_has_accumulated_first_row
            + is_padding.clone() * hash_table_log_derivative_is_default_initial;

        // Lookup Argument with Lookup Table
        let compressed_row_lo =
            lookup_input_weight.clone() * look_in_lo + lookup_output_weight.clone() * look_out_lo;
        let compressed_row_hi =
            lookup_input_weight * look_in_hi + lookup_output_weight * look_out_hi;
        let lookup_table_log_derivative_is_default_initial =
            lookup_table_client_log_derivative.clone() - lookup_arg_default_initial.clone();
        let lookup_table_log_derivative_has_accumulated_first_row =
            (lookup_table_client_log_derivative - lookup_arg_default_initial)
                * (lookup_indeterminate.clone() - compressed_row_lo.clone())
                * (lookup_indeterminate.clone() - compressed_row_hi.clone())
                - two() * lookup_indeterminate
                + compressed_row_lo
                + compressed_row_hi;
        let lookup_table_log_derivative_is_initialized_correctly = (one() - is_padding.clone())
            * lookup_table_log_derivative_has_accumulated_first_row
            + is_padding * lookup_table_log_derivative_is_default_initial;

        vec![
            hash_table_log_derivative_is_initialized_correctly,
            lookup_table_log_derivative_is_initialized_correctly,
        ]
    }

    fn consistency_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let row =
            |col_id: Self::MainColumn| circuit_builder.input(Main(col_id.master_main_index()));

        let one = circuit_builder.b_constant(1);
        let is_padding = row(Self::MainColumn::IsPadding);
        let is_padding_is_0_or_1 = is_padding.clone() * (one - is_padding);

        vec![is_padding_is_0_or_1]
    }

    fn transition_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let constant = |c: u64| circuit_builder.b_constant(c);

        let curr_main_row = |column_idx: Self::MainColumn| {
            circuit_builder.input(CurrentMain(column_idx.master_main_index()))
        };
        let next_main_row = |column_idx: Self::MainColumn| {
            circuit_builder.input(NextMain(column_idx.master_main_index()))
        };
        let curr_aux_row = |column_idx: Self::AuxColumn| {
            circuit_builder.input(CurrentAux(column_idx.master_aux_index()))
        };
        let next_aux_row = |column_idx: Self::AuxColumn| {
            circuit_builder.input(NextAux(column_idx.master_aux_index()))
        };

        let one = constant(1);
        let two = constant(2);
        let two_pow_8 = constant(1 << 8);

        let is_padding = curr_main_row(Self::MainColumn::IsPadding);
        let hash_table_server_log_derivative =
            curr_aux_row(Self::AuxColumn::HashTableServerLogDerivative);
        let lookup_table_client_log_derivative =
            curr_aux_row(Self::AuxColumn::LookupTableClientLogDerivative);

        let is_padding_next = next_main_row(Self::MainColumn::IsPadding);
        let look_in_hi_next = next_main_row(Self::MainColumn::LookInHi);
        let look_in_lo_next = next_main_row(Self::MainColumn::LookInLo);
        let look_out_hi_next = next_main_row(Self::MainColumn::LookOutHi);
        let look_out_lo_next = next_main_row(Self::MainColumn::LookOutLo);
        let lookup_multiplicity_next = next_main_row(Self::MainColumn::LookupMultiplicity);
        let hash_table_server_log_derivative_next =
            next_aux_row(Self::AuxColumn::HashTableServerLogDerivative);
        let lookup_table_client_log_derivative_next =
            next_aux_row(Self::AuxColumn::LookupTableClientLogDerivative);

        let hash_indeterminate = challenge(HashCascadeLookupIndeterminate);
        let hash_input_weight = challenge(HashCascadeLookInWeight);
        let hash_output_weight = challenge(HashCascadeLookOutWeight);

        let lookup_indeterminate = challenge(CascadeLookupIndeterminate);
        let lookup_input_weight = challenge(LookupTableInputWeight);
        let lookup_output_weight = challenge(LookupTableOutputWeight);

        // Contiguous padding: if current row is padding, then so is next row
        let if_current_row_is_padding_row_then_next_row_is_padding_row =
            is_padding * (one.clone() - is_padding_next.clone());

        // Lookup Argument with Hash Table
        let compressed_next_row_hash = hash_input_weight
            * (two_pow_8.clone() * look_in_hi_next.clone() + look_in_lo_next.clone())
            + hash_output_weight
                * (two_pow_8 * look_out_hi_next.clone() + look_out_lo_next.clone());
        let hash_table_log_derivative_remains = hash_table_server_log_derivative_next.clone()
            - hash_table_server_log_derivative.clone();
        let hash_table_log_derivative_accumulates_next_row = (hash_table_server_log_derivative_next
            - hash_table_server_log_derivative)
            * (hash_indeterminate - compressed_next_row_hash)
            - lookup_multiplicity_next;
        let hash_table_log_derivative_updates_correctly = (one.clone() - is_padding_next.clone())
            * hash_table_log_derivative_accumulates_next_row
            + is_padding_next.clone() * hash_table_log_derivative_remains;

        // Lookup Argument with Lookup Table
        let compressed_row_lo_next = lookup_input_weight.clone() * look_in_lo_next
            + lookup_output_weight.clone() * look_out_lo_next;
        let compressed_row_hi_next =
            lookup_input_weight * look_in_hi_next + lookup_output_weight * look_out_hi_next;
        let lookup_table_log_derivative_remains = lookup_table_client_log_derivative_next.clone()
            - lookup_table_client_log_derivative.clone();
        let lookup_table_log_derivative_accumulates_next_row =
            (lookup_table_client_log_derivative_next - lookup_table_client_log_derivative)
                * (lookup_indeterminate.clone() - compressed_row_lo_next.clone())
                * (lookup_indeterminate.clone() - compressed_row_hi_next.clone())
                - two * lookup_indeterminate
                + compressed_row_lo_next
                + compressed_row_hi_next;
        let lookup_table_log_derivative_updates_correctly = (one - is_padding_next.clone())
            * lookup_table_log_derivative_accumulates_next_row
            + is_padding_next * lookup_table_log_derivative_remains;

        vec![
            if_current_row_is_padding_row_then_next_row_is_padding_row,
            hash_table_log_derivative_updates_correctly,
            lookup_table_log_derivative_updates_correctly,
        ]
    }

    fn terminal_constraints(
        _circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        // no further constraints
        vec![]
    }
}
