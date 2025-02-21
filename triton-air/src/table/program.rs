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
use crate::cross_table_argument::EvalArg;
use crate::cross_table_argument::LookupArg;
use crate::table_column::MasterAuxColumn;
use crate::table_column::MasterMainColumn;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ProgramTable;

impl crate::private::Seal for ProgramTable {}

impl AIR for ProgramTable {
    type MainColumn = crate::table_column::ProgramMainColumn;
    type AuxColumn = crate::table_column::ProgramAuxColumn;

    fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let x_constant = |xfe| circuit_builder.x_constant(xfe);
        let main_row = |col: Self::MainColumn| circuit_builder.input(Main(col.master_main_index()));
        let aux_row = |col: Self::AuxColumn| circuit_builder.input(Aux(col.master_aux_index()));

        let address = main_row(Self::MainColumn::Address);
        let instruction = main_row(Self::MainColumn::Instruction);
        let index_in_chunk = main_row(Self::MainColumn::IndexInChunk);
        let is_hash_input_padding = main_row(Self::MainColumn::IsHashInputPadding);
        let instruction_lookup_log_derivative =
            aux_row(Self::AuxColumn::InstructionLookupServerLogDerivative);
        let prepare_chunk_running_evaluation =
            aux_row(Self::AuxColumn::PrepareChunkRunningEvaluation);
        let send_chunk_running_evaluation = aux_row(Self::AuxColumn::SendChunkRunningEvaluation);

        let lookup_arg_initial = x_constant(LookupArg::default_initial());
        let eval_arg_initial = x_constant(EvalArg::default_initial());

        let program_attestation_prepare_chunk_indeterminate =
            challenge(ChallengeId::ProgramAttestationPrepareChunkIndeterminate);

        let first_address_is_zero = address;
        let index_in_chunk_is_zero = index_in_chunk;
        let hash_input_padding_indicator_is_zero = is_hash_input_padding;

        let instruction_lookup_log_derivative_is_initialized_correctly =
            instruction_lookup_log_derivative - lookup_arg_initial;

        let prepare_chunk_running_evaluation_has_absorbed_first_instruction =
            prepare_chunk_running_evaluation
                - eval_arg_initial.clone() * program_attestation_prepare_chunk_indeterminate
                - instruction;

        let send_chunk_running_evaluation_is_default_initial =
            send_chunk_running_evaluation - eval_arg_initial;

        vec![
            first_address_is_zero,
            index_in_chunk_is_zero,
            hash_input_padding_indicator_is_zero,
            instruction_lookup_log_derivative_is_initialized_correctly,
            prepare_chunk_running_evaluation_has_absorbed_first_instruction,
            send_chunk_running_evaluation_is_default_initial,
        ]
    }

    fn consistency_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let main_row = |col: Self::MainColumn| circuit_builder.input(Main(col.master_main_index()));

        let one = constant(1);
        let max_index_in_chunk = constant((Tip5::RATE - 1).try_into().unwrap());

        let index_in_chunk = main_row(Self::MainColumn::IndexInChunk);
        let max_minus_index_in_chunk_inv = main_row(Self::MainColumn::MaxMinusIndexInChunkInv);
        let is_hash_input_padding = main_row(Self::MainColumn::IsHashInputPadding);
        let is_table_padding = main_row(Self::MainColumn::IsTablePadding);

        let max_minus_index_in_chunk = max_index_in_chunk - index_in_chunk;
        let max_minus_index_in_chunk_inv_is_zero_or_the_inverse_of_max_minus_index_in_chunk =
            (one.clone() - max_minus_index_in_chunk.clone() * max_minus_index_in_chunk_inv.clone())
                * max_minus_index_in_chunk_inv.clone();
        let max_minus_index_in_chunk_is_zero_or_the_inverse_of_max_minus_index_in_chunk_inv =
            (one.clone() - max_minus_index_in_chunk.clone() * max_minus_index_in_chunk_inv)
                * max_minus_index_in_chunk;

        let is_hash_input_padding_is_bit =
            is_hash_input_padding.clone() * (is_hash_input_padding - one.clone());
        let is_table_padding_is_bit = is_table_padding.clone() * (is_table_padding - one);

        vec![
            max_minus_index_in_chunk_inv_is_zero_or_the_inverse_of_max_minus_index_in_chunk,
            max_minus_index_in_chunk_is_zero_or_the_inverse_of_max_minus_index_in_chunk_inv,
            is_hash_input_padding_is_bit,
            is_table_padding_is_bit,
        ]
    }

    fn transition_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let constant = |c: u64| circuit_builder.b_constant(c);

        let current_main_row =
            |col: Self::MainColumn| circuit_builder.input(CurrentMain(col.master_main_index()));
        let next_main_row =
            |col: Self::MainColumn| circuit_builder.input(NextMain(col.master_main_index()));
        let current_aux_row =
            |col: Self::AuxColumn| circuit_builder.input(CurrentAux(col.master_aux_index()));
        let next_aux_row =
            |col: Self::AuxColumn| circuit_builder.input(NextAux(col.master_aux_index()));

        let one = constant(1);
        let rate_minus_one = constant(u64::try_from(Tip5::RATE).unwrap() - 1);

        let prepare_chunk_indeterminate =
            challenge(ChallengeId::ProgramAttestationPrepareChunkIndeterminate);
        let send_chunk_indeterminate =
            challenge(ChallengeId::ProgramAttestationSendChunkIndeterminate);

        let address = current_main_row(Self::MainColumn::Address);
        let instruction = current_main_row(Self::MainColumn::Instruction);
        let lookup_multiplicity = current_main_row(Self::MainColumn::LookupMultiplicity);
        let index_in_chunk = current_main_row(Self::MainColumn::IndexInChunk);
        let max_minus_index_in_chunk_inv =
            current_main_row(Self::MainColumn::MaxMinusIndexInChunkInv);
        let is_hash_input_padding = current_main_row(Self::MainColumn::IsHashInputPadding);
        let is_table_padding = current_main_row(Self::MainColumn::IsTablePadding);
        let log_derivative = current_aux_row(Self::AuxColumn::InstructionLookupServerLogDerivative);
        let prepare_chunk_running_evaluation =
            current_aux_row(Self::AuxColumn::PrepareChunkRunningEvaluation);
        let send_chunk_running_evaluation =
            current_aux_row(Self::AuxColumn::SendChunkRunningEvaluation);

        let address_next = next_main_row(Self::MainColumn::Address);
        let instruction_next = next_main_row(Self::MainColumn::Instruction);
        let index_in_chunk_next = next_main_row(Self::MainColumn::IndexInChunk);
        let max_minus_index_in_chunk_inv_next =
            next_main_row(Self::MainColumn::MaxMinusIndexInChunkInv);
        let is_hash_input_padding_next = next_main_row(Self::MainColumn::IsHashInputPadding);
        let is_table_padding_next = next_main_row(Self::MainColumn::IsTablePadding);
        let log_derivative_next =
            next_aux_row(Self::AuxColumn::InstructionLookupServerLogDerivative);
        let prepare_chunk_running_evaluation_next =
            next_aux_row(Self::AuxColumn::PrepareChunkRunningEvaluation);
        let send_chunk_running_evaluation_next =
            next_aux_row(Self::AuxColumn::SendChunkRunningEvaluation);

        let address_increases_by_one = address_next - (address.clone() + one.clone());
        let is_table_padding_is_0_or_remains_unchanged =
            is_table_padding.clone() * (is_table_padding_next.clone() - is_table_padding);

        let index_in_chunk_cycles_correctly = (one.clone()
            - max_minus_index_in_chunk_inv.clone()
                * (rate_minus_one.clone() - index_in_chunk.clone()))
            * index_in_chunk_next.clone()
            + max_minus_index_in_chunk_inv.clone()
                * (index_in_chunk_next.clone() - index_in_chunk.clone() - one.clone());

        let hash_input_indicator_is_0_or_remains_unchanged =
            is_hash_input_padding.clone() * (is_hash_input_padding_next.clone() - one.clone());

        let first_hash_input_padding_is_1 = (is_hash_input_padding.clone() - one.clone())
            * is_hash_input_padding_next
            * (instruction_next.clone() - one.clone());

        let hash_input_padding_is_0_after_the_first_1 =
            is_hash_input_padding.clone() * instruction_next.clone();

        let next_row_is_table_padding_row = is_table_padding_next.clone() - one.clone();
        let table_padding_starts_when_hash_input_padding_is_active_and_index_in_chunk_is_zero =
            is_hash_input_padding.clone()
                * (one.clone()
                    - max_minus_index_in_chunk_inv.clone()
                        * (rate_minus_one.clone() - index_in_chunk.clone()))
                * next_row_is_table_padding_row.clone();

        let log_derivative_remains = log_derivative_next.clone() - log_derivative.clone();
        let compressed_row = challenge(ChallengeId::ProgramAddressWeight) * address
            + challenge(ChallengeId::ProgramInstructionWeight) * instruction
            + challenge(ChallengeId::ProgramNextInstructionWeight) * instruction_next.clone();

        let indeterminate = challenge(ChallengeId::InstructionLookupIndeterminate);
        let log_derivative_updates = (log_derivative_next - log_derivative)
            * (indeterminate - compressed_row)
            - lookup_multiplicity;
        let log_derivative_updates_if_and_only_if_not_a_padding_row =
            (one.clone() - is_hash_input_padding.clone()) * log_derivative_updates
                + is_hash_input_padding * log_derivative_remains;

        let prepare_chunk_running_evaluation_absorbs_next_instruction =
            prepare_chunk_running_evaluation_next.clone()
                - prepare_chunk_indeterminate.clone() * prepare_chunk_running_evaluation
                - instruction_next.clone();
        let prepare_chunk_running_evaluation_resets_and_absorbs_next_instruction =
            prepare_chunk_running_evaluation_next.clone()
                - prepare_chunk_indeterminate
                - instruction_next;
        let index_in_chunk_is_max = rate_minus_one.clone() - index_in_chunk.clone();
        let index_in_chunk_is_not_max =
            one.clone() - max_minus_index_in_chunk_inv * (rate_minus_one.clone() - index_in_chunk);
        let prepare_chunk_running_evaluation_resets_every_rate_rows_and_absorbs_next_instruction =
            index_in_chunk_is_max * prepare_chunk_running_evaluation_absorbs_next_instruction
                + index_in_chunk_is_not_max
                    * prepare_chunk_running_evaluation_resets_and_absorbs_next_instruction;

        let send_chunk_running_evaluation_absorbs_next_chunk = send_chunk_running_evaluation_next
            .clone()
            - send_chunk_indeterminate * send_chunk_running_evaluation.clone()
            - prepare_chunk_running_evaluation_next;
        let send_chunk_running_evaluation_does_not_change =
            send_chunk_running_evaluation_next - send_chunk_running_evaluation;
        let index_in_chunk_next_is_max = rate_minus_one - index_in_chunk_next;
        let index_in_chunk_next_is_not_max =
            one - max_minus_index_in_chunk_inv_next * index_in_chunk_next_is_max.clone();

        let send_chunk_running_eval_absorbs_chunk_iff_index_in_chunk_next_is_max_and_not_padding =
            send_chunk_running_evaluation_absorbs_next_chunk
                * next_row_is_table_padding_row
                * index_in_chunk_next_is_not_max
                + send_chunk_running_evaluation_does_not_change.clone() * is_table_padding_next
                + send_chunk_running_evaluation_does_not_change * index_in_chunk_next_is_max;

        vec![
            address_increases_by_one,
            is_table_padding_is_0_or_remains_unchanged,
            index_in_chunk_cycles_correctly,
            hash_input_indicator_is_0_or_remains_unchanged,
            first_hash_input_padding_is_1,
            hash_input_padding_is_0_after_the_first_1,
            table_padding_starts_when_hash_input_padding_is_active_and_index_in_chunk_is_zero,
            log_derivative_updates_if_and_only_if_not_a_padding_row,
            prepare_chunk_running_evaluation_resets_every_rate_rows_and_absorbs_next_instruction,
            send_chunk_running_eval_absorbs_chunk_iff_index_in_chunk_next_is_max_and_not_padding,
        ]
    }

    fn terminal_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let constant = |c: u64| circuit_builder.b_constant(c);
        let main_row = |col: Self::MainColumn| circuit_builder.input(Main(col.master_main_index()));

        let index_in_chunk = main_row(Self::MainColumn::IndexInChunk);
        let is_hash_input_padding = main_row(Self::MainColumn::IsHashInputPadding);
        let is_table_padding = main_row(Self::MainColumn::IsTablePadding);

        let hash_input_padding_is_one = is_hash_input_padding - constant(1);

        let max_index_in_chunk = Tip5::RATE as u64 - 1;
        let index_in_chunk_is_max_or_row_is_padding_row =
            (index_in_chunk - constant(max_index_in_chunk)) * (is_table_padding - constant(1));

        vec![
            hash_input_padding_is_one,
            index_in_chunk_is_max_or_row_is_padding_row,
        ]
    }
}
