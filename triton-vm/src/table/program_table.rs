use std::cmp::Ordering;

use constraint_builder::DualRowIndicator::*;
use constraint_builder::SingleRowIndicator::*;
use constraint_builder::*;
use ndarray::s;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use num_traits::One;
use num_traits::Zero;
use strum::EnumCount;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::profiler::profiler;
use crate::table::challenges::ChallengeId::*;
use crate::table::challenges::Challenges;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::EvalArg;
use crate::table::cross_table_argument::LookupArg;
use crate::table::master_table::TableId;
use crate::table::table_column::ProgramBaseTableColumn::*;
use crate::table::table_column::ProgramExtTableColumn::*;
use crate::table::table_column::*;

pub const BASE_WIDTH: usize = ProgramBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = ProgramExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ProgramTable;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ExtProgramTable;

impl ExtProgramTable {
    pub fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let x_constant = |xfe| circuit_builder.x_constant(xfe);
        let base_row = |col: ProgramBaseTableColumn| {
            circuit_builder.input(BaseRow(col.master_base_table_index()))
        };
        let ext_row = |col: ProgramExtTableColumn| {
            circuit_builder.input(ExtRow(col.master_ext_table_index()))
        };

        let address = base_row(Address);
        let instruction = base_row(Instruction);
        let index_in_chunk = base_row(IndexInChunk);
        let is_hash_input_padding = base_row(IsHashInputPadding);
        let instruction_lookup_log_derivative = ext_row(InstructionLookupServerLogDerivative);
        let prepare_chunk_running_evaluation = ext_row(PrepareChunkRunningEvaluation);
        let send_chunk_running_evaluation = ext_row(SendChunkRunningEvaluation);

        let lookup_arg_initial = x_constant(LookupArg::default_initial());
        let eval_arg_initial = x_constant(EvalArg::default_initial());

        let program_attestation_prepare_chunk_indeterminate =
            challenge(ProgramAttestationPrepareChunkIndeterminate);

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

    pub fn consistency_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let base_row = |col: ProgramBaseTableColumn| {
            circuit_builder.input(BaseRow(col.master_base_table_index()))
        };

        let one = constant(1);
        let max_index_in_chunk = constant((Tip5::RATE - 1).try_into().unwrap());

        let index_in_chunk = base_row(IndexInChunk);
        let max_minus_index_in_chunk_inv = base_row(MaxMinusIndexInChunkInv);
        let is_hash_input_padding = base_row(IsHashInputPadding);
        let is_table_padding = base_row(IsTablePadding);

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

    pub fn transition_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let constant = |c: u64| circuit_builder.b_constant(c);

        let current_base_row = |col: ProgramBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProgramBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };
        let current_ext_row = |col: ProgramExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_ext_row = |col: ProgramExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };

        let one = constant(1);
        let rate_minus_one = constant(u64::try_from(Tip5::RATE).unwrap() - 1);

        let prepare_chunk_indeterminate = challenge(ProgramAttestationPrepareChunkIndeterminate);
        let send_chunk_indeterminate = challenge(ProgramAttestationSendChunkIndeterminate);

        let address = current_base_row(Address);
        let instruction = current_base_row(Instruction);
        let lookup_multiplicity = current_base_row(LookupMultiplicity);
        let index_in_chunk = current_base_row(IndexInChunk);
        let max_minus_index_in_chunk_inv = current_base_row(MaxMinusIndexInChunkInv);
        let is_hash_input_padding = current_base_row(IsHashInputPadding);
        let is_table_padding = current_base_row(IsTablePadding);
        let log_derivative = current_ext_row(InstructionLookupServerLogDerivative);
        let prepare_chunk_running_evaluation = current_ext_row(PrepareChunkRunningEvaluation);
        let send_chunk_running_evaluation = current_ext_row(SendChunkRunningEvaluation);

        let address_next = next_base_row(Address);
        let instruction_next = next_base_row(Instruction);
        let index_in_chunk_next = next_base_row(IndexInChunk);
        let max_minus_index_in_chunk_inv_next = next_base_row(MaxMinusIndexInChunkInv);
        let is_hash_input_padding_next = next_base_row(IsHashInputPadding);
        let is_table_padding_next = next_base_row(IsTablePadding);
        let log_derivative_next = next_ext_row(InstructionLookupServerLogDerivative);
        let prepare_chunk_running_evaluation_next = next_ext_row(PrepareChunkRunningEvaluation);
        let send_chunk_running_evaluation_next = next_ext_row(SendChunkRunningEvaluation);

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
        let compressed_row = challenge(ProgramAddressWeight) * address
            + challenge(ProgramInstructionWeight) * instruction
            + challenge(ProgramNextInstructionWeight) * instruction_next.clone();

        let indeterminate = challenge(InstructionLookupIndeterminate);
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

        let send_chunk_running_eval_absorbs_chunk_iff_index_in_chunk_next_is_max_and_not_padding_row =
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
            send_chunk_running_eval_absorbs_chunk_iff_index_in_chunk_next_is_max_and_not_padding_row,
        ]
    }

    pub fn terminal_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let constant = |c: u64| circuit_builder.b_constant(c);
        let base_row = |col: ProgramBaseTableColumn| {
            circuit_builder.input(BaseRow(col.master_base_table_index()))
        };

        let index_in_chunk = base_row(IndexInChunk);
        let is_hash_input_padding = base_row(IsHashInputPadding);
        let is_table_padding = base_row(IsTablePadding);

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

impl ProgramTable {
    pub fn fill_trace(
        program_table: &mut ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
    ) {
        let max_index_in_chunk = bfe!(Tip5::RATE as u64 - 1);

        let instructions = aet.program.to_bwords();
        let program_len = instructions.len();
        let padded_program_len = aet.height_of_table(TableId::Program);

        let one_iter = bfe_array![1].into_iter();
        let zero_iter = bfe_array![0].into_iter();
        let padding_iter = one_iter.chain(zero_iter.cycle());
        let padded_instructions = instructions.into_iter().chain(padding_iter);
        let padded_instructions = padded_instructions.take(padded_program_len);

        for (row_idx, instruction) in padded_instructions.enumerate() {
            let address = u64::try_from(row_idx).unwrap();
            let address = bfe!(address);

            let lookup_multiplicity = match row_idx.cmp(&program_len) {
                Ordering::Less => aet.instruction_multiplicities[row_idx],
                _ => 0,
            };
            let lookup_multiplicity = bfe!(lookup_multiplicity);
            let index_in_chunk = bfe!((row_idx % Tip5::RATE) as u64);

            let max_minus_index_in_chunk_inv =
                (max_index_in_chunk - index_in_chunk).inverse_or_zero();

            let is_hash_input_padding = match row_idx.cmp(&program_len) {
                Ordering::Less => bfe!(0),
                _ => bfe!(1),
            };

            let mut current_row = program_table.row_mut(row_idx);
            current_row[Address.base_table_index()] = address;
            current_row[Instruction.base_table_index()] = instruction;
            current_row[LookupMultiplicity.base_table_index()] = lookup_multiplicity;
            current_row[IndexInChunk.base_table_index()] = index_in_chunk;
            current_row[MaxMinusIndexInChunkInv.base_table_index()] = max_minus_index_in_chunk_inv;
            current_row[IsHashInputPadding.base_table_index()] = is_hash_input_padding;
        }
    }

    pub fn pad_trace(mut program_table: ArrayViewMut2<BFieldElement>, program_len: usize) {
        let addresses =
            (program_len..program_table.nrows()).map(|a| bfe!(u64::try_from(a).unwrap()));
        let addresses = Array1::from_iter(addresses);
        let address_column = program_table.slice_mut(s![program_len.., Address.base_table_index()]);
        addresses.move_into(address_column);

        let indices_in_chunks = (program_len..program_table.nrows())
            .map(|idx| idx % Tip5::RATE)
            .map(|ac| bfe!(u64::try_from(ac).unwrap()));
        let indices_in_chunks = Array1::from_iter(indices_in_chunks);
        let index_in_chunk_column =
            program_table.slice_mut(s![program_len.., IndexInChunk.base_table_index()]);
        indices_in_chunks.move_into(index_in_chunk_column);

        let max_minus_indices_in_chunks_inverses = (program_len..program_table.nrows())
            .map(|idx| Tip5::RATE - 1 - (idx % Tip5::RATE))
            .map(|ac| BFieldElement::new(ac.try_into().unwrap()))
            .map(|bfe| bfe.inverse_or_zero());
        let max_minus_indices_in_chunks_inverses =
            Array1::from_iter(max_minus_indices_in_chunks_inverses);
        let max_minus_index_in_chunk_inv_column = program_table.slice_mut(s![
            program_len..,
            MaxMinusIndexInChunkInv.base_table_index()
        ]);
        max_minus_indices_in_chunks_inverses.move_into(max_minus_index_in_chunk_inv_column);

        program_table
            .slice_mut(s![program_len.., IsHashInputPadding.base_table_index()])
            .fill(BFieldElement::one());
        program_table
            .slice_mut(s![program_len.., IsTablePadding.base_table_index()])
            .fill(BFieldElement::one());
    }

    pub fn extend(
        base_table: ArrayView2<BFieldElement>,
        mut ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        profiler!(start "program table");
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());

        let mut instruction_lookup_log_derivative = LookupArg::default_initial();
        let mut prepare_chunk_running_evaluation = EvalArg::default_initial();
        let mut send_chunk_running_evaluation = EvalArg::default_initial();

        for (idx, consecutive_rows) in base_table.windows([2, BASE_WIDTH]).into_iter().enumerate() {
            let row = consecutive_rows.row(0);
            let next_row = consecutive_rows.row(1);
            let mut extension_row = ext_table.row_mut(idx);

            // In the Program Table, the logarithmic derivative for the instruction lookup
            // argument does record the initial in the first row, as an exception to all other
            // table-linking arguments.
            // This is necessary because an instruction's potential argument, or else the next
            // instruction, is recorded in the next row. To be able to check correct initialization
            // of the logarithmic derivative, both the current and the next row must be accessible
            // to the constraint. Only transition constraints can access both rows. Hence, the
            // initial value of the logarithmic derivative must be independent of the second row.
            // The logarithmic derivative's final value, allowing for a meaningful cross-table
            // argument, is recorded in the first padding row. This row is guaranteed to exist
            // due to the hash-input padding mechanics.
            extension_row[InstructionLookupServerLogDerivative.ext_table_index()] =
                instruction_lookup_log_derivative;

            instruction_lookup_log_derivative = Self::update_instruction_lookup_log_derivative(
                challenges,
                row,
                next_row,
                instruction_lookup_log_derivative,
            );
            prepare_chunk_running_evaluation = Self::update_prepare_chunk_running_evaluation(
                row,
                challenges,
                prepare_chunk_running_evaluation,
            );
            send_chunk_running_evaluation = Self::update_send_chunk_running_evaluation(
                row,
                challenges,
                send_chunk_running_evaluation,
                prepare_chunk_running_evaluation,
            );

            extension_row[PrepareChunkRunningEvaluation.ext_table_index()] =
                prepare_chunk_running_evaluation;
            extension_row[SendChunkRunningEvaluation.ext_table_index()] =
                send_chunk_running_evaluation;
        }

        // special treatment for the last row
        let base_rows_iter = base_table.rows().into_iter();
        let ext_rows_iter = ext_table.rows_mut().into_iter();
        let last_base_row = base_rows_iter.last().unwrap();
        let mut last_ext_row = ext_rows_iter.last().unwrap();

        prepare_chunk_running_evaluation = Self::update_prepare_chunk_running_evaluation(
            last_base_row,
            challenges,
            prepare_chunk_running_evaluation,
        );
        send_chunk_running_evaluation = Self::update_send_chunk_running_evaluation(
            last_base_row,
            challenges,
            send_chunk_running_evaluation,
            prepare_chunk_running_evaluation,
        );

        last_ext_row[InstructionLookupServerLogDerivative.ext_table_index()] =
            instruction_lookup_log_derivative;
        last_ext_row[PrepareChunkRunningEvaluation.ext_table_index()] =
            prepare_chunk_running_evaluation;
        last_ext_row[SendChunkRunningEvaluation.ext_table_index()] = send_chunk_running_evaluation;

        profiler!(stop "program table");
    }

    fn update_instruction_lookup_log_derivative(
        challenges: &Challenges,
        row: ArrayView1<BFieldElement>,
        next_row: ArrayView1<BFieldElement>,
        instruction_lookup_log_derivative: XFieldElement,
    ) -> XFieldElement {
        if row[IsHashInputPadding.base_table_index()].is_one() {
            return instruction_lookup_log_derivative;
        }
        instruction_lookup_log_derivative
            + Self::instruction_lookup_log_derivative_summand(row, next_row, challenges)
    }

    fn instruction_lookup_log_derivative_summand(
        row: ArrayView1<BFieldElement>,
        next_row: ArrayView1<BFieldElement>,
        challenges: &Challenges,
    ) -> XFieldElement {
        let compressed_row = row[Address.base_table_index()] * challenges[ProgramAddressWeight]
            + row[Instruction.base_table_index()] * challenges[ProgramInstructionWeight]
            + next_row[Instruction.base_table_index()] * challenges[ProgramNextInstructionWeight];
        (challenges[InstructionLookupIndeterminate] - compressed_row).inverse()
            * row[LookupMultiplicity.base_table_index()]
    }

    fn update_prepare_chunk_running_evaluation(
        row: ArrayView1<BFieldElement>,
        challenges: &Challenges,
        prepare_chunk_running_evaluation: XFieldElement,
    ) -> XFieldElement {
        let running_evaluation_resets = row[IndexInChunk.base_table_index()].is_zero();
        let prepare_chunk_running_evaluation = match running_evaluation_resets {
            true => EvalArg::default_initial(),
            false => prepare_chunk_running_evaluation,
        };

        prepare_chunk_running_evaluation * challenges[ProgramAttestationPrepareChunkIndeterminate]
            + row[Instruction.base_table_index()]
    }

    fn update_send_chunk_running_evaluation(
        row: ArrayView1<BFieldElement>,
        challenges: &Challenges,
        send_chunk_running_evaluation: XFieldElement,
        prepare_chunk_running_evaluation: XFieldElement,
    ) -> XFieldElement {
        let index_in_chunk = row[IndexInChunk.base_table_index()];
        let is_table_padding_row = row[IsTablePadding.base_table_index()].is_one();
        let max_index_in_chunk = Tip5::RATE as u64 - 1;
        let running_evaluation_needs_update =
            !is_table_padding_row && index_in_chunk.value() == max_index_in_chunk;

        if !running_evaluation_needs_update {
            return send_chunk_running_evaluation;
        }

        send_chunk_running_evaluation * challenges[ProgramAttestationSendChunkIndeterminate]
            + prepare_chunk_running_evaluation
    }
}
