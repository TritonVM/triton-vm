use std::cmp::Ordering;

use air::challenge_id::ChallengeId;
use air::cross_table_argument::CrossTableArg;
use air::cross_table_argument::EvalArg;
use air::cross_table_argument::LookupArg;
use air::table::TableId;
use air::table::program::ProgramTable;
use air::table_column::MasterAuxColumn;
use air::table_column::MasterMainColumn;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use ndarray::s;
use num_traits::One;
use num_traits::Zero;
use strum::EnumCount;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::challenges::Challenges;
use crate::profiler::profiler;
use crate::table::TraceTable;

type MainColumn = <ProgramTable as air::AIR>::MainColumn;
type AuxColumn = <ProgramTable as air::AIR>::AuxColumn;

impl TraceTable for ProgramTable {
    type FillParam = ();
    type FillReturnInfo = ();

    fn fill(mut program_table: ArrayViewMut2<BFieldElement>, aet: &AlgebraicExecutionTrace, _: ()) {
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
            current_row[MainColumn::Address.main_index()] = address;
            current_row[MainColumn::Instruction.main_index()] = instruction;
            current_row[MainColumn::LookupMultiplicity.main_index()] = lookup_multiplicity;
            current_row[MainColumn::IndexInChunk.main_index()] = index_in_chunk;
            current_row[MainColumn::MaxMinusIndexInChunkInv.main_index()] =
                max_minus_index_in_chunk_inv;
            current_row[MainColumn::IsHashInputPadding.main_index()] = is_hash_input_padding;
        }
    }

    fn pad(mut program_table: ArrayViewMut2<BFieldElement>, program_len: usize) {
        let addresses =
            (program_len..program_table.nrows()).map(|a| bfe!(u64::try_from(a).unwrap()));
        let addresses = Array1::from_iter(addresses);
        let address_column =
            program_table.slice_mut(s![program_len.., MainColumn::Address.main_index()]);
        addresses.move_into(address_column);

        let indices_in_chunks = (program_len..program_table.nrows())
            .map(|idx| idx % Tip5::RATE)
            .map(|ac| bfe!(u64::try_from(ac).unwrap()));
        let indices_in_chunks = Array1::from_iter(indices_in_chunks);
        let index_in_chunk_column =
            program_table.slice_mut(s![program_len.., MainColumn::IndexInChunk.main_index()]);
        indices_in_chunks.move_into(index_in_chunk_column);

        let max_minus_indices_in_chunks_inverses = (program_len..program_table.nrows())
            .map(|idx| Tip5::RATE - 1 - (idx % Tip5::RATE))
            .map(|ac| BFieldElement::new(ac.try_into().unwrap()))
            .map(|bfe| bfe.inverse_or_zero());
        let max_minus_indices_in_chunks_inverses =
            Array1::from_iter(max_minus_indices_in_chunks_inverses);
        let max_minus_index_in_chunk_inv_column = program_table.slice_mut(s![
            program_len..,
            MainColumn::MaxMinusIndexInChunkInv.main_index()
        ]);
        max_minus_indices_in_chunks_inverses.move_into(max_minus_index_in_chunk_inv_column);

        program_table
            .slice_mut(s![
                program_len..,
                MainColumn::IsHashInputPadding.main_index()
            ])
            .fill(BFieldElement::one());
        program_table
            .slice_mut(s![program_len.., MainColumn::IsTablePadding.main_index()])
            .fill(BFieldElement::one());
    }

    fn extend(
        main_table: ArrayView2<BFieldElement>,
        mut aux_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        profiler!(start "program table");
        assert_eq!(MainColumn::COUNT, main_table.ncols());
        assert_eq!(AuxColumn::COUNT, aux_table.ncols());
        assert_eq!(main_table.nrows(), aux_table.nrows());

        let mut instruction_lookup_log_derivative = LookupArg::default_initial();
        let mut prepare_chunk_running_evaluation = EvalArg::default_initial();
        let mut send_chunk_running_evaluation = EvalArg::default_initial();

        for (idx, consecutive_rows) in main_table
            .windows([2, MainColumn::COUNT])
            .into_iter()
            .enumerate()
        {
            let row = consecutive_rows.row(0);
            let next_row = consecutive_rows.row(1);
            let mut auxiliary_row = aux_table.row_mut(idx);

            // In the Program Table, the logarithmic derivative for the
            // instruction lookup argument does record the initial in the first
            // row, as an exception to all other table-linking arguments.
            // This is necessary because an instruction's potential argument, or
            // else the next instruction, is recorded in the next row. To be
            // able to check correct initialization of the logarithmic
            // derivative, both the current and the next row must be accessible
            // to the constraint. Only transition constraints can access both
            // rows. Hence, the initial value of the logarithmic derivative must
            // be independent of the second row. The logarithmic derivative's
            // final value, allowing for a meaningful cross-table argument, is
            // recorded in the first padding row. This row is guaranteed to
            // exist due to the hash-input padding mechanics.
            auxiliary_row[AuxColumn::InstructionLookupServerLogDerivative.aux_index()] =
                instruction_lookup_log_derivative;

            instruction_lookup_log_derivative = update_instruction_lookup_log_derivative(
                challenges,
                row,
                next_row,
                instruction_lookup_log_derivative,
            );
            prepare_chunk_running_evaluation = update_prepare_chunk_running_evaluation(
                row,
                challenges,
                prepare_chunk_running_evaluation,
            );
            send_chunk_running_evaluation = update_send_chunk_running_evaluation(
                row,
                challenges,
                send_chunk_running_evaluation,
                prepare_chunk_running_evaluation,
            );

            auxiliary_row[AuxColumn::PrepareChunkRunningEvaluation.aux_index()] =
                prepare_chunk_running_evaluation;
            auxiliary_row[AuxColumn::SendChunkRunningEvaluation.aux_index()] =
                send_chunk_running_evaluation;
        }

        // special treatment for the last row
        let last_main_row = main_table.rows().into_iter().next_back().unwrap();
        let mut last_aux_row = aux_table.rows_mut().into_iter().next_back().unwrap();

        prepare_chunk_running_evaluation = update_prepare_chunk_running_evaluation(
            last_main_row,
            challenges,
            prepare_chunk_running_evaluation,
        );
        send_chunk_running_evaluation = update_send_chunk_running_evaluation(
            last_main_row,
            challenges,
            send_chunk_running_evaluation,
            prepare_chunk_running_evaluation,
        );

        last_aux_row[AuxColumn::InstructionLookupServerLogDerivative.aux_index()] =
            instruction_lookup_log_derivative;
        last_aux_row[AuxColumn::PrepareChunkRunningEvaluation.aux_index()] =
            prepare_chunk_running_evaluation;
        last_aux_row[AuxColumn::SendChunkRunningEvaluation.aux_index()] =
            send_chunk_running_evaluation;

        profiler!(stop "program table");
    }
}

fn update_instruction_lookup_log_derivative(
    challenges: &Challenges,
    row: ArrayView1<BFieldElement>,
    next_row: ArrayView1<BFieldElement>,
    instruction_lookup_log_derivative: XFieldElement,
) -> XFieldElement {
    if row[MainColumn::IsHashInputPadding.main_index()].is_one() {
        return instruction_lookup_log_derivative;
    }
    instruction_lookup_log_derivative
        + instruction_lookup_log_derivative_summand(row, next_row, challenges)
}

fn instruction_lookup_log_derivative_summand(
    row: ArrayView1<BFieldElement>,
    next_row: ArrayView1<BFieldElement>,
    challenges: &Challenges,
) -> XFieldElement {
    let compressed_row = row[MainColumn::Address.main_index()]
        * challenges[ChallengeId::ProgramAddressWeight]
        + row[MainColumn::Instruction.main_index()]
            * challenges[ChallengeId::ProgramInstructionWeight]
        + next_row[MainColumn::Instruction.main_index()]
            * challenges[ChallengeId::ProgramNextInstructionWeight];
    (challenges[ChallengeId::InstructionLookupIndeterminate] - compressed_row).inverse()
        * row[MainColumn::LookupMultiplicity.main_index()]
}

fn update_prepare_chunk_running_evaluation(
    row: ArrayView1<BFieldElement>,
    challenges: &Challenges,
    prepare_chunk_running_evaluation: XFieldElement,
) -> XFieldElement {
    let running_evaluation_resets = row[MainColumn::IndexInChunk.main_index()].is_zero();
    let prepare_chunk_running_evaluation = if running_evaluation_resets {
        EvalArg::default_initial()
    } else {
        prepare_chunk_running_evaluation
    };

    prepare_chunk_running_evaluation
        * challenges[ChallengeId::ProgramAttestationPrepareChunkIndeterminate]
        + row[MainColumn::Instruction.main_index()]
}

fn update_send_chunk_running_evaluation(
    row: ArrayView1<BFieldElement>,
    challenges: &Challenges,
    send_chunk_running_evaluation: XFieldElement,
    prepare_chunk_running_evaluation: XFieldElement,
) -> XFieldElement {
    let index_in_chunk = row[MainColumn::IndexInChunk.main_index()];
    let is_table_padding_row = row[MainColumn::IsTablePadding.main_index()].is_one();
    let max_index_in_chunk = Tip5::RATE as u64 - 1;
    let running_evaluation_needs_update =
        !is_table_padding_row && index_in_chunk.value() == max_index_in_chunk;

    if !running_evaluation_needs_update {
        return send_chunk_running_evaluation;
    }

    send_chunk_running_evaluation
        * challenges[ChallengeId::ProgramAttestationSendChunkIndeterminate]
        + prepare_chunk_running_evaluation
}
