use std::cmp::Ordering;

use air::challenge_id::ChallengeId;
use air::cross_table_argument::CrossTableArg;
use air::cross_table_argument::EvalArg;
use air::cross_table_argument::LookupArg;
use air::table::TableId;
use air::table::program::ProgramTable;
use air::table_column::MasterAuxColumn;
use air::table_column::MasterMainColumn;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use ndarray::s;
use num_traits::ConstZero;
use num_traits::One;
use num_traits::Zero;
use strum::EnumCount;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::challenges::Challenges;
use crate::ndarray_helper::par_fill_column_with;
use crate::profiler::profiler;
use crate::table::TraceTable;
use crate::table::running_arguments::par_log_derivative_into;
use crate::table::running_arguments::par_running_affine_maps_into;
use crate::table::running_arguments::par_running_evaluation_into;
use crate::table::running_arguments::with_contiguous_column;

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
        let padding_rows = program_len..program_table.nrows();
        let address_column = MainColumn::Address.main_index();
        par_fill_column_with(
            &mut program_table,
            address_column,
            padding_rows.clone(),
            |a| bfe!(u64::try_from(a).unwrap()),
        );

        // The remaining columns only depend on the index in the chunk, which
        // takes few different values. In particular, few inverses suffice.
        let indices_in_chunk: [_; Tip5::RATE] =
            std::array::from_fn(|index| bfe!(u64::try_from(index).unwrap()));
        let index_in_chunk_column = MainColumn::IndexInChunk.main_index();
        par_fill_column_with(
            &mut program_table,
            index_in_chunk_column,
            padding_rows.clone(),
            |idx| indices_in_chunk[idx % Tip5::RATE],
        );

        let max_minus_indices_in_chunk_inverses: [_; Tip5::RATE] = std::array::from_fn(|index| {
            let max_minus_index = u64::try_from(Tip5::RATE - 1 - index).unwrap();
            BFieldElement::new(max_minus_index).inverse_or_zero()
        });
        let max_minus_index_in_chunk_inv_column = MainColumn::MaxMinusIndexInChunkInv.main_index();
        par_fill_column_with(
            &mut program_table,
            max_minus_index_in_chunk_inv_column,
            padding_rows,
            |idx| max_minus_indices_in_chunk_inverses[idx % Tip5::RATE],
        );

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
        //
        // In other words, every row records the contributions of all the
        // rows before it.
        let instruction_lookup_fraction = |row_idx: usize| {
            let previous_row_idx = row_idx.checked_sub(1)?;
            let previous_row = main_table.row(previous_row_idx);
            let row = main_table.row(row_idx);
            let is_hash_input_padding =
                previous_row[MainColumn::IsHashInputPadding.main_index()].is_one();
            (!is_hash_input_padding)
                .then(|| instruction_lookup_log_derivative_summand(previous_row, row, challenges))
        };
        let instruction_lookup_log_derivative =
            aux_table.column_mut(AuxColumn::InstructionLookupServerLogDerivative.aux_index());
        with_contiguous_column(instruction_lookup_log_derivative, |column| {
            par_log_derivative_into(column, LookupArg::default_initial(), |row_idx| {
                [instruction_lookup_fraction(row_idx)]
            });
        });

        // The running evaluation that prepares a chunk is reset at the
        // beginning of every chunk: such a row maps any running evaluation to
        // the same value.
        let prepare_chunk_indeterminate =
            challenges[ChallengeId::ProgramAttestationPrepareChunkIndeterminate];
        let prepare_chunk_map = |row_idx: usize| {
            let row = main_table.row(row_idx);
            let instruction = row[MainColumn::Instruction.main_index()];
            let running_evaluation_resets = row[MainColumn::IndexInChunk.main_index()].is_zero();
            if running_evaluation_resets {
                let reset_value = EvalArg::default_initial() * prepare_chunk_indeterminate;
                Some((XFieldElement::ZERO, reset_value + instruction))
            } else {
                Some((prepare_chunk_indeterminate, instruction.lift()))
            }
        };

        // A chunk is sent when its last element has been prepared.
        let max_index_in_chunk = Tip5::RATE as u64 - 1;
        let sends_chunk = |row_idx: usize| {
            let row = main_table.row(row_idx);
            let index_in_chunk = row[MainColumn::IndexInChunk.main_index()];
            let is_table_padding_row = row[MainColumn::IsTablePadding.main_index()].is_one();
            !is_table_padding_row && index_in_chunk.value() == max_index_in_chunk
        };

        let (prepare_chunk_running_evaluation, send_chunk_running_evaluation) = aux_table
            .multi_slice_mut((
                s![.., AuxColumn::PrepareChunkRunningEvaluation.aux_index()],
                s![.., AuxColumn::SendChunkRunningEvaluation.aux_index()],
            ));
        with_contiguous_column(prepare_chunk_running_evaluation, |prepared_chunks| {
            let initial = EvalArg::default_initial();
            par_running_affine_maps_into(prepared_chunks, initial, prepare_chunk_map);

            let prepared_chunks = &*prepared_chunks;
            with_contiguous_column(send_chunk_running_evaluation, |column| {
                par_running_evaluation_into(
                    column,
                    challenges[ChallengeId::ProgramAttestationSendChunkIndeterminate],
                    initial,
                    |row_idx| sends_chunk(row_idx).then(|| prepared_chunks[row_idx]),
                );
            });
        });

        profiler!(stop "program table");
    }
}

fn instruction_lookup_log_derivative_summand(
    row: ArrayView1<BFieldElement>,
    next_row: ArrayView1<BFieldElement>,
    challenges: &Challenges,
) -> (XFieldElement, XFieldElement) {
    let compressed_row = row[MainColumn::Address.main_index()]
        * challenges[ChallengeId::ProgramAddressWeight]
        + row[MainColumn::Instruction.main_index()]
            * challenges[ChallengeId::ProgramInstructionWeight]
        + next_row[MainColumn::Instruction.main_index()]
            * challenges[ChallengeId::ProgramNextInstructionWeight];
    let denominator = challenges[ChallengeId::InstructionLookupIndeterminate] - compressed_row;
    let numerator = row[MainColumn::LookupMultiplicity.main_index()].lift();

    (denominator, numerator)
}
