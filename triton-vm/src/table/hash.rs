use air::challenge_id::ChallengeId::*;
use air::cross_table_argument::CrossTableArg;
use air::cross_table_argument::EvalArg;
use air::cross_table_argument::LookupArg;
use air::table::hash::HashTable;
use air::table::hash::HashTableMode;
use air::table::hash::PermutationTrace;
use air::table::hash::MONTGOMERY_MODULUS;
use air::table::hash::NUM_ROUND_CONSTANTS;
use air::table_column::HashBaseTableColumn::*;
use air::table_column::HashExtTableColumn::*;
use air::table_column::MasterBaseTableColumn;
use air::table_column::MasterExtTableColumn;
use air::AIR;
use constraint_circuit::ConstraintCircuitBuilder;
use constraint_circuit::ConstraintCircuitMonad;
use constraint_circuit::DualRowIndicator;
use constraint_circuit::DualRowIndicator::*;
use constraint_circuit::InputIndicator;
use constraint_circuit::SingleRowIndicator;
use constraint_circuit::SingleRowIndicator::*;
use isa::instruction::AnInstruction::Hash;
use isa::instruction::AnInstruction::SpongeAbsorb;
use isa::instruction::AnInstruction::SpongeInit;
use isa::instruction::AnInstruction::SpongeSqueeze;
use isa::instruction::Instruction;
use itertools::Itertools;
use ndarray::*;
use num_traits::Zero;
use strum::Display;
use strum::EnumCount;
use strum::EnumIter;
use strum::IntoEnumIterator;
use twenty_first::prelude::tip5::NUM_ROUNDS;
use twenty_first::prelude::tip5::NUM_SPLIT_AND_LOOKUP;
use twenty_first::prelude::tip5::RATE;
use twenty_first::prelude::tip5::ROUND_CONSTANTS;
use twenty_first::prelude::tip5::STATE_SIZE;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::challenges::Challenges;
use crate::profiler::profiler;
use crate::table::TraceTable;

/// Return the 16-bit chunks of the “un-Montgomery'd” representation, in little-endian chunk
/// order. This (basically) translates to the application of `σ(R·x)` for input `x`, which
/// are the first two steps in Tip5's split-and-lookup S-Box.
/// `R` is the Montgomery modulus, _i.e._, `R = 2^64 mod p`.
/// `σ` as described in the paper decomposes the 64-bit input into 8-bit limbs, whereas
/// this method decomposes into 16-bit limbs for arithmetization reasons; the 16-bit limbs
/// are split into 8-bit limbs in the Cascade Table.
/// For a more in-depth explanation of all the necessary steps in the split-and-lookup S-Box,
/// see the [Tip5 paper](https://eprint.iacr.org/2023/107.pdf).
///
/// Note: this is distinct from the seemingly similar [`raw_u16s`](BFieldElement::raw_u16s).
pub(crate) fn base_field_element_into_16_bit_limbs(x: BFieldElement) -> [u16; 4] {
    let r_times_x = (MONTGOMERY_MODULUS * x).value();
    [0, 16, 32, 48].map(|shift| ((r_times_x >> shift) & 0xffff) as u16)
}

/// Convert a permutation trace to a segment in the Hash Table.
///
/// **Note**: The current instruction [`CI`] is _not_ set.
pub(crate) fn trace_to_table_rows(trace: PermutationTrace) -> Array2<BFieldElement> {
    let mut table_rows = Array2::default([0, <HashTable as AIR>::MainColumn::COUNT]);
    for (round_number, &trace_row) in trace.iter().enumerate() {
        let table_row = trace_row_to_table_row(trace_row, round_number);
        table_rows.push_row(table_row.view()).unwrap();
    }
    table_rows
}

pub(crate) fn trace_row_to_table_row(
    trace_row: [BFieldElement; STATE_SIZE],
    round_number: usize,
) -> Array1<BFieldElement> {
    let row = Array1::zeros([<HashTable as AIR>::MainColumn::COUNT]);
    let row = fill_row_with_round_number(row, round_number);
    let row = fill_row_with_split_state_elements_using_trace_row(row, trace_row);
    let row = fill_row_with_unsplit_state_elements_using_trace_row(row, trace_row);
    let row = fill_row_with_state_inverses_using_trace_row(row, trace_row);
    fill_row_with_round_constants_for_round(row, round_number)
}

fn fill_row_with_round_number(
    mut row: Array1<BFieldElement>,
    round_number: usize,
) -> Array1<BFieldElement> {
    row[RoundNumber.base_table_index()] = bfe!(round_number as u64);
    row
}

fn fill_row_with_split_state_elements_using_trace_row(
    row: Array1<BFieldElement>,
    trace_row: [BFieldElement; STATE_SIZE],
) -> Array1<BFieldElement> {
    let row = fill_split_state_element_0_of_row_using_trace_row(row, trace_row);
    let row = fill_split_state_element_1_of_row_using_trace_row(row, trace_row);
    let row = fill_split_state_element_2_of_row_using_trace_row(row, trace_row);
    fill_split_state_element_3_of_row_using_trace_row(row, trace_row)
}

fn fill_split_state_element_0_of_row_using_trace_row(
    mut row: Array1<BFieldElement>,
    trace_row: [BFieldElement; STATE_SIZE],
) -> Array1<BFieldElement> {
    let limbs = base_field_element_into_16_bit_limbs(trace_row[0]);
    let look_in_split = limbs.map(|limb| bfe!(limb));
    row[State0LowestLkIn.base_table_index()] = look_in_split[0];
    row[State0MidLowLkIn.base_table_index()] = look_in_split[1];
    row[State0MidHighLkIn.base_table_index()] = look_in_split[2];
    row[State0HighestLkIn.base_table_index()] = look_in_split[3];

    let look_out_split = limbs.map(crate::table::cascade::lookup_16_bit_limb);
    row[State0LowestLkOut.base_table_index()] = look_out_split[0];
    row[State0MidLowLkOut.base_table_index()] = look_out_split[1];
    row[State0MidHighLkOut.base_table_index()] = look_out_split[2];
    row[State0HighestLkOut.base_table_index()] = look_out_split[3];

    row
}

fn fill_split_state_element_1_of_row_using_trace_row(
    mut row: Array1<BFieldElement>,
    trace_row: [BFieldElement; STATE_SIZE],
) -> Array1<BFieldElement> {
    let limbs = base_field_element_into_16_bit_limbs(trace_row[1]);
    let look_in_split = limbs.map(|limb| bfe!(limb));
    row[State1LowestLkIn.base_table_index()] = look_in_split[0];
    row[State1MidLowLkIn.base_table_index()] = look_in_split[1];
    row[State1MidHighLkIn.base_table_index()] = look_in_split[2];
    row[State1HighestLkIn.base_table_index()] = look_in_split[3];

    let look_out_split = limbs.map(crate::table::cascade::lookup_16_bit_limb);
    row[State1LowestLkOut.base_table_index()] = look_out_split[0];
    row[State1MidLowLkOut.base_table_index()] = look_out_split[1];
    row[State1MidHighLkOut.base_table_index()] = look_out_split[2];
    row[State1HighestLkOut.base_table_index()] = look_out_split[3];

    row
}

fn fill_split_state_element_2_of_row_using_trace_row(
    mut row: Array1<BFieldElement>,
    trace_row: [BFieldElement; STATE_SIZE],
) -> Array1<BFieldElement> {
    let limbs = base_field_element_into_16_bit_limbs(trace_row[2]);
    let look_in_split = limbs.map(|limb| bfe!(limb));
    row[State2LowestLkIn.base_table_index()] = look_in_split[0];
    row[State2MidLowLkIn.base_table_index()] = look_in_split[1];
    row[State2MidHighLkIn.base_table_index()] = look_in_split[2];
    row[State2HighestLkIn.base_table_index()] = look_in_split[3];

    let look_out_split = limbs.map(crate::table::cascade::lookup_16_bit_limb);
    row[State2LowestLkOut.base_table_index()] = look_out_split[0];
    row[State2MidLowLkOut.base_table_index()] = look_out_split[1];
    row[State2MidHighLkOut.base_table_index()] = look_out_split[2];
    row[State2HighestLkOut.base_table_index()] = look_out_split[3];

    row
}

fn fill_split_state_element_3_of_row_using_trace_row(
    mut row: Array1<BFieldElement>,
    trace_row: [BFieldElement; STATE_SIZE],
) -> Array1<BFieldElement> {
    let limbs = base_field_element_into_16_bit_limbs(trace_row[3]);
    let look_in_split = limbs.map(|limb| bfe!(limb));
    row[State3LowestLkIn.base_table_index()] = look_in_split[0];
    row[State3MidLowLkIn.base_table_index()] = look_in_split[1];
    row[State3MidHighLkIn.base_table_index()] = look_in_split[2];
    row[State3HighestLkIn.base_table_index()] = look_in_split[3];

    let look_out_split = limbs.map(crate::table::cascade::lookup_16_bit_limb);
    row[State3LowestLkOut.base_table_index()] = look_out_split[0];
    row[State3MidLowLkOut.base_table_index()] = look_out_split[1];
    row[State3MidHighLkOut.base_table_index()] = look_out_split[2];
    row[State3HighestLkOut.base_table_index()] = look_out_split[3];

    row
}

fn fill_row_with_unsplit_state_elements_using_trace_row(
    mut row: Array1<BFieldElement>,
    trace_row: [BFieldElement; STATE_SIZE],
) -> Array1<BFieldElement> {
    row[State4.base_table_index()] = trace_row[4];
    row[State5.base_table_index()] = trace_row[5];
    row[State6.base_table_index()] = trace_row[6];
    row[State7.base_table_index()] = trace_row[7];
    row[State8.base_table_index()] = trace_row[8];
    row[State9.base_table_index()] = trace_row[9];
    row[State10.base_table_index()] = trace_row[10];
    row[State11.base_table_index()] = trace_row[11];
    row[State12.base_table_index()] = trace_row[12];
    row[State13.base_table_index()] = trace_row[13];
    row[State14.base_table_index()] = trace_row[14];
    row[State15.base_table_index()] = trace_row[15];
    row
}

fn fill_row_with_state_inverses_using_trace_row(
    mut row: Array1<BFieldElement>,
    trace_row: [BFieldElement; STATE_SIZE],
) -> Array1<BFieldElement> {
    row[State0Inv.base_table_index()] = inverse_or_zero_of_highest_2_limbs(trace_row[0]);
    row[State1Inv.base_table_index()] = inverse_or_zero_of_highest_2_limbs(trace_row[1]);
    row[State2Inv.base_table_index()] = inverse_or_zero_of_highest_2_limbs(trace_row[2]);
    row[State3Inv.base_table_index()] = inverse_or_zero_of_highest_2_limbs(trace_row[3]);
    row
}

/// The inverse-or-zero of (2^32 - 1 - 2^16·`highest` - `mid_high`) where `highest`
/// is the most significant limb of the given `state_element`, and `mid_high` the second-most
/// significant limb.
fn inverse_or_zero_of_highest_2_limbs(state_element: BFieldElement) -> BFieldElement {
    let limbs = base_field_element_into_16_bit_limbs(state_element);
    let highest: u64 = limbs[3].into();
    let mid_high: u64 = limbs[2].into();
    let high_limbs = bfe!((highest << 16) + mid_high);
    let two_pow_32_minus_1 = bfe!((1_u64 << 32) - 1);
    let to_invert = two_pow_32_minus_1 - high_limbs;
    to_invert.inverse_or_zero()
}

fn fill_row_with_round_constants_for_round(
    mut row: Array1<BFieldElement>,
    round_number: usize,
) -> Array1<BFieldElement> {
    let round_constants = HashTable::tip5_round_constants_by_round_number(round_number);
    let [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15] = round_constants;
    row[Constant0.base_table_index()] = r0;
    row[Constant1.base_table_index()] = r1;
    row[Constant2.base_table_index()] = r2;
    row[Constant3.base_table_index()] = r3;
    row[Constant4.base_table_index()] = r4;
    row[Constant5.base_table_index()] = r5;
    row[Constant6.base_table_index()] = r6;
    row[Constant7.base_table_index()] = r7;
    row[Constant8.base_table_index()] = r8;
    row[Constant9.base_table_index()] = r9;
    row[Constant10.base_table_index()] = r10;
    row[Constant11.base_table_index()] = r11;
    row[Constant12.base_table_index()] = r12;
    row[Constant13.base_table_index()] = r13;
    row[Constant14.base_table_index()] = r14;
    row[Constant15.base_table_index()] = r15;
    row
}

impl TraceTable for HashTable {
    type FillParam = ();
    type FillReturnInfo = ();

    fn fill(mut main_table: ArrayViewMut2<BFieldElement>, aet: &AlgebraicExecutionTrace, _: ()) {
        let program_hash_part_start = 0;
        let program_hash_part_end = program_hash_part_start + aet.program_hash_trace.nrows();
        let sponge_part_start = program_hash_part_end;
        let sponge_part_end = sponge_part_start + aet.sponge_trace.nrows();
        let hash_part_start = sponge_part_end;
        let hash_part_end = hash_part_start + aet.hash_trace.nrows();

        let (mut program_hash_part, mut sponge_part, mut hash_part) = main_table.multi_slice_mut((
            s![program_hash_part_start..program_hash_part_end, ..],
            s![sponge_part_start..sponge_part_end, ..],
            s![hash_part_start..hash_part_end, ..],
        ));

        program_hash_part.assign(&aet.program_hash_trace);
        sponge_part.assign(&aet.sponge_trace);
        hash_part.assign(&aet.hash_trace);

        let mode_column_idx = Mode.base_table_index();
        let mut program_hash_mode_column = program_hash_part.column_mut(mode_column_idx);
        let mut sponge_mode_column = sponge_part.column_mut(mode_column_idx);
        let mut hash_mode_column = hash_part.column_mut(mode_column_idx);

        program_hash_mode_column.fill(HashTableMode::ProgramHashing.into());
        sponge_mode_column.fill(HashTableMode::Sponge.into());
        hash_mode_column.fill(HashTableMode::Hash.into());
    }

    fn pad(mut main_table: ArrayViewMut2<BFieldElement>, table_length: usize) {
        let inverse_of_high_limbs = inverse_or_zero_of_highest_2_limbs(bfe!(0));
        for column_id in [State0Inv, State1Inv, State2Inv, State3Inv] {
            let column_index = column_id.base_table_index();
            let slice_info = s![table_length.., column_index];
            let mut column = main_table.slice_mut(slice_info);
            column.fill(inverse_of_high_limbs);
        }

        let round_constants = Self::tip5_round_constants_by_round_number(0);
        for (round_constant_idx, &round_constant) in round_constants.iter().enumerate() {
            let round_constant_column =
                HashTable::round_constant_column_by_index(round_constant_idx);
            let round_constant_column_idx = round_constant_column.base_table_index();
            let slice_info = s![table_length.., round_constant_column_idx];
            let mut column = main_table.slice_mut(slice_info);
            column.fill(round_constant);
        }

        let mode_column_index = Mode.base_table_index();
        let mode_column_slice_info = s![table_length.., mode_column_index];
        let mut mode_column = main_table.slice_mut(mode_column_slice_info);
        mode_column.fill(HashTableMode::Pad.into());

        let instruction_column_index = CI.base_table_index();
        let instruction_column_slice_info = s![table_length.., instruction_column_index];
        let mut instruction_column = main_table.slice_mut(instruction_column_slice_info);
        instruction_column.fill(Instruction::Hash.opcode_b());
    }

    fn extend(
        main_table: ArrayView2<BFieldElement>,
        mut aux_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        profiler!(start "hash table");
        assert_eq!(Self::MainColumn::COUNT, main_table.ncols());
        assert_eq!(Self::AuxColumn::COUNT, aux_table.ncols());
        assert_eq!(main_table.nrows(), aux_table.nrows());

        let ci_weight = challenges[HashCIWeight];
        let hash_digest_eval_indeterminate = challenges[HashDigestIndeterminate];
        let hash_input_eval_indeterminate = challenges[HashInputIndeterminate];
        let sponge_eval_indeterminate = challenges[SpongeIndeterminate];
        let cascade_indeterminate = challenges[HashCascadeLookupIndeterminate];
        let send_chunk_indeterminate = challenges[ProgramAttestationSendChunkIndeterminate];

        let mut hash_input_running_evaluation = EvalArg::default_initial();
        let mut hash_digest_running_evaluation = EvalArg::default_initial();
        let mut sponge_running_evaluation = EvalArg::default_initial();
        let mut cascade_state_0_highest_log_derivative = LookupArg::default_initial();
        let mut cascade_state_0_mid_high_log_derivative = LookupArg::default_initial();
        let mut cascade_state_0_mid_low_log_derivative = LookupArg::default_initial();
        let mut cascade_state_0_lowest_log_derivative = LookupArg::default_initial();
        let mut cascade_state_1_highest_log_derivative = LookupArg::default_initial();
        let mut cascade_state_1_mid_high_log_derivative = LookupArg::default_initial();
        let mut cascade_state_1_mid_low_log_derivative = LookupArg::default_initial();
        let mut cascade_state_1_lowest_log_derivative = LookupArg::default_initial();
        let mut cascade_state_2_highest_log_derivative = LookupArg::default_initial();
        let mut cascade_state_2_mid_high_log_derivative = LookupArg::default_initial();
        let mut cascade_state_2_mid_low_log_derivative = LookupArg::default_initial();
        let mut cascade_state_2_lowest_log_derivative = LookupArg::default_initial();
        let mut cascade_state_3_highest_log_derivative = LookupArg::default_initial();
        let mut cascade_state_3_mid_high_log_derivative = LookupArg::default_initial();
        let mut cascade_state_3_mid_low_log_derivative = LookupArg::default_initial();
        let mut cascade_state_3_lowest_log_derivative = LookupArg::default_initial();
        let mut receive_chunk_running_evaluation = EvalArg::default_initial();

        let two_pow_16 = bfe!(1_u64 << 16);
        let two_pow_32 = bfe!(1_u64 << 32);
        let two_pow_48 = bfe!(1_u64 << 48);

        let montgomery_modulus_inverse = MONTGOMERY_MODULUS.inverse();
        let re_compose_state_element =
            |row: ArrayView1<BFieldElement>,
             highest: Self::MainColumn,
             mid_high: Self::MainColumn,
             mid_low: Self::MainColumn,
             lowest: Self::MainColumn| {
                (row[highest.base_table_index()] * two_pow_48
                    + row[mid_high.base_table_index()] * two_pow_32
                    + row[mid_low.base_table_index()] * two_pow_16
                    + row[lowest.base_table_index()])
                    * montgomery_modulus_inverse
            };

        let rate_registers = |row: ArrayView1<BFieldElement>| {
            let state_0 = re_compose_state_element(
                row,
                State0HighestLkIn,
                State0MidHighLkIn,
                State0MidLowLkIn,
                State0LowestLkIn,
            );
            let state_1 = re_compose_state_element(
                row,
                State1HighestLkIn,
                State1MidHighLkIn,
                State1MidLowLkIn,
                State1LowestLkIn,
            );
            let state_2 = re_compose_state_element(
                row,
                State2HighestLkIn,
                State2MidHighLkIn,
                State2MidLowLkIn,
                State2LowestLkIn,
            );
            let state_3 = re_compose_state_element(
                row,
                State3HighestLkIn,
                State3MidHighLkIn,
                State3MidLowLkIn,
                State3LowestLkIn,
            );
            [
                state_0,
                state_1,
                state_2,
                state_3,
                row[State4.base_table_index()],
                row[State5.base_table_index()],
                row[State6.base_table_index()],
                row[State7.base_table_index()],
                row[State8.base_table_index()],
                row[State9.base_table_index()],
            ]
        };

        let state_weights = &challenges[StackWeight0..StackWeight10];
        let compressed_row = |row: ArrayView1<BFieldElement>| -> XFieldElement {
            rate_registers(row)
                .iter()
                .zip_eq(state_weights.iter())
                .map(|(&state, &weight)| weight * state)
                .sum()
        };

        let cascade_look_in_weight = challenges[HashCascadeLookInWeight];
        let cascade_look_out_weight = challenges[HashCascadeLookOutWeight];

        let log_derivative_summand =
            |row: ArrayView1<BFieldElement>,
             lk_in_col: Self::MainColumn,
             lk_out_col: Self::MainColumn| {
                let compressed_elements = cascade_indeterminate
                    - cascade_look_in_weight * row[lk_in_col.base_table_index()]
                    - cascade_look_out_weight * row[lk_out_col.base_table_index()];
                compressed_elements.inverse()
            };

        for row_idx in 0..main_table.nrows() {
            let row = main_table.row(row_idx);

            let mode = row[Mode.base_table_index()];
            let in_program_hashing_mode = mode == HashTableMode::ProgramHashing.into();
            let in_sponge_mode = mode == HashTableMode::Sponge.into();
            let in_hash_mode = mode == HashTableMode::Hash.into();
            let in_pad_mode = mode == HashTableMode::Pad.into();

            let round_number = row[RoundNumber.base_table_index()];
            let in_round_0 = round_number.is_zero();
            let in_last_round = round_number == (NUM_ROUNDS as u64).into();

            let current_instruction = row[CI.base_table_index()];
            let current_instruction_is_sponge_init =
                current_instruction == Instruction::SpongeInit.opcode_b();

            if in_program_hashing_mode && in_round_0 {
                let compressed_chunk_of_instructions = EvalArg::compute_terminal(
                    &rate_registers(row),
                    EvalArg::default_initial(),
                    challenges[ProgramAttestationPrepareChunkIndeterminate],
                );
                receive_chunk_running_evaluation = receive_chunk_running_evaluation
                    * send_chunk_indeterminate
                    + compressed_chunk_of_instructions
            }

            if in_sponge_mode && in_round_0 && current_instruction_is_sponge_init {
                sponge_running_evaluation = sponge_running_evaluation * sponge_eval_indeterminate
                    + ci_weight * current_instruction
            }

            if in_sponge_mode && in_round_0 && !current_instruction_is_sponge_init {
                sponge_running_evaluation = sponge_running_evaluation * sponge_eval_indeterminate
                    + ci_weight * current_instruction
                    + compressed_row(row)
            }

            if in_hash_mode && in_round_0 {
                hash_input_running_evaluation = hash_input_running_evaluation
                    * hash_input_eval_indeterminate
                    + compressed_row(row)
            }

            if in_hash_mode && in_last_round {
                let compressed_digest: XFieldElement = rate_registers(row)[..Digest::LEN]
                    .iter()
                    .zip_eq(state_weights[..Digest::LEN].iter())
                    .map(|(&state, &weight)| weight * state)
                    .sum();
                hash_digest_running_evaluation = hash_digest_running_evaluation
                    * hash_digest_eval_indeterminate
                    + compressed_digest
            }

            if !in_pad_mode && !in_last_round && !current_instruction_is_sponge_init {
                cascade_state_0_highest_log_derivative +=
                    log_derivative_summand(row, State0HighestLkIn, State0HighestLkOut);
                cascade_state_0_mid_high_log_derivative +=
                    log_derivative_summand(row, State0MidHighLkIn, State0MidHighLkOut);
                cascade_state_0_mid_low_log_derivative +=
                    log_derivative_summand(row, State0MidLowLkIn, State0MidLowLkOut);
                cascade_state_0_lowest_log_derivative +=
                    log_derivative_summand(row, State0LowestLkIn, State0LowestLkOut);
                cascade_state_1_highest_log_derivative +=
                    log_derivative_summand(row, State1HighestLkIn, State1HighestLkOut);
                cascade_state_1_mid_high_log_derivative +=
                    log_derivative_summand(row, State1MidHighLkIn, State1MidHighLkOut);
                cascade_state_1_mid_low_log_derivative +=
                    log_derivative_summand(row, State1MidLowLkIn, State1MidLowLkOut);
                cascade_state_1_lowest_log_derivative +=
                    log_derivative_summand(row, State1LowestLkIn, State1LowestLkOut);
                cascade_state_2_highest_log_derivative +=
                    log_derivative_summand(row, State2HighestLkIn, State2HighestLkOut);
                cascade_state_2_mid_high_log_derivative +=
                    log_derivative_summand(row, State2MidHighLkIn, State2MidHighLkOut);
                cascade_state_2_mid_low_log_derivative +=
                    log_derivative_summand(row, State2MidLowLkIn, State2MidLowLkOut);
                cascade_state_2_lowest_log_derivative +=
                    log_derivative_summand(row, State2LowestLkIn, State2LowestLkOut);
                cascade_state_3_highest_log_derivative +=
                    log_derivative_summand(row, State3HighestLkIn, State3HighestLkOut);
                cascade_state_3_mid_high_log_derivative +=
                    log_derivative_summand(row, State3MidHighLkIn, State3MidHighLkOut);
                cascade_state_3_mid_low_log_derivative +=
                    log_derivative_summand(row, State3MidLowLkIn, State3MidLowLkOut);
                cascade_state_3_lowest_log_derivative +=
                    log_derivative_summand(row, State3LowestLkIn, State3LowestLkOut);
            }

            let mut extension_row = aux_table.row_mut(row_idx);
            extension_row[ReceiveChunkRunningEvaluation.ext_table_index()] =
                receive_chunk_running_evaluation;
            extension_row[HashInputRunningEvaluation.ext_table_index()] =
                hash_input_running_evaluation;
            extension_row[HashDigestRunningEvaluation.ext_table_index()] =
                hash_digest_running_evaluation;
            extension_row[SpongeRunningEvaluation.ext_table_index()] = sponge_running_evaluation;
            extension_row[CascadeState0HighestClientLogDerivative.ext_table_index()] =
                cascade_state_0_highest_log_derivative;
            extension_row[CascadeState0MidHighClientLogDerivative.ext_table_index()] =
                cascade_state_0_mid_high_log_derivative;
            extension_row[CascadeState0MidLowClientLogDerivative.ext_table_index()] =
                cascade_state_0_mid_low_log_derivative;
            extension_row[CascadeState0LowestClientLogDerivative.ext_table_index()] =
                cascade_state_0_lowest_log_derivative;
            extension_row[CascadeState1HighestClientLogDerivative.ext_table_index()] =
                cascade_state_1_highest_log_derivative;
            extension_row[CascadeState1MidHighClientLogDerivative.ext_table_index()] =
                cascade_state_1_mid_high_log_derivative;
            extension_row[CascadeState1MidLowClientLogDerivative.ext_table_index()] =
                cascade_state_1_mid_low_log_derivative;
            extension_row[CascadeState1LowestClientLogDerivative.ext_table_index()] =
                cascade_state_1_lowest_log_derivative;
            extension_row[CascadeState2HighestClientLogDerivative.ext_table_index()] =
                cascade_state_2_highest_log_derivative;
            extension_row[CascadeState2MidHighClientLogDerivative.ext_table_index()] =
                cascade_state_2_mid_high_log_derivative;
            extension_row[CascadeState2MidLowClientLogDerivative.ext_table_index()] =
                cascade_state_2_mid_low_log_derivative;
            extension_row[CascadeState2LowestClientLogDerivative.ext_table_index()] =
                cascade_state_2_lowest_log_derivative;
            extension_row[CascadeState3HighestClientLogDerivative.ext_table_index()] =
                cascade_state_3_highest_log_derivative;
            extension_row[CascadeState3MidHighClientLogDerivative.ext_table_index()] =
                cascade_state_3_mid_high_log_derivative;
            extension_row[CascadeState3MidLowClientLogDerivative.ext_table_index()] =
                cascade_state_3_mid_low_log_derivative;
            extension_row[CascadeState3LowestClientLogDerivative.ext_table_index()] =
                cascade_state_3_lowest_log_derivative;
        }
        profiler!(stop "hash table");
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use air::table::TableId;
    use air::table_column::HashBaseTableColumn;
    use air::AIR;
    use std::collections::HashMap;

    use crate::shared_tests::ProgramAndInput;
    use crate::stark::tests::master_tables_for_low_security_level;
    use crate::table::master_table::MasterTable;
    use crate::triton_asm;
    use crate::triton_program;
    use crate::vm::VM;

    use super::*;

    #[test]
    fn hash_table_mode_discriminant_is_unique() {
        let mut discriminants_and_modes = HashMap::new();
        for mode in HashTableMode::iter() {
            let discriminant = u32::from(mode);
            let maybe_entry = discriminants_and_modes.insert(discriminant, mode);
            if let Some(entry) = maybe_entry {
                panic!("Discriminant collision for {discriminant} between {entry} and {mode}.");
            }
        }
    }

    #[test]
    fn terminal_constraints_hold_for_sponge_init_edge_case() {
        let many_sponge_inits = triton_asm![sponge_init; 23_631];
        let many_squeeze_absorbs = (0..2_100)
            .flat_map(|_| triton_asm!(sponge_squeeze sponge_absorb))
            .collect_vec();
        let program = triton_program! {
            {&many_sponge_inits}
            {&many_squeeze_absorbs}
            sponge_init
            halt
        };

        let (aet, _) = VM::trace_execution(&program, [].into(), [].into()).unwrap();
        dbg!(aet.height());
        dbg!(aet.padded_height());
        dbg!(aet.height_of_table(TableId::Hash));
        dbg!(aet.height_of_table(TableId::OpStack));
        dbg!(aet.height_of_table(TableId::Cascade));

        let (_, _, master_base_table, master_ext_table, challenges) =
            master_tables_for_low_security_level(ProgramAndInput::new(program));
        let challenges = &challenges.challenges;

        let master_base_trace_table = master_base_table.trace_table();
        let master_ext_trace_table = master_ext_table.trace_table();

        let last_row = master_base_trace_table.slice(s![-1.., ..]);
        let last_opcode = last_row[[0, HashBaseTableColumn::CI.master_base_table_index()]];
        let last_instruction: Instruction = last_opcode.value().try_into().unwrap();
        assert_eq!(Instruction::SpongeInit, last_instruction);

        let circuit_builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in HashTable::terminal_constraints(&circuit_builder)
            .into_iter()
            .map(|constraint_monad| constraint_monad.consume())
            .enumerate()
        {
            let evaluated_constraint = constraint.evaluate(
                master_base_trace_table.slice(s![-1.., ..]),
                master_ext_trace_table.slice(s![-1.., ..]),
                challenges,
            );
            assert_eq!(
                xfe!(0),
                evaluated_constraint,
                "Terminal constraint {constraint_idx} failed."
            );
        }
    }
}
