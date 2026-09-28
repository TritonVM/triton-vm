use air::challenge_id::ChallengeId;
use air::cross_table_argument::CrossTableArg;
use air::cross_table_argument::EvalArg;
use air::cross_table_argument::LookupArg;
use air::table::hash::HashTable;
use air::table::hash::HashTableMode;
use air::table::hash::MONTGOMERY_MODULUS;
use air::table::hash::PermutationTrace;
use air::table_column::MasterAuxColumn;
use air::table_column::MasterMainColumn;
use isa::instruction::Instruction;
use itertools::Itertools;
use ndarray::prelude::*;
use num_traits::ConstOne;
use num_traits::Zero;
use rayon::prelude::*;
use strum::EnumCount;
use strum::IntoEnumIterator;
use twenty_first::prelude::tip5::NUM_ROUNDS;
use twenty_first::prelude::tip5::STATE_SIZE;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::challenges::Challenges;
use crate::ndarray_helper::COL_AXIS;
use crate::profiler::profiler;
use crate::table::TraceTable;
use crate::table::running_arguments::par_log_derivative;
use crate::table::running_arguments::par_running_evaluation;

type MainColumn = <HashTable as air::AIR>::MainColumn;
type AuxColumn = <HashTable as air::AIR>::AuxColumn;

/// Return the 16-bit chunks of the “un-Montgomery'd” representation, in
/// little-endian chunk order. This (basically) translates to the application of
/// `σ(R·x)` for input `x`, which are the first two steps in Tip5's
/// split-and-lookup S-Box. `R` is the Montgomery modulus, _i.e._, `R = 2^64 mod
/// p`. `σ` as described in the paper decomposes the 64-bit input into 8-bit
/// limbs, whereas this method decomposes into 16-bit limbs for arithmetization
/// reasons; the 16-bit limbs are split into 8-bit limbs in the Cascade Table.
/// For a more in-depth explanation of all the necessary steps in the
/// split-and-lookup S-Box, see the
/// [Tip5 paper](https://eprint.iacr.org/2023/107.pdf).
///
/// Note: this is distinct from the seemingly similar
/// [`raw_u16s`](BFieldElement::raw_u16s).
pub(crate) fn base_field_element_into_16_bit_limbs(x: BFieldElement) -> [u16; 4] {
    let r_times_x = (MONTGOMERY_MODULUS * x).value();
    [0, 16, 32, 48].map(|shift| ((r_times_x >> shift) & 0xffff) as u16)
}

/// Convert a permutation trace to a segment in the Hash Table.
///
/// **Note**: The current instruction [`CI`][ci] is _not_ set.
///
/// [ci]: air::table_column::HashMainColumn::CI
pub(crate) fn trace_to_table_rows(trace: PermutationTrace) -> Array2<BFieldElement> {
    let mut table_rows = Array2::default([0, MainColumn::COUNT]);
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
    let row = Array1::zeros([MainColumn::COUNT]);
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
    row[MainColumn::RoundNumber.main_index()] = bfe!(round_number as u64);
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
    row[MainColumn::State0LowestLkIn.main_index()] = look_in_split[0];
    row[MainColumn::State0MidLowLkIn.main_index()] = look_in_split[1];
    row[MainColumn::State0MidHighLkIn.main_index()] = look_in_split[2];
    row[MainColumn::State0HighestLkIn.main_index()] = look_in_split[3];

    let look_out_split = limbs.map(crate::table::cascade::lookup_16_bit_limb);
    row[MainColumn::State0LowestLkOut.main_index()] = look_out_split[0];
    row[MainColumn::State0MidLowLkOut.main_index()] = look_out_split[1];
    row[MainColumn::State0MidHighLkOut.main_index()] = look_out_split[2];
    row[MainColumn::State0HighestLkOut.main_index()] = look_out_split[3];

    row
}

fn fill_split_state_element_1_of_row_using_trace_row(
    mut row: Array1<BFieldElement>,
    trace_row: [BFieldElement; STATE_SIZE],
) -> Array1<BFieldElement> {
    let limbs = base_field_element_into_16_bit_limbs(trace_row[1]);
    let look_in_split = limbs.map(|limb| bfe!(limb));
    row[MainColumn::State1LowestLkIn.main_index()] = look_in_split[0];
    row[MainColumn::State1MidLowLkIn.main_index()] = look_in_split[1];
    row[MainColumn::State1MidHighLkIn.main_index()] = look_in_split[2];
    row[MainColumn::State1HighestLkIn.main_index()] = look_in_split[3];

    let look_out_split = limbs.map(crate::table::cascade::lookup_16_bit_limb);
    row[MainColumn::State1LowestLkOut.main_index()] = look_out_split[0];
    row[MainColumn::State1MidLowLkOut.main_index()] = look_out_split[1];
    row[MainColumn::State1MidHighLkOut.main_index()] = look_out_split[2];
    row[MainColumn::State1HighestLkOut.main_index()] = look_out_split[3];

    row
}

fn fill_split_state_element_2_of_row_using_trace_row(
    mut row: Array1<BFieldElement>,
    trace_row: [BFieldElement; STATE_SIZE],
) -> Array1<BFieldElement> {
    let limbs = base_field_element_into_16_bit_limbs(trace_row[2]);
    let look_in_split = limbs.map(|limb| bfe!(limb));
    row[MainColumn::State2LowestLkIn.main_index()] = look_in_split[0];
    row[MainColumn::State2MidLowLkIn.main_index()] = look_in_split[1];
    row[MainColumn::State2MidHighLkIn.main_index()] = look_in_split[2];
    row[MainColumn::State2HighestLkIn.main_index()] = look_in_split[3];

    let look_out_split = limbs.map(crate::table::cascade::lookup_16_bit_limb);
    row[MainColumn::State2LowestLkOut.main_index()] = look_out_split[0];
    row[MainColumn::State2MidLowLkOut.main_index()] = look_out_split[1];
    row[MainColumn::State2MidHighLkOut.main_index()] = look_out_split[2];
    row[MainColumn::State2HighestLkOut.main_index()] = look_out_split[3];

    row
}

fn fill_split_state_element_3_of_row_using_trace_row(
    mut row: Array1<BFieldElement>,
    trace_row: [BFieldElement; STATE_SIZE],
) -> Array1<BFieldElement> {
    let limbs = base_field_element_into_16_bit_limbs(trace_row[3]);
    let look_in_split = limbs.map(|limb| bfe!(limb));
    row[MainColumn::State3LowestLkIn.main_index()] = look_in_split[0];
    row[MainColumn::State3MidLowLkIn.main_index()] = look_in_split[1];
    row[MainColumn::State3MidHighLkIn.main_index()] = look_in_split[2];
    row[MainColumn::State3HighestLkIn.main_index()] = look_in_split[3];

    let look_out_split = limbs.map(crate::table::cascade::lookup_16_bit_limb);
    row[MainColumn::State3LowestLkOut.main_index()] = look_out_split[0];
    row[MainColumn::State3MidLowLkOut.main_index()] = look_out_split[1];
    row[MainColumn::State3MidHighLkOut.main_index()] = look_out_split[2];
    row[MainColumn::State3HighestLkOut.main_index()] = look_out_split[3];

    row
}

fn fill_row_with_unsplit_state_elements_using_trace_row(
    mut row: Array1<BFieldElement>,
    trace_row: [BFieldElement; STATE_SIZE],
) -> Array1<BFieldElement> {
    row[MainColumn::State4.main_index()] = trace_row[4];
    row[MainColumn::State5.main_index()] = trace_row[5];
    row[MainColumn::State6.main_index()] = trace_row[6];
    row[MainColumn::State7.main_index()] = trace_row[7];
    row[MainColumn::State8.main_index()] = trace_row[8];
    row[MainColumn::State9.main_index()] = trace_row[9];
    row[MainColumn::State10.main_index()] = trace_row[10];
    row[MainColumn::State11.main_index()] = trace_row[11];
    row[MainColumn::State12.main_index()] = trace_row[12];
    row[MainColumn::State13.main_index()] = trace_row[13];
    row[MainColumn::State14.main_index()] = trace_row[14];
    row[MainColumn::State15.main_index()] = trace_row[15];
    row
}

fn fill_row_with_state_inverses_using_trace_row(
    mut row: Array1<BFieldElement>,
    trace_row: [BFieldElement; STATE_SIZE],
) -> Array1<BFieldElement> {
    row[MainColumn::State0Inv.main_index()] = inverse_or_zero_of_highest_2_limbs(trace_row[0]);
    row[MainColumn::State1Inv.main_index()] = inverse_or_zero_of_highest_2_limbs(trace_row[1]);
    row[MainColumn::State2Inv.main_index()] = inverse_or_zero_of_highest_2_limbs(trace_row[2]);
    row[MainColumn::State3Inv.main_index()] = inverse_or_zero_of_highest_2_limbs(trace_row[3]);
    row
}

/// The inverse-or-zero of (2^32 - 1 - 2^16·`highest` - `mid_high`) where
/// `highest` is the most significant limb of the given `state_element`, and
/// `mid_high` the second-most significant limb.
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
    row[MainColumn::Constant0.main_index()] = round_constants[0];
    row[MainColumn::Constant1.main_index()] = round_constants[1];
    row[MainColumn::Constant2.main_index()] = round_constants[2];
    row[MainColumn::Constant3.main_index()] = round_constants[3];
    row[MainColumn::Constant4.main_index()] = round_constants[4];
    row[MainColumn::Constant5.main_index()] = round_constants[5];
    row[MainColumn::Constant6.main_index()] = round_constants[6];
    row[MainColumn::Constant7.main_index()] = round_constants[7];
    row[MainColumn::Constant8.main_index()] = round_constants[8];
    row[MainColumn::Constant9.main_index()] = round_constants[9];
    row[MainColumn::Constant10.main_index()] = round_constants[10];
    row[MainColumn::Constant11.main_index()] = round_constants[11];
    row[MainColumn::Constant12.main_index()] = round_constants[12];
    row[MainColumn::Constant13.main_index()] = round_constants[13];
    row[MainColumn::Constant14.main_index()] = round_constants[14];
    row[MainColumn::Constant15.main_index()] = round_constants[15];
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

        let mode_column_idx = MainColumn::Mode.main_index();
        let mut program_hash_mode_column = program_hash_part.column_mut(mode_column_idx);
        let mut sponge_mode_column = sponge_part.column_mut(mode_column_idx);
        let mut hash_mode_column = hash_part.column_mut(mode_column_idx);

        program_hash_mode_column.fill(HashTableMode::ProgramHashing.into());
        sponge_mode_column.fill(HashTableMode::Sponge.into());
        hash_mode_column.fill(HashTableMode::Hash.into());
    }

    fn pad(mut main_table: ArrayViewMut2<BFieldElement>, table_length: usize) {
        let inverse_of_high_limbs = inverse_or_zero_of_highest_2_limbs(bfe!(0));
        for column_id in [
            MainColumn::State0Inv,
            MainColumn::State1Inv,
            MainColumn::State2Inv,
            MainColumn::State3Inv,
        ] {
            let column_index = column_id.main_index();
            let slice_info = s![table_length.., column_index];
            let mut column = main_table.slice_mut(slice_info);
            column.fill(inverse_of_high_limbs);
        }

        let round_constants = Self::tip5_round_constants_by_round_number(0);
        for (round_constant_idx, &round_constant) in round_constants.iter().enumerate() {
            let round_constant_column =
                HashTable::round_constant_column_by_index(round_constant_idx);
            let round_constant_column_idx = round_constant_column.main_index();
            let slice_info = s![table_length.., round_constant_column_idx];
            let mut column = main_table.slice_mut(slice_info);
            column.fill(round_constant);
        }

        let mode_column_index = MainColumn::Mode.main_index();
        let mode_column_slice_info = s![table_length.., mode_column_index];
        let mut mode_column = main_table.slice_mut(mode_column_slice_info);
        mode_column.fill(HashTableMode::Pad.into());

        let instruction_column_index = MainColumn::CI.main_index();
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
        assert_eq!(MainColumn::COUNT, main_table.ncols());
        assert_eq!(AuxColumn::COUNT, aux_table.ncols());
        assert_eq!(main_table.nrows(), aux_table.nrows());

        let ci_weight = challenges[ChallengeId::HashCIWeight];
        let hash_digest_eval_indeterminate = challenges[ChallengeId::HashDigestIndeterminate];
        let hash_input_eval_indeterminate = challenges[ChallengeId::HashInputIndeterminate];
        let sponge_eval_indeterminate = challenges[ChallengeId::SpongeIndeterminate];
        let cascade_indeterminate = challenges[ChallengeId::HashCascadeLookupIndeterminate];
        let send_chunk_indeterminate =
            challenges[ChallengeId::ProgramAttestationSendChunkIndeterminate];
        let prepare_chunk_indeterminate =
            challenges[ChallengeId::ProgramAttestationPrepareChunkIndeterminate];

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
                (row[highest.main_index()] * two_pow_48
                    + row[mid_high.main_index()] * two_pow_32
                    + row[mid_low.main_index()] * two_pow_16
                    + row[lowest.main_index()])
                    * montgomery_modulus_inverse
            };

        let rate_registers = |row: ArrayView1<BFieldElement>| {
            let state_0 = re_compose_state_element(
                row,
                MainColumn::State0HighestLkIn,
                MainColumn::State0MidHighLkIn,
                MainColumn::State0MidLowLkIn,
                MainColumn::State0LowestLkIn,
            );
            let state_1 = re_compose_state_element(
                row,
                MainColumn::State1HighestLkIn,
                MainColumn::State1MidHighLkIn,
                MainColumn::State1MidLowLkIn,
                MainColumn::State1LowestLkIn,
            );
            let state_2 = re_compose_state_element(
                row,
                MainColumn::State2HighestLkIn,
                MainColumn::State2MidHighLkIn,
                MainColumn::State2MidLowLkIn,
                MainColumn::State2LowestLkIn,
            );
            let state_3 = re_compose_state_element(
                row,
                MainColumn::State3HighestLkIn,
                MainColumn::State3MidHighLkIn,
                MainColumn::State3MidLowLkIn,
                MainColumn::State3LowestLkIn,
            );
            [
                state_0,
                state_1,
                state_2,
                state_3,
                row[MainColumn::State4.main_index()],
                row[MainColumn::State5.main_index()],
                row[MainColumn::State6.main_index()],
                row[MainColumn::State7.main_index()],
                row[MainColumn::State8.main_index()],
                row[MainColumn::State9.main_index()],
            ]
        };

        let state_weights = &challenges[ChallengeId::StackWeight0..ChallengeId::StackWeight10];
        let compressed_row = |row: ArrayView1<BFieldElement>| -> XFieldElement {
            rate_registers(row)
                .iter()
                .zip_eq(state_weights.iter())
                .map(|(&state, &weight)| weight * state)
                .sum()
        };
        let compressed_digest = |row: ArrayView1<BFieldElement>| -> XFieldElement {
            rate_registers(row)[..Digest::LEN]
                .iter()
                .zip_eq(state_weights[..Digest::LEN].iter())
                .map(|(&state, &weight)| weight * state)
                .sum()
        };

        let cascade_look_in_weight = challenges[ChallengeId::HashCascadeLookInWeight];
        let cascade_look_out_weight = challenges[ChallengeId::HashCascadeLookOutWeight];

        let mode = |row: ArrayView1<BFieldElement>| row[MainColumn::Mode.main_index()];
        let in_program_hashing_mode =
            |row: ArrayView1<BFieldElement>| mode(row) == HashTableMode::ProgramHashing.into();
        let in_sponge_mode =
            |row: ArrayView1<BFieldElement>| mode(row) == HashTableMode::Sponge.into();
        let in_hash_mode = |row: ArrayView1<BFieldElement>| mode(row) == HashTableMode::Hash.into();
        let in_pad_mode = |row: ArrayView1<BFieldElement>| mode(row) == HashTableMode::Pad.into();

        let round_number =
            |row: ArrayView1<BFieldElement>| row[MainColumn::RoundNumber.main_index()];
        let in_round_0 = |row: ArrayView1<BFieldElement>| round_number(row).is_zero();
        let in_last_round =
            |row: ArrayView1<BFieldElement>| round_number(row) == (NUM_ROUNDS as u64).into();

        let current_instruction = |row: ArrayView1<BFieldElement>| row[MainColumn::CI.main_index()];
        let current_instruction_is_sponge_init = |row: ArrayView1<BFieldElement>| {
            current_instruction(row) == Instruction::SpongeInit.opcode_b()
        };

        // The contribution of each row to the various running evaluations,
        // if any. The running evaluations are then computed in parallel.
        let receive_chunk_addend = |row: ArrayView1<BFieldElement>| {
            (in_program_hashing_mode(row) && in_round_0(row)).then(|| {
                EvalArg::compute_terminal(
                    &rate_registers(row),
                    EvalArg::default_initial(),
                    prepare_chunk_indeterminate,
                )
            })
        };
        let sponge_addend = |row: ArrayView1<BFieldElement>| {
            (in_sponge_mode(row) && in_round_0(row)).then(|| {
                let mut addend = ci_weight * current_instruction(row);
                if !current_instruction_is_sponge_init(row) {
                    addend += compressed_row(row);
                }
                addend
            })
        };
        let hash_input_addend = |row: ArrayView1<BFieldElement>| {
            (in_hash_mode(row) && in_round_0(row)).then(|| compressed_row(row))
        };
        let hash_digest_addend = |row: ArrayView1<BFieldElement>| {
            (in_hash_mode(row) && in_last_round(row)).then(|| compressed_digest(row))
        };

        let rows = || {
            (0..main_table.nrows())
                .into_par_iter()
                .map(|row_idx| main_table.row(row_idx))
        };
        let running_evaluation =
            |addend: &(dyn Fn(ArrayView1<BFieldElement>) -> Option<XFieldElement> + Sync),
             indeterminate| {
                let addends = rows().map(addend).collect::<Vec<_>>();
                par_running_evaluation(&addends, indeterminate, EvalArg::default_initial())
            };

        // The contribution of each row to the cascade table lookups' log
        // derivatives, if any. The inversions are batched.
        let contributes_to_cascade_lookup = |row: ArrayView1<BFieldElement>| {
            !in_pad_mode(row) && !in_last_round(row) && !current_instruction_is_sponge_init(row)
        };
        let cascade_log_derivative = |lk_in_col: Self::MainColumn, lk_out_col: Self::MainColumn| {
            let fractions = rows()
                .map(|row| {
                    contributes_to_cascade_lookup(row).then(|| {
                        let denominator = cascade_indeterminate
                            - cascade_look_in_weight * row[lk_in_col.main_index()]
                            - cascade_look_out_weight * row[lk_out_col.main_index()];
                        (denominator, XFieldElement::ONE)
                    })
                })
                .collect::<Vec<_>>();
            par_log_derivative(&fractions, LookupArg::default_initial())
        };

        let all_aux_columns = AuxColumn::iter().collect_vec();
        aux_table
            .axis_iter_mut(COL_AXIS)
            .into_par_iter()
            .zip(all_aux_columns)
            .enumerate()
            .for_each(|(column_idx, (target_column, aux_column))| {
                debug_assert_eq!(column_idx, aux_column.aux_index());
                let column = match aux_column {
                    AuxColumn::ReceiveChunkRunningEvaluation => {
                        running_evaluation(&receive_chunk_addend, send_chunk_indeterminate)
                    }
                    AuxColumn::HashInputRunningEvaluation => {
                        running_evaluation(&hash_input_addend, hash_input_eval_indeterminate)
                    }
                    AuxColumn::HashDigestRunningEvaluation => {
                        running_evaluation(&hash_digest_addend, hash_digest_eval_indeterminate)
                    }
                    AuxColumn::SpongeRunningEvaluation => {
                        running_evaluation(&sponge_addend, sponge_eval_indeterminate)
                    }
                    AuxColumn::CascadeState0HighestClientLogDerivative => cascade_log_derivative(
                        MainColumn::State0HighestLkIn,
                        MainColumn::State0HighestLkOut,
                    ),
                    AuxColumn::CascadeState0MidHighClientLogDerivative => cascade_log_derivative(
                        MainColumn::State0MidHighLkIn,
                        MainColumn::State0MidHighLkOut,
                    ),
                    AuxColumn::CascadeState0MidLowClientLogDerivative => cascade_log_derivative(
                        MainColumn::State0MidLowLkIn,
                        MainColumn::State0MidLowLkOut,
                    ),
                    AuxColumn::CascadeState0LowestClientLogDerivative => cascade_log_derivative(
                        MainColumn::State0LowestLkIn,
                        MainColumn::State0LowestLkOut,
                    ),
                    AuxColumn::CascadeState1HighestClientLogDerivative => cascade_log_derivative(
                        MainColumn::State1HighestLkIn,
                        MainColumn::State1HighestLkOut,
                    ),
                    AuxColumn::CascadeState1MidHighClientLogDerivative => cascade_log_derivative(
                        MainColumn::State1MidHighLkIn,
                        MainColumn::State1MidHighLkOut,
                    ),
                    AuxColumn::CascadeState1MidLowClientLogDerivative => cascade_log_derivative(
                        MainColumn::State1MidLowLkIn,
                        MainColumn::State1MidLowLkOut,
                    ),
                    AuxColumn::CascadeState1LowestClientLogDerivative => cascade_log_derivative(
                        MainColumn::State1LowestLkIn,
                        MainColumn::State1LowestLkOut,
                    ),
                    AuxColumn::CascadeState2HighestClientLogDerivative => cascade_log_derivative(
                        MainColumn::State2HighestLkIn,
                        MainColumn::State2HighestLkOut,
                    ),
                    AuxColumn::CascadeState2MidHighClientLogDerivative => cascade_log_derivative(
                        MainColumn::State2MidHighLkIn,
                        MainColumn::State2MidHighLkOut,
                    ),
                    AuxColumn::CascadeState2MidLowClientLogDerivative => cascade_log_derivative(
                        MainColumn::State2MidLowLkIn,
                        MainColumn::State2MidLowLkOut,
                    ),
                    AuxColumn::CascadeState2LowestClientLogDerivative => cascade_log_derivative(
                        MainColumn::State2LowestLkIn,
                        MainColumn::State2LowestLkOut,
                    ),
                    AuxColumn::CascadeState3HighestClientLogDerivative => cascade_log_derivative(
                        MainColumn::State3HighestLkIn,
                        MainColumn::State3HighestLkOut,
                    ),
                    AuxColumn::CascadeState3MidHighClientLogDerivative => cascade_log_derivative(
                        MainColumn::State3MidHighLkIn,
                        MainColumn::State3MidHighLkOut,
                    ),
                    AuxColumn::CascadeState3MidLowClientLogDerivative => cascade_log_derivative(
                        MainColumn::State3MidLowLkIn,
                        MainColumn::State3MidLowLkOut,
                    ),
                    AuxColumn::CascadeState3LowestClientLogDerivative => cascade_log_derivative(
                        MainColumn::State3LowestLkIn,
                        MainColumn::State3LowestLkOut,
                    ),
                };
                Array1::from(column).move_into(target_column);
            });
        profiler!(stop "hash table");
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) mod tests {
    use air::AIR;
    use air::table::TableId;
    use constraint_circuit::ConstraintCircuitBuilder;
    use std::collections::HashMap;
    use strum::IntoEnumIterator;

    use crate::shared_tests::TestableProgram;
    use crate::table::master_table::MasterTable;
    use crate::tests::test;
    use crate::triton_asm;
    use crate::triton_program;
    use crate::vm::VM;

    use super::*;

    #[macro_rules_attr::apply(test)]
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

    #[macro_rules_attr::apply(test)]
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

        let (aet, _) = VM::trace_execution(program.clone(), [].into(), [].into()).unwrap();
        dbg!(aet.height());
        dbg!(aet.padded_height());
        dbg!(aet.height_of_table(TableId::Hash));
        dbg!(aet.height_of_table(TableId::OpStack));
        dbg!(aet.height_of_table(TableId::Cascade));

        let proof_artifacts = TestableProgram::new(program).generate_proof_artifacts();
        let challenges = &proof_artifacts.challenges.challenges;
        let master_main_trace_table = proof_artifacts.master_main_table.trace_table();
        let master_aux_trace_table = proof_artifacts.master_aux_table.trace_table();

        let last_row = master_main_trace_table.slice(s![-1.., ..]);
        let last_opcode = last_row[[0, MainColumn::CI.master_main_index()]];
        let last_instruction: Instruction = last_opcode.value().try_into().unwrap();
        assert_eq!(Instruction::SpongeInit, last_instruction);

        let circuit_builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in HashTable::terminal_constraints(&circuit_builder)
            .into_iter()
            .map(|constraint_monad| constraint_monad.consume())
            .enumerate()
        {
            let evaluated_constraint = constraint.evaluate(
                master_main_trace_table.slice(s![-1.., ..]),
                master_aux_trace_table.slice(s![-1.., ..]),
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
