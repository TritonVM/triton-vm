use constraint_circuit::ConstraintCircuitBuilder;
use constraint_circuit::ConstraintCircuitMonad;
use constraint_circuit::DualRowIndicator;
use constraint_circuit::DualRowIndicator::CurrentAux;
use constraint_circuit::DualRowIndicator::CurrentMain;
use constraint_circuit::DualRowIndicator::NextAux;
use constraint_circuit::DualRowIndicator::NextMain;
use constraint_circuit::InputIndicator;
use constraint_circuit::SingleRowIndicator;
use constraint_circuit::SingleRowIndicator::Aux;
use constraint_circuit::SingleRowIndicator::Main;
use isa::instruction::Instruction;
use itertools::Itertools;
use strum::Display;
use strum::EnumCount;
use strum::EnumIter;
use strum::IntoEnumIterator;
use twenty_first::prelude::tip5::NUM_ROUNDS;
use twenty_first::prelude::*;

use crate::AIR;
use crate::challenge_id::ChallengeId;
use crate::cross_table_argument::CrossTableArg;
use crate::cross_table_argument::EvalArg;
use crate::cross_table_argument::LookupArg;
use crate::table_column::MasterAuxColumn;
use crate::table_column::MasterMainColumn;

pub const MONTGOMERY_MODULUS: BFieldElement =
    BFieldElement::new(((1_u128 << 64) % BFieldElement::P as u128) as u64);

const POWER_MAP_EXPONENT: u64 = 7;
const NUM_ROUND_CONSTANTS: usize = tip5::STATE_SIZE;

pub const PERMUTATION_TRACE_LENGTH: usize = NUM_ROUNDS + 1;

pub type PermutationTrace = [[BFieldElement; tip5::STATE_SIZE]; PERMUTATION_TRACE_LENGTH];

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct HashTable;

impl crate::private::Seal for HashTable {}

type MainColumn = <HashTable as AIR>::MainColumn;
type AuxColumn = <HashTable as AIR>::AuxColumn;

impl HashTable {
    /// Get the MDS matrix's entry in row `row_idx` and column `col_idx`.
    const fn mds_matrix_entry(row_idx: usize, col_idx: usize) -> BFieldElement {
        assert!(row_idx < tip5::STATE_SIZE);
        assert!(col_idx < tip5::STATE_SIZE);
        let index_in_matrix_defining_column =
            (tip5::STATE_SIZE + row_idx - col_idx) % tip5::STATE_SIZE;
        let mds_matrix_entry = tip5::MDS_MATRIX_FIRST_COLUMN[index_in_matrix_defining_column];
        BFieldElement::new(mds_matrix_entry as u64)
    }

    /// The round constants for round `r` if it is a valid round number in the
    /// Tip5 permutation, and the zero vector otherwise.
    pub fn tip5_round_constants_by_round_number(r: usize) -> [BFieldElement; NUM_ROUND_CONSTANTS] {
        if r >= NUM_ROUNDS {
            return bfe_array![0; NUM_ROUND_CONSTANTS];
        }

        let range_start = NUM_ROUND_CONSTANTS * r;
        let range_end = NUM_ROUND_CONSTANTS * (r + 1);
        tip5::ROUND_CONSTANTS[range_start..range_end]
            .try_into()
            .unwrap()
    }

    /// Construct one of the states 0 through 3 from its constituent limbs.
    /// For example, state 0 (prior to it being looked up in the
    /// split-and-lookup S-Box, which is usually the desired version of the
    /// state) is constructed from limbs [`State0HighestLkIn`][hi] through
    /// [`State0LowestLkIn`][lo].
    ///
    /// States 4 through 15 are directly accessible. See also the slightly
    /// related [`Self::state_column_by_index`].
    ///
    /// [hi]: crate::table_column::HashMainColumn::State0HighestLkIn
    /// [lo]: crate::table_column::HashMainColumn::State0LowestLkIn
    fn re_compose_16_bit_limbs<II: InputIndicator>(
        circuit_builder: &ConstraintCircuitBuilder<II>,
        highest: ConstraintCircuitMonad<II>,
        mid_high: ConstraintCircuitMonad<II>,
        mid_low: ConstraintCircuitMonad<II>,
        lowest: ConstraintCircuitMonad<II>,
    ) -> ConstraintCircuitMonad<II> {
        let constant = |c: u64| circuit_builder.b_constant(c);
        let montgomery_modulus_inv = circuit_builder.b_constant(MONTGOMERY_MODULUS.inverse());

        let sum_of_shifted_limbs = highest * constant(1 << 48)
            + mid_high * constant(1 << 32)
            + mid_low * constant(1 << 16)
            + lowest;
        sum_of_shifted_limbs * montgomery_modulus_inv
    }

    /// A constraint circuit evaluating to zero if and only if the given
    /// `round_number_circuit_node` is not equal to the given
    /// `round_number_to_deselect`.
    fn round_number_deselector<II: InputIndicator>(
        circuit_builder: &ConstraintCircuitBuilder<II>,
        round_number_circuit_node: &ConstraintCircuitMonad<II>,
        round_number_to_deselect: usize,
    ) -> ConstraintCircuitMonad<II> {
        assert!(
            round_number_to_deselect <= NUM_ROUNDS,
            "Round number must be in [0, {NUM_ROUNDS}] but got {round_number_to_deselect}."
        );
        let constant = |c: u64| circuit_builder.b_constant(c);

        // To not subtract zero from the first factor: some special casing.
        let first_factor = match round_number_to_deselect {
            0 => constant(1),
            _ => round_number_circuit_node.clone(),
        };
        (1..=NUM_ROUNDS)
            .filter(|&r| r != round_number_to_deselect)
            .map(|r| round_number_circuit_node.clone() - constant(r as u64))
            .fold(first_factor, |a, b| a * b)
    }

    /// A constraint circuit evaluating to zero if and only if the given
    /// `mode_circuit_node` is equal to the given `mode_to_select`.
    fn select_mode<II: InputIndicator>(
        circuit_builder: &ConstraintCircuitBuilder<II>,
        mode_circuit_node: &ConstraintCircuitMonad<II>,
        mode_to_select: HashTableMode,
    ) -> ConstraintCircuitMonad<II> {
        mode_circuit_node.clone() - circuit_builder.b_constant(mode_to_select)
    }

    /// A constraint circuit evaluating to zero if and only if the given
    /// `mode_circuit_node` is not equal to the given `mode_to_deselect`.
    fn mode_deselector<II: InputIndicator>(
        circuit_builder: &ConstraintCircuitBuilder<II>,
        mode_circuit_node: &ConstraintCircuitMonad<II>,
        mode_to_deselect: HashTableMode,
    ) -> ConstraintCircuitMonad<II> {
        let constant = |c: u64| circuit_builder.b_constant(c);
        HashTableMode::iter()
            .filter(|&mode| mode != mode_to_deselect)
            .map(|mode| mode_circuit_node.clone() - constant(mode.into()))
            .fold(constant(1), |accumulator, factor| accumulator * factor)
    }

    fn instruction_deselector<II: InputIndicator>(
        circuit_builder: &ConstraintCircuitBuilder<II>,
        current_instruction_node: &ConstraintCircuitMonad<II>,
        instruction_to_deselect: Instruction,
    ) -> ConstraintCircuitMonad<II> {
        let constant = |c: u64| circuit_builder.b_constant(c);
        let opcode = |instruction: Instruction| circuit_builder.b_constant(instruction.opcode_b());
        let relevant_instructions = [
            Instruction::Hash,
            Instruction::SpongeInit,
            Instruction::SpongeAbsorb,
            Instruction::SpongeSqueeze,
        ];
        assert!(relevant_instructions.contains(&instruction_to_deselect));

        relevant_instructions
            .iter()
            .filter(|&instruction| instruction != &instruction_to_deselect)
            .map(|&instruction| current_instruction_node.clone() - opcode(instruction))
            .fold(constant(1), |accumulator, factor| accumulator * factor)
    }

    /// The [main column][main_col] for the round constant corresponding to the
    /// given index. Valid indices are 0 through 15, corresponding to the 16
    /// round constants [`Constant0`][c0] through [`Constant15`][c15].
    ///
    /// [main_col]: crate::table_column::HashMainColumn
    /// [c0]: crate::table_column::HashMainColumn::Constant0
    /// [c15]: crate::table_column::HashMainColumn::Constant15
    pub fn round_constant_column_by_index(index: usize) -> MainColumn {
        match index {
            0 => MainColumn::Constant0,
            1 => MainColumn::Constant1,
            2 => MainColumn::Constant2,
            3 => MainColumn::Constant3,
            4 => MainColumn::Constant4,
            5 => MainColumn::Constant5,
            6 => MainColumn::Constant6,
            7 => MainColumn::Constant7,
            8 => MainColumn::Constant8,
            9 => MainColumn::Constant9,
            10 => MainColumn::Constant10,
            11 => MainColumn::Constant11,
            12 => MainColumn::Constant12,
            13 => MainColumn::Constant13,
            14 => MainColumn::Constant14,
            15 => MainColumn::Constant15,
            _ => panic!("invalid constant column index"),
        }
    }

    /// The [`HashMainColumn`][MainColumn] for the state corresponding to the
    /// given index. Valid indices are 4 through 15, corresponding to the 12
    /// state columns [`State4`][state_4] through [`State15`][state_15].
    ///
    /// States with indices 0 through 3 have to be assembled from the respective
    /// limbs; see [`Self::re_compose_states_0_through_3_before_lookup`]
    /// or [`Self::re_compose_16_bit_limbs`].
    ///
    /// [state_4]: crate::table_column::HashMainColumn::State4
    /// [state_15]: crate::table_column::HashMainColumn::State15
    fn state_column_by_index(index: usize) -> MainColumn {
        match index {
            4 => MainColumn::State4,
            5 => MainColumn::State5,
            6 => MainColumn::State6,
            7 => MainColumn::State7,
            8 => MainColumn::State8,
            9 => MainColumn::State9,
            10 => MainColumn::State10,
            11 => MainColumn::State11,
            12 => MainColumn::State12,
            13 => MainColumn::State13,
            14 => MainColumn::State14,
            15 => MainColumn::State15,
            _ => panic!("invalid state column index"),
        }
    }

    fn indicate_column_index_in_main_row(column: MainColumn) -> SingleRowIndicator {
        Main(column.master_main_index())
    }

    fn indicate_column_index_in_current_main_row(column: MainColumn) -> DualRowIndicator {
        CurrentMain(column.master_main_index())
    }

    fn indicate_column_index_in_next_main_row(column: MainColumn) -> DualRowIndicator {
        NextMain(column.master_main_index())
    }

    fn re_compose_states_0_through_3_before_lookup<II: InputIndicator>(
        circuit_builder: &ConstraintCircuitBuilder<II>,
        main_row_to_input_indicator: fn(MainColumn) -> II,
    ) -> [ConstraintCircuitMonad<II>; 4] {
        let input = |input_indicator: II| circuit_builder.input(input_indicator);
        let state_0 = Self::re_compose_16_bit_limbs(
            circuit_builder,
            input(main_row_to_input_indicator(MainColumn::State0HighestLkIn)),
            input(main_row_to_input_indicator(MainColumn::State0MidHighLkIn)),
            input(main_row_to_input_indicator(MainColumn::State0MidLowLkIn)),
            input(main_row_to_input_indicator(MainColumn::State0LowestLkIn)),
        );
        let state_1 = Self::re_compose_16_bit_limbs(
            circuit_builder,
            input(main_row_to_input_indicator(MainColumn::State1HighestLkIn)),
            input(main_row_to_input_indicator(MainColumn::State1MidHighLkIn)),
            input(main_row_to_input_indicator(MainColumn::State1MidLowLkIn)),
            input(main_row_to_input_indicator(MainColumn::State1LowestLkIn)),
        );
        let state_2 = Self::re_compose_16_bit_limbs(
            circuit_builder,
            input(main_row_to_input_indicator(MainColumn::State2HighestLkIn)),
            input(main_row_to_input_indicator(MainColumn::State2MidHighLkIn)),
            input(main_row_to_input_indicator(MainColumn::State2MidLowLkIn)),
            input(main_row_to_input_indicator(MainColumn::State2LowestLkIn)),
        );
        let state_3 = Self::re_compose_16_bit_limbs(
            circuit_builder,
            input(main_row_to_input_indicator(MainColumn::State3HighestLkIn)),
            input(main_row_to_input_indicator(MainColumn::State3MidHighLkIn)),
            input(main_row_to_input_indicator(MainColumn::State3MidLowLkIn)),
            input(main_row_to_input_indicator(MainColumn::State3LowestLkIn)),
        );
        [state_0, state_1, state_2, state_3]
    }

    fn tip5_constraints_as_circuits(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> (
        [ConstraintCircuitMonad<DualRowIndicator>; tip5::STATE_SIZE],
        [ConstraintCircuitMonad<DualRowIndicator>; tip5::STATE_SIZE],
    ) {
        let constant = |c: u64| circuit_builder.b_constant(c);
        let b_constant = |c| circuit_builder.b_constant(c);
        let current_main_row = |column_idx: MainColumn| {
            circuit_builder.input(CurrentMain(column_idx.master_main_index()))
        };
        let next_main_row = |column_idx: MainColumn| {
            circuit_builder.input(NextMain(column_idx.master_main_index()))
        };

        let state_0_after_lookup = Self::re_compose_16_bit_limbs(
            circuit_builder,
            current_main_row(MainColumn::State0HighestLkOut),
            current_main_row(MainColumn::State0MidHighLkOut),
            current_main_row(MainColumn::State0MidLowLkOut),
            current_main_row(MainColumn::State0LowestLkOut),
        );
        let state_1_after_lookup = Self::re_compose_16_bit_limbs(
            circuit_builder,
            current_main_row(MainColumn::State1HighestLkOut),
            current_main_row(MainColumn::State1MidHighLkOut),
            current_main_row(MainColumn::State1MidLowLkOut),
            current_main_row(MainColumn::State1LowestLkOut),
        );
        let state_2_after_lookup = Self::re_compose_16_bit_limbs(
            circuit_builder,
            current_main_row(MainColumn::State2HighestLkOut),
            current_main_row(MainColumn::State2MidHighLkOut),
            current_main_row(MainColumn::State2MidLowLkOut),
            current_main_row(MainColumn::State2LowestLkOut),
        );
        let state_3_after_lookup = Self::re_compose_16_bit_limbs(
            circuit_builder,
            current_main_row(MainColumn::State3HighestLkOut),
            current_main_row(MainColumn::State3MidHighLkOut),
            current_main_row(MainColumn::State3MidLowLkOut),
            current_main_row(MainColumn::State3LowestLkOut),
        );

        let state_part_before_power_map: [_; tip5::STATE_SIZE - tip5::NUM_SPLIT_AND_LOOKUP] = [
            MainColumn::State4,
            MainColumn::State5,
            MainColumn::State6,
            MainColumn::State7,
            MainColumn::State8,
            MainColumn::State9,
            MainColumn::State10,
            MainColumn::State11,
            MainColumn::State12,
            MainColumn::State13,
            MainColumn::State14,
            MainColumn::State15,
        ]
        .map(current_main_row);

        let state_part_after_power_map = {
            let mut exponentiation_accumulator = state_part_before_power_map.clone();
            for _ in 1..POWER_MAP_EXPONENT {
                for (i, state) in exponentiation_accumulator.iter_mut().enumerate() {
                    *state = state.clone() * state_part_before_power_map[i].clone();
                }
            }
            exponentiation_accumulator
        };

        let state_after_s_box_application = [
            state_0_after_lookup,
            state_1_after_lookup,
            state_2_after_lookup,
            state_3_after_lookup,
            state_part_after_power_map[0].clone(),
            state_part_after_power_map[1].clone(),
            state_part_after_power_map[2].clone(),
            state_part_after_power_map[3].clone(),
            state_part_after_power_map[4].clone(),
            state_part_after_power_map[5].clone(),
            state_part_after_power_map[6].clone(),
            state_part_after_power_map[7].clone(),
            state_part_after_power_map[8].clone(),
            state_part_after_power_map[9].clone(),
            state_part_after_power_map[10].clone(),
            state_part_after_power_map[11].clone(),
        ];

        let mut state_after_matrix_multiplication = vec![constant(0); tip5::STATE_SIZE];
        for (row_idx, acc) in state_after_matrix_multiplication.iter_mut().enumerate() {
            for (col_idx, state) in state_after_s_box_application.iter().enumerate() {
                let matrix_entry = b_constant(Self::mds_matrix_entry(row_idx, col_idx));
                *acc = acc.clone() + matrix_entry * state.clone();
            }
        }

        let round_constants: [_; tip5::STATE_SIZE] = [
            MainColumn::Constant0,
            MainColumn::Constant1,
            MainColumn::Constant2,
            MainColumn::Constant3,
            MainColumn::Constant4,
            MainColumn::Constant5,
            MainColumn::Constant6,
            MainColumn::Constant7,
            MainColumn::Constant8,
            MainColumn::Constant9,
            MainColumn::Constant10,
            MainColumn::Constant11,
            MainColumn::Constant12,
            MainColumn::Constant13,
            MainColumn::Constant14,
            MainColumn::Constant15,
        ]
        .map(current_main_row);

        let state_after_round_constant_addition = state_after_matrix_multiplication
            .into_iter()
            .zip_eq(round_constants)
            .map(|(st, rndc)| st + rndc)
            .collect_vec();

        let [state_0_next, state_1_next, state_2_next, state_3_next] =
            Self::re_compose_states_0_through_3_before_lookup(
                circuit_builder,
                Self::indicate_column_index_in_next_main_row,
            );
        let state_next = [
            state_0_next,
            state_1_next,
            state_2_next,
            state_3_next,
            next_main_row(MainColumn::State4),
            next_main_row(MainColumn::State5),
            next_main_row(MainColumn::State6),
            next_main_row(MainColumn::State7),
            next_main_row(MainColumn::State8),
            next_main_row(MainColumn::State9),
            next_main_row(MainColumn::State10),
            next_main_row(MainColumn::State11),
            next_main_row(MainColumn::State12),
            next_main_row(MainColumn::State13),
            next_main_row(MainColumn::State14),
            next_main_row(MainColumn::State15),
        ];

        let round_number_next = next_main_row(MainColumn::RoundNumber);
        let hash_function_round_correctly_performs_update = state_after_round_constant_addition
            .into_iter()
            .zip_eq(state_next.clone())
            .map(|(state_element, state_element_next)| {
                round_number_next.clone() * (state_element - state_element_next)
            })
            .collect_vec()
            .try_into()
            .unwrap();

        (state_next, hash_function_round_correctly_performs_update)
    }

    fn cascade_log_derivative_update_circuit(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        look_in_column: MainColumn,
        look_out_column: MainColumn,
        cascade_log_derivative_column: AuxColumn,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let challenge = |c| circuit_builder.challenge(c);
        let opcode = |instruction: Instruction| circuit_builder.b_constant(instruction.opcode_b());
        let constant = |c: u32| circuit_builder.b_constant(c);
        let next_main_row = |column_idx: MainColumn| {
            circuit_builder.input(NextMain(column_idx.master_main_index()))
        };
        let current_aux_row = |column_idx: AuxColumn| {
            circuit_builder.input(CurrentAux(column_idx.master_aux_index()))
        };
        let next_aux_row =
            |column_idx: AuxColumn| circuit_builder.input(NextAux(column_idx.master_aux_index()));

        let cascade_indeterminate = challenge(ChallengeId::HashCascadeLookupIndeterminate);
        let look_in_weight = challenge(ChallengeId::HashCascadeLookInWeight);
        let look_out_weight = challenge(ChallengeId::HashCascadeLookOutWeight);

        let ci_next = next_main_row(MainColumn::CI);
        let mode_next = next_main_row(MainColumn::Mode);
        let round_number_next = next_main_row(MainColumn::RoundNumber);
        let cascade_log_derivative = current_aux_row(cascade_log_derivative_column);
        let cascade_log_derivative_next = next_aux_row(cascade_log_derivative_column);

        let compressed_row = look_in_weight * next_main_row(look_in_column)
            + look_out_weight * next_main_row(look_out_column);

        let cascade_log_derivative_remains =
            cascade_log_derivative_next.clone() - cascade_log_derivative.clone();
        let cascade_log_derivative_updates = (cascade_log_derivative_next - cascade_log_derivative)
            * (cascade_indeterminate - compressed_row)
            - constant(1);

        let next_row_is_padding_row_or_round_number_next_is_max_or_ci_next_is_sponge_init =
            Self::select_mode(circuit_builder, &mode_next, HashTableMode::Pad)
                * (round_number_next.clone() - constant(NUM_ROUNDS as u32))
                * (ci_next.clone() - opcode(Instruction::SpongeInit));
        let round_number_next_is_not_num_rounds =
            Self::round_number_deselector(circuit_builder, &round_number_next, NUM_ROUNDS);
        let current_instruction_next_is_not_sponge_init =
            Self::instruction_deselector(circuit_builder, &ci_next, Instruction::SpongeInit);

        next_row_is_padding_row_or_round_number_next_is_max_or_ci_next_is_sponge_init
            * cascade_log_derivative_updates
            + round_number_next_is_not_num_rounds * cascade_log_derivative_remains.clone()
            + current_instruction_next_is_not_sponge_init * cascade_log_derivative_remains
    }
}

impl AIR for HashTable {
    type MainColumn = crate::table_column::HashMainColumn;
    type AuxColumn = crate::table_column::HashAuxColumn;

    fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let constant = |c: u64| circuit_builder.b_constant(c);

        let main_row =
            |column: Self::MainColumn| circuit_builder.input(Main(column.master_main_index()));
        let aux_row =
            |column: Self::AuxColumn| circuit_builder.input(Aux(column.master_aux_index()));

        let running_evaluation_initial = circuit_builder.x_constant(EvalArg::default_initial());
        let lookup_arg_default_initial = circuit_builder.x_constant(LookupArg::default_initial());

        let mode = main_row(Self::MainColumn::Mode);
        let running_evaluation_hash_input = aux_row(Self::AuxColumn::HashInputRunningEvaluation);
        let running_evaluation_hash_digest = aux_row(Self::AuxColumn::HashDigestRunningEvaluation);
        let running_evaluation_sponge = aux_row(Self::AuxColumn::SpongeRunningEvaluation);
        let running_evaluation_receive_chunk =
            aux_row(Self::AuxColumn::ReceiveChunkRunningEvaluation);

        let cascade_indeterminate = challenge(ChallengeId::HashCascadeLookupIndeterminate);
        let look_in_weight = challenge(ChallengeId::HashCascadeLookInWeight);
        let look_out_weight = challenge(ChallengeId::HashCascadeLookOutWeight);
        let prepare_chunk_indeterminate =
            challenge(ChallengeId::ProgramAttestationPrepareChunkIndeterminate);
        let receive_chunk_indeterminate =
            challenge(ChallengeId::ProgramAttestationSendChunkIndeterminate);

        // First chunk of the program is received correctly. Relates to program
        // attestation.
        let [state_0, state_1, state_2, state_3] =
            Self::re_compose_states_0_through_3_before_lookup(
                circuit_builder,
                Self::indicate_column_index_in_main_row,
            );
        let state_rate_part: [_; tip5::RATE] = [
            state_0,
            state_1,
            state_2,
            state_3,
            main_row(Self::MainColumn::State4),
            main_row(Self::MainColumn::State5),
            main_row(Self::MainColumn::State6),
            main_row(Self::MainColumn::State7),
            main_row(Self::MainColumn::State8),
            main_row(Self::MainColumn::State9),
        ];
        let compressed_chunk = state_rate_part
            .into_iter()
            .fold(running_evaluation_initial.clone(), |acc, state_element| {
                acc * prepare_chunk_indeterminate.clone() + state_element
            });
        let running_evaluation_receive_chunk_is_initialized_correctly =
            running_evaluation_receive_chunk
                - receive_chunk_indeterminate * running_evaluation_initial.clone()
                - compressed_chunk;

        // The lookup arguments with the Cascade Table for the S-Boxes are
        // initialized correctly.
        let cascade_log_derivative_init_circuit =
            |look_in_column, look_out_column, cascade_log_derivative_column| {
                let look_in = main_row(look_in_column);
                let look_out = main_row(look_out_column);
                let compressed_row =
                    look_in_weight.clone() * look_in + look_out_weight.clone() * look_out;
                let cascade_log_derivative = aux_row(cascade_log_derivative_column);
                (cascade_log_derivative - lookup_arg_default_initial.clone())
                    * (cascade_indeterminate.clone() - compressed_row)
                    - constant(1)
            };

        // miscellaneous initial constraints
        let mode_is_program_hashing =
            Self::select_mode(circuit_builder, &mode, HashTableMode::ProgramHashing);
        let round_number_is_0 = main_row(Self::MainColumn::RoundNumber);
        let running_evaluation_hash_input_is_default_initial =
            running_evaluation_hash_input - running_evaluation_initial.clone();
        let running_evaluation_hash_digest_is_default_initial =
            running_evaluation_hash_digest - running_evaluation_initial.clone();
        let running_evaluation_sponge_is_default_initial =
            running_evaluation_sponge - running_evaluation_initial;

        vec![
            mode_is_program_hashing,
            round_number_is_0,
            running_evaluation_hash_input_is_default_initial,
            running_evaluation_hash_digest_is_default_initial,
            running_evaluation_sponge_is_default_initial,
            running_evaluation_receive_chunk_is_initialized_correctly,
            cascade_log_derivative_init_circuit(
                Self::MainColumn::State0HighestLkIn,
                Self::MainColumn::State0HighestLkOut,
                Self::AuxColumn::CascadeState0HighestClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                Self::MainColumn::State0MidHighLkIn,
                Self::MainColumn::State0MidHighLkOut,
                Self::AuxColumn::CascadeState0MidHighClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                Self::MainColumn::State0MidLowLkIn,
                Self::MainColumn::State0MidLowLkOut,
                Self::AuxColumn::CascadeState0MidLowClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                Self::MainColumn::State0LowestLkIn,
                Self::MainColumn::State0LowestLkOut,
                Self::AuxColumn::CascadeState0LowestClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                Self::MainColumn::State1HighestLkIn,
                Self::MainColumn::State1HighestLkOut,
                Self::AuxColumn::CascadeState1HighestClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                Self::MainColumn::State1MidHighLkIn,
                Self::MainColumn::State1MidHighLkOut,
                Self::AuxColumn::CascadeState1MidHighClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                Self::MainColumn::State1MidLowLkIn,
                Self::MainColumn::State1MidLowLkOut,
                Self::AuxColumn::CascadeState1MidLowClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                Self::MainColumn::State1LowestLkIn,
                Self::MainColumn::State1LowestLkOut,
                Self::AuxColumn::CascadeState1LowestClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                Self::MainColumn::State2HighestLkIn,
                Self::MainColumn::State2HighestLkOut,
                Self::AuxColumn::CascadeState2HighestClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                Self::MainColumn::State2MidHighLkIn,
                Self::MainColumn::State2MidHighLkOut,
                Self::AuxColumn::CascadeState2MidHighClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                Self::MainColumn::State2MidLowLkIn,
                Self::MainColumn::State2MidLowLkOut,
                Self::AuxColumn::CascadeState2MidLowClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                Self::MainColumn::State2LowestLkIn,
                Self::MainColumn::State2LowestLkOut,
                Self::AuxColumn::CascadeState2LowestClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                Self::MainColumn::State3HighestLkIn,
                Self::MainColumn::State3HighestLkOut,
                Self::AuxColumn::CascadeState3HighestClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                Self::MainColumn::State3MidHighLkIn,
                Self::MainColumn::State3MidHighLkOut,
                Self::AuxColumn::CascadeState3MidHighClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                Self::MainColumn::State3MidLowLkIn,
                Self::MainColumn::State3MidLowLkOut,
                Self::AuxColumn::CascadeState3MidLowClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                Self::MainColumn::State3LowestLkIn,
                Self::MainColumn::State3LowestLkOut,
                Self::AuxColumn::CascadeState3LowestClientLogDerivative,
            ),
        ]
    }

    fn consistency_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let opcode = |instruction: Instruction| circuit_builder.b_constant(instruction.opcode_b());
        let constant = |c: u64| circuit_builder.b_constant(c);
        let main_row = |column_id: Self::MainColumn| {
            circuit_builder.input(Main(column_id.master_main_index()))
        };

        let mode = main_row(Self::MainColumn::Mode);
        let ci = main_row(Self::MainColumn::CI);
        let round_number = main_row(Self::MainColumn::RoundNumber);

        let ci_is_hash = ci.clone() - opcode(Instruction::Hash);
        let ci_is_sponge_init = ci.clone() - opcode(Instruction::SpongeInit);
        let ci_is_sponge_absorb = ci.clone() - opcode(Instruction::SpongeAbsorb);
        let ci_is_sponge_squeeze = ci - opcode(Instruction::SpongeSqueeze);

        let mode_is_not_hash = Self::mode_deselector(circuit_builder, &mode, HashTableMode::Hash);
        let round_number_is_not_0 =
            Self::round_number_deselector(circuit_builder, &round_number, 0);

        let mode_is_a_valid_mode =
            Self::mode_deselector(circuit_builder, &mode, HashTableMode::Pad)
                * Self::select_mode(circuit_builder, &mode, HashTableMode::Pad);

        let if_mode_is_not_sponge_then_ci_is_hash =
            Self::select_mode(circuit_builder, &mode, HashTableMode::Sponge) * ci_is_hash.clone();

        let if_mode_is_sponge_then_ci_is_a_sponge_instruction =
            Self::mode_deselector(circuit_builder, &mode, HashTableMode::Sponge)
                * ci_is_sponge_init
                * ci_is_sponge_absorb.clone()
                * ci_is_sponge_squeeze.clone();

        let if_padding_mode_then_round_number_is_0 =
            Self::mode_deselector(circuit_builder, &mode, HashTableMode::Pad)
                * round_number.clone();

        let if_ci_is_sponge_init_then_ = ci_is_hash * ci_is_sponge_absorb * ci_is_sponge_squeeze;
        let if_ci_is_sponge_init_then_round_number_is_0 =
            if_ci_is_sponge_init_then_.clone() * round_number.clone();

        let if_ci_is_sponge_init_then_rate_is_0 = (10..=15).map(|state_index| {
            let state_element = main_row(Self::state_column_by_index(state_index));
            if_ci_is_sponge_init_then_.clone() * state_element
        });

        let if_mode_is_hash_and_round_no_is_0_then_ = round_number_is_not_0 * mode_is_not_hash;
        let if_mode_is_hash_and_round_no_is_0_then_states_10_through_15_are_1 =
            (10..=15).map(|state_index| {
                let state_element = main_row(Self::state_column_by_index(state_index));
                if_mode_is_hash_and_round_no_is_0_then_.clone() * (state_element - constant(1))
            });

        // consistency of the inverse of the highest 2 limbs minus 2^32 - 1
        let one = constant(1);
        let two_pow_16 = constant(1 << 16);
        let two_pow_32 = constant(1 << 32);
        let state_0_hi_limbs_minus_2_pow_32 = two_pow_32.clone()
            - one.clone()
            - main_row(Self::MainColumn::State0HighestLkIn) * two_pow_16.clone()
            - main_row(Self::MainColumn::State0MidHighLkIn);
        let state_1_hi_limbs_minus_2_pow_32 = two_pow_32.clone()
            - one.clone()
            - main_row(Self::MainColumn::State1HighestLkIn) * two_pow_16.clone()
            - main_row(Self::MainColumn::State1MidHighLkIn);
        let state_2_hi_limbs_minus_2_pow_32 = two_pow_32.clone()
            - one.clone()
            - main_row(Self::MainColumn::State2HighestLkIn) * two_pow_16.clone()
            - main_row(Self::MainColumn::State2MidHighLkIn);
        let state_3_hi_limbs_minus_2_pow_32 = two_pow_32
            - one.clone()
            - main_row(Self::MainColumn::State3HighestLkIn) * two_pow_16.clone()
            - main_row(Self::MainColumn::State3MidHighLkIn);

        let state_0_hi_limbs_inv = main_row(Self::MainColumn::State0Inv);
        let state_1_hi_limbs_inv = main_row(Self::MainColumn::State1Inv);
        let state_2_hi_limbs_inv = main_row(Self::MainColumn::State2Inv);
        let state_3_hi_limbs_inv = main_row(Self::MainColumn::State3Inv);

        let state_0_hi_limbs_are_not_all_1s =
            state_0_hi_limbs_minus_2_pow_32.clone() * state_0_hi_limbs_inv.clone() - one.clone();
        let state_1_hi_limbs_are_not_all_1s =
            state_1_hi_limbs_minus_2_pow_32.clone() * state_1_hi_limbs_inv.clone() - one.clone();
        let state_2_hi_limbs_are_not_all_1s =
            state_2_hi_limbs_minus_2_pow_32.clone() * state_2_hi_limbs_inv.clone() - one.clone();
        let state_3_hi_limbs_are_not_all_1s =
            state_3_hi_limbs_minus_2_pow_32.clone() * state_3_hi_limbs_inv.clone() - one;

        let state_0_hi_limbs_inv_is_inv_or_is_zero =
            state_0_hi_limbs_are_not_all_1s.clone() * state_0_hi_limbs_inv;
        let state_1_hi_limbs_inv_is_inv_or_is_zero =
            state_1_hi_limbs_are_not_all_1s.clone() * state_1_hi_limbs_inv;
        let state_2_hi_limbs_inv_is_inv_or_is_zero =
            state_2_hi_limbs_are_not_all_1s.clone() * state_2_hi_limbs_inv;
        let state_3_hi_limbs_inv_is_inv_or_is_zero =
            state_3_hi_limbs_are_not_all_1s.clone() * state_3_hi_limbs_inv;

        let state_0_hi_limbs_inv_is_inv_or_state_0_hi_limbs_is_zero =
            state_0_hi_limbs_are_not_all_1s.clone() * state_0_hi_limbs_minus_2_pow_32;
        let state_1_hi_limbs_inv_is_inv_or_state_1_hi_limbs_is_zero =
            state_1_hi_limbs_are_not_all_1s.clone() * state_1_hi_limbs_minus_2_pow_32;
        let state_2_hi_limbs_inv_is_inv_or_state_2_hi_limbs_is_zero =
            state_2_hi_limbs_are_not_all_1s.clone() * state_2_hi_limbs_minus_2_pow_32;
        let state_3_hi_limbs_inv_is_inv_or_state_3_hi_limbs_is_zero =
            state_3_hi_limbs_are_not_all_1s.clone() * state_3_hi_limbs_minus_2_pow_32;

        // consistent decomposition into limbs
        let state_0_lo_limbs = main_row(Self::MainColumn::State0MidLowLkIn) * two_pow_16.clone()
            + main_row(Self::MainColumn::State0LowestLkIn);
        let state_1_lo_limbs = main_row(Self::MainColumn::State1MidLowLkIn) * two_pow_16.clone()
            + main_row(Self::MainColumn::State1LowestLkIn);
        let state_2_lo_limbs = main_row(Self::MainColumn::State2MidLowLkIn) * two_pow_16.clone()
            + main_row(Self::MainColumn::State2LowestLkIn);
        let state_3_lo_limbs = main_row(Self::MainColumn::State3MidLowLkIn) * two_pow_16
            + main_row(Self::MainColumn::State3LowestLkIn);

        let if_state_0_hi_limbs_are_all_1_then_state_0_lo_limbs_are_all_0 =
            state_0_hi_limbs_are_not_all_1s * state_0_lo_limbs;
        let if_state_1_hi_limbs_are_all_1_then_state_1_lo_limbs_are_all_0 =
            state_1_hi_limbs_are_not_all_1s * state_1_lo_limbs;
        let if_state_2_hi_limbs_are_all_1_then_state_2_lo_limbs_are_all_0 =
            state_2_hi_limbs_are_not_all_1s * state_2_lo_limbs;
        let if_state_3_hi_limbs_are_all_1_then_state_3_lo_limbs_are_all_0 =
            state_3_hi_limbs_are_not_all_1s * state_3_lo_limbs;

        let mut constraints = vec![
            mode_is_a_valid_mode,
            if_mode_is_not_sponge_then_ci_is_hash,
            if_mode_is_sponge_then_ci_is_a_sponge_instruction,
            if_padding_mode_then_round_number_is_0,
            if_ci_is_sponge_init_then_round_number_is_0,
            state_0_hi_limbs_inv_is_inv_or_is_zero,
            state_1_hi_limbs_inv_is_inv_or_is_zero,
            state_2_hi_limbs_inv_is_inv_or_is_zero,
            state_3_hi_limbs_inv_is_inv_or_is_zero,
            state_0_hi_limbs_inv_is_inv_or_state_0_hi_limbs_is_zero,
            state_1_hi_limbs_inv_is_inv_or_state_1_hi_limbs_is_zero,
            state_2_hi_limbs_inv_is_inv_or_state_2_hi_limbs_is_zero,
            state_3_hi_limbs_inv_is_inv_or_state_3_hi_limbs_is_zero,
            if_state_0_hi_limbs_are_all_1_then_state_0_lo_limbs_are_all_0,
            if_state_1_hi_limbs_are_all_1_then_state_1_lo_limbs_are_all_0,
            if_state_2_hi_limbs_are_all_1_then_state_2_lo_limbs_are_all_0,
            if_state_3_hi_limbs_are_all_1_then_state_3_lo_limbs_are_all_0,
        ];

        constraints.extend(if_ci_is_sponge_init_then_rate_is_0);
        constraints.extend(if_mode_is_hash_and_round_no_is_0_then_states_10_through_15_are_1);

        for round_constant_column_idx in 0..NUM_ROUND_CONSTANTS {
            let round_constant_column =
                Self::round_constant_column_by_index(round_constant_column_idx);
            let round_constant_column_circuit = main_row(round_constant_column);
            let mut round_constant_constraint_circuit = constant(0);
            for round_idx in 0..NUM_ROUNDS {
                let round_constants = Self::tip5_round_constants_by_round_number(round_idx);
                let round_constant = round_constants[round_constant_column_idx];
                let round_constant = circuit_builder.b_constant(round_constant);
                let round_deselector_circuit =
                    Self::round_number_deselector(circuit_builder, &round_number, round_idx);
                round_constant_constraint_circuit = round_constant_constraint_circuit
                    + round_deselector_circuit
                        * (round_constant_column_circuit.clone() - round_constant);
            }
            constraints.push(round_constant_constraint_circuit);
        }

        constraints
    }

    fn transition_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let opcode = |instruction: Instruction| circuit_builder.b_constant(instruction.opcode_b());
        let constant = |c: u64| circuit_builder.b_constant(c);

        let opcode_hash = opcode(Instruction::Hash);
        let opcode_sponge_init = opcode(Instruction::SpongeInit);
        let opcode_sponge_absorb = opcode(Instruction::SpongeAbsorb);
        let opcode_sponge_squeeze = opcode(Instruction::SpongeSqueeze);

        let current_main_row = |column_idx: Self::MainColumn| {
            circuit_builder.input(CurrentMain(column_idx.master_main_index()))
        };
        let next_main_row = |column_idx: Self::MainColumn| {
            circuit_builder.input(NextMain(column_idx.master_main_index()))
        };
        let current_aux_row = |column_idx: Self::AuxColumn| {
            circuit_builder.input(CurrentAux(column_idx.master_aux_index()))
        };
        let next_aux_row = |column_idx: Self::AuxColumn| {
            circuit_builder.input(NextAux(column_idx.master_aux_index()))
        };

        let running_evaluation_initial = circuit_builder.x_constant(EvalArg::default_initial());

        let prepare_chunk_indeterminate =
            challenge(ChallengeId::ProgramAttestationPrepareChunkIndeterminate);
        let receive_chunk_indeterminate =
            challenge(ChallengeId::ProgramAttestationSendChunkIndeterminate);
        let compress_program_digest_indeterminate =
            challenge(ChallengeId::CompressProgramDigestIndeterminate);
        let expected_program_digest = challenge(ChallengeId::CompressedProgramDigest);
        let hash_input_eval_indeterminate = challenge(ChallengeId::HashInputIndeterminate);
        let hash_digest_eval_indeterminate = challenge(ChallengeId::HashDigestIndeterminate);
        let sponge_indeterminate = challenge(ChallengeId::SpongeIndeterminate);

        let mode = current_main_row(Self::MainColumn::Mode);
        let ci = current_main_row(Self::MainColumn::CI);
        let round_number = current_main_row(Self::MainColumn::RoundNumber);
        let running_evaluation_receive_chunk =
            current_aux_row(Self::AuxColumn::ReceiveChunkRunningEvaluation);
        let running_evaluation_hash_input =
            current_aux_row(Self::AuxColumn::HashInputRunningEvaluation);
        let running_evaluation_hash_digest =
            current_aux_row(Self::AuxColumn::HashDigestRunningEvaluation);
        let running_evaluation_sponge = current_aux_row(Self::AuxColumn::SpongeRunningEvaluation);

        let mode_next = next_main_row(Self::MainColumn::Mode);
        let ci_next = next_main_row(Self::MainColumn::CI);
        let round_number_next = next_main_row(Self::MainColumn::RoundNumber);
        let running_evaluation_receive_chunk_next =
            next_aux_row(Self::AuxColumn::ReceiveChunkRunningEvaluation);
        let running_evaluation_hash_input_next =
            next_aux_row(Self::AuxColumn::HashInputRunningEvaluation);
        let running_evaluation_hash_digest_next =
            next_aux_row(Self::AuxColumn::HashDigestRunningEvaluation);
        let running_evaluation_sponge_next = next_aux_row(Self::AuxColumn::SpongeRunningEvaluation);

        let [state_0, state_1, state_2, state_3] =
            Self::re_compose_states_0_through_3_before_lookup(
                circuit_builder,
                Self::indicate_column_index_in_current_main_row,
            );

        let state_current = [
            state_0,
            state_1,
            state_2,
            state_3,
            current_main_row(Self::MainColumn::State4),
            current_main_row(Self::MainColumn::State5),
            current_main_row(Self::MainColumn::State6),
            current_main_row(Self::MainColumn::State7),
            current_main_row(Self::MainColumn::State8),
            current_main_row(Self::MainColumn::State9),
            current_main_row(Self::MainColumn::State10),
            current_main_row(Self::MainColumn::State11),
            current_main_row(Self::MainColumn::State12),
            current_main_row(Self::MainColumn::State13),
            current_main_row(Self::MainColumn::State14),
            current_main_row(Self::MainColumn::State15),
        ];

        let (state_next, hash_function_round_correctly_performs_update) =
            Self::tip5_constraints_as_circuits(circuit_builder);

        let state_weights = [
            ChallengeId::StackWeight0,
            ChallengeId::StackWeight1,
            ChallengeId::StackWeight2,
            ChallengeId::StackWeight3,
            ChallengeId::StackWeight4,
            ChallengeId::StackWeight5,
            ChallengeId::StackWeight6,
            ChallengeId::StackWeight7,
            ChallengeId::StackWeight8,
            ChallengeId::StackWeight9,
            ChallengeId::StackWeight10,
            ChallengeId::StackWeight11,
            ChallengeId::StackWeight12,
            ChallengeId::StackWeight13,
            ChallengeId::StackWeight14,
            ChallengeId::StackWeight15,
        ]
        .map(challenge);

        let round_number_is_not_num_rounds =
            Self::round_number_deselector(circuit_builder, &round_number, NUM_ROUNDS);

        let round_number_is_0_through_4_or_round_number_next_is_0 =
            round_number_is_not_num_rounds * round_number_next.clone();

        let next_mode_is_padding_mode_or_round_number_is_num_rounds_or_increments_by_one =
            Self::select_mode(circuit_builder, &mode_next, HashTableMode::Pad)
                * (ci.clone() - opcode_sponge_init.clone())
                * (round_number.clone() - constant(NUM_ROUNDS as u64))
                * (round_number_next.clone() - round_number.clone() - constant(1));

        // compress the digest by computing the terminal of an evaluation
        // argument
        let compressed_digest = state_current[..Digest::LEN].iter().fold(
            running_evaluation_initial.clone(),
            |acc, digest_element| {
                acc * compress_program_digest_indeterminate.clone() + digest_element.clone()
            },
        );
        let if_mode_changes_from_program_hashing_then_current_digest_is_expected_program_digest =
            Self::mode_deselector(circuit_builder, &mode, HashTableMode::ProgramHashing)
                * Self::select_mode(circuit_builder, &mode_next, HashTableMode::ProgramHashing)
                * (compressed_digest - expected_program_digest);

        let if_mode_is_program_hashing_and_next_mode_is_sponge_then_ci_next_is_sponge_init =
            Self::mode_deselector(circuit_builder, &mode, HashTableMode::ProgramHashing)
                * Self::mode_deselector(circuit_builder, &mode_next, HashTableMode::Sponge)
                * (ci_next.clone() - opcode_sponge_init.clone());

        let if_round_number_is_not_max_and_ci_is_not_sponge_init_then_ci_doesnt_change =
            (round_number.clone() - constant(NUM_ROUNDS as u64))
                * (ci.clone() - opcode_sponge_init.clone())
                * (ci_next.clone() - ci.clone());

        let if_round_number_is_not_max_and_ci_is_not_sponge_init_then_mode_doesnt_change =
            (round_number - constant(NUM_ROUNDS as u64))
                * (ci.clone() - opcode_sponge_init.clone())
                * (mode_next.clone() - mode.clone());

        let if_mode_is_sponge_then_mode_next_is_sponge_or_hash_or_pad =
            Self::mode_deselector(circuit_builder, &mode, HashTableMode::Sponge)
                * Self::select_mode(circuit_builder, &mode_next, HashTableMode::Sponge)
                * Self::select_mode(circuit_builder, &mode_next, HashTableMode::Hash)
                * Self::select_mode(circuit_builder, &mode_next, HashTableMode::Pad);

        let if_mode_is_hash_then_mode_next_is_hash_or_pad =
            Self::mode_deselector(circuit_builder, &mode, HashTableMode::Hash)
                * Self::select_mode(circuit_builder, &mode_next, HashTableMode::Hash)
                * Self::select_mode(circuit_builder, &mode_next, HashTableMode::Pad);

        let if_mode_is_pad_then_mode_next_is_pad =
            Self::mode_deselector(circuit_builder, &mode, HashTableMode::Pad)
                * Self::select_mode(circuit_builder, &mode_next, HashTableMode::Pad);

        let difference_of_capacity_registers = state_current[tip5::RATE..]
            .iter()
            .zip_eq(state_next[tip5::RATE..].iter())
            .map(|(current, next)| next.clone() - current.clone())
            .collect_vec();
        let randomized_sum_of_capacity_differences = state_weights[tip5::RATE..]
            .iter()
            .zip_eq(difference_of_capacity_registers)
            .map(|(weight, state_difference)| weight.clone() * state_difference)
            .sum::<ConstraintCircuitMonad<_>>();

        let capacity_doesnt_change_at_section_start_when_program_hashing_or_absorbing =
            Self::round_number_deselector(circuit_builder, &round_number_next, 0)
                * Self::select_mode(circuit_builder, &mode_next, HashTableMode::Hash)
                * Self::select_mode(circuit_builder, &mode_next, HashTableMode::Pad)
                * (ci_next.clone() - opcode_sponge_init.clone())
                * randomized_sum_of_capacity_differences.clone();

        let difference_of_state_registers = state_current
            .iter()
            .zip_eq(state_next.iter())
            .map(|(current, next)| next.clone() - current.clone())
            .collect_vec();
        let randomized_sum_of_state_differences = state_weights
            .iter()
            .zip_eq(difference_of_state_registers.iter())
            .map(|(weight, state_difference)| weight.clone() * state_difference.clone())
            .sum();
        let if_round_number_next_is_0_and_ci_next_is_squeeze_then_state_doesnt_change =
            Self::round_number_deselector(circuit_builder, &round_number_next, 0)
                * Self::instruction_deselector(
                    circuit_builder,
                    &ci_next,
                    Instruction::SpongeSqueeze,
                )
                * randomized_sum_of_state_differences;

        // Evaluation Arguments

        // If (and only if) the row number in the next row is 0 and the mode in
        // the next row is `hash`, update running evaluation “hash input.”
        let running_evaluation_hash_input_remains =
            running_evaluation_hash_input_next.clone() - running_evaluation_hash_input.clone();
        let tip5_input = state_next[..tip5::RATE].to_owned();
        let compressed_row_from_processor = tip5_input
            .into_iter()
            .zip_eq(state_weights[..tip5::RATE].iter())
            .map(|(state, weight)| weight.clone() * state)
            .sum();

        let running_evaluation_hash_input_updates = running_evaluation_hash_input_next
            - hash_input_eval_indeterminate * running_evaluation_hash_input
            - compressed_row_from_processor;
        let running_evaluation_hash_input_is_updated_correctly =
            Self::round_number_deselector(circuit_builder, &round_number_next, 0)
                * Self::mode_deselector(circuit_builder, &mode_next, HashTableMode::Hash)
                * running_evaluation_hash_input_updates
                + round_number_next.clone() * running_evaluation_hash_input_remains.clone()
                + Self::select_mode(circuit_builder, &mode_next, HashTableMode::Hash)
                    * running_evaluation_hash_input_remains;

        // If (and only if) the row number in the next row is NUM_ROUNDS and the
        // current instruction in the next row corresponds to `hash`, update
        // running evaluation “hash digest.”
        let round_number_next_is_num_rounds =
            round_number_next.clone() - constant(NUM_ROUNDS as u64);
        let running_evaluation_hash_digest_remains =
            running_evaluation_hash_digest_next.clone() - running_evaluation_hash_digest.clone();
        let hash_digest = state_next[..Digest::LEN].to_owned();
        let compressed_row_hash_digest = hash_digest
            .into_iter()
            .zip_eq(state_weights[..Digest::LEN].iter())
            .map(|(state, weight)| weight.clone() * state)
            .sum();
        let running_evaluation_hash_digest_updates = running_evaluation_hash_digest_next
            - hash_digest_eval_indeterminate * running_evaluation_hash_digest
            - compressed_row_hash_digest;
        let running_evaluation_hash_digest_is_updated_correctly =
            Self::round_number_deselector(circuit_builder, &round_number_next, NUM_ROUNDS)
                * Self::mode_deselector(circuit_builder, &mode_next, HashTableMode::Hash)
                * running_evaluation_hash_digest_updates
                + round_number_next_is_num_rounds * running_evaluation_hash_digest_remains.clone()
                + Self::select_mode(circuit_builder, &mode_next, HashTableMode::Hash)
                    * running_evaluation_hash_digest_remains;

        // The running evaluation for “Sponge” updates correctly.
        let compressed_row_next = state_weights[..tip5::RATE]
            .iter()
            .zip_eq(state_next[..tip5::RATE].iter())
            .map(|(weight, st_next)| weight.clone() * st_next.clone())
            .sum();
        let running_evaluation_sponge_has_accumulated_ci = running_evaluation_sponge_next.clone()
            - sponge_indeterminate * running_evaluation_sponge.clone()
            - challenge(ChallengeId::HashCIWeight) * ci_next.clone();
        let running_evaluation_sponge_has_accumulated_next_row =
            running_evaluation_sponge_has_accumulated_ci.clone() - compressed_row_next;
        let if_round_no_next_0_and_ci_next_is_spongy_then_running_evaluation_sponge_updates =
            Self::round_number_deselector(circuit_builder, &round_number_next, 0)
                * (ci_next.clone() - opcode_hash)
                * running_evaluation_sponge_has_accumulated_next_row;

        let running_evaluation_sponge_remains =
            running_evaluation_sponge_next - running_evaluation_sponge;
        let if_round_no_next_is_not_0_then_running_evaluation_sponge_remains =
            round_number_next.clone() * running_evaluation_sponge_remains.clone();
        let if_ci_next_is_not_spongy_then_running_evaluation_sponge_remains = (ci_next.clone()
            - opcode_sponge_init)
            * (ci_next.clone() - opcode_sponge_absorb)
            * (ci_next - opcode_sponge_squeeze)
            * running_evaluation_sponge_remains;
        let running_evaluation_sponge_is_updated_correctly =
            if_round_no_next_0_and_ci_next_is_spongy_then_running_evaluation_sponge_updates
                + if_round_no_next_is_not_0_then_running_evaluation_sponge_remains
                + if_ci_next_is_not_spongy_then_running_evaluation_sponge_remains;

        // program attestation: absorb RATE instructions if in the right mode on
        // the right row
        let compressed_chunk = state_next[..tip5::RATE]
            .iter()
            .fold(running_evaluation_initial, |acc, rate_element| {
                acc * prepare_chunk_indeterminate.clone() + rate_element.clone()
            });
        let receive_chunk_running_evaluation_absorbs_chunk_of_instructions =
            running_evaluation_receive_chunk_next.clone()
                - receive_chunk_indeterminate * running_evaluation_receive_chunk.clone()
                - compressed_chunk;
        let receive_chunk_running_evaluation_remains =
            running_evaluation_receive_chunk_next - running_evaluation_receive_chunk;
        let receive_chunk_of_instructions_iff_next_mode_is_prog_hashing_and_next_round_number_is_0 =
            Self::round_number_deselector(circuit_builder, &round_number_next, 0)
                * Self::mode_deselector(circuit_builder, &mode_next, HashTableMode::ProgramHashing)
                * receive_chunk_running_evaluation_absorbs_chunk_of_instructions
                + round_number_next * receive_chunk_running_evaluation_remains.clone()
                + Self::select_mode(circuit_builder, &mode_next, HashTableMode::ProgramHashing)
                    * receive_chunk_running_evaluation_remains;

        let constraints = vec![
            round_number_is_0_through_4_or_round_number_next_is_0,
            next_mode_is_padding_mode_or_round_number_is_num_rounds_or_increments_by_one,
            receive_chunk_of_instructions_iff_next_mode_is_prog_hashing_and_next_round_number_is_0,
            if_mode_changes_from_program_hashing_then_current_digest_is_expected_program_digest,
            if_mode_is_program_hashing_and_next_mode_is_sponge_then_ci_next_is_sponge_init,
            if_round_number_is_not_max_and_ci_is_not_sponge_init_then_ci_doesnt_change,
            if_round_number_is_not_max_and_ci_is_not_sponge_init_then_mode_doesnt_change,
            if_mode_is_sponge_then_mode_next_is_sponge_or_hash_or_pad,
            if_mode_is_hash_then_mode_next_is_hash_or_pad,
            if_mode_is_pad_then_mode_next_is_pad,
            capacity_doesnt_change_at_section_start_when_program_hashing_or_absorbing,
            if_round_number_next_is_0_and_ci_next_is_squeeze_then_state_doesnt_change,
            running_evaluation_hash_input_is_updated_correctly,
            running_evaluation_hash_digest_is_updated_correctly,
            running_evaluation_sponge_is_updated_correctly,
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                Self::MainColumn::State0HighestLkIn,
                Self::MainColumn::State0HighestLkOut,
                Self::AuxColumn::CascadeState0HighestClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                Self::MainColumn::State0MidHighLkIn,
                Self::MainColumn::State0MidHighLkOut,
                Self::AuxColumn::CascadeState0MidHighClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                Self::MainColumn::State0MidLowLkIn,
                Self::MainColumn::State0MidLowLkOut,
                Self::AuxColumn::CascadeState0MidLowClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                Self::MainColumn::State0LowestLkIn,
                Self::MainColumn::State0LowestLkOut,
                Self::AuxColumn::CascadeState0LowestClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                Self::MainColumn::State1HighestLkIn,
                Self::MainColumn::State1HighestLkOut,
                Self::AuxColumn::CascadeState1HighestClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                Self::MainColumn::State1MidHighLkIn,
                Self::MainColumn::State1MidHighLkOut,
                Self::AuxColumn::CascadeState1MidHighClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                Self::MainColumn::State1MidLowLkIn,
                Self::MainColumn::State1MidLowLkOut,
                Self::AuxColumn::CascadeState1MidLowClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                Self::MainColumn::State1LowestLkIn,
                Self::MainColumn::State1LowestLkOut,
                Self::AuxColumn::CascadeState1LowestClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                Self::MainColumn::State2HighestLkIn,
                Self::MainColumn::State2HighestLkOut,
                Self::AuxColumn::CascadeState2HighestClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                Self::MainColumn::State2MidHighLkIn,
                Self::MainColumn::State2MidHighLkOut,
                Self::AuxColumn::CascadeState2MidHighClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                Self::MainColumn::State2MidLowLkIn,
                Self::MainColumn::State2MidLowLkOut,
                Self::AuxColumn::CascadeState2MidLowClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                Self::MainColumn::State2LowestLkIn,
                Self::MainColumn::State2LowestLkOut,
                Self::AuxColumn::CascadeState2LowestClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                Self::MainColumn::State3HighestLkIn,
                Self::MainColumn::State3HighestLkOut,
                Self::AuxColumn::CascadeState3HighestClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                Self::MainColumn::State3MidHighLkIn,
                Self::MainColumn::State3MidHighLkOut,
                Self::AuxColumn::CascadeState3MidHighClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                Self::MainColumn::State3MidLowLkIn,
                Self::MainColumn::State3MidLowLkOut,
                Self::AuxColumn::CascadeState3MidLowClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                Self::MainColumn::State3LowestLkIn,
                Self::MainColumn::State3LowestLkOut,
                Self::AuxColumn::CascadeState3LowestClientLogDerivative,
            ),
        ];

        [
            constraints,
            hash_function_round_correctly_performs_update.to_vec(),
        ]
        .concat()
    }

    fn terminal_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let opcode = |instruction: Instruction| circuit_builder.b_constant(instruction.opcode_b());
        let constant = |c: u64| circuit_builder.b_constant(c);
        let main_row = |column_idx: Self::MainColumn| {
            circuit_builder.input(Main(column_idx.master_main_index()))
        };

        let mode = main_row(Self::MainColumn::Mode);
        let round_number = main_row(Self::MainColumn::RoundNumber);

        let compress_program_digest_indeterminate =
            challenge(ChallengeId::CompressProgramDigestIndeterminate);
        let expected_program_digest = challenge(ChallengeId::CompressedProgramDigest);

        let max_round_number = constant(NUM_ROUNDS as u64);

        let [state_0, state_1, state_2, state_3] =
            Self::re_compose_states_0_through_3_before_lookup(
                circuit_builder,
                Self::indicate_column_index_in_main_row,
            );
        let state_4 = main_row(Self::MainColumn::State4);
        let program_digest = [state_0, state_1, state_2, state_3, state_4];
        let compressed_digest = program_digest.into_iter().fold(
            circuit_builder.x_constant(EvalArg::default_initial()),
            |acc, digest_element| {
                acc * compress_program_digest_indeterminate.clone() + digest_element
            },
        );
        let if_mode_is_program_hashing_then_current_digest_is_expected_program_digest =
            Self::mode_deselector(circuit_builder, &mode, HashTableMode::ProgramHashing)
                * (compressed_digest - expected_program_digest);

        let if_mode_is_not_pad_and_ci_is_not_sponge_init_then_round_number_is_max_round_number =
            Self::select_mode(circuit_builder, &mode, HashTableMode::Pad)
                * (main_row(Self::MainColumn::CI) - opcode(Instruction::SpongeInit))
                * (round_number - max_round_number);

        vec![
            if_mode_is_program_hashing_then_current_digest_is_expected_program_digest,
            if_mode_is_not_pad_and_ci_is_not_sponge_init_then_round_number_is_max_round_number,
        ]
    }
}

/// The current “mode” of the Hash Table. The Hash Table can be in one of four
/// distinct modes:
///
/// 1. Hashing the [`Program`][program]. This is part of program attestation.
/// 1. Processing all Sponge instructions, _i.e._, `sponge_init`,
///    `sponge_absorb`, `sponge_absorb_mem`, and `sponge_squeeze`.
/// 1. Processing the `hash` instruction.
/// 1. Padding mode.
///
/// Changing the mode is only possible when the current
/// [`RoundNumber`][round_no] is [`NUM_ROUNDS`]. The mode evolves as
/// [`ProgramHashing`][prog_hash] → [`Sponge`][sponge] → [`Hash`][hash] →
/// [`Pad`][pad]. Once mode [`Pad`][pad] is reached, it is not possible to
/// change the mode anymore. Skipping any or all of the modes
/// [`Sponge`][sponge], [`Hash`][hash], or [`Pad`][pad] is possible in
/// principle:
/// - if no Sponge instructions are executed, mode [`Sponge`][sponge] will be
///   skipped,
/// - if no `hash` instruction is executed, mode [`Hash`][hash] will be skipped,
///   and
/// - if the Hash Table does not require any padding, mode [`Pad`][pad] will be
///   skipped.
///
/// It is not possible to skip mode [`ProgramHashing`][prog_hash]:
/// the [`Program`][program] is always hashed.
/// The empty program is not valid since any valid [`Program`][program] must
/// execute instruction `halt`.
///
/// [round_no]: crate::table_column::HashMainColumn::RoundNumber
/// [program]: isa::program::Program
/// [prog_hash]: HashTableMode::ProgramHashing
/// [sponge]: HashTableMode::Sponge
/// [hash]: type@HashTableMode::Hash
/// [pad]: HashTableMode::Pad
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum HashTableMode {
    /// The mode in which the [`Program`][program] is hashed. This is part of
    /// program attestation.
    ///
    /// [program]: isa::program::Program
    ProgramHashing,

    /// The mode in which Sponge instructions, _i.e._, `sponge_init`,
    /// `sponge_absorb`, `sponge_absorb_mem`, and `sponge_squeeze`, are
    /// processed.
    Sponge,

    /// The mode in which the `hash` instruction is processed.
    Hash,

    /// Indicator for padding rows.
    Pad,
}

impl From<HashTableMode> for u32 {
    fn from(mode: HashTableMode) -> Self {
        match mode {
            HashTableMode::ProgramHashing => 1,
            HashTableMode::Sponge => 2,
            HashTableMode::Hash => 3,
            HashTableMode::Pad => 0,
        }
    }
}

impl From<HashTableMode> for u64 {
    fn from(mode: HashTableMode) -> Self {
        let discriminant: u32 = mode.into();
        discriminant.into()
    }
}

impl From<HashTableMode> for BFieldElement {
    fn from(mode: HashTableMode) -> Self {
        let discriminant: u32 = mode.into();
        discriminant.into()
    }
}
