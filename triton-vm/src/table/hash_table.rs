use constraint_builder::ConstraintCircuitBuilder;
use constraint_builder::ConstraintCircuitMonad;
use constraint_builder::DualRowIndicator;
use constraint_builder::DualRowIndicator::*;
use constraint_builder::InputIndicator;
use constraint_builder::SingleRowIndicator;
use constraint_builder::SingleRowIndicator::*;
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
use twenty_first::prelude::tip5::MDS_MATRIX_FIRST_COLUMN;
use twenty_first::prelude::tip5::NUM_ROUNDS;
use twenty_first::prelude::tip5::NUM_SPLIT_AND_LOOKUP;
use twenty_first::prelude::tip5::RATE;
use twenty_first::prelude::tip5::ROUND_CONSTANTS;
use twenty_first::prelude::tip5::STATE_SIZE;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::profiler::profiler;
use crate::table::cascade_table::CascadeTable;
use crate::table::challenges::ChallengeId::*;
use crate::table::challenges::Challenges;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::EvalArg;
use crate::table::cross_table_argument::LookupArg;
use crate::table::table_column::HashBaseTableColumn;
use crate::table::table_column::HashBaseTableColumn::*;
use crate::table::table_column::HashExtTableColumn;
use crate::table::table_column::HashExtTableColumn::*;
use crate::table::table_column::MasterBaseTableColumn;
use crate::table::table_column::MasterExtTableColumn;

pub const BASE_WIDTH: usize = HashBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = HashExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

/// See [`HashTable::base_field_element_into_16_bit_limbs`] for more details.
const MONTGOMERY_MODULUS: BFieldElement =
    BFieldElement::new(((1_u128 << 64) % BFieldElement::P as u128) as u64);

pub const POWER_MAP_EXPONENT: u64 = 7;
pub const NUM_ROUND_CONSTANTS: usize = STATE_SIZE;

pub(crate) const PERMUTATION_TRACE_LENGTH: usize = NUM_ROUNDS + 1;

pub type PermutationTrace = [[BFieldElement; STATE_SIZE]; PERMUTATION_TRACE_LENGTH];

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct HashTable;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ExtHashTable;

/// The current “mode” of the Hash Table. The Hash Table can be in one of four distinct modes:
///
/// 1. Hashing the [`Program`][program]. This is part of program attestation.
/// 1. Processing all Sponge instructions, _i.e._, `sponge_init`,
///     `sponge_absorb`, `sponge_absorb_mem`, and `sponge_squeeze`.
/// 1. Processing the `hash` instruction.
/// 1. Padding mode.
///
/// Changing the mode is only possible when the current [`RoundNumber`] is [`NUM_ROUNDS`].
/// The mode evolves as
/// [`ProgramHashing`][prog_hash] → [`Sponge`][sponge] → [`Hash`][hash] → [`Pad`][pad].
/// Once mode [`Pad`][pad] is reached, it is not possible to change the mode anymore.
/// Skipping any or all of the modes [`Sponge`][sponge], [`Hash`][hash], or [`Pad`][pad]
/// is possible in principle:
/// - if no Sponge instructions are executed, mode [`Sponge`][sponge] will be skipped,
/// - if no `hash` instruction is executed, mode [`Hash`][hash] will be skipped, and
/// - if the Hash Table does not require any padding, mode [`Pad`][pad] will be skipped.
///
/// It is not possible to skip mode [`ProgramHashing`][prog_hash]:
/// the [`Program`][program] is always hashed.
/// The empty program is not valid since any valid [`Program`][program] must execute
/// instruction `halt`.
///
/// [program]: isa::program::Program
/// [prog_hash]: HashTableMode::ProgramHashing
/// [sponge]: HashTableMode::Sponge
/// [hash]: type@HashTableMode::Hash
/// [pad]: HashTableMode::Pad
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum HashTableMode {
    /// The mode in which the [`Program`][program] is hashed. This is part of program attestation.
    ///
    /// [program]: isa::program::Program
    ProgramHashing,

    /// The mode in which Sponge instructions, _i.e._, `sponge_init`,
    /// `sponge_absorb`, `sponge_absorb_mem`, and `sponge_squeeze`, are processed.
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

impl ExtHashTable {
    /// Construct one of the states 0 through 3 from its constituent limbs.
    /// For example, state 0 (prior to it being looked up in the split-and-lookup S-Box, which is
    /// usually the desired version of the state) is constructed from limbs
    /// [`State0HighestLkIn`] through [`State0LowestLkIn`].
    ///
    /// States 4 through 15 are directly accessible. See also the slightly related
    /// [`Self::state_column_by_index`].
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

    pub fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let constant = |c: u64| circuit_builder.b_constant(c);

        let base_row = |column: HashBaseTableColumn| {
            circuit_builder.input(BaseRow(column.master_base_table_index()))
        };
        let ext_row = |column: HashExtTableColumn| {
            circuit_builder.input(ExtRow(column.master_ext_table_index()))
        };

        let running_evaluation_initial = circuit_builder.x_constant(EvalArg::default_initial());
        let lookup_arg_default_initial = circuit_builder.x_constant(LookupArg::default_initial());

        let mode = base_row(Mode);
        let running_evaluation_hash_input = ext_row(HashInputRunningEvaluation);
        let running_evaluation_hash_digest = ext_row(HashDigestRunningEvaluation);
        let running_evaluation_sponge = ext_row(SpongeRunningEvaluation);
        let running_evaluation_receive_chunk = ext_row(ReceiveChunkRunningEvaluation);

        let cascade_indeterminate = challenge(HashCascadeLookupIndeterminate);
        let look_in_weight = challenge(HashCascadeLookInWeight);
        let look_out_weight = challenge(HashCascadeLookOutWeight);
        let prepare_chunk_indeterminate = challenge(ProgramAttestationPrepareChunkIndeterminate);
        let receive_chunk_indeterminate = challenge(ProgramAttestationSendChunkIndeterminate);

        // First chunk of the program is received correctly. Relates to program attestation.
        let [state_0, state_1, state_2, state_3] =
            Self::re_compose_states_0_through_3_before_lookup(
                circuit_builder,
                Self::indicate_column_index_in_base_row,
            );
        let state_rate_part: [_; RATE] = [
            state_0,
            state_1,
            state_2,
            state_3,
            base_row(State4),
            base_row(State5),
            base_row(State6),
            base_row(State7),
            base_row(State8),
            base_row(State9),
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

        // The lookup arguments with the Cascade Table for the S-Boxes are initialized correctly.
        let cascade_log_derivative_init_circuit =
            |look_in_column, look_out_column, cascade_log_derivative_column| {
                let look_in = base_row(look_in_column);
                let look_out = base_row(look_out_column);
                let compressed_row =
                    look_in_weight.clone() * look_in + look_out_weight.clone() * look_out;
                let cascade_log_derivative = ext_row(cascade_log_derivative_column);
                (cascade_log_derivative - lookup_arg_default_initial.clone())
                    * (cascade_indeterminate.clone() - compressed_row)
                    - constant(1)
            };

        // miscellaneous initial constraints
        let mode_is_program_hashing =
            Self::select_mode(circuit_builder, &mode, HashTableMode::ProgramHashing);
        let round_number_is_0 = base_row(RoundNumber);
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
                State0HighestLkIn,
                State0HighestLkOut,
                CascadeState0HighestClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                State0MidHighLkIn,
                State0MidHighLkOut,
                CascadeState0MidHighClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                State0MidLowLkIn,
                State0MidLowLkOut,
                CascadeState0MidLowClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                State0LowestLkIn,
                State0LowestLkOut,
                CascadeState0LowestClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                State1HighestLkIn,
                State1HighestLkOut,
                CascadeState1HighestClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                State1MidHighLkIn,
                State1MidHighLkOut,
                CascadeState1MidHighClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                State1MidLowLkIn,
                State1MidLowLkOut,
                CascadeState1MidLowClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                State1LowestLkIn,
                State1LowestLkOut,
                CascadeState1LowestClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                State2HighestLkIn,
                State2HighestLkOut,
                CascadeState2HighestClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                State2MidHighLkIn,
                State2MidHighLkOut,
                CascadeState2MidHighClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                State2MidLowLkIn,
                State2MidLowLkOut,
                CascadeState2MidLowClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                State2LowestLkIn,
                State2LowestLkOut,
                CascadeState2LowestClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                State3HighestLkIn,
                State3HighestLkOut,
                CascadeState3HighestClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                State3MidHighLkIn,
                State3MidHighLkOut,
                CascadeState3MidHighClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                State3MidLowLkIn,
                State3MidLowLkOut,
                CascadeState3MidLowClientLogDerivative,
            ),
            cascade_log_derivative_init_circuit(
                State3LowestLkIn,
                State3LowestLkOut,
                CascadeState3LowestClientLogDerivative,
            ),
        ]
    }

    /// A constraint circuit evaluating to zero if and only if the given
    /// `round_number_circuit_node` is not equal to the given `round_number_to_deselect`.
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

    /// A constraint circuit evaluating to zero if and only if the given `mode_circuit_node` is
    /// equal to the given `mode_to_select`.
    fn select_mode<II: InputIndicator>(
        circuit_builder: &ConstraintCircuitBuilder<II>,
        mode_circuit_node: &ConstraintCircuitMonad<II>,
        mode_to_select: HashTableMode,
    ) -> ConstraintCircuitMonad<II> {
        mode_circuit_node.clone() - circuit_builder.b_constant(mode_to_select)
    }

    /// A constraint circuit evaluating to zero if and only if the given `mode_circuit_node` is
    /// not equal to the given `mode_to_deselect`.
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
        let relevant_instructions = [Hash, SpongeInit, SpongeAbsorb, SpongeSqueeze];
        assert!(relevant_instructions.contains(&instruction_to_deselect));

        relevant_instructions
            .iter()
            .filter(|&instruction| instruction != &instruction_to_deselect)
            .map(|&instruction| current_instruction_node.clone() - opcode(instruction))
            .fold(constant(1), |accumulator, factor| accumulator * factor)
    }

    pub fn consistency_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let opcode = |instruction: Instruction| circuit_builder.b_constant(instruction.opcode_b());
        let constant = |c: u64| circuit_builder.b_constant(c);
        let base_row = |column_id: HashBaseTableColumn| {
            circuit_builder.input(BaseRow(column_id.master_base_table_index()))
        };

        let mode = base_row(Mode);
        let ci = base_row(CI);
        let round_number = base_row(RoundNumber);

        let ci_is_hash = ci.clone() - opcode(Hash);
        let ci_is_sponge_init = ci.clone() - opcode(SpongeInit);
        let ci_is_sponge_absorb = ci.clone() - opcode(SpongeAbsorb);
        let ci_is_sponge_squeeze = ci - opcode(SpongeSqueeze);

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
            let state_element = base_row(Self::state_column_by_index(state_index));
            if_ci_is_sponge_init_then_.clone() * state_element
        });

        let if_mode_is_hash_and_round_no_is_0_then_ = round_number_is_not_0 * mode_is_not_hash;
        let if_mode_is_hash_and_round_no_is_0_then_states_10_through_15_are_1 =
            (10..=15).map(|state_index| {
                let state_element = base_row(Self::state_column_by_index(state_index));
                if_mode_is_hash_and_round_no_is_0_then_.clone() * (state_element - constant(1))
            });

        // consistency of the inverse of the highest 2 limbs minus 2^32 - 1
        let one = constant(1);
        let two_pow_16 = constant(1 << 16);
        let two_pow_32 = constant(1 << 32);
        let state_0_hi_limbs_minus_2_pow_32 = two_pow_32.clone()
            - one.clone()
            - base_row(State0HighestLkIn) * two_pow_16.clone()
            - base_row(State0MidHighLkIn);
        let state_1_hi_limbs_minus_2_pow_32 = two_pow_32.clone()
            - one.clone()
            - base_row(State1HighestLkIn) * two_pow_16.clone()
            - base_row(State1MidHighLkIn);
        let state_2_hi_limbs_minus_2_pow_32 = two_pow_32.clone()
            - one.clone()
            - base_row(State2HighestLkIn) * two_pow_16.clone()
            - base_row(State2MidHighLkIn);
        let state_3_hi_limbs_minus_2_pow_32 = two_pow_32
            - one.clone()
            - base_row(State3HighestLkIn) * two_pow_16.clone()
            - base_row(State3MidHighLkIn);

        let state_0_hi_limbs_inv = base_row(State0Inv);
        let state_1_hi_limbs_inv = base_row(State1Inv);
        let state_2_hi_limbs_inv = base_row(State2Inv);
        let state_3_hi_limbs_inv = base_row(State3Inv);

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
        let state_0_lo_limbs =
            base_row(State0MidLowLkIn) * two_pow_16.clone() + base_row(State0LowestLkIn);
        let state_1_lo_limbs =
            base_row(State1MidLowLkIn) * two_pow_16.clone() + base_row(State1LowestLkIn);
        let state_2_lo_limbs =
            base_row(State2MidLowLkIn) * two_pow_16.clone() + base_row(State2LowestLkIn);
        let state_3_lo_limbs = base_row(State3MidLowLkIn) * two_pow_16 + base_row(State3LowestLkIn);

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
            let round_constant_column_circuit = base_row(round_constant_column);
            let mut round_constant_constraint_circuit = constant(0);
            for round_idx in 0..NUM_ROUNDS {
                let round_constants = HashTable::tip5_round_constants_by_round_number(round_idx);
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

    /// The [`HashBaseTableColumn`] for the round constant corresponding to the given index.
    /// Valid indices are 0 through 15, corresponding to the 16 round constants
    /// [`Constant0`] through [`Constant15`].
    fn round_constant_column_by_index(index: usize) -> HashBaseTableColumn {
        match index {
            0 => Constant0,
            1 => Constant1,
            2 => Constant2,
            3 => Constant3,
            4 => Constant4,
            5 => Constant5,
            6 => Constant6,
            7 => Constant7,
            8 => Constant8,
            9 => Constant9,
            10 => Constant10,
            11 => Constant11,
            12 => Constant12,
            13 => Constant13,
            14 => Constant14,
            15 => Constant15,
            _ => panic!("invalid constant column index"),
        }
    }

    /// The [`HashBaseTableColumn`] for the state corresponding to the given index.
    /// Valid indices are 4 through 15, corresponding to the 12 state columns
    /// [`State4`] through [`State15`].
    ///
    /// States with indices 0 through 3 have to be assembled from the respective limbs;
    /// see [`Self::re_compose_states_0_through_3_before_lookup`]
    /// or [`Self::re_compose_16_bit_limbs`].
    fn state_column_by_index(index: usize) -> HashBaseTableColumn {
        match index {
            4 => State4,
            5 => State5,
            6 => State6,
            7 => State7,
            8 => State8,
            9 => State9,
            10 => State10,
            11 => State11,
            12 => State12,
            13 => State13,
            14 => State14,
            15 => State15,
            _ => panic!("invalid state column index"),
        }
    }

    pub fn transition_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let opcode = |instruction: Instruction| circuit_builder.b_constant(instruction.opcode_b());
        let constant = |c: u64| circuit_builder.b_constant(c);

        let opcode_hash = opcode(Hash);
        let opcode_sponge_init = opcode(SpongeInit);
        let opcode_sponge_absorb = opcode(SpongeAbsorb);
        let opcode_sponge_squeeze = opcode(SpongeSqueeze);

        let current_base_row = |column_idx: HashBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(column_idx.master_base_table_index()))
        };
        let next_base_row = |column_idx: HashBaseTableColumn| {
            circuit_builder.input(NextBaseRow(column_idx.master_base_table_index()))
        };
        let current_ext_row = |column_idx: HashExtTableColumn| {
            circuit_builder.input(CurrentExtRow(column_idx.master_ext_table_index()))
        };
        let next_ext_row = |column_idx: HashExtTableColumn| {
            circuit_builder.input(NextExtRow(column_idx.master_ext_table_index()))
        };

        let running_evaluation_initial = circuit_builder.x_constant(EvalArg::default_initial());

        let prepare_chunk_indeterminate = challenge(ProgramAttestationPrepareChunkIndeterminate);
        let receive_chunk_indeterminate = challenge(ProgramAttestationSendChunkIndeterminate);
        let compress_program_digest_indeterminate = challenge(CompressProgramDigestIndeterminate);
        let expected_program_digest = challenge(CompressedProgramDigest);
        let hash_input_eval_indeterminate = challenge(HashInputIndeterminate);
        let hash_digest_eval_indeterminate = challenge(HashDigestIndeterminate);
        let sponge_indeterminate = challenge(SpongeIndeterminate);

        let mode = current_base_row(Mode);
        let ci = current_base_row(CI);
        let round_number = current_base_row(RoundNumber);
        let running_evaluation_receive_chunk = current_ext_row(ReceiveChunkRunningEvaluation);
        let running_evaluation_hash_input = current_ext_row(HashInputRunningEvaluation);
        let running_evaluation_hash_digest = current_ext_row(HashDigestRunningEvaluation);
        let running_evaluation_sponge = current_ext_row(SpongeRunningEvaluation);

        let mode_next = next_base_row(Mode);
        let ci_next = next_base_row(CI);
        let round_number_next = next_base_row(RoundNumber);
        let running_evaluation_receive_chunk_next = next_ext_row(ReceiveChunkRunningEvaluation);
        let running_evaluation_hash_input_next = next_ext_row(HashInputRunningEvaluation);
        let running_evaluation_hash_digest_next = next_ext_row(HashDigestRunningEvaluation);
        let running_evaluation_sponge_next = next_ext_row(SpongeRunningEvaluation);

        let [state_0, state_1, state_2, state_3] =
            Self::re_compose_states_0_through_3_before_lookup(
                circuit_builder,
                Self::indicate_column_index_in_current_base_row,
            );

        let state_current = [
            state_0,
            state_1,
            state_2,
            state_3,
            current_base_row(State4),
            current_base_row(State5),
            current_base_row(State6),
            current_base_row(State7),
            current_base_row(State8),
            current_base_row(State9),
            current_base_row(State10),
            current_base_row(State11),
            current_base_row(State12),
            current_base_row(State13),
            current_base_row(State14),
            current_base_row(State15),
        ];

        let (state_next, hash_function_round_correctly_performs_update) =
            Self::tip5_constraints_as_circuits(circuit_builder);

        let state_weights = [
            StackWeight0,
            StackWeight1,
            StackWeight2,
            StackWeight3,
            StackWeight4,
            StackWeight5,
            StackWeight6,
            StackWeight7,
            StackWeight8,
            StackWeight9,
            StackWeight10,
            StackWeight11,
            StackWeight12,
            StackWeight13,
            StackWeight14,
            StackWeight15,
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

        // compress the digest by computing the terminal of an evaluation argument
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

        let difference_of_capacity_registers = state_current[RATE..]
            .iter()
            .zip_eq(state_next[RATE..].iter())
            .map(|(current, next)| next.clone() - current.clone())
            .collect_vec();
        let randomized_sum_of_capacity_differences = state_weights[RATE..]
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
                * Self::instruction_deselector(circuit_builder, &ci_next, SpongeSqueeze)
                * randomized_sum_of_state_differences;

        // Evaluation Arguments

        // If (and only if) the row number in the next row is 0 and the mode in the next row is
        // `hash`, update running evaluation “hash input.”
        let running_evaluation_hash_input_remains =
            running_evaluation_hash_input_next.clone() - running_evaluation_hash_input.clone();
        let tip5_input = state_next[..RATE].to_owned();
        let compressed_row_from_processor = tip5_input
            .into_iter()
            .zip_eq(state_weights[..RATE].iter())
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

        // If (and only if) the row number in the next row is NUM_ROUNDS and the current instruction
        // in the next row corresponds to `hash`, update running evaluation “hash digest.”
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
        let compressed_row_next = state_weights[..RATE]
            .iter()
            .zip_eq(state_next[..RATE].iter())
            .map(|(weight, st_next)| weight.clone() * st_next.clone())
            .sum();
        let running_evaluation_sponge_has_accumulated_ci = running_evaluation_sponge_next.clone()
            - sponge_indeterminate * running_evaluation_sponge.clone()
            - challenge(HashCIWeight) * ci_next.clone();
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

        // program attestation: absorb RATE instructions if in the right mode on the right row
        let compressed_chunk = state_next[..RATE]
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
                State0HighestLkIn,
                State0HighestLkOut,
                CascadeState0HighestClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                State0MidHighLkIn,
                State0MidHighLkOut,
                CascadeState0MidHighClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                State0MidLowLkIn,
                State0MidLowLkOut,
                CascadeState0MidLowClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                State0LowestLkIn,
                State0LowestLkOut,
                CascadeState0LowestClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                State1HighestLkIn,
                State1HighestLkOut,
                CascadeState1HighestClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                State1MidHighLkIn,
                State1MidHighLkOut,
                CascadeState1MidHighClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                State1MidLowLkIn,
                State1MidLowLkOut,
                CascadeState1MidLowClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                State1LowestLkIn,
                State1LowestLkOut,
                CascadeState1LowestClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                State2HighestLkIn,
                State2HighestLkOut,
                CascadeState2HighestClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                State2MidHighLkIn,
                State2MidHighLkOut,
                CascadeState2MidHighClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                State2MidLowLkIn,
                State2MidLowLkOut,
                CascadeState2MidLowClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                State2LowestLkIn,
                State2LowestLkOut,
                CascadeState2LowestClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                State3HighestLkIn,
                State3HighestLkOut,
                CascadeState3HighestClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                State3MidHighLkIn,
                State3MidHighLkOut,
                CascadeState3MidHighClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                State3MidLowLkIn,
                State3MidLowLkOut,
                CascadeState3MidLowClientLogDerivative,
            ),
            Self::cascade_log_derivative_update_circuit(
                circuit_builder,
                State3LowestLkIn,
                State3LowestLkOut,
                CascadeState3LowestClientLogDerivative,
            ),
        ];

        [
            constraints,
            hash_function_round_correctly_performs_update.to_vec(),
        ]
            .concat()
    }

    fn indicate_column_index_in_base_row(column: HashBaseTableColumn) -> SingleRowIndicator {
        BaseRow(column.master_base_table_index())
    }

    fn indicate_column_index_in_current_base_row(column: HashBaseTableColumn) -> DualRowIndicator {
        CurrentBaseRow(column.master_base_table_index())
    }

    fn indicate_column_index_in_next_base_row(column: HashBaseTableColumn) -> DualRowIndicator {
        NextBaseRow(column.master_base_table_index())
    }

    fn re_compose_states_0_through_3_before_lookup<II: InputIndicator>(
        circuit_builder: &ConstraintCircuitBuilder<II>,
        base_row_to_input_indicator: fn(HashBaseTableColumn) -> II,
    ) -> [ConstraintCircuitMonad<II>; 4] {
        let input = |input_indicator: II| circuit_builder.input(input_indicator);
        let state_0 = Self::re_compose_16_bit_limbs(
            circuit_builder,
            input(base_row_to_input_indicator(State0HighestLkIn)),
            input(base_row_to_input_indicator(State0MidHighLkIn)),
            input(base_row_to_input_indicator(State0MidLowLkIn)),
            input(base_row_to_input_indicator(State0LowestLkIn)),
        );
        let state_1 = Self::re_compose_16_bit_limbs(
            circuit_builder,
            input(base_row_to_input_indicator(State1HighestLkIn)),
            input(base_row_to_input_indicator(State1MidHighLkIn)),
            input(base_row_to_input_indicator(State1MidLowLkIn)),
            input(base_row_to_input_indicator(State1LowestLkIn)),
        );
        let state_2 = Self::re_compose_16_bit_limbs(
            circuit_builder,
            input(base_row_to_input_indicator(State2HighestLkIn)),
            input(base_row_to_input_indicator(State2MidHighLkIn)),
            input(base_row_to_input_indicator(State2MidLowLkIn)),
            input(base_row_to_input_indicator(State2LowestLkIn)),
        );
        let state_3 = Self::re_compose_16_bit_limbs(
            circuit_builder,
            input(base_row_to_input_indicator(State3HighestLkIn)),
            input(base_row_to_input_indicator(State3MidHighLkIn)),
            input(base_row_to_input_indicator(State3MidLowLkIn)),
            input(base_row_to_input_indicator(State3LowestLkIn)),
        );
        [state_0, state_1, state_2, state_3]
    }

    fn tip5_constraints_as_circuits(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> (
        [ConstraintCircuitMonad<DualRowIndicator>; STATE_SIZE],
        [ConstraintCircuitMonad<DualRowIndicator>; STATE_SIZE],
    ) {
        let constant = |c: u64| circuit_builder.b_constant(c);
        let b_constant = |c| circuit_builder.b_constant(c);
        let current_base_row = |column_idx: HashBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(column_idx.master_base_table_index()))
        };
        let next_base_row = |column_idx: HashBaseTableColumn| {
            circuit_builder.input(NextBaseRow(column_idx.master_base_table_index()))
        };

        let state_0_after_lookup = Self::re_compose_16_bit_limbs(
            circuit_builder,
            current_base_row(State0HighestLkOut),
            current_base_row(State0MidHighLkOut),
            current_base_row(State0MidLowLkOut),
            current_base_row(State0LowestLkOut),
        );
        let state_1_after_lookup = Self::re_compose_16_bit_limbs(
            circuit_builder,
            current_base_row(State1HighestLkOut),
            current_base_row(State1MidHighLkOut),
            current_base_row(State1MidLowLkOut),
            current_base_row(State1LowestLkOut),
        );
        let state_2_after_lookup = Self::re_compose_16_bit_limbs(
            circuit_builder,
            current_base_row(State2HighestLkOut),
            current_base_row(State2MidHighLkOut),
            current_base_row(State2MidLowLkOut),
            current_base_row(State2LowestLkOut),
        );
        let state_3_after_lookup = Self::re_compose_16_bit_limbs(
            circuit_builder,
            current_base_row(State3HighestLkOut),
            current_base_row(State3MidHighLkOut),
            current_base_row(State3MidLowLkOut),
            current_base_row(State3LowestLkOut),
        );

        let state_part_before_power_map: [_; STATE_SIZE - NUM_SPLIT_AND_LOOKUP] = [
            State4, State5, State6, State7, State8, State9, State10, State11, State12, State13,
            State14, State15,
        ]
            .map(current_base_row);

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

        let mut state_after_matrix_multiplication = vec![constant(0); STATE_SIZE];
        for (row_idx, acc) in state_after_matrix_multiplication.iter_mut().enumerate() {
            for (col_idx, state) in state_after_s_box_application.iter().enumerate() {
                let matrix_entry = b_constant(HashTable::mds_matrix_entry(row_idx, col_idx));
                *acc = acc.clone() + matrix_entry * state.clone();
            }
        }

        let round_constants: [_; STATE_SIZE] = [
            Constant0, Constant1, Constant2, Constant3, Constant4, Constant5, Constant6, Constant7,
            Constant8, Constant9, Constant10, Constant11, Constant12, Constant13, Constant14,
            Constant15,
        ]
            .map(current_base_row);

        let state_after_round_constant_addition = state_after_matrix_multiplication
            .into_iter()
            .zip_eq(round_constants)
            .map(|(st, rndc)| st + rndc)
            .collect_vec();

        let [state_0_next, state_1_next, state_2_next, state_3_next] =
            Self::re_compose_states_0_through_3_before_lookup(
                circuit_builder,
                Self::indicate_column_index_in_next_base_row,
            );
        let state_next = [
            state_0_next,
            state_1_next,
            state_2_next,
            state_3_next,
            next_base_row(State4),
            next_base_row(State5),
            next_base_row(State6),
            next_base_row(State7),
            next_base_row(State8),
            next_base_row(State9),
            next_base_row(State10),
            next_base_row(State11),
            next_base_row(State12),
            next_base_row(State13),
            next_base_row(State14),
            next_base_row(State15),
        ];

        let round_number_next = next_base_row(RoundNumber);
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
        look_in_column: HashBaseTableColumn,
        look_out_column: HashBaseTableColumn,
        cascade_log_derivative_column: HashExtTableColumn,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let challenge = |c| circuit_builder.challenge(c);
        let opcode = |instruction: Instruction| circuit_builder.b_constant(instruction.opcode_b());
        let constant = |c: u32| circuit_builder.b_constant(c);
        let next_base_row = |column_idx: HashBaseTableColumn| {
            circuit_builder.input(NextBaseRow(column_idx.master_base_table_index()))
        };
        let current_ext_row = |column_idx: HashExtTableColumn| {
            circuit_builder.input(CurrentExtRow(column_idx.master_ext_table_index()))
        };
        let next_ext_row = |column_idx: HashExtTableColumn| {
            circuit_builder.input(NextExtRow(column_idx.master_ext_table_index()))
        };

        let cascade_indeterminate = challenge(HashCascadeLookupIndeterminate);
        let look_in_weight = challenge(HashCascadeLookInWeight);
        let look_out_weight = challenge(HashCascadeLookOutWeight);

        let ci_next = next_base_row(CI);
        let mode_next = next_base_row(Mode);
        let round_number_next = next_base_row(RoundNumber);
        let cascade_log_derivative = current_ext_row(cascade_log_derivative_column);
        let cascade_log_derivative_next = next_ext_row(cascade_log_derivative_column);

        let compressed_row = look_in_weight * next_base_row(look_in_column)
            + look_out_weight * next_base_row(look_out_column);

        let cascade_log_derivative_remains =
            cascade_log_derivative_next.clone() - cascade_log_derivative.clone();
        let cascade_log_derivative_updates = (cascade_log_derivative_next - cascade_log_derivative)
            * (cascade_indeterminate - compressed_row)
            - constant(1);

        let next_row_is_padding_row_or_round_number_next_is_max_or_ci_next_is_sponge_init =
            Self::select_mode(circuit_builder, &mode_next, HashTableMode::Pad)
                * (round_number_next.clone() - constant(NUM_ROUNDS as u32))
                * (ci_next.clone() - opcode(SpongeInit));
        let round_number_next_is_not_num_rounds =
            Self::round_number_deselector(circuit_builder, &round_number_next, NUM_ROUNDS);
        let current_instruction_next_is_not_sponge_init =
            Self::instruction_deselector(circuit_builder, &ci_next, SpongeInit);

        next_row_is_padding_row_or_round_number_next_is_max_or_ci_next_is_sponge_init
            * cascade_log_derivative_updates
            + round_number_next_is_not_num_rounds * cascade_log_derivative_remains.clone()
            + current_instruction_next_is_not_sponge_init * cascade_log_derivative_remains
    }

    pub fn terminal_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let opcode = |instruction: Instruction| circuit_builder.b_constant(instruction.opcode_b());
        let constant = |c: u64| circuit_builder.b_constant(c);
        let base_row = |column_idx: HashBaseTableColumn| {
            circuit_builder.input(BaseRow(column_idx.master_base_table_index()))
        };

        let mode = base_row(Mode);
        let round_number = base_row(RoundNumber);

        let compress_program_digest_indeterminate = challenge(CompressProgramDigestIndeterminate);
        let expected_program_digest = challenge(CompressedProgramDigest);

        let max_round_number = constant(NUM_ROUNDS as u64);

        let [state_0, state_1, state_2, state_3] =
            Self::re_compose_states_0_through_3_before_lookup(
                circuit_builder,
                Self::indicate_column_index_in_base_row,
            );
        let state_4 = base_row(State4);
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
                * (base_row(CI) - opcode(SpongeInit))
                * (round_number - max_round_number);

        vec![
            if_mode_is_program_hashing_then_current_digest_is_expected_program_digest,
            if_mode_is_not_pad_and_ci_is_not_sponge_init_then_round_number_is_max_round_number,
        ]
    }
}

impl HashTable {
    /// Get the MDS matrix's entry in row `row_idx` and column `col_idx`.
    pub const fn mds_matrix_entry(row_idx: usize, col_idx: usize) -> BFieldElement {
        assert!(row_idx < STATE_SIZE);
        assert!(col_idx < STATE_SIZE);
        let index_in_matrix_defining_column = (STATE_SIZE + row_idx - col_idx) % STATE_SIZE;
        let mds_matrix_entry = MDS_MATRIX_FIRST_COLUMN[index_in_matrix_defining_column];
        BFieldElement::new(mds_matrix_entry as u64)
    }

    /// The round constants for round `r` if it is a valid round number in the Tip5 permutation,
    /// and the zero vector otherwise.
    pub fn tip5_round_constants_by_round_number(r: usize) -> [BFieldElement; NUM_ROUND_CONSTANTS] {
        if r >= NUM_ROUNDS {
            return bfe_array![0; NUM_ROUND_CONSTANTS];
        }

        let range_start = NUM_ROUND_CONSTANTS * r;
        let range_end = NUM_ROUND_CONSTANTS * (r + 1);
        ROUND_CONSTANTS[range_start..range_end].try_into().unwrap()
    }

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
    pub fn base_field_element_into_16_bit_limbs(x: BFieldElement) -> [u16; 4] {
        let r_times_x = (MONTGOMERY_MODULUS * x).value();
        [0, 16, 32, 48].map(|shift| ((r_times_x >> shift) & 0xffff) as u16)
    }

    /// Convert a permutation trace to a segment in the Hash Table.
    ///
    /// **Note**: The current instruction [`CI`] is _not_ set.
    pub fn trace_to_table_rows(trace: PermutationTrace) -> Array2<BFieldElement> {
        let mut table_rows = Array2::default([0, BASE_WIDTH]);
        for (round_number, &trace_row) in trace.iter().enumerate() {
            let table_row = Self::trace_row_to_table_row(trace_row, round_number);
            table_rows.push_row(table_row.view()).unwrap();
        }
        table_rows
    }

    pub fn trace_row_to_table_row(
        trace_row: [BFieldElement; STATE_SIZE],
        round_number: usize,
    ) -> Array1<BFieldElement> {
        let row = Array1::zeros([BASE_WIDTH]);
        let row = Self::fill_row_with_round_number(row, round_number);
        let row = Self::fill_row_with_split_state_elements_using_trace_row(row, trace_row);
        let row = Self::fill_row_with_unsplit_state_elements_using_trace_row(row, trace_row);
        let row = Self::fill_row_with_state_inverses_using_trace_row(row, trace_row);
        Self::fill_row_with_round_constants_for_round(row, round_number)
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
        let row = Self::fill_split_state_element_0_of_row_using_trace_row(row, trace_row);
        let row = Self::fill_split_state_element_1_of_row_using_trace_row(row, trace_row);
        let row = Self::fill_split_state_element_2_of_row_using_trace_row(row, trace_row);
        Self::fill_split_state_element_3_of_row_using_trace_row(row, trace_row)
    }

    fn fill_split_state_element_0_of_row_using_trace_row(
        mut row: Array1<BFieldElement>,
        trace_row: [BFieldElement; STATE_SIZE],
    ) -> Array1<BFieldElement> {
        let limbs = Self::base_field_element_into_16_bit_limbs(trace_row[0]);
        let look_in_split = limbs.map(|limb| bfe!(limb));
        row[State0LowestLkIn.base_table_index()] = look_in_split[0];
        row[State0MidLowLkIn.base_table_index()] = look_in_split[1];
        row[State0MidHighLkIn.base_table_index()] = look_in_split[2];
        row[State0HighestLkIn.base_table_index()] = look_in_split[3];

        let look_out_split = limbs.map(CascadeTable::lookup_16_bit_limb);
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
        let limbs = Self::base_field_element_into_16_bit_limbs(trace_row[1]);
        let look_in_split = limbs.map(|limb| bfe!(limb));
        row[State1LowestLkIn.base_table_index()] = look_in_split[0];
        row[State1MidLowLkIn.base_table_index()] = look_in_split[1];
        row[State1MidHighLkIn.base_table_index()] = look_in_split[2];
        row[State1HighestLkIn.base_table_index()] = look_in_split[3];

        let look_out_split = limbs.map(CascadeTable::lookup_16_bit_limb);
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
        let limbs = Self::base_field_element_into_16_bit_limbs(trace_row[2]);
        let look_in_split = limbs.map(|limb| bfe!(limb));
        row[State2LowestLkIn.base_table_index()] = look_in_split[0];
        row[State2MidLowLkIn.base_table_index()] = look_in_split[1];
        row[State2MidHighLkIn.base_table_index()] = look_in_split[2];
        row[State2HighestLkIn.base_table_index()] = look_in_split[3];

        let look_out_split = limbs.map(CascadeTable::lookup_16_bit_limb);
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
        let limbs = Self::base_field_element_into_16_bit_limbs(trace_row[3]);
        let look_in_split = limbs.map(|limb| bfe!(limb));
        row[State3LowestLkIn.base_table_index()] = look_in_split[0];
        row[State3MidLowLkIn.base_table_index()] = look_in_split[1];
        row[State3MidHighLkIn.base_table_index()] = look_in_split[2];
        row[State3HighestLkIn.base_table_index()] = look_in_split[3];

        let look_out_split = limbs.map(CascadeTable::lookup_16_bit_limb);
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
        row[State0Inv.base_table_index()] = Self::inverse_or_zero_of_highest_2_limbs(trace_row[0]);
        row[State1Inv.base_table_index()] = Self::inverse_or_zero_of_highest_2_limbs(trace_row[1]);
        row[State2Inv.base_table_index()] = Self::inverse_or_zero_of_highest_2_limbs(trace_row[2]);
        row[State3Inv.base_table_index()] = Self::inverse_or_zero_of_highest_2_limbs(trace_row[3]);
        row
    }

    /// The inverse-or-zero of (2^32 - 1 - 2^16·`highest` - `mid_high`) where `highest`
    /// is the most significant limb of the given `state_element`, and `mid_high` the second-most
    /// significant limb.
    fn inverse_or_zero_of_highest_2_limbs(state_element: BFieldElement) -> BFieldElement {
        let limbs = Self::base_field_element_into_16_bit_limbs(state_element);
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
        let round_constants = Self::tip5_round_constants_by_round_number(round_number);
        row[Constant0.base_table_index()] = round_constants[0];
        row[Constant1.base_table_index()] = round_constants[1];
        row[Constant2.base_table_index()] = round_constants[2];
        row[Constant3.base_table_index()] = round_constants[3];
        row[Constant4.base_table_index()] = round_constants[4];
        row[Constant5.base_table_index()] = round_constants[5];
        row[Constant6.base_table_index()] = round_constants[6];
        row[Constant7.base_table_index()] = round_constants[7];
        row[Constant8.base_table_index()] = round_constants[8];
        row[Constant9.base_table_index()] = round_constants[9];
        row[Constant10.base_table_index()] = round_constants[10];
        row[Constant11.base_table_index()] = round_constants[11];
        row[Constant12.base_table_index()] = round_constants[12];
        row[Constant13.base_table_index()] = round_constants[13];
        row[Constant14.base_table_index()] = round_constants[14];
        row[Constant15.base_table_index()] = round_constants[15];
        row
    }

    pub fn fill_trace(
        hash_table: &mut ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
    ) {
        let program_hash_part_start = 0;
        let program_hash_part_end = program_hash_part_start + aet.program_hash_trace.nrows();
        let sponge_part_start = program_hash_part_end;
        let sponge_part_end = sponge_part_start + aet.sponge_trace.nrows();
        let hash_part_start = sponge_part_end;
        let hash_part_end = hash_part_start + aet.hash_trace.nrows();

        let (mut program_hash_part, mut sponge_part, mut hash_part) = hash_table.multi_slice_mut((
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

    pub fn pad_trace(mut hash_table: ArrayViewMut2<BFieldElement>, hash_table_length: usize) {
        let inverse_of_high_limbs = Self::inverse_or_zero_of_highest_2_limbs(bfe!(0));
        for column_id in [State0Inv, State1Inv, State2Inv, State3Inv] {
            let column_index = column_id.base_table_index();
            let slice_info = s![hash_table_length.., column_index];
            let mut column = hash_table.slice_mut(slice_info);
            column.fill(inverse_of_high_limbs);
        }

        let round_constants = Self::tip5_round_constants_by_round_number(0);
        for (round_constant_idx, &round_constant) in round_constants.iter().enumerate() {
            let round_constant_column =
                ExtHashTable::round_constant_column_by_index(round_constant_idx);
            let round_constant_column_idx = round_constant_column.base_table_index();
            let slice_info = s![hash_table_length.., round_constant_column_idx];
            let mut column = hash_table.slice_mut(slice_info);
            column.fill(round_constant);
        }

        let mode_column_index = Mode.base_table_index();
        let mode_column_slice_info = s![hash_table_length.., mode_column_index];
        let mut mode_column = hash_table.slice_mut(mode_column_slice_info);
        mode_column.fill(HashTableMode::Pad.into());

        let instruction_column_index = CI.base_table_index();
        let instruction_column_slice_info = s![hash_table_length.., instruction_column_index];
        let mut instruction_column = hash_table.slice_mut(instruction_column_slice_info);
        instruction_column.fill(Instruction::Hash.opcode_b());
    }

    pub fn extend(
        base_table: ArrayView2<BFieldElement>,
        mut ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        profiler!(start "hash table");
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());

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
             highest: HashBaseTableColumn,
             mid_high: HashBaseTableColumn,
             mid_low: HashBaseTableColumn,
             lowest: HashBaseTableColumn| {
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
             lk_in_col: HashBaseTableColumn,
             lk_out_col: HashBaseTableColumn| {
                let compressed_elements = cascade_indeterminate
                    - cascade_look_in_weight * row[lk_in_col.base_table_index()]
                    - cascade_look_out_weight * row[lk_out_col.base_table_index()];
                compressed_elements.inverse()
            };

        for row_idx in 0..base_table.nrows() {
            let row = base_table.row(row_idx);

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

            let mut extension_row = ext_table.row_mut(row_idx);
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
    use std::collections::HashMap;

    use crate::shared_tests::ProgramAndInput;
    use crate::stark::tests::master_tables_for_low_security_level;
    use crate::table::master_table::MasterTable;
    use crate::table::master_table::TableId;
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
        for (constraint_idx, constraint) in ExtHashTable::terminal_constraints(&circuit_builder)
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
