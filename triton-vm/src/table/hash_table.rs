use itertools::Itertools;
use ndarray::s;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use num_traits::Zero;
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::b_field_element::BFIELD_ONE;
use twenty_first::shared_math::b_field_element::BFIELD_ZERO;
use twenty_first::shared_math::tip5::DIGEST_LENGTH;
use twenty_first::shared_math::tip5::MDS_MATRIX_FIRST_COLUMN;
use twenty_first::shared_math::tip5::NUM_ROUNDS;
use twenty_first::shared_math::tip5::NUM_SPLIT_AND_LOOKUP;
use twenty_first::shared_math::tip5::RATE;
use twenty_first::shared_math::tip5::ROUND_CONSTANTS;
use twenty_first::shared_math::tip5::STATE_SIZE;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use triton_opcodes::instruction::Instruction;

use crate::table::cascade_table::CascadeTable;
use crate::table::challenges::ChallengeId::*;
use crate::table::challenges::Challenges;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::ConstraintCircuitMonad;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::DualRowIndicator::*;
use crate::table::constraint_circuit::InputIndicator;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::*;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::EvalArg;
use crate::table::cross_table_argument::LookupArg;
use crate::table::table_column::HashBaseTableColumn;
use crate::table::table_column::HashBaseTableColumn::*;
use crate::table::table_column::HashExtTableColumn;
use crate::table::table_column::HashExtTableColumn::*;
use crate::table::table_column::MasterBaseTableColumn;
use crate::table::table_column::MasterExtTableColumn;
use crate::vm::AlgebraicExecutionTrace;

pub const BASE_WIDTH: usize = HashBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = HashExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

pub const POWER_MAP_EXPONENT: u64 = 7;
pub const NUM_ROUND_CONSTANTS: usize = STATE_SIZE;

#[derive(Debug, Clone)]
pub struct HashTable {}

#[derive(Debug, Clone)]
pub struct ExtHashTable {}

impl ExtHashTable {
    fn re_compose_16_bit_limbs<II: InputIndicator>(
        circuit_builder: &ConstraintCircuitBuilder<II>,
        highest: ConstraintCircuitMonad<II>,
        mid_high: ConstraintCircuitMonad<II>,
        mid_low: ConstraintCircuitMonad<II>,
        lowest: ConstraintCircuitMonad<II>,
    ) -> ConstraintCircuitMonad<II> {
        let constant = |c: u64| circuit_builder.b_constant(c.into());

        // R is the field element congruent to 2^64 modulo p, accounting for native
        // representation of field elements in Montgomery form.
        // State elements are multiplied by R prior to decomposition into limbs.
        // After re-composition from limbs, the result must be multiplied by R_inv.
        // See `HashTable::base_field_element_16_bit_limbs` for some more details.
        let capital_r = BFieldElement::new(((1_u128 << 64) % BFieldElement::P as u128) as u64);
        let capital_r_inv = circuit_builder.b_constant(capital_r.inverse());

        (highest * constant(1 << 48)
            + mid_high * constant(1 << 32)
            + mid_low * constant(1 << 16)
            + lowest)
            * capital_r_inv
    }

    pub fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let constant = |c: u64| circuit_builder.b_constant(c.into());
        let b_constant = |c| circuit_builder.b_constant(c);

        let base_row = |column: HashBaseTableColumn| {
            circuit_builder.input(BaseRow(column.master_base_table_index()))
        };
        let ext_row = |column: HashExtTableColumn| {
            circuit_builder.input(ExtRow(column.master_ext_table_index()))
        };

        let running_evaluation_initial = circuit_builder.x_constant(EvalArg::default_initial());

        let round_number = base_row(RoundNumber);
        let ci = base_row(CI);
        let running_evaluation_hash_input = ext_row(HashInputRunningEvaluation);
        let running_evaluation_hash_digest = ext_row(HashDigestRunningEvaluation);
        let running_evaluation_sponge = ext_row(SpongeRunningEvaluation);

        let one = constant(1);

        let state_0 = Self::re_compose_16_bit_limbs(
            circuit_builder,
            base_row(State0HighestLkIn),
            base_row(State0MidHighLkIn),
            base_row(State0MidLowLkIn),
            base_row(State0LowestLkIn),
        );
        let state_1 = Self::re_compose_16_bit_limbs(
            circuit_builder,
            base_row(State1HighestLkIn),
            base_row(State1MidHighLkIn),
            base_row(State1MidLowLkIn),
            base_row(State1LowestLkIn),
        );
        let state_2 = Self::re_compose_16_bit_limbs(
            circuit_builder,
            base_row(State2HighestLkIn),
            base_row(State2MidHighLkIn),
            base_row(State2MidLowLkIn),
            base_row(State2LowestLkIn),
        );
        let state_3 = Self::re_compose_16_bit_limbs(
            circuit_builder,
            base_row(State3HighestLkIn),
            base_row(State3MidHighLkIn),
            base_row(State3MidLowLkIn),
            base_row(State3LowestLkIn),
        );

        let state = [
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

        let state_weights = [
            HashStateWeight0,
            HashStateWeight1,
            HashStateWeight2,
            HashStateWeight3,
            HashStateWeight4,
            HashStateWeight5,
            HashStateWeight6,
            HashStateWeight7,
            HashStateWeight8,
            HashStateWeight9,
        ];
        let compressed_row: ConstraintCircuitMonad<_> = state_weights
            .into_iter()
            .zip_eq(state.into_iter())
            .map(|(weight, state)| challenge(weight) * state)
            .sum();

        let round_number_is_neg_1_or_0 =
            (round_number.clone() + one.clone()) * round_number.clone();

        let ci_is_hash = ci.clone() - b_constant(Instruction::Hash.opcode_b());
        let ci_is_absorb_init = ci - b_constant(Instruction::AbsorbInit.opcode_b());
        let current_instruction_is_absorb_init_or_hash =
            ci_is_absorb_init.clone() * ci_is_hash.clone();

        // Evaluation Argument “hash input”
        // If the round number is -1, the running evaluation is the default initial.
        // If the current instruction is AbsorbInit, the running evaluation is the default initial.
        // Else, the first update has been applied to the running evaluation.
        let hash_input_indeterminate = challenge(HashInputIndeterminate);
        let running_evaluation_hash_input_is_default_initial =
            running_evaluation_hash_input.clone() - running_evaluation_initial.clone();
        let running_evaluation_hash_input_has_accumulated_first_row = running_evaluation_hash_input
            - running_evaluation_initial.clone() * hash_input_indeterminate
            - compressed_row.clone();
        let running_evaluation_hash_input_is_initialized_correctly = (round_number.clone() + one)
            * ci_is_absorb_init.clone()
            * running_evaluation_hash_input_has_accumulated_first_row
            + ci_is_hash.clone() * running_evaluation_hash_input_is_default_initial.clone()
            + round_number * running_evaluation_hash_input_is_default_initial;

        // Evaluation Argument “hash digest”
        let running_evaluation_hash_digest_is_default_initial =
            running_evaluation_hash_digest - running_evaluation_initial.clone();

        // Evaluation Argument “Sponge”
        let sponge_indeterminate = challenge(SpongeIndeterminate);
        let running_evaluation_sponge_is_default_initial =
            running_evaluation_sponge.clone() - running_evaluation_initial.clone();
        let running_evaluation_sponge_has_accumulated_first_row = running_evaluation_sponge
            - running_evaluation_initial * sponge_indeterminate
            - challenge(HashCIWeight) * b_constant(Instruction::AbsorbInit.opcode_b())
            - compressed_row;
        let running_evaluation_sponge_absorb_is_initialized_correctly = ci_is_hash
            * running_evaluation_sponge_has_accumulated_first_row
            + ci_is_absorb_init * running_evaluation_sponge_is_default_initial;

        let mut constraints = vec![
            round_number_is_neg_1_or_0,
            current_instruction_is_absorb_init_or_hash,
            running_evaluation_hash_input_is_initialized_correctly,
            running_evaluation_hash_digest_is_default_initial,
            running_evaluation_sponge_absorb_is_initialized_correctly,
        ];
        constraints
            .append(&mut Self::all_cascade_log_derivative_init_circuits(circuit_builder).to_vec());
        constraints
    }

    fn round_number_deselector<II: InputIndicator>(
        circuit_builder: &ConstraintCircuitBuilder<II>,
        round_number_circuit_node: &ConstraintCircuitMonad<II>,
        round_number_to_deselect: isize,
    ) -> ConstraintCircuitMonad<II> {
        assert!(
            -1 <= round_number_to_deselect && round_number_to_deselect <= NUM_ROUNDS as isize,
            "Round number to deselect must be in the range [-1, {NUM_ROUNDS}] \
            but got {round_number_to_deselect}."
        );
        // Because BFieldElements cannot be built from signed integers, we special-case -1.
        let constant = |c: u64| circuit_builder.b_constant(c.into());
        let factor_for_neg_1 = match round_number_to_deselect == -1 {
            true => constant(1),
            false => round_number_circuit_node.clone() + constant(1),
        };
        (0..=NUM_ROUNDS as isize)
            .filter(|&r| r != round_number_to_deselect)
            .map(|r| round_number_circuit_node.clone() - constant(r as u64))
            .fold(factor_for_neg_1, |a, b| a * b)
    }

    fn cascade_log_derivative_init_circuit(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
        look_in_column: HashBaseTableColumn,
        look_out_column: HashBaseTableColumn,
        cascade_log_derivative_column: HashExtTableColumn,
    ) -> ConstraintCircuitMonad<SingleRowIndicator> {
        let challenge = |c| circuit_builder.challenge(c);
        let constant = |c: u64| circuit_builder.b_constant(c.into());
        let base_row = |column_idx: HashBaseTableColumn| {
            circuit_builder.input(BaseRow(column_idx.master_base_table_index()))
        };
        let ext_row = |column_idx: HashExtTableColumn| {
            circuit_builder.input(ExtRow(column_idx.master_ext_table_index()))
        };

        let cascade_indeterminate = challenge(HashCascadeLookupIndeterminate);
        let look_in_weight = challenge(HashCascadeLookInWeight);
        let look_out_weight = challenge(HashCascadeLookOutWeight);

        let round_number = base_row(RoundNumber);
        let cascade_log_derivative = ext_row(cascade_log_derivative_column);
        let default_initial = circuit_builder.x_constant(LookupArg::default_initial());

        let compressed_row =
            look_in_weight * base_row(look_in_column) + look_out_weight * base_row(look_out_column);

        let cascade_log_derivative_is_default_initial =
            cascade_log_derivative.clone() - default_initial.clone();
        let cascade_log_derivative_updates = (cascade_log_derivative - default_initial)
            * (cascade_indeterminate - compressed_row)
            - constant(1);

        let round_number_next_is_neg_1 = round_number.clone() + constant(1);
        let round_number_next_is_0 = round_number;

        round_number_next_is_neg_1 * cascade_log_derivative_updates
            + round_number_next_is_0 * cascade_log_derivative_is_default_initial
    }

    fn all_cascade_log_derivative_init_circuits(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> [ConstraintCircuitMonad<SingleRowIndicator>; 4 * NUM_SPLIT_AND_LOOKUP] {
        [
            Self::cascade_log_derivative_init_circuit(
                circuit_builder,
                State0HighestLkIn,
                State0HighestLkOut,
                CascadeState0HighestClientLogDerivative,
            ),
            Self::cascade_log_derivative_init_circuit(
                circuit_builder,
                State0MidHighLkIn,
                State0MidHighLkOut,
                CascadeState0MidHighClientLogDerivative,
            ),
            Self::cascade_log_derivative_init_circuit(
                circuit_builder,
                State0MidLowLkIn,
                State0MidLowLkOut,
                CascadeState0MidLowClientLogDerivative,
            ),
            Self::cascade_log_derivative_init_circuit(
                circuit_builder,
                State0LowestLkIn,
                State0LowestLkOut,
                CascadeState0LowestClientLogDerivative,
            ),
            Self::cascade_log_derivative_init_circuit(
                circuit_builder,
                State1HighestLkIn,
                State1HighestLkOut,
                CascadeState1HighestClientLogDerivative,
            ),
            Self::cascade_log_derivative_init_circuit(
                circuit_builder,
                State1MidHighLkIn,
                State1MidHighLkOut,
                CascadeState1MidHighClientLogDerivative,
            ),
            Self::cascade_log_derivative_init_circuit(
                circuit_builder,
                State1MidLowLkIn,
                State1MidLowLkOut,
                CascadeState1MidLowClientLogDerivative,
            ),
            Self::cascade_log_derivative_init_circuit(
                circuit_builder,
                State1LowestLkIn,
                State1LowestLkOut,
                CascadeState1LowestClientLogDerivative,
            ),
            Self::cascade_log_derivative_init_circuit(
                circuit_builder,
                State2HighestLkIn,
                State2HighestLkOut,
                CascadeState2HighestClientLogDerivative,
            ),
            Self::cascade_log_derivative_init_circuit(
                circuit_builder,
                State2MidHighLkIn,
                State2MidHighLkOut,
                CascadeState2MidHighClientLogDerivative,
            ),
            Self::cascade_log_derivative_init_circuit(
                circuit_builder,
                State2MidLowLkIn,
                State2MidLowLkOut,
                CascadeState2MidLowClientLogDerivative,
            ),
            Self::cascade_log_derivative_init_circuit(
                circuit_builder,
                State2LowestLkIn,
                State2LowestLkOut,
                CascadeState2LowestClientLogDerivative,
            ),
            Self::cascade_log_derivative_init_circuit(
                circuit_builder,
                State3HighestLkIn,
                State3HighestLkOut,
                CascadeState3HighestClientLogDerivative,
            ),
            Self::cascade_log_derivative_init_circuit(
                circuit_builder,
                State3MidHighLkIn,
                State3MidHighLkOut,
                CascadeState3MidHighClientLogDerivative,
            ),
            Self::cascade_log_derivative_init_circuit(
                circuit_builder,
                State3MidLowLkIn,
                State3MidLowLkOut,
                CascadeState3MidLowClientLogDerivative,
            ),
            Self::cascade_log_derivative_init_circuit(
                circuit_builder,
                State3LowestLkIn,
                State3LowestLkOut,
                CascadeState3LowestClientLogDerivative,
            ),
        ]
    }

    pub fn consistency_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let constant = |c: u64| circuit_builder.b_constant(c.into());
        let base_row = |column_id: HashBaseTableColumn| {
            circuit_builder.input(BaseRow(column_id.master_base_table_index()))
        };

        let round_number = base_row(RoundNumber);
        let ci = base_row(CI);
        let state10 = base_row(State10);
        let state11 = base_row(State11);
        let state12 = base_row(State12);
        let state13 = base_row(State13);
        let state14 = base_row(State14);
        let state15 = base_row(State15);

        let ci_is_hash = ci.clone() - constant(Instruction::Hash.opcode() as u64);
        let ci_is_absorb_init = ci.clone() - constant(Instruction::AbsorbInit.opcode() as u64);
        let ci_is_absorb = ci.clone() - constant(Instruction::Absorb.opcode() as u64);
        let ci_is_squeeze = ci - constant(Instruction::Squeeze.opcode() as u64);

        let round_number_is_not_neg_1 =
            Self::round_number_deselector(circuit_builder, &round_number, -1);
        let round_number_is_not_0 =
            Self::round_number_deselector(circuit_builder, &round_number, 0);

        let if_padding_row_then_ci_is_hash = round_number_is_not_neg_1 * ci_is_hash.clone();

        let if_ci_is_hash_and_round_no_is_0_then_ = round_number_is_not_0.clone()
            * ci_is_absorb_init
            * ci_is_absorb.clone()
            * ci_is_squeeze.clone();
        let if_ci_is_hash_and_round_no_is_0_then_state_10_is_1 =
            if_ci_is_hash_and_round_no_is_0_then_.clone() * (state10.clone() - constant(1));
        let if_ci_is_hash_and_round_no_is_0_then_state_11_is_1 =
            if_ci_is_hash_and_round_no_is_0_then_.clone() * (state11.clone() - constant(1));
        let if_ci_is_hash_and_round_no_is_0_then_state_12_is_1 =
            if_ci_is_hash_and_round_no_is_0_then_.clone() * (state12.clone() - constant(1));
        let if_ci_is_hash_and_round_no_is_0_then_state_13_is_1 =
            if_ci_is_hash_and_round_no_is_0_then_.clone() * (state13.clone() - constant(1));
        let if_ci_is_hash_and_round_no_is_0_then_state_14_is_1 =
            if_ci_is_hash_and_round_no_is_0_then_.clone() * (state14.clone() - constant(1));
        let if_ci_is_hash_and_round_no_is_0_then_state_15_is_1 =
            if_ci_is_hash_and_round_no_is_0_then_ * (state15.clone() - constant(1));

        let if_ci_is_absorb_init_and_round_no_is_0_then_ =
            round_number_is_not_0 * ci_is_hash * ci_is_absorb * ci_is_squeeze;
        let if_ci_is_absorb_init_and_round_no_is_0_then_state_10_is_0 =
            if_ci_is_absorb_init_and_round_no_is_0_then_.clone() * state10;
        let if_ci_is_absorb_init_and_round_no_is_0_then_state_11_is_0 =
            if_ci_is_absorb_init_and_round_no_is_0_then_.clone() * state11;
        let if_ci_is_absorb_init_and_round_no_is_0_then_state_12_is_0 =
            if_ci_is_absorb_init_and_round_no_is_0_then_.clone() * state12;
        let if_ci_is_absorb_init_and_round_no_is_0_then_state_13_is_0 =
            if_ci_is_absorb_init_and_round_no_is_0_then_.clone() * state13;
        let if_ci_is_absorb_init_and_round_no_is_0_then_state_14_is_0 =
            if_ci_is_absorb_init_and_round_no_is_0_then_.clone() * state14;
        let if_ci_is_absorb_init_and_round_no_is_0_then_state_15_is_0 =
            if_ci_is_absorb_init_and_round_no_is_0_then_ * state15;

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
            if_padding_row_then_ci_is_hash,
            if_ci_is_hash_and_round_no_is_0_then_state_10_is_1,
            if_ci_is_hash_and_round_no_is_0_then_state_11_is_1,
            if_ci_is_hash_and_round_no_is_0_then_state_12_is_1,
            if_ci_is_hash_and_round_no_is_0_then_state_13_is_1,
            if_ci_is_hash_and_round_no_is_0_then_state_14_is_1,
            if_ci_is_hash_and_round_no_is_0_then_state_15_is_1,
            if_ci_is_absorb_init_and_round_no_is_0_then_state_10_is_0,
            if_ci_is_absorb_init_and_round_no_is_0_then_state_11_is_0,
            if_ci_is_absorb_init_and_round_no_is_0_then_state_12_is_0,
            if_ci_is_absorb_init_and_round_no_is_0_then_state_13_is_0,
            if_ci_is_absorb_init_and_round_no_is_0_then_state_14_is_0,
            if_ci_is_absorb_init_and_round_no_is_0_then_state_15_is_0,
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

        for round_constant_column_idx in 0..NUM_ROUND_CONSTANTS {
            let round_constant_column =
                Self::round_constant_column_by_index(round_constant_column_idx);
            let round_constant_column_circuit = base_row(round_constant_column);
            let mut round_constant_constraint_circuit = constant(0);
            for round_idx in 0..NUM_ROUNDS {
                let round_constant_idx_for_current_row =
                    NUM_ROUND_CONSTANTS * round_idx + round_constant_column_idx;
                let round_constant_for_current_row =
                    circuit_builder.b_constant(ROUND_CONSTANTS[round_constant_idx_for_current_row]);
                let round_deselector_circuit = Self::round_number_deselector(
                    circuit_builder,
                    &round_number,
                    round_idx as isize,
                );
                round_constant_constraint_circuit = round_constant_constraint_circuit
                    + round_deselector_circuit
                        * (round_constant_column_circuit.clone() - round_constant_for_current_row);
            }
            constraints.push(round_constant_constraint_circuit);
        }

        constraints
    }

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

    pub fn transition_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let constant = |c: u64| circuit_builder.b_constant(c.into());
        let b_constant = |c| circuit_builder.b_constant(c);

        let opcode_hash = b_constant(Instruction::Hash.opcode_b());
        let opcode_absorb_init = b_constant(Instruction::AbsorbInit.opcode_b());
        let opcode_absorb = b_constant(Instruction::Absorb.opcode_b());
        let opcode_squeeze = b_constant(Instruction::Squeeze.opcode_b());

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

        let hash_input_eval_indeterminate = challenge(HashInputIndeterminate);
        let hash_digest_eval_indeterminate = challenge(HashDigestIndeterminate);
        let sponge_indeterminate = challenge(SpongeIndeterminate);

        let round_number = current_base_row(RoundNumber);
        let ci = current_base_row(CI);
        let running_evaluation_hash_input = current_ext_row(HashInputRunningEvaluation);
        let running_evaluation_hash_digest = current_ext_row(HashDigestRunningEvaluation);
        let running_evaluation_sponge = current_ext_row(SpongeRunningEvaluation);

        let round_number_next = next_base_row(RoundNumber);
        let ci_next = next_base_row(CI);
        let running_evaluation_hash_input_next = next_ext_row(HashInputRunningEvaluation);
        let running_evaluation_hash_digest_next = next_ext_row(HashDigestRunningEvaluation);
        let running_evaluation_sponge_next = next_ext_row(SpongeRunningEvaluation);

        let state_0 = Self::re_compose_16_bit_limbs(
            circuit_builder,
            current_base_row(State0HighestLkIn),
            current_base_row(State0MidHighLkIn),
            current_base_row(State0MidLowLkIn),
            current_base_row(State0LowestLkIn),
        );
        let state_1 = Self::re_compose_16_bit_limbs(
            circuit_builder,
            current_base_row(State1HighestLkIn),
            current_base_row(State1MidHighLkIn),
            current_base_row(State1MidLowLkIn),
            current_base_row(State1LowestLkIn),
        );
        let state_2 = Self::re_compose_16_bit_limbs(
            circuit_builder,
            current_base_row(State2HighestLkIn),
            current_base_row(State2MidHighLkIn),
            current_base_row(State2MidLowLkIn),
            current_base_row(State2LowestLkIn),
        );
        let state_3 = Self::re_compose_16_bit_limbs(
            circuit_builder,
            current_base_row(State3HighestLkIn),
            current_base_row(State3MidHighLkIn),
            current_base_row(State3MidLowLkIn),
            current_base_row(State3LowestLkIn),
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
            HashStateWeight0,
            HashStateWeight1,
            HashStateWeight2,
            HashStateWeight3,
            HashStateWeight4,
            HashStateWeight5,
            HashStateWeight6,
            HashStateWeight7,
            HashStateWeight8,
            HashStateWeight9,
            HashStateWeight10,
            HashStateWeight11,
            HashStateWeight12,
            HashStateWeight13,
            HashStateWeight14,
            HashStateWeight15,
        ]
        .map(challenge);

        // round numbers evolve as
        // 0 -> 1 -> 2 -> 3 -> 4 -> 5, and
        // 5 -> -1 or 5 -> 0, and
        // -1 -> -1

        let round_number_is_not_neg_1 =
            Self::round_number_deselector(circuit_builder, &round_number, -1);
        let round_number_is_not_5 =
            Self::round_number_deselector(circuit_builder, &round_number, 5);

        let round_number_is_0_through_5_or_round_number_next_is_neg_1 =
            round_number_is_not_neg_1 * (round_number_next.clone() + constant(1));

        let round_number_is_neg_1_through_4_or_round_number_next_is_0_or_neg_1 =
            round_number_is_not_5
                * round_number_next.clone()
                * (round_number_next.clone() + constant(1));

        let round_number_is_neg_1_or_5_or_increments_by_one = (round_number.clone() + constant(1))
            * (round_number.clone() - constant(NUM_ROUNDS as u64))
            * (round_number_next.clone() - round_number.clone() - constant(1));

        let if_ci_is_hash_then_ci_doesnt_change = (ci.clone() - opcode_absorb_init.clone())
            * (ci.clone() - opcode_absorb.clone())
            * (ci.clone() - opcode_squeeze.clone())
            * (ci_next.clone() - opcode_hash.clone());

        let if_round_number_is_not_5_then_ci_doesnt_change =
            (round_number - constant(NUM_ROUNDS as u64)) * (ci_next.clone() - ci);

        // copy capacity between rounds with index 5 and 0 if instruction is “absorb”
        let round_number_next_is_not_0 =
            Self::round_number_deselector(circuit_builder, &round_number_next, 0);
        let round_number_next_is_0 = round_number_next.clone();

        let difference_of_capacity_registers = state_current[RATE..]
            .iter()
            .zip_eq(state_next[RATE..].iter())
            .map(|(current, next)| next.clone() - current.clone())
            .collect_vec();
        let randomized_sum_of_capacity_differences = state_weights[RATE..]
            .iter()
            .zip_eq(difference_of_capacity_registers)
            .map(|(weight, state_difference)| weight.clone() * state_difference)
            .sum();
        let if_round_number_next_is_0_and_ci_next_is_absorb_then_capacity_doesnt_change =
            round_number_next_is_not_0.clone()
                * (ci_next.clone() - opcode_hash.clone())
                * (ci_next.clone() - opcode_absorb_init.clone())
                * (ci_next.clone() - opcode_squeeze.clone())
                * randomized_sum_of_capacity_differences;

        // copy entire state between rounds with index 5 and 0 if instruction is “squeeze”
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
            round_number_next_is_not_0.clone()
                * (ci_next.clone() - opcode_hash.clone())
                * (ci_next.clone() - opcode_absorb_init.clone())
                * (ci_next.clone() - opcode_absorb.clone())
                * randomized_sum_of_state_differences;

        // Evaluation Arguments

        // If (and only if) the row number in the next row is 0 and the current instruction in
        // the next row is corresponds to `hash`, update running evaluation “hash input.”
        let ci_next_is_not_hash = (ci_next.clone() - opcode_absorb_init.clone())
            * (ci_next.clone() - opcode_absorb.clone())
            * (ci_next.clone() - opcode_squeeze.clone());
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
        let running_evaluation_hash_input_is_updated_correctly = round_number_next_is_not_0.clone()
            * ci_next_is_not_hash.clone()
            * running_evaluation_hash_input_updates
            + round_number_next_is_0.clone() * running_evaluation_hash_input_remains.clone()
            + (ci_next.clone() - opcode_hash.clone()) * running_evaluation_hash_input_remains;

        // If (and only if) the row number in the next row is 5 and the current instruction in
        // the next row corresponds to `hash`, update running evaluation “hash digest.”
        let round_number_next_is_5 = round_number_next.clone() - constant(NUM_ROUNDS as u64);
        let round_number_next_is_not_5 =
            Self::round_number_deselector(circuit_builder, &round_number_next, NUM_ROUNDS as isize);
        let running_evaluation_hash_digest_remains =
            running_evaluation_hash_digest_next.clone() - running_evaluation_hash_digest.clone();
        let hash_digest = state_next[..DIGEST_LENGTH].to_owned();
        let compressed_row_hash_digest = hash_digest
            .into_iter()
            .zip_eq(state_weights[..DIGEST_LENGTH].iter())
            .map(|(state, weight)| weight.clone() * state)
            .sum();
        let running_evaluation_hash_digest_updates = running_evaluation_hash_digest_next
            - hash_digest_eval_indeterminate * running_evaluation_hash_digest
            - compressed_row_hash_digest;
        let running_evaluation_hash_digest_is_updated_correctly = round_number_next_is_not_5
            * ci_next_is_not_hash
            * running_evaluation_hash_digest_updates
            + round_number_next_is_5 * running_evaluation_hash_digest_remains.clone()
            + (ci_next.clone() - opcode_hash.clone()) * running_evaluation_hash_digest_remains;

        // The running evaluation for “Sponge” updates correctly.
        let compressed_row_next = state_weights[..RATE]
            .iter()
            .zip_eq(state_next[..RATE].iter())
            .map(|(weight, st_next)| weight.clone() * st_next.clone())
            .sum();
        let running_evaluation_sponge_has_accumulated_next_row = running_evaluation_sponge_next
            .clone()
            - sponge_indeterminate * running_evaluation_sponge.clone()
            - challenge(HashCIWeight) * ci_next.clone()
            - compressed_row_next;
        let if_round_no_next_0_and_ci_next_is_spongy_then_running_eval_sponge_updates =
            round_number_next_is_not_0
                * (ci_next.clone() - opcode_hash)
                * running_evaluation_sponge_has_accumulated_next_row;

        let running_evaluation_sponge_absorb_remains =
            running_evaluation_sponge_next - running_evaluation_sponge;
        let if_round_no_next_is_not_0_then_running_evaluation_sponge_absorb_remains =
            round_number_next_is_0 * running_evaluation_sponge_absorb_remains.clone();
        let if_ci_next_is_not_spongy_then_running_evaluation_sponge_absorb_remains =
            (ci_next.clone() - opcode_absorb_init)
                * (ci_next.clone() - opcode_absorb)
                * (ci_next - opcode_squeeze)
                * running_evaluation_sponge_absorb_remains;
        let running_evaluation_sponge_is_updated_correctly =
            if_round_no_next_0_and_ci_next_is_spongy_then_running_eval_sponge_updates
                + if_round_no_next_is_not_0_then_running_evaluation_sponge_absorb_remains
                + if_ci_next_is_not_spongy_then_running_evaluation_sponge_absorb_remains;

        [
            vec![
                round_number_is_0_through_5_or_round_number_next_is_neg_1,
                round_number_is_neg_1_through_4_or_round_number_next_is_0_or_neg_1,
                round_number_is_neg_1_or_5_or_increments_by_one,
                if_ci_is_hash_then_ci_doesnt_change,
                if_round_number_is_not_5_then_ci_doesnt_change,
            ],
            hash_function_round_correctly_performs_update.to_vec(),
            vec![
                if_round_number_next_is_0_and_ci_next_is_absorb_then_capacity_doesnt_change,
                if_round_number_next_is_0_and_ci_next_is_squeeze_then_state_doesnt_change,
                running_evaluation_hash_input_is_updated_correctly,
                running_evaluation_hash_digest_is_updated_correctly,
                running_evaluation_sponge_is_updated_correctly,
            ],
            Self::all_cascade_log_derivative_update_circuits(circuit_builder).to_vec(),
        ]
        .concat()
    }

    fn tip5_constraints_as_circuits(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> (
        [ConstraintCircuitMonad<DualRowIndicator>; STATE_SIZE],
        [ConstraintCircuitMonad<DualRowIndicator>; STATE_SIZE],
    ) {
        let constant = |c: u64| circuit_builder.b_constant(c.into());
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

        #[allow(clippy::needless_range_loop)]
        let state_after_matrix_multiplication: [_; STATE_SIZE] = {
            let mut result_vec = Vec::with_capacity(STATE_SIZE);
            for row_idx in 0..STATE_SIZE {
                let mut current_accumulator = constant(0);
                for col_idx in 0..STATE_SIZE {
                    let mds_matrix_entry =
                        b_constant(HashTable::mds_matrix_entry(row_idx, col_idx));
                    let state_entry = state_after_s_box_application[col_idx].clone();
                    current_accumulator = current_accumulator + mds_matrix_entry * state_entry;
                }
                result_vec.push(current_accumulator);
            }
            result_vec.try_into().unwrap()
        };

        let round_constants: [_; STATE_SIZE] = [
            Constant0, Constant1, Constant2, Constant3, Constant4, Constant5, Constant6, Constant7,
            Constant8, Constant9, Constant10, Constant11, Constant12, Constant13, Constant14,
            Constant15,
        ]
        .map(current_base_row);

        let state_after_round_constant_addition: [_; STATE_SIZE] =
            state_after_matrix_multiplication
                .into_iter()
                .zip_eq(round_constants)
                .map(|(st, rndc)| st + rndc)
                .collect_vec()
                .try_into()
                .unwrap();

        let state_0_next = Self::re_compose_16_bit_limbs(
            circuit_builder,
            next_base_row(State0HighestLkIn),
            next_base_row(State0MidHighLkIn),
            next_base_row(State0MidLowLkIn),
            next_base_row(State0LowestLkIn),
        );
        let state_1_next = Self::re_compose_16_bit_limbs(
            circuit_builder,
            next_base_row(State1HighestLkIn),
            next_base_row(State1MidHighLkIn),
            next_base_row(State1MidLowLkIn),
            next_base_row(State1LowestLkIn),
        );
        let state_2_next = Self::re_compose_16_bit_limbs(
            circuit_builder,
            next_base_row(State2HighestLkIn),
            next_base_row(State2MidHighLkIn),
            next_base_row(State2MidLowLkIn),
            next_base_row(State2LowestLkIn),
        );
        let state_3_next = Self::re_compose_16_bit_limbs(
            circuit_builder,
            next_base_row(State3HighestLkIn),
            next_base_row(State3MidHighLkIn),
            next_base_row(State3MidLowLkIn),
            next_base_row(State3LowestLkIn),
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

        let round_number = current_base_row(RoundNumber);
        let hash_function_round_correctly_performs_update = state_after_round_constant_addition
            .into_iter()
            .zip_eq(state_next.clone().into_iter())
            .map(|(state_element, state_element_next)| {
                (round_number.clone() + constant(1))
                    * (round_number.clone() - constant(NUM_ROUNDS as u64))
                    * (state_element - state_element_next)
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
        let constant = |c: u64| circuit_builder.b_constant(c.into());
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

        let round_number_next_is_neg_1_or_5 = (round_number_next.clone() + constant(1))
            * (round_number_next.clone() - constant(NUM_ROUNDS as u64));
        let round_number_next_is_0_through_4 = round_number_next.clone()
            * (round_number_next.clone() - constant(1))
            * (round_number_next.clone() - constant(2))
            * (round_number_next.clone() - constant(3))
            * (round_number_next - constant(4));

        round_number_next_is_neg_1_or_5 * cascade_log_derivative_updates
            + round_number_next_is_0_through_4 * cascade_log_derivative_remains
    }

    fn all_cascade_log_derivative_update_circuits(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> [ConstraintCircuitMonad<DualRowIndicator>; 4 * NUM_SPLIT_AND_LOOKUP] {
        [
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
        ]
    }

    pub fn terminal_constraints(
        _circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        // no more constraints
        vec![]
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

    /// The round constants for round `round_number` if it is a valid round number in the Tip5
    /// permutation, and the zero vector otherwise.
    pub fn tip5_round_constants_by_round_number(
        round_number: usize,
    ) -> [BFieldElement; NUM_ROUND_CONSTANTS] {
        match round_number {
            i if i < NUM_ROUNDS => ROUND_CONSTANTS
                [NUM_ROUND_CONSTANTS * i..NUM_ROUND_CONSTANTS * (i + 1)]
                .try_into()
                .unwrap(),
            _ => [BFIELD_ZERO; NUM_ROUND_CONSTANTS],
        }
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
        let capital_r = BFieldElement::new(((1_u128 << 64) % BFieldElement::P as u128) as u64);
        let r_times_x = capital_r * x;
        [0, 16, 32, 48].map(|shift| ((r_times_x.value() >> shift) & 0xffff) as u16)
    }

    /// Given a trace of the Tip5 permutation, construct a trace corresponding to the columns of
    /// the Hash Table. This includes
    ///
    /// - adding the round number
    /// - adding the round constants,
    /// - decomposing the first [`NUM_SPLIT_AND_LOOKUP`] (== 4) state elements into their
    ///     constituent limbs,
    /// - setting the inverse-or-zero for proving correct limb decomposition, and
    /// - adding the looked-up value for each limb.
    ///
    /// The current instruction is not set.
    pub fn convert_to_hash_table_rows(
        hash_permutation_trace: [[BFieldElement; STATE_SIZE]; NUM_ROUNDS + 1],
    ) -> Array2<BFieldElement> {
        let mut hash_trace_addendum = Array2::zeros([NUM_ROUNDS + 1, BASE_WIDTH]);
        for (round_number, mut row) in hash_trace_addendum.rows_mut().into_iter().enumerate() {
            let trace_row = hash_permutation_trace[round_number];
            row[RoundNumber.base_table_index()] = BFieldElement::from(round_number as u64);

            let st_0_limbs = Self::base_field_element_into_16_bit_limbs(trace_row[0]);
            let st_0_look_in_split = st_0_limbs.map(|limb| BFieldElement::new(limb as u64));
            row[State0LowestLkIn.base_table_index()] = st_0_look_in_split[0];
            row[State0MidLowLkIn.base_table_index()] = st_0_look_in_split[1];
            row[State0MidHighLkIn.base_table_index()] = st_0_look_in_split[2];
            row[State0HighestLkIn.base_table_index()] = st_0_look_in_split[3];

            let st_0_look_out_split = st_0_limbs.map(CascadeTable::lookup_16_bit_limb);
            row[State0LowestLkOut.base_table_index()] = st_0_look_out_split[0];
            row[State0MidLowLkOut.base_table_index()] = st_0_look_out_split[1];
            row[State0MidHighLkOut.base_table_index()] = st_0_look_out_split[2];
            row[State0HighestLkOut.base_table_index()] = st_0_look_out_split[3];

            let st_1_limbs = Self::base_field_element_into_16_bit_limbs(trace_row[1]);
            let st_1_look_in_split = st_1_limbs.map(|limb| BFieldElement::new(limb as u64));
            row[State1LowestLkIn.base_table_index()] = st_1_look_in_split[0];
            row[State1MidLowLkIn.base_table_index()] = st_1_look_in_split[1];
            row[State1MidHighLkIn.base_table_index()] = st_1_look_in_split[2];
            row[State1HighestLkIn.base_table_index()] = st_1_look_in_split[3];

            let st_1_look_out_split = st_1_limbs.map(CascadeTable::lookup_16_bit_limb);
            row[State1LowestLkOut.base_table_index()] = st_1_look_out_split[0];
            row[State1MidLowLkOut.base_table_index()] = st_1_look_out_split[1];
            row[State1MidHighLkOut.base_table_index()] = st_1_look_out_split[2];
            row[State1HighestLkOut.base_table_index()] = st_1_look_out_split[3];

            let st_2_limbs = Self::base_field_element_into_16_bit_limbs(trace_row[2]);
            let st_2_look_in_split = st_2_limbs.map(|limb| BFieldElement::new(limb as u64));
            row[State2LowestLkIn.base_table_index()] = st_2_look_in_split[0];
            row[State2MidLowLkIn.base_table_index()] = st_2_look_in_split[1];
            row[State2MidHighLkIn.base_table_index()] = st_2_look_in_split[2];
            row[State2HighestLkIn.base_table_index()] = st_2_look_in_split[3];

            let st_2_look_out_split = st_2_limbs.map(CascadeTable::lookup_16_bit_limb);
            row[State2LowestLkOut.base_table_index()] = st_2_look_out_split[0];
            row[State2MidLowLkOut.base_table_index()] = st_2_look_out_split[1];
            row[State2MidHighLkOut.base_table_index()] = st_2_look_out_split[2];
            row[State2HighestLkOut.base_table_index()] = st_2_look_out_split[3];

            let st_3_limbs = Self::base_field_element_into_16_bit_limbs(trace_row[3]);
            let st_3_look_in_split = st_3_limbs.map(|limb| BFieldElement::new(limb as u64));
            row[State3LowestLkIn.base_table_index()] = st_3_look_in_split[0];
            row[State3MidLowLkIn.base_table_index()] = st_3_look_in_split[1];
            row[State3MidHighLkIn.base_table_index()] = st_3_look_in_split[2];
            row[State3HighestLkIn.base_table_index()] = st_3_look_in_split[3];

            let st_3_look_out_split = st_3_limbs.map(CascadeTable::lookup_16_bit_limb);
            row[State3LowestLkOut.base_table_index()] = st_3_look_out_split[0];
            row[State3MidLowLkOut.base_table_index()] = st_3_look_out_split[1];
            row[State3MidHighLkOut.base_table_index()] = st_3_look_out_split[2];
            row[State3HighestLkOut.base_table_index()] = st_3_look_out_split[3];

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

            row[State0Inv.base_table_index()] =
                Self::inverse_or_zero_of_highest_2_limbs(trace_row[0]);
            row[State1Inv.base_table_index()] =
                Self::inverse_or_zero_of_highest_2_limbs(trace_row[1]);
            row[State2Inv.base_table_index()] =
                Self::inverse_or_zero_of_highest_2_limbs(trace_row[2]);
            row[State3Inv.base_table_index()] =
                Self::inverse_or_zero_of_highest_2_limbs(trace_row[3]);

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
        }
        hash_trace_addendum
    }

    /// The inverse-or-zero of (2^32 - 1 - 2^16·`highest` - `mid_high`) where `highest`
    /// is the most significant limb of the given `state_element`, and `mid_high` the second-most
    /// significant limb.
    fn inverse_or_zero_of_highest_2_limbs(state_element: BFieldElement) -> BFieldElement {
        let limbs = Self::base_field_element_into_16_bit_limbs(state_element);
        let highest = limbs[3] as u64;
        let mid_high = limbs[2] as u64;
        let high_limbs = BFieldElement::new((highest << 16) + mid_high);
        let two_pow_32_minus_1 = BFieldElement::new((1 << 32) - 1);
        let to_invert = two_pow_32_minus_1 - high_limbs;
        to_invert.inverse_or_zero()
    }

    pub fn fill_trace(
        hash_table: &mut ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
    ) {
        let sponge_part_start = 0;
        let sponge_part_end = sponge_part_start + aet.sponge_trace.nrows();
        let hash_part_start = sponge_part_end;
        let hash_part_end = hash_part_start + aet.hash_trace.nrows();

        let sponge_part = hash_table.slice_mut(s![sponge_part_start..sponge_part_end, ..]);
        aet.sponge_trace.clone().move_into(sponge_part);
        let hash_part = hash_table.slice_mut(s![hash_part_start..hash_part_end, ..]);
        aet.hash_trace.clone().move_into(hash_part);
    }

    pub fn pad_trace(hash_table: &mut ArrayViewMut2<BFieldElement>, hash_table_length: usize) {
        hash_table
            .slice_mut(s![hash_table_length.., CI.base_table_index()])
            .fill(Instruction::Hash.opcode_b());
        hash_table
            .slice_mut(s![hash_table_length.., RoundNumber.base_table_index()])
            .fill(-BFIELD_ONE);

        let inverse_of_high_limbs_plus_1_minus_2_pow_32 =
            Self::inverse_or_zero_of_highest_2_limbs(BFIELD_ZERO);
        hash_table
            .slice_mut(s![hash_table_length.., State0Inv.base_table_index()])
            .fill(inverse_of_high_limbs_plus_1_minus_2_pow_32);
        hash_table
            .slice_mut(s![hash_table_length.., State1Inv.base_table_index()])
            .fill(inverse_of_high_limbs_plus_1_minus_2_pow_32);
        hash_table
            .slice_mut(s![hash_table_length.., State2Inv.base_table_index()])
            .fill(inverse_of_high_limbs_plus_1_minus_2_pow_32);
        hash_table
            .slice_mut(s![hash_table_length.., State3Inv.base_table_index()])
            .fill(inverse_of_high_limbs_plus_1_minus_2_pow_32);
    }

    pub fn extend(
        base_table: ArrayView2<BFieldElement>,
        mut ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());

        let ci_weight = challenges.get_challenge(HashCIWeight);
        let hash_digest_eval_indeterminate = challenges.get_challenge(HashDigestIndeterminate);
        let hash_input_eval_indeterminate = challenges.get_challenge(HashInputIndeterminate);
        let sponge_eval_indeterminate = challenges.get_challenge(SpongeIndeterminate);
        let cascade_indeterminate = challenges.get_challenge(HashCascadeLookupIndeterminate);

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

        let two_pow_16 = BFieldElement::from(1_u64 << 16);
        let two_pow_32 = BFieldElement::from(1_u64 << 32);
        let two_pow_48 = BFieldElement::from(1_u64 << 48);

        // Before decomposition into limbs, the state element is multiplied by R (capital_r).
        // After re-composition, we need to divide by R to get the original state element.
        // See `Self::base_field_element_16_bit_limbs` for more details.
        let capital_r = BFieldElement::new(((1_u128 << 64) % BFieldElement::P as u128) as u64);
        let capital_r_inv = capital_r.inverse();

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
                    * capital_r_inv
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

        let state_weights = [
            challenges.get_challenge(HashStateWeight0),
            challenges.get_challenge(HashStateWeight1),
            challenges.get_challenge(HashStateWeight2),
            challenges.get_challenge(HashStateWeight3),
            challenges.get_challenge(HashStateWeight4),
            challenges.get_challenge(HashStateWeight5),
            challenges.get_challenge(HashStateWeight6),
            challenges.get_challenge(HashStateWeight7),
            challenges.get_challenge(HashStateWeight8),
            challenges.get_challenge(HashStateWeight9),
        ];

        let cascade_look_in_weight = challenges.get_challenge(HashCascadeLookInWeight);
        let cascade_look_out_weight = challenges.get_challenge(HashCascadeLookOutWeight);

        let opcode_hash = Instruction::Hash.opcode_b();
        let opcode_absorb_init = Instruction::AbsorbInit.opcode_b();
        let opcode_absorb = Instruction::Absorb.opcode_b();
        let opcode_squeeze = Instruction::Squeeze.opcode_b();

        for row_idx in 0..base_table.nrows() {
            let row = base_table.row(row_idx);
            let current_instruction = row[CI.base_table_index()];
            let round_number = row[RoundNumber.base_table_index()];

            if round_number.value() == NUM_ROUNDS as u64 && current_instruction == opcode_hash {
                // add compressed digest to running evaluation “hash digest”
                let compressed_hash_digest: XFieldElement = rate_registers(row)[..DIGEST_LENGTH]
                    .iter()
                    .zip_eq(state_weights[..DIGEST_LENGTH].iter())
                    .map(|(&state, &weight)| weight * state)
                    .sum();
                hash_digest_running_evaluation = hash_digest_running_evaluation
                    * hash_digest_eval_indeterminate
                    + compressed_hash_digest;
            }

            if round_number.is_zero() {
                let compressed_row: XFieldElement = state_weights
                    .iter()
                    .zip_eq(rate_registers(row).iter())
                    .map(|(&weight, &element)| weight * element)
                    .sum();

                if current_instruction == opcode_hash {
                    hash_input_running_evaluation = hash_input_running_evaluation
                        * hash_input_eval_indeterminate
                        + compressed_row;
                }

                if current_instruction == opcode_absorb_init
                    || current_instruction == opcode_absorb
                    || current_instruction == opcode_squeeze
                {
                    sponge_running_evaluation = sponge_running_evaluation
                        * sponge_eval_indeterminate
                        + ci_weight * current_instruction
                        + compressed_row;
                }
            }

            if (0..NUM_ROUNDS as u64).contains(&round_number.value()) {
                cascade_state_0_highest_log_derivative += (cascade_indeterminate
                    - cascade_look_in_weight * row[State0HighestLkIn.base_table_index()]
                    - cascade_look_out_weight * row[State0HighestLkOut.base_table_index()])
                .inverse();
                cascade_state_0_mid_high_log_derivative += (cascade_indeterminate
                    - cascade_look_in_weight * row[State0MidHighLkIn.base_table_index()]
                    - cascade_look_out_weight * row[State0MidHighLkOut.base_table_index()])
                .inverse();
                cascade_state_0_mid_low_log_derivative += (cascade_indeterminate
                    - cascade_look_in_weight * row[State0MidLowLkIn.base_table_index()]
                    - cascade_look_out_weight * row[State0MidLowLkOut.base_table_index()])
                .inverse();
                cascade_state_0_lowest_log_derivative += (cascade_indeterminate
                    - cascade_look_in_weight * row[State0LowestLkIn.base_table_index()]
                    - cascade_look_out_weight * row[State0LowestLkOut.base_table_index()])
                .inverse();
                cascade_state_1_highest_log_derivative += (cascade_indeterminate
                    - cascade_look_in_weight * row[State1HighestLkIn.base_table_index()]
                    - cascade_look_out_weight * row[State1HighestLkOut.base_table_index()])
                .inverse();
                cascade_state_1_mid_high_log_derivative += (cascade_indeterminate
                    - cascade_look_in_weight * row[State1MidHighLkIn.base_table_index()]
                    - cascade_look_out_weight * row[State1MidHighLkOut.base_table_index()])
                .inverse();
                cascade_state_1_mid_low_log_derivative += (cascade_indeterminate
                    - cascade_look_in_weight * row[State1MidLowLkIn.base_table_index()]
                    - cascade_look_out_weight * row[State1MidLowLkOut.base_table_index()])
                .inverse();
                cascade_state_1_lowest_log_derivative += (cascade_indeterminate
                    - cascade_look_in_weight * row[State1LowestLkIn.base_table_index()]
                    - cascade_look_out_weight * row[State1LowestLkOut.base_table_index()])
                .inverse();
                cascade_state_2_highest_log_derivative += (cascade_indeterminate
                    - cascade_look_in_weight * row[State2HighestLkIn.base_table_index()]
                    - cascade_look_out_weight * row[State2HighestLkOut.base_table_index()])
                .inverse();
                cascade_state_2_mid_high_log_derivative += (cascade_indeterminate
                    - cascade_look_in_weight * row[State2MidHighLkIn.base_table_index()]
                    - cascade_look_out_weight * row[State2MidHighLkOut.base_table_index()])
                .inverse();
                cascade_state_2_mid_low_log_derivative += (cascade_indeterminate
                    - cascade_look_in_weight * row[State2MidLowLkIn.base_table_index()]
                    - cascade_look_out_weight * row[State2MidLowLkOut.base_table_index()])
                .inverse();
                cascade_state_2_lowest_log_derivative += (cascade_indeterminate
                    - cascade_look_in_weight * row[State2LowestLkIn.base_table_index()]
                    - cascade_look_out_weight * row[State2LowestLkOut.base_table_index()])
                .inverse();
                cascade_state_3_highest_log_derivative += (cascade_indeterminate
                    - cascade_look_in_weight * row[State3HighestLkIn.base_table_index()]
                    - cascade_look_out_weight * row[State3HighestLkOut.base_table_index()])
                .inverse();
                cascade_state_3_mid_high_log_derivative += (cascade_indeterminate
                    - cascade_look_in_weight * row[State3MidHighLkIn.base_table_index()]
                    - cascade_look_out_weight * row[State3MidHighLkOut.base_table_index()])
                .inverse();
                cascade_state_3_mid_low_log_derivative += (cascade_indeterminate
                    - cascade_look_in_weight * row[State3MidLowLkIn.base_table_index()]
                    - cascade_look_out_weight * row[State3MidLowLkOut.base_table_index()])
                .inverse();
                cascade_state_3_lowest_log_derivative += (cascade_indeterminate
                    - cascade_look_in_weight * row[State3LowestLkIn.base_table_index()]
                    - cascade_look_out_weight * row[State3LowestLkOut.base_table_index()])
                .inverse();
            }

            let mut extension_row = ext_table.row_mut(row_idx);
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
    }
}

#[cfg(test)]
mod constraint_tests {
    use num_traits::Zero;

    use crate::stark::triton_stark_tests::parse_simulate_pad_extend;
    use crate::table::extension_table::Evaluable;
    use crate::table::master_table::MasterTable;

    use super::*;

    #[test]
    fn hash_table_satisfies_constraints_test() {
        let source_code = "hash hash hash halt";
        let (_, _, _, master_base_table, master_ext_table, challenges) =
            parse_simulate_pad_extend(source_code, vec![], vec![]);
        assert_eq!(
            master_base_table.master_base_matrix.nrows(),
            master_ext_table.master_ext_matrix.nrows()
        );
        let master_base_trace_table = master_base_table.trace_table();
        let master_ext_trace_table = master_ext_table.trace_table();
        assert_eq!(
            master_base_trace_table.nrows(),
            master_ext_trace_table.nrows()
        );

        let num_rows = master_base_trace_table.nrows();
        let first_base_row = master_base_trace_table.row(0).map(|e| e.lift());
        let first_base_row = first_base_row.view();
        let first_ext_row = master_ext_trace_table.row(0);
        for (idx, v) in
            ExtHashTable::evaluate_initial_constraints(first_base_row, first_ext_row, &challenges)
                .iter()
                .enumerate()
        {
            assert!(v.is_zero(), "Initial constraint {idx} failed.");
        }

        for row_idx in 0..num_rows {
            let base_row = master_base_trace_table.row(row_idx).map(|e| e.lift());
            let base_row = base_row.view();
            let ext_row = master_ext_trace_table.row(row_idx);
            for (constraint_idx, v) in
                ExtHashTable::evaluate_consistency_constraints(base_row, ext_row, &challenges)
                    .iter()
                    .enumerate()
            {
                assert!(
                    v.is_zero(),
                    "consistency constraint {constraint_idx} failed in row {row_idx}"
                );
            }
        }

        for row_idx in 0..num_rows - 1 {
            let base_row = master_base_trace_table.row(row_idx).map(|e| e.lift());
            let base_row = base_row.view();
            let ext_row = master_ext_trace_table.row(row_idx);
            let next_base_row = master_base_trace_table.row(row_idx + 1).map(|e| e.lift());
            let next_base_row = next_base_row.view();
            let next_ext_row = master_ext_trace_table.row(row_idx + 1);
            for (constraint_idx, v) in ExtHashTable::evaluate_transition_constraints(
                base_row,
                ext_row,
                next_base_row,
                next_ext_row,
                &challenges,
            )
            .iter()
            .enumerate()
            {
                assert!(
                    v.is_zero(),
                    "transition constraint {constraint_idx} failed in row {row_idx}",
                );
            }
        }

        let last_base_row = master_base_trace_table.row(num_rows - 1).map(|e| e.lift());
        let last_base_row = last_base_row.view();
        let last_ext_row = master_ext_trace_table.row(num_rows - 1);
        for (idx, v) in
            ExtHashTable::evaluate_terminal_constraints(last_base_row, last_ext_row, &challenges)
                .iter()
                .enumerate()
        {
            assert!(v.is_zero(), "Terminal constraint {idx} failed.");
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub fn constraints_evaluate_to_zero(
        master_base_trace_table: ArrayView2<BFieldElement>,
        master_ext_trace_table: ArrayView2<XFieldElement>,
        challenges: &Challenges,
    ) -> bool {
        let zero = XFieldElement::zero();
        assert_eq!(
            master_base_trace_table.nrows(),
            master_ext_trace_table.nrows()
        );

        let circuit_builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in ExtHashTable::initial_constraints(&circuit_builder)
            .into_iter()
            .map(|constraint_monad| constraint_monad.consume())
            .enumerate()
        {
            let evaluated_constraint = constraint.evaluate(
                master_base_trace_table.slice(s![..1, ..]),
                master_ext_trace_table.slice(s![..1, ..]),
                challenges,
            );
            assert_eq!(
                zero, evaluated_constraint,
                "Initial constraint {constraint_idx} failed."
            );
        }

        let circuit_builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in ExtHashTable::consistency_constraints(&circuit_builder)
            .into_iter()
            .map(|constraint_monad| constraint_monad.consume())
            .enumerate()
        {
            for row_idx in 0..master_base_trace_table.nrows() {
                let evaluated_constraint = constraint.evaluate(
                    master_base_trace_table.slice(s![row_idx..row_idx + 1, ..]),
                    master_ext_trace_table.slice(s![row_idx..row_idx + 1, ..]),
                    challenges,
                );
                assert_eq!(
                    zero, evaluated_constraint,
                    "Consistency constraint {constraint_idx} failed on row {row_idx}."
                );
            }
        }

        let circuit_builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in ExtHashTable::transition_constraints(&circuit_builder)
            .into_iter()
            .map(|constraint_monad| constraint_monad.consume())
            .enumerate()
        {
            for row_idx in 0..master_base_trace_table.nrows() - 1 {
                let evaluated_constraint = constraint.evaluate(
                    master_base_trace_table.slice(s![row_idx..row_idx + 2, ..]),
                    master_ext_trace_table.slice(s![row_idx..row_idx + 2, ..]),
                    challenges,
                );
                assert_eq!(
                    zero, evaluated_constraint,
                    "Transition constraint {constraint_idx} failed on row {row_idx}."
                );
            }
        }

        let circuit_builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in ExtHashTable::terminal_constraints(&circuit_builder)
            .into_iter()
            .map(|constraint_monad| constraint_monad.consume())
            .enumerate()
        {
            let evaluated_constraint = constraint.evaluate(
                master_base_trace_table.slice(s![master_base_trace_table.nrows() - 1.., ..]),
                master_ext_trace_table.slice(s![master_ext_trace_table.nrows() - 1.., ..]),
                challenges,
            );
            assert_eq!(
                zero, evaluated_constraint,
                "Terminal constraint {constraint_idx} failed."
            );
        }

        true
    }
}
