use itertools::Itertools;
use ndarray::s;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use num_traits::One;
use strum::EnumCount;
use triton_opcodes::instruction::Instruction;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::b_field_element::BFIELD_ONE;
use twenty_first::shared_math::tip5::DIGEST_LENGTH;
use twenty_first::shared_math::tip5::MDS_MATRIX_FIRST_COLUMN;
use twenty_first::shared_math::tip5::NUM_ROUNDS;
use twenty_first::shared_math::tip5::NUM_SPLIT_AND_LOOKUP;
use twenty_first::shared_math::tip5::RATE;
use twenty_first::shared_math::tip5::ROUND_CONSTANTS;
use twenty_first::shared_math::tip5::STATE_SIZE;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::challenges::ChallengeId::*;
use crate::table::challenges::Challenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::ConstraintCircuitMonad;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::DualRowIndicator::*;
use crate::table::constraint_circuit::InputIndicator;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::*;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::EvalArg;
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
    pub fn ext_initial_constraints_as_circuits() -> Vec<ConstraintCircuit<SingleRowIndicator>> {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let challenge = |c| circuit_builder.challenge(c);
        let constant = |c| circuit_builder.b_constant(c);
        let one = constant(BFIELD_ONE);

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

        let two_pow_16 = constant(BFieldElement::new(1_u64 << 16));
        let two_pow_32 = constant(BFieldElement::new(1_u64 << 32));
        let two_pow_48 = constant(BFieldElement::new(1_u64 << 48));

        let state_0 = base_row(State0HighestLkIn) * two_pow_48.clone()
            + base_row(State0MidHighLkIn) * two_pow_32.clone()
            + base_row(State0MidLowLkIn) * two_pow_16.clone()
            + base_row(State0LowestLkIn);
        let state_1 = base_row(State1HighestLkIn) * two_pow_48.clone()
            + base_row(State1MidHighLkIn) * two_pow_32.clone()
            + base_row(State1MidLowLkIn) * two_pow_16.clone()
            + base_row(State1LowestLkIn);
        let state_2 = base_row(State2HighestLkIn) * two_pow_48.clone()
            + base_row(State2MidHighLkIn) * two_pow_32.clone()
            + base_row(State2MidLowLkIn) * two_pow_16.clone()
            + base_row(State2LowestLkIn);
        let state_3 = base_row(State3HighestLkIn) * two_pow_48.clone()
            + base_row(State3MidHighLkIn) * two_pow_32.clone()
            + base_row(State3MidLowLkIn) * two_pow_16.clone()
            + base_row(State3LowestLkIn);

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

        let ci_is_hash = ci.clone() - constant(Instruction::Hash.opcode_b());
        let ci_is_absorb_init = ci - constant(Instruction::AbsorbInit.opcode_b());
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
        let running_evaluation_hash_input_is_initialized_correctly = (round_number.clone()
            + one.clone())
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
            - challenge(HashCIWeight) * constant(Instruction::AbsorbInit.opcode_b())
            - compressed_row;
        let running_evaluation_sponge_absorb_is_initialized_correctly = ci_is_hash
            * running_evaluation_sponge_has_accumulated_first_row
            + ci_is_absorb_init * running_evaluation_sponge_is_default_initial;

        let mut constraints = [
            round_number_is_neg_1_or_0,
            current_instruction_is_absorb_init_or_hash,
            running_evaluation_hash_input_is_initialized_correctly,
            running_evaluation_hash_digest_is_default_initial,
            running_evaluation_sponge_absorb_is_initialized_correctly,
        ];

        ConstraintCircuitMonad::constant_folding(&mut constraints);
        constraints.map(|circuit| circuit.consume()).to_vec()
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
        let constant = |c: u64| circuit_builder.b_constant(c.into());
        (-1..=NUM_ROUNDS as isize)
            .filter(|&r| r != round_number_to_deselect)
            .map(|r| round_number_circuit_node.clone() - constant(r as u64))
            .fold(constant(1), |a, b| a * b)
    }

    pub fn ext_consistency_constraints_as_circuits() -> Vec<ConstraintCircuit<SingleRowIndicator>> {
        let circuit_builder = ConstraintCircuitBuilder::new();
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
            Self::round_number_deselector(&circuit_builder, &round_number, -1);
        let round_number_is_not_0 =
            Self::round_number_deselector(&circuit_builder, &round_number, 0);

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
            if_ci_is_hash_and_round_no_is_0_then_.clone() * (state15.clone() - constant(1));

        let if_ci_is_absorb_init_and_round_no_is_0_then_ = round_number_is_not_0.clone()
            * ci_is_hash
            * ci_is_absorb.clone()
            * ci_is_squeeze.clone();
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
            if_ci_is_absorb_init_and_round_no_is_0_then_.clone() * state15;

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
                    &circuit_builder,
                    &round_number,
                    round_idx as isize,
                );
                round_constant_constraint_circuit = round_constant_constraint_circuit
                    + round_deselector_circuit
                        * (round_constant_column_circuit.clone() - round_constant_for_current_row);
            }
            constraints.push(round_constant_constraint_circuit);
        }

        ConstraintCircuitMonad::constant_folding(&mut constraints);
        constraints
            .into_iter()
            .map(|circuit| circuit.consume())
            .collect()
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

    pub fn ext_transition_constraints_as_circuits() -> Vec<ConstraintCircuit<DualRowIndicator>> {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let challenge = |c| circuit_builder.challenge(c);
        let constant = |c: u64| circuit_builder.b_constant(c.into());

        let opcode_hash = constant(Instruction::Hash.opcode() as u64);
        let opcode_absorb_init = constant(Instruction::AbsorbInit.opcode() as u64);
        let opcode_absorb = constant(Instruction::Absorb.opcode() as u64);
        let opcode_squeeze = constant(Instruction::Squeeze.opcode() as u64);

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

        let two_pow_16 = constant(1 << 16);
        let two_pow_32 = constant(1 << 32);
        let two_pow_48 = constant(1 << 48);

        let state_0 = current_base_row(State0HighestLkIn) * two_pow_48.clone()
            + current_base_row(State0MidHighLkIn) * two_pow_32.clone()
            + current_base_row(State0MidLowLkIn) * two_pow_16.clone()
            + current_base_row(State0LowestLkIn);
        let state_1 = current_base_row(State1HighestLkIn) * two_pow_48.clone()
            + current_base_row(State1MidHighLkIn) * two_pow_32.clone()
            + current_base_row(State1MidLowLkIn) * two_pow_16.clone()
            + current_base_row(State1LowestLkIn);
        let state_2 = current_base_row(State2HighestLkIn) * two_pow_48.clone()
            + current_base_row(State2MidHighLkIn) * two_pow_32.clone()
            + current_base_row(State2MidLowLkIn) * two_pow_16.clone()
            + current_base_row(State2LowestLkIn);
        let state_3 = current_base_row(State3HighestLkIn) * two_pow_48.clone()
            + current_base_row(State3MidHighLkIn) * two_pow_32.clone()
            + current_base_row(State3MidLowLkIn) * two_pow_16.clone()
            + current_base_row(State3LowestLkIn);

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
            Self::tip5_constraints_as_circuits(&circuit_builder);

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
            Self::round_number_deselector(&circuit_builder, &round_number, -1);
        let round_number_is_not_5 =
            Self::round_number_deselector(&circuit_builder, &round_number, 5);

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
            (round_number.clone() - constant(NUM_ROUNDS as u64)) * (ci_next.clone() - ci);

        // copy capacity between rounds with index 5 and 0 if instruction is “absorb”
        let round_number_next_is_not_0 =
            Self::round_number_deselector(&circuit_builder, &round_number_next, 0);
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
        let round_number_next_is_not_5 = Self::round_number_deselector(
            &circuit_builder,
            &round_number_next,
            NUM_ROUNDS as isize,
        );
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
            - sponge_indeterminate.clone() * running_evaluation_sponge.clone()
            - challenge(HashCIWeight) * ci_next.clone()
            - compressed_row_next;
        let if_round_no_next_0_and_ci_next_is_spongy_then_running_eval_sponge_updates =
            round_number_next_is_not_0.clone()
                * (ci_next.clone() - opcode_hash.clone())
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

        let mut constraints = [
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
        ]
        .concat();

        ConstraintCircuitMonad::constant_folding(&mut constraints);
        constraints
            .into_iter()
            .map(|circuit| circuit.consume())
            .collect()
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

        let two_pow_16 = constant(1 << 16);
        let two_pow_32 = constant(1 << 32);
        let two_pow_48 = constant(1 << 48);

        let state_0_after_lookup = current_base_row(State0HighestLkOut) * two_pow_48.clone()
            + current_base_row(State0MidHighLkOut) * two_pow_32.clone()
            + current_base_row(State0MidLowLkOut) * two_pow_16.clone()
            + current_base_row(State0LowestLkOut);
        let state_1_after_lookup = current_base_row(State1HighestLkOut) * two_pow_48.clone()
            + current_base_row(State1MidHighLkOut) * two_pow_32.clone()
            + current_base_row(State1MidLowLkOut) * two_pow_16.clone()
            + current_base_row(State1LowestLkOut);
        let state_2_after_lookup = current_base_row(State2HighestLkOut) * two_pow_48.clone()
            + current_base_row(State2MidHighLkOut) * two_pow_32.clone()
            + current_base_row(State2MidLowLkOut) * two_pow_16.clone()
            + current_base_row(State2LowestLkOut);
        let state_3_after_lookup = current_base_row(State3HighestLkOut) * two_pow_48.clone()
            + current_base_row(State3MidHighLkOut) * two_pow_32.clone()
            + current_base_row(State3MidLowLkOut) * two_pow_16.clone()
            + current_base_row(State3LowestLkOut);

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

        let state_0_next = next_base_row(State0HighestLkIn) * two_pow_48.clone()
            + next_base_row(State0MidHighLkIn) * two_pow_32.clone()
            + next_base_row(State0MidLowLkIn) * two_pow_16.clone()
            + next_base_row(State0LowestLkIn);
        let state_1_next = next_base_row(State1HighestLkIn) * two_pow_48.clone()
            + next_base_row(State1MidHighLkIn) * two_pow_32.clone()
            + next_base_row(State1MidLowLkIn) * two_pow_16.clone()
            + next_base_row(State1LowestLkIn);
        let state_2_next = next_base_row(State2HighestLkIn) * two_pow_48.clone()
            + next_base_row(State2MidHighLkIn) * two_pow_32.clone()
            + next_base_row(State2MidLowLkIn) * two_pow_16.clone()
            + next_base_row(State2LowestLkIn);
        let state_3_next = next_base_row(State3HighestLkIn) * two_pow_48
            + next_base_row(State3MidHighLkIn) * two_pow_32
            + next_base_row(State3MidLowLkIn) * two_pow_16
            + next_base_row(State3LowestLkIn);

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

    pub fn ext_terminal_constraints_as_circuits() -> Vec<ConstraintCircuit<SingleRowIndicator>> {
        // no more constraints
        vec![]
    }
}

impl HashTable {
    /// Get the MDS matrix's entry in row `row_idx` and column `col_idx`.
    pub const fn mds_matrix_entry(row_idx: usize, col_idx: usize) -> BFieldElement {
        assert!(row_idx < STATE_SIZE);
        assert!(col_idx < STATE_SIZE);
        let index_in_matrix_defining_column = (row_idx - col_idx) % STATE_SIZE;
        let mds_matrix_entry = MDS_MATRIX_FIRST_COLUMN[index_in_matrix_defining_column];
        BFieldElement::new(mds_matrix_entry as u64)
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

        let mut hash_input_running_evaluation = EvalArg::default_initial();
        let mut hash_digest_running_evaluation = EvalArg::default_initial();
        let mut sponge_running_evaluation = EvalArg::default_initial();

        let two_pow_16 = BFieldElement::from(1_u64 << 16);
        let two_pow_32 = BFieldElement::from(1_u64 << 32);
        let two_pow_48 = BFieldElement::from(1_u64 << 48);
        let rate_registers = |row: ArrayView1<BFieldElement>| {
            [
                row[State0HighestLkIn.base_table_index()] * two_pow_48
                    + row[State0MidHighLkIn.base_table_index()] * two_pow_32
                    + row[State0MidLowLkIn.base_table_index()] * two_pow_16
                    + row[State0LowestLkIn.base_table_index()],
                row[State1HighestLkIn.base_table_index()] * two_pow_48
                    + row[State1MidHighLkIn.base_table_index()] * two_pow_32
                    + row[State1MidLowLkIn.base_table_index()] * two_pow_16
                    + row[State1LowestLkIn.base_table_index()],
                row[State2HighestLkIn.base_table_index()] * two_pow_48
                    + row[State2MidHighLkIn.base_table_index()] * two_pow_32
                    + row[State2MidLowLkIn.base_table_index()] * two_pow_16
                    + row[State2LowestLkIn.base_table_index()],
                row[State3HighestLkIn.base_table_index()] * two_pow_48
                    + row[State3MidHighLkIn.base_table_index()] * two_pow_32
                    + row[State3MidLowLkIn.base_table_index()] * two_pow_16
                    + row[State3LowestLkIn.base_table_index()],
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

        let opcode_hash = Instruction::Hash.opcode_b();
        let opcode_absorb_init = Instruction::AbsorbInit.opcode_b();
        let opcode_absorb = Instruction::Absorb.opcode_b();
        let opcode_squeeze = Instruction::Squeeze.opcode_b();

        let previous_row = Array1::zeros([BASE_WIDTH]);
        let mut previous_row = previous_row.view();
        for row_idx in 0..base_table.nrows() {
            let current_row = base_table.row(row_idx);
            let current_instruction = current_row[CI.base_table_index()];

            if current_row[RoundNumber.base_table_index()].value() == NUM_ROUNDS as u64 + 1
                && current_instruction == opcode_hash
            {
                // add compressed digest to running evaluation “hash digest”
                let compressed_hash_digest: XFieldElement = rate_registers(current_row)
                    [..DIGEST_LENGTH]
                    .iter()
                    .zip_eq(state_weights[..DIGEST_LENGTH].iter())
                    .map(|(&state, &weight)| weight * state)
                    .sum();
                hash_digest_running_evaluation = hash_digest_running_evaluation
                    * hash_digest_eval_indeterminate
                    + compressed_hash_digest;
            }

            // all remaining Evaluation Arguments only get updated if the round number is 1
            if current_row[RoundNumber.base_table_index()].is_one() {
                let elements_for_hash_input_and_sponge_operations = match current_instruction {
                    op if op == opcode_hash || op == opcode_absorb_init || op == opcode_squeeze => {
                        rate_registers(current_row)
                    }
                    op if op == opcode_absorb => {
                        let rate_previous_row = rate_registers(previous_row);
                        let rate_current_row = rate_registers(current_row);
                        rate_current_row
                            .iter()
                            .zip_eq(rate_previous_row.iter())
                            .map(|(&current_state, &previous_state)| current_state - previous_state)
                            .collect_vec()
                            .try_into()
                            .unwrap()
                    }
                    _ => panic!("Opcode must be of `hash`, `absorb_init`, `absorb`, or `squeeze`."),
                };
                let compressed_row_hash_input_and_sponge_operations: XFieldElement = state_weights
                    .iter()
                    .zip_eq(elements_for_hash_input_and_sponge_operations.iter())
                    .map(|(&weight, &element)| weight * element)
                    .sum();

                match current_instruction {
                    ci if ci == opcode_hash => {
                        hash_input_running_evaluation = hash_input_running_evaluation
                            * hash_input_eval_indeterminate
                            + compressed_row_hash_input_and_sponge_operations;
                    }
                    ci if ci == opcode_absorb_init
                        || ci == opcode_absorb
                        || ci == opcode_squeeze =>
                    {
                        sponge_running_evaluation = sponge_running_evaluation
                            * sponge_eval_indeterminate
                            + ci_weight * ci
                            + compressed_row_hash_input_and_sponge_operations;
                    }
                    _ => panic!("Opcode must be of `hash`, `absorb_init`, `absorb`, or `squeeze`."),
                }
            }

            let mut extension_row = ext_table.row_mut(row_idx);
            extension_row[HashInputRunningEvaluation.ext_table_index()] =
                hash_input_running_evaluation;
            extension_row[HashDigestRunningEvaluation.ext_table_index()] =
                hash_digest_running_evaluation;
            extension_row[SpongeRunningEvaluation.ext_table_index()] = sponge_running_evaluation;

            previous_row = current_row;
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
        let (_, _, master_base_table, master_ext_table, challenges) =
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
        let first_base_row = master_base_trace_table.row(0);
        let first_ext_row = master_ext_trace_table.row(0);
        for (idx, v) in
            ExtHashTable::evaluate_initial_constraints(first_base_row, first_ext_row, &challenges)
                .iter()
                .enumerate()
        {
            assert!(v.is_zero(), "Initial constraint {idx} failed.");
        }

        for row_idx in 0..num_rows {
            let base_row = master_base_trace_table.row(row_idx);
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
            let base_row = master_base_trace_table.row(row_idx);
            let ext_row = master_ext_trace_table.row(row_idx);
            let next_base_row = master_base_trace_table.row(row_idx + 1);
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

        let last_base_row = master_base_trace_table.row(num_rows - 1);
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
