use itertools::Itertools;
use ndarray::s;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use num_traits::One;
use strum::EnumCount;
use strum_macros::Display;
use strum_macros::EnumCount as EnumCountMacro;
use strum_macros::EnumIter;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::rescue_prime_regular::ALPHA;
use twenty_first::shared_math::rescue_prime_regular::DIGEST_LENGTH;
use twenty_first::shared_math::rescue_prime_regular::MDS;
use twenty_first::shared_math::rescue_prime_regular::MDS_INV;
use twenty_first::shared_math::rescue_prime_regular::NUM_ROUNDS;
use twenty_first::shared_math::rescue_prime_regular::ROUND_CONSTANTS;
use twenty_first::shared_math::rescue_prime_regular::STATE_SIZE;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::challenges::TableChallenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::ConstraintCircuitMonad;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::DualRowIndicator::*;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::*;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::EvalArg;
use crate::table::hash_table::HashTableChallengeId::*;
use crate::table::master_table::NUM_BASE_COLUMNS;
use crate::table::master_table::NUM_EXT_COLUMNS;
use crate::table::table_column::BaseTableColumn;
use crate::table::table_column::ExtTableColumn;
use crate::table::table_column::HashBaseTableColumn;
use crate::table::table_column::HashBaseTableColumn::*;
use crate::table::table_column::HashExtTableColumn;
use crate::table::table_column::HashExtTableColumn::*;
use crate::table::table_column::MasterBaseTableColumn;
use crate::table::table_column::MasterExtTableColumn;
use crate::vm::AlgebraicExecutionTrace;

pub const HASH_TABLE_NUM_PERMUTATION_ARGUMENTS: usize = 0;
pub const HASH_TABLE_NUM_EVALUATION_ARGUMENTS: usize = 2;
pub const HASH_TABLE_NUM_EXTENSION_CHALLENGES: usize = HashTableChallengeId::COUNT;

pub const BASE_WIDTH: usize = HashBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = HashExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

pub const NUM_ROUND_CONSTANTS: usize = STATE_SIZE * 2;
pub const TOTAL_NUM_CONSTANTS: usize = NUM_ROUND_CONSTANTS * NUM_ROUNDS;

#[derive(Debug, Clone)]
pub struct HashTable {}

#[derive(Debug, Clone)]
pub struct ExtHashTable {}

impl ExtHashTable {
    pub fn ext_initial_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            HashTableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let challenge = |c| circuit_builder.challenge(c);
        let one = circuit_builder.b_constant(1_u32.into());

        let running_evaluation_initial = circuit_builder.x_constant(EvalArg::default_initial());

        let round_number = circuit_builder.input(BaseRow(ROUNDNUMBER.master_base_table_index()));
        let running_evaluation_from_processor = circuit_builder.input(ExtRow(
            FromProcessorRunningEvaluation.master_ext_table_index(),
        ));
        let running_evaluation_to_processor = circuit_builder.input(ExtRow(
            ToProcessorRunningEvaluation.master_ext_table_index(),
        ));
        let state = [
            STATE0, STATE1, STATE2, STATE3, STATE4, STATE5, STATE6, STATE7, STATE8, STATE9,
        ]
        .map(|st| circuit_builder.input(BaseRow(st.master_base_table_index())));

        let round_number_is_0_or_1 = round_number.clone() * (round_number.clone() - one.clone());

        // Evaluation Argument “from processor”
        // If the round number is 0, the running evaluation is the default initial.
        // Else, the first update has been applied to the running evaluation.
        let running_evaluation_from_processor_is_default_initial =
            running_evaluation_from_processor.clone() - running_evaluation_initial.clone();
        let compressed_row = [
            challenge(StackInputWeight0),
            challenge(StackInputWeight1),
            challenge(StackInputWeight2),
            challenge(StackInputWeight3),
            challenge(StackInputWeight4),
            challenge(StackInputWeight5),
            challenge(StackInputWeight6),
            challenge(StackInputWeight7),
            challenge(StackInputWeight8),
            challenge(StackInputWeight9),
        ]
        .into_iter()
        .zip_eq(state.into_iter())
        .map(|(w, s)| s * w)
        .sum();
        let from_processor_indeterminate = challenge(FromProcessorEvalIndeterminate);
        let running_evaluation_from_processor_is_updated = running_evaluation_from_processor
            - running_evaluation_initial.clone() * from_processor_indeterminate
            - compressed_row;
        let running_evaluation_from_processor_is_updated_if_and_only_if_not_a_padding_row =
            round_number.clone() * running_evaluation_from_processor_is_updated
                + (one - round_number) * running_evaluation_from_processor_is_default_initial;

        // Evaluation Argument “to processor”
        let running_evaluation_to_processor_is_default_initial =
            running_evaluation_to_processor - running_evaluation_initial;

        [
            round_number_is_0_or_1,
            running_evaluation_from_processor_is_updated_if_and_only_if_not_a_padding_row,
            running_evaluation_to_processor_is_default_initial,
        ]
        .map(|circuit| circuit.consume())
        .to_vec()
    }

    pub fn ext_consistency_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            HashTableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let constant = |c: u64| circuit_builder.b_constant(c.into());

        let round_number = circuit_builder.input(BaseRow(ROUNDNUMBER.master_base_table_index()));
        let state10 = circuit_builder.input(BaseRow(STATE10.master_base_table_index()));
        let state11 = circuit_builder.input(BaseRow(STATE11.master_base_table_index()));
        let state12 = circuit_builder.input(BaseRow(STATE12.master_base_table_index()));
        let state13 = circuit_builder.input(BaseRow(STATE13.master_base_table_index()));
        let state14 = circuit_builder.input(BaseRow(STATE14.master_base_table_index()));
        let state15 = circuit_builder.input(BaseRow(STATE15.master_base_table_index()));

        let round_number_deselector = |round_number_to_deselect| {
            (0..=NUM_ROUNDS + 1)
                .filter(|&r| r != round_number_to_deselect)
                .map(|r| round_number.clone() - constant(r as u64))
                .fold(constant(1), |a, b| a * b)
        };

        let round_number_is_not_1_or = round_number_deselector(1);
        let mut consistency_constraint_circuits = vec![
            round_number_is_not_1_or.clone() * (state10 - constant(1)), // <-- domain separation bit
            round_number_is_not_1_or.clone() * state11,
            round_number_is_not_1_or.clone() * state12,
            round_number_is_not_1_or.clone() * state13,
            round_number_is_not_1_or.clone() * state14,
            round_number_is_not_1_or * state15,
        ];

        let round_constant_offset = CONSTANT0A.master_base_table_index();
        for round_constant_col_index in 0..NUM_ROUND_CONSTANTS {
            let round_constant_input =
                circuit_builder.input(BaseRow(round_constant_col_index + round_constant_offset));
            let round_constant_constraint_circuit = (1..=NUM_ROUNDS)
                .map(|i| {
                    let round_constant_idx =
                        NUM_ROUND_CONSTANTS * (i - 1) + round_constant_col_index;
                    let round_constant_needed = constant(ROUND_CONSTANTS[round_constant_idx]);
                    round_number_deselector(i)
                        * (round_constant_input.clone() - round_constant_needed)
                })
                .sum();
            consistency_constraint_circuits.push(round_constant_constraint_circuit);
        }

        consistency_constraint_circuits
            .into_iter()
            .map(|circuit| circuit.consume())
            .collect()
    }

    pub fn ext_transition_constraints_as_circuits() -> Vec<
        ConstraintCircuit<HashTableChallenges, DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>>,
    > {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let constant = |c: u64| circuit_builder.b_constant(c.into());

        let from_processor_eval_indeterminate =
            circuit_builder.challenge(FromProcessorEvalIndeterminate);
        let to_processor_eval_indeterminate =
            circuit_builder.challenge(ToProcessorEvalIndeterminate);

        let round_number =
            circuit_builder.input(CurrentBaseRow(ROUNDNUMBER.master_base_table_index()));
        let running_evaluation_from_processor = circuit_builder.input(CurrentExtRow(
            FromProcessorRunningEvaluation.master_ext_table_index(),
        ));
        let running_evaluation_to_processor = circuit_builder.input(CurrentExtRow(
            ToProcessorRunningEvaluation.master_ext_table_index(),
        ));

        let round_number_next =
            circuit_builder.input(NextBaseRow(ROUNDNUMBER.master_base_table_index()));
        let running_evaluation_from_processor_next = circuit_builder.input(NextExtRow(
            FromProcessorRunningEvaluation.master_ext_table_index(),
        ));
        let running_evaluation_to_processor_next = circuit_builder.input(NextExtRow(
            ToProcessorRunningEvaluation.master_ext_table_index(),
        ));

        // round number
        // round numbers evolve as
        // 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9, and
        // 9 -> 1 or 9 -> 0, and
        // 0 -> 0

        // if round number is 0, then next round number is 0
        // DNF: rn in {1, ..., 9} ∨ rn* = 0
        let round_number_is_1_through_9_or_round_number_next_is_0 = (1..=NUM_ROUNDS + 1)
            .map(|r| constant(r as u64) - round_number.clone())
            .fold(constant(1), |a, b| a * b)
            * round_number_next.clone();

        // if round number is 9, then next round number is 0 or 1
        // DNF: rn =/= 9 ∨ rn* = 0 ∨ rn* = 1
        let round_number_is_0_through_8_or_round_number_next_is_0_or_1 = (0..=NUM_ROUNDS)
            .map(|r| constant(r as u64) - round_number.clone())
            .fold(constant(1), |a, b| a * b)
            * (constant(1) - round_number_next.clone())
            * round_number_next.clone();

        // if round number is in {1, ..., 8} then next round number is +1
        // DNF: (rn == 0 ∨ rn == 9) ∨ rn* = rn + 1
        let round_number_is_0_or_9_or_increments_by_one = round_number.clone()
            * (constant(NUM_ROUNDS as u64 + 1) - round_number.clone())
            * (round_number_next.clone() - round_number.clone() - constant(1));

        // Rescue-XLIX

        let round_constants_a: [_; STATE_SIZE] = [
            CONSTANT0A,
            CONSTANT1A,
            CONSTANT2A,
            CONSTANT3A,
            CONSTANT4A,
            CONSTANT5A,
            CONSTANT6A,
            CONSTANT7A,
            CONSTANT8A,
            CONSTANT9A,
            CONSTANT10A,
            CONSTANT11A,
            CONSTANT12A,
            CONSTANT13A,
            CONSTANT14A,
            CONSTANT15A,
        ]
        .map(|c| circuit_builder.input(CurrentBaseRow(c.master_base_table_index())));
        let round_constants_b: [_; STATE_SIZE] = [
            CONSTANT0B,
            CONSTANT1B,
            CONSTANT2B,
            CONSTANT3B,
            CONSTANT4B,
            CONSTANT5B,
            CONSTANT6B,
            CONSTANT7B,
            CONSTANT8B,
            CONSTANT9B,
            CONSTANT10B,
            CONSTANT11B,
            CONSTANT12B,
            CONSTANT13B,
            CONSTANT14B,
            CONSTANT15B,
        ]
        .map(|c| circuit_builder.input(CurrentBaseRow(c.master_base_table_index())));

        let state: [_; STATE_SIZE] = [
            STATE0, STATE1, STATE2, STATE3, STATE4, STATE5, STATE6, STATE7, STATE8, STATE9,
            STATE10, STATE11, STATE12, STATE13, STATE14, STATE15,
        ];
        let current_state =
            state.map(|s| circuit_builder.input(CurrentBaseRow(s.master_base_table_index())));
        let next_state =
            state.map(|s| circuit_builder.input(NextBaseRow(s.master_base_table_index())));

        // left-hand-side, starting at current round and going forward

        let after_sbox = {
            let mut exponentiation_accumulator = current_state.to_vec();
            for _ in 1..ALPHA {
                for i in 0..exponentiation_accumulator.len() {
                    exponentiation_accumulator[i] =
                        exponentiation_accumulator[i].clone() * current_state[i].clone();
                }
            }
            exponentiation_accumulator
        };
        let after_mds = (0..STATE_SIZE)
            .map(|i| {
                (0..STATE_SIZE)
                    .map(|j| constant(MDS[i * STATE_SIZE + j]) * after_sbox[j].clone())
                    .sum::<ConstraintCircuitMonad<_, _>>()
            })
            .collect_vec();

        let after_constants = after_mds
            .into_iter()
            .zip_eq(round_constants_a)
            .map(|(st, rndc)| st + rndc)
            .collect_vec();

        // right hand side; move backwards
        let before_constants = next_state
            .clone()
            .into_iter()
            .zip_eq(round_constants_b)
            .map(|(st, rndc)| st - rndc)
            .collect_vec();
        let before_mds = (0..STATE_SIZE)
            .map(|i| {
                (0..STATE_SIZE)
                    .map(|j| constant(MDS_INV[i * STATE_SIZE + j]) * before_constants[j].clone())
                    .sum::<ConstraintCircuitMonad<_, _>>()
            })
            .collect_vec();

        let before_sbox = {
            let mut exponentiation_accumulator = before_mds.clone();
            for _ in 1..ALPHA {
                for i in 0..exponentiation_accumulator.len() {
                    exponentiation_accumulator[i] =
                        exponentiation_accumulator[i].clone() * before_mds[i].clone();
                }
            }
            exponentiation_accumulator
        };

        // Equate left hand side to right hand side. Ignore if padding row or after final round.

        let hash_function_round_correctly_performs_update = after_constants
            .into_iter()
            .zip_eq(before_sbox.into_iter())
            .map(|(lhs, rhs)| {
                round_number.clone()
                    * (round_number.clone() - constant(NUM_ROUNDS as u64 + 1))
                    * (lhs - rhs)
            })
            .collect_vec();

        // Evaluation Arguments

        // from Processor Table to Hash Table
        // If (and only if) the next row number is 1, update running evaluation “from processor.”
        let running_evaluation_from_processor_remains = running_evaluation_from_processor_next
            .clone()
            - running_evaluation_from_processor.clone();
        let xlix_input = next_state[0..2 * DIGEST_LENGTH].to_owned();
        let stack_input_weights = [
            StackInputWeight0,
            StackInputWeight1,
            StackInputWeight2,
            StackInputWeight3,
            StackInputWeight4,
            StackInputWeight5,
            StackInputWeight6,
            StackInputWeight7,
            StackInputWeight8,
            StackInputWeight9,
        ]
        .map(|w| circuit_builder.challenge(w));
        let compressed_row_from_processor = xlix_input
            .into_iter()
            .zip_eq(stack_input_weights.into_iter())
            .map(|(state, weight)| weight * state)
            .sum();

        let running_evaluation_from_processor_updates = running_evaluation_from_processor_next
            - from_processor_eval_indeterminate * running_evaluation_from_processor
            - compressed_row_from_processor;
        let round_number_next_unequal_1 = (0..=NUM_ROUNDS + 1)
            .filter(|&r| r != 1)
            .map(|r| round_number_next.clone() - constant(r as u64))
            .fold(constant(1), |a, b| a * b);
        let running_evaluation_from_processor_is_updated_correctly =
            running_evaluation_from_processor_remains * (round_number_next.clone() - constant(1))
                + running_evaluation_from_processor_updates * round_number_next_unequal_1;

        // from Hash Table to Processor Table
        // If (and only if) the next row number is 9, update running evaluation “to processor.”
        let running_evaluation_to_processor_remains =
            running_evaluation_to_processor_next.clone() - running_evaluation_to_processor.clone();
        let xlix_digest = next_state[0..DIGEST_LENGTH].to_owned();
        let digest_output_weights = [
            DigestOutputWeight0,
            DigestOutputWeight1,
            DigestOutputWeight2,
            DigestOutputWeight3,
            DigestOutputWeight4,
        ]
        .map(|w| circuit_builder.challenge(w));
        let compressed_row_to_processor = xlix_digest
            .into_iter()
            .zip_eq(digest_output_weights.into_iter())
            .map(|(state, weight)| weight * state)
            .sum();
        let running_evaluation_to_processor_updates = running_evaluation_to_processor_next
            - to_processor_eval_indeterminate * running_evaluation_to_processor
            - compressed_row_to_processor;
        let round_number_next_leq_number_of_rounds = (0..=NUM_ROUNDS)
            .map(|r| round_number_next.clone() - constant(r as u64))
            .fold(constant(1), |a, b| a * b);
        let running_evaluation_to_processor_is_updated_correctly =
            running_evaluation_to_processor_remains
                * (round_number_next - constant(NUM_ROUNDS as u64 + 1))
                + running_evaluation_to_processor_updates * round_number_next_leq_number_of_rounds;

        [
            vec![
                round_number_is_1_through_9_or_round_number_next_is_0,
                round_number_is_0_through_8_or_round_number_next_is_0_or_1,
                round_number_is_0_or_9_or_increments_by_one,
            ],
            hash_function_round_correctly_performs_update,
            vec![
                running_evaluation_from_processor_is_updated_correctly,
                running_evaluation_to_processor_is_updated_correctly,
            ],
        ]
        .concat()
        .into_iter()
        .map(|circuit| circuit.consume())
        .collect()
    }

    pub fn ext_terminal_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            HashTableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        // no more constraints
        vec![]
    }
}

impl HashTable {
    pub fn fill_trace(
        hash_table: &mut ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
    ) {
        let hash_table_to_fill = hash_table.slice_mut(s![0..aet.hash_matrix.nrows(), ..]);
        aet.hash_matrix.clone().move_into(hash_table_to_fill);
    }

    pub fn pad_trace(_hash_table: &mut ArrayViewMut2<BFieldElement>) {
        // Hash Table is padded with all-zero rows. It is also initialized with all-zero rows.
        // Hence, no need to do anything.
    }

    pub fn extend(
        base_table: ArrayView2<BFieldElement>,
        mut ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &HashTableChallenges,
    ) {
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());
        let mut from_processor_running_evaluation = EvalArg::default_initial();
        let mut to_processor_running_evaluation = EvalArg::default_initial();

        for row_idx in 0..base_table.nrows() {
            let current_row = base_table.row(row_idx);

            // Add compressed input to running evaluation if round index marks beginning of hashing
            if current_row[ROUNDNUMBER.base_table_index()].is_one() {
                let state_for_input = [
                    current_row[STATE0.base_table_index()],
                    current_row[STATE1.base_table_index()],
                    current_row[STATE2.base_table_index()],
                    current_row[STATE3.base_table_index()],
                    current_row[STATE4.base_table_index()],
                    current_row[STATE5.base_table_index()],
                    current_row[STATE6.base_table_index()],
                    current_row[STATE7.base_table_index()],
                    current_row[STATE8.base_table_index()],
                    current_row[STATE9.base_table_index()],
                ];
                let stack_input_weights = [
                    challenges.stack_input_weight0,
                    challenges.stack_input_weight1,
                    challenges.stack_input_weight2,
                    challenges.stack_input_weight3,
                    challenges.stack_input_weight4,
                    challenges.stack_input_weight5,
                    challenges.stack_input_weight6,
                    challenges.stack_input_weight7,
                    challenges.stack_input_weight8,
                    challenges.stack_input_weight9,
                ];
                let compressed_state_for_input: XFieldElement = state_for_input
                    .iter()
                    .zip_eq(stack_input_weights.iter())
                    .map(|(&state, &weight)| weight * state)
                    .sum();
                from_processor_running_evaluation = from_processor_running_evaluation
                    * challenges.from_processor_eval_indeterminate
                    + compressed_state_for_input;
            }

            // Add compressed digest to running evaluation if round index marks end of hashing
            if current_row[ROUNDNUMBER.base_table_index()].value() == NUM_ROUNDS as u64 + 1 {
                let state_for_output = [
                    current_row[STATE0.base_table_index()],
                    current_row[STATE1.base_table_index()],
                    current_row[STATE2.base_table_index()],
                    current_row[STATE3.base_table_index()],
                    current_row[STATE4.base_table_index()],
                ];
                let digest_output_weights = [
                    challenges.digest_output_weight0,
                    challenges.digest_output_weight1,
                    challenges.digest_output_weight2,
                    challenges.digest_output_weight3,
                    challenges.digest_output_weight4,
                ];
                let compressed_state_for_output: XFieldElement = state_for_output
                    .iter()
                    .zip_eq(digest_output_weights.iter())
                    .map(|(&state, &weight)| weight * state)
                    .sum();
                to_processor_running_evaluation = to_processor_running_evaluation
                    * challenges.to_processor_eval_indeterminate
                    + compressed_state_for_output;
            }

            let mut extension_row = ext_table.row_mut(row_idx);
            extension_row[FromProcessorRunningEvaluation.ext_table_index()] =
                from_processor_running_evaluation;
            extension_row[ToProcessorRunningEvaluation.ext_table_index()] =
                to_processor_running_evaluation;
        }
    }
}

#[derive(Debug, Copy, Clone, Display, EnumCountMacro, EnumIter, PartialEq, Eq, Hash)]
pub enum HashTableChallengeId {
    FromProcessorEvalIndeterminate,
    ToProcessorEvalIndeterminate,

    StackInputWeight0,
    StackInputWeight1,
    StackInputWeight2,
    StackInputWeight3,
    StackInputWeight4,
    StackInputWeight5,
    StackInputWeight6,
    StackInputWeight7,
    StackInputWeight8,
    StackInputWeight9,

    DigestOutputWeight0,
    DigestOutputWeight1,
    DigestOutputWeight2,
    DigestOutputWeight3,
    DigestOutputWeight4,
}

impl From<HashTableChallengeId> for usize {
    fn from(val: HashTableChallengeId) -> Self {
        val as usize
    }
}

#[derive(Debug, Clone)]
pub struct HashTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the hash table.
    pub from_processor_eval_indeterminate: XFieldElement,
    pub to_processor_eval_indeterminate: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    // There are 2 * DIGEST_LENGTH elements of these
    pub stack_input_weight0: XFieldElement,
    pub stack_input_weight1: XFieldElement,
    pub stack_input_weight2: XFieldElement,
    pub stack_input_weight3: XFieldElement,
    pub stack_input_weight4: XFieldElement,
    pub stack_input_weight5: XFieldElement,
    pub stack_input_weight6: XFieldElement,
    pub stack_input_weight7: XFieldElement,
    pub stack_input_weight8: XFieldElement,
    pub stack_input_weight9: XFieldElement,

    // There are DIGEST_LENGTH elements of these
    pub digest_output_weight0: XFieldElement,
    pub digest_output_weight1: XFieldElement,
    pub digest_output_weight2: XFieldElement,
    pub digest_output_weight3: XFieldElement,
    pub digest_output_weight4: XFieldElement,
}

impl TableChallenges for HashTableChallenges {
    type Id = HashTableChallengeId;

    #[inline]
    fn get_challenge(&self, id: Self::Id) -> XFieldElement {
        match id {
            FromProcessorEvalIndeterminate => self.from_processor_eval_indeterminate,
            ToProcessorEvalIndeterminate => self.to_processor_eval_indeterminate,
            StackInputWeight0 => self.stack_input_weight0,
            StackInputWeight1 => self.stack_input_weight1,
            StackInputWeight2 => self.stack_input_weight2,
            StackInputWeight3 => self.stack_input_weight3,
            StackInputWeight4 => self.stack_input_weight4,
            StackInputWeight5 => self.stack_input_weight5,
            StackInputWeight6 => self.stack_input_weight6,
            StackInputWeight7 => self.stack_input_weight7,
            StackInputWeight8 => self.stack_input_weight8,
            StackInputWeight9 => self.stack_input_weight9,
            DigestOutputWeight0 => self.digest_output_weight0,
            DigestOutputWeight1 => self.digest_output_weight1,
            DigestOutputWeight2 => self.digest_output_weight2,
            DigestOutputWeight3 => self.digest_output_weight3,
            DigestOutputWeight4 => self.digest_output_weight4,
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
