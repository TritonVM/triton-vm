use itertools::Itertools;
use num_traits::Zero;
use strum::EnumCount;
use strum_macros::{Display, EnumCount as EnumCountMacro, EnumIter};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::MPolynomial;
use twenty_first::shared_math::rescue_prime_regular::DIGEST_LENGTH;
use twenty_first::shared_math::rescue_prime_regular::{
    ALPHA, MDS, MDS_INV, NUM_ROUNDS, ROUND_CONSTANTS, STATE_SIZE,
};
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::arithmetic_domain::ArithmeticDomain;
use crate::cross_table_arguments::{CrossTableArg, EvalArg};
use crate::table::base_table::Extendable;
use crate::table::challenges::TableChallenges;
use crate::table::constraint_circuit::DualRowIndicator::{CurrentRow, NextRow};
use crate::table::constraint_circuit::SingleRowIndicator::Row;
use crate::table::constraint_circuit::{
    ConstraintCircuit, ConstraintCircuitBuilder, ConstraintCircuitMonad, DualRowIndicator,
    SingleRowIndicator,
};
use crate::table::hash_table::HashTableChallengeId::*;
use crate::table::table_column::HashBaseTableColumn::{self, *};
use crate::table::table_column::HashExtTableColumn::{self, *};

use super::base_table::{InheritsFromTable, Table, TableLike};
use super::challenges::AllChallenges;
use super::extension_table::{ExtensionTable, QuotientableExtensionTable};

pub const HASH_TABLE_NUM_PERMUTATION_ARGUMENTS: usize = 0;
pub const HASH_TABLE_NUM_EVALUATION_ARGUMENTS: usize = 2;

/// This is 15 because it combines: 10 stack_input_weights and 5 digest_output_weights.
pub const HASH_TABLE_NUM_EXTENSION_CHALLENGES: usize = 15;

pub const BASE_WIDTH: usize = HashBaseTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + HashExtTableColumn::COUNT;

pub const NUM_ROUND_CONSTANTS: usize = STATE_SIZE * 2;
pub const TOTAL_NUM_CONSTANTS: usize = NUM_ROUND_CONSTANTS * NUM_ROUNDS;

#[derive(Debug, Clone)]
pub struct HashTable {
    inherited_table: Table<BFieldElement>,
}

impl InheritsFromTable<BFieldElement> for HashTable {
    fn inherited_table(&self) -> &Table<BFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<BFieldElement> {
        &mut self.inherited_table
    }
}

#[derive(Debug, Clone)]
pub struct ExtHashTable {
    pub(crate) inherited_table: Table<XFieldElement>,
}

impl Default for ExtHashTable {
    fn default() -> Self {
        Self {
            inherited_table: Table::new(
                BASE_WIDTH,
                FULL_WIDTH,
                vec![],
                "EmptyExtHashTable".to_string(),
            ),
        }
    }
}

impl QuotientableExtensionTable for ExtHashTable {}

impl InheritsFromTable<XFieldElement> for ExtHashTable {
    fn inherited_table(&self) -> &Table<XFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<XFieldElement> {
        &mut self.inherited_table
    }
}

impl TableLike<BFieldElement> for HashTable {}

impl Extendable for HashTable {
    fn get_padding_rows(&self) -> (Option<usize>, Vec<Vec<BFieldElement>>) {
        (None, vec![vec![BFieldElement::zero(); BASE_WIDTH]])
    }
}

impl TableLike<XFieldElement> for ExtHashTable {}

impl ExtHashTable {
    pub fn ext_initial_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<HashTableChallenges, SingleRowIndicator<FULL_WIDTH>>> {
        let circuit_builder = ConstraintCircuitBuilder::new(FULL_WIDTH);
        let challenge = |c| circuit_builder.challenge(c);
        let one = circuit_builder.b_constant(1_u32.into());

        let running_evaluation_initial = circuit_builder.x_constant(EvalArg::default_initial());

        let round_number = circuit_builder.input(Row(ROUNDNUMBER.into()));
        let running_evaluation_from_processor =
            circuit_builder.input(Row(FromProcessorRunningEvaluation.into()));
        let running_evaluation_to_processor =
            circuit_builder.input(Row(ToProcessorRunningEvaluation.into()));
        let state = [
            STATE0, STATE1, STATE2, STATE3, STATE4, STATE5, STATE6, STATE7, STATE8, STATE9,
        ]
        .map(|st| circuit_builder.input(Row(st.into())));

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

    pub fn ext_consistency_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<HashTableChallenges, SingleRowIndicator<FULL_WIDTH>>> {
        let circuit_builder = ConstraintCircuitBuilder::new(FULL_WIDTH);
        let constant = |c: u64| circuit_builder.b_constant(c.into());

        let round_number = circuit_builder.input(Row(ROUNDNUMBER.into()));
        let state10 = circuit_builder.input(Row(STATE10.into()));
        let state11 = circuit_builder.input(Row(STATE11.into()));
        let state12 = circuit_builder.input(Row(STATE12.into()));
        let state13 = circuit_builder.input(Row(STATE13.into()));
        let state14 = circuit_builder.input(Row(STATE14.into()));
        let state15 = circuit_builder.input(Row(STATE15.into()));

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

        let round_constant_offset: usize = CONSTANT0A.into();
        for round_constant_col_index in 0..NUM_ROUND_CONSTANTS {
            let round_constant_input =
                circuit_builder.input(Row(round_constant_col_index + round_constant_offset));
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

    pub fn ext_transition_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<HashTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
        let circuit_builder = ConstraintCircuitBuilder::new(2 * FULL_WIDTH);
        let constant = |c: u64| circuit_builder.b_constant(c.into());

        let from_processor_eval_indeterminate =
            circuit_builder.challenge(FromProcessorEvalIndeterminate);
        let to_processor_eval_indeterminate =
            circuit_builder.challenge(ToProcessorEvalIndeterminate);

        let round_number = circuit_builder.input(CurrentRow(ROUNDNUMBER.into()));
        let running_evaluation_from_processor =
            circuit_builder.input(CurrentRow(FromProcessorRunningEvaluation.into()));
        let running_evaluation_to_processor =
            circuit_builder.input(CurrentRow(ToProcessorRunningEvaluation.into()));

        let round_number_next = circuit_builder.input(NextRow(ROUNDNUMBER.into()));
        let running_evaluation_from_processor_next =
            circuit_builder.input(NextRow(FromProcessorRunningEvaluation.into()));
        let running_evaluation_to_processor_next =
            circuit_builder.input(NextRow(ToProcessorRunningEvaluation.into()));

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
        .map(|c| circuit_builder.input(CurrentRow(c.into())));
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
        .map(|c| circuit_builder.input(CurrentRow(c.into())));

        let state: [_; STATE_SIZE] = [
            STATE0, STATE1, STATE2, STATE3, STATE4, STATE5, STATE6, STATE7, STATE8, STATE9,
            STATE10, STATE11, STATE12, STATE13, STATE14, STATE15,
        ];
        let current_state = state.map(|s| circuit_builder.input(CurrentRow(s.into())));
        let next_state = state.map(|s| circuit_builder.input(NextRow(s.into())));

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

    pub fn ext_terminal_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<HashTableChallenges, SingleRowIndicator<FULL_WIDTH>>> {
        // no more constraints
        vec![]
    }

    fn ext_initial_constraints(
        challenges: &HashTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        Self::ext_initial_constraints_as_circuits()
            .into_iter()
            .map(|circuit| circuit.partial_evaluate(challenges))
            .collect_vec()
    }

    fn ext_consistency_constraints(
        challenges: &HashTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        Self::ext_consistency_constraints_as_circuits()
            .into_iter()
            .map(|circuit| circuit.partial_evaluate(challenges))
            .collect_vec()
    }

    fn ext_transition_constraints(
        challenges: &HashTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        Self::ext_transition_constraints_as_circuits()
            .into_iter()
            .map(|circuit| circuit.partial_evaluate(challenges))
            .collect_vec()
    }

    fn ext_terminal_constraints(
        challenges: &HashTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        Self::ext_terminal_constraints_as_circuits()
            .into_iter()
            .map(|circuit| circuit.partial_evaluate(challenges))
            .collect_vec()
    }
}

impl HashTable {
    pub fn new(inherited_table: Table<BFieldElement>) -> Self {
        Self { inherited_table }
    }

    pub fn new_prover(matrix: Vec<Vec<BFieldElement>>) -> Self {
        let inherited_table = Table::new(BASE_WIDTH, FULL_WIDTH, matrix, "HashTable".to_string());
        Self { inherited_table }
    }

    pub fn to_quotient_and_fri_domain_table(
        &self,
        quotient_domain: &ArithmeticDomain<BFieldElement>,
        fri_domain: &ArithmeticDomain<BFieldElement>,
        num_trace_randomizers: usize,
    ) -> (Self, Self) {
        let base_columns = 0..self.base_width();
        let (quotient_domain_table, fri_domain_table) = self.dual_low_degree_extension(
            quotient_domain,
            fri_domain,
            num_trace_randomizers,
            base_columns,
        );
        (
            Self::new(quotient_domain_table),
            Self::new(fri_domain_table),
        )
    }

    pub fn extend(&self, challenges: &HashTableChallenges) -> ExtHashTable {
        let mut from_processor_running_evaluation = EvalArg::default_initial();
        let mut to_processor_running_evaluation = EvalArg::default_initial();

        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        for row in self.data().iter() {
            let mut extension_row = [0.into(); FULL_WIDTH];
            extension_row[..BASE_WIDTH]
                .copy_from_slice(&row.iter().map(|elem| elem.lift()).collect_vec());

            // Add compressed input to running evaluation if round index marks beginning of hashing
            if row[usize::from(ROUNDNUMBER)].value() == 1 {
                let state_for_input = [
                    extension_row[usize::from(STATE0)],
                    extension_row[usize::from(STATE1)],
                    extension_row[usize::from(STATE2)],
                    extension_row[usize::from(STATE3)],
                    extension_row[usize::from(STATE4)],
                    extension_row[usize::from(STATE5)],
                    extension_row[usize::from(STATE6)],
                    extension_row[usize::from(STATE7)],
                    extension_row[usize::from(STATE8)],
                    extension_row[usize::from(STATE9)],
                ];
                let compressed_state_for_input: XFieldElement = state_for_input
                    .iter()
                    .zip_eq(
                        [
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
                        ]
                        .iter(),
                    )
                    .map(|(&state, &weight)| weight * state)
                    .sum();

                from_processor_running_evaluation = from_processor_running_evaluation
                    * challenges.from_processor_eval_indeterminate
                    + compressed_state_for_input;
            }
            extension_row[usize::from(FromProcessorRunningEvaluation)] =
                from_processor_running_evaluation;

            // Add compressed digest to running evaluation if round index marks end of hashing
            if row[usize::from(ROUNDNUMBER)].value() == NUM_ROUNDS as u64 + 1 {
                let state_for_output = [
                    extension_row[usize::from(STATE0)],
                    extension_row[usize::from(STATE1)],
                    extension_row[usize::from(STATE2)],
                    extension_row[usize::from(STATE3)],
                    extension_row[usize::from(STATE4)],
                ];
                let compressed_state_for_output: XFieldElement = state_for_output
                    .iter()
                    .zip_eq(
                        [
                            challenges.digest_output_weight0,
                            challenges.digest_output_weight1,
                            challenges.digest_output_weight2,
                            challenges.digest_output_weight3,
                            challenges.digest_output_weight4,
                        ]
                        .iter(),
                    )
                    .map(|(&state, &weight)| weight * state)
                    .sum();

                to_processor_running_evaluation = to_processor_running_evaluation
                    * challenges.to_processor_eval_indeterminate
                    + compressed_state_for_output;
            }
            extension_row[usize::from(ToProcessorRunningEvaluation)] =
                to_processor_running_evaluation;

            extension_matrix.push(extension_row.to_vec());
        }

        assert_eq!(self.data().len(), extension_matrix.len());
        let extension_table = self.new_from_lifted_matrix(extension_matrix);

        ExtHashTable {
            inherited_table: extension_table,
        }
    }

    pub fn for_verifier() -> ExtHashTable {
        let inherited_table =
            Table::new(BASE_WIDTH, FULL_WIDTH, vec![], "ExtHashTable".to_string());
        let base_table = Self { inherited_table };
        let empty_matrix: Vec<Vec<XFieldElement>> = vec![];
        let extension_table = base_table.new_from_lifted_matrix(empty_matrix);

        ExtHashTable {
            inherited_table: extension_table,
        }
    }
}

impl ExtHashTable {
    pub fn new(inherited_table: Table<XFieldElement>) -> Self {
        Self { inherited_table }
    }

    pub fn to_quotient_and_fri_domain_table(
        &self,
        quotient_domain: &ArithmeticDomain<BFieldElement>,
        fri_domain: &ArithmeticDomain<BFieldElement>,
        num_trace_randomizers: usize,
    ) -> (Self, Self) {
        let ext_columns = self.base_width()..self.full_width();
        let (quotient_domain_table_ext, fri_domain_table_ext) = self.dual_low_degree_extension(
            quotient_domain,
            fri_domain,
            num_trace_randomizers,
            ext_columns,
        );
        (
            Self::new(quotient_domain_table_ext),
            Self::new(fri_domain_table_ext),
        )
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

impl ExtensionTable for ExtHashTable {
    fn dynamic_initial_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtHashTable::ext_initial_constraints(&challenges.hash_table_challenges)
    }

    fn dynamic_consistency_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtHashTable::ext_consistency_constraints(&challenges.hash_table_challenges)
    }

    fn dynamic_transition_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtHashTable::ext_transition_constraints(&challenges.hash_table_challenges)
    }

    fn dynamic_terminal_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtHashTable::ext_terminal_constraints(&challenges.hash_table_challenges)
    }
}

#[cfg(test)]
mod constraint_tests {
    use crate::table::extension_table::Evaluable;
    use crate::vm::Program;

    use super::*;

    #[test]
    fn table_satisfies_constraints_test() {
        let program = Program::from_code("hash hash hash halt").unwrap();

        let (aet, _, maybe_err) = program.simulate_no_input();

        if let Some(e) = maybe_err {
            panic!("Program execution failed: {e}");
        }

        let challenges = AllChallenges::placeholder();
        let ext_hash_table =
            HashTable::new_prover(aet.hash_matrix.iter().map(|r| r.to_vec()).collect())
                .extend(&challenges.hash_table_challenges);

        for v in ext_hash_table.evaluate_initial_constraints(&ext_hash_table.data()[0], &challenges)
        {
            assert!(v.is_zero());
        }

        for (i, row) in ext_hash_table.data().iter().enumerate() {
            for (j, v) in ext_hash_table
                .evaluate_consistency_constraints(row, &challenges)
                .iter()
                .enumerate()
            {
                assert!(v.is_zero(), "consistency constraint {j} failed in row {i}");
            }
        }

        for (i, (current_row, next_row)) in ext_hash_table.data().iter().tuple_windows().enumerate()
        {
            for (j, v) in ext_hash_table
                .evaluate_transition_constraints(current_row, next_row, &challenges)
                .iter()
                .enumerate()
            {
                assert!(v.is_zero(), "transition constraint {j} failed in row {i}",);
            }
        }

        for v in ext_hash_table.evaluate_terminal_constraints(
            &ext_hash_table.data()[ext_hash_table.data().len() - 1],
            &challenges,
        ) {
            assert!(v.is_zero());
        }
    }
}
