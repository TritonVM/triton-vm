use std::ops::Add;
use std::ops::Mul;

use itertools::Itertools;
use num_traits::{One, Zero};
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::{Degree, MPolynomial};
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::rescue_prime_regular::DIGEST_LENGTH;
use twenty_first::shared_math::rescue_prime_regular::{
    ALPHA, CAPACITY, MDS, MDS_INV, NUM_ROUNDS, ROUND_CONSTANTS, STATE_SIZE,
};
use twenty_first::shared_math::traits::ModPowU32;
use twenty_first::shared_math::traits::ModPowU64;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::cross_table_arguments::{CrossTableArg, EvalArg};
use crate::fri_domain::FriDomain;
use crate::table::base_table::Extendable;
use crate::table::extension_table::Evaluable;
use crate::table::table_collection::interpolant_degree;
use crate::table::table_column::HashBaseTableColumn::{self, *};
use crate::table::table_column::HashExtTableColumn::{self, *};

use super::base_table::{InheritsFromTable, Table, TableLike};
use super::challenges::AllChallenges;
use super::extension_table::{ExtensionTable, Quotientable, QuotientableExtensionTable};

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

impl Evaluable for ExtHashTable {
    fn evaluate_consistency_constraints(
        &self,
        evaluation_point: &[XFieldElement],
        _challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {
        let constant = |x| BFieldElement::new(x as u64).lift();
        let round_number = evaluation_point[usize::from(ROUNDNUMBER)];
        let state10 = evaluation_point[usize::from(STATE10)];
        let state11 = evaluation_point[usize::from(STATE11)];
        let state12 = evaluation_point[usize::from(STATE12)];
        let state13 = evaluation_point[usize::from(STATE13)];
        let state14 = evaluation_point[usize::from(STATE14)];
        let state15 = evaluation_point[usize::from(STATE15)];

        let round_number_is_not_1_or = (0..=NUM_ROUNDS + 1)
            .filter(|&r| r != 1)
            .map(|r| round_number - constant(r))
            .fold(XFieldElement::one(), XFieldElement::mul);

        let mut evaluated_consistency_constraints = vec![
            round_number_is_not_1_or * (state10 - XFieldElement::one()), // <-- domain separation bit
            round_number_is_not_1_or * state11,
            round_number_is_not_1_or * state12,
            round_number_is_not_1_or * state13,
            round_number_is_not_1_or * state14,
            round_number_is_not_1_or * state15,
        ];

        let round_constant_offset = usize::from(CONSTANT0A);
        for round_constant_idx in 0..NUM_ROUND_CONSTANTS {
            let round_constant_column: HashBaseTableColumn =
                // wrap
                (round_constant_idx + round_constant_offset).try_into().unwrap();
            evaluated_consistency_constraints.push(
                round_number
                    * (round_number - XFieldElement::from(NUM_ROUNDS as u32 + 1))
                    * (Self::round_constants_interpolant(round_constant_column)
                        .evaluate(&evaluation_point[usize::from(ROUNDNUMBER)])
                        - evaluation_point[usize::from(round_constant_column)]),
            );
        }

        evaluated_consistency_constraints
    }

    fn evaluate_transition_constraints(
        &self,
        evaluation_point: &[XFieldElement],
        challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {
        let constant = |c: u64| BFieldElement::new(c).lift();
        let from_processor_eval_indeterminate = challenges
            .hash_table_challenges
            .from_processor_eval_indeterminate;
        let to_processor_eval_indeterminate = challenges
            .hash_table_challenges
            .to_processor_eval_indeterminate;

        let round_number = evaluation_point[usize::from(ROUNDNUMBER)];
        let running_evaluation_from_processor =
            evaluation_point[usize::from(FromProcessorRunningEvaluation)];
        let running_evaluation_to_processor =
            evaluation_point[usize::from(ToProcessorRunningEvaluation)];
        let round_number_next = evaluation_point[FULL_WIDTH + usize::from(ROUNDNUMBER)];
        let running_evaluation_from_processor_next =
            evaluation_point[FULL_WIDTH + usize::from(FromProcessorRunningEvaluation)];
        let running_evaluation_to_processor_next =
            evaluation_point[FULL_WIDTH + usize::from(ToProcessorRunningEvaluation)];

        let mut constraint_evaluations: Vec<XFieldElement> = vec![];

        // round number
        // round numbers evolve as
        // 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9, and
        // 9 -> 1 or 9 -> 0, and
        // 0 -> 0

        // 1. round number belongs to {0, ..., 9}
        // => consistency constraint

        // 2. if round number is 0, then next round number is 0
        // DNF: rn in {1, ..., 9} ∨ rn* = 0
        let mut evaluation = (1..=NUM_ROUNDS + 1)
            .map(|r| constant(r as u64) - round_number)
            .fold(constant(1), XFieldElement::mul);
        evaluation *= round_number_next;
        constraint_evaluations.push(evaluation);

        // 3. if round number is 9, then next round number is 0 or 1
        // DNF: rn =/= 9 ∨ rn* = 0 ∨ rn* = 1
        evaluation = (0..=NUM_ROUNDS)
            .map(|r| constant(r as u64) - round_number)
            .fold(constant(1), XFieldElement::mul);
        evaluation *= constant(1) - round_number_next;
        evaluation *= round_number_next;
        constraint_evaluations.push(evaluation);

        // 4. if round number is in {1, ..., 8} then next round number is +1
        // DNF: (rn == 0 ∨ rn == 9) ∨ rn* = rn + 1
        evaluation = round_number
            * (constant(NUM_ROUNDS as u64 + 1) - round_number)
            * (round_number_next - round_number - constant(1));
        constraint_evaluations.push(evaluation);

        // Rescue-XLIX

        // left-hand-side, starting at current round and going forward
        let current_state: Vec<XFieldElement> = (0..STATE_SIZE)
            .map(|i| evaluation_point[usize::from(STATE0) + i])
            .collect_vec();
        let after_sbox = current_state
            .into_iter()
            .map(|c| c.mod_pow_u64(ALPHA))
            .collect_vec();
        let after_mds = (0..STATE_SIZE)
            .map(|i| {
                (0..STATE_SIZE)
                    .map(|j| constant(MDS[i * STATE_SIZE + j]) * after_sbox[j])
                    .fold(constant(0), XFieldElement::add)
            })
            .collect_vec();
        let round_constants =
            evaluation_point[usize::from(CONSTANT0A)..=usize::from(CONSTANT15B)].to_vec();
        let after_constants = after_mds
            .into_iter()
            .zip_eq(&round_constants[..(NUM_ROUND_CONSTANTS / 2)])
            .map(|(st, rndc)| st + rndc.to_owned())
            .collect_vec();

        // right hand side; move backwards
        let next_state: Vec<XFieldElement> = (0..STATE_SIZE)
            .map(|i| evaluation_point[FULL_WIDTH + usize::from(STATE0) + i])
            .collect_vec();
        let before_constants = next_state
            .into_iter()
            .zip_eq(&round_constants[(NUM_ROUND_CONSTANTS / 2)..])
            .map(|(st, rndc)| st - rndc.to_owned())
            .collect_vec();
        let before_mds = (0..STATE_SIZE)
            .map(|i| {
                (0..STATE_SIZE)
                    .map(|j| constant(MDS_INV[i * STATE_SIZE + j]) * before_constants[j])
                    .fold(constant(0), XFieldElement::add)
            })
            .collect_vec();
        let before_sbox = before_mds
            .iter()
            .map(|c| (*c).mod_pow_u32(ALPHA as u32))
            .collect_vec();

        // Equate left hand side to right hand side. Ignore if padding row or after final round.
        constraint_evaluations.append(
            &mut after_constants
                .into_iter()
                .zip_eq(before_sbox.into_iter())
                .map(|(lhs, rhs)| {
                    round_number * (round_number - constant(NUM_ROUNDS as u64 + 1)) * (lhs - rhs)
                })
                .collect_vec(),
        );

        // Evaluation Arguments

        // from Processor Table to Hash Table
        // If (and only if) the next row number is 1, update running evaluation “from processor.”
        let running_evaluation_from_processor_remains =
            running_evaluation_from_processor_next - running_evaluation_from_processor;
        let xlix_input = (0..2 * DIGEST_LENGTH)
            .map(|i| evaluation_point[FULL_WIDTH + usize::from(STATE0) + i])
            .collect_vec();
        let compressed_row_from_processor = [
            challenges.hash_table_challenges.stack_input_weight0,
            challenges.hash_table_challenges.stack_input_weight1,
            challenges.hash_table_challenges.stack_input_weight2,
            challenges.hash_table_challenges.stack_input_weight3,
            challenges.hash_table_challenges.stack_input_weight4,
            challenges.hash_table_challenges.stack_input_weight5,
            challenges.hash_table_challenges.stack_input_weight6,
            challenges.hash_table_challenges.stack_input_weight7,
            challenges.hash_table_challenges.stack_input_weight8,
            challenges.hash_table_challenges.stack_input_weight9,
        ]
        .into_iter()
        .zip_eq(xlix_input.into_iter())
        .map(|(weight, state)| weight * state)
        .sum();
        let running_evaluation_from_processor_updates = running_evaluation_from_processor_next
            - from_processor_eval_indeterminate * running_evaluation_from_processor
            - compressed_row_from_processor;
        let round_number_next_unequal_1 = (0..=NUM_ROUNDS + 1)
            .filter(|&r| r != 1)
            .map(|r| round_number_next - constant(r as u64))
            .fold(XFieldElement::one(), XFieldElement::mul);
        let running_evaluation_from_processor_is_updated_correctly =
            running_evaluation_from_processor_remains * (round_number_next - constant(1))
                + running_evaluation_from_processor_updates * round_number_next_unequal_1;
        constraint_evaluations.push(running_evaluation_from_processor_is_updated_correctly);

        // from Hash Table to Processor Table
        // If (and only if) the next row number is 9, update running evaluation “to processor.”
        let running_evaluation_to_processor_remains =
            running_evaluation_to_processor_next - running_evaluation_to_processor;
        let xlix_digest = (0..DIGEST_LENGTH)
            .map(|i| evaluation_point[FULL_WIDTH + usize::from(STATE0) + i])
            .collect_vec();
        let compressed_row_to_processor = [
            challenges.hash_table_challenges.digest_output_weight0,
            challenges.hash_table_challenges.digest_output_weight1,
            challenges.hash_table_challenges.digest_output_weight2,
            challenges.hash_table_challenges.digest_output_weight3,
            challenges.hash_table_challenges.digest_output_weight4,
        ]
        .into_iter()
        .zip_eq(xlix_digest.into_iter())
        .map(|(weight, state)| weight * state)
        .sum();
        let running_evaluation_to_processor_updates = running_evaluation_to_processor_next
            - to_processor_eval_indeterminate * running_evaluation_to_processor
            - compressed_row_to_processor;
        let round_number_next_leq_number_of_rounds = (0..=NUM_ROUNDS)
            .map(|r| round_number_next - constant(r as u64))
            .fold(XFieldElement::one(), XFieldElement::mul);
        let running_evaluation_to_processor_is_updated_correctly =
            running_evaluation_to_processor_remains
                * (round_number_next - constant(NUM_ROUNDS as u64 + 1))
                + running_evaluation_to_processor_updates * round_number_next_leq_number_of_rounds;
        constraint_evaluations.push(running_evaluation_to_processor_is_updated_correctly);

        constraint_evaluations
    }
}

impl Quotientable for ExtHashTable {
    fn get_consistency_quotient_degree_bounds(
        &self,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Vec<Degree> {
        let zerofier_degree = padded_height as Degree;
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);
        let capacity_degree_bounds =
            vec![interpolant_degree * (NUM_ROUNDS + 1 + 1) as Degree; CAPACITY];
        let round_constant_degree_bounds =
            vec![interpolant_degree * (NUM_ROUNDS + 1) as Degree; NUM_ROUND_CONSTANTS];

        [capacity_degree_bounds, round_constant_degree_bounds]
            .concat()
            .into_iter()
            .map(|degree_bound| degree_bound - zerofier_degree)
            .collect_vec()
    }

    fn get_transition_quotient_degree_bounds(
        &self,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Vec<Degree> {
        let zerofier_degree = padded_height as Degree - 1;
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);
        let round_number_bounds = vec![
            interpolant_degree * (NUM_ROUNDS + 1 + 1) as Degree,
            interpolant_degree * (NUM_ROUNDS + 1 + 1 + 1) as Degree,
            interpolant_degree * 3,
        ];
        let state_evolution_bounds =
            vec![interpolant_degree * (ALPHA + 1 + 1) as Degree; STATE_SIZE];
        let eval_arg_degrees = vec![interpolant_degree * (NUM_ROUNDS + 1 + 1) as Degree; 2];

        [
            round_number_bounds,
            state_evolution_bounds,
            eval_arg_degrees,
        ]
        .concat()
        .into_iter()
        .map(|degree_bound| degree_bound - zerofier_degree)
        .collect_vec()
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
    fn ext_initial_constraints(
        challenges: &HashTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let constant = |xfe| MPolynomial::from_constant(xfe, FULL_WIDTH);
        let one = constant(XFieldElement::one());
        let running_evaluation_initial = constant(EvalArg::default_initial());

        let variables = MPolynomial::variables(FULL_WIDTH);
        let round_number = variables[usize::from(ROUNDNUMBER)].clone();
        let running_evaluation_from_processor =
            variables[usize::from(FromProcessorRunningEvaluation)].clone();
        let running_evaluation_to_processor =
            variables[usize::from(ToProcessorRunningEvaluation)].clone();
        let state = (0..2 * DIGEST_LENGTH)
            .map(|i| variables[usize::from(STATE0) + i].clone())
            .collect_vec();

        let round_number_is_0_or_1 = round_number.clone() * (round_number.clone() - one.clone());

        // Evaluation Argument “from processor”
        // If the round number is 0, the running evaluation is the default initial.
        // Else, the first update has been applied to the running evaluation.
        let running_evaluation_from_processor_is_default_initial =
            running_evaluation_from_processor.clone() - running_evaluation_initial.clone();
        let compressed_row = [
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
        .iter()
        .zip_eq(state.iter())
        .map(|(&w, s)| s.clone() * constant(w))
        .sum();
        let from_processor_indeterminate = constant(challenges.from_processor_eval_indeterminate);
        let running_evaluation_from_processor_is_updated = running_evaluation_from_processor
            - running_evaluation_initial.clone() * from_processor_indeterminate
            - compressed_row;
        let running_evaluation_from_processor_is_updated_if_and_only_if_not_a_padding_row =
            round_number.clone() * running_evaluation_from_processor_is_updated
                + (one - round_number) * running_evaluation_from_processor_is_default_initial;

        // Evaluation Argument “to processor”
        let running_evaluation_to_processor_is_default_initial =
            running_evaluation_to_processor - running_evaluation_initial;

        vec![
            round_number_is_0_or_1,
            running_evaluation_from_processor_is_updated_if_and_only_if_not_a_padding_row,
            running_evaluation_to_processor_is_default_initial,
        ]
    }

    /// The implementation below is kept around for debugging purposes. This table evaluates the
    /// consistency constraints directly by implementing the respective method in trait
    /// `Evaluable`, and does not use the polynomials below.
    #[allow(unreachable_code)]
    fn ext_consistency_constraints(
        _challenges: &HashTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        panic!("ext_consistency_constraints should never be called; method is bypassed statically");
        let constant = |c| MPolynomial::from_constant(BFieldElement::new(c).lift(), FULL_WIDTH);
        let variables = MPolynomial::variables(FULL_WIDTH);

        let round_number = variables[usize::from(ROUNDNUMBER)].clone();
        let state10 = variables[usize::from(STATE10)].clone();
        let state11 = variables[usize::from(STATE11)].clone();
        let state12 = variables[usize::from(STATE12)].clone();
        let state13 = variables[usize::from(STATE13)].clone();
        let state14 = variables[usize::from(STATE14)].clone();
        let state15 = variables[usize::from(STATE15)].clone();

        // 1. if round number is 1, then capacity is zero
        // DNF: rn =/= 1 ∨ cap = 0
        let round_number_is_not_1_or = (0..=NUM_ROUNDS + 1)
            .filter(|&r| r != 1)
            .map(|r| round_number.clone() - constant(r as u64))
            .fold(constant(1), MPolynomial::mul);

        let mut consistency_polynomials = vec![
            round_number_is_not_1_or.clone() * (state10 - constant(1)), // <-- domain separation bit
            round_number_is_not_1_or.clone() * state11,
            round_number_is_not_1_or.clone() * state12,
            round_number_is_not_1_or.clone() * state13,
            round_number_is_not_1_or.clone() * state14,
            round_number_is_not_1_or * state15,
        ];

        // 2. round number is in {0, ..., 9}
        let polynomial = (0..=NUM_ROUNDS + 1)
            .map(|r| constant(r as u64) - round_number.clone())
            .fold(constant(1), MPolynomial::mul);
        consistency_polynomials.push(polynomial);

        // 3. round constants
        // if round number is zero, we don't care
        // otherwise, make sure the constant is correct
        let round_constant_offset = usize::from(CONSTANT0A);
        for round_constant_idx in 0..NUM_ROUND_CONSTANTS {
            let round_constant_column: HashBaseTableColumn =
                // wrap
                (round_constant_idx + round_constant_offset).try_into().unwrap();
            let round_constant = &variables[usize::from(round_constant_column)];
            let interpolant = Self::round_constants_interpolant(round_constant_column);
            let multivariate_interpolant =
                MPolynomial::lift(interpolant, usize::from(ROUNDNUMBER), FULL_WIDTH);
            consistency_polynomials.push(
                round_number.clone()
                    * (round_number.clone() - constant(NUM_ROUNDS as u64 + 1))
                    * (multivariate_interpolant - round_constant.clone()),
            );
        }

        consistency_polynomials
    }

    fn round_constants_interpolant(
        round_constant: HashBaseTableColumn,
    ) -> Polynomial<XFieldElement> {
        let round_constant_idx = usize::from(round_constant) - usize::from(CONSTANT0A);
        let domain = (1..=NUM_ROUNDS)
            .map(|x| BFieldElement::new(x as u64).lift())
            .collect_vec();
        let abscissae = (1..=NUM_ROUNDS)
            .map(|i| ROUND_CONSTANTS[NUM_ROUND_CONSTANTS * (i - 1) + round_constant_idx])
            .map(|x| BFieldElement::new(x).lift())
            .collect_vec();
        Polynomial::lagrange_interpolate(&domain, &abscissae)
    }

    /// The implementation below is kept around for debugging purposes. This table evaluates the
    /// transition constraints directly by implementing the respective method in trait
    /// `Evaluable`, and does not use the polynomials below.
    #[allow(unreachable_code)]
    fn ext_transition_constraints(
        _challenges: &HashTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        panic!("ext_transition_constraints should never be called; method is bypassed statically");
        let constant = |c| MPolynomial::from_constant(BFieldElement::new(c).lift(), 2 * FULL_WIDTH);
        let variables = MPolynomial::variables(2 * FULL_WIDTH);

        let round_number = variables[usize::from(ROUNDNUMBER)].clone();
        let round_number_next = variables[FULL_WIDTH + usize::from(ROUNDNUMBER)].clone();

        let mut constraint_polynomials: Vec<MPolynomial<XFieldElement>> = vec![];

        // round number
        // round numbers evolve as
        // 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9, and
        // 9 -> 1 or 9 -> 0, and
        // 0 -> 0

        // 1. round number belongs to {0, ..., 9}
        // => consistency constraint

        // 2. if round number is 0, then next round number is 0
        // DNF: rn in {1, ..., 9} ∨ rn* = 0
        let mut polynomial = (1..=NUM_ROUNDS + 1)
            .map(|r| constant(r as u64) - round_number.clone())
            .fold(constant(1_u64), MPolynomial::mul);
        polynomial *= round_number_next.clone();
        constraint_polynomials.push(polynomial);

        // 3. if round number is 9, then next round number is 0 or 1
        // DNF: rn =/= 9 ∨ rn* = 0 ∨ rn* = 1
        polynomial = (0..=NUM_ROUNDS)
            .map(|r| constant(r as u64) - round_number.clone())
            .fold(constant(1), MPolynomial::mul);
        polynomial *= constant(1) - round_number_next.clone();
        polynomial *= round_number_next.clone();
        constraint_polynomials.push(polynomial);

        // 4. if round number is in {1, ..., 8} then next round number is +1
        // DNF: (rn == 0 ∨ rn == 9) ∨ rn* = rn + 1
        polynomial = round_number.clone()
            * (constant(NUM_ROUNDS as u64 + 1) - round_number.clone())
            * (round_number_next.clone() - round_number.clone() - constant(1));
        constraint_polynomials.push(polynomial);

        // Rescue-XLIX

        // left-hand-side, starting at current round and going forward
        let current_state: Vec<MPolynomial<XFieldElement>> = (0..STATE_SIZE)
            .map(|i| variables[usize::from(STATE0) + i].clone())
            .collect_vec();
        let after_sbox = current_state
            .iter()
            .map(|c| c.pow(ALPHA as u8))
            .collect_vec();
        let after_mds = (0..STATE_SIZE)
            .map(|i| {
                (0..STATE_SIZE)
                    .map(|j| constant(MDS[i * STATE_SIZE + j]) * after_sbox[j].clone())
                    .fold(constant(0), MPolynomial::add)
            })
            .collect_vec();
        let round_constants =
            variables[usize::from(CONSTANT0A)..=usize::from(CONSTANT15B)].to_vec();
        let after_constants = after_mds
            .into_iter()
            .zip_eq(&round_constants[..(NUM_ROUND_CONSTANTS / 2)])
            .map(|(st, rndc)| st + rndc.to_owned())
            .collect_vec();

        // right hand side; move backwards
        let next_state: Vec<MPolynomial<XFieldElement>> = (0..STATE_SIZE)
            .map(|i| variables[FULL_WIDTH + usize::from(STATE0) + i].clone())
            .collect_vec();
        let before_constants = next_state
            .into_iter()
            .zip_eq(&round_constants[(NUM_ROUND_CONSTANTS / 2)..])
            .map(|(st, rndc)| st - rndc.to_owned())
            .collect_vec();
        let before_mds = (0..STATE_SIZE)
            .map(|i| {
                (0..STATE_SIZE)
                    .map(|j| constant(MDS_INV[i * STATE_SIZE + j]) * before_constants[j].clone())
                    .fold(constant(0), MPolynomial::add)
            })
            .collect_vec();
        let before_sbox = before_mds.iter().map(|c| c.pow(ALPHA as u8)).collect_vec();

        // equate left hand side to right hand side
        // (and ignore if padding row)
        constraint_polynomials.append(
            &mut after_constants
                .into_iter()
                .zip_eq(before_sbox.into_iter())
                .map(|(lhs, rhs)| {
                    round_number.clone()
                        * (round_number.clone() - constant(NUM_ROUNDS as u64 + 1))
                        * (lhs - rhs)
                })
                .collect_vec(),
        );

        constraint_polynomials
    }

    fn ext_terminal_constraints(
        _challenges: &HashTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        vec![]
    }
}

impl HashTable {
    pub fn new_prover(matrix: Vec<Vec<BFieldElement>>) -> Self {
        let inherited_table = Table::new(BASE_WIDTH, FULL_WIDTH, matrix, "HashTable".to_string());
        Self { inherited_table }
    }

    pub fn to_fri_domain_table(
        &self,
        fri_domain: &FriDomain<BFieldElement>,
        omicron: BFieldElement,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Self {
        let base_columns = 0..self.base_width();
        let fri_domain_codewords = self.low_degree_extension(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
            base_columns,
        );
        let inherited_table = self.inherited_table.with_data(fri_domain_codewords);
        Self { inherited_table }
    }

    pub fn extend(
        &self,
        challenges: &HashTableChallenges,
        interpolant_degree: Degree,
    ) -> ExtHashTable {
        let mut from_processor_running_evaluation = EvalArg::default_initial();
        let mut to_processor_running_evaluation = EvalArg::default_initial();

        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        for row in self.data().iter() {
            let mut extension_row = [0.into(); FULL_WIDTH];
            extension_row[..BASE_WIDTH]
                .copy_from_slice(&row.iter().map(|elem| elem.lift()).collect_vec());

            // Add compressed input to running evaluation if round index marks beginning of hashing
            if row[usize::from(HashBaseTableColumn::ROUNDNUMBER)].value() == 1 {
                let state_for_input = [
                    extension_row[usize::from(HashBaseTableColumn::STATE0)],
                    extension_row[usize::from(HashBaseTableColumn::STATE1)],
                    extension_row[usize::from(HashBaseTableColumn::STATE2)],
                    extension_row[usize::from(HashBaseTableColumn::STATE3)],
                    extension_row[usize::from(HashBaseTableColumn::STATE4)],
                    extension_row[usize::from(HashBaseTableColumn::STATE5)],
                    extension_row[usize::from(HashBaseTableColumn::STATE6)],
                    extension_row[usize::from(HashBaseTableColumn::STATE7)],
                    extension_row[usize::from(HashBaseTableColumn::STATE8)],
                    extension_row[usize::from(HashBaseTableColumn::STATE9)],
                ];
                let compressed_state_for_input = state_for_input
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
            if row[usize::from(HashBaseTableColumn::ROUNDNUMBER)].value() == NUM_ROUNDS as u64 + 1 {
                let state_for_output = [
                    extension_row[usize::from(HashBaseTableColumn::STATE0)],
                    extension_row[usize::from(HashBaseTableColumn::STATE1)],
                    extension_row[usize::from(HashBaseTableColumn::STATE2)],
                    extension_row[usize::from(HashBaseTableColumn::STATE3)],
                    extension_row[usize::from(HashBaseTableColumn::STATE4)],
                ];
                let compressed_state_for_output = state_for_output
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
        let padded_height = extension_matrix.len();
        let extension_table = self.extension(
            extension_matrix,
            interpolant_degree,
            padded_height,
            ExtHashTable::ext_initial_constraints(challenges),
            vec![],
            vec![],
            ExtHashTable::ext_terminal_constraints(challenges),
        );

        ExtHashTable {
            inherited_table: extension_table,
        }
    }

    pub fn for_verifier(
        interpolant_degree: Degree,
        padded_height: usize,
        all_challenges: &AllChallenges,
    ) -> ExtHashTable {
        let inherited_table =
            Table::new(BASE_WIDTH, FULL_WIDTH, vec![], "ExtHashTable".to_string());
        let base_table = Self { inherited_table };
        let empty_matrix: Vec<Vec<XFieldElement>> = vec![];
        let extension_table = base_table.extension(
            empty_matrix,
            interpolant_degree,
            padded_height,
            ExtHashTable::ext_initial_constraints(&all_challenges.hash_table_challenges),
            // The Hash Table bypasses the symbolic representation of transition and consistency
            // constraints. As a result, there is nothing to memoize. Since the memoization
            // dictionary is never used, it can't hurt to supply empty databases.
            vec![],
            vec![],
            ExtHashTable::ext_terminal_constraints(&all_challenges.hash_table_challenges),
        );

        ExtHashTable {
            inherited_table: extension_table,
        }
    }
}

impl ExtHashTable {
    pub fn to_fri_domain_table(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        omicron: XFieldElement,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Self {
        let ext_columns = self.base_width()..self.full_width();
        let fri_domain_codewords_ext = self.low_degree_extension(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
            ext_columns,
        );

        let inherited_table = self.inherited_table.with_data(fri_domain_codewords_ext);
        ExtHashTable { inherited_table }
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
        challenges: &super::challenges::AllChallenges,
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
    use twenty_first::shared_math::other::roundup_npo2;

    use crate::vm::Program;

    use super::*;

    #[test]
    fn table_satisfies_constraints_test() {
        let program = Program::from_code("hash hash hash halt").unwrap();

        let (aet, maybe_err, _) = program.simulate_with_input(&[], &[]);

        if let Some(e) = maybe_err {
            panic!("Program execution failed: {e}");
        }

        let padded_height = roundup_npo2(aet.hash_matrix.len() as u64) as usize;
        let num_trace_randomizers = 0;
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);

        let challenges = AllChallenges::placeholder();
        let ext_hash_table =
            HashTable::new_prover(aet.hash_matrix.iter().map(|r| r.to_vec()).collect())
                .extend(&challenges.hash_table_challenges, interpolant_degree);

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
            let eval_point = [current_row.to_vec(), next_row.to_vec()].concat();
            for (j, v) in ext_hash_table
                .evaluate_transition_constraints(&eval_point, &challenges)
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
