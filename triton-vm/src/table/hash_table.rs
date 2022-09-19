use std::ops::Add;
use std::ops::Mul;

use itertools::Itertools;
use num_traits::{One, Zero};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::{Degree, MPolynomial};
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::rescue_prime_regular::DIGEST_LENGTH;
use twenty_first::shared_math::rescue_prime_regular::{
    ALPHA, CAPACITY, MDS, MDS_INV, NUM_ROUNDS, ROUND_CONSTANTS, STATE_SIZE,
};
use twenty_first::shared_math::traits::ModPowU64;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::fri_domain::FriDomain;
use crate::stark::StarkHasher;
use crate::table::base_table::Extendable;
use crate::table::extension_table::Evaluable;
use crate::table::table_column::HashTableColumn::*;

use super::base_table::{self, InheritsFromTable, Table, TableLike};
use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use super::extension_table::{ExtensionTable, Quotientable, QuotientableExtensionTable};
use super::table_column::HashTableColumn;

pub const HASH_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 0;
pub const HASH_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 2;
pub const HASH_TABLE_INITIALS_COUNT: usize =
    HASH_TABLE_PERMUTATION_ARGUMENTS_COUNT + HASH_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 18 because it combines: 12 stack_input_weights and 6 digest_output_weights.
pub const HASH_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 18;

pub const BASE_WIDTH: usize = 49;
pub const FULL_WIDTH: usize = 53; // BASE_WIDTH + 2 * INITIALS_COUNT

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
    inherited_table: Table<XFieldElement>,
}

impl Evaluable for ExtHashTable {
    fn evaluate_consistency_constraints(
        &self,
        evaluation_point: &[XFieldElement],
    ) -> Vec<XFieldElement> {
        let constant = |x| BFieldElement::new(x as u64).lift();
        let round_number = evaluation_point[ROUNDNUMBER as usize];
        let state10 = evaluation_point[STATE10 as usize];
        let state11 = evaluation_point[STATE11 as usize];
        let state12 = evaluation_point[STATE12 as usize];
        let state13 = evaluation_point[STATE13 as usize];
        let state14 = evaluation_point[STATE14 as usize];
        let state15 = evaluation_point[STATE15 as usize];

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

        let round_constant_offset = CONSTANT0A as usize;
        for round_constant_idx in 0..NUM_ROUND_CONSTANTS {
            let round_constant_column: HashTableColumn =
                (round_constant_idx + round_constant_offset).into();
            evaluated_consistency_constraints.push(
                round_number
                    * (round_number - XFieldElement::from(9))
                    * (Self::round_constants_interpolant(round_constant_column)
                        .evaluate(&evaluation_point[ROUNDNUMBER as usize])
                        - evaluation_point[round_constant_column as usize]),
            );
        }

        evaluated_consistency_constraints
    }

    fn evaluate_transition_constraints(
        &self,
        evaluation_point: &[XFieldElement],
    ) -> Vec<XFieldElement> {
        let constant = |c: u64| BFieldElement::new(c).lift();

        let round_number = evaluation_point[ROUNDNUMBER as usize];
        let round_number_next = evaluation_point[FULL_WIDTH + ROUNDNUMBER as usize];

        let mut constraint_evaluations: Vec<XFieldElement> = vec![];

        // round number
        // round numbers evolve as
        // 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9, and
        // 9 -> 1 or 9 -> 0, and
        // 0 -> 0

        // 1. round number belongs to {0, ..., 9}
        // => consistency constraint

        // 2. if round number is 0, then next round number is 0
        // DNF: rn in {1, ..., 9} \/ rn* = 0
        let mut evaluation = (1..=NUM_ROUNDS + 1)
            .map(|r| constant(r as u64) - round_number)
            .fold(constant(1), XFieldElement::mul);
        evaluation *= round_number_next;
        constraint_evaluations.push(evaluation);

        // 3. if round number is 9, then next round number is 0 or 1
        // DNF: rn =/= 9 \/ rn* = 0 \/ rn* = 1
        evaluation = (0..=8)
            .map(|r| constant(r) - round_number)
            .fold(constant(1), XFieldElement::mul);
        evaluation *= constant(1) - round_number_next;
        evaluation *= round_number_next;
        constraint_evaluations.push(evaluation);

        // 4. if round number is in {1, ..., 8} then next round number is +1
        // DNF: (rn == 0 \/ rn == 9) \/ rn* = rn + 1
        evaluation = round_number
            * (constant(9) - round_number)
            * (round_number_next - round_number - constant(1));
        constraint_evaluations.push(evaluation);

        // Rescue-XLIX

        // left-hand-side, starting at current round and going forward
        let current_state: Vec<XFieldElement> = (0..STATE_SIZE)
            .map(|i| evaluation_point[HashTableColumn::STATE0 as usize + i])
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
        let round_constants = evaluation_point
            [(HashTableColumn::CONSTANT0A as usize)..=(HashTableColumn::CONSTANT15B as usize)]
            .to_vec();
        let after_constants = after_mds
            .into_iter()
            .zip_eq(&round_constants[..(NUM_ROUND_CONSTANTS / 2)])
            .map(|(st, rndc)| st + rndc.to_owned())
            .collect_vec();

        // right hand side; move backwards
        let next_state: Vec<XFieldElement> = (0..STATE_SIZE)
            .map(|i| evaluation_point[FULL_WIDTH + HashTableColumn::STATE0 as usize + i])
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
            .map(|c| (*c).mod_pow_u64(ALPHA))
            .collect_vec();

        // equate left hand side to right hand side
        // (and ignore if padding row)
        constraint_evaluations.append(
            &mut after_constants
                .into_iter()
                .zip_eq(before_sbox.into_iter())
                .map(|(lhs, rhs)| round_number * (round_number - constant(9)) * (lhs - rhs))
                .collect_vec(),
        );

        constraint_evaluations
    }
}

impl Quotientable for ExtHashTable {
    fn get_consistency_quotient_degree_bounds(&self) -> Vec<Degree> {
        let capacity_degree_bounds =
            vec![self.interpolant_degree() * (NUM_ROUNDS + 1 + 1) as Degree; CAPACITY];
        let round_constant_degree_bounds =
            vec![self.interpolant_degree() * (NUM_ROUNDS + 1) as Degree; NUM_ROUND_CONSTANTS];

        [capacity_degree_bounds, round_constant_degree_bounds].concat()
    }

    fn get_transition_quotient_degree_bounds(&self) -> Vec<Degree> {
        let round_number_bounds = vec![
            self.interpolant_degree() * (NUM_ROUNDS + 1 + 1) as Degree,
            self.interpolant_degree() * (NUM_ROUNDS + 1 + 1 + 1) as Degree,
            self.interpolant_degree() * 3,
        ];
        let state_evolution_bounds =
            vec![self.interpolant_degree() * (ALPHA + 1 + 1) as Degree; STATE_SIZE];

        [round_number_bounds, state_evolution_bounds].concat()
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
    fn ext_boundary_constraints() -> Vec<MPolynomial<XFieldElement>> {
        let one = MPolynomial::from_constant(1.into(), FULL_WIDTH);
        let variables = MPolynomial::variables(FULL_WIDTH, 1.into());

        let round_number = variables[ROUNDNUMBER as usize].clone();
        let round_number_is_0_or_1 = round_number.clone() * (round_number - one);
        vec![round_number_is_0_or_1]
    }

    /// The implementation below is kept around for debugging purposes. This table evaluates the
    /// consistency constraints directly by implementing the respective method in trait
    /// `Evaluable`, and does not use the polynomials below.
    #[allow(unreachable_code)]
    fn ext_consistency_constraints() -> Vec<MPolynomial<XFieldElement>> {
        panic!("ext_consistency_constraints should never be called; method is bypassed statically");
        let constant = |c| MPolynomial::from_constant(BFieldElement::new(c).lift(), FULL_WIDTH);
        let variables = MPolynomial::variables(FULL_WIDTH, 1.into());

        let round_number = variables[ROUNDNUMBER as usize].clone();
        let state10 = variables[STATE10 as usize].clone();
        let state11 = variables[STATE11 as usize].clone();
        let state12 = variables[STATE12 as usize].clone();
        let state13 = variables[STATE13 as usize].clone();
        let state14 = variables[STATE14 as usize].clone();
        let state15 = variables[STATE15 as usize].clone();

        // 1. if round number is 1, then capacity is zero
        // DNF: rn =/= 1 \/ cap = 0
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
        let nine = constant(9);
        let round_constant_offset = CONSTANT0A as usize;
        for round_constant_idx in 0..NUM_ROUND_CONSTANTS {
            let round_constant_column: HashTableColumn =
                (round_constant_idx + round_constant_offset).into();
            let round_constant = &variables[round_constant_column as usize];
            let interpolant = Self::round_constants_interpolant(round_constant_column);
            let multivariate_interpolant =
                MPolynomial::lift(interpolant, ROUNDNUMBER as usize, FULL_WIDTH);
            consistency_polynomials.push(
                round_number.clone()
                    * (round_number.clone() - nine)
                    * (multivariate_interpolant - round_constant.clone()),
            );
        }

        consistency_polynomials
    }

    fn round_constants_interpolant(round_constant: HashTableColumn) -> Polynomial<XFieldElement> {
        let round_constant_idx = (round_constant as usize) - (CONSTANT0A as usize);
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
        let variables = MPolynomial::variables(2 * FULL_WIDTH, XFieldElement::one());

        let round_number = variables[ROUNDNUMBER as usize].clone();
        let round_number_next = variables[FULL_WIDTH + ROUNDNUMBER as usize].clone();

        let mut constraint_polynomials: Vec<MPolynomial<XFieldElement>> = vec![];

        // round number
        // round numbers evolve as
        // 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9, and
        // 9 -> 1 or 9 -> 0, and
        // 0 -> 0

        // 1. round number belongs to {0, ..., 9}
        // => consistency constraint

        // 2. if round number is 0, then next round number is 0
        // DNF: rn in {1, ..., 9} \/ rn* = 0
        let mut polynomial = (1..=NUM_ROUNDS + 1)
            .map(|r| constant(r as u64) - round_number.clone())
            .fold(constant(1_u64), MPolynomial::mul);
        polynomial *= round_number_next.clone();
        constraint_polynomials.push(polynomial);

        // 3. if round number is 9, then next round number is 0 or 1
        // DNF: rn =/= 9 \/ rn* = 0 \/ rn* = 1
        polynomial = (0..=8)
            .map(|r| constant(r) - round_number.clone())
            .fold(constant(1), MPolynomial::mul);
        polynomial *= constant(1) - round_number_next.clone();
        polynomial *= round_number_next.clone();
        constraint_polynomials.push(polynomial);

        // 4. if round number is in {1, ..., 8} then next round number is +1
        // DNF: (rn == 0 \/ rn == 9) \/ rn* = rn + 1
        polynomial = round_number.clone()
            * (constant(9) - round_number.clone())
            * (round_number_next.clone() - round_number.clone() - constant(1));
        constraint_polynomials.push(polynomial);

        // Rescue-XLIX

        // left-hand-side, starting at current round and going forward
        let current_state: Vec<MPolynomial<XFieldElement>> = (0..STATE_SIZE)
            .map(|i| variables[HashTableColumn::STATE0 as usize + i].clone())
            .collect_vec();
        let after_sbox = current_state.iter().map(|c| c.pow(ALPHA)).collect_vec();
        let after_mds = (0..STATE_SIZE)
            .map(|i| {
                (0..STATE_SIZE)
                    .map(|j| constant(MDS[i * STATE_SIZE + j]) * after_sbox[j].clone())
                    .fold(constant(0), MPolynomial::add)
            })
            .collect_vec();
        let round_constants = variables
            [(HashTableColumn::CONSTANT0A as usize)..=(HashTableColumn::CONSTANT15B as usize)]
            .to_vec();
        let after_constants = after_mds
            .into_iter()
            .zip_eq(&round_constants[..(NUM_ROUND_CONSTANTS / 2)])
            .map(|(st, rndc)| st + rndc.to_owned())
            .collect_vec();

        // right hand side; move backwards
        let next_state: Vec<MPolynomial<XFieldElement>> = (0..STATE_SIZE)
            .map(|i| variables[FULL_WIDTH + HashTableColumn::STATE0 as usize + i].clone())
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
        let before_sbox = before_mds.iter().map(|c| c.pow(ALPHA)).collect_vec();

        // equate left hand side to right hand side
        // (and ignore if padding row)
        constraint_polynomials.append(
            &mut after_constants
                .into_iter()
                .zip_eq(before_sbox.into_iter())
                .map(|(lhs, rhs)| {
                    round_number.clone() * (round_number.clone() - constant(9)) * (lhs - rhs)
                })
                .collect_vec(),
        );

        constraint_polynomials
    }

    fn ext_terminal_constraints(
        _challenges: &HashTableChallenges,
        _terminals: &HashTableEndpoints,
    ) -> Vec<MPolynomial<XFieldElement>> {
        vec![]
    }
}

impl HashTable {
    pub fn new_prover(num_trace_randomizers: usize, matrix: Vec<Vec<BFieldElement>>) -> Self {
        let unpadded_height = matrix.len();
        let padded_height = base_table::padded_height(unpadded_height);

        let omicron = base_table::derive_omicron(padded_height as u64);
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            matrix,
            "HashTable".to_string(),
        );

        Self { inherited_table }
    }

    pub fn codeword_table(&self, fri_domain: &FriDomain<BFieldElement>) -> Self {
        let base_columns = 0..self.base_width();
        let codewords = self.low_degree_extension(fri_domain, base_columns);

        let inherited_table = self.inherited_table.with_data(codewords);
        Self { inherited_table }
    }

    pub fn extend(
        &self,
        challenges: &HashTableChallenges,
        initials: &HashTableEndpoints,
    ) -> (ExtHashTable, HashTableEndpoints) {
        let mut from_processor_running_sum = initials.from_processor_eval_sum;
        let mut to_processor_running_sum = initials.to_processor_eval_sum;

        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        for row in self.data().iter() {
            let mut extension_row = Vec::with_capacity(FULL_WIDTH);
            extension_row.extend(row.iter().map(|elem| elem.lift()));

            // Compress input values into single value (independent of round index)
            let state_for_input = [
                extension_row[HashTableColumn::STATE0 as usize],
                extension_row[HashTableColumn::STATE1 as usize],
                extension_row[HashTableColumn::STATE2 as usize],
                extension_row[HashTableColumn::STATE3 as usize],
                extension_row[HashTableColumn::STATE4 as usize],
                extension_row[HashTableColumn::STATE5 as usize],
                extension_row[HashTableColumn::STATE6 as usize],
                extension_row[HashTableColumn::STATE7 as usize],
                extension_row[HashTableColumn::STATE8 as usize],
                extension_row[HashTableColumn::STATE9 as usize],
                extension_row[HashTableColumn::STATE10 as usize],
                extension_row[HashTableColumn::STATE11 as usize],
            ];
            let compressed_state_for_input = state_for_input
                .iter()
                .zip(challenges.stack_input_weights.iter())
                .map(|(state, weight)| *weight * *state)
                .fold(XFieldElement::zero(), |sum, summand| sum + summand);
            extension_row.push(compressed_state_for_input);

            // Add compressed input to running sum if round index marks beginning of hashing
            extension_row.push(from_processor_running_sum);
            if row[HashTableColumn::ROUNDNUMBER as usize].value() == 1 {
                from_processor_running_sum = from_processor_running_sum
                    * challenges.from_processor_eval_row_weight
                    + compressed_state_for_input;
            }

            // Compress digest values into single value (independent of round index)
            let state_for_output = [
                extension_row[HashTableColumn::STATE0 as usize],
                extension_row[HashTableColumn::STATE1 as usize],
                extension_row[HashTableColumn::STATE2 as usize],
                extension_row[HashTableColumn::STATE3 as usize],
                extension_row[HashTableColumn::STATE4 as usize],
                extension_row[HashTableColumn::STATE5 as usize],
            ];
            let compressed_state_for_output = state_for_output
                .iter()
                .zip(challenges.digest_output_weights.iter())
                .map(|(state, weight)| *weight * *state)
                .fold(XFieldElement::zero(), |sum, summand| sum + summand);
            extension_row.push(compressed_state_for_output);

            // Add compressed digest to running sum if round index marks end of hashing
            extension_row.push(to_processor_running_sum);
            if row[HashTableColumn::ROUNDNUMBER as usize].value() == 8 {
                to_processor_running_sum = to_processor_running_sum
                    * challenges.to_processor_eval_row_weight
                    + compressed_state_for_output;
            }

            extension_matrix.push(extension_row);
        }

        let terminals = HashTableEndpoints {
            from_processor_eval_sum: from_processor_running_sum,
            to_processor_eval_sum: to_processor_running_sum,
        };

        let extension_table = self.extension(
            extension_matrix,
            ExtHashTable::ext_boundary_constraints(),
            vec![],
            vec![],
            ExtHashTable::ext_terminal_constraints(challenges, &terminals),
        );

        (
            ExtHashTable {
                inherited_table: extension_table,
            },
            terminals,
        )
    }

    pub fn for_verifier(
        num_trace_randomizers: usize,
        padded_height: usize,
        all_challenges: &AllChallenges,
        all_terminals: &AllEndpoints<StarkHasher>,
    ) -> ExtHashTable {
        let omicron = base_table::derive_omicron(padded_height as u64);
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            vec![],
            "ExtHashTable".to_string(),
        );
        let base_table = Self { inherited_table };
        let empty_matrix: Vec<Vec<XFieldElement>> = vec![];
        let extension_table = base_table.extension(
            empty_matrix,
            ExtHashTable::ext_boundary_constraints(),
            // The Hash Table bypasses the symbolic representation of transition and consistency constraints.
            // As a result, there is nothing to memoize. Since the memoization dictionary is never used, it
            // can't hurt to supply empty databases.
            vec![], // ExtHashTable::ext_transition_constraints(&all_challenges.hash_table_challenges),
            vec![], // ExtHashTable::ext_consistency_constraints(),
            ExtHashTable::ext_terminal_constraints(
                &all_challenges.hash_table_challenges,
                &all_terminals.hash_table_endpoints,
            ),
        );

        ExtHashTable {
            inherited_table: extension_table,
        }
    }
}

impl ExtHashTable {
    pub fn with_padded_height(num_trace_randomizers: usize, padded_height: usize) -> Self {
        let matrix: Vec<Vec<XFieldElement>> = vec![];

        let omicron = base_table::derive_omicron(padded_height as u64);
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            matrix,
            "ExtHashTable".to_string(),
        );

        Self { inherited_table }
    }

    pub fn ext_codeword_table(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        base_codewords: &[Vec<BFieldElement>],
    ) -> Self {
        let ext_columns = self.base_width()..self.full_width();
        let ext_codewords = self.low_degree_extension(fri_domain, ext_columns);

        let lifted_base_codewords = base_codewords
            .iter()
            .map(|base_codeword| base_codeword.iter().map(|bfe| bfe.lift()).collect_vec())
            .collect_vec();
        let all_codewords = vec![lifted_base_codewords, ext_codewords].concat();
        assert_eq!(self.full_width(), all_codewords.len());

        let inherited_table = self.inherited_table.with_data(all_codewords);
        ExtHashTable { inherited_table }
    }
}

#[derive(Debug, Clone)]
pub struct HashTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the hash table.
    pub from_processor_eval_row_weight: XFieldElement,
    pub to_processor_eval_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub stack_input_weights: [XFieldElement; 2 * DIGEST_LENGTH],
    pub digest_output_weights: [XFieldElement; DIGEST_LENGTH],
}

#[derive(Debug, Clone)]
pub struct HashTableEndpoints {
    /// Values randomly generated by the prover for zero-knowledge.
    pub from_processor_eval_sum: XFieldElement,
    pub to_processor_eval_sum: XFieldElement,
}

impl ExtensionTable for ExtHashTable {
    fn dynamic_boundary_constraints(&self) -> Vec<MPolynomial<XFieldElement>> {
        ExtHashTable::ext_boundary_constraints()
    }

    fn dynamic_transition_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtHashTable::ext_transition_constraints(&challenges.hash_table_challenges)
    }

    fn dynamic_consistency_constraints(&self) -> Vec<MPolynomial<XFieldElement>> {
        ExtHashTable::ext_consistency_constraints()
    }

    fn dynamic_terminal_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
        terminals: &super::challenges_endpoints::AllEndpoints<StarkHasher>,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtHashTable::ext_terminal_constraints(
            &challenges.hash_table_challenges,
            &terminals.hash_table_endpoints,
        )
    }
}

#[cfg(test)]
mod constraint_tests {
    use crate::vm::Program;

    use super::*;

    #[test]
    fn table_satisfies_constraints_test() {
        let program = Program::from_code("hash hash hash halt").unwrap();
        let (base_matrices, _, _) = program.simulate_with_input(&[], &[]);
        let (ext_hash_table, _) = HashTable::new_prover(
            0,
            base_matrices
                .hash_matrix
                .iter()
                .map(|r| r.to_vec())
                .collect(),
        )
        .extend(
            &AllChallenges::dummy().hash_table_challenges,
            &AllEndpoints::<StarkHasher>::dummy().hash_table_endpoints,
        );

        for v in ext_hash_table.evaluate_boundary_constraints(&ext_hash_table.data()[0]) {
            assert!(v.is_zero());
        }

        for (i, row) in ext_hash_table.data().iter().enumerate() {
            for (j, v) in ext_hash_table
                .evaluate_consistency_constraints(&row)
                .iter()
                .enumerate()
            {
                assert!(
                    v.is_zero(),
                    "consistency constraint {} failed in row {}: {:?}",
                    j,
                    i,
                    row
                );
            }
        }

        for (i, (current_row, next_row)) in ext_hash_table.data().iter().tuple_windows().enumerate()
        {
            let eval_point = [current_row.to_vec(), next_row.to_vec()].concat();
            for (j, v) in ext_hash_table
                .evaluate_transition_constraints(&eval_point)
                .iter()
                .enumerate()
            {
                assert!(
                    v.is_zero(),
                    "transition constraint {} failed in row {}",
                    j,
                    i
                );
            }
        }

        for v in ext_hash_table
            .evaluate_terminal_constraints(&ext_hash_table.data()[ext_hash_table.data().len() - 1])
        {
            assert!(v.is_zero());
        }
    }
}
