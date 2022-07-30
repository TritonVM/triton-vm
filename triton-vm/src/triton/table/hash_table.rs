use super::base_table::{self, BaseTable, HasBaseTable, Table};
use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use super::extension_table::ExtensionTable;
use super::table_column::HashTableColumn;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::stark::triton::fri_domain::FriDomain;
use crate::shared_math::stark::triton::state::DIGEST_LEN;
use crate::shared_math::x_field_element::XFieldElement;
use itertools::Itertools;

pub const HASH_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 0;
pub const HASH_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 2;
pub const HASH_TABLE_INITIALS_COUNT: usize =
    HASH_TABLE_PERMUTATION_ARGUMENTS_COUNT + HASH_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 18 because it combines: 12 stack_input_weights and 6 digest_output_weights.
pub const HASH_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 18;

pub const BASE_WIDTH: usize = 17;
pub const FULL_WIDTH: usize = 21; // BASE_WIDTH + 2 * INITIALS_COUNT

type BWord = BFieldElement;
type XWord = XFieldElement;

#[derive(Debug, Clone)]
pub struct HashTable {
    base: BaseTable<BWord>,
}

impl HasBaseTable<BWord> for HashTable {
    fn to_base(&self) -> &BaseTable<BWord> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<BWord> {
        &mut self.base
    }
}

#[derive(Debug, Clone)]
pub struct ExtHashTable {
    base: BaseTable<XFieldElement>,
}

impl HasBaseTable<XFieldElement> for ExtHashTable {
    fn to_base(&self) -> &BaseTable<XFieldElement> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<XFieldElement> {
        &mut self.base
    }
}

impl Table<BWord> for HashTable {
    fn get_padding_row(&self) -> Vec<BWord> {
        vec![0.into(); BASE_WIDTH]
    }
}

impl Table<XFieldElement> for ExtHashTable {
    fn get_padding_row(&self) -> Vec<XFieldElement> {
        panic!("Extension tables don't get padded");
    }
}

impl ExtensionTable for ExtHashTable {
    fn ext_boundary_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        let variables: Vec<MPolynomial<XWord>> = MPolynomial::variables(FULL_WIDTH, 1.into());
        let one = MPolynomial::<XFieldElement>::from_constant(1.into(), FULL_WIDTH);

        let rnd_nmbr = variables[usize::from(HashTableColumn::ROUNDNUMBER)].clone();

        // 1. The round number rnd_nmbr starts at 1.
        let rnd_nmbr_starts_at_one = rnd_nmbr - one;

        // TODO this constraint breaks – figure out why!
        vec![rnd_nmbr_starts_at_one];
        vec![]
    }

    fn ext_consistency_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        fn constant(constant: u32) -> MPolynomial<XWord> {
            MPolynomial::from_constant(constant.into(), FULL_WIDTH)
        }

        let variables: Vec<MPolynomial<XWord>> = MPolynomial::variables(FULL_WIDTH, 1.into());

        let rnd_nmbr = variables[usize::from(HashTableColumn::ROUNDNUMBER)].clone();
        let state12 = variables[usize::from(HashTableColumn::STATE12)].clone();
        let state13 = variables[usize::from(HashTableColumn::STATE13)].clone();
        let state14 = variables[usize::from(HashTableColumn::STATE14)].clone();
        let state15 = variables[usize::from(HashTableColumn::STATE15)].clone();

        // Common factor:
        /*
            (rnd_nmbr.clone() - constant(0))
            * (rnd_nmbr.clone() - constant(2))
            * (rnd_nmbr.clone() - constant(3))
            * (rnd_nmbr.clone() - constant(4))
            * (rnd_nmbr.clone() - constant(5))
            * (rnd_nmbr.clone() - constant(6))
            * (rnd_nmbr.clone() - constant(7))
            * (rnd_nmbr.clone() - constant(8));
        */

        let common_factor = (0..=0)
            .chain(2..=8)
            .into_iter()
            .map(|n| rnd_nmbr.clone() - constant(n))
            .fold(constant(1), |acc, x| acc * x);

        // 1. If the round number is 1, register state12 is 0.
        let if_rnd_nmbr_is_1_then_state12_is_0 = common_factor.clone() * state12;

        // 2. If the round number is 1, register state13 is 0.
        let if_rnd_nmbr_is_1_then_state13_is_0 = common_factor.clone() * state13;

        // 3. If the round number is 1, register state14 is 0.
        let if_rnd_nmbr_is_1_then_state14_is_0 = common_factor.clone() * state14;

        // 4. If the round number is 1, register state15 is 0.
        let if_rnd_nmbr_is_1_then_state15_is_0 = common_factor * state15;

        vec![
            if_rnd_nmbr_is_1_then_state12_is_0,
            if_rnd_nmbr_is_1_then_state13_is_0,
            if_rnd_nmbr_is_1_then_state14_is_0,
            if_rnd_nmbr_is_1_then_state15_is_0,
        ]
    }

    fn ext_transition_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        fn constant(constant: u32) -> MPolynomial<XWord> {
            MPolynomial::from_constant(constant.into(), 2 * FULL_WIDTH)
        }

        let variables: Vec<MPolynomial<XWord>> = MPolynomial::variables(2 * FULL_WIDTH, 1.into());

        let rnd_nmbr = variables[usize::from(HashTableColumn::ROUNDNUMBER)].clone();
        let rnd_nmbr_next =
            variables[FULL_WIDTH + usize::from(HashTableColumn::ROUNDNUMBER)].clone();

        let helper = |rnd_nmbr_arg, rnd_nmbr_next_arg| -> MPolynomial<XFieldElement> {
            let mut prod = rnd_nmbr_next.clone() - constant(rnd_nmbr_next_arg);
            for i in 0..=8 {
                if i != rnd_nmbr_arg {
                    prod *= rnd_nmbr.clone() - constant(i)
                }
            }
            prod
        };

        // 1. If the round number is 0, the next round number is 0.
        let if_rnd_nmbr_is_0_then_next_rnd_nmbr_is_0 = helper(0, 0);
        // 2. If the round number is 1, the next round number is 2.
        let if_rnd_nmbr_is_1_then_next_rnd_nmbr_is_2 = helper(1, 2);
        // 3. If the round number is 2, the next round number is 3.
        let if_rnd_nmbr_is_2_then_next_rnd_nmbr_is_3 = helper(2, 3);
        // 4. If the round number is 3, the next round number is 4.
        let if_rnd_nmbr_is_3_then_next_rnd_nmbr_is_4 = helper(3, 4);
        // 5. If the round number is 4, the next round number is 5.
        let if_rnd_nmbr_is_4_then_next_rnd_nmbr_is_5 = helper(4, 5);
        // 6. If the round number is 5, the next round number is 6.
        let if_rnd_nmbr_is_5_then_next_rnd_nmbr_is_6 = helper(5, 6);
        // 7. If the round number is 6, the next round number is 7.
        let if_rnd_nmbr_is_6_then_next_rnd_nmbr_is_7 = helper(6, 7);
        // 8. If the round number is 7, the next round number is 8.
        let if_rnd_nmbr_is_7_then_next_rnd_nmbr_is_8 = helper(7, 8);
        // 9. If the round number is 8, the next round number is either 0 or 1.
        let if_rnd_nmbr_is_8_then_next_rnd_nmbr_is_0_or_1 =
            helper(8, 0) * (rnd_nmbr_next - constant(1));
        // 10. The remaining 7·16 = 112 constraints are left as an exercise to the reader.
        // TODO
        //let _remaining = todo!();

        vec![
            if_rnd_nmbr_is_0_then_next_rnd_nmbr_is_0,
            if_rnd_nmbr_is_1_then_next_rnd_nmbr_is_2,
            if_rnd_nmbr_is_2_then_next_rnd_nmbr_is_3,
            if_rnd_nmbr_is_3_then_next_rnd_nmbr_is_4,
            if_rnd_nmbr_is_4_then_next_rnd_nmbr_is_5,
            if_rnd_nmbr_is_5_then_next_rnd_nmbr_is_6,
            if_rnd_nmbr_is_6_then_next_rnd_nmbr_is_7,
            if_rnd_nmbr_is_7_then_next_rnd_nmbr_is_8,
            if_rnd_nmbr_is_8_then_next_rnd_nmbr_is_0_or_1,
        ]
    }

    fn ext_terminal_constraints(
        &self,
        _challenges: &AllChallenges,
        _terminals: &AllEndpoints,
    ) -> Vec<MPolynomial<XWord>> {
        vec![]
    }
}

impl HashTable {
    pub fn new_prover(num_trace_randomizers: usize, matrix: Vec<Vec<BWord>>) -> Self {
        let unpadded_height = matrix.len();
        let padded_height = base_table::pad_height(unpadded_height, num_trace_randomizers);

        let omicron = base_table::derive_omicron(padded_height as u64);
        let base = BaseTable::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            matrix,
            "HashTable".to_string(),
        );

        Self { base }
    }

    pub fn codeword_table(&self, fri_domain: &FriDomain<BWord>) -> Self {
        let base_columns = 0..self.base_width();
        let codewords =
            self.low_degree_extension(fri_domain, self.num_trace_randomizers(), base_columns);

        let base = self.base.with_data(codewords);
        Self { base }
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
                .fold(XWord::ring_zero(), |sum, summand| sum + summand);
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
                .fold(XWord::ring_zero(), |sum, summand| sum + summand);
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

        let base = self.base.with_lifted_data(extension_matrix);
        let table = ExtHashTable { base };
        let terminals = HashTableEndpoints {
            from_processor_eval_sum: from_processor_running_sum,
            to_processor_eval_sum: to_processor_running_sum,
        };

        (table, terminals)
    }
}

impl ExtHashTable {
    pub fn with_padded_height(num_trace_randomizers: usize, padded_height: usize) -> Self {
        let matrix: Vec<Vec<XWord>> = vec![];

        let omicron = base_table::derive_omicron(padded_height as u64);
        let base = BaseTable::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            matrix,
            "ExtHashTable".to_string(),
        );

        Self { base }
    }

    pub fn ext_codeword_table(
        &self,
        fri_domain: &FriDomain<XWord>,
        base_codewords: &[Vec<BWord>],
    ) -> Self {
        // Extension Tables do not have a randomized trace
        let num_trace_randomizers = 0;
        let ext_columns = self.base_width()..self.full_width();
        let ext_codewords =
            self.low_degree_extension(fri_domain, num_trace_randomizers, ext_columns);

        let lifted_base_codewords = base_codewords
            .iter()
            .map(|base_codeword| base_codeword.iter().map(|bfe| bfe.lift()).collect_vec())
            .collect_vec();
        let all_codewords = vec![lifted_base_codewords, ext_codewords].concat();
        assert_eq!(self.full_width(), all_codewords.len());

        let base = self.base.with_data(all_codewords);
        ExtHashTable { base }
    }
}

#[derive(Debug, Clone)]
pub struct HashTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the hash table.
    pub from_processor_eval_row_weight: XFieldElement,
    pub to_processor_eval_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub stack_input_weights: [XFieldElement; 2 * DIGEST_LEN],
    pub digest_output_weights: [XFieldElement; DIGEST_LEN],
}

#[derive(Debug, Clone)]
pub struct HashTableEndpoints {
    /// Values randomly generated by the prover for zero-knowledge.
    pub from_processor_eval_sum: XFieldElement,
    pub to_processor_eval_sum: XFieldElement,
}
