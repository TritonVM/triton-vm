use super::base_table::{self, BaseTable, BaseTableTrait, HasBaseTable};
use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use super::extension_table::{ExtensionTable, Quotientable, QuotientableExtensionTable};
use super::table_collection::TableId;
use super::table_column::HashTableColumn;
use crate::fri_domain::FriDomain;
use crate::state::DIGEST_LEN;
use itertools::Itertools;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::MPolynomial;
use twenty_first::shared_math::x_field_element::XFieldElement;

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

impl Quotientable for ExtHashTable {}
impl QuotientableExtensionTable for ExtHashTable {}

impl HasBaseTable<XFieldElement> for ExtHashTable {
    fn to_base(&self) -> &BaseTable<XFieldElement> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<XFieldElement> {
        &mut self.base
    }
}

impl BaseTableTrait<BWord> for HashTable {
    fn get_padding_row(&self) -> Vec<BWord> {
        vec![0.into(); BASE_WIDTH]
    }
}

impl BaseTableTrait<XFieldElement> for ExtHashTable {
    fn get_padding_row(&self) -> Vec<XFieldElement> {
        panic!("Extension tables don't get padded");
    }
}

impl ExtHashTable {
    fn ext_boundary_constraints(_challenges: &HashTableChallenges) -> Vec<MPolynomial<XWord>> {
        let variables: Vec<MPolynomial<XWord>> = MPolynomial::variables(FULL_WIDTH, 1.into());
        let one = MPolynomial::<XFieldElement>::from_constant(1.into(), FULL_WIDTH);

        let rnd_nmbr = variables[usize::from(HashTableColumn::ROUNDNUMBER)].clone();

        // 1. The round number rnd_nmbr starts at 1.
        let rnd_nmbr_starts_at_one = rnd_nmbr - one;

        vec![rnd_nmbr_starts_at_one]
    }

    fn ext_consistency_constraints(_challenges: &HashTableChallenges) -> Vec<MPolynomial<XWord>> {
        fn constant(constant: u32) -> MPolynomial<XWord> {
            MPolynomial::from_constant(constant.into(), FULL_WIDTH)
        }

        let variables: Vec<MPolynomial<XWord>> = MPolynomial::variables(FULL_WIDTH, 1.into());

        let rnd_nmbr = variables[usize::from(HashTableColumn::ROUNDNUMBER)].clone();
        let state12 = variables[usize::from(HashTableColumn::STATE12)].clone();
        let state13 = variables[usize::from(HashTableColumn::STATE13)].clone();
        let state14 = variables[usize::from(HashTableColumn::STATE14)].clone();
        let state15 = variables[usize::from(HashTableColumn::STATE15)].clone();

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

    fn ext_transition_constraints(_challenges: &HashTableChallenges) -> Vec<MPolynomial<XWord>> {
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
        _challenges: &HashTableChallenges,
        _terminals: &HashTableEndpoints,
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
            TableId::HashTable,
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

        let terminals = HashTableEndpoints {
            from_processor_eval_sum: from_processor_running_sum,
            to_processor_eval_sum: to_processor_running_sum,
        };

        let base = self.base.with_lifted_data(extension_matrix);
        let table = BaseTable::extension(
            base,
            ExtHashTable::ext_boundary_constraints(challenges),
            ExtHashTable::ext_transition_constraints(&challenges),
            ExtHashTable::ext_consistency_constraints(&challenges),
            ExtHashTable::ext_terminal_constraints(&challenges, &terminals),
        );

        (ExtHashTable { base: table }, terminals)
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
            TableId::HashTable,
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

    pub fn for_verifier(
        num_trace_randomizers: usize,
        padded_height: usize,
        all_challenges: &AllChallenges,
        all_terminals: &AllEndpoints,
    ) -> Self {
        let omicron = base_table::derive_omicron(padded_height as u64);
        let base = BaseTable::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            vec![],
            "ExtHashTable".to_string(),
            TableId::HashTable,
        );
        let table = BaseTable::extension(
            base,
            ExtHashTable::ext_boundary_constraints(&all_challenges.hash_table_challenges),
            ExtHashTable::ext_transition_constraints(&all_challenges.hash_table_challenges),
            ExtHashTable::ext_consistency_constraints(&all_challenges.hash_table_challenges),
            ExtHashTable::ext_terminal_constraints(
                &all_challenges.hash_table_challenges,
                &all_terminals.hash_table_endpoints,
            ),
        );

        ExtHashTable { base: table }
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

impl ExtensionTable for ExtHashTable {
    fn dynamic_boundary_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtHashTable::ext_boundary_constraints(&challenges.hash_table_challenges)
    }

    fn dynamic_transition_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtHashTable::ext_transition_constraints(&challenges.hash_table_challenges)
    }

    fn dynamic_consistency_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtHashTable::ext_consistency_constraints(&challenges.hash_table_challenges)
    }

    fn dynamic_terminal_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
        terminals: &super::challenges_endpoints::AllEndpoints,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtHashTable::ext_terminal_constraints(
            &challenges.hash_table_challenges,
            &terminals.hash_table_endpoints,
        )
    }
}
