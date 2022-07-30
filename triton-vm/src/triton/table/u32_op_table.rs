use super::base_table::{self, BaseTable, HasBaseTable, Table};
use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use super::extension_table::ExtensionTable;
use super::table_column::U32OpTableColumn;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::stark::triton::fri_domain::FriDomain;
use crate::shared_math::x_field_element::XFieldElement;
use itertools::Itertools;

pub const U32_OP_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 5;
pub const U32_OP_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 0;
pub const U32_OP_TABLE_INITIALS_COUNT: usize =
    U32_OP_TABLE_PERMUTATION_ARGUMENTS_COUNT + U32_OP_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 14 because it combines: (lt, and, xor, div) x (lhs, rhs, result) + (rev) x (lhs, result)
pub const U32_OP_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 14;

pub const BASE_WIDTH: usize = 7;
pub const FULL_WIDTH: usize = 17; // BASE_WIDTH + 2 * INITIALS_COUNT

type BWord = BFieldElement;
type XWord = XFieldElement;

#[derive(Debug, Clone)]
pub struct U32OpTable {
    base: BaseTable<BWord>,
}

impl HasBaseTable<BWord> for U32OpTable {
    fn to_base(&self) -> &BaseTable<BWord> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<BWord> {
        &mut self.base
    }
}

#[derive(Debug, Clone)]
pub struct ExtU32OpTable {
    base: BaseTable<XFieldElement>,
}

impl HasBaseTable<XFieldElement> for ExtU32OpTable {
    fn to_base(&self) -> &BaseTable<XFieldElement> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<XFieldElement> {
        &mut self.base
    }
}

impl Table<BWord> for U32OpTable {
    fn get_padding_row(&self) -> Vec<BWord> {
        vec![0.into(); BASE_WIDTH]
    }
}

impl Table<XFieldElement> for ExtU32OpTable {
    fn get_padding_row(&self) -> Vec<XFieldElement> {
        panic!("Extension tables don't get padded");
    }
}

impl ExtensionTable for ExtU32OpTable {
    fn ext_boundary_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        vec![]
    }

    fn ext_consistency_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        vec![]
    }

    fn ext_transition_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        vec![]
    }

    fn ext_terminal_constraints(
        &self,
        _challenges: &AllChallenges,
        _terminals: &AllEndpoints,
    ) -> Vec<MPolynomial<XWord>> {
        vec![]
    }
}

impl U32OpTable {
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
            "U32OpTable".to_string(),
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
        challenges: &U32OpTableChallenges,
        initials: &U32OpTableEndpoints,
    ) -> (ExtU32OpTable, U32OpTableEndpoints) {
        let mut lt_running_product = initials.processor_lt_perm_product;
        let mut and_running_product = initials.processor_and_perm_product;
        let mut xor_running_product = initials.processor_xor_perm_product;
        let mut reverse_running_product = initials.processor_reverse_perm_product;
        let mut div_running_product = initials.processor_div_perm_product;

        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        for row in self.data().iter() {
            let mut extension_row = Vec::with_capacity(FULL_WIDTH);
            extension_row.extend(row.iter().map(|elem| elem.lift()));

            // lhs and rhs are needed for _all_ of U32Table's Permutation Arguments
            let lhs = extension_row[U32OpTableColumn::LHS as usize];
            let rhs = extension_row[U32OpTableColumn::RHS as usize];

            // Compress (lhs, rhs, lt) into single value
            let lt = extension_row[U32OpTableColumn::LT as usize];
            let compressed_row_for_lt = lhs * challenges.lt_lhs_weight
                + rhs * challenges.lt_rhs_weight
                + lt * challenges.lt_result_weight;
            extension_row.push(compressed_row_for_lt);

            // Multiply compressed value into running product for lt
            extension_row.push(lt_running_product);
            lt_running_product *= challenges.processor_lt_perm_row_weight - compressed_row_for_lt;

            // Compress (lhs, rhs, and) into single value
            let and = extension_row[U32OpTableColumn::AND as usize];
            let compressed_row_for_and = lhs * challenges.and_lhs_weight
                + rhs * challenges.and_rhs_weight
                + and * challenges.and_result_weight;
            extension_row.push(compressed_row_for_and);

            // Multiply compressed value into running product for and
            extension_row.push(and_running_product);
            and_running_product *=
                challenges.processor_and_perm_row_weight - compressed_row_for_and;

            // Compress (lhs, rhs, xor) into single value
            let xor = extension_row[U32OpTableColumn::XOR as usize];
            let compressed_row_for_xor = lhs * challenges.xor_lhs_weight
                + rhs * challenges.xor_rhs_weight
                + xor * challenges.xor_result_weight;
            extension_row.push(compressed_row_for_xor);

            // Multiply compressed value into running product for xor
            extension_row.push(xor_running_product);
            xor_running_product *=
                challenges.processor_xor_perm_row_weight - compressed_row_for_xor;

            // Compress (lhs, reverse) into single value
            let reverse = extension_row[U32OpTableColumn::REV as usize];
            let compressed_row_for_reverse =
                lhs * challenges.reverse_lhs_weight + reverse * challenges.reverse_result_weight;
            extension_row.push(compressed_row_for_reverse);

            // Multiply compressed value into running product for reverse
            extension_row.push(reverse_running_product);
            reverse_running_product *=
                challenges.processor_reverse_perm_row_weight - compressed_row_for_reverse;

            // Compress (lhs, rhs, lt) into single value for div
            let lt_for_div = extension_row[U32OpTableColumn::LT as usize];
            let compressed_row_for_div = lhs * challenges.div_divisor_weight
                + rhs * challenges.div_remainder_weight
                + lt_for_div * challenges.div_result_weight;
            extension_row.push(compressed_row_for_div);

            // Multiply compressed value into running product for div
            extension_row.push(div_running_product);
            div_running_product *=
                challenges.processor_div_perm_row_weight - compressed_row_for_div;

            extension_matrix.push(extension_row);
        }

        let base = self.base.with_lifted_data(extension_matrix);
        let table = ExtU32OpTable { base };
        let terminals = U32OpTableEndpoints {
            processor_lt_perm_product: lt_running_product,
            processor_and_perm_product: and_running_product,
            processor_xor_perm_product: xor_running_product,
            processor_reverse_perm_product: reverse_running_product,
            processor_div_perm_product: div_running_product,
        };

        (table, terminals)
    }
}

impl ExtU32OpTable {
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
            "ExtU32OpTable".to_string(),
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
        ExtU32OpTable { base }
    }
}

#[derive(Debug, Clone)]
pub struct U32OpTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the u32 op table.
    pub processor_lt_perm_row_weight: XFieldElement,
    pub processor_and_perm_row_weight: XFieldElement,
    pub processor_xor_perm_row_weight: XFieldElement,
    pub processor_reverse_perm_row_weight: XFieldElement,
    pub processor_div_perm_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub lt_lhs_weight: XFieldElement,
    pub lt_rhs_weight: XFieldElement,
    pub lt_result_weight: XFieldElement,

    pub and_lhs_weight: XFieldElement,
    pub and_rhs_weight: XFieldElement,
    pub and_result_weight: XFieldElement,

    pub xor_lhs_weight: XFieldElement,
    pub xor_rhs_weight: XFieldElement,
    pub xor_result_weight: XFieldElement,

    pub reverse_lhs_weight: XFieldElement,
    pub reverse_result_weight: XFieldElement,

    pub div_divisor_weight: XFieldElement,
    pub div_remainder_weight: XFieldElement,
    pub div_result_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct U32OpTableEndpoints {
    /// Values randomly generated by the prover for zero-knowledge.
    pub processor_lt_perm_product: XFieldElement,
    pub processor_and_perm_product: XFieldElement,
    pub processor_xor_perm_product: XFieldElement,
    pub processor_reverse_perm_product: XFieldElement,
    pub processor_div_perm_product: XFieldElement,
}
