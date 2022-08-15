use super::base_table::{self, BaseTable, BaseTableTrait, HasBaseTable};
use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use super::extension_table::{ExtensionTable, Quotientable, QuotientableExtensionTable};
use super::table_collection::TableId;
use super::table_column::U32OpTableColumn;
use crate::fri_domain::FriDomain;
use crate::instruction::Instruction;
use itertools::Itertools;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::MPolynomial;
use twenty_first::shared_math::traits::IdentityValues;
use twenty_first::shared_math::x_field_element::XFieldElement;

pub const U32_OP_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 1;
pub const U32_OP_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 0;
pub const U32_OP_TABLE_INITIALS_COUNT: usize =
    U32_OP_TABLE_PERMUTATION_ARGUMENTS_COUNT + U32_OP_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 4 because it combines: ci, lhs, rhs, result
pub const U32_OP_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 4;

pub const BASE_WIDTH: usize = 8;
pub const FULL_WIDTH: usize = 10; // BASE_WIDTH + 2 * INITIALS_COUNT

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

impl Quotientable for ExtU32OpTable {}
impl QuotientableExtensionTable for ExtU32OpTable {}

impl HasBaseTable<XFieldElement> for ExtU32OpTable {
    fn to_base(&self) -> &BaseTable<XFieldElement> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<XFieldElement> {
        &mut self.base
    }
}

impl BaseTableTrait<BWord> for U32OpTable {
    fn get_padding_row(&self) -> Vec<BWord> {
        vec![0.into(); BASE_WIDTH]
    }
}

impl BaseTableTrait<XFieldElement> for ExtU32OpTable {
    fn get_padding_row(&self) -> Vec<XFieldElement> {
        panic!("Extension tables don't get padded");
    }
}

impl ExtU32OpTable {
    fn ext_boundary_constraints(_challenges: &U32OpTableChallenges) -> Vec<MPolynomial<XWord>> {
        vec![]
    }

    // TODO actually use consistency constraints
    fn ext_consistency_constraints(_challenges: &U32OpTableChallenges) -> Vec<MPolynomial<XWord>> {
        vec![]
    }

    fn ext_transition_constraints(_challenges: &U32OpTableChallenges) -> Vec<MPolynomial<XWord>> {
        vec![]
    }

    fn ext_terminal_constraints(
        _challenges: &U32OpTableChallenges,
        _terminals: &U32OpTableEndpoints,
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
            TableId::U32OpTable,
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
        let mut running_product = initials.processor_perm_product;
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());

        for row in self.data().iter() {
            let mut extension_row = Vec::with_capacity(FULL_WIDTH);
            extension_row.extend(row.iter().map(|elem| elem.lift()));

            let current_instruction: Instruction = row[U32OpTableColumn::CI as usize]
                .value()
                .try_into()
                .expect("CI does not correspond to any instruction.");

            // ci, lhs, and rhs are needed for _all_ of U32Table's Permutation Arguments
            let ci = extension_row[U32OpTableColumn::CI as usize];
            let lhs = extension_row[U32OpTableColumn::LHS as usize];
            let rhs = match current_instruction {
                Instruction::Reverse => 0.into(),
                _ => extension_row[U32OpTableColumn::RHS as usize],
            };
            let result = match current_instruction {
                Instruction::Lt => extension_row[U32OpTableColumn::LT as usize],
                Instruction::And => extension_row[U32OpTableColumn::AND as usize],
                Instruction::Xor => extension_row[U32OpTableColumn::XOR as usize],
                Instruction::Reverse => extension_row[U32OpTableColumn::REV as usize],
                Instruction::Div => extension_row[U32OpTableColumn::LT as usize],
                // halt is used for padding
                Instruction::Halt => XFieldElement::ring_zero(),
                x => panic!("Unknown instruction '{x}' in the U32 Table."),
            };

            // Compress (ci, lhs, rhs, result) into single value
            let compressed_row = ci * challenges.ci_weight
                + lhs * challenges.lhs_weight
                + rhs * challenges.rhs_weight
                + result * challenges.result_weight;
            extension_row.push(compressed_row);

            // Multiply compressed value into running product if indicator is set
            extension_row.push(running_product);
            if row[U32OpTableColumn::IDC as usize].is_one() {
                running_product *= challenges.processor_perm_row_weight - compressed_row;
            }

            extension_matrix.push(extension_row);
        }

        let terminals = U32OpTableEndpoints {
            processor_perm_product: running_product,
        };

        let base = self.base.with_lifted_data(extension_matrix);
        let table = BaseTable::extension(
            base,
            ExtU32OpTable::ext_boundary_constraints(challenges),
            ExtU32OpTable::ext_transition_constraints(&challenges),
            ExtU32OpTable::ext_consistency_constraints(&challenges),
            ExtU32OpTable::ext_terminal_constraints(&challenges, &terminals),
        );

        (ExtU32OpTable { base: table }, terminals)
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
            TableId::U32OpTable,
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
            "ExtU32OpTable".to_string(),
            TableId::U32OpTable,
        );
        let table = BaseTable::extension(
            base,
            ExtU32OpTable::ext_boundary_constraints(&all_challenges.u32_op_table_challenges),
            ExtU32OpTable::ext_transition_constraints(&all_challenges.u32_op_table_challenges),
            ExtU32OpTable::ext_consistency_constraints(&all_challenges.u32_op_table_challenges),
            ExtU32OpTable::ext_terminal_constraints(
                &all_challenges.u32_op_table_challenges,
                &all_terminals.u32_op_table_endpoints,
            ),
        );

        ExtU32OpTable { base: table }
    }
}

#[derive(Debug, Clone)]
pub struct U32OpTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the u32 op table.
    pub processor_perm_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub lhs_weight: XFieldElement,
    pub rhs_weight: XFieldElement,
    pub ci_weight: XFieldElement,
    pub result_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct U32OpTableEndpoints {
    /// Values randomly generated by the prover for zero-knowledge.
    pub processor_perm_product: XFieldElement,
}

impl ExtensionTable for ExtU32OpTable {
    fn dynamic_boundary_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtU32OpTable::ext_boundary_constraints(&challenges.u32_op_table_challenges)
    }

    fn dynamic_transition_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtU32OpTable::ext_transition_constraints(&challenges.u32_op_table_challenges)
    }

    fn dynamic_consistency_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtU32OpTable::ext_consistency_constraints(&challenges.u32_op_table_challenges)
    }

    fn dynamic_terminal_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
        terminals: &super::challenges_endpoints::AllEndpoints,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtU32OpTable::ext_terminal_constraints(
            &challenges.u32_op_table_challenges,
            &terminals.u32_op_table_endpoints,
        )
    }
}
