use itertools::Itertools;
use num_traits::{One, Zero};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::MPolynomial;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::fri_domain::FriDomain;
use crate::stark::StarkHasher;
use crate::table::base_table::Extendable;
use crate::table::extension_table::Evaluable;

use super::base_table::{self, InheritsFromTable, Table, TableLike};
use super::challenges_endpoints::{AllChallenges, AllTerminals};
use super::extension_table::{ExtensionTable, Quotientable, QuotientableExtensionTable};
use super::table_column::OpStackTableColumn;

pub const OP_STACK_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 1;
pub const OP_STACK_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 0;
pub const OP_STACK_TABLE_INITIALS_COUNT: usize =
    OP_STACK_TABLE_PERMUTATION_ARGUMENTS_COUNT + OP_STACK_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 4 because it combines: clk, ci, osv, osp
pub const OP_STACK_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 4;

pub const BASE_WIDTH: usize = 4;
pub const FULL_WIDTH: usize = 6; // BASE_WIDTH + 2 * INITIALS_COUNT

#[derive(Debug, Clone)]
pub struct OpStackTable {
    inherited_table: Table<BFieldElement>,
}

impl InheritsFromTable<BFieldElement> for OpStackTable {
    fn inherited_table(&self) -> &Table<BFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<BFieldElement> {
        &mut self.inherited_table
    }
}

#[derive(Debug, Clone)]
pub struct ExtOpStackTable {
    inherited_table: Table<XFieldElement>,
}

impl Evaluable for ExtOpStackTable {}
impl Quotientable for ExtOpStackTable {}
impl QuotientableExtensionTable for ExtOpStackTable {}

impl InheritsFromTable<XFieldElement> for ExtOpStackTable {
    fn inherited_table(&self) -> &Table<XFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<XFieldElement> {
        &mut self.inherited_table
    }
}

impl TableLike<BFieldElement> for OpStackTable {}

impl Extendable for OpStackTable {
    fn get_padding_rows(&self) -> (Option<usize>, Vec<Vec<BFieldElement>>) {
        let max_clock = self.data().len() as u64 - 1;
        if let Some((idx, padding_template)) = self
            .data()
            .iter()
            .enumerate()
            .find(|(_, row)| row[OpStackTableColumn::CLK as usize].value() == max_clock)
        {
            let mut padding_row = padding_template.clone();
            padding_row[OpStackTableColumn::CLK as usize] += BFieldElement::one();
            (Some(idx + 1), vec![padding_row])
        } else {
            let mut padding_row = vec![BFieldElement::zero(); BASE_WIDTH];
            padding_row[OpStackTableColumn::OSP as usize] = BFieldElement::new(16);
            (None, vec![padding_row])
        }
    }
}

impl TableLike<XFieldElement> for ExtOpStackTable {}

impl ExtOpStackTable {
    fn ext_initial_constraints() -> Vec<MPolynomial<XFieldElement>> {
        use OpStackTableColumn::*;

        let variables: Vec<MPolynomial<XFieldElement>> =
            MPolynomial::variables(FULL_WIDTH, 1.into());
        let clk = variables[CLK as usize].clone();
        let osv = variables[OSV as usize].clone();
        let osp = variables[OSP as usize].clone();
        let sixteen = MPolynomial::from_constant(16.into(), FULL_WIDTH);

        // 1. clk is 0.
        let clk_is_0 = clk;

        // 2. osv is 0.
        let osv_is_0 = osv;

        // 3. osp is the number of available stack registers, i.e., 16.
        let osp_is_16 = osp - sixteen;

        vec![clk_is_0, osv_is_0, osp_is_16]
    }

    fn ext_consistency_constraints(
        _challenges: &OpStackTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }

    fn ext_transition_constraints(
        _challenges: &OpStackTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        use OpStackTableColumn::*;

        let variables: Vec<MPolynomial<XFieldElement>> =
            MPolynomial::variables(2 * FULL_WIDTH, 1.into());
        // let clk = variables[usize::from(CLK)].clone();
        let ib1_shrink_stack = variables[usize::from(IB1ShrinkStack)].clone();
        let osv = variables[usize::from(OSV)].clone();
        let osp = variables[usize::from(OSP)].clone();
        let osp_next = variables[FULL_WIDTH + usize::from(OSP)].clone();
        let osv_next = variables[FULL_WIDTH + usize::from(OSV)].clone();
        let one = MPolynomial::from_constant(1.into(), FULL_WIDTH);

        // the osp increases by 1 or the osp does not change
        //
        // $(osp' - (osp + 1))·(osp' - osp) = 0$
        let osp_increases_by_1_or_does_not_change =
            (osp_next.clone() - (osp.clone() + one.clone())) * (osp_next.clone() - osp.clone());

        // the osp increases by 1 or the osv does not change OR the ci shrinks the OpStack
        //
        // $ (osp' - (osp + 1)) · (osv' - osv) · (1 - ib1) = 0$
        let osp_increases_by_1_or_osv_does_not_change_or_shrink_stack =
            (osp_next - (osp + one.clone())) * (osv_next - osv) * (one - ib1_shrink_stack);

        vec![
            osp_increases_by_1_or_does_not_change,
            osp_increases_by_1_or_osv_does_not_change_or_shrink_stack,
        ]
    }

    fn ext_terminal_constraints(
        _challenges: &OpStackTableChallenges,
        _terminals: &OpStackTableEndpoints,
    ) -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }
}

impl OpStackTable {
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
            "OpStackTable".to_string(),
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
        challenges: &OpStackTableChallenges,
        initials: &OpStackTableEndpoints,
    ) -> (ExtOpStackTable, OpStackTableEndpoints) {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        let mut running_product = initials.processor_perm_product;

        for row in self.data().iter() {
            let mut extension_row = Vec::with_capacity(FULL_WIDTH);
            extension_row.extend(row.iter().map(|elem| elem.lift()));

            let (clk, ci, osv, osp) = (
                extension_row[OpStackTableColumn::CLK as usize],
                extension_row[OpStackTableColumn::IB1ShrinkStack as usize],
                extension_row[OpStackTableColumn::OSV as usize],
                extension_row[OpStackTableColumn::OSP as usize],
            );

            let (clk_w, ci_w, osp_w, osv_w) = (
                challenges.clk_weight,
                challenges.ci_weight,
                challenges.osv_weight,
                challenges.osp_weight,
            );

            // 1. Compress multiple values within one row so they become one value.
            let compressed_row_for_permutation_argument =
                clk * clk_w + ci * ci_w + osv * osv_w + osp * osp_w;

            extension_row.push(compressed_row_for_permutation_argument);

            // 2. Compute the running *product* of the compressed column (permutation value)
            extension_row.push(running_product);
            running_product *=
                challenges.processor_perm_row_weight - compressed_row_for_permutation_argument;

            extension_matrix.push(extension_row);
        }

        let terminals = OpStackTableEndpoints {
            processor_perm_product: running_product,
        };

        let inherited_table = self.extension(
            extension_matrix,
            ExtOpStackTable::ext_initial_constraints(),
            ExtOpStackTable::ext_consistency_constraints(challenges),
            ExtOpStackTable::ext_transition_constraints(challenges),
            ExtOpStackTable::ext_terminal_constraints(challenges, &terminals),
        );
        (ExtOpStackTable { inherited_table }, terminals)
    }

    pub fn for_verifier(
        num_trace_randomizers: usize,
        padded_height: usize,
        all_challenges: &AllChallenges,
        all_terminals: &AllTerminals<StarkHasher>,
    ) -> ExtOpStackTable {
        let omicron = base_table::derive_omicron(padded_height as u64);
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            vec![],
            "ExtOpStackTable".to_string(),
        );
        let base_table = Self { inherited_table };
        let empty_matrix: Vec<Vec<XFieldElement>> = vec![];
        let extension_table = base_table.extension(
            empty_matrix,
            ExtOpStackTable::ext_initial_constraints(),
            ExtOpStackTable::ext_consistency_constraints(&all_challenges.op_stack_table_challenges),
            ExtOpStackTable::ext_transition_constraints(&all_challenges.op_stack_table_challenges),
            ExtOpStackTable::ext_terminal_constraints(
                &all_challenges.op_stack_table_challenges,
                &all_terminals.op_stack_table_endpoints,
            ),
        );

        ExtOpStackTable {
            inherited_table: extension_table,
        }
    }
}

impl ExtOpStackTable {
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
            "ExtOpStackTable".to_string(),
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
        ExtOpStackTable { inherited_table }
    }
}

#[derive(Debug, Clone)]
pub struct OpStackTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the op-stack table.
    pub processor_perm_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub clk_weight: XFieldElement,
    pub ci_weight: XFieldElement,
    pub osv_weight: XFieldElement,
    pub osp_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct OpStackTableEndpoints {
    /// Values randomly generated by the prover for zero-knowledge.
    pub processor_perm_product: XFieldElement,
}

impl ExtensionTable for ExtOpStackTable {
    fn dynamic_initial_constraints(&self) -> Vec<MPolynomial<XFieldElement>> {
        ExtOpStackTable::ext_initial_constraints()
    }

    fn dynamic_consistency_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtOpStackTable::ext_consistency_constraints(&challenges.op_stack_table_challenges)
    }

    fn dynamic_transition_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtOpStackTable::ext_transition_constraints(&challenges.op_stack_table_challenges)
    }

    fn dynamic_terminal_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
        terminals: &super::challenges_endpoints::AllTerminals<StarkHasher>,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtOpStackTable::ext_terminal_constraints(
            &challenges.op_stack_table_challenges,
            &terminals.op_stack_table_endpoints,
        )
    }
}
