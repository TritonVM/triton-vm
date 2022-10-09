use itertools::Itertools;
use num_traits::{One, Zero};
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::{Degree, MPolynomial};
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::cross_table_arguments::{CrossTableArg, PermArg};
use crate::fri_domain::FriDomain;
use crate::table::base_table::Extendable;
use crate::table::extension_table::Evaluable;
use crate::table::table_column::OpStackBaseTableColumn::{self, *};
use crate::table::table_column::OpStackExtTableColumn::{self, *};

use super::base_table::{InheritsFromTable, Table, TableLike};
use super::challenges::AllChallenges;
use super::extension_table::{ExtensionTable, Quotientable, QuotientableExtensionTable};

pub const OP_STACK_TABLE_NUM_PERMUTATION_ARGUMENTS: usize = 1;
pub const OP_STACK_TABLE_NUM_EVALUATION_ARGUMENTS: usize = 0;

/// This is 4 because it combines: clk, ci, osv, osp
pub const OP_STACK_TABLE_NUM_EXTENSION_CHALLENGES: usize = 4;

pub const BASE_WIDTH: usize = OpStackBaseTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + OpStackExtTableColumn::COUNT;

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

impl Default for ExtOpStackTable {
    fn default() -> Self {
        Self {
            inherited_table: Table::new(
                BASE_WIDTH,
                FULL_WIDTH,
                vec![],
                "EmptyExtOpStackTable".to_string(),
            ),
        }
    }
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
            .find(|(_, row)| row[usize::from(OpStackBaseTableColumn::CLK)].value() == max_clock)
        {
            let mut padding_row = padding_template.clone();
            padding_row[usize::from(OpStackBaseTableColumn::CLK)] += BFieldElement::one();
            (Some(idx + 1), vec![padding_row])
        } else {
            let mut padding_row = vec![BFieldElement::zero(); BASE_WIDTH];
            padding_row[usize::from(OpStackBaseTableColumn::OSP)] = BFieldElement::new(16);
            (None, vec![padding_row])
        }
    }
}

impl TableLike<XFieldElement> for ExtOpStackTable {}

impl ExtOpStackTable {
    fn ext_initial_constraints(
        _challenges: &OpStackTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        use OpStackBaseTableColumn::*;

        let variables: Vec<MPolynomial<XFieldElement>> = MPolynomial::variables(FULL_WIDTH);
        let clk = variables[usize::from(CLK)].clone();
        let osp = variables[usize::from(OSP)].clone();
        let osv = variables[usize::from(OSV)].clone();
        let sixteen = MPolynomial::from_constant(16.into(), FULL_WIDTH);

        let clk_is_0 = clk;
        let osv_is_0 = osv;
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
        use OpStackBaseTableColumn::*;

        let variables: Vec<MPolynomial<XFieldElement>> = MPolynomial::variables(2 * FULL_WIDTH);
        let ib1_shrink_stack = variables[usize::from(IB1ShrinkStack)].clone();
        let osp = variables[usize::from(OSP)].clone();
        let osv = variables[usize::from(OSV)].clone();
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
    ) -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }
}

impl OpStackTable {
    pub fn new_prover(matrix: Vec<Vec<BFieldElement>>) -> Self {
        let inherited_table =
            Table::new(BASE_WIDTH, FULL_WIDTH, matrix, "OpStackTable".to_string());
        Self { inherited_table }
    }

    pub fn codeword_table(
        &self,
        fri_domain: &FriDomain<BFieldElement>,
        omicron: BFieldElement,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Self {
        let base_columns = 0..self.base_width();
        let codewords = self.low_degree_extension(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
            base_columns,
        );
        let inherited_table = self.inherited_table.with_data(codewords);
        Self { inherited_table }
    }

    pub fn extend(
        &self,
        challenges: &OpStackTableChallenges,
        interpolant_degree: Degree,
    ) -> ExtOpStackTable {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        let mut running_product = PermArg::default_initial();
        let mut all_clock_jump_differences_running_product = PermArg::default_initial();

        let mut previous_row: Option<Vec<BFieldElement>> = None;
        for row in self.data().iter() {
            let mut extension_row = [0.into(); FULL_WIDTH];
            extension_row[..BASE_WIDTH]
                .copy_from_slice(&row.iter().map(|elem| elem.lift()).collect_vec());

            let clk = extension_row[usize::from(CLK)];
            let ib1 = extension_row[usize::from(IB1ShrinkStack)];
            let osp = extension_row[usize::from(OSP)];
            let osv = extension_row[usize::from(OSV)];

            let clk_w = challenges.clk_weight;
            let ib1_w = challenges.ib1_weight;
            let osp_w = challenges.osp_weight;
            let osv_w = challenges.osv_weight;

            // compress multiple values within one row so they become one value
            let compressed_row_for_permutation_argument =
                clk * clk_w + ib1 * ib1_w + osp * osp_w + osv * osv_w;

            // compute the running *product* of the compressed column (for permutation argument)
            running_product *=
                challenges.processor_perm_row_weight - compressed_row_for_permutation_argument;
            extension_row[usize::from(RunningProductPermArg)] = running_product;

            // clock jump difference
            if let Some(prow) = previous_row {
                if prow[usize::from(OSP)] == row[usize::from(OSP)] {
                    // if row[usize::from(IsPadding)] { // todo: we can't recognize padding rows atm
                    let clock_jump_difference =
                        (row[usize::from(CLK)] - prow[usize::from(CLK)]).lift();
                    all_clock_jump_differences_running_product *=
                        challenges.all_clock_jump_differences_weight - clock_jump_difference;
                }
            }
            extension_row[usize::from(AllClockJumpDifferencesPermArg)] =
                all_clock_jump_differences_running_product;

            previous_row = Some(row.clone());
            extension_matrix.push(extension_row.to_vec());
        }

        assert_eq!(self.data().len(), extension_matrix.len());
        let padded_height = extension_matrix.len();
        let inherited_table = self.extension(
            extension_matrix,
            interpolant_degree,
            padded_height,
            ExtOpStackTable::ext_initial_constraints(challenges),
            ExtOpStackTable::ext_consistency_constraints(challenges),
            ExtOpStackTable::ext_transition_constraints(challenges),
            ExtOpStackTable::ext_terminal_constraints(challenges),
        );
        ExtOpStackTable { inherited_table }
    }

    pub fn for_verifier(
        interpolant_degree: Degree,
        padded_height: usize,
        all_challenges: &AllChallenges,
    ) -> ExtOpStackTable {
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            vec![],
            "ExtOpStackTable".to_string(),
        );
        let base_table = Self { inherited_table };
        let empty_matrix: Vec<Vec<XFieldElement>> = vec![];
        let extension_table = base_table.extension(
            empty_matrix,
            interpolant_degree,
            padded_height,
            ExtOpStackTable::ext_initial_constraints(&all_challenges.op_stack_table_challenges),
            ExtOpStackTable::ext_consistency_constraints(&all_challenges.op_stack_table_challenges),
            ExtOpStackTable::ext_transition_constraints(&all_challenges.op_stack_table_challenges),
            ExtOpStackTable::ext_terminal_constraints(&all_challenges.op_stack_table_challenges),
        );

        ExtOpStackTable {
            inherited_table: extension_table,
        }
    }
}

impl ExtOpStackTable {
    pub fn ext_codeword_table(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        omicron: XFieldElement,
        padded_height: usize,
        num_trace_randomizers: usize,
        base_codewords: &[Vec<BFieldElement>],
    ) -> Self {
        let ext_columns = self.base_width()..self.full_width();
        let ext_codewords = self.low_degree_extension(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
            ext_columns,
        );

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
    pub ib1_weight: XFieldElement,
    pub osv_weight: XFieldElement,
    pub osp_weight: XFieldElement,

    /// Weight for accumulating all clock jump differences
    pub all_clock_jump_differences_weight: XFieldElement,
}

impl ExtensionTable for ExtOpStackTable {
    fn dynamic_initial_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtOpStackTable::ext_initial_constraints(&challenges.op_stack_table_challenges)
    }

    fn dynamic_consistency_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtOpStackTable::ext_consistency_constraints(&challenges.op_stack_table_challenges)
    }

    fn dynamic_transition_constraints(
        &self,
        challenges: &super::challenges::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtOpStackTable::ext_transition_constraints(&challenges.op_stack_table_challenges)
    }

    fn dynamic_terminal_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtOpStackTable::ext_terminal_constraints(&challenges.op_stack_table_challenges)
    }
}
