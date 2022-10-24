use itertools::Itertools;
use num_traits::One;
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::{Degree, MPolynomial};
use twenty_first::shared_math::traits::Inverse;
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
    pub(crate) inherited_table: Table<XFieldElement>,
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
        panic!("This function should not be called: the Op Stack Table implements `.pad` directly.")
    }

    // todo deduplicate this function. Other copy in jump_stack_table.rs
    fn pad(&mut self, padded_height: usize) {
        let max_clock = self.data().len() as u64 - 1;
        let num_padding_rows = padded_height - self.data().len();

        let template_index = self
            .data()
            .iter()
            .enumerate()
            .find(|(_, row)| row[usize::from(CLK)].value() == max_clock)
            .map(|(idx, _)| idx)
            .unwrap();
        let insertion_index = template_index + 1;

        let padding_template = &mut self.mut_data()[template_index];
        padding_template[usize::from(InverseOfClkDiffMinusOne)] = 0_u64.into();

        let mut padding_rows = vec![];
        while padding_rows.len() < num_padding_rows {
            let mut padding_row = padding_template.clone();
            padding_row[usize::from(CLK)] += (padding_rows.len() as u32 + 1).into();
            padding_rows.push(padding_row)
        }

        if let Some(row) = padding_rows.last_mut() {
            if let Some(next_row) = self.data().get(insertion_index) {
                let clk_diff = next_row[usize::from(CLK)] - row[usize::from(CLK)];
                row[usize::from(InverseOfClkDiffMinusOne)] =
                    (clk_diff - 1_u64.into()).inverse_or_zero();
            }
        }

        let old_tail_length = self.data().len() - insertion_index;
        self.mut_data().append(&mut padding_rows);
        self.mut_data()[insertion_index..].rotate_left(old_tail_length);

        assert_eq!(padded_height, self.data().len());
    }
}

impl TableLike<XFieldElement> for ExtOpStackTable {}

impl ExtOpStackTable {
    fn ext_initial_constraints(
        challenges: &OpStackTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        use OpStackBaseTableColumn::*;

        let variables: Vec<MPolynomial<XFieldElement>> = MPolynomial::variables(FULL_WIDTH);
        let clk = variables[usize::from(CLK)].clone();
        let ib1 = variables[usize::from(IB1ShrinkStack)].clone();
        let osp = variables[usize::from(OSP)].clone();
        let osv = variables[usize::from(OSV)].clone();
        let sixteen = MPolynomial::from_constant(16.into(), FULL_WIDTH);
        let rppa = variables[usize::from(RunningProductPermArg)].clone();
        let rpcjd = variables[usize::from(AllClockJumpDifferencesPermArg)].clone();
        let one = MPolynomial::from_constant(XFieldElement::one(), FULL_WIDTH);

        let clk_is_0 = clk.clone();
        let osv_is_0 = osv.clone();
        let osp_is_16 = osp.clone() - sixteen;

        // The running product for the permutation argument `rppa`
        // starts off having accumulated the first row.
        let constant = |xfe| MPolynomial::from_constant(xfe, FULL_WIDTH);
        let compressed_row = constant(challenges.clk_weight) * clk
            + constant(challenges.ib1_weight) * ib1
            + constant(challenges.osp_weight) * osp
            + constant(challenges.osv_weight) * osv;
        let alpha = constant(challenges.processor_perm_row_eval_point);
        let rppa_initial = alpha - compressed_row;
        let rppa_starts_correctly = rppa - rppa_initial;

        // The running product for clock jump differences starts with
        // one
        let rpcjd_starts_correctly = rpcjd - one;

        vec![
            clk_is_0,
            osv_is_0,
            osp_is_16,
            rppa_starts_correctly,
            rpcjd_starts_correctly,
        ]
    }

    fn ext_consistency_constraints(
        _challenges: &OpStackTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }

    fn ext_transition_constraints(
        challenges: &OpStackTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        use OpStackBaseTableColumn::*;

        let variables: Vec<MPolynomial<XFieldElement>> = MPolynomial::variables(2 * FULL_WIDTH);

        let clk = variables[usize::from(CLK)].clone();
        let ib1_shrink_stack = variables[usize::from(IB1ShrinkStack)].clone();
        let osp = variables[usize::from(OSP)].clone();
        let osv = variables[usize::from(OSV)].clone();
        let clk_di = variables[usize::from(InverseOfClkDiffMinusOne)].clone();
        let rpcjd = variables[usize::from(AllClockJumpDifferencesPermArg)].clone();
        let rppa = variables[usize::from(RunningProductPermArg)].clone();

        let clk_next = variables[FULL_WIDTH + usize::from(CLK)].clone();
        let ib1_shrink_stack_next = variables[FULL_WIDTH + usize::from(IB1ShrinkStack)].clone();
        let osp_next = variables[FULL_WIDTH + usize::from(OSP)].clone();
        let osv_next = variables[FULL_WIDTH + usize::from(OSV)].clone();
        let rpcjd_next =
            variables[FULL_WIDTH + usize::from(AllClockJumpDifferencesPermArg)].clone();
        let rppa_next = variables[FULL_WIDTH + usize::from(RunningProductPermArg)].clone();
        let one = MPolynomial::from_constant(XFieldElement::one(), 2 * FULL_WIDTH);

        // the osp increases by 1 or the osp does not change
        //
        // $(osp' - (osp + 1))·(osp' - osp) = 0$
        let osp_increases_by_1_or_does_not_change =
            (osp_next.clone() - (osp.clone() + one.clone())) * (osp_next.clone() - osp.clone());

        // the osp increases by 1 or the osv does not change OR the ci shrinks the OpStack
        //
        // $ (osp' - (osp + 1)) · (osv' - osv) · (1 - ib1) = 0$
        let osp_increases_by_1_or_osv_does_not_change_or_shrink_stack = (osp_next.clone()
            - (osp.clone() + one.clone()))
            * (osv_next.clone() - osv)
            * (one.clone() - ib1_shrink_stack);

        // The clock jump difference inverse is consistent
        // with the clock cycles.
        //     clk_di' = (clk' - clk - 1)^-1 unless osp changes
        // <=> (osp' - osp - 1) * (1 - clk_di' * (clk' - clk - 1)) * clk_di' = 0
        //  /\ (osp' - osp - 1) * (1 - clk_di' * (clk' - clk - 1)) * (clk' - clk - 1) = 0
        let osp_changes = osp_next.clone() - osp.clone() - one.clone();
        let clk_diff_minus_one = clk_next.clone() - clk.clone() - one.clone();
        let clkdi_is_cdmo_inverse = clk_di.clone() * clk_diff_minus_one.clone() - one.clone();
        let clk_di_is_zero_or_cdmo_inverse_or_osp_changes =
            osp_changes.clone() * clkdi_is_cdmo_inverse.clone() * clk_di.clone();
        let cdmo_is_zero_or_clkdi_inverse_or_osp_changes =
            osp_changes * clkdi_is_cdmo_inverse * clk_diff_minus_one;

        // The running product for clock jump differences `rpcjd`
        // accumulates a factor (beta - clk' + clk) if
        //  - the op stack pointer `osp` remains the same; and
        //  - the clock jump difference is 2 or greater.
        //
        //   (clk' - clk - 1) * (1 - osp' + osp) * (cjdrp' - cjdrp * (beta - clk' + clk))
        // + (1 - (clk' - clk - 1) * clk_di) * (cjdrp' - cjdrp)
        // + (osp' - osp) * (cjdrp' - cjdrp)
        let constant = |xfe| MPolynomial::from_constant(xfe, 2 * FULL_WIDTH);
        let beta = constant(challenges.all_clock_jump_differences_eval_point);
        let cjdrp_updates_correctly = (clk_next.clone() - clk.clone() - one.clone())
            * (one.clone() - osp_next.clone() + osp.clone())
            * (rpcjd_next.clone() - rpcjd.clone() * (beta - clk_next.clone() + clk.clone()))
            + (one.clone() - (clk_next.clone() - clk - one) * clk_di)
                * (rpcjd_next.clone() - rpcjd.clone())
            + (osp_next.clone() - osp) * (rpcjd_next - rpcjd);

        // The running product for the permutation argument `rppa` is updated correctly.
        let alpha = constant(challenges.processor_perm_row_eval_point);
        let compressed_row = constant(challenges.clk_weight) * clk_next
            + constant(challenges.ib1_weight) * ib1_shrink_stack_next
            + constant(challenges.osp_weight) * osp_next
            + constant(challenges.osv_weight) * osv_next;
        let rppa_updates_correctly = rppa_next - rppa * (alpha - compressed_row);

        vec![
            osp_increases_by_1_or_does_not_change,
            osp_increases_by_1_or_osv_does_not_change_or_shrink_stack,
            clk_di_is_zero_or_cdmo_inverse_or_osp_changes,
            cdmo_is_zero_or_clkdi_inverse_or_osp_changes,
            cjdrp_updates_correctly,
            rppa_updates_correctly,
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
                challenges.processor_perm_row_eval_point - compressed_row_for_permutation_argument;
            extension_row[usize::from(RunningProductPermArg)] = running_product;

            // clock jump difference
            if let Some(prow) = previous_row {
                if prow[usize::from(OSP)] == row[usize::from(OSP)] {
                    let clock_jump_difference =
                        (row[usize::from(CLK)] - prow[usize::from(CLK)]).lift();
                    if clock_jump_difference != XFieldElement::one() {
                        all_clock_jump_differences_running_product *= challenges
                            .all_clock_jump_differences_eval_point
                            - clock_jump_difference;
                    }
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
    pub fn lde(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        omicron: XFieldElement,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Self {
        let ext_columns = self.base_width()..self.full_width();
        let ext_codewords = self.low_degree_extension(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
            ext_columns,
        );

        let inherited_table = self.inherited_table.with_data(ext_codewords);
        ExtOpStackTable { inherited_table }
    }
}

#[derive(Debug, Clone)]
pub struct OpStackTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the op-stack table.
    pub processor_perm_row_eval_point: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub clk_weight: XFieldElement,
    pub ib1_weight: XFieldElement,
    pub osv_weight: XFieldElement,
    pub osp_weight: XFieldElement,

    /// Weight for accumulating all clock jump differences
    pub all_clock_jump_differences_eval_point: XFieldElement,
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
