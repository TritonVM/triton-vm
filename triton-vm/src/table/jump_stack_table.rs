use itertools::Itertools;
use num_traits::One;
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::{Degree, MPolynomial};
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::cross_table_arguments::{CrossTableArg, PermArg};
use crate::fri_domain::FriDomain;
use crate::instruction::Instruction;
use crate::table::base_table::Extendable;
use crate::table::extension_table::Evaluable;
use crate::table::table_column::JumpStackBaseTableColumn::{self, *};
use crate::table::table_column::JumpStackExtTableColumn::{self, *};

use super::base_table::{InheritsFromTable, Table, TableLike};
use super::challenges::AllChallenges;
use super::extension_table::{ExtensionTable, Quotientable, QuotientableExtensionTable};

pub const JUMP_STACK_TABLE_NUM_PERMUTATION_ARGUMENTS: usize = 1;
pub const JUMP_STACK_TABLE_NUM_EVALUATION_ARGUMENTS: usize = 0;

/// This is 5 because it combines: clk, ci, jsp, jso, jsd,
pub const JUMP_STACK_TABLE_NUM_EXTENSION_CHALLENGES: usize = 5;

pub const BASE_WIDTH: usize = JumpStackBaseTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + JumpStackExtTableColumn::COUNT;

#[derive(Debug, Clone)]
pub struct JumpStackTable {
    inherited_table: Table<BFieldElement>,
}

impl InheritsFromTable<BFieldElement> for JumpStackTable {
    fn inherited_table(&self) -> &Table<BFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<BFieldElement> {
        &mut self.inherited_table
    }
}

#[derive(Debug, Clone)]
pub struct ExtJumpStackTable {
    pub(crate) inherited_table: Table<XFieldElement>,
}

impl Default for ExtJumpStackTable {
    fn default() -> Self {
        Self {
            inherited_table: Table::new(
                BASE_WIDTH,
                FULL_WIDTH,
                vec![],
                "EmptyExtJumpStackTable".to_string(),
            ),
        }
    }
}

impl Evaluable for ExtJumpStackTable {}
impl Quotientable for ExtJumpStackTable {}
impl QuotientableExtensionTable for ExtJumpStackTable {}

impl InheritsFromTable<XFieldElement> for ExtJumpStackTable {
    fn inherited_table(&self) -> &Table<XFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<XFieldElement> {
        &mut self.inherited_table
    }
}

impl TableLike<BFieldElement> for JumpStackTable {}

impl Extendable for JumpStackTable {
    fn get_padding_rows(&self) -> (Option<usize>, Vec<Vec<BFieldElement>>) {
        panic!(
            "This function should not be called: the Jump Stack Table implements `.pad` directly."
        )
    }

    // todo deduplicate this function. Other copy in op_stack_table.rs
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

impl TableLike<XFieldElement> for ExtJumpStackTable {}

impl ExtJumpStackTable {
    fn ext_initial_constraints(
        challenges: &JumpStackTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let variables: Vec<MPolynomial<XFieldElement>> = MPolynomial::variables(FULL_WIDTH);

        // 1. Cycle count clk is 0.
        let clk = variables[usize::from(CLK)].clone();

        // 2. Jump Stack Pointer jsp is 0.
        let jsp = variables[usize::from(JSP)].clone();

        // 3. Jump Stack Origin jso is 0.
        let jso = variables[usize::from(JSO)].clone();

        // 4. Jump Stack Destination jsd is 0.
        let jsd = variables[usize::from(JSD)].clone();

        let ci = variables[usize::from(JumpStackBaseTableColumn::CI)].clone();
        let rppa = variables[usize::from(JumpStackExtTableColumn::RunningProductPermArg)].clone();
        let rpcjd =
            variables[usize::from(JumpStackExtTableColumn::AllClockJumpDifferencesPermArg)].clone();

        // rppa starts off having accumulated the first row
        let constant = |xfe| MPolynomial::from_constant(xfe, FULL_WIDTH);
        let alpha = constant(challenges.processor_perm_indeterminate);
        let compressed_row = constant(challenges.clk_weight) * clk.clone()
            + constant(challenges.ci_weight) * ci
            + constant(challenges.jsp_weight) * jsp.clone()
            + constant(challenges.jso_weight) * jsp.clone()
            + constant(challenges.jsd_weight) * jsd.clone();
        let rppa_starts_correctly = rppa - (alpha - compressed_row);

        // rpcjd starts off as 1
        let one = constant(XFieldElement::one());
        let rpcjd_starts_with_one = rpcjd - one;

        vec![
            clk,
            jsp,
            jso,
            jsd,
            rppa_starts_correctly,
            rpcjd_starts_with_one,
        ]
    }

    fn ext_consistency_constraints(
        _challenges: &JumpStackTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }

    fn ext_transition_constraints(
        challenges: &JumpStackTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let constant = |xfe| MPolynomial::from_constant(xfe, 2 * FULL_WIDTH);
        let one = constant(XFieldElement::one());
        let call_opcode = constant(Instruction::Call(Default::default()).opcode_b().lift());
        let return_opcode = constant(Instruction::Return.opcode_b().lift());

        let variables = MPolynomial::variables(2 * FULL_WIDTH);
        let clk = variables[usize::from(CLK)].clone();
        let ci = variables[usize::from(CI)].clone();
        let jsp = variables[usize::from(JSP)].clone();
        let jso = variables[usize::from(JSO)].clone();
        let jsd = variables[usize::from(JSD)].clone();
        let clk_di = variables[usize::from(InverseOfClkDiffMinusOne)].clone();
        let rppa = variables[usize::from(RunningProductPermArg)].clone();
        let rpcjd = variables[usize::from(AllClockJumpDifferencesPermArg)].clone();
        let clk_next = variables[FULL_WIDTH + usize::from(CLK)].clone();
        let ci_next = variables[FULL_WIDTH + usize::from(CI)].clone();
        let jsp_next = variables[FULL_WIDTH + usize::from(JSP)].clone();
        let jso_next = variables[FULL_WIDTH + usize::from(JSO)].clone();
        let jsd_next = variables[FULL_WIDTH + usize::from(JSD)].clone();
        let clk_di_next = variables[FULL_WIDTH + usize::from(InverseOfClkDiffMinusOne)].clone();
        let rppa_next = variables[FULL_WIDTH + usize::from(RunningProductPermArg)].clone();
        let rpcjd_next =
            variables[FULL_WIDTH + usize::from(AllClockJumpDifferencesPermArg)].clone();

        // 1. The jump stack pointer jsp increases by 1
        //      or the jump stack pointer jsp does not change
        let jsp_inc_or_stays =
            (jsp_next.clone() - (jsp.clone() + one.clone())) * (jsp_next.clone() - jsp.clone());

        // 2. The jump stack pointer jsp increases by 1
        //      or current instruction ci is return
        //      or the jump stack origin jso does not change
        let jsp_inc_by_one_or_ci_is_return =
            (jsp_next.clone() - (jsp.clone() + one.clone())) * (ci.clone() - return_opcode.clone());
        let jsp_inc_or_jso_stays_or_ci_is_ret =
            jsp_inc_by_one_or_ci_is_return.clone() * (jso_next.clone() - jso);

        // 3. The jump stack pointer jsp increases by 1
        //      or current instruction ci is return
        //      or the jump stack destination jsd does not change
        let jsp_inc_or_jsd_stays_or_ci_ret =
            jsp_inc_by_one_or_ci_is_return * (jsd_next.clone() - jsd);

        // 4. The jump stack pointer jsp increases by 1
        //      or the cycle count clk increases by 1
        //      or current instruction ci is call
        //      or current instruction ci is return
        let jsp_inc_or_clk_inc_or_ci_call_or_ci_ret = (jsp_next.clone()
            - (jsp.clone() + one.clone()))
            * (clk_next.clone() - (clk.clone() + one.clone()))
            * (ci.clone() - call_opcode)
            * (ci - return_opcode);

        // 5. If the memory pointer `jsp` does not change, then
        // `clk_di'` is the inverse-or-zero of the clock jump
        // difference minus one.
        let jsp_changes = jsp_next.clone() - jsp.clone() - one.clone();
        let clock_diff_minus_one = clk_next.clone() - clk.clone() - one.clone();
        let clkdi_is_inverse_of_clock_diff_minus_one = clk_di_next * clock_diff_minus_one.clone();
        let clkdi_is_zero_or_clkdi_is_inverse_of_clock_diff_minus_one_or_jsp_changes =
            clk_di.clone() * clkdi_is_inverse_of_clock_diff_minus_one.clone() * jsp_changes.clone();
        let clock_diff_minus_one_is_zero_or_clock_diff_minus_one_is_clkdi_inverse_or_jsp_changes =
            clock_diff_minus_one.clone() * clkdi_is_inverse_of_clock_diff_minus_one * jsp_changes;

        // 6. The running product for the permutation argument `rppa`
        //  accumulates one row in each row, relative to weights `a`,
        //  `b`, `c`, `d`, `e`, and indeterminate `α`.
        let compressed_row = constant(challenges.clk_weight) * clk_next.clone()
            + constant(challenges.ci_weight) * ci_next
            + constant(challenges.jsp_weight) * jsp_next.clone()
            + constant(challenges.jso_weight) * jso_next
            + constant(challenges.jsd_weight) * jsd_next;
        let rppa_updates_correctly =
            rppa_next - rppa * (constant(challenges.processor_perm_indeterminate) - compressed_row);

        // 7. The running product for clock jump differences `rpcjd`
        // accumulates a factor `(clk' - clk - 1)` (relative to
        // indeterminate `β`) if a) the clock jump difference is
        // greater than 1, and if b) the jump stack pointer does not
        // change; and remains the same otherwise.
        //
        //   (1 - (clk' - clk - 1) · clk_di) · (rpcjd' - rpcjd)
        // + (jsp' - jsp) · (rpcjd' - rpcjd)
        // + (clk' - clk - 1) · (jsp' - jsp - 1)
        //     · (rpcjd' - rpcjd · (β - clk' + clk))`
        let indeterminate =
            constant(challenges.all_clock_jump_differences_multi_perm_indeterminate);
        let rpcjd_remains = rpcjd_next.clone() - rpcjd.clone();
        let jsp_diff = jsp_next - jsp;
        let rpcjd_update = rpcjd_next - rpcjd * (indeterminate - clk_next.clone() + clk.clone());
        let rpcjd_remains_if_clk_increments_by_one =
            (one.clone() - clock_diff_minus_one * clk_di) * rpcjd_remains.clone();
        let rpcjd_remains_if_jsp_changes = jsp_diff.clone() * rpcjd_remains;
        let rpcjd_updates_if_jsp_remains_and_clk_jumps =
            (clk_next - clk - one.clone()) * (jsp_diff - one) * rpcjd_update;
        let rpcjd_updates_correctly = rpcjd_remains_if_clk_increments_by_one
            + rpcjd_remains_if_jsp_changes
            + rpcjd_updates_if_jsp_remains_and_clk_jumps;

        vec![
            jsp_inc_or_stays,
            jsp_inc_or_jso_stays_or_ci_is_ret,
            jsp_inc_or_jsd_stays_or_ci_ret,
            jsp_inc_or_clk_inc_or_ci_call_or_ci_ret,
            clkdi_is_zero_or_clkdi_is_inverse_of_clock_diff_minus_one_or_jsp_changes,
            clock_diff_minus_one_is_zero_or_clock_diff_minus_one_is_clkdi_inverse_or_jsp_changes,
            rppa_updates_correctly,
            rpcjd_updates_correctly,
        ]
    }

    fn ext_terminal_constraints(
        _challenges: &JumpStackTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        vec![]
    }
}

impl JumpStackTable {
    pub fn new_prover(matrix: Vec<Vec<BFieldElement>>) -> Self {
        let inherited_table =
            Table::new(BASE_WIDTH, FULL_WIDTH, matrix, "JumpStackTable".to_string());
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
        challenges: &JumpStackTableChallenges,
        interpolant_degree: Degree,
    ) -> ExtJumpStackTable {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        let mut running_product = PermArg::default_initial();
        let mut all_clock_jump_differences_running_product = PermArg::default_initial();

        let mut previous_row: Option<Vec<BFieldElement>> = None;
        for row in self.data().iter() {
            let mut extension_row = [0.into(); FULL_WIDTH];
            extension_row[..BASE_WIDTH]
                .copy_from_slice(&row.iter().map(|elem| elem.lift()).collect_vec());

            let (clk, ci, jsp, jso, jsd) = (
                extension_row[usize::from(CLK)],
                extension_row[usize::from(CI)],
                extension_row[usize::from(JSP)],
                extension_row[usize::from(JSO)],
                extension_row[usize::from(JSD)],
            );

            let (clk_w, ci_w, jsp_w, jso_w, jsd_w) = (
                challenges.clk_weight,
                challenges.ci_weight,
                challenges.jsp_weight,
                challenges.jso_weight,
                challenges.jsd_weight,
            );

            // compress multiple values within one row so they become one value
            let compressed_row_for_permutation_argument =
                clk * clk_w + ci * ci_w + jsp * jsp_w + jso * jso_w + jsd * jsd_w;

            // compute the running *product* of the compressed column (for permutation argument)
            running_product *=
                challenges.processor_perm_indeterminate - compressed_row_for_permutation_argument;
            extension_row[usize::from(RunningProductPermArg)] = running_product;

            // clock jump difference
            if let Some(prow) = previous_row {
                if prow[usize::from(JSP)] == row[usize::from(JSP)] {
                    let clock_jump_difference =
                        (row[usize::from(CLK)] - prow[usize::from(CLK)]).lift();
                    if clock_jump_difference != XFieldElement::one() {
                        all_clock_jump_differences_running_product *= challenges
                            .all_clock_jump_differences_multi_perm_indeterminate
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
            ExtJumpStackTable::ext_initial_constraints(challenges),
            ExtJumpStackTable::ext_consistency_constraints(challenges),
            ExtJumpStackTable::ext_transition_constraints(challenges),
            ExtJumpStackTable::ext_terminal_constraints(challenges),
        );
        ExtJumpStackTable { inherited_table }
    }

    pub fn for_verifier(
        interpolant_degree: Degree,
        padded_height: usize,
        all_challenges: &AllChallenges,
    ) -> ExtJumpStackTable {
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            vec![],
            "ExtJumpStackTable".to_string(),
        );
        let base_table = Self { inherited_table };
        let empty_matrix: Vec<Vec<XFieldElement>> = vec![];
        let extension_table = base_table.extension(
            empty_matrix,
            interpolant_degree,
            padded_height,
            ExtJumpStackTable::ext_initial_constraints(&all_challenges.jump_stack_table_challenges),
            ExtJumpStackTable::ext_consistency_constraints(
                &all_challenges.jump_stack_table_challenges,
            ),
            ExtJumpStackTable::ext_transition_constraints(
                &all_challenges.jump_stack_table_challenges,
            ),
            ExtJumpStackTable::ext_terminal_constraints(
                &all_challenges.jump_stack_table_challenges,
            ),
        );

        ExtJumpStackTable {
            inherited_table: extension_table,
        }
    }
}

impl ExtJumpStackTable {
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
        ExtJumpStackTable { inherited_table }
    }
}

#[derive(Debug, Clone)]
pub struct JumpStackTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the op-stack table.
    pub processor_perm_indeterminate: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub clk_weight: XFieldElement,
    pub ci_weight: XFieldElement,
    pub jsp_weight: XFieldElement,
    pub jso_weight: XFieldElement,
    pub jsd_weight: XFieldElement,

    /// Weight for accumulating all clock jump differences
    pub all_clock_jump_differences_multi_perm_indeterminate: XFieldElement,
}

impl ExtensionTable for ExtJumpStackTable {
    fn dynamic_initial_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtJumpStackTable::ext_initial_constraints(&challenges.jump_stack_table_challenges)
    }

    fn dynamic_consistency_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtJumpStackTable::ext_consistency_constraints(&challenges.jump_stack_table_challenges)
    }

    fn dynamic_transition_constraints(
        &self,
        challenges: &super::challenges::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtJumpStackTable::ext_transition_constraints(&challenges.jump_stack_table_challenges)
    }

    fn dynamic_terminal_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtJumpStackTable::ext_terminal_constraints(&challenges.jump_stack_table_challenges)
    }
}
