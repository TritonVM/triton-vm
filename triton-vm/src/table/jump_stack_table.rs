use itertools::Itertools;
use num_traits::One;
use strum::EnumCount;
use strum_macros::{Display, EnumCount as EnumCountMacro, EnumIter};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use JumpStackTableChallengeId::*;

use crate::cross_table_arguments::{CrossTableArg, PermArg};
use crate::instruction::Instruction;
use crate::table::base_table::Extendable;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::Row;
use crate::table::table_column::JumpStackBaseTableColumn::{self, *};
use crate::table::table_column::JumpStackExtTableColumn::{self, *};

use super::base_table::{InheritsFromTable, Table, TableLike};
use super::challenges::TableChallenges;
use super::constraint_circuit::DualRowIndicator::*;
use super::constraint_circuit::{ConstraintCircuit, ConstraintCircuitBuilder, DualRowIndicator};
use super::extension_table::{ExtensionTable, QuotientableExtensionTable};

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

    fn pad(&mut self, padded_height: usize) {
        let max_clock = self.data().len() as u64 - 1;
        let num_padding_rows = padded_height - self.data().len();

        let template_index = self
            .data()
            .iter()
            .enumerate()
            .find(|(_, row)| row[usize::from(CLK)].value() == max_clock)
            .map(|(idx, _)| idx)
            .expect("Jump Stack Table must contain row with clock cycle equal to max cycle.");
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
                    (clk_diff - BFieldElement::one()).inverse_or_zero();
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
    pub fn ext_initial_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<JumpStackTableChallenges, SingleRowIndicator<FULL_WIDTH>>> {
        let circuit_builder = ConstraintCircuitBuilder::new(FULL_WIDTH);
        let one = circuit_builder.b_constant(1_u32.into());

        let clk = circuit_builder.input(Row(CLK.into()));
        let jsp = circuit_builder.input(Row(JSP.into()));
        let jso = circuit_builder.input(Row(JSO.into()));
        let jsd = circuit_builder.input(Row(JSD.into()));
        let ci = circuit_builder.input(Row(CI.into()));
        let rppa = circuit_builder.input(Row(RunningProductPermArg.into()));
        let rpcjd = circuit_builder.input(Row(AllClockJumpDifferencesPermArg.into()));

        let processor_perm_indeterminate = circuit_builder.challenge(ProcessorPermRowIndeterminate);
        // note: `clk`, `jsp`, `jso`, and `jsd` are all constrained to be 0 and can thus be omitted.
        let compressed_row = circuit_builder.challenge(CiWeight) * ci;
        let rppa_starts_correctly = rppa - (processor_perm_indeterminate - compressed_row);

        let rpcjd_starts_with_one = rpcjd - one;

        [
            clk,
            jsp,
            jso,
            jsd,
            rppa_starts_correctly,
            rpcjd_starts_with_one,
        ]
        .map(|circuit| circuit.consume())
        .to_vec()
    }

    pub fn ext_consistency_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<JumpStackTableChallenges, SingleRowIndicator<FULL_WIDTH>>> {
        // no further constraints
        vec![]
    }

    pub fn ext_transition_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<JumpStackTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
        let circuit_builder = ConstraintCircuitBuilder::new(2 * FULL_WIDTH);
        let one = circuit_builder.b_constant(1u32.into());
        let call_opcode =
            circuit_builder.b_constant(Instruction::Call(Default::default()).opcode_b());
        let return_opcode = circuit_builder.b_constant(Instruction::Return.opcode_b());

        let clk = circuit_builder.input(CurrentRow(CLK.into()));
        let ci = circuit_builder.input(CurrentRow(CI.into()));
        let jsp = circuit_builder.input(CurrentRow(JSP.into()));
        let jso = circuit_builder.input(CurrentRow(JSO.into()));
        let jsd = circuit_builder.input(CurrentRow(JSD.into()));
        let clk_di = circuit_builder.input(CurrentRow(InverseOfClkDiffMinusOne.into()));
        let rppa = circuit_builder.input(CurrentRow(RunningProductPermArg.into()));
        let rpcjd = circuit_builder.input(CurrentRow(AllClockJumpDifferencesPermArg.into()));

        let clk_next = circuit_builder.input(NextRow(CLK.into()));
        let ci_next = circuit_builder.input(NextRow(CI.into()));
        let jsp_next = circuit_builder.input(NextRow(JSP.into()));
        let jso_next = circuit_builder.input(NextRow(JSO.into()));
        let jsd_next = circuit_builder.input(NextRow(JSD.into()));
        let clk_di_next = circuit_builder.input(NextRow(InverseOfClkDiffMinusOne.into()));
        let rppa_next = circuit_builder.input(NextRow(RunningProductPermArg.into()));
        let rpcjd_next = circuit_builder.input(NextRow(AllClockJumpDifferencesPermArg.into()));

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
        let compressed_row = circuit_builder.challenge(ClkWeight) * clk_next.clone()
            + circuit_builder.challenge(CiWeight) * ci_next
            + circuit_builder.challenge(JspWeight) * jsp_next.clone()
            + circuit_builder.challenge(JsoWeight) * jso_next
            + circuit_builder.challenge(JsdWeight) * jsd_next;

        let rppa_updates_correctly = rppa_next
            - rppa * (circuit_builder.challenge(ProcessorPermRowIndeterminate) - compressed_row);

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
            circuit_builder.challenge(AllClockJumpDifferencesMultiPermIndeterminate);
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

        [
            jsp_inc_or_stays,
            jsp_inc_or_jso_stays_or_ci_is_ret,
            jsp_inc_or_jsd_stays_or_ci_ret,
            jsp_inc_or_clk_inc_or_ci_call_or_ci_ret,
            clkdi_is_zero_or_clkdi_is_inverse_of_clock_diff_minus_one_or_jsp_changes,
            clock_diff_minus_one_is_zero_or_clock_diff_minus_one_is_clkdi_inverse_or_jsp_changes,
            rppa_updates_correctly,
            rpcjd_updates_correctly,
        ]
        .map(|circuit| circuit.consume())
        .to_vec()
    }

    pub fn ext_terminal_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<JumpStackTableChallenges, SingleRowIndicator<FULL_WIDTH>>> {
        vec![]
    }
}

impl JumpStackTable {
    pub fn new(inherited_table: Table<BFieldElement>) -> Self {
        Self { inherited_table }
    }

    pub fn new_prover(matrix: Vec<Vec<BFieldElement>>) -> Self {
        let inherited_table =
            Table::new(BASE_WIDTH, FULL_WIDTH, matrix, "JumpStackTable".to_string());
        Self { inherited_table }
    }

    pub fn extend(&self, challenges: &JumpStackTableChallenges) -> ExtJumpStackTable {
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
        let inherited_table = self.new_from_lifted_matrix(extension_matrix);
        ExtJumpStackTable { inherited_table }
    }

    pub fn for_verifier() -> ExtJumpStackTable {
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            vec![],
            "ExtJumpStackTable".to_string(),
        );
        let base_table = Self { inherited_table };
        let empty_matrix: Vec<Vec<XFieldElement>> = vec![];
        let extension_table = base_table.new_from_lifted_matrix(empty_matrix);

        ExtJumpStackTable {
            inherited_table: extension_table,
        }
    }
}

impl ExtJumpStackTable {
    pub fn new(inherited_table: Table<XFieldElement>) -> Self {
        Self { inherited_table }
    }
}

#[derive(Debug, Copy, Clone, Display, EnumCountMacro, EnumIter, PartialEq, Eq, Hash)]
pub enum JumpStackTableChallengeId {
    ProcessorPermRowIndeterminate,
    ClkWeight,
    CiWeight,
    JspWeight,
    JsoWeight,
    JsdWeight,
    AllClockJumpDifferencesMultiPermIndeterminate,
}

impl From<JumpStackTableChallengeId> for usize {
    fn from(val: JumpStackTableChallengeId) -> Self {
        val as usize
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

impl TableChallenges for JumpStackTableChallenges {
    type Id = JumpStackTableChallengeId;

    #[inline]
    fn get_challenge(&self, id: Self::Id) -> XFieldElement {
        match id {
            ProcessorPermRowIndeterminate => self.processor_perm_indeterminate,
            ClkWeight => self.clk_weight,
            CiWeight => self.ci_weight,
            JspWeight => self.jsp_weight,
            JsoWeight => self.jso_weight,
            JsdWeight => self.jsd_weight,
            AllClockJumpDifferencesMultiPermIndeterminate => {
                self.all_clock_jump_differences_multi_perm_indeterminate
            }
        }
    }
}

impl ExtensionTable for ExtJumpStackTable {}
