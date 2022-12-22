use std::cmp::Ordering;

use ndarray::parallel::prelude::*;
use ndarray::s;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use ndarray::Axis;
use num_traits::One;
use num_traits::Zero;
use strum::EnumCount;
use strum_macros::Display;
use strum_macros::EnumCount as EnumCountMacro;
use strum_macros::EnumIter;
use triton_opcodes::instruction::Instruction;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use std::fmt::Display;
use std::fmt::Formatter;
use JumpStackTableChallengeId::*;

use crate::table::challenges::TableChallenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::DualRowIndicator::*;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::*;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::PermArg;
use crate::table::master_table::NUM_BASE_COLUMNS;
use crate::table::master_table::NUM_EXT_COLUMNS;
use crate::table::table_column::BaseTableColumn;
use crate::table::table_column::ExtTableColumn;
use crate::table::table_column::JumpStackBaseTableColumn;
use crate::table::table_column::JumpStackBaseTableColumn::*;
use crate::table::table_column::JumpStackExtTableColumn;
use crate::table::table_column::JumpStackExtTableColumn::*;
use crate::table::table_column::MasterBaseTableColumn;
use crate::table::table_column::MasterExtTableColumn;
use crate::table::table_column::ProcessorBaseTableColumn;
use crate::vm::AlgebraicExecutionTrace;

pub const JUMP_STACK_TABLE_NUM_PERMUTATION_ARGUMENTS: usize = 1;
pub const JUMP_STACK_TABLE_NUM_EVALUATION_ARGUMENTS: usize = 0;
pub const JUMP_STACK_TABLE_NUM_EXTENSION_CHALLENGES: usize = JumpStackTableChallengeId::COUNT;

pub const BASE_WIDTH: usize = JumpStackBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = JumpStackExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

#[derive(Debug, Clone)]
pub struct JumpStackTable {}

#[derive(Debug, Clone)]
pub struct ExtJumpStackTable {}

impl ExtJumpStackTable {
    pub fn ext_initial_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            JumpStackTableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let one = circuit_builder.b_constant(1_u32.into());

        let clk = circuit_builder.input(BaseRow(CLK.master_base_table_index()));
        let jsp = circuit_builder.input(BaseRow(JSP.master_base_table_index()));
        let jso = circuit_builder.input(BaseRow(JSO.master_base_table_index()));
        let jsd = circuit_builder.input(BaseRow(JSD.master_base_table_index()));
        let ci = circuit_builder.input(BaseRow(CI.master_base_table_index()));
        let rppa = circuit_builder.input(ExtRow(RunningProductPermArg.master_ext_table_index()));
        let rpcjd = circuit_builder.input(ExtRow(
            AllClockJumpDifferencesPermArg.master_ext_table_index(),
        ));

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

    pub fn ext_consistency_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            JumpStackTableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        // no further constraints
        vec![]
    }

    pub fn ext_transition_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            JumpStackTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let one = circuit_builder.b_constant(1u32.into());
        let call_opcode =
            circuit_builder.b_constant(Instruction::Call(Default::default()).opcode_b());
        let return_opcode = circuit_builder.b_constant(Instruction::Return.opcode_b());

        let clk = circuit_builder.input(CurrentBaseRow(CLK.master_base_table_index()));
        let ci = circuit_builder.input(CurrentBaseRow(CI.master_base_table_index()));
        let jsp = circuit_builder.input(CurrentBaseRow(JSP.master_base_table_index()));
        let jso = circuit_builder.input(CurrentBaseRow(JSO.master_base_table_index()));
        let jsd = circuit_builder.input(CurrentBaseRow(JSD.master_base_table_index()));
        let clk_di = circuit_builder.input(CurrentBaseRow(
            InverseOfClkDiffMinusOne.master_base_table_index(),
        ));
        let rppa = circuit_builder.input(CurrentExtRow(
            RunningProductPermArg.master_ext_table_index(),
        ));
        let rpcjd = circuit_builder.input(CurrentExtRow(
            AllClockJumpDifferencesPermArg.master_ext_table_index(),
        ));

        let clk_next = circuit_builder.input(NextBaseRow(CLK.master_base_table_index()));
        let ci_next = circuit_builder.input(NextBaseRow(CI.master_base_table_index()));
        let jsp_next = circuit_builder.input(NextBaseRow(JSP.master_base_table_index()));
        let jso_next = circuit_builder.input(NextBaseRow(JSO.master_base_table_index()));
        let jsd_next = circuit_builder.input(NextBaseRow(JSD.master_base_table_index()));
        let clk_di_next = circuit_builder.input(NextBaseRow(
            InverseOfClkDiffMinusOne.master_base_table_index(),
        ));
        let rppa_next =
            circuit_builder.input(NextExtRow(RunningProductPermArg.master_ext_table_index()));
        let rpcjd_next = circuit_builder.input(NextExtRow(
            AllClockJumpDifferencesPermArg.master_ext_table_index(),
        ));

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

    pub fn ext_terminal_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            JumpStackTableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        vec![]
    }
}

impl JumpStackTable {
    /// Fills the trace table in-place and returns all clock jump differences greater than 1.
    pub fn fill_trace(
        jump_stack_table: &mut ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
    ) -> Vec<BFieldElement> {
        // Store the registers relevant for the Jump Stack Table, i.e., CLK, CI, JSP, JSO, JSD,
        // with JSP as the key. Preserves, thus allows reusing, the order of the processor's
        // rows, which are sorted by CLK.
        let mut pre_processed_jump_stack_table: Vec<Vec<_>> = vec![];
        for processor_row in aet.processor_matrix.rows() {
            let clk = processor_row[ProcessorBaseTableColumn::CLK.base_table_index()];
            let ci = processor_row[ProcessorBaseTableColumn::CI.base_table_index()];
            let jsp = processor_row[ProcessorBaseTableColumn::JSP.base_table_index()];
            let jso = processor_row[ProcessorBaseTableColumn::JSO.base_table_index()];
            let jsd = processor_row[ProcessorBaseTableColumn::JSD.base_table_index()];
            // The (honest) prover can only grow the Jump Stack's size by at most 1 per execution
            // step. Hence, the following (a) works, and (b) sorts.
            let jsp_val = jsp.value() as usize;
            let jump_stack_row = (clk, ci, jso, jsd);
            match jsp_val.cmp(&pre_processed_jump_stack_table.len()) {
                Ordering::Less => pre_processed_jump_stack_table[jsp_val].push(jump_stack_row),
                Ordering::Equal => pre_processed_jump_stack_table.push(vec![jump_stack_row]),
                Ordering::Greater => panic!("JSP must increase by at most 1 per execution step."),
            }
        }

        // Move the rows into the Jump Stack Table, sorted by JSP first, CLK second.
        let mut jump_stack_table_row = 0;
        for (jsp_val, rows_with_this_jsp) in pre_processed_jump_stack_table.into_iter().enumerate()
        {
            let jsp = BFieldElement::new(jsp_val as u64);
            for (clk, ci, jso, jsd) in rows_with_this_jsp {
                jump_stack_table[(jump_stack_table_row, CLK.base_table_index())] = clk;
                jump_stack_table[(jump_stack_table_row, CI.base_table_index())] = ci;
                jump_stack_table[(jump_stack_table_row, JSP.base_table_index())] = jsp;
                jump_stack_table[(jump_stack_table_row, JSO.base_table_index())] = jso;
                jump_stack_table[(jump_stack_table_row, JSD.base_table_index())] = jsd;
                jump_stack_table_row += 1;
            }
        }
        assert_eq!(aet.processor_matrix.nrows(), jump_stack_table_row);

        // Set inverse of (clock difference - 1). Also, collect all clock jump differences
        // greater than 1.
        // The Jump Stack Table and the Processor Table have the same length.
        let mut clock_jump_differences_greater_than_1 = vec![];
        for row_idx in 0..aet.processor_matrix.nrows() - 1 {
            let (mut curr_row, next_row) =
                jump_stack_table.multi_slice_mut((s![row_idx, ..], s![row_idx + 1, ..]));
            let clk_diff = next_row[CLK.base_table_index()] - curr_row[CLK.base_table_index()];
            let clk_diff_minus_1 = clk_diff - BFieldElement::one();
            let clk_diff_minus_1_inverse = clk_diff_minus_1.inverse_or_zero();
            curr_row[InverseOfClkDiffMinusOne.base_table_index()] = clk_diff_minus_1_inverse;

            if curr_row[JSP.base_table_index()] == next_row[JSP.base_table_index()]
                && clk_diff.value() > 1
            {
                clock_jump_differences_greater_than_1.push(clk_diff);
            }
        }
        clock_jump_differences_greater_than_1
    }

    pub fn pad_trace(
        jump_stack_table: &mut ArrayViewMut2<BFieldElement>,
        processor_table_len: usize,
    ) {
        assert!(
            processor_table_len > 0,
            "Processor Table must have at least 1 row."
        );

        // Set up indices for relevant sections of the table.
        let padded_height = jump_stack_table.nrows();
        let num_padding_rows = padded_height - processor_table_len;
        let max_clk_before_padding = processor_table_len - 1;
        let max_clk_before_padding_row_idx = jump_stack_table
            .rows()
            .into_iter()
            .enumerate()
            .find(|(_, row)| row[CLK.base_table_index()].value() as usize == max_clk_before_padding)
            .map(|(idx, _)| idx)
            .expect("Jump Stack Table must contain row with clock cycle equal to max cycle.");
        let rows_to_move_source_section_start = max_clk_before_padding_row_idx + 1;
        let rows_to_move_source_section_end = processor_table_len;
        let num_rows_to_move = rows_to_move_source_section_end - rows_to_move_source_section_start;
        let rows_to_move_dest_section_start = rows_to_move_source_section_start + num_padding_rows;
        let rows_to_move_dest_section_end = rows_to_move_dest_section_start + num_rows_to_move;
        let padding_section_start = rows_to_move_source_section_start;
        let padding_section_end = padding_section_start + num_padding_rows;
        assert_eq!(padded_height, rows_to_move_dest_section_end);

        // Move all rows below the row with highest CLK to the end of the table – if they exist.
        if num_rows_to_move > 0 {
            let rows_to_move_source_range =
                rows_to_move_source_section_start..rows_to_move_source_section_end;
            let rows_to_move_dest_range =
                rows_to_move_dest_section_start..rows_to_move_dest_section_end;
            let rows_to_move = jump_stack_table
                .slice(s![rows_to_move_source_range, ..])
                .to_owned();
            rows_to_move
                .move_into(&mut jump_stack_table.slice_mut(s![rows_to_move_dest_range, ..]));
        }

        // Fill the created gap with padding rows, i.e., with (adjusted) copies of the last row
        // before the gap. This is the padding section.
        let mut padding_row_template = jump_stack_table
            .row(max_clk_before_padding_row_idx)
            .to_owned();
        padding_row_template[InverseOfClkDiffMinusOne.base_table_index()] = BFieldElement::zero();
        let mut padding_section =
            jump_stack_table.slice_mut(s![padding_section_start..padding_section_end, ..]);
        padding_section
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|padding_row| padding_row_template.clone().move_into(padding_row));

        // CLK keeps increasing by 1 also in the padding section.
        let new_clk_values = Array1::from_iter(
            (processor_table_len..padded_height).map(|clk| BFieldElement::new(clk as u64)),
        );
        new_clk_values.move_into(padding_section.slice_mut(s![.., CLK.base_table_index()]));

        // InverseOfClkDiffMinusOne must be consistent at the padding section's boundaries.
        jump_stack_table[[
            max_clk_before_padding_row_idx,
            InverseOfClkDiffMinusOne.base_table_index(),
        ]] = BFieldElement::zero();
        if num_rows_to_move > 0 && rows_to_move_dest_section_start > 0 {
            let max_clk_after_padding = padded_height - 1;
            let clk_diff_minus_one_at_padding_section_lower_boundary = jump_stack_table
                [[rows_to_move_dest_section_start, CLK.base_table_index()]]
                - BFieldElement::new(max_clk_after_padding as u64)
                - BFieldElement::one();
            let last_row_in_padding_section_idx = rows_to_move_dest_section_start - 1;
            jump_stack_table[[
                last_row_in_padding_section_idx,
                InverseOfClkDiffMinusOne.base_table_index(),
            ]] = clk_diff_minus_one_at_padding_section_lower_boundary.inverse_or_zero();
        }
    }

    pub fn extend(
        base_table: ArrayView2<BFieldElement>,
        mut ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &JumpStackTableChallenges,
    ) {
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());
        let mut running_product = PermArg::default_initial();
        let mut all_clock_jump_differences_running_product = PermArg::default_initial();
        let mut previous_row: Option<ArrayView1<BFieldElement>> = None;

        for row_idx in 0..base_table.nrows() {
            let current_row = base_table.row(row_idx);
            let clk = current_row[CLK.base_table_index()];
            let ci = current_row[CI.base_table_index()];
            let jsp = current_row[JSP.base_table_index()];
            let jso = current_row[JSO.base_table_index()];
            let jsd = current_row[JSD.base_table_index()];

            let compressed_row_for_permutation_argument = clk * challenges.clk_weight
                + ci * challenges.ci_weight
                + jsp * challenges.jsp_weight
                + jso * challenges.jso_weight
                + jsd * challenges.jsd_weight;
            running_product *=
                challenges.processor_perm_indeterminate - compressed_row_for_permutation_argument;

            // clock jump difference
            if let Some(prev_row) = previous_row {
                if prev_row[JSP.base_table_index()] == current_row[JSP.base_table_index()] {
                    let clock_jump_difference =
                        current_row[CLK.base_table_index()] - prev_row[CLK.base_table_index()];
                    if !clock_jump_difference.is_one() {
                        all_clock_jump_differences_running_product *= challenges
                            .all_clock_jump_differences_multi_perm_indeterminate
                            - clock_jump_difference;
                    }
                }
            }

            let mut extension_row = ext_table.row_mut(row_idx);
            extension_row[RunningProductPermArg.ext_table_index()] = running_product;
            extension_row[AllClockJumpDifferencesPermArg.ext_table_index()] =
                all_clock_jump_differences_running_product;
            previous_row = Some(current_row);
        }
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
    /// permutation/evaluation column of the jump-stack table.
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

pub struct JumpStackMatrixRow {
    pub row: [BFieldElement; BASE_WIDTH],
}

impl Display for JumpStackMatrixRow {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let width = 5;
        write!(
            f,
            "│ CLK: {:>width$} │ CI:  {:>width$} │ \
            JSP: {:>width$} │ JSO: {:>width$} │ JSD: {:>width$} │",
            self.row[CLK.base_table_index()].value(),
            self.row[CI.base_table_index()].value(),
            self.row[JSP.base_table_index()].value(),
            self.row[JSO.base_table_index()].value(),
            self.row[JSD.base_table_index()].value(),
        )
    }
}
