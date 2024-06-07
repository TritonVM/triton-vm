use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::Range;

use itertools::Itertools;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use strum::EnumCount;
use strum::IntoEnumIterator;
use twenty_first::math::traits::FiniteField;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::instruction::Instruction;
use crate::ndarray_helper::contiguous_column_slices;
use crate::ndarray_helper::horizontal_multi_slice_mut;
use crate::profiler::profiler;
use crate::table::challenges::ChallengeId::*;
use crate::table::challenges::Challenges;
use crate::table::constraint_circuit::DualRowIndicator::*;
use crate::table::constraint_circuit::SingleRowIndicator::*;
use crate::table::constraint_circuit::*;
use crate::table::cross_table_argument::*;
use crate::table::table_column::JumpStackBaseTableColumn::*;
use crate::table::table_column::JumpStackExtTableColumn::*;
use crate::table::table_column::*;

pub const BASE_WIDTH: usize = JumpStackBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = JumpStackExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct JumpStackTable;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ExtJumpStackTable;

impl ExtJumpStackTable {
    pub fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let clk = circuit_builder.input(BaseRow(CLK.master_base_table_index()));
        let jsp = circuit_builder.input(BaseRow(JSP.master_base_table_index()));
        let jso = circuit_builder.input(BaseRow(JSO.master_base_table_index()));
        let jsd = circuit_builder.input(BaseRow(JSD.master_base_table_index()));
        let ci = circuit_builder.input(BaseRow(CI.master_base_table_index()));
        let rppa = circuit_builder.input(ExtRow(RunningProductPermArg.master_ext_table_index()));
        let clock_jump_diff_log_derivative = circuit_builder.input(ExtRow(
            ClockJumpDifferenceLookupClientLogDerivative.master_ext_table_index(),
        ));

        let processor_perm_indeterminate = circuit_builder.challenge(JumpStackIndeterminate);
        // note: `clk`, `jsp`, `jso`, and `jsd` are all constrained to be 0 and can thus be omitted.
        let compressed_row = circuit_builder.challenge(JumpStackCiWeight) * ci;
        let rppa_starts_correctly = rppa - (processor_perm_indeterminate - compressed_row);

        // A clock jump difference of 0 is not allowed. Hence, the initial is recorded.
        let clock_jump_diff_log_derivative_starts_correctly = clock_jump_diff_log_derivative
            - circuit_builder.x_constant(LookupArg::default_initial());

        vec![
            clk,
            jsp,
            jso,
            jsd,
            rppa_starts_correctly,
            clock_jump_diff_log_derivative_starts_correctly,
        ]
    }

    pub fn consistency_constraints(
        _circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        // no further constraints
        vec![]
    }

    pub fn transition_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let one = || circuit_builder.b_constant(1);
        let call_opcode =
            circuit_builder.b_constant(Instruction::Call(BFieldElement::default()).opcode_b());
        let return_opcode = circuit_builder.b_constant(Instruction::Return.opcode_b());

        let clk = circuit_builder.input(CurrentBaseRow(CLK.master_base_table_index()));
        let ci = circuit_builder.input(CurrentBaseRow(CI.master_base_table_index()));
        let jsp = circuit_builder.input(CurrentBaseRow(JSP.master_base_table_index()));
        let jso = circuit_builder.input(CurrentBaseRow(JSO.master_base_table_index()));
        let jsd = circuit_builder.input(CurrentBaseRow(JSD.master_base_table_index()));
        let rppa = circuit_builder.input(CurrentExtRow(
            RunningProductPermArg.master_ext_table_index(),
        ));
        let clock_jump_diff_log_derivative = circuit_builder.input(CurrentExtRow(
            ClockJumpDifferenceLookupClientLogDerivative.master_ext_table_index(),
        ));

        let clk_next = circuit_builder.input(NextBaseRow(CLK.master_base_table_index()));
        let ci_next = circuit_builder.input(NextBaseRow(CI.master_base_table_index()));
        let jsp_next = circuit_builder.input(NextBaseRow(JSP.master_base_table_index()));
        let jso_next = circuit_builder.input(NextBaseRow(JSO.master_base_table_index()));
        let jsd_next = circuit_builder.input(NextBaseRow(JSD.master_base_table_index()));
        let rppa_next =
            circuit_builder.input(NextExtRow(RunningProductPermArg.master_ext_table_index()));
        let clock_jump_diff_log_derivative_next = circuit_builder.input(NextExtRow(
            ClockJumpDifferenceLookupClientLogDerivative.master_ext_table_index(),
        ));

        // 1. The jump stack pointer jsp increases by 1
        //      or the jump stack pointer jsp does not change
        let jsp_inc_or_stays =
            (jsp_next.clone() - jsp.clone() - one()) * (jsp_next.clone() - jsp.clone());

        // 2. The jump stack pointer jsp increases by 1
        //      or current instruction ci is return
        //      or the jump stack origin jso does not change
        let jsp_inc_by_one_or_ci_is_return =
            (jsp_next.clone() - jsp.clone() - one()) * (ci.clone() - return_opcode.clone());
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
        let jsp_inc_or_clk_inc_or_ci_call_or_ci_ret = (jsp_next.clone() - jsp.clone() - one())
            * (clk_next.clone() - clk.clone() - one())
            * (ci.clone() - call_opcode)
            * (ci - return_opcode);

        // The running product for the permutation argument `rppa` accumulates one row in each
        // row, relative to weights `a`, `b`, `c`, `d`, `e`, and indeterminate `α`.
        let compressed_row = circuit_builder.challenge(JumpStackClkWeight) * clk_next.clone()
            + circuit_builder.challenge(JumpStackCiWeight) * ci_next
            + circuit_builder.challenge(JumpStackJspWeight) * jsp_next.clone()
            + circuit_builder.challenge(JumpStackJsoWeight) * jso_next
            + circuit_builder.challenge(JumpStackJsdWeight) * jsd_next;

        let rppa_updates_correctly =
            rppa_next - rppa * (circuit_builder.challenge(JumpStackIndeterminate) - compressed_row);

        // The running sum of the logarithmic derivative for the clock jump difference Lookup
        // Argument accumulates a summand of `clk_diff` if and only if the `jsp` does not change.
        // Expressed differently:
        // - the `jsp` changes or the log derivative accumulates a summand, and
        // - the `jsp` does not change or the log derivative does not change.
        let log_derivative_remains =
            clock_jump_diff_log_derivative_next.clone() - clock_jump_diff_log_derivative.clone();
        let clk_diff = clk_next - clk;
        let log_derivative_accumulates = (clock_jump_diff_log_derivative_next
            - clock_jump_diff_log_derivative)
            * (circuit_builder.challenge(ClockJumpDifferenceLookupIndeterminate) - clk_diff)
            - one();
        let log_derivative_updates_correctly = (jsp_next.clone() - jsp.clone() - one())
            * log_derivative_accumulates
            + (jsp_next - jsp) * log_derivative_remains;

        vec![
            jsp_inc_or_stays,
            jsp_inc_or_jso_stays_or_ci_is_ret,
            jsp_inc_or_jsd_stays_or_ci_ret,
            jsp_inc_or_clk_inc_or_ci_call_or_ci_ret,
            rppa_updates_correctly,
            log_derivative_updates_correctly,
        ]
    }

    pub fn terminal_constraints(
        _circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        // no further constraints
        vec![]
    }
}

impl JumpStackTable {
    /// Fills the trace table in-place and returns all clock jump differences.
    pub fn fill_trace(
        jump_stack_table: &mut ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
    ) -> Vec<BFieldElement> {
        // Store the registers relevant for the Jump Stack Table, i.e., CLK, CI, JSP, JSO, JSD,
        // with JSP as the key. Preserves, thus allows reusing, the order of the processor's
        // rows, which are sorted by CLK.
        let mut pre_processed_jump_stack_table: Vec<Vec<_>> = vec![];
        for processor_row in aet.processor_trace.rows() {
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
            let jsp = bfe!(jsp_val as u64);
            for (clk, ci, jso, jsd) in rows_with_this_jsp {
                jump_stack_table[(jump_stack_table_row, CLK.base_table_index())] = clk;
                jump_stack_table[(jump_stack_table_row, CI.base_table_index())] = ci;
                jump_stack_table[(jump_stack_table_row, JSP.base_table_index())] = jsp;
                jump_stack_table[(jump_stack_table_row, JSO.base_table_index())] = jso;
                jump_stack_table[(jump_stack_table_row, JSD.base_table_index())] = jsd;
                jump_stack_table_row += 1;
            }
        }
        assert_eq!(aet.processor_trace.nrows(), jump_stack_table_row);

        // Collect all clock jump differences.
        // The Jump Stack Table and the Processor Table have the same length.
        let mut clock_jump_differences = vec![];
        for row_idx in 0..aet.processor_trace.nrows() - 1 {
            let curr_row = jump_stack_table.row(row_idx);
            let next_row = jump_stack_table.row(row_idx + 1);
            let clk_diff = next_row[CLK.base_table_index()] - curr_row[CLK.base_table_index()];
            if curr_row[JSP.base_table_index()] == next_row[JSP.base_table_index()] {
                clock_jump_differences.push(clk_diff);
            }
        }
        clock_jump_differences
    }

    pub fn pad_trace(
        mut jump_stack_table: ArrayViewMut2<BFieldElement>,
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

        // Fill the created gap with padding rows, i.e., with copies of the last row before the
        // gap. This is the padding section.
        let padding_row_template = jump_stack_table
            .row(max_clk_before_padding_row_idx)
            .to_owned();
        let mut padding_section =
            jump_stack_table.slice_mut(s![padding_section_start..padding_section_end, ..]);
        padding_section
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|padding_row| padding_row_template.clone().move_into(padding_row));

        // CLK keeps increasing by 1 also in the padding section.
        let new_clk_values =
            Array1::from_iter((processor_table_len..padded_height).map(|clk| bfe!(clk as u64)));
        new_clk_values.move_into(padding_section.slice_mut(s![.., CLK.base_table_index()]));
    }

    pub fn extend(
        base_table: ArrayView2<BFieldElement>,
        mut ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        profiler!(start "jump stack table");
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());

        // use strum::IntoEnumIterator;
        let extension_column_indices = JumpStackExtTableColumn::iter()
            .map(|column| column.ext_table_index())
            .collect_vec();
        let extension_column_slices = horizontal_multi_slice_mut(
            ext_table.view_mut(),
            &contiguous_column_slices(&extension_column_indices),
        );
        let extension_functions = [
            Self::extension_column_running_product_permutation_argument,
            Self::extension_column_clock_jump_diff_lookup_log_derivative,
        ];

        extension_functions
            .into_par_iter()
            .zip_eq(extension_column_slices)
            .for_each(|(generator, slice)| {
                generator(base_table, challenges).move_into(slice);
            });

        profiler!(stop "jump stack table");
    }

    fn extension_column_running_product_permutation_argument(
        base_table: ArrayView2<BFieldElement>,
        challenges: &Challenges,
    ) -> Array2<XFieldElement> {
        let mut running_product = PermArg::default_initial();
        let mut extension_column = Vec::with_capacity(base_table.nrows());
        for row in base_table.rows() {
            let compressed_row = row[CLK.base_table_index()] * challenges[JumpStackClkWeight]
                + row[CI.base_table_index()] * challenges[JumpStackCiWeight]
                + row[JSP.base_table_index()] * challenges[JumpStackJspWeight]
                + row[JSO.base_table_index()] * challenges[JumpStackJsoWeight]
                + row[JSD.base_table_index()] * challenges[JumpStackJsdWeight];
            running_product *= challenges[JumpStackIndeterminate] - compressed_row;
            extension_column.push(running_product);
        }
        Array2::from_shape_vec((base_table.nrows(), 1), extension_column).unwrap()
    }

    fn extension_column_clock_jump_diff_lookup_log_derivative(
        base_table: ArrayView2<BFieldElement>,
        challenges: &Challenges,
    ) -> Array2<XFieldElement> {
        // - use memoization to avoid recomputing inverses
        // - precompute common values through batch inversion
        const PRECOMPUTE_INVERSES_OF: Range<u64> = 0..100;
        let indeterminate = challenges[ClockJumpDifferenceLookupIndeterminate];
        let to_invert = PRECOMPUTE_INVERSES_OF
            .map(|i| indeterminate - bfe!(i))
            .collect();
        let mut inverses_dictionary = PRECOMPUTE_INVERSES_OF
            .zip_eq(XFieldElement::batch_inversion(to_invert))
            .map(|(i, inv)| (bfe!(i), inv))
            .collect::<HashMap<_, _>>();

        // populate extension column using memoization
        let mut cjd_lookup_log_derivative = LookupArg::default_initial();
        let mut extension_column = Vec::with_capacity(base_table.nrows());
        extension_column.push(cjd_lookup_log_derivative);
        for (previous_row, current_row) in base_table.rows().into_iter().tuple_windows() {
            if previous_row[JSP.base_table_index()] == current_row[JSP.base_table_index()] {
                let previous_clock = previous_row[CLK.base_table_index()];
                let current_clock = current_row[CLK.base_table_index()];
                let clock_jump_difference = current_clock - previous_clock;
                let &mut inverse = inverses_dictionary
                    .entry(clock_jump_difference)
                    .or_insert_with(|| (indeterminate - clock_jump_difference).inverse());
                cjd_lookup_log_derivative += inverse;
            }
            extension_column.push(cjd_lookup_log_derivative);
        }
        Array2::from_shape_vec((base_table.nrows(), 1), extension_column).unwrap()
    }
}
