use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::Range;

use air::challenge_id::ChallengeId;
use air::cross_table_argument::CrossTableArg;
use air::cross_table_argument::LookupArg;
use air::cross_table_argument::PermArg;
use air::table::jump_stack::JumpStackTable;
use air::table_column::MasterAuxColumn;
use air::table_column::MasterMainColumn;
use air::table_column::ProcessorMainColumn;
use itertools::Itertools;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use strum::EnumCount;
use strum::IntoEnumIterator;
use twenty_first::math::traits::FiniteField;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::challenges::Challenges;
use crate::ndarray_helper::ROW_AXIS;
use crate::ndarray_helper::contiguous_column_slices;
use crate::ndarray_helper::horizontal_multi_slice_mut;
use crate::profiler::profiler;
use crate::table::TraceTable;

type MainColumn = <JumpStackTable as air::AIR>::MainColumn;
type AuxColumn = <JumpStackTable as air::AIR>::AuxColumn;

fn auxiliary_column_running_product_permutation_argument(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    let mut running_product = PermArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    for row in main_table.rows() {
        let compressed_row = row[MainColumn::CLK.main_index()]
            * challenges[ChallengeId::JumpStackClkWeight]
            + row[MainColumn::CI.main_index()] * challenges[ChallengeId::JumpStackCiWeight]
            + row[MainColumn::JSP.main_index()] * challenges[ChallengeId::JumpStackJspWeight]
            + row[MainColumn::JSO.main_index()] * challenges[ChallengeId::JumpStackJsoWeight]
            + row[MainColumn::JSD.main_index()] * challenges[ChallengeId::JumpStackJsdWeight];
        running_product *= challenges[ChallengeId::JumpStackIndeterminate] - compressed_row;
        auxiliary_column.push(running_product);
    }
    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

fn auxiliary_column_clock_jump_diff_lookup_log_derivative(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    // - use memoization to avoid recomputing inverses
    // - precompute common values through batch inversion
    const PRECOMPUTE_INVERSES_OF: Range<u64> = 0..100;
    let indeterminate = challenges[ChallengeId::ClockJumpDifferenceLookupIndeterminate];
    let to_invert = PRECOMPUTE_INVERSES_OF
        .map(|i| indeterminate - bfe!(i))
        .collect();
    let mut inverses_dictionary = PRECOMPUTE_INVERSES_OF
        .zip_eq(XFieldElement::batch_inversion(to_invert))
        .map(|(i, inv)| (bfe!(i), inv))
        .collect::<HashMap<_, _>>();

    // populate auxiliary column using memoization
    let mut cjd_lookup_log_derivative = LookupArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    auxiliary_column.push(cjd_lookup_log_derivative);
    for (previous_row, current_row) in main_table.rows().into_iter().tuple_windows() {
        if previous_row[MainColumn::JSP.main_index()] == current_row[MainColumn::JSP.main_index()] {
            let previous_clock = previous_row[MainColumn::CLK.main_index()];
            let current_clock = current_row[MainColumn::CLK.main_index()];
            let clock_jump_difference = current_clock - previous_clock;
            let &mut inverse = inverses_dictionary
                .entry(clock_jump_difference)
                .or_insert_with(|| (indeterminate - clock_jump_difference).inverse());
            cjd_lookup_log_derivative += inverse;
        }
        auxiliary_column.push(cjd_lookup_log_derivative);
    }
    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

impl TraceTable for JumpStackTable {
    type FillParam = ();
    type FillReturnInfo = Vec<BFieldElement>;

    fn fill(
        mut jump_stack_table: ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
        _: Self::FillParam,
    ) -> Self::FillReturnInfo {
        // Store the registers relevant for the Jump Stack Table,
        // i.e., CLK, CI, JSP, JSO, JSD, with JSP as the key. Preserves, thus
        // allows reusing, the order of the processor's rows, which are sorted
        // by CLK.
        let mut pre_processed_jump_stack_table: Vec<Vec<_>> = vec![];
        for processor_row in aet.processor_trace.rows() {
            let clk = processor_row[ProcessorMainColumn::CLK.main_index()];
            let ci = processor_row[ProcessorMainColumn::CI.main_index()];
            let jsp = processor_row[ProcessorMainColumn::JSP.main_index()];
            let jso = processor_row[ProcessorMainColumn::JSO.main_index()];
            let jsd = processor_row[ProcessorMainColumn::JSD.main_index()];

            // The (honest) prover can only grow the Jump Stack's size by at
            // most 1 per execution step. Hence, the following (a) works, and
            // (b) sorts.
            let jsp_val = jsp.value() as usize;
            let jump_stack_row = (clk, ci, jso, jsd);
            match jsp_val.cmp(&pre_processed_jump_stack_table.len()) {
                Ordering::Less => pre_processed_jump_stack_table[jsp_val].push(jump_stack_row),
                Ordering::Equal => pre_processed_jump_stack_table.push(vec![jump_stack_row]),
                Ordering::Greater => panic!("JSP must increase by at most 1 per execution step."),
            }
        }

        // move rows into the Jump Stack Table, sorted by JSP first, CLK second
        let mut jump_stack_table_row = 0;
        for (jsp_val, rows_with_this_jsp) in pre_processed_jump_stack_table.into_iter().enumerate()
        {
            let jsp = bfe!(jsp_val as u64);
            for (clk, ci, jso, jsd) in rows_with_this_jsp {
                jump_stack_table[(jump_stack_table_row, MainColumn::CLK.main_index())] = clk;
                jump_stack_table[(jump_stack_table_row, MainColumn::CI.main_index())] = ci;
                jump_stack_table[(jump_stack_table_row, MainColumn::JSP.main_index())] = jsp;
                jump_stack_table[(jump_stack_table_row, MainColumn::JSO.main_index())] = jso;
                jump_stack_table[(jump_stack_table_row, MainColumn::JSD.main_index())] = jsd;
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
            let clk_diff =
                next_row[MainColumn::CLK.main_index()] - curr_row[MainColumn::CLK.main_index()];
            if curr_row[MainColumn::JSP.main_index()] == next_row[MainColumn::JSP.main_index()] {
                clock_jump_differences.push(clk_diff);
            }
        }
        clock_jump_differences
    }

    fn pad(mut jump_stack_table: ArrayViewMut2<BFieldElement>, table_len: usize) {
        assert!(table_len > 0, "Processor Table must have at least 1 row.");

        // Set up indices for relevant sections of the table.
        let padded_height = jump_stack_table.nrows();
        let num_padding_rows = padded_height - table_len;
        let max_clk_before_padding = table_len - 1;
        let max_clk_before_padding_row_idx = jump_stack_table
            .rows()
            .into_iter()
            .enumerate()
            .find(|(_, row)| {
                row[MainColumn::CLK.main_index()].value() as usize == max_clk_before_padding
            })
            .map(|(idx, _)| idx)
            .expect("Jump Stack Table must contain row with clock cycle equal to max cycle.");
        let rows_to_move_source_section_start = max_clk_before_padding_row_idx + 1;
        let rows_to_move_source_section_end = table_len;
        let num_rows_to_move = rows_to_move_source_section_end - rows_to_move_source_section_start;
        let rows_to_move_dest_section_start = rows_to_move_source_section_start + num_padding_rows;
        let rows_to_move_dest_section_end = rows_to_move_dest_section_start + num_rows_to_move;
        let padding_section_start = rows_to_move_source_section_start;
        let padding_section_end = padding_section_start + num_padding_rows;
        assert_eq!(padded_height, rows_to_move_dest_section_end);

        // Move all rows below the row with the highest CLK to the end of the
        // table – if they exist.
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

        // Fill the created gap with padding rows, i.e., with copies of the last
        // row before the gap. This is the padding section.
        let padding_row_template = jump_stack_table
            .row(max_clk_before_padding_row_idx)
            .to_owned();
        let mut padding_section =
            jump_stack_table.slice_mut(s![padding_section_start..padding_section_end, ..]);
        padding_section
            .axis_iter_mut(ROW_AXIS)
            .into_par_iter()
            .for_each(|padding_row| padding_row_template.clone().move_into(padding_row));

        // CLK keeps increasing by 1 also in the padding section.
        let new_clk_values =
            Array1::from_iter((table_len..padded_height).map(|clk| bfe!(clk as u64)));
        new_clk_values.move_into(padding_section.slice_mut(s![.., MainColumn::CLK.main_index()]));
    }

    fn extend(
        main_table: ArrayView2<BFieldElement>,
        mut aux_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        profiler!(start "jump stack table");
        assert_eq!(MainColumn::COUNT, main_table.ncols());
        assert_eq!(AuxColumn::COUNT, aux_table.ncols());
        assert_eq!(main_table.nrows(), aux_table.nrows());

        // use strum::IntoEnumIterator;
        let auxiliary_column_indices = AuxColumn::iter()
            .map(|column| column.aux_index())
            .collect_vec();
        let auxiliary_column_slices = horizontal_multi_slice_mut(
            aux_table.view_mut(),
            &contiguous_column_slices(&auxiliary_column_indices),
        );
        let extension_functions = [
            auxiliary_column_running_product_permutation_argument,
            auxiliary_column_clock_jump_diff_lookup_log_derivative,
        ];

        extension_functions
            .into_par_iter()
            .zip_eq(auxiliary_column_slices)
            .for_each(|(generator, slice)| {
                generator(main_table, challenges).move_into(slice);
            });

        profiler!(stop "jump stack table");
    }
}
