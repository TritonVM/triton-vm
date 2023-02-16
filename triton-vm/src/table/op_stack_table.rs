use std::cmp::Ordering;

use ndarray::parallel::prelude::*;
use ndarray::s;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use ndarray::Axis;
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::op_stack::OP_STACK_REG_COUNT;
use crate::table::challenges::ChallengeId::*;
use crate::table::challenges::Challenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::DualRowIndicator::*;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::*;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::LookupArg;
use crate::table::cross_table_argument::PermArg;
use crate::table::master_table::NUM_BASE_COLUMNS;
use crate::table::master_table::NUM_EXT_COLUMNS;
use crate::table::table_column::MasterBaseTableColumn;
use crate::table::table_column::MasterExtTableColumn;
use crate::table::table_column::OpStackBaseTableColumn;
use crate::table::table_column::OpStackBaseTableColumn::*;
use crate::table::table_column::OpStackExtTableColumn;
use crate::table::table_column::OpStackExtTableColumn::*;
use crate::table::table_column::ProcessorBaseTableColumn;
use crate::vm::AlgebraicExecutionTrace;

pub const BASE_WIDTH: usize = OpStackBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = OpStackExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

#[derive(Debug, Clone)]
pub struct OpStackTable {}

#[derive(Debug, Clone)]
pub struct ExtOpStackTable {}

impl ExtOpStackTable {
    pub fn ext_initial_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>>> {
        let circuit_builder = ConstraintCircuitBuilder::new();

        let clk = circuit_builder.input(BaseRow(CLK.master_base_table_index()));
        let ib1 = circuit_builder.input(BaseRow(IB1ShrinkStack.master_base_table_index()));
        let osp = circuit_builder.input(BaseRow(OSP.master_base_table_index()));
        let osv = circuit_builder.input(BaseRow(OSV.master_base_table_index()));
        let rppa = circuit_builder.input(ExtRow(RunningProductPermArg.master_ext_table_index()));
        let clock_jump_diff_log_derivative = circuit_builder.input(ExtRow(
            ClockJumpDifferenceLookupClientLogDerivative.master_ext_table_index(),
        ));

        let clk_is_0 = clk;
        let osv_is_0 = osv;
        let osp_is_16 = osp - circuit_builder.b_constant(16_u32.into());

        // The running product for the permutation argument `rppa` starts off having accumulated the
        // first row. Note that `clk` and `osv` are constrained to be 0, and `osp` to be 16.
        let compressed_row = circuit_builder.challenge(OpStackIb1Weight) * ib1
            + circuit_builder.challenge(OpStackOspWeight)
                * circuit_builder.b_constant(16_u32.into());
        let processor_perm_indeterminate = circuit_builder.challenge(OpStackIndeterminate);
        let rppa_initial = processor_perm_indeterminate - compressed_row;
        let rppa_starts_correctly = rppa - rppa_initial;

        let clock_jump_diff_log_derivative_is_initialized_correctly = clock_jump_diff_log_derivative
            - circuit_builder.x_constant(LookupArg::default_initial());

        [
            clk_is_0,
            osv_is_0,
            osp_is_16,
            rppa_starts_correctly,
            clock_jump_diff_log_derivative_is_initialized_correctly,
        ]
        .map(|circuit| circuit.consume())
        .to_vec()
    }

    pub fn ext_consistency_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>>> {
        // no further constraints
        vec![]
    }

    pub fn ext_transition_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>>> {
        let circuit_builder =
            ConstraintCircuitBuilder::<DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>>::new();
        let one = circuit_builder.b_constant(1u32.into());

        let clk = circuit_builder.input(CurrentBaseRow(CLK.master_base_table_index()));
        let ib1_shrink_stack =
            circuit_builder.input(CurrentBaseRow(IB1ShrinkStack.master_base_table_index()));
        let osp = circuit_builder.input(CurrentBaseRow(OSP.master_base_table_index()));
        let osv = circuit_builder.input(CurrentBaseRow(OSV.master_base_table_index()));
        let rppa = circuit_builder.input(CurrentExtRow(
            RunningProductPermArg.master_ext_table_index(),
        ));
        let clock_jump_diff_log_derivative = circuit_builder.input(CurrentExtRow(
            ClockJumpDifferenceLookupClientLogDerivative.master_ext_table_index(),
        ));

        let clk_next = circuit_builder.input(NextBaseRow(CLK.master_base_table_index()));
        let ib1_shrink_stack_next =
            circuit_builder.input(NextBaseRow(IB1ShrinkStack.master_base_table_index()));
        let osp_next = circuit_builder.input(NextBaseRow(OSP.master_base_table_index()));
        let osv_next = circuit_builder.input(NextBaseRow(OSV.master_base_table_index()));
        let rppa_next =
            circuit_builder.input(NextExtRow(RunningProductPermArg.master_ext_table_index()));
        let clock_jump_diff_log_derivative_next = circuit_builder.input(NextExtRow(
            ClockJumpDifferenceLookupClientLogDerivative.master_ext_table_index(),
        ));

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

        // The running product for the permutation argument `rppa` is updated correctly.
        let alpha = circuit_builder.challenge(OpStackIndeterminate);
        let compressed_row = circuit_builder.challenge(OpStackClkWeight) * clk_next.clone()
            + circuit_builder.challenge(OpStackIb1Weight) * ib1_shrink_stack_next
            + circuit_builder.challenge(OpStackOspWeight) * osp_next.clone()
            + circuit_builder.challenge(OpStackOsvWeight) * osv_next;

        let rppa_updates_correctly = rppa_next - rppa * (alpha - compressed_row);

        // The running sum of the logarithmic derivative for the clock jump difference Lookup
        // Argument accumulates a summand of `clk_diff` if and only if the `osp` does not change.
        // Expressed differently:
        // - the `osp` changes or the log derivative accumulates a summand, and
        // - the `osp` does not change or the log derivative does not change.
        let log_derivative_remains =
            clock_jump_diff_log_derivative_next.clone() - clock_jump_diff_log_derivative.clone();
        let clk_diff = clk_next - clk;
        let log_derivative_accumulates = (clock_jump_diff_log_derivative_next
            - clock_jump_diff_log_derivative)
            * (circuit_builder.challenge(ClockJumpDifferenceLookupIndeterminate) - clk_diff)
            - one.clone();
        let log_derivative_updates_correctly = (osp_next.clone() - osp.clone() - one)
            * log_derivative_accumulates
            + (osp_next - osp) * log_derivative_remains;

        [
            osp_increases_by_1_or_does_not_change,
            osp_increases_by_1_or_osv_does_not_change_or_shrink_stack,
            rppa_updates_correctly,
            log_derivative_updates_correctly,
        ]
        .map(|circuit| circuit.consume())
        .to_vec()
    }

    pub fn ext_terminal_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>>> {
        // no further constraints
        vec![]
    }
}

impl OpStackTable {
    /// Fills the trace table in-place and returns all clock jump differences.
    pub fn fill_trace(
        op_stack_table: &mut ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
    ) -> Vec<BFieldElement> {
        // Store the registers relevant for the Op Stack Table, i.e., CLK, IB1, OSP, and OSV,
        // with OSP as the key. Preserves, thus allows reusing, the order of the processor's
        // rows, which are sorted by CLK.
        let mut pre_processed_op_stack_table: Vec<Vec<_>> = vec![];
        for processor_row in aet.processor_trace.rows() {
            let clk = processor_row[ProcessorBaseTableColumn::CLK.base_table_index()];
            let ib1 = processor_row[ProcessorBaseTableColumn::IB1.base_table_index()];
            let osp = processor_row[ProcessorBaseTableColumn::OSP.base_table_index()];
            let osv = processor_row[ProcessorBaseTableColumn::OSV.base_table_index()];
            // The (honest) prover can only grow the Op Stack's size by at most 1 per execution
            // step. Hence, the following (a) works, and (b) sorts.
            let osp_minus_16 = osp.value() as usize - OP_STACK_REG_COUNT;
            let op_stack_row = (clk, ib1, osv);
            match osp_minus_16.cmp(&pre_processed_op_stack_table.len()) {
                Ordering::Less => pre_processed_op_stack_table[osp_minus_16].push(op_stack_row),
                Ordering::Equal => pre_processed_op_stack_table.push(vec![op_stack_row]),
                Ordering::Greater => panic!("OSP must increase by at most 1 per execution step."),
            }
        }

        // Move the rows into the Op Stack Table, sorted by OSP first, CLK second.
        let mut op_stack_table_row = 0;
        for (osp_minus_16, rows_with_this_osp) in
            pre_processed_op_stack_table.into_iter().enumerate()
        {
            let osp = BFieldElement::new((osp_minus_16 + OP_STACK_REG_COUNT) as u64);
            for (clk, ib1, osv) in rows_with_this_osp {
                op_stack_table[[op_stack_table_row, CLK.base_table_index()]] = clk;
                op_stack_table[[op_stack_table_row, IB1ShrinkStack.base_table_index()]] = ib1;
                op_stack_table[[op_stack_table_row, OSP.base_table_index()]] = osp;
                op_stack_table[[op_stack_table_row, OSV.base_table_index()]] = osv;
                op_stack_table_row += 1;
            }
        }
        assert_eq!(aet.processor_trace.nrows(), op_stack_table_row);

        // Collect all clock jump differences.
        // The Op Stack Table and the Processor Table have the same length.
        let mut clock_jump_differences = vec![];
        for row_idx in 0..aet.processor_trace.nrows() - 1 {
            let curr_row = op_stack_table.row(row_idx);
            let next_row = op_stack_table.row(row_idx + 1);
            let clk_diff = next_row[CLK.base_table_index()] - curr_row[CLK.base_table_index()];
            if curr_row[OSP.base_table_index()] == next_row[OSP.base_table_index()] {
                clock_jump_differences.push(clk_diff);
            }
        }
        clock_jump_differences
    }

    pub fn pad_trace(
        op_stack_table: &mut ArrayViewMut2<BFieldElement>,
        processor_table_len: usize,
    ) {
        assert!(
            processor_table_len > 0,
            "Processor Table must have at least 1 row."
        );

        // Set up indices for relevant sections of the table.
        let padded_height = op_stack_table.nrows();
        let num_padding_rows = padded_height - processor_table_len;
        let max_clk_before_padding = processor_table_len - 1;
        let max_clk_before_padding_row_idx = op_stack_table
            .rows()
            .into_iter()
            .enumerate()
            .find(|(_, row)| row[CLK.base_table_index()].value() as usize == max_clk_before_padding)
            .map(|(idx, _)| idx)
            .expect("Op Stack Table must contain row with clock cycle equal to max cycle.");
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
            let rows_to_move = op_stack_table
                .slice(s![rows_to_move_source_range, ..])
                .to_owned();
            rows_to_move.move_into(&mut op_stack_table.slice_mut(s![rows_to_move_dest_range, ..]));
        }

        // Fill the created gap with padding rows, i.e., with copies of the last row before the
        // gap. This is the padding section.
        let padding_row_template = op_stack_table
            .row(max_clk_before_padding_row_idx)
            .to_owned();
        let mut padding_section =
            op_stack_table.slice_mut(s![padding_section_start..padding_section_end, ..]);
        padding_section
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|padding_row| padding_row_template.clone().move_into(padding_row));

        // CLK keeps increasing by 1 also in the padding section.
        let new_clk_values = Array1::from_iter(
            (processor_table_len..padded_height).map(|clk| BFieldElement::new(clk as u64)),
        );
        new_clk_values.move_into(padding_section.slice_mut(s![.., CLK.base_table_index()]));
    }

    pub fn extend(
        base_table: ArrayView2<BFieldElement>,
        mut ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());

        let clk_weight = challenges.get_challenge(OpStackClkWeight);
        let ib1_weight = challenges.get_challenge(OpStackIb1Weight);
        let osp_weight = challenges.get_challenge(OpStackOspWeight);
        let osv_weight = challenges.get_challenge(OpStackOsvWeight);
        let perm_arg_indeterminate = challenges.get_challenge(OpStackIndeterminate);
        let clock_jump_difference_lookup_indeterminate =
            challenges.get_challenge(ClockJumpDifferenceLookupIndeterminate);

        let mut running_product = PermArg::default_initial();
        let mut clock_jump_diff_lookup_log_derivative = LookupArg::default_initial();
        let mut previous_row: Option<ArrayView1<BFieldElement>> = None;

        for row_idx in 0..base_table.nrows() {
            let current_row = base_table.row(row_idx);
            let clk = current_row[CLK.base_table_index()];
            let ib1 = current_row[IB1ShrinkStack.base_table_index()];
            let osp = current_row[OSP.base_table_index()];
            let osv = current_row[OSV.base_table_index()];

            let compressed_row_for_permutation_argument =
                clk * clk_weight + ib1 * ib1_weight + osp * osp_weight + osv * osv_weight;
            running_product *= perm_arg_indeterminate - compressed_row_for_permutation_argument;

            // clock jump difference
            if let Some(prev_row) = previous_row {
                if prev_row[OSP.base_table_index()] == current_row[OSP.base_table_index()] {
                    let clock_jump_difference =
                        current_row[CLK.base_table_index()] - prev_row[CLK.base_table_index()];
                    clock_jump_diff_lookup_log_derivative +=
                        (clock_jump_difference_lookup_indeterminate - clock_jump_difference)
                            .inverse();
                }
            }

            let mut extension_row = ext_table.row_mut(row_idx);
            extension_row[RunningProductPermArg.ext_table_index()] = running_product;
            extension_row[ClockJumpDifferenceLookupClientLogDerivative.ext_table_index()] =
                clock_jump_diff_lookup_log_derivative;
            previous_row = Some(current_row);
        }
    }
}
