use std::cmp::Ordering;
use std::collections::HashMap;

use arbitrary::Arbitrary;
use itertools::Itertools;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use strum::EnumCount;
use twenty_first::math::traits::FiniteField;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::ndarray_helper::contiguous_column_slices;
use crate::ndarray_helper::horizontal_multi_slice_mut;
use crate::op_stack::OpStackElement;
use crate::op_stack::UnderflowIO;
use crate::profiler::profiler;
use crate::table::challenges::ChallengeId::*;
use crate::table::challenges::Challenges;
use crate::table::constraint_circuit::DualRowIndicator::*;
use crate::table::constraint_circuit::SingleRowIndicator::*;
use crate::table::constraint_circuit::*;
use crate::table::cross_table_argument::*;
use crate::table::master_table::TableId;
use crate::table::table_column::OpStackBaseTableColumn::*;
use crate::table::table_column::OpStackExtTableColumn::*;
use crate::table::table_column::*;

pub const BASE_WIDTH: usize = OpStackBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = OpStackExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

/// The value indicating a padding row in the op stack table. Stored in the `ib1_shrink_stack`
/// column.
pub(crate) const PADDING_VALUE: BFieldElement = BFieldElement::new(2);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct OpStackTable;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ExtOpStackTable;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub struct OpStackTableEntry {
    pub clk: u32,
    pub op_stack_pointer: BFieldElement,
    pub underflow_io: UnderflowIO,
}

impl OpStackTableEntry {
    pub fn new(clk: u32, op_stack_pointer: BFieldElement, underflow_io: UnderflowIO) -> Self {
        Self {
            clk,
            op_stack_pointer,
            underflow_io,
        }
    }

    pub fn shrinks_stack(&self) -> bool {
        self.underflow_io.shrinks_stack()
    }

    pub fn grows_stack(&self) -> bool {
        self.underflow_io.grows_stack()
    }

    pub fn from_underflow_io_sequence(
        clk: u32,
        op_stack_pointer_after_sequence_execution: BFieldElement,
        mut underflow_io_sequence: Vec<UnderflowIO>,
    ) -> Vec<Self> {
        UnderflowIO::canonicalize_sequence(&mut underflow_io_sequence);
        assert!(UnderflowIO::is_uniform_sequence(&underflow_io_sequence));

        let sequence_length: BFieldElement =
            u32::try_from(underflow_io_sequence.len()).unwrap().into();
        let mut op_stack_pointer = match UnderflowIO::is_writing_sequence(&underflow_io_sequence) {
            true => op_stack_pointer_after_sequence_execution - sequence_length,
            false => op_stack_pointer_after_sequence_execution + sequence_length,
        };
        let mut op_stack_table_entries = vec![];
        for underflow_io in underflow_io_sequence {
            if underflow_io.shrinks_stack() {
                op_stack_pointer.decrement();
            }
            let op_stack_table_entry = Self::new(clk, op_stack_pointer, underflow_io);
            op_stack_table_entries.push(op_stack_table_entry);
            if underflow_io.grows_stack() {
                op_stack_pointer.increment();
            }
        }
        op_stack_table_entries
    }

    pub fn to_base_table_row(self) -> Array1<BFieldElement> {
        let shrink_stack_indicator = match self.shrinks_stack() {
            true => bfe!(1),
            false => bfe!(0),
        };

        let mut row = Array1::zeros(BASE_WIDTH);
        row[CLK.base_table_index()] = self.clk.into();
        row[IB1ShrinkStack.base_table_index()] = shrink_stack_indicator;
        row[StackPointer.base_table_index()] = self.op_stack_pointer;
        row[FirstUnderflowElement.base_table_index()] = self.underflow_io.payload();
        row
    }
}

impl ExtOpStackTable {
    pub fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let constant = |c| circuit_builder.b_constant(c);
        let x_constant = |c| circuit_builder.x_constant(c);
        let base_row = |column: OpStackBaseTableColumn| {
            circuit_builder.input(BaseRow(column.master_base_table_index()))
        };
        let ext_row = |column: OpStackExtTableColumn| {
            circuit_builder.input(ExtRow(column.master_ext_table_index()))
        };

        let initial_stack_length = u32::try_from(OpStackElement::COUNT).unwrap();
        let initial_stack_length = constant(initial_stack_length.into());
        let padding_indicator = constant(PADDING_VALUE);

        let stack_pointer_is_16 = base_row(StackPointer) - initial_stack_length.clone();

        let compressed_row = challenge(OpStackClkWeight) * base_row(CLK)
            + challenge(OpStackIb1Weight) * base_row(IB1ShrinkStack)
            + challenge(OpStackPointerWeight) * initial_stack_length
            + challenge(OpStackFirstUnderflowElementWeight) * base_row(FirstUnderflowElement);
        let rppa_initial = challenge(OpStackIndeterminate) - compressed_row;
        let rppa_has_accumulated_first_row = ext_row(RunningProductPermArg) - rppa_initial;

        let rppa_is_default_initial =
            ext_row(RunningProductPermArg) - x_constant(PermArg::default_initial());

        let first_row_is_padding_row = base_row(IB1ShrinkStack) - padding_indicator;
        let first_row_is_not_padding_row =
            base_row(IB1ShrinkStack) * (base_row(IB1ShrinkStack) - constant(bfe!(1)));

        let rppa_starts_correctly = rppa_has_accumulated_first_row * first_row_is_padding_row
            + rppa_is_default_initial * first_row_is_not_padding_row;

        let lookup_argument_initial = x_constant(LookupArg::default_initial());
        let clock_jump_diff_log_derivative_is_initialized_correctly =
            ext_row(ClockJumpDifferenceLookupClientLogDerivative) - lookup_argument_initial;

        vec![
            stack_pointer_is_16,
            rppa_starts_correctly,
            clock_jump_diff_log_derivative_is_initialized_correctly,
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
        let constant = |c| circuit_builder.b_constant(c);
        let challenge = |c| circuit_builder.challenge(c);
        let current_base_row = |column: OpStackBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(column.master_base_table_index()))
        };
        let current_ext_row = |column: OpStackExtTableColumn| {
            circuit_builder.input(CurrentExtRow(column.master_ext_table_index()))
        };
        let next_base_row = |column: OpStackBaseTableColumn| {
            circuit_builder.input(NextBaseRow(column.master_base_table_index()))
        };
        let next_ext_row = |column: OpStackExtTableColumn| {
            circuit_builder.input(NextExtRow(column.master_ext_table_index()))
        };

        let one = constant(1_u32.into());
        let padding_indicator = constant(PADDING_VALUE);

        let clk = current_base_row(CLK);
        let ib1_shrink_stack = current_base_row(IB1ShrinkStack);
        let stack_pointer = current_base_row(StackPointer);
        let first_underflow_element = current_base_row(FirstUnderflowElement);
        let rppa = current_ext_row(RunningProductPermArg);
        let clock_jump_diff_log_derivative =
            current_ext_row(ClockJumpDifferenceLookupClientLogDerivative);

        let clk_next = next_base_row(CLK);
        let ib1_shrink_stack_next = next_base_row(IB1ShrinkStack);
        let stack_pointer_next = next_base_row(StackPointer);
        let first_underflow_element_next = next_base_row(FirstUnderflowElement);
        let rppa_next = next_ext_row(RunningProductPermArg);
        let clock_jump_diff_log_derivative_next =
            next_ext_row(ClockJumpDifferenceLookupClientLogDerivative);

        let stack_pointer_increases_by_1_or_does_not_change =
            (stack_pointer_next.clone() - stack_pointer.clone() - one.clone())
                * (stack_pointer_next.clone() - stack_pointer.clone());

        let stack_pointer_inc_by_1_or_underflow_element_doesnt_change_or_next_ci_grows_stack =
            (stack_pointer_next.clone() - stack_pointer.clone() - one.clone())
                * (first_underflow_element_next.clone() - first_underflow_element.clone())
                * ib1_shrink_stack_next.clone();

        let next_row_is_padding_row = ib1_shrink_stack_next.clone() - padding_indicator.clone();
        let if_current_row_is_padding_row_then_next_row_is_padding_row = ib1_shrink_stack.clone()
            * (ib1_shrink_stack - one.clone())
            * next_row_is_padding_row.clone();

        // The running product for the permutation argument `rppa` is updated correctly.
        let compressed_row = circuit_builder.challenge(OpStackClkWeight) * clk_next.clone()
            + circuit_builder.challenge(OpStackIb1Weight) * ib1_shrink_stack_next.clone()
            + circuit_builder.challenge(OpStackPointerWeight) * stack_pointer_next.clone()
            + circuit_builder.challenge(OpStackFirstUnderflowElementWeight)
                * first_underflow_element_next;

        let rppa_updates =
            rppa_next.clone() - rppa.clone() * (challenge(OpStackIndeterminate) - compressed_row);

        let next_row_is_not_padding_row =
            ib1_shrink_stack_next.clone() * (ib1_shrink_stack_next.clone() - one.clone());
        let rppa_remains = rppa_next - rppa;

        let rppa_updates_correctly = rppa_updates * next_row_is_padding_row.clone()
            + rppa_remains * next_row_is_not_padding_row.clone();

        let clk_diff = clk_next - clk;
        let log_derivative_accumulates = (clock_jump_diff_log_derivative_next.clone()
            - clock_jump_diff_log_derivative.clone())
            * (challenge(ClockJumpDifferenceLookupIndeterminate) - clk_diff)
            - one.clone();
        let log_derivative_remains =
            clock_jump_diff_log_derivative_next.clone() - clock_jump_diff_log_derivative.clone();

        let log_derivative_accumulates_or_stack_pointer_changes_or_next_row_is_padding_row =
            log_derivative_accumulates
                * (stack_pointer_next.clone() - stack_pointer.clone() - one.clone())
                * next_row_is_padding_row;
        let log_derivative_remains_or_stack_pointer_doesnt_change =
            log_derivative_remains.clone() * (stack_pointer_next.clone() - stack_pointer.clone());
        let log_derivatve_remains_or_next_row_is_not_padding_row =
            log_derivative_remains * next_row_is_not_padding_row;

        let log_derivative_updates_correctly =
            log_derivative_accumulates_or_stack_pointer_changes_or_next_row_is_padding_row
                + log_derivative_remains_or_stack_pointer_doesnt_change
                + log_derivatve_remains_or_next_row_is_not_padding_row;

        vec![
            stack_pointer_increases_by_1_or_does_not_change,
            stack_pointer_inc_by_1_or_underflow_element_doesnt_change_or_next_ci_grows_stack,
            if_current_row_is_padding_row_then_next_row_is_padding_row,
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

impl OpStackTable {
    /// Fills the trace table in-place and returns all clock jump differences.
    pub fn fill_trace(
        op_stack_table: &mut ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
    ) -> Vec<BFieldElement> {
        let mut op_stack_table =
            op_stack_table.slice_mut(s![0..aet.height_of_table(TableId::OpStack), ..]);
        let trace_iter = aet.op_stack_underflow_trace.rows().into_iter();

        let sorted_rows =
            trace_iter.sorted_by(|row_0, row_1| Self::compare_rows(row_0.view(), row_1.view()));
        for (row_index, row) in sorted_rows.enumerate() {
            op_stack_table.row_mut(row_index).assign(&row);
        }

        Self::clock_jump_differences(op_stack_table.view())
    }

    fn compare_rows(
        row_0: ArrayView1<BFieldElement>,
        row_1: ArrayView1<BFieldElement>,
    ) -> Ordering {
        let stack_pointer_0 = row_0[StackPointer.base_table_index()].value();
        let stack_pointer_1 = row_1[StackPointer.base_table_index()].value();
        let compare_stack_pointers = stack_pointer_0.cmp(&stack_pointer_1);

        let clk_0 = row_0[CLK.base_table_index()].value();
        let clk_1 = row_1[CLK.base_table_index()].value();
        let compare_clocks = clk_0.cmp(&clk_1);

        compare_stack_pointers.then(compare_clocks)
    }

    fn clock_jump_differences(op_stack_table: ArrayView2<BFieldElement>) -> Vec<BFieldElement> {
        let mut clock_jump_differences = vec![];
        for consecutive_rows in op_stack_table.axis_windows(Axis(0), 2) {
            let current_row = consecutive_rows.row(0);
            let next_row = consecutive_rows.row(1);
            let current_stack_pointer = current_row[StackPointer.base_table_index()];
            let next_stack_pointer = next_row[StackPointer.base_table_index()];
            if current_stack_pointer == next_stack_pointer {
                let current_clk = current_row[CLK.base_table_index()];
                let next_clk = next_row[CLK.base_table_index()];
                let clk_difference = next_clk - current_clk;
                clock_jump_differences.push(clk_difference);
            }
        }
        clock_jump_differences
    }

    pub fn pad_trace(mut op_stack_table: ArrayViewMut2<BFieldElement>, op_stack_table_len: usize) {
        let last_row_index = op_stack_table_len.saturating_sub(1);
        let mut padding_row = op_stack_table.row(last_row_index).to_owned();
        padding_row[IB1ShrinkStack.base_table_index()] = PADDING_VALUE;
        if op_stack_table_len == 0 {
            let first_stack_pointer = u32::try_from(OpStackElement::COUNT).unwrap().into();
            padding_row[StackPointer.base_table_index()] = first_stack_pointer;
        }

        let mut padding_section = op_stack_table.slice_mut(s![op_stack_table_len.., ..]);
        padding_section
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| row.assign(&padding_row));
    }

    pub fn extend(
        base_table: ArrayView2<BFieldElement>,
        mut ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        profiler!(start "op stack table");
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());

        let extension_column_indices = [
            RunningProductPermArg.ext_table_index(),
            ClockJumpDifferenceLookupClientLogDerivative.ext_table_index(),
        ];
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
            .zip_eq(extension_column_slices.into_par_iter())
            .for_each(|(generator, slice)| {
                generator(base_table, challenges).move_into(slice);
            });

        profiler!(stop "op stack table");
    }

    fn extension_column_running_product_permutation_argument(
        base_table: ArrayView2<BFieldElement>,
        challenges: &Challenges,
    ) -> Array2<XFieldElement> {
        let perm_arg_indeterminate = challenges[OpStackIndeterminate];

        let mut running_product = PermArg::default_initial();
        let mut extension_column = Vec::with_capacity(base_table.nrows());
        for row in base_table.rows() {
            if row[IB1ShrinkStack.base_table_index()] != PADDING_VALUE {
                let compressed_row = row[CLK.base_table_index()] * challenges[OpStackClkWeight]
                    + row[IB1ShrinkStack.base_table_index()] * challenges[OpStackIb1Weight]
                    + row[StackPointer.base_table_index()] * challenges[OpStackPointerWeight]
                    + row[FirstUnderflowElement.base_table_index()]
                        * challenges[OpStackFirstUnderflowElementWeight];
                running_product *= perm_arg_indeterminate - compressed_row;
            }
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
        const INVERSES_DICTIONARY_INITIAL_POPULATION: u64 = 100;
        let cjd_lookup_indeterminate = challenges[ClockJumpDifferenceLookupIndeterminate];
        let to_invert = (0..INVERSES_DICTIONARY_INITIAL_POPULATION)
            .map(|i| cjd_lookup_indeterminate - bfe!(i))
            .collect_vec();
        let inverses = XFieldElement::batch_inversion(to_invert);
        let mut inverses_dictionary = (0_u64..)
            .zip(inverses)
            .map(|(i, inv)| (bfe!(i), inv))
            .collect::<HashMap<_, _>>();

        // populate extension column using memoization
        let mut cjd_lookup_log_derivative = LookupArg::default_initial();
        let mut extension_column = Vec::with_capacity(base_table.nrows());
        extension_column.push(cjd_lookup_log_derivative);
        for (previous_row, current_row) in base_table.rows().into_iter().tuple_windows() {
            if current_row[IB1ShrinkStack.base_table_index()] == PADDING_VALUE {
                break;
            };

            let previous_stack_pointer = previous_row[StackPointer.base_table_index()];
            let current_stack_pointer = current_row[StackPointer.base_table_index()];
            if previous_stack_pointer == current_stack_pointer {
                let previous_clock = previous_row[CLK.base_table_index()];
                let current_clock = current_row[CLK.base_table_index()];
                let clock_jump_difference = current_clock - previous_clock;
                let &mut inverse = inverses_dictionary
                    .entry(clock_jump_difference)
                    .or_insert_with(|| {
                        (cjd_lookup_indeterminate - clock_jump_difference).inverse()
                    });
                cjd_lookup_log_derivative += inverse;
            }
            extension_column.push(cjd_lookup_log_derivative);
        }

        // fill padding section
        extension_column.resize(base_table.nrows(), cjd_lookup_log_derivative);
        Array2::from_shape_vec((base_table.nrows(), 1), extension_column).unwrap()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use assert2::assert;
    use itertools::Itertools;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use crate::op_stack::OpStackElement;

    use super::*;

    #[proptest]
    fn op_stack_table_entry_either_shrinks_stack_or_grows_stack(
        #[strategy(arb())] entry: OpStackTableEntry,
    ) {
        let shrinks_stack = entry.shrinks_stack();
        let grows_stack = entry.grows_stack();
        assert!(shrinks_stack ^ grows_stack);
    }

    #[proptest]
    fn op_stack_pointer_in_sequence_of_op_stack_table_entries(
        clk: u32,
        #[strategy(OpStackElement::COUNT..1024)] stack_pointer: usize,
        #[strategy(vec(arb(), ..OpStackElement::COUNT))] base_field_elements: Vec<BFieldElement>,
        sequence_of_writes: bool,
    ) {
        let sequence_length = u64::try_from(base_field_elements.len()).unwrap();
        let stack_pointer = u64::try_from(stack_pointer).unwrap();

        let underflow_io_operation = match sequence_of_writes {
            true => UnderflowIO::Write,
            false => UnderflowIO::Read,
        };
        let underflow_io = base_field_elements
            .into_iter()
            .map(underflow_io_operation)
            .collect();

        let op_stack_pointer = stack_pointer.into();
        let entries =
            OpStackTableEntry::from_underflow_io_sequence(clk, op_stack_pointer, underflow_io);
        let op_stack_pointers = entries
            .iter()
            .map(|entry| entry.op_stack_pointer.value())
            .sorted()
            .collect_vec();

        let expected_stack_pointer_range = match sequence_of_writes {
            true => stack_pointer - sequence_length..stack_pointer,
            false => stack_pointer..stack_pointer + sequence_length,
        };
        let expected_op_stack_pointers = expected_stack_pointer_range.collect_vec();
        prop_assert_eq!(expected_op_stack_pointers, op_stack_pointers);
    }

    #[proptest]
    fn clk_stays_same_in_sequence_of_op_stack_table_entries(
        clk: u32,
        #[strategy(OpStackElement::COUNT..1024)] stack_pointer: usize,
        #[strategy(vec(arb(), ..OpStackElement::COUNT))] base_field_elements: Vec<BFieldElement>,
        sequence_of_writes: bool,
    ) {
        let underflow_io_operation = match sequence_of_writes {
            true => UnderflowIO::Write,
            false => UnderflowIO::Read,
        };
        let underflow_io = base_field_elements
            .into_iter()
            .map(underflow_io_operation)
            .collect();

        let op_stack_pointer = u64::try_from(stack_pointer).unwrap().into();
        let entries =
            OpStackTableEntry::from_underflow_io_sequence(clk, op_stack_pointer, underflow_io);
        let clk_values = entries.iter().map(|entry| entry.clk).collect_vec();
        let all_clk_values_are_clk = clk_values.iter().all(|&c| c == clk);
        prop_assert!(all_clk_values_are_clk);
    }

    #[proptest]
    fn compare_rows_with_unequal_stack_pointer_and_equal_clk(
        stack_pointer_0: u64,
        stack_pointer_1: u64,
        clk: u64,
    ) {
        let mut row_0 = Array1::zeros(BASE_WIDTH);
        row_0[StackPointer.base_table_index()] = stack_pointer_0.into();
        row_0[CLK.base_table_index()] = clk.into();

        let mut row_1 = Array1::zeros(BASE_WIDTH);
        row_1[StackPointer.base_table_index()] = stack_pointer_1.into();
        row_1[CLK.base_table_index()] = clk.into();

        let stack_pointer_comparison = stack_pointer_0.cmp(&stack_pointer_1);
        let row_comparison = OpStackTable::compare_rows(row_0.view(), row_1.view());

        prop_assert_eq!(stack_pointer_comparison, row_comparison);
    }

    #[proptest]
    fn compare_rows_with_equal_stack_pointer_and_unequal_clk(
        stack_pointer: u64,
        clk_0: u64,
        clk_1: u64,
    ) {
        let mut row_0 = Array1::zeros(BASE_WIDTH);
        row_0[StackPointer.base_table_index()] = stack_pointer.into();
        row_0[CLK.base_table_index()] = clk_0.into();

        let mut row_1 = Array1::zeros(BASE_WIDTH);
        row_1[StackPointer.base_table_index()] = stack_pointer.into();
        row_1[CLK.base_table_index()] = clk_1.into();

        let clk_comparison = clk_0.cmp(&clk_1);
        let row_comparison = OpStackTable::compare_rows(row_0.view(), row_1.view());

        prop_assert_eq!(clk_comparison, row_comparison);
    }
}
