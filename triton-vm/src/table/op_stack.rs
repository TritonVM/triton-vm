use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::Range;

use air::challenge_id::ChallengeId;
use air::cross_table_argument::CrossTableArg;
use air::cross_table_argument::LookupArg;
use air::cross_table_argument::PermArg;
use air::table::TableId;
use air::table::op_stack::OpStackTable;
use air::table::op_stack::PADDING_VALUE;
use air::table_column::MasterAuxColumn;
use air::table_column::MasterMainColumn;
use air::table_column::OpStackAuxColumn;
use arbitrary::Arbitrary;
use isa::op_stack::OpStackElement;
use isa::op_stack::UnderflowIO;
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

type MainColumn = <OpStackTable as air::AIR>::MainColumn;
type AuxColumn = <OpStackTable as air::AIR>::AuxColumn;

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

    pub fn to_main_table_row(self) -> Array1<BFieldElement> {
        let shrink_stack_indicator = if self.shrinks_stack() {
            bfe!(1)
        } else {
            bfe!(0)
        };

        let mut row = Array1::zeros(MainColumn::COUNT);
        row[MainColumn::CLK.main_index()] = self.clk.into();
        row[MainColumn::IB1ShrinkStack.main_index()] = shrink_stack_indicator;
        row[MainColumn::StackPointer.main_index()] = self.op_stack_pointer;
        row[MainColumn::FirstUnderflowElement.main_index()] = self.underflow_io.payload();
        row
    }
}

fn auxiliary_column_running_product_permutation_argument(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    let perm_arg_indeterminate = challenges[ChallengeId::OpStackIndeterminate];

    let mut running_product = PermArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    for row in main_table.rows() {
        if row[MainColumn::IB1ShrinkStack.main_index()] != PADDING_VALUE {
            let compressed_row = row[MainColumn::CLK.main_index()]
                * challenges[ChallengeId::OpStackClkWeight]
                + row[MainColumn::IB1ShrinkStack.main_index()]
                    * challenges[ChallengeId::OpStackIb1Weight]
                + row[MainColumn::StackPointer.main_index()]
                    * challenges[ChallengeId::OpStackPointerWeight]
                + row[MainColumn::FirstUnderflowElement.main_index()]
                    * challenges[ChallengeId::OpStackFirstUnderflowElementWeight];
            running_product *= perm_arg_indeterminate - compressed_row;
        }
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
    let cjd_lookup_indeterminate = challenges[ChallengeId::ClockJumpDifferenceLookupIndeterminate];
    let to_invert = PRECOMPUTE_INVERSES_OF
        .map(|i| cjd_lookup_indeterminate - bfe!(i))
        .collect_vec();
    let inverses = XFieldElement::batch_inversion(to_invert);
    let mut inverses_dictionary = PRECOMPUTE_INVERSES_OF
        .zip_eq(inverses)
        .map(|(i, inv)| (bfe!(i), inv))
        .collect::<HashMap<_, _>>();

    // populate auxiliary column using memoization
    let mut cjd_lookup_log_derivative = LookupArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    auxiliary_column.push(cjd_lookup_log_derivative);
    for (previous_row, current_row) in main_table.rows().into_iter().tuple_windows() {
        if current_row[MainColumn::IB1ShrinkStack.main_index()] == PADDING_VALUE {
            break;
        };

        let previous_stack_pointer = previous_row[MainColumn::StackPointer.main_index()];
        let current_stack_pointer = current_row[MainColumn::StackPointer.main_index()];
        if previous_stack_pointer == current_stack_pointer {
            let previous_clock = previous_row[MainColumn::CLK.main_index()];
            let current_clock = current_row[MainColumn::CLK.main_index()];
            let clock_jump_difference = current_clock - previous_clock;
            let &mut inverse = inverses_dictionary
                .entry(clock_jump_difference)
                .or_insert_with(|| (cjd_lookup_indeterminate - clock_jump_difference).inverse());
            cjd_lookup_log_derivative += inverse;
        }
        auxiliary_column.push(cjd_lookup_log_derivative);
    }

    // fill padding section
    auxiliary_column.resize(main_table.nrows(), cjd_lookup_log_derivative);
    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

impl TraceTable for OpStackTable {
    type FillParam = ();
    type FillReturnInfo = Vec<BFieldElement>;

    fn fill(
        mut op_stack_table: ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
        _: Self::FillParam,
    ) -> Vec<BFieldElement> {
        let mut op_stack_table =
            op_stack_table.slice_mut(s![0..aet.height_of_table(TableId::OpStack), ..]);
        let trace_iter = aet.op_stack_underflow_trace.rows().into_iter();

        let sorted_rows =
            trace_iter.sorted_by(|row_0, row_1| compare_rows(row_0.view(), row_1.view()));
        for (row_index, row) in sorted_rows.enumerate() {
            op_stack_table.row_mut(row_index).assign(&row);
        }

        clock_jump_differences(op_stack_table.view())
    }

    fn pad(mut op_stack_table: ArrayViewMut2<BFieldElement>, op_stack_table_len: usize) {
        let last_row_index = op_stack_table_len.saturating_sub(1);
        let mut padding_row = op_stack_table.row(last_row_index).to_owned();
        padding_row[MainColumn::IB1ShrinkStack.main_index()] = PADDING_VALUE;
        if op_stack_table_len == 0 {
            let first_stack_pointer = u32::try_from(OpStackElement::COUNT).unwrap().into();
            padding_row[MainColumn::StackPointer.main_index()] = first_stack_pointer;
        }

        let mut padding_section = op_stack_table.slice_mut(s![op_stack_table_len.., ..]);
        padding_section
            .axis_iter_mut(ROW_AXIS)
            .into_par_iter()
            .for_each(|mut row| row.assign(&padding_row));
    }

    fn extend(
        main_table: ArrayView2<BFieldElement>,
        mut aux_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        profiler!(start "op stack table");
        assert_eq!(MainColumn::COUNT, main_table.ncols());
        assert_eq!(AuxColumn::COUNT, aux_table.ncols());
        assert_eq!(main_table.nrows(), aux_table.nrows());

        let auxiliary_column_indices = OpStackAuxColumn::iter()
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

        profiler!(stop "op stack table");
    }
}

fn compare_rows(row_0: ArrayView1<BFieldElement>, row_1: ArrayView1<BFieldElement>) -> Ordering {
    let stack_pointer_0 = row_0[MainColumn::StackPointer.main_index()].value();
    let stack_pointer_1 = row_1[MainColumn::StackPointer.main_index()].value();
    let compare_stack_pointers = stack_pointer_0.cmp(&stack_pointer_1);

    let clk_0 = row_0[MainColumn::CLK.main_index()].value();
    let clk_1 = row_1[MainColumn::CLK.main_index()].value();
    let compare_clocks = clk_0.cmp(&clk_1);

    compare_stack_pointers.then(compare_clocks)
}

fn clock_jump_differences(op_stack_table: ArrayView2<BFieldElement>) -> Vec<BFieldElement> {
    let mut clock_jump_differences = vec![];
    for consecutive_rows in op_stack_table.axis_windows(ROW_AXIS, 2) {
        let current_row = consecutive_rows.row(0);
        let next_row = consecutive_rows.row(1);
        let current_stack_pointer = current_row[MainColumn::StackPointer.main_index()];
        let next_stack_pointer = next_row[MainColumn::StackPointer.main_index()];
        if current_stack_pointer == next_stack_pointer {
            let current_clk = current_row[MainColumn::CLK.main_index()];
            let next_clk = next_row[MainColumn::CLK.main_index()];
            let clk_difference = next_clk - current_clk;
            clock_jump_differences.push(clk_difference);
        }
    }
    clock_jump_differences
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) mod tests {
    use assert2::assert;
    use isa::op_stack::OpStackElement;
    use itertools::Itertools;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

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
        let mut row_0 = Array1::zeros(MainColumn::COUNT);
        row_0[MainColumn::StackPointer.main_index()] = stack_pointer_0.into();
        row_0[MainColumn::CLK.main_index()] = clk.into();

        let mut row_1 = Array1::zeros(MainColumn::COUNT);
        row_1[MainColumn::StackPointer.main_index()] = stack_pointer_1.into();
        row_1[MainColumn::CLK.main_index()] = clk.into();

        let stack_pointer_comparison = stack_pointer_0.cmp(&stack_pointer_1);
        let row_comparison = compare_rows(row_0.view(), row_1.view());

        prop_assert_eq!(stack_pointer_comparison, row_comparison);
    }

    #[proptest]
    fn compare_rows_with_equal_stack_pointer_and_unequal_clk(
        stack_pointer: u64,
        clk_0: u64,
        clk_1: u64,
    ) {
        let mut row_0 = Array1::zeros(MainColumn::COUNT);
        row_0[MainColumn::StackPointer.main_index()] = stack_pointer.into();
        row_0[MainColumn::CLK.main_index()] = clk_0.into();

        let mut row_1 = Array1::zeros(MainColumn::COUNT);
        row_1[MainColumn::StackPointer.main_index()] = stack_pointer.into();
        row_1[MainColumn::CLK.main_index()] = clk_1.into();

        let clk_comparison = clk_0.cmp(&clk_1);
        let row_comparison = compare_rows(row_0.view(), row_1.view());

        prop_assert_eq!(clk_comparison, row_comparison);
    }
}
