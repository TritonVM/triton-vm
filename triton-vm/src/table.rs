use arbitrary::Arbitrary;
use itertools::Itertools;
use twenty_first::prelude::x_field_element::EXTENSION_DEGREE;
use twenty_first::prelude::*;

use crate::instruction::AnInstruction;
use crate::instruction::LabelledInstruction;
use crate::op_stack::NumberOfWords;
pub use crate::stark::NUM_QUOTIENT_SEGMENTS;
pub use crate::table::master_table::NUM_BASE_COLUMNS;
pub use crate::table::master_table::NUM_EXT_COLUMNS;

pub mod cascade_table;
pub mod challenges;
pub mod constraint_circuit;
#[rustfmt::skip]
pub mod constraints;
pub mod cross_table_argument;
#[rustfmt::skip]
pub mod degree_lowering_table;
pub mod extension_table;
pub mod hash_table;
pub mod jump_stack_table;
pub mod lookup_table;
pub mod master_table;
pub mod op_stack_table;
pub mod processor_table;
pub mod program_table;
pub mod ram_table;
pub mod table_column;
#[rustfmt::skip]
mod tasm_air_constraints;
pub mod u32_table;

/// A single row of a [`MasterBaseTable`][table].
///
/// Usually, the elements in the table are [`BFieldElement`]s. For out-of-domain rows, which is
/// relevant for “Domain Extension to Eliminate Pretenders” (DEEP), the elements are
/// [`XFieldElement`]s.
///
/// [table]: master_table::MasterBaseTable
pub type BaseRow<T> = [T; NUM_BASE_COLUMNS];

/// A single row of a [`MasterExtensionTable`][table].
///
/// [table]: master_table::MasterExtTable
pub type ExtensionRow = [XFieldElement; NUM_EXT_COLUMNS];

/// An element of the split-up quotient polynomial.
///
/// See also [`NUM_QUOTIENT_SEGMENTS`].
pub type QuotientSegments = [XFieldElement; NUM_QUOTIENT_SEGMENTS];

/// Memory layout guarantees for the [Triton assembly AIR constraint evaluator][tasm_air].
///
/// [tasm_air]: tasm_air_constraints::air_constraint_evaluation_tasm
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub struct TasmConstraintEvaluationMemoryLayout {
    /// Pointer to a region of memory that is reserved for constraint evaluation. The size of the
    /// region must be at least [`MEM_PAGE_SIZE`][mem_page_size] [`BFieldElement`]s.
    ///
    /// [mem_page_size]: TasmConstraintEvaluationMemoryLayout::MEM_PAGE_SIZE
    pub free_mem_page_ptr: BFieldElement,

    /// Pointer to an array of [`XFieldElement`]s of length [`NUM_BASE_COLUMNS`].
    pub curr_base_row_ptr: BFieldElement,

    /// Pointer to an array of [`XFieldElement`]s of length [`NUM_EXT_COLUMNS`].
    pub curr_ext_row_ptr: BFieldElement,

    /// Pointer to an array of [`XFieldElement`]s of length [`NUM_BASE_COLUMNS`].
    pub next_base_row_ptr: BFieldElement,

    /// Pointer to an array of [`XFieldElement`]s of length [`NUM_EXT_COLUMNS`].
    pub next_ext_row_ptr: BFieldElement,

    /// Pointer to an array of [`XFieldElement`]s of length [`NUM_CHALLENGES`][num_challenges].
    ///
    /// [num_challenges]: challenges::Challenges::count()
    pub challenges_ptr: BFieldElement,
}

/// The emitted Triton assembly has the following signature:
///
/// # Signature
///
/// ```text
/// BEFORE: _
/// AFTER:  _ *evaluated_constraints
/// ```
/// # Requirements
///
/// In order for this method to emit Triton assembly, various memory regions need to be
/// declared. This is done through [`TasmConstraintEvaluationMemoryLayout`]. The memory
/// layout must be [integral].
///
/// # Guarantees
///
/// - The emitted code does not declare any labels.
/// - The emitted code is “straight-line”, _i.e._, does not contain any of the instructions
/// `call`, `return`, `recurse`, or `skiz`.
/// - The emitted code does not contain instruction `halt`.
/// - All memory write access of the emitted code is within the bounds of the memory region
/// pointed to by `*free_memory_page`.
/// - `*evaluated_constraints` points to an array of [`XFieldElement`]s of length
///   [`NUM_CONSTRAINTS`][total]. Of these,
/// - the first [`NUM_INITIAL_CONSTRAINTS`][init] elements are the evaluated initial constraints,
/// - the next [`NUM_CONSISTENCY_CONSTRAINTS`][cons] elements are the evaluated consistency constraints,
/// - the next [`NUM_TRANSITION_CONSTRAINTS`][tran] elements are the evaluated transition constraints,
/// - the last [`NUM_TERMINAL_CONSTRAINTS`][term] elements are the evaluated terminal constraints.
///
/// [integral]: TasmConstraintEvaluationMemoryLayout::is_integral
/// [total]: master_table::MasterExtTable::NUM_CONSTRAINTS
/// [init]: master_table::MasterExtTable::NUM_INITIAL_CONSTRAINTS
/// [cons]: master_table::MasterExtTable::NUM_CONSISTENCY_CONSTRAINTS
/// [tran]: master_table::MasterExtTable::NUM_TRANSITION_CONSTRAINTS
/// [term]: master_table::MasterExtTable::NUM_TERMINAL_CONSTRAINTS
pub fn air_constraint_evaluation_tasm(
    mem_layout: TasmConstraintEvaluationMemoryLayout,
) -> Vec<LabelledInstruction> {
    TasmConstraintInstantiator::new(mem_layout).instantiate_constraints()
}

impl TasmConstraintEvaluationMemoryLayout {
    /// The minimal required size of a memory page in [`BFieldElement`]s.
    pub const MEM_PAGE_SIZE: usize = 1 << 32;

    /// Determine if the memory layout's constraints are met, _i.e._, whether the various pointers
    /// point to large enough regions of memory.
    pub fn is_integral(self) -> bool {
        let memory_regions = self.into_memory_regions();
        if memory_regions.iter().unique().count() != memory_regions.len() {
            return false;
        }

        let disjoint_from_all_others = |region| {
            let mut all_other_regions = memory_regions.iter().filter(|&r| r != region);
            all_other_regions.all(|r| r.disjoint_from(region))
        };
        memory_regions.iter().all(disjoint_from_all_others)
    }

    fn into_memory_regions(self) -> Box<[MemoryRegion]> {
        let all_regions = [
            MemoryRegion::new(self.free_mem_page_ptr, Self::MEM_PAGE_SIZE),
            MemoryRegion::new(self.curr_base_row_ptr, NUM_BASE_COLUMNS),
            MemoryRegion::new(self.curr_ext_row_ptr, NUM_EXT_COLUMNS),
            MemoryRegion::new(self.next_base_row_ptr, NUM_BASE_COLUMNS),
            MemoryRegion::new(self.next_ext_row_ptr, NUM_EXT_COLUMNS),
            MemoryRegion::new(self.challenges_ptr, challenges::Challenges::count()),
        ];
        Box::new(all_regions)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) struct MemoryRegion {
    start: u64,
    size: u64,
}

impl MemoryRegion {
    pub fn new<A: Into<u64>>(address: A, size: usize) -> Self {
        let start = address.into();
        let size = u64::try_from(size).unwrap();
        Self { start, size }
    }

    pub fn disjoint_from(self, other: &Self) -> bool {
        !self.overlaps(other)
    }

    pub fn overlaps(self, other: &Self) -> bool {
        self.contains_address(other.start) || other.contains_address(self.start)
    }

    pub fn contains_address<A: Into<u64>>(self, addr: A) -> bool {
        (self.start..self.start + self.size).contains(&addr.into())
    }
}

struct TasmConstraintInstantiator {
    mem_layout: TasmConstraintEvaluationMemoryLayout,

    /// The number of elements written to the output list.
    elements_written: usize,
}

impl TasmConstraintInstantiator {
    /// An offset from the [memory layout][layout]'s `free_mem_page_ptr`, in number of
    /// [`XFieldElement`]s. Indicates the start of the to-be-returned array.
    ///
    /// [layout]: TasmConstraintEvaluationMemoryLayout
    const OUT_ARRAY_OFFSET: usize = {
        let mem_page_size = TasmConstraintEvaluationMemoryLayout::MEM_PAGE_SIZE;
        let max_num_words_for_evaluated_constraints = 1 << 16; // magic!
        let out_array_offset_in_words = mem_page_size - max_num_words_for_evaluated_constraints;
        assert!(out_array_offset_in_words % EXTENSION_DEGREE == 0);
        out_array_offset_in_words / EXTENSION_DEGREE
    };

    fn new(mem_layout: TasmConstraintEvaluationMemoryLayout) -> Self {
        let elements_written = 0;
        Self {
            mem_layout,
            elements_written,
        }
    }

    fn instantiate_constraints(&mut self) -> Vec<LabelledInstruction> {
        [
            self.instantiate_initial_constraints(),
            self.instantiate_consistency_constraints(),
            self.instantiate_transition_constraints(),
            self.instantiate_terminal_constraints(),
            self.prepare_return_values(),
        ]
        .concat()
    }

    fn load_ext_field_constant(xfe: XFieldElement) -> Vec<LabelledInstruction> {
        let [c0, c1, c2] = xfe
            .coefficients
            .map(AnInstruction::Push)
            .map(LabelledInstruction::Instruction);
        vec![c2, c1, c0]
    }

    fn load_ext_field_element_from_list(
        &self,
        list: IOList,
        element_index: usize,
    ) -> Vec<LabelledInstruction> {
        let list_offset = self.list_offset(list);

        let word_offset = element_index * EXTENSION_DEGREE;
        let start_to_read_offset = EXTENSION_DEGREE - 1;
        let word_index = word_offset + start_to_read_offset;
        let word_index = bfe!(word_index as u64);

        let push_address = AnInstruction::Push(list_offset + word_index);
        let read_mem = AnInstruction::ReadMem(NumberOfWords::N3);
        let pop = AnInstruction::Pop(NumberOfWords::N1);

        [push_address, read_mem, pop]
            .map(LabelledInstruction::Instruction)
            .to_vec()
    }

    fn store_ext_field_element(&self, element_index: usize) -> Vec<LabelledInstruction> {
        let list_offset = self.list_offset(IOList::FreeMemPage);

        let word_offset = element_index * EXTENSION_DEGREE;
        let word_index = bfe!(word_offset as u64);

        let push_address = AnInstruction::Push(list_offset + word_index);
        let write_mem = AnInstruction::WriteMem(NumberOfWords::N3);
        let pop = AnInstruction::Pop(NumberOfWords::N1);

        [push_address, write_mem, pop]
            .map(LabelledInstruction::Instruction)
            .to_vec()
    }

    fn write_into_output_list(&mut self) -> Vec<LabelledInstruction> {
        let element_index = Self::OUT_ARRAY_OFFSET + self.elements_written;
        self.elements_written += 1;
        self.store_ext_field_element(element_index)
    }

    fn prepare_return_values(&mut self) -> Vec<LabelledInstruction> {
        let list_offset = self.list_offset(IOList::FreeMemPage);

        let out_array_offset_in_num_bfes = Self::OUT_ARRAY_OFFSET * EXTENSION_DEGREE;
        let out_array_offset = bfe!(u64::try_from(out_array_offset_in_num_bfes).unwrap());

        [list_offset + out_array_offset]
            .map(AnInstruction::Push)
            .map(LabelledInstruction::Instruction)
            .to_vec()
    }
}

macro_rules! io_list {
    ($($list:ident => $ptr:ident,)*) => {
        #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
        enum IOList {
            $($list,)*
        }

        impl TasmConstraintInstantiator {
            fn list_offset(&self, list: IOList) -> BFieldElement {
                match list {
                    $(IOList::$list => self.mem_layout.$ptr,)*
                }
            }
        }
    };
}

io_list!(
    FreeMemPage => free_mem_page_ptr,
    CurrBaseRow => curr_base_row_ptr,
    CurrExtRow => curr_ext_row_ptr,
    NextBaseRow => next_base_row_ptr,
    NextExtRow => next_ext_row_ptr,
    Challenges => challenges_ptr,
);

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;
    use twenty_first::bfe;

    use super::*;

    #[proptest]
    fn size_0_memory_region_contains_no_addresses(
        #[strategy(arb())] region_start: BFieldElement,
        #[strategy(arb())] address: BFieldElement,
    ) {
        let region = MemoryRegion::new(region_start, 0);
        prop_assert!(!region.contains_address(region_start));
        prop_assert!(!region.contains_address(address));
    }

    #[proptest]
    fn size_1_memory_region_contains_only_start_address(
        #[strategy(arb())] region_start: BFieldElement,
        #[strategy(arb())]
        #[filter(#region_start != #address)]
        address: BFieldElement,
    ) {
        let region = MemoryRegion::new(region_start, 1);
        prop_assert!(region.contains_address(region_start));
        prop_assert!(!region.contains_address(address));
    }

    #[test]
    fn definitely_integral_memory_layout_is_detected_as_integral() {
        let mem_page_size = TasmConstraintEvaluationMemoryLayout::MEM_PAGE_SIZE as u64;
        let mem_page = |i| bfe!(i * mem_page_size);
        let layout = TasmConstraintEvaluationMemoryLayout {
            free_mem_page_ptr: mem_page(0),
            curr_base_row_ptr: mem_page(1),
            curr_ext_row_ptr: mem_page(2),
            next_base_row_ptr: mem_page(3),
            next_ext_row_ptr: mem_page(4),
            challenges_ptr: mem_page(5),
        };
        assert!(layout.is_integral());
    }

    #[test]
    fn definitely_non_integral_memory_layout_is_detected_as_non_integral() {
        let layout = TasmConstraintEvaluationMemoryLayout {
            free_mem_page_ptr: bfe!(0),
            curr_base_row_ptr: bfe!(1),
            curr_ext_row_ptr: bfe!(2),
            next_base_row_ptr: bfe!(3),
            next_ext_row_ptr: bfe!(4),
            challenges_ptr: bfe!(5),
        };
        assert!(!layout.is_integral());
    }
}
