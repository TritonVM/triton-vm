use arbitrary::Arbitrary;
use itertools::Itertools;
use twenty_first::prelude::BFieldElement;
use twenty_first::prelude::XFieldElement;

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
pub mod tasm_air_constraints;
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
    /// region must be at least 2^32 [`BFieldElement`]s.
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

impl TasmConstraintEvaluationMemoryLayout {
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
            MemoryRegion::new(self.free_mem_page_ptr, 1 << 32),
            MemoryRegion::new(self.curr_base_row_ptr, NUM_BASE_COLUMNS as u64),
            MemoryRegion::new(self.curr_ext_row_ptr, NUM_EXT_COLUMNS as u64),
            MemoryRegion::new(self.next_base_row_ptr, NUM_BASE_COLUMNS as u64),
            MemoryRegion::new(self.next_ext_row_ptr, NUM_EXT_COLUMNS as u64),
            MemoryRegion::new(self.challenges_ptr, challenges::Challenges::count() as u64),
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
    pub fn new(pointer: BFieldElement, size: u64) -> Self {
        let start = pointer.value();
        Self { start, size }
    }

    pub fn disjoint_from(self, other: &Self) -> bool {
        !self.overlaps(other)
    }

    pub fn overlaps(self, other: &Self) -> bool {
        self.contains_pointer(other.start) || other.contains_pointer(self.start)
    }

    pub fn contains_pointer(self, ptr: u64) -> bool {
        (self.start..self.start + self.size).contains(&ptr)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use super::*;

    #[proptest]
    fn size_0_memory_region_contains_no_addresses(#[strategy(arb())] pointer: BFieldElement) {
        let one = BFieldElement::new(1);
        let region = MemoryRegion::new(pointer, 0);

        prop_assert!(!region.contains_pointer((pointer - one).value()));
        prop_assert!(!region.contains_pointer(pointer.value()));
        prop_assert!(!region.contains_pointer((pointer + one).value()));
    }

    #[proptest]
    fn size_1_memory_region_contains_only_start_address(#[strategy(arb())] pointer: BFieldElement) {
        let one = BFieldElement::new(1);
        let region = MemoryRegion::new(pointer, 1);

        prop_assert!(!region.contains_pointer((pointer - one).value()));
        prop_assert!(region.contains_pointer(pointer.value()));
        prop_assert!(!region.contains_pointer((pointer + one).value()));
    }

    #[test]
    fn definitely_integral_memory_layout_is_detected_as_integral() {
        let mem_page = |i: u64| BFieldElement::new(i * (1 << 32));
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
        let bfe = BFieldElement::new;
        let layout = TasmConstraintEvaluationMemoryLayout {
            free_mem_page_ptr: bfe(0),
            curr_base_row_ptr: bfe(1),
            curr_ext_row_ptr: bfe(2),
            next_base_row_ptr: bfe(3),
            next_ext_row_ptr: bfe(4),
            challenges_ptr: bfe(5),
        };
        assert!(!layout.is_integral());
    }
}
