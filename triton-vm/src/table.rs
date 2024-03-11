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
            MemoryRegion::new(self.challenges_ptr, challenges::Challenges::COUNT),
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
