use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;

use arbitrary::Arbitrary;
use itertools::Itertools;
use strum::EnumCount;
use strum::EnumIter;
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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, EnumCount, EnumIter)]
pub enum ConstraintType {
    /// Pertains only to the first row of the execution trace.
    Initial,

    /// Pertains to each row of the execution trace.
    Consistency,

    /// Pertains to each pair of consecutive rows of the execution trace.
    Transition,

    /// Pertains only to the last row of the execution trace.
    Terminal,
}

impl Display for ConstraintType {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            ConstraintType::Initial => write!(f, "initial"),
            ConstraintType::Consistency => write!(f, "consistency"),
            ConstraintType::Transition => write!(f, "transition"),
            ConstraintType::Terminal => write!(f, "terminal"),
        }
    }
}

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

/// Memory layout guarantees for the [*dynamic* Triton assembly AIR constraint evaluator][tasm_air].
///
/// [tasm_air]: tasm_air_constraints::dynamic_air_constraint_evaluation_tasm
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub struct DynamicTasmConstraintEvaluationMemoryLayout {
    /// Pointer to a region of memory that is reserved for a) pointers to {current,
    /// next} {main, aux} rows, and b) intermediate values in the course of constraint
    /// evaluation. The size of the region must be at least [`MEM_PAGE_SIZE`][mem_page_size]
    /// [`BFieldElement`]s.
    ///
    /// [mem_page_size]: TasmConstraintEvaluationMemoryLayout::MEM_PAGE_SIZE
    pub free_mem_page_ptr: BFieldElement,

    /// Pointer to an array of [`XFieldElement`]s of length [`NUM_CHALLENGES`][num_challenges].
    ///
    /// [num_challenges]: challenges::Challenges::COUNT
    pub challenges_ptr: BFieldElement,
}

/// Memory layout guarantees for the [*static* Triton assembly AIR constraint evaluator][tasm_air].
///
/// [tasm_air]: tasm_air_constraints::static_air_constraint_evaluation_tasm
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub struct StaticTasmConstraintEvaluationMemoryLayout {
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
    /// [num_challenges]: challenges::Challenges::COUNT
    pub challenges_ptr: BFieldElement,
}

pub trait IntegralMemoryLayout {
    /// The minimal required size of a memory page in [`BFieldElement`]s.
    const MEM_PAGE_SIZE: usize = 1 << 32;

    /// Determine if the memory layout's constraints are met, _i.e._, whether the various pointers
    /// point to large enough regions of memory.
    fn is_integral(&self) -> bool {
        let memory_regions = self.memory_regions();
        if memory_regions.iter().unique().count() != memory_regions.len() {
            return false;
        }

        let disjoint_from_all_others = |region| {
            let mut all_other_regions = memory_regions.iter().filter(|&r| r != region);
            all_other_regions.all(|r| r.disjoint_from(region))
        };
        memory_regions.iter().all(disjoint_from_all_others)
    }

    fn memory_regions(&self) -> Box<[MemoryRegion]>;
}

impl IntegralMemoryLayout for StaticTasmConstraintEvaluationMemoryLayout {
    fn memory_regions(&self) -> Box<[MemoryRegion]> {
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

impl IntegralMemoryLayout for DynamicTasmConstraintEvaluationMemoryLayout {
    fn memory_regions(&self) -> Box<[MemoryRegion]> {
        let all_regions = [
            MemoryRegion::new(self.free_mem_page_ptr, Self::MEM_PAGE_SIZE),
            MemoryRegion::new(self.challenges_ptr, challenges::Challenges::COUNT),
        ];
        Box::new(all_regions)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct MemoryRegion {
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

    impl Default for StaticTasmConstraintEvaluationMemoryLayout {
        /// For testing purposes only.
        fn default() -> Self {
            let mem_page_size = StaticTasmConstraintEvaluationMemoryLayout::MEM_PAGE_SIZE as u64;
            let mem_page = |i| bfe!(i * mem_page_size);
            StaticTasmConstraintEvaluationMemoryLayout {
                free_mem_page_ptr: mem_page(0),
                curr_base_row_ptr: mem_page(1),
                curr_ext_row_ptr: mem_page(2),
                next_base_row_ptr: mem_page(3),
                next_ext_row_ptr: mem_page(4),
                challenges_ptr: mem_page(5),
            }
        }
    }

    impl Default for DynamicTasmConstraintEvaluationMemoryLayout {
        /// For testing purposes only.
        fn default() -> Self {
            let mem_page_size = DynamicTasmConstraintEvaluationMemoryLayout::MEM_PAGE_SIZE as u64;
            let mem_page = |i| bfe!(i * mem_page_size);
            DynamicTasmConstraintEvaluationMemoryLayout {
                free_mem_page_ptr: mem_page(0),
                challenges_ptr: mem_page(1),
            }
        }
    }

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
        assert!(StaticTasmConstraintEvaluationMemoryLayout::default().is_integral());
        assert!(DynamicTasmConstraintEvaluationMemoryLayout::default().is_integral());
    }

    #[test]
    fn definitely_non_integral_memory_layout_is_detected_as_non_integral() {
        let layout = StaticTasmConstraintEvaluationMemoryLayout {
            free_mem_page_ptr: bfe!(0),
            curr_base_row_ptr: bfe!(1),
            curr_ext_row_ptr: bfe!(2),
            next_base_row_ptr: bfe!(3),
            next_ext_row_ptr: bfe!(4),
            challenges_ptr: bfe!(5),
        };
        assert!(!layout.is_integral());

        let layout = DynamicTasmConstraintEvaluationMemoryLayout {
            free_mem_page_ptr: bfe!(0),
            challenges_ptr: bfe!(5),
        };
        assert!(!layout.is_integral());
    }
}
