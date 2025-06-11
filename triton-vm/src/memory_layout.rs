pub use constraint_builder::codegen::MEM_PAGE_SIZE;

use air::challenge_id::ChallengeId;
use arbitrary::Arbitrary;
use itertools::Itertools;
use strum::EnumCount;
use twenty_first::prelude::*;

use crate::table::master_table::MasterAuxTable;
use crate::table::master_table::MasterMainTable;

/// Memory layout guarantees for the [Triton assembly AIR constraint
/// evaluator][tasm_air] with input lists at dynamically known memory locations.
///
/// [tasm_air]: crate::constraints::dynamic_air_constraint_evaluation_tasm
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub struct DynamicTasmConstraintEvaluationMemoryLayout {
    /// Pointer to a region of memory that is reserved for (a) pointers to
    /// {current, next} {main, aux} rows, and (b) intermediate values in the
    /// course of constraint evaluation. The size of the region must be at
    /// least [`MEM_PAGE_SIZE`] [`BFieldElement`]s.
    pub free_mem_page_ptr: BFieldElement,

    /// Pointer to an array of [`XFieldElement`]s of length
    /// [`NUM_CHALLENGES`][num_challenges].
    ///
    /// [num_challenges]: ChallengeId::COUNT
    pub challenges_ptr: BFieldElement,
}

/// Memory layout guarantees for the [Triton assembly AIR constraint
/// evaluator][tasm_air] with input lists at statically known memory locations.
///
/// [tasm_air]: crate::constraints::static_air_constraint_evaluation_tasm
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub struct StaticTasmConstraintEvaluationMemoryLayout {
    /// Pointer to a region of memory that is reserved for constraint
    /// evaluation. The size of the region must be at least
    /// [`MEM_PAGE_SIZE`] [`BFieldElement`]s.
    pub free_mem_page_ptr: BFieldElement,

    /// Pointer to an array of [`XFieldElement`]s of length
    /// [`MasterMainTable::NUM_COLUMNS`].
    pub curr_main_row_ptr: BFieldElement,

    /// Pointer to an array of [`XFieldElement`]s of length
    /// [`MasterAuxTable::NUM_COLUMNS`].
    pub curr_aux_row_ptr: BFieldElement,

    /// Pointer to an array of [`XFieldElement`]s of length
    /// [`MasterMainTable::NUM_COLUMNS`].
    pub next_main_row_ptr: BFieldElement,

    /// Pointer to an array of [`XFieldElement`]s of length
    /// [`MasterAuxTable::NUM_COLUMNS`].
    pub next_aux_row_ptr: BFieldElement,

    /// Pointer to an array of [`XFieldElement`]s of length
    /// [`NUM_CHALLENGES`][num_challenges].
    ///
    /// [num_challenges]: ChallengeId::COUNT
    pub challenges_ptr: BFieldElement,
}

pub trait IntegralMemoryLayout {
    /// Determine if the memory layout's constraints are met, _i.e._, whether
    /// the various pointers point to large enough regions of memory.
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
            MemoryRegion::new(self.free_mem_page_ptr, MEM_PAGE_SIZE),
            MemoryRegion::new(self.curr_main_row_ptr, MasterMainTable::NUM_COLUMNS),
            MemoryRegion::new(self.curr_aux_row_ptr, MasterAuxTable::NUM_COLUMNS),
            MemoryRegion::new(self.next_main_row_ptr, MasterMainTable::NUM_COLUMNS),
            MemoryRegion::new(self.next_aux_row_ptr, MasterAuxTable::NUM_COLUMNS),
            MemoryRegion::new(self.challenges_ptr, ChallengeId::COUNT),
        ];
        Box::new(all_regions)
    }
}

impl IntegralMemoryLayout for DynamicTasmConstraintEvaluationMemoryLayout {
    fn memory_regions(&self) -> Box<[MemoryRegion]> {
        let all_regions = [
            MemoryRegion::new(self.free_mem_page_ptr, MEM_PAGE_SIZE),
            MemoryRegion::new(self.challenges_ptr, ChallengeId::COUNT),
        ];
        Box::new(all_regions)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct MemoryRegion {
    start: BFieldElement,
    size: u64,
}

impl MemoryRegion {
    pub fn new<A: Into<u64>>(address: A, size: usize) -> Self {
        let start = bfe!(address.into());
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
        // move all arithmetic to u128 to avoid overflows
        let addr = u128::from(addr.into());
        let start = u128::from(self.start.value());
        let end = start + u128::from(self.size);
        (start..end).contains(&addr)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;
    use twenty_first::bfe;

    use super::*;

    impl Default for StaticTasmConstraintEvaluationMemoryLayout {
        /// For testing purposes only.
        fn default() -> Self {
            let mem_page_size = MEM_PAGE_SIZE as u64;
            let mem_page = |i| bfe!(i * mem_page_size);
            StaticTasmConstraintEvaluationMemoryLayout {
                free_mem_page_ptr: mem_page(0),
                curr_main_row_ptr: mem_page(1),
                curr_aux_row_ptr: mem_page(2),
                next_main_row_ptr: mem_page(3),
                next_aux_row_ptr: mem_page(4),
                challenges_ptr: mem_page(5),
            }
        }
    }

    impl Default for DynamicTasmConstraintEvaluationMemoryLayout {
        /// For testing purposes only.
        fn default() -> Self {
            let mem_page_size = MEM_PAGE_SIZE as u64;
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
            curr_main_row_ptr: bfe!(1),
            curr_aux_row_ptr: bfe!(2),
            next_main_row_ptr: bfe!(3),
            next_aux_row_ptr: bfe!(4),
            challenges_ptr: bfe!(5),
        };
        assert!(!layout.is_integral());

        let layout = DynamicTasmConstraintEvaluationMemoryLayout {
            free_mem_page_ptr: bfe!(0),
            challenges_ptr: bfe!(5),
        };
        assert!(!layout.is_integral());
    }

    #[test]
    fn memory_layout_integrity_check_does_not_panic_due_to_arithmetic_overflow() {
        let mem_layout = DynamicTasmConstraintEvaluationMemoryLayout {
            free_mem_page_ptr: bfe!(BFieldElement::MAX),
            challenges_ptr: bfe!(1_u64 << 63),
        };
        assert!(mem_layout.is_integral());
    }
}
