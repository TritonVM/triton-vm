use ndarray::s;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::b_field_element::BFIELD_ONE;
use twenty_first::shared_math::b_field_element::BFIELD_ZERO;
use twenty_first::shared_math::tip5::Tip5;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::challenges::Challenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::table_column::LookupBaseTableColumn;
use crate::table::table_column::LookupBaseTableColumn::*;
use crate::table::table_column::LookupExtTableColumn;
use crate::table::table_column::LookupExtTableColumn::*;
use crate::vm::AlgebraicExecutionTrace;

pub const BASE_WIDTH: usize = LookupBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = LookupExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

/// The lookup table for [Tip5](https://eprint.iacr.org/2023/107.pdf)'s split-and-lookup S-Box,
/// which uses the Fermat cube map over the finite field with 2^8 + 1 elements.
pub const LOOKUP_TABLE: [BFieldElement; 1 << 8] = lookup_table();

/// Compute the lookup table for the Fermat cube map. Helper method for the [`LOOKUP_TABLE`],
/// which should be used instead.
const fn lookup_table() -> [BFieldElement; 1 << 8] {
    let mut lookup_table = [BFIELD_ZERO; 1 << 8];
    let mut i = 0;
    while i < (1 << 8) {
        lookup_table[i] = BFieldElement::new(Tip5::LOOKUP_TABLE[i] as u64);
        i += 1;
    }
    lookup_table
}

pub struct LookupTable {}

pub struct ExtLookupTable {}

impl LookupTable {
    pub fn fill_trace(
        lookup_table: &mut ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
    ) {
    }

    pub fn pad_trace(lookup_table: &mut ArrayViewMut2<BFieldElement>) {
        // The Lookup Table is always fully populated.
        let lookup_table_length: usize = 1 << 8;
        lookup_table
            .slice_mut(s![lookup_table_length.., IsPadding.base_table_index()])
            .fill(BFIELD_ONE);
    }

    pub fn extend(
        base_table: ArrayView2<BFieldElement>,
        mut ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());
    }
}

impl ExtLookupTable {
    pub fn ext_initial_constraints_as_circuits() -> Vec<ConstraintCircuit<SingleRowIndicator>> {
        vec![]
    }

    pub fn ext_consistency_constraints_as_circuits() -> Vec<ConstraintCircuit<SingleRowIndicator>> {
        vec![]
    }

    pub fn ext_transition_constraints_as_circuits() -> Vec<ConstraintCircuit<DualRowIndicator>> {
        vec![]
    }

    pub fn ext_terminal_constraints_as_circuits() -> Vec<ConstraintCircuit<SingleRowIndicator>> {
        vec![]
    }
}
