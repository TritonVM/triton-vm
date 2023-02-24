use ndarray::s;
use ndarray::Array1;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::b_field_element::BFIELD_ONE;
use twenty_first::shared_math::tip5;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::challenges::Challenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::table_column::LookupBaseTableColumn;
use crate::table::table_column::LookupBaseTableColumn::*;
use crate::table::table_column::LookupExtTableColumn;
use crate::table::table_column::MasterBaseTableColumn;
use crate::vm::AlgebraicExecutionTrace;

pub const BASE_WIDTH: usize = LookupBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = LookupExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

pub struct LookupTable {}

pub struct ExtLookupTable {}

impl LookupTable {
    pub fn fill_trace(
        lookup_table: &mut ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
    ) {
        assert!(lookup_table.nrows() >= 1 << 8);

        let lookup_input = Array1::from_iter((0_u64..1 << 8).map(BFieldElement::new));
        let lookup_output = Array1::from_iter(
            (0..1 << 8).map(|i| BFieldElement::new(tip5::LOOKUP_TABLE[i] as u64)),
        );
        let lookup_multiplicities = Array1::from_iter(
            aet.lookup_table_lookup_multiplicities
                .map(BFieldElement::new),
        );

        let lookup_input_column =
            lookup_table.slice_mut(s![..1_usize << 8, LookIn.base_table_index()]);
        lookup_input.move_into(lookup_input_column);
        let lookup_output_column =
            lookup_table.slice_mut(s![..1_usize << 8, LookOut.base_table_index()]);
        lookup_output.move_into(lookup_output_column);
        let lookup_multiplicities_column =
            lookup_table.slice_mut(s![..1_usize << 8, LookupMultiplicity.base_table_index()]);
        lookup_multiplicities.move_into(lookup_multiplicities_column);
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
