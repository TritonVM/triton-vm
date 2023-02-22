use ndarray::s;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::b_field_element::BFIELD_ONE;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::challenges::Challenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::table_column::CascadeBaseTableColumn;
use crate::table::table_column::CascadeExtTableColumn;
use crate::vm::AlgebraicExecutionTrace;

pub const BASE_WIDTH: usize = CascadeBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = CascadeExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

pub struct CascadeTable {}

pub struct ExtCascadeTable {}

impl CascadeTable {
    pub fn fill_trace(
        cascade_table: &mut ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
    ) {
    }

    pub fn pad_trace(
        cascade_table: &mut ArrayViewMut2<BFieldElement>,
        cascade_table_length: usize,
    ) {
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

impl ExtCascadeTable {
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
