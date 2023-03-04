use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::tip5;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::challenges::Challenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::ConstraintCircuitMonad;
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

    pub fn lookup_16_bit_limb(to_look_up: u16) -> BFieldElement {
        let to_look_up_lo = (to_look_up & 0xff) as usize;
        let to_look_up_hi = ((to_look_up >> 8) & 0xff) as usize;
        let looked_up_lo = tip5::LOOKUP_TABLE[to_look_up_lo] as u64;
        let looked_up_hi = tip5::LOOKUP_TABLE[to_look_up_hi] as u64;
        let looked_up = (looked_up_hi << 8) | looked_up_lo;
        BFieldElement::from_raw_u64(looked_up)
    }
}

impl ExtCascadeTable {
    pub fn ext_initial_constraints_as_circuits() -> Vec<ConstraintCircuit<SingleRowIndicator>> {
        let mut constraints = [];
        ConstraintCircuitMonad::constant_folding(&mut constraints);
        constraints.map(|circuit| circuit.consume()).to_vec()
    }

    pub fn ext_consistency_constraints_as_circuits() -> Vec<ConstraintCircuit<SingleRowIndicator>> {
        let mut constraints = [];
        ConstraintCircuitMonad::constant_folding(&mut constraints);
        constraints.map(|circuit| circuit.consume()).to_vec()
    }

    pub fn ext_transition_constraints_as_circuits() -> Vec<ConstraintCircuit<DualRowIndicator>> {
        let mut constraints = [];
        ConstraintCircuitMonad::constant_folding(&mut constraints);
        constraints.map(|circuit| circuit.consume()).to_vec()
    }

    pub fn ext_terminal_constraints_as_circuits() -> Vec<ConstraintCircuit<SingleRowIndicator>> {
        let mut constraints = [];
        ConstraintCircuitMonad::constant_folding(&mut constraints);
        constraints.map(|circuit| circuit.consume()).to_vec()
    }
}
