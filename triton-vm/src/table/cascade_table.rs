use ndarray::s;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use num_traits::One;
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::b_field_element::BFIELD_ONE;
use twenty_first::shared_math::tip5;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::challenges::ChallengeId::*;
use crate::table::challenges::Challenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::ConstraintCircuitMonad;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::LookupArg;
use crate::table::table_column::CascadeBaseTableColumn;
use crate::table::table_column::CascadeBaseTableColumn::*;
use crate::table::table_column::CascadeExtTableColumn;
use crate::table::table_column::CascadeExtTableColumn::*;
use crate::table::table_column::MasterBaseTableColumn;
use crate::table::table_column::MasterExtTableColumn;
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
        for (row_idx, (&to_look_up, &multiplicity)) in
            aet.cascade_table_lookup_multiplicities.iter().enumerate()
        {
            let to_look_up_lo = (to_look_up & 0xff) as u8;
            let to_look_up_hi = ((to_look_up >> 8) & 0xff) as u8;

            let mut row = cascade_table.row_mut(row_idx);
            row[LookInLo.base_table_index()] = BFieldElement::from_raw_u64(to_look_up_lo as u64);
            row[LookInHi.base_table_index()] = BFieldElement::from_raw_u64(to_look_up_hi as u64);
            row[LookOutLo.base_table_index()] = Self::lookup_8_bit_limb(to_look_up_lo);
            row[LookOutHi.base_table_index()] = Self::lookup_8_bit_limb(to_look_up_hi);
            row[LookupMultiplicity.base_table_index()] = BFieldElement::new(multiplicity);
        }
    }

    pub fn pad_trace(
        cascade_table: &mut ArrayViewMut2<BFieldElement>,
        cascade_table_length: usize,
    ) {
        cascade_table
            .slice_mut(s![cascade_table_length.., IsPadding.base_table_index()])
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

        let mut lookup_table_log_derivative = LookupArg::default_initial();

        let lookup_input_weight = challenges.get_challenge(LookupTableInputWeight);
        let lookup_output_weight = challenges.get_challenge(LookupTableOutputWeight);
        let lookup_indeterminate = challenges.get_challenge(CascadeLookupIndeterminate);

        for row_idx in 0..base_table.nrows() {
            let base_row = base_table.row(row_idx);
            let is_padding = base_row[IsPadding.base_table_index()].is_one();

            if !is_padding {
                let compressed_row_lo = lookup_input_weight * base_row[LookInLo.base_table_index()]
                    + lookup_output_weight * base_row[LookOutLo.base_table_index()];
                let compressed_row_hi = lookup_input_weight * base_row[LookInHi.base_table_index()]
                    + lookup_output_weight * base_row[LookOutHi.base_table_index()];
                lookup_table_log_derivative += (lookup_indeterminate - compressed_row_lo).inverse();
                lookup_table_log_derivative += (lookup_indeterminate - compressed_row_hi).inverse();
            }

            let mut extension_row = ext_table.row_mut(row_idx);
            extension_row[LookupTableClientLogDerivative.ext_table_index()] =
                lookup_table_log_derivative;
        }
    }

    fn lookup_8_bit_limb(to_look_up: u8) -> BFieldElement {
        let looked_up = tip5::LOOKUP_TABLE[to_look_up as usize] as u64;
        BFieldElement::from_raw_u64(looked_up)
    }

    pub fn lookup_16_bit_limb(to_look_up: u16) -> BFieldElement {
        let to_look_up_lo = (to_look_up & 0xff) as u8;
        let to_look_up_hi = ((to_look_up >> 8) & 0xff) as u8;
        let looked_up_lo = Self::lookup_8_bit_limb(to_look_up_lo);
        let looked_up_hi = Self::lookup_8_bit_limb(to_look_up_hi);
        let two_pow_8 = BFieldElement::new(1 << 8);
        two_pow_8 * looked_up_hi + looked_up_lo
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
