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

use crate::aet::AlgebraicExecutionTrace;
use crate::table::challenges::ChallengeId;
use crate::table::challenges::ChallengeId::*;
use crate::table::challenges::Challenges;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::ConstraintCircuitMonad;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::DualRowIndicator::*;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::*;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::LookupArg;
use crate::table::table_column::CascadeBaseTableColumn;
use crate::table::table_column::CascadeBaseTableColumn::*;
use crate::table::table_column::CascadeExtTableColumn;
use crate::table::table_column::CascadeExtTableColumn::*;
use crate::table::table_column::MasterBaseTableColumn;
use crate::table::table_column::MasterExtTableColumn;

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
            row[LookInLo.base_table_index()] = BFieldElement::new(to_look_up_lo as u64);
            row[LookInHi.base_table_index()] = BFieldElement::new(to_look_up_hi as u64);
            row[LookOutLo.base_table_index()] = Self::lookup_8_bit_limb(to_look_up_lo);
            row[LookOutHi.base_table_index()] = Self::lookup_8_bit_limb(to_look_up_hi);
            row[LookupMultiplicity.base_table_index()] = BFieldElement::new(multiplicity);
        }
    }

    pub fn pad_trace(mut cascade_table: ArrayViewMut2<BFieldElement>, cascade_table_length: usize) {
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

        let mut hash_table_log_derivative = LookupArg::default_initial();
        let mut lookup_table_log_derivative = LookupArg::default_initial();

        let two_pow_8 = BFieldElement::new(1 << 8);

        let hash_indeterminate = challenges[HashCascadeLookupIndeterminate];
        let hash_input_weight = challenges[HashCascadeLookInWeight];
        let hash_output_weight = challenges[HashCascadeLookOutWeight];

        let lookup_indeterminate = challenges[CascadeLookupIndeterminate];
        let lookup_input_weight = challenges[LookupTableInputWeight];
        let lookup_output_weight = challenges[LookupTableOutputWeight];

        for row_idx in 0..base_table.nrows() {
            let base_row = base_table.row(row_idx);
            let is_padding = base_row[IsPadding.base_table_index()].is_one();

            if !is_padding {
                let look_in = two_pow_8 * base_row[LookInHi.base_table_index()]
                    + base_row[LookInLo.base_table_index()];
                let look_out = two_pow_8 * base_row[LookOutHi.base_table_index()]
                    + base_row[LookOutLo.base_table_index()];
                let compressed_row_hash =
                    hash_input_weight * look_in + hash_output_weight * look_out;
                let lookup_multiplicity = base_row[LookupMultiplicity.base_table_index()];
                hash_table_log_derivative +=
                    (hash_indeterminate - compressed_row_hash).inverse() * lookup_multiplicity;

                let compressed_row_lo = lookup_input_weight * base_row[LookInLo.base_table_index()]
                    + lookup_output_weight * base_row[LookOutLo.base_table_index()];
                let compressed_row_hi = lookup_input_weight * base_row[LookInHi.base_table_index()]
                    + lookup_output_weight * base_row[LookOutHi.base_table_index()];
                lookup_table_log_derivative += (lookup_indeterminate - compressed_row_lo).inverse();
                lookup_table_log_derivative += (lookup_indeterminate - compressed_row_hi).inverse();
            }

            let mut extension_row = ext_table.row_mut(row_idx);
            extension_row[HashTableServerLogDerivative.ext_table_index()] =
                hash_table_log_derivative;
            extension_row[LookupTableClientLogDerivative.ext_table_index()] =
                lookup_table_log_derivative;
        }
    }

    fn lookup_8_bit_limb(to_look_up: u8) -> BFieldElement {
        let looked_up = tip5::LOOKUP_TABLE[to_look_up as usize] as u64;
        BFieldElement::new(looked_up)
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
    pub fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let base_row = |col_id: CascadeBaseTableColumn| {
            circuit_builder.input(BaseRow(col_id.master_base_table_index()))
        };
        let ext_row = |col_id: CascadeExtTableColumn| {
            circuit_builder.input(ExtRow(col_id.master_ext_table_index()))
        };
        let challenge = |challenge_id: ChallengeId| circuit_builder.challenge(challenge_id);

        let one = circuit_builder.b_constant(BFIELD_ONE);
        let two = circuit_builder.b_constant(BFieldElement::new(2));
        let two_pow_8 = circuit_builder.b_constant(BFieldElement::new(1 << 8));
        let lookup_arg_default_initial = circuit_builder.x_constant(LookupArg::default_initial());

        let is_padding = base_row(IsPadding);
        let look_in_hi = base_row(LookInHi);
        let look_in_lo = base_row(LookInLo);
        let look_out_hi = base_row(LookOutHi);
        let look_out_lo = base_row(LookOutLo);
        let lookup_multiplicity = base_row(LookupMultiplicity);
        let hash_table_server_log_derivative = ext_row(HashTableServerLogDerivative);
        let lookup_table_client_log_derivative = ext_row(LookupTableClientLogDerivative);

        let hash_indeterminate = challenge(HashCascadeLookupIndeterminate);
        let hash_input_weight = challenge(HashCascadeLookInWeight);
        let hash_output_weight = challenge(HashCascadeLookOutWeight);

        let lookup_indeterminate = challenge(CascadeLookupIndeterminate);
        let lookup_input_weight = challenge(LookupTableInputWeight);
        let lookup_output_weight = challenge(LookupTableOutputWeight);

        // Lookup Argument with Hash Table
        let compressed_row_hash = hash_input_weight
            * (two_pow_8.clone() * look_in_hi.clone() + look_in_lo.clone())
            + hash_output_weight * (two_pow_8 * look_out_hi.clone() + look_out_lo.clone());
        let hash_table_log_derivative_is_default_initial =
            hash_table_server_log_derivative.clone() - lookup_arg_default_initial.clone();
        let hash_table_log_derivative_has_accumulated_first_row = (hash_table_server_log_derivative
            - lookup_arg_default_initial.clone())
            * (hash_indeterminate - compressed_row_hash)
            - lookup_multiplicity;
        let hash_table_log_derivative_is_initialized_correctly = (one.clone() - is_padding.clone())
            * hash_table_log_derivative_has_accumulated_first_row
            + is_padding.clone() * hash_table_log_derivative_is_default_initial;

        // Lookup Argument with Lookup Table
        let compressed_row_lo =
            lookup_input_weight.clone() * look_in_lo + lookup_output_weight.clone() * look_out_lo;
        let compressed_row_hi =
            lookup_input_weight * look_in_hi + lookup_output_weight * look_out_hi;
        let lookup_table_log_derivative_is_default_initial =
            lookup_table_client_log_derivative.clone() - lookup_arg_default_initial.clone();
        let lookup_table_log_derivative_has_accumulated_first_row =
            (lookup_table_client_log_derivative - lookup_arg_default_initial)
                * (lookup_indeterminate.clone() - compressed_row_lo.clone())
                * (lookup_indeterminate.clone() - compressed_row_hi.clone())
                - two * lookup_indeterminate
                + compressed_row_lo
                + compressed_row_hi;
        let lookup_table_log_derivative_is_initialized_correctly = (one - is_padding.clone())
            * lookup_table_log_derivative_has_accumulated_first_row
            + is_padding * lookup_table_log_derivative_is_default_initial;

        vec![
            hash_table_log_derivative_is_initialized_correctly,
            lookup_table_log_derivative_is_initialized_correctly,
        ]
    }

    pub fn consistency_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let base_row = |col_id: CascadeBaseTableColumn| {
            circuit_builder.input(BaseRow(col_id.master_base_table_index()))
        };

        let one = circuit_builder.b_constant(BFIELD_ONE);
        let is_padding = base_row(IsPadding);
        let is_padding_is_0_or_1 = is_padding.clone() * (one - is_padding);

        vec![is_padding_is_0_or_1]
    }

    pub fn transition_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let constant = |c: u64| circuit_builder.b_constant(c.into());

        let current_base_row = |column_idx: CascadeBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(column_idx.master_base_table_index()))
        };
        let next_base_row = |column_idx: CascadeBaseTableColumn| {
            circuit_builder.input(NextBaseRow(column_idx.master_base_table_index()))
        };
        let current_ext_row = |column_idx: CascadeExtTableColumn| {
            circuit_builder.input(CurrentExtRow(column_idx.master_ext_table_index()))
        };
        let next_ext_row = |column_idx: CascadeExtTableColumn| {
            circuit_builder.input(NextExtRow(column_idx.master_ext_table_index()))
        };

        let one = constant(1);
        let two = constant(2);
        let two_pow_8 = constant(1 << 8);

        let is_padding = current_base_row(IsPadding);
        let hash_table_server_log_derivative = current_ext_row(HashTableServerLogDerivative);
        let lookup_table_client_log_derivative = current_ext_row(LookupTableClientLogDerivative);

        let is_padding_next = next_base_row(IsPadding);
        let look_in_hi_next = next_base_row(LookInHi);
        let look_in_lo_next = next_base_row(LookInLo);
        let look_out_hi_next = next_base_row(LookOutHi);
        let look_out_lo_next = next_base_row(LookOutLo);
        let lookup_multiplicity_next = next_base_row(LookupMultiplicity);
        let hash_table_server_log_derivative_next = next_ext_row(HashTableServerLogDerivative);
        let lookup_table_client_log_derivative_next = next_ext_row(LookupTableClientLogDerivative);

        let hash_indeterminate = challenge(HashCascadeLookupIndeterminate);
        let hash_input_weight = challenge(HashCascadeLookInWeight);
        let hash_output_weight = challenge(HashCascadeLookOutWeight);

        let lookup_indeterminate = challenge(CascadeLookupIndeterminate);
        let lookup_input_weight = challenge(LookupTableInputWeight);
        let lookup_output_weight = challenge(LookupTableOutputWeight);

        // Padding region is contiguous: if current row is padding, then next row is padding.
        let if_current_row_is_padding_row_then_next_row_is_padding_row =
            is_padding * (one.clone() - is_padding_next.clone());

        // Lookup Argument with Hash Table
        let compressed_next_row_hash = hash_input_weight
            * (two_pow_8.clone() * look_in_hi_next.clone() + look_in_lo_next.clone())
            + hash_output_weight
                * (two_pow_8 * look_out_hi_next.clone() + look_out_lo_next.clone());
        let hash_table_log_derivative_remains = hash_table_server_log_derivative_next.clone()
            - hash_table_server_log_derivative.clone();
        let hash_table_log_derivative_accumulates_next_row = (hash_table_server_log_derivative_next
            - hash_table_server_log_derivative)
            * (hash_indeterminate - compressed_next_row_hash)
            - lookup_multiplicity_next;
        let hash_table_log_derivative_updates_correctly = (one.clone() - is_padding_next.clone())
            * hash_table_log_derivative_accumulates_next_row
            + is_padding_next.clone() * hash_table_log_derivative_remains;

        // Lookup Argument with Lookup Table
        let compressed_row_lo_next = lookup_input_weight.clone() * look_in_lo_next
            + lookup_output_weight.clone() * look_out_lo_next;
        let compressed_row_hi_next =
            lookup_input_weight * look_in_hi_next + lookup_output_weight * look_out_hi_next;
        let lookup_table_log_derivative_remains = lookup_table_client_log_derivative_next.clone()
            - lookup_table_client_log_derivative.clone();
        let lookup_table_log_derivative_accumulates_next_row =
            (lookup_table_client_log_derivative_next - lookup_table_client_log_derivative)
                * (lookup_indeterminate.clone() - compressed_row_lo_next.clone())
                * (lookup_indeterminate.clone() - compressed_row_hi_next.clone())
                - two * lookup_indeterminate
                + compressed_row_lo_next
                + compressed_row_hi_next;
        let lookup_table_log_derivative_updates_correctly = (one - is_padding_next.clone())
            * lookup_table_log_derivative_accumulates_next_row
            + is_padding_next * lookup_table_log_derivative_remains;

        vec![
            if_current_row_is_padding_row_then_next_row_is_padding_row,
            hash_table_log_derivative_updates_correctly,
            lookup_table_log_derivative_updates_correctly,
        ]
    }

    pub fn terminal_constraints(
        _circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        // no further constraints
        vec![]
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use num_traits::Zero;

    pub fn constraints_evaluate_to_zero(
        master_base_trace_table: ArrayView2<BFieldElement>,
        master_ext_trace_table: ArrayView2<XFieldElement>,
        challenges: &Challenges,
    ) -> bool {
        let zero = XFieldElement::zero();
        assert_eq!(
            master_base_trace_table.nrows(),
            master_ext_trace_table.nrows()
        );

        let circuit_builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in ExtCascadeTable::initial_constraints(&circuit_builder)
            .into_iter()
            .map(|constraint_monad| constraint_monad.consume())
            .enumerate()
        {
            let evaluated_constraint = constraint.evaluate(
                master_base_trace_table.slice(s![..1, ..]),
                master_ext_trace_table.slice(s![..1, ..]),
                challenges,
            );
            assert_eq!(
                zero, evaluated_constraint,
                "Initial constraint {constraint_idx} failed."
            );
        }

        let circuit_builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in
            ExtCascadeTable::consistency_constraints(&circuit_builder)
                .into_iter()
                .map(|constraint_monad| constraint_monad.consume())
                .enumerate()
        {
            for row_idx in 0..master_base_trace_table.nrows() {
                let evaluated_constraint = constraint.evaluate(
                    master_base_trace_table.slice(s![row_idx..row_idx + 1, ..]),
                    master_ext_trace_table.slice(s![row_idx..row_idx + 1, ..]),
                    challenges,
                );
                assert_eq!(
                    zero, evaluated_constraint,
                    "Consistency constraint {constraint_idx} failed on row {row_idx}."
                );
            }
        }

        let circuit_builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in
            ExtCascadeTable::transition_constraints(&circuit_builder)
                .into_iter()
                .map(|constraint_monad| constraint_monad.consume())
                .enumerate()
        {
            for row_idx in 0..master_base_trace_table.nrows() - 1 {
                let evaluated_constraint = constraint.evaluate(
                    master_base_trace_table.slice(s![row_idx..row_idx + 2, ..]),
                    master_ext_trace_table.slice(s![row_idx..row_idx + 2, ..]),
                    challenges,
                );
                assert_eq!(
                    zero, evaluated_constraint,
                    "Transition constraint {constraint_idx} failed on row {row_idx}."
                );
            }
        }

        let circuit_builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in ExtCascadeTable::terminal_constraints(&circuit_builder)
            .into_iter()
            .map(|constraint_monad| constraint_monad.consume())
            .enumerate()
        {
            let evaluated_constraint = constraint.evaluate(
                master_base_trace_table.slice(s![-1.., ..]),
                master_ext_trace_table.slice(s![-1.., ..]),
                challenges,
            );
            assert_eq!(
                zero, evaluated_constraint,
                "Terminal constraint {constraint_idx} failed."
            );
        }

        true
    }
}
