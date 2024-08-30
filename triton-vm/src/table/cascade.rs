use air::challenge_id::ChallengeId;
use air::challenge_id::ChallengeId::*;
use air::cross_table_argument::CrossTableArg;
use air::cross_table_argument::LookupArg;
use air::table::cascade::CascadeTable;
use air::table_column::CascadeBaseTableColumn;
use air::table_column::CascadeBaseTableColumn::*;
use air::table_column::CascadeExtTableColumn;
use air::table_column::CascadeExtTableColumn::*;
use air::table_column::MasterBaseTableColumn;
use air::table_column::MasterExtTableColumn;
use air::AIR;
use constraint_circuit::ConstraintCircuitBuilder;
use constraint_circuit::ConstraintCircuitMonad;
use constraint_circuit::DualRowIndicator;
use constraint_circuit::DualRowIndicator::*;
use constraint_circuit::SingleRowIndicator;
use constraint_circuit::SingleRowIndicator::*;
use ndarray::s;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use num_traits::ConstOne;
use num_traits::One;
use strum::EnumCount;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::challenges::Challenges;
use crate::profiler::profiler;
use crate::table::TraceTable;

fn lookup_8_bit_limb(to_look_up: u8) -> BFieldElement {
    tip5::LOOKUP_TABLE[usize::from(to_look_up)].into()
}

pub(crate) fn lookup_16_bit_limb(to_look_up: u16) -> BFieldElement {
    let to_look_up_lo = (to_look_up & 0xff) as u8;
    let to_look_up_hi = ((to_look_up >> 8) & 0xff) as u8;
    let looked_up_lo = lookup_8_bit_limb(to_look_up_lo);
    let looked_up_hi = lookup_8_bit_limb(to_look_up_hi);
    bfe!(1 << 8) * looked_up_hi + looked_up_lo
}

impl TraceTable for CascadeTable {
    type FillParam = ();
    type FillReturnInfo = ();

    fn fill(mut main_table: ArrayViewMut2<BFieldElement>, aet: &AlgebraicExecutionTrace, _: ()) {
        for (row_idx, (&to_look_up, &multiplicity)) in
            aet.cascade_table_lookup_multiplicities.iter().enumerate()
        {
            let to_look_up_lo = (to_look_up & 0xff) as u8;
            let to_look_up_hi = ((to_look_up >> 8) & 0xff) as u8;

            let mut row = main_table.row_mut(row_idx);
            row[LookInLo.base_table_index()] = bfe!(to_look_up_lo);
            row[LookInHi.base_table_index()] = bfe!(to_look_up_hi);
            row[LookOutLo.base_table_index()] = lookup_8_bit_limb(to_look_up_lo);
            row[LookOutHi.base_table_index()] = lookup_8_bit_limb(to_look_up_hi);
            row[LookupMultiplicity.base_table_index()] = bfe!(multiplicity);
        }
    }

    fn pad(mut main_table: ArrayViewMut2<BFieldElement>, cascade_table_length: usize) {
        main_table
            .slice_mut(s![cascade_table_length.., IsPadding.base_table_index()])
            .fill(BFieldElement::ONE);
    }

    fn extend(
        main_table: ArrayView2<BFieldElement>,
        mut aux_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        profiler!(start "cascade table");
        assert_eq!(Self::MainColumn::COUNT, main_table.ncols());
        assert_eq!(Self::AuxColumn::COUNT, aux_table.ncols());
        assert_eq!(main_table.nrows(), aux_table.nrows());

        let mut hash_table_log_derivative = LookupArg::default_initial();
        let mut lookup_table_log_derivative = LookupArg::default_initial();

        let two_pow_8 = bfe!(1 << 8);

        let hash_indeterminate = challenges[HashCascadeLookupIndeterminate];
        let hash_input_weight = challenges[HashCascadeLookInWeight];
        let hash_output_weight = challenges[HashCascadeLookOutWeight];

        let lookup_indeterminate = challenges[CascadeLookupIndeterminate];
        let lookup_input_weight = challenges[LookupTableInputWeight];
        let lookup_output_weight = challenges[LookupTableOutputWeight];

        for row_idx in 0..main_table.nrows() {
            let base_row = main_table.row(row_idx);
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

            let mut extension_row = aux_table.row_mut(row_idx);
            extension_row[HashTableServerLogDerivative.ext_table_index()] =
                hash_table_log_derivative;
            extension_row[LookupTableClientLogDerivative.ext_table_index()] =
                lookup_table_log_derivative;
        }
        profiler!(stop "cascade table");
    }
}
