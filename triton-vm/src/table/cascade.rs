use air::challenge_id::ChallengeId;
use air::cross_table_argument::CrossTableArg;
use air::cross_table_argument::LookupArg;
use air::table::cascade::CascadeTable;
use air::table_column::MasterAuxColumn;
use air::table_column::MasterMainColumn;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use ndarray::s;
use num_traits::ConstOne;
use num_traits::One;
use rayon::prelude::*;
use strum::EnumCount;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::challenges::Challenges;
use crate::ndarray_helper::ROW_AXIS;
use crate::profiler::profiler;
use crate::table::TraceTable;
use crate::table::running_arguments::par_log_derivative_into;
use crate::table::running_arguments::with_contiguous_column;

type MainColumn = <CascadeTable as air::AIR>::MainColumn;
type AuxColumn = <CascadeTable as air::AIR>::AuxColumn;

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
        let rows_and_multiplicities = main_table
            .axis_iter_mut(ROW_AXIS)
            .into_par_iter()
            .zip(&aet.cascade_table_lookup_multiplicities);

        rows_and_multiplicities.for_each(|(mut row, (&to_look_up, &multiplicity))| {
            let to_look_up_lo = (to_look_up & 0xff) as u8;
            let to_look_up_hi = ((to_look_up >> 8) & 0xff) as u8;

            row[MainColumn::LookInLo.main_index()] = bfe!(to_look_up_lo);
            row[MainColumn::LookInHi.main_index()] = bfe!(to_look_up_hi);
            row[MainColumn::LookOutLo.main_index()] = lookup_8_bit_limb(to_look_up_lo);
            row[MainColumn::LookOutHi.main_index()] = lookup_8_bit_limb(to_look_up_hi);
            row[MainColumn::LookupMultiplicity.main_index()] = bfe!(multiplicity);
        });
    }

    fn pad(mut main_table: ArrayViewMut2<BFieldElement>, cascade_table_length: usize) {
        main_table
            .slice_mut(s![
                cascade_table_length..,
                MainColumn::IsPadding.main_index()
            ])
            .fill(BFieldElement::ONE);
    }

    fn extend(
        main_table: ArrayView2<BFieldElement>,
        mut aux_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        profiler!(start "cascade table");
        assert_eq!(MainColumn::COUNT, main_table.ncols());
        assert_eq!(AuxColumn::COUNT, aux_table.ncols());
        assert_eq!(main_table.nrows(), aux_table.nrows());

        let two_pow_8 = bfe!(1 << 8);

        let hash_indeterminate = challenges[ChallengeId::HashCascadeLookupIndeterminate];
        let hash_input_weight = challenges[ChallengeId::HashCascadeLookInWeight];
        let hash_output_weight = challenges[ChallengeId::HashCascadeLookOutWeight];

        let lookup_indeterminate = challenges[ChallengeId::CascadeLookupIndeterminate];
        let lookup_input_weight = challenges[ChallengeId::LookupTableInputWeight];
        let lookup_output_weight = challenges[ChallengeId::LookupTableOutputWeight];

        let is_padding =
            |row: ArrayView1<BFieldElement>| row[MainColumn::IsPadding.main_index()].is_one();

        // one fraction per (non-padding) row
        let hash_table_fraction = |row_idx: usize| {
            let row = main_table.row(row_idx);
            let fraction = (!is_padding(row)).then(|| {
                let look_in = two_pow_8 * row[MainColumn::LookInHi.main_index()]
                    + row[MainColumn::LookInLo.main_index()];
                let look_out = two_pow_8 * row[MainColumn::LookOutHi.main_index()]
                    + row[MainColumn::LookOutLo.main_index()];
                let compressed_row = hash_input_weight * look_in + hash_output_weight * look_out;
                let lookup_multiplicity = row[MainColumn::LookupMultiplicity.main_index()];
                (
                    hash_indeterminate - compressed_row,
                    lookup_multiplicity.lift(),
                )
            });
            [fraction]
        };

        // two fractions per (non-padding) row
        let lookup_table_fractions = |row_idx: usize| {
            let row = main_table.row(row_idx);
            let fraction = |[in_col, out_col]: [MainColumn; 2]| {
                (!is_padding(row)).then(|| {
                    let compressed_row = lookup_input_weight * row[in_col.main_index()]
                        + lookup_output_weight * row[out_col.main_index()];
                    (lookup_indeterminate - compressed_row, XFieldElement::ONE)
                })
            };
            [
                fraction([MainColumn::LookInLo, MainColumn::LookOutLo]),
                fraction([MainColumn::LookInHi, MainColumn::LookOutHi]),
            ]
        };

        let initial = LookupArg::default_initial();
        let hash_table_log_derivative =
            aux_table.column_mut(AuxColumn::HashTableServerLogDerivative.aux_index());
        with_contiguous_column(hash_table_log_derivative, |column| {
            par_log_derivative_into(column, initial, hash_table_fraction)
        });
        let lookup_table_log_derivative =
            aux_table.column_mut(AuxColumn::LookupTableClientLogDerivative.aux_index());
        with_contiguous_column(lookup_table_log_derivative, |column| {
            par_log_derivative_into(column, initial, lookup_table_fractions)
        });
        profiler!(stop "cascade table");
    }
}
