use air::challenge_id::ChallengeId;
use air::cross_table_argument::CrossTableArg;
use air::cross_table_argument::LookupArg;
use air::table::cascade::CascadeTable;
use air::table_column::MasterAuxColumn;
use air::table_column::MasterMainColumn;
use ndarray::Array1;
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
use crate::table::running_arguments::par_fractions;
use crate::table::running_arguments::par_log_derivative;
use crate::table::running_arguments::par_running_sum;

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

        let rows = || {
            (0..main_table.nrows())
                .into_par_iter()
                .map(|row_idx| main_table.row(row_idx))
        };
        let is_padding =
            |row: ArrayView1<BFieldElement>| row[MainColumn::IsPadding.main_index()].is_one();

        // one fraction per (non-padding) row; inversions are batched
        let hash_table_fractions = rows()
            .map(|row| {
                (!is_padding(row)).then(|| {
                    let look_in = two_pow_8 * row[MainColumn::LookInHi.main_index()]
                        + row[MainColumn::LookInLo.main_index()];
                    let look_out = two_pow_8 * row[MainColumn::LookOutHi.main_index()]
                        + row[MainColumn::LookOutLo.main_index()];
                    let compressed_row =
                        hash_input_weight * look_in + hash_output_weight * look_out;
                    let lookup_multiplicity = row[MainColumn::LookupMultiplicity.main_index()];
                    (
                        hash_indeterminate - compressed_row,
                        lookup_multiplicity.lift(),
                    )
                })
            })
            .collect::<Vec<_>>();
        let hash_table_log_derivative =
            par_log_derivative(&hash_table_fractions, LookupArg::default_initial());

        // two fractions per (non-padding) row
        let lookup_table_fraction = |row: ArrayView1<BFieldElement>, lo_or_hi: [MainColumn; 2]| {
            (!is_padding(row)).then(|| {
                let [in_col, out_col] = lo_or_hi;
                let compressed_row = lookup_input_weight * row[in_col.main_index()]
                    + lookup_output_weight * row[out_col.main_index()];
                (lookup_indeterminate - compressed_row, XFieldElement::ONE)
            })
        };
        let lookup_table_fractions_lo = rows()
            .map(|row| lookup_table_fraction(row, [MainColumn::LookInLo, MainColumn::LookOutLo]))
            .collect::<Vec<_>>();
        let lookup_table_fractions_hi = rows()
            .map(|row| lookup_table_fraction(row, [MainColumn::LookInHi, MainColumn::LookOutHi]))
            .collect::<Vec<_>>();
        let lookup_table_summands = par_fractions(&lookup_table_fractions_lo)
            .into_par_iter()
            .zip(par_fractions(&lookup_table_fractions_hi))
            .map(|(lo, hi)| lo + hi)
            .collect();
        let lookup_table_log_derivative =
            par_running_sum(lookup_table_summands, LookupArg::default_initial());

        Array1::from(hash_table_log_derivative)
            .move_into(aux_table.column_mut(AuxColumn::HashTableServerLogDerivative.aux_index()));
        Array1::from(lookup_table_log_derivative)
            .move_into(aux_table.column_mut(AuxColumn::LookupTableClientLogDerivative.aux_index()));
        profiler!(stop "cascade table");
    }
}
