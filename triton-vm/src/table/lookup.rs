use air::challenge_id::ChallengeId;
use air::cross_table_argument::CrossTableArg;
use air::cross_table_argument::EvalArg;
use air::cross_table_argument::LookupArg;
use air::table::lookup::LookupTable;
use air::table_column::MasterAuxColumn;
use air::table_column::MasterMainColumn;
use itertools::Itertools;
use ndarray::prelude::*;
use num_traits::ConstOne;
use num_traits::One;
use rayon::prelude::*;
use strum::EnumCount;
use strum::IntoEnumIterator;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::challenges::Challenges;
use crate::ndarray_helper::contiguous_column_slices;
use crate::ndarray_helper::horizontal_multi_slice_mut;
use crate::profiler::profiler;
use crate::table::TraceTable;

type MainColumn = <LookupTable as air::AIR>::MainColumn;
type AuxColumn = <LookupTable as air::AIR>::AuxColumn;

fn auxiliary_column_cascade_running_sum_log_derivative(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    let look_in_weight = challenges[ChallengeId::LookupTableInputWeight];
    let look_out_weight = challenges[ChallengeId::LookupTableOutputWeight];
    let indeterminate = challenges[ChallengeId::CascadeLookupIndeterminate];

    let mut cascade_table_running_sum_log_derivative = LookupArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    for row in main_table.rows() {
        if row[MainColumn::IsPadding.main_index()].is_one() {
            break;
        }

        let lookup_input = row[MainColumn::LookIn.main_index()];
        let lookup_output = row[MainColumn::LookOut.main_index()];
        let compressed_row = lookup_input * look_in_weight + lookup_output * look_out_weight;

        let lookup_multiplicity = row[MainColumn::LookupMultiplicity.main_index()];
        cascade_table_running_sum_log_derivative +=
            (indeterminate - compressed_row).inverse() * lookup_multiplicity;

        auxiliary_column.push(cascade_table_running_sum_log_derivative);
    }

    // fill padding section
    auxiliary_column.resize(main_table.nrows(), cascade_table_running_sum_log_derivative);
    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

fn auxiliary_column_public_running_evaluation(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    let mut running_evaluation = EvalArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    for row in main_table.rows() {
        if row[MainColumn::IsPadding.main_index()].is_one() {
            break;
        }

        running_evaluation = running_evaluation
            * challenges[ChallengeId::LookupTablePublicIndeterminate]
            + row[MainColumn::LookOut.main_index()];
        auxiliary_column.push(running_evaluation);
    }

    // fill padding section
    auxiliary_column.resize(main_table.nrows(), running_evaluation);
    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

impl TraceTable for LookupTable {
    type FillParam = ();
    type FillReturnInfo = ();

    fn fill(mut main_table: ArrayViewMut2<BFieldElement>, aet: &AlgebraicExecutionTrace, _: ()) {
        const LOOKUP_TABLE_LEN: usize = tip5::LOOKUP_TABLE.len();
        assert!(main_table.nrows() >= LOOKUP_TABLE_LEN);

        // Lookup Table input
        let lookup_input = Array1::from_iter((0..LOOKUP_TABLE_LEN).map(|i| bfe!(i as u64)));
        let lookup_input_column =
            main_table.slice_mut(s![..LOOKUP_TABLE_LEN, MainColumn::LookIn.main_index()]);
        lookup_input.move_into(lookup_input_column);

        // Lookup Table output
        let lookup_output = Array1::from_iter(tip5::LOOKUP_TABLE.map(BFieldElement::from));
        let lookup_output_column =
            main_table.slice_mut(s![..LOOKUP_TABLE_LEN, MainColumn::LookOut.main_index()]);
        lookup_output.move_into(lookup_output_column);

        // Lookup Table multiplicities
        let lookup_multiplicities = Array1::from_iter(
            aet.lookup_table_lookup_multiplicities
                .map(BFieldElement::new),
        );
        let lookup_multiplicities_column = main_table.slice_mut(s![
            ..LOOKUP_TABLE_LEN,
            MainColumn::LookupMultiplicity.main_index()
        ]);
        lookup_multiplicities.move_into(lookup_multiplicities_column);
    }

    fn pad(mut lookup_table: ArrayViewMut2<BFieldElement>, table_length: usize) {
        lookup_table
            .slice_mut(s![table_length.., MainColumn::IsPadding.main_index()])
            .fill(BFieldElement::ONE);
    }

    fn extend(
        main_table: ArrayView2<BFieldElement>,
        mut aux_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        profiler!(start "lookup table");
        assert_eq!(MainColumn::COUNT, main_table.ncols());
        assert_eq!(AuxColumn::COUNT, aux_table.ncols());
        assert_eq!(main_table.nrows(), aux_table.nrows());

        let auxiliary_column_indices = AuxColumn::iter()
            .map(|column| column.aux_index())
            .collect_vec();
        let auxiliary_column_slices = horizontal_multi_slice_mut(
            aux_table.view_mut(),
            &contiguous_column_slices(&auxiliary_column_indices),
        );
        let extension_functions = [
            auxiliary_column_cascade_running_sum_log_derivative,
            auxiliary_column_public_running_evaluation,
        ];

        extension_functions
            .into_par_iter()
            .zip_eq(auxiliary_column_slices)
            .for_each(|(generator, slice)| {
                generator(main_table, challenges).move_into(slice);
            });

        profiler!(stop "lookup table");
    }
}
