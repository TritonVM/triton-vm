use ndarray::s;
use ndarray::Array1;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use num_traits::Zero;
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::b_field_element::BFIELD_ONE;
use twenty_first::shared_math::tip5;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::challenges::ChallengeId::CascadeLookupIndeterminate;
use crate::table::challenges::ChallengeId::LookupTableInputWeight;
use crate::table::challenges::ChallengeId::LookupTableOutputWeight;
use crate::table::challenges::ChallengeId::LookupTablePublicIndeterminate;
use crate::table::challenges::Challenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::EvalArg;
use crate::table::cross_table_argument::LookupArg;
use crate::table::table_column::LookupBaseTableColumn;
use crate::table::table_column::LookupBaseTableColumn::*;
use crate::table::table_column::LookupExtTableColumn;
use crate::table::table_column::LookupExtTableColumn::*;
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

        let look_in_weight = challenges.get_challenge(LookupTableInputWeight);
        let look_out_weight = challenges.get_challenge(LookupTableOutputWeight);
        let cascade_indeterminate = challenges.get_challenge(CascadeLookupIndeterminate);
        let public_indeterminate = challenges.get_challenge(LookupTablePublicIndeterminate);

        let mut cascade_table_running_sum_log_derivative = LookupArg::default_initial();
        let mut public_running_evaluation = EvalArg::default_initial();

        for row_idx in 0..base_table.nrows() {
            let base_row = base_table.row(row_idx);

            let lookup_input = base_row[LookIn.base_table_index()];
            let lookup_output = base_row[LookOut.base_table_index()];
            let lookup_multiplicity = base_row[LookupMultiplicity.base_table_index()];
            let is_padding = base_row[IsPadding.base_table_index()];

            if is_padding.is_zero() {
                let compressed_row =
                    lookup_input * look_in_weight + lookup_output * look_out_weight;
                cascade_table_running_sum_log_derivative +=
                    (cascade_indeterminate - compressed_row).inverse() * lookup_multiplicity;

                public_running_evaluation =
                    public_running_evaluation * public_indeterminate + lookup_output;
            }

            let mut ext_row = ext_table.row_mut(row_idx);
            ext_row[CascadeTableServerLogDerivative.ext_table_index()] =
                cascade_table_running_sum_log_derivative;
            ext_row[PublicEvaluationArgument.ext_table_index()] = public_running_evaluation;
        }
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
