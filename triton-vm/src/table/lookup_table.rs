use ndarray::s;
use ndarray::Array1;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use num_traits::One;
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::b_field_element::BFIELD_ONE;
use twenty_first::shared_math::tip5;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::challenges::ChallengeId;
use crate::table::challenges::ChallengeId::CascadeLookupIndeterminate;
use crate::table::challenges::ChallengeId::LookupTableInputWeight;
use crate::table::challenges::ChallengeId::LookupTableOutputWeight;
use crate::table::challenges::ChallengeId::LookupTablePublicIndeterminate;
use crate::table::challenges::Challenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::ConstraintCircuitMonad;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::DualRowIndicator::*;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::*;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::EvalArg;
use crate::table::cross_table_argument::LookupArg;
use crate::table::table_column::LookupBaseTableColumn;
use crate::table::table_column::LookupBaseTableColumn::*;
use crate::table::table_column::LookupExtTableColumn;
use crate::table::table_column::LookupExtTableColumn::*;
use crate::table::table_column::MasterBaseTableColumn;
use crate::table::table_column::MasterExtTableColumn;
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

        // Lookup Table input
        let lookup_input = Array1::from_iter((0_u64..1 << 8).map(BFieldElement::new));
        let lookup_input_column =
            lookup_table.slice_mut(s![..1_usize << 8, LookIn.base_table_index()]);
        lookup_input.move_into(lookup_input_column);

        // Lookup Table output
        let lookup_output = Array1::from_iter(
            (0..1 << 8).map(|i| BFieldElement::new(tip5::LOOKUP_TABLE[i] as u64)),
        );
        let lookup_output_column =
            lookup_table.slice_mut(s![..1_usize << 8, LookOut.base_table_index()]);
        lookup_output.move_into(lookup_output_column);

        // Lookup Table multiplicities
        let lookup_multiplicities = Array1::from_iter(
            aet.lookup_table_lookup_multiplicities
                .map(BFieldElement::new),
        );
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
            let is_padding = base_row[IsPadding.base_table_index()].is_one();

            if !is_padding {
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
        let circuit_builder = ConstraintCircuitBuilder::new();

        let base_row = |col_id: LookupBaseTableColumn| {
            circuit_builder.input(BaseRow(col_id.master_base_table_index()))
        };
        let ext_row = |col_id: LookupExtTableColumn| {
            circuit_builder.input(ExtRow(col_id.master_ext_table_index()))
        };
        let challenge = |challenge_id: ChallengeId| circuit_builder.challenge(challenge_id);

        let lookup_input = base_row(LookIn);
        let lookup_output = base_row(LookOut);
        let lookup_multiplicity = base_row(LookupMultiplicity);
        let cascade_table_server_log_derivative = ext_row(CascadeTableServerLogDerivative);
        let public_evaluation_argument = ext_row(PublicEvaluationArgument);

        let lookup_input_is_0 = lookup_input;

        // Lookup Argument with Cascade Table
        // note: `lookup_input` is known to be 0 and thus doesn't appear in the compressed row
        let lookup_argument_default_initial =
            circuit_builder.x_constant(LookupArg::default_initial());
        let cascade_table_indeterminate = challenge(CascadeLookupIndeterminate);
        let compressed_row = lookup_output.clone() * challenge(LookupTableOutputWeight);
        let cascade_table_log_derivative_is_initialized_correctly =
            (cascade_table_server_log_derivative - lookup_argument_default_initial)
                * (cascade_table_indeterminate - compressed_row)
                - lookup_multiplicity;

        // public Evaluation Argument
        let eval_argument_default_initial = circuit_builder.x_constant(EvalArg::default_initial());
        let public_indeterminate = challenge(LookupTablePublicIndeterminate);
        let public_evaluation_argument_is_initialized_correctly = public_evaluation_argument
            - eval_argument_default_initial * public_indeterminate
            - lookup_output;

        let mut constraints = [
            lookup_input_is_0,
            cascade_table_log_derivative_is_initialized_correctly,
            public_evaluation_argument_is_initialized_correctly,
        ];
        ConstraintCircuitMonad::constant_folding(&mut constraints);
        constraints.map(|circuit| circuit.consume()).to_vec()
    }

    pub fn ext_consistency_constraints_as_circuits() -> Vec<ConstraintCircuit<SingleRowIndicator>> {
        vec![]
    }

    pub fn ext_transition_constraints_as_circuits() -> Vec<ConstraintCircuit<DualRowIndicator>> {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let one = circuit_builder.b_constant(BFIELD_ONE);

        let current_base_row = |col_id: LookupBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col_id.master_base_table_index()))
        };
        let next_base_row = |col_id: LookupBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col_id.master_base_table_index()))
        };
        let current_ext_row = |col_id: LookupExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col_id.master_ext_table_index()))
        };
        let next_ext_row = |col_id: LookupExtTableColumn| {
            circuit_builder.input(NextExtRow(col_id.master_ext_table_index()))
        };
        let challenge = |challenge_id: ChallengeId| circuit_builder.challenge(challenge_id);

        let lookup_input = current_base_row(LookIn);
        let is_padding = current_base_row(IsPadding);
        let cascade_table_server_log_derivative = current_ext_row(CascadeTableServerLogDerivative);
        let public_evaluation_argument = current_ext_row(PublicEvaluationArgument);

        let lookup_input_next = next_base_row(LookIn);
        let lookup_output_next = next_base_row(LookOut);
        let lookup_multiplicity_next = next_base_row(LookupMultiplicity);
        let is_padding_next = next_base_row(IsPadding);
        let cascade_table_server_log_derivative_next =
            next_ext_row(CascadeTableServerLogDerivative);
        let public_evaluation_argument_next = next_ext_row(PublicEvaluationArgument);

        // Padding section is contiguous: if the current row is a padding row, then the next row
        // is also a padding row.
        let if_current_row_is_padding_row_then_next_row_is_padding_row =
            is_padding * (one.clone() - is_padding_next.clone());

        // Lookup Table's input increments by 1 if and only if the next row is not a padding row
        let if_next_row_is_padding_row_then_lookup_input_next_is_0 =
            is_padding_next.clone() * lookup_input_next.clone();
        let if_next_row_is_not_padding_row_then_lookup_input_next_increments_by_1 = (one.clone()
            - is_padding_next.clone())
            * (lookup_input_next.clone() - lookup_input - one.clone());
        let lookup_input_increments_if_and_only_if_next_row_is_not_padding_row =
            if_next_row_is_padding_row_then_lookup_input_next_is_0
                + if_next_row_is_not_padding_row_then_lookup_input_next_increments_by_1;

        // Lookup Argument with Cascade Table
        let cascade_table_indeterminate = challenge(CascadeLookupIndeterminate);
        let compressed_row = lookup_input_next * challenge(LookupTableInputWeight)
            + lookup_output_next.clone() * challenge(LookupTableOutputWeight);
        let cascade_table_log_derivative_remains = cascade_table_server_log_derivative_next.clone()
            - cascade_table_server_log_derivative.clone();
        let cascade_table_log_derivative_updates = (cascade_table_server_log_derivative_next
            - cascade_table_server_log_derivative)
            * (cascade_table_indeterminate - compressed_row)
            - lookup_multiplicity_next;
        let cascade_table_log_derivative_updates_if_and_only_if_next_row_is_not_padding_row =
            (one.clone() - is_padding_next.clone()) * cascade_table_log_derivative_updates
                + is_padding_next.clone() * cascade_table_log_derivative_remains;

        // public Evaluation Argument
        let public_indeterminate = challenge(LookupTablePublicIndeterminate);
        let public_evaluation_argument_remains =
            public_evaluation_argument_next.clone() - public_evaluation_argument.clone();
        let public_evaluation_argument_updates = public_evaluation_argument_next
            - public_evaluation_argument * public_indeterminate
            - lookup_output_next;
        let public_evaluation_argument_updates_if_and_only_if_next_row_is_not_padding_row =
            (one - is_padding_next.clone()) * public_evaluation_argument_updates
                + is_padding_next * public_evaluation_argument_remains;

        let mut constraints = [
            if_current_row_is_padding_row_then_next_row_is_padding_row,
            lookup_input_increments_if_and_only_if_next_row_is_not_padding_row,
            cascade_table_log_derivative_updates_if_and_only_if_next_row_is_not_padding_row,
            public_evaluation_argument_updates_if_and_only_if_next_row_is_not_padding_row,
        ];
        ConstraintCircuitMonad::constant_folding(&mut constraints);
        constraints.map(|circuit| circuit.consume()).to_vec()
    }

    pub fn ext_terminal_constraints_as_circuits() -> Vec<ConstraintCircuit<SingleRowIndicator>> {
        vec![]
    }
}
