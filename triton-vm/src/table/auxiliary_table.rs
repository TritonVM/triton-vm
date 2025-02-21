use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;

use itertools::Itertools;
use ndarray::ArrayView1;
use twenty_first::math::traits::FiniteField;
use twenty_first::prelude::*;

use crate::challenges::Challenges;
use crate::table::ConstraintType;
use crate::table::master_table::MasterAuxTable;

include!(concat!(env!("OUT_DIR"), "/evaluate_constraints.rs"));

// The implementations of these functions are generated in `build.rs`.
pub trait Evaluable<FF: FiniteField> {
    fn evaluate_initial_constraints(
        main_row: ArrayView1<FF>,
        aux_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement>;

    fn evaluate_consistency_constraints(
        main_row: ArrayView1<FF>,
        aux_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement>;

    fn evaluate_transition_constraints(
        current_main_row: ArrayView1<FF>,
        current_aux_row: ArrayView1<XFieldElement>,
        next_main_row: ArrayView1<FF>,
        next_aux_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement>;

    fn evaluate_terminal_constraints(
        main_row: ArrayView1<FF>,
        aux_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement>;
}

/// Helps debugging and benchmarking. The maximal degree achieved in any table
/// dictates the length of the FRI domain, which in turn is responsible for the
/// main performance bottleneck.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct DegreeWithOrigin {
    pub degree: isize,
    pub interpolant_degree: isize,
    pub zerofier_degree: isize,
    pub origin_index: usize,
    pub origin_table_height: usize,

    /// Can be used to determine the degree bounds for the quotient polynomials:
    /// the degree of the zerofier polynomials differ between the constraint
    /// types.
    pub origin_constraint_type: ConstraintType,
}

impl Display for DegreeWithOrigin {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        assert!(self.degree > 0, "constant constraints make no sense");
        let zerofier_corrected_degree = self.degree + self.zerofier_degree;
        let degree = zerofier_corrected_degree / self.interpolant_degree;
        let idx = self.origin_index;
        let constraint_type = self.origin_constraint_type;
        write!(
            f,
            "Degree of polynomial {idx:02} of type “{constraint_type}” is {degree}."
        )
    }
}

/// Compute the degrees of the quotients from all AIR constraints that apply to
/// the table.
pub(crate) fn all_degrees_with_origin(
    interpolant_degree: isize,
    padded_height: usize,
) -> Vec<DegreeWithOrigin> {
    let initial_degrees_with_origin =
        MasterAuxTable::initial_quotient_degree_bounds(interpolant_degree)
            .into_iter()
            .enumerate()
            .map(|(origin_index, degree)| DegreeWithOrigin {
                degree,
                interpolant_degree,
                zerofier_degree: 1,
                origin_index,
                origin_table_height: padded_height,
                origin_constraint_type: ConstraintType::Initial,
            })
            .collect_vec();

    let consistency_degrees_with_origin =
        MasterAuxTable::consistency_quotient_degree_bounds(interpolant_degree, padded_height)
            .into_iter()
            .enumerate()
            .map(|(origin_index, degree)| DegreeWithOrigin {
                degree,
                interpolant_degree,
                zerofier_degree: padded_height as isize,
                origin_index,
                origin_table_height: padded_height,
                origin_constraint_type: ConstraintType::Consistency,
            })
            .collect();

    let transition_degrees_with_origin =
        MasterAuxTable::transition_quotient_degree_bounds(interpolant_degree, padded_height)
            .into_iter()
            .enumerate()
            .map(|(origin_index, degree)| DegreeWithOrigin {
                degree,
                interpolant_degree,
                zerofier_degree: padded_height as isize - 1,
                origin_index,
                origin_table_height: padded_height,
                origin_constraint_type: ConstraintType::Transition,
            })
            .collect();

    let terminal_degrees_with_origin =
        MasterAuxTable::terminal_quotient_degree_bounds(interpolant_degree)
            .into_iter()
            .enumerate()
            .map(|(origin_index, degree)| DegreeWithOrigin {
                degree,
                interpolant_degree,
                zerofier_degree: 1,
                origin_index,
                origin_table_height: padded_height,
                origin_constraint_type: ConstraintType::Terminal,
            })
            .collect();

    [
        initial_degrees_with_origin,
        consistency_degrees_with_origin,
        transition_degrees_with_origin,
        terminal_degrees_with_origin,
    ]
    .concat()
}
