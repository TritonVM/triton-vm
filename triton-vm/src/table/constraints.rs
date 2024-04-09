//! This file is a placeholder for auto-generated code
//! Run `cargo run --bin constraint-evaluation-generator`
//! to fill in this file with optimized constraints.

use crate::table::challenges::Challenges;
use ndarray::ArrayView1;
use twenty_first::prelude::BFieldElement;
use twenty_first::prelude::XFieldElement;

use crate::table::extension_table::Evaluable;
use crate::table::extension_table::Quotientable;
use crate::table::master_table::MasterExtTable;

pub(crate) const ERROR_MESSAGE_GENERATE_CONSTRAINTS: &str =
    "Constraints must be in place. Run: `cargo run --bin constraint-evaluation-generator`";
const ERROR_MESSAGE_GENERATE_DEGREE_BOUNDS: &str =
    "Degree bounds must be in place. Run: `cargo run --bin constraint-evaluation-generator`";

impl Evaluable<BFieldElement> for MasterExtTable {
    fn evaluate_initial_constraints(
        _: ArrayView1<BFieldElement>,
        _: ArrayView1<XFieldElement>,
        _: &Challenges,
    ) -> Vec<XFieldElement> {
        panic!("{ERROR_MESSAGE_GENERATE_CONSTRAINTS}")
    }

    fn evaluate_consistency_constraints(
        _: ArrayView1<BFieldElement>,
        _: ArrayView1<XFieldElement>,
        _: &Challenges,
    ) -> Vec<XFieldElement> {
        panic!("{ERROR_MESSAGE_GENERATE_CONSTRAINTS}")
    }

    fn evaluate_transition_constraints(
        _: ArrayView1<BFieldElement>,
        _: ArrayView1<XFieldElement>,
        _: ArrayView1<BFieldElement>,
        _: ArrayView1<XFieldElement>,
        _: &Challenges,
    ) -> Vec<XFieldElement> {
        panic!("{ERROR_MESSAGE_GENERATE_CONSTRAINTS}")
    }

    fn evaluate_terminal_constraints(
        _: ArrayView1<BFieldElement>,
        _: ArrayView1<XFieldElement>,
        _: &Challenges,
    ) -> Vec<XFieldElement> {
        panic!("{ERROR_MESSAGE_GENERATE_CONSTRAINTS}")
    }
}

impl Evaluable<XFieldElement> for MasterExtTable {
    fn evaluate_initial_constraints(
        _: ArrayView1<XFieldElement>,
        _: ArrayView1<XFieldElement>,
        _: &Challenges,
    ) -> Vec<XFieldElement> {
        panic!("{ERROR_MESSAGE_GENERATE_CONSTRAINTS}")
    }

    fn evaluate_consistency_constraints(
        _: ArrayView1<XFieldElement>,
        _: ArrayView1<XFieldElement>,
        _: &Challenges,
    ) -> Vec<XFieldElement> {
        panic!("{ERROR_MESSAGE_GENERATE_CONSTRAINTS}")
    }

    fn evaluate_transition_constraints(
        _: ArrayView1<XFieldElement>,
        _: ArrayView1<XFieldElement>,
        _: ArrayView1<XFieldElement>,
        _: ArrayView1<XFieldElement>,
        _: &Challenges,
    ) -> Vec<XFieldElement> {
        panic!("{ERROR_MESSAGE_GENERATE_CONSTRAINTS}")
    }

    fn evaluate_terminal_constraints(
        _: ArrayView1<XFieldElement>,
        _: ArrayView1<XFieldElement>,
        _: &Challenges,
    ) -> Vec<XFieldElement> {
        panic!("{ERROR_MESSAGE_GENERATE_CONSTRAINTS}")
    }
}

impl Quotientable for MasterExtTable {
    const NUM_INITIAL_CONSTRAINTS: usize = 0;
    const NUM_CONSISTENCY_CONSTRAINTS: usize = 0;
    const NUM_TRANSITION_CONSTRAINTS: usize = 0;
    const NUM_TERMINAL_CONSTRAINTS: usize = 0;

    fn initial_quotient_degree_bounds(_: isize) -> Vec<isize> {
        panic!("{ERROR_MESSAGE_GENERATE_DEGREE_BOUNDS}")
    }

    fn consistency_quotient_degree_bounds(_: isize, _: usize) -> Vec<isize> {
        panic!("{ERROR_MESSAGE_GENERATE_DEGREE_BOUNDS}")
    }

    fn transition_quotient_degree_bounds(_: isize, _: usize) -> Vec<isize> {
        panic!("{ERROR_MESSAGE_GENERATE_DEGREE_BOUNDS}")
    }

    fn terminal_quotient_degree_bounds(_: isize) -> Vec<isize> {
        panic!("{ERROR_MESSAGE_GENERATE_DEGREE_BOUNDS}")
    }
}
