use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::challenges::AllChallenges;
use crate::table::extension_table::Evaluable;
use crate::table::extension_table::Quotientable;
use crate::table::op_stack_table::ExtOpStackTable;

// This file is a placeholder for auto-generated code
// Run `cargo run --bin constraint-evaluation-generator`
// to fill in this file with optimized constraints.
impl Evaluable for ExtOpStackTable {
    fn evaluate_initial_constraints(
        &self,
        _evaluation_point: &[XFieldElement],
        _challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {
        panic!(
            "Constraints must be in place. Run `cargo run --bin constraint-evaluation-generator`."
        )
    }

    fn evaluate_consistency_constraints(
        &self,
        _evaluation_point: &[XFieldElement],
        _challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {
        panic!(
            "Constraints must be in place. Run `cargo run --bin constraint-evaluation-generator`."
        )
    }

    fn evaluate_transition_constraints(
        &self,
        _current_row: &[XFieldElement],
        _next_row: &[XFieldElement],
        _challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {
        panic!(
            "Constraints must be in place. Run `cargo run --bin constraint-evaluation-generator`."
        )
    }

    fn evaluate_terminal_constraints(
        &self,
        _evaluation_point: &[XFieldElement],
        _challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {
        panic!(
            "Constraints must be in place. Run `cargo run --bin constraint-evaluation-generator`."
        )
    }
}

impl Quotientable for ExtOpStackTable {
    fn get_initial_quotient_degree_bounds(
        &self,
        _padded_height: usize,
        _num_trace_randomizers: usize,
    ) -> Vec<Degree> {
        panic!(
            "Degree bounds must be in place. Run `cargo run --bin constraint-evaluation-generator`."
        )
    }

    fn get_consistency_quotient_degree_bounds(
        &self,
        _padded_height: usize,
        _num_trace_randomizers: usize,
    ) -> Vec<Degree> {
        panic!(
            "Degree bounds must be in place. Run `cargo run --bin constraint-evaluation-generator`."
        )
    }

    fn get_transition_quotient_degree_bounds(
        &self,
        _padded_height: usize,
        _num_trace_randomizers: usize,
    ) -> Vec<Degree> {
        panic!(
            "Degree bounds must be in place. Run `cargo run --bin constraint-evaluation-generator`."
        )
    }

    fn get_terminal_quotient_degree_bounds(
        &self,
        _padded_height: usize,
        _num_trace_randomizers: usize,
    ) -> Vec<Degree> {
        panic!(
            "Degree bounds must be in place. Run `cargo run --bin constraint-evaluation-generator`."
        )
    }
}
