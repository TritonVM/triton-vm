use std::collections::HashSet;

use itertools::Itertools;
use triton_vm::table::base_table::InheritsFromTable;
use triton_vm::table::challenges::TableChallenges;
use triton_vm::table::constraint_circuit::{CircuitExpression, CircuitId, ConstraintCircuit};
use triton_vm::table::program_table::ExtProgramTable;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;

fn main() {
    println!("Generate those constraint evaluators!");

    let table = ExtProgramTable::default();
    let mut constraint_circuits = ExtProgramTable::ext_transition_constraints_as_circuits();
    gen(
        &table,
        "program_table",
        "ExtProgramTable",
        &mut constraint_circuits,
    );
}

fn gen<Table: InheritsFromTable<XFieldElement>, T: TableChallenges>(
    table: &Table,
    table_name_snake: &str,
    table_name_camel: &str,
    constraint_circuits: &mut [ConstraintCircuit<T>],
) {
    // Assert that all node IDs are unique (sanity check)
    ConstraintCircuit::assert_has_unique_ids(constraint_circuits);

    // Delete redundant nodes
    ConstraintCircuit::constant_folding(&mut constraint_circuits.iter_mut().collect_vec());

    // Count number of times each node is visited
    ConstraintCircuit::traverse_multiple(constraint_circuits);

    // Get the max count that each node is visited
    let mut max_visited = 0;
    for constraint in constraint_circuits.iter() {
        max_visited = std::cmp::max(max_visited, constraint.get_max_visited_counter());
    }

    let mut requested_visited = max_visited;

    // Declare shared values
    // In the main function we predeclare all variables with a visit count of more than 1
    let mut shared_evaluations: Vec<String> = vec![];
    while requested_visited > 1 {
        shared_evaluations.push(evaluate_nodes_with_visit_count(
            requested_visited,
            constraint_circuits,
        ));
        requested_visited -= 1;
    }

    let filename = format!("triton-vm/src/table/{table_name_snake}_autogen.rs");
    let code = "todo!()";

    let template = format!(
        "
use twenty_first::shared_math::x_field_element::XFieldElement;

use super::challenges::AllChallenges;
use super::extension_table::Evaluable;
use super::{table_name_snake}::{table_name_camel};

impl Evaluable for {table_name_camel} {{
    fn evaluate_transition_constraints(
        &self,
        current_row: &[XFieldElement],
        next_row: &[XFieldElement],
        challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {{
        {code}
    }}
}}
"
    );

    if std::env::var("OUTPUT_RUST_SOURCE_CODE").is_ok() {
        use std::fs;
        fs::write(filename, template).expect("Unable to write file");
    }
}

/// Produce the code to evaluate code for all nodes that share a value number of
/// times visited. A value for all nodes with a higher count than the provided are assumed
/// to be in scope.
fn evaluate_nodes_with_visit_count<T: TableChallenges>(
    visited_count: usize,
    circuits: &[ConstraintCircuit<T>],
) -> String {
    let mut in_scope: HashSet<CircuitId> = HashSet::new();
    let mut output = String::default();

    for circuit in circuits.iter() {
        declare_single_node_with_visit_count(visited_count, circuit, &mut in_scope, &mut output);
    }

    output
}

fn declare_single_node_with_visit_count<T: TableChallenges>(
    requested_visited_count: usize,
    circuit: &ConstraintCircuit<T>,
    in_scope: &mut HashSet<CircuitId>,
    output: &mut String,
) {
    println!("requested_visited_count = {requested_visited_count}");
    if circuit.visited_counter < requested_visited_count {
        // If the visited counter is not there yet, make a recursive call. We are
        // not yet ready to bind this node's ID to a value.
        if let CircuitExpression::BinaryOperation(_binop, lhs, rhs) = &circuit.expression {
            declare_single_node_with_visit_count(
                requested_visited_count,
                &lhs.as_ref().borrow(),
                in_scope,
                output,
            );
            declare_single_node_with_visit_count(
                requested_visited_count,
                &rhs.as_ref().borrow(),
                in_scope,
                output,
            );
        }
        return;
    }

    // If this node has already been declared, or visit counter is higher than requested,
    // than the node value *must* already be in scope. We should not redeclare it.
    // We also do not declare nodes that are e.g `point[3]` since they are already in scope
    // through the `points` input argument, and we do not declare constants.
    if circuit.visited_counter > requested_visited_count
        || in_scope.contains(&circuit.id)
        || matches!(circuit.expression, CircuitExpression::BConstant(_))
        || matches!(circuit.expression, CircuitExpression::XConstant(_))
        || circuit.get_linear_one_index().is_some()
    {
        return;
    }

    // If this line is met, it means that the visit count is as requested, and that
    // the value is not in scope. So it must be added to the scope. We find the
    // expression for the value, and then put it into scope through a let expression
    if circuit.visited_counter == requested_visited_count && !in_scope.contains(&circuit.id) {
        let binding_name = get_binding_name(circuit);
        output.push_str(&format!("let {binding_name} =\n"));
        evaluate_single_node(requested_visited_count, circuit, in_scope, output);
        output.push_str("\n");

        let new_insertion = in_scope.insert(circuit.id.clone());
        assert!(new_insertion);
    }
}

/// Return a variable name for the node. Returns `point[n]` if node is just
/// a value from the codewords. Otherwise returns the ID of the circuit.
fn get_binding_name<T: TableChallenges>(circuit: &ConstraintCircuit<T>) -> String {
    match &circuit.expression {
        CircuitExpression::XConstant(xfe) => print_xfe(*xfe),
        CircuitExpression::BConstant(bfe) => print_bfe(*bfe),
        CircuitExpression::Input(idx) => {
            if *idx >= circuit.var_count {
                let idx = idx - circuit.var_count;
                format!("next_row[{idx}]")
            } else {
                format!("current_row[{idx}]")
            }
        }
        CircuitExpression::Challenge(challenge_id) => {
            let challenge_index: usize = (*challenge_id).into();
            format!("challenges[{challenge_index}]")
        }
        CircuitExpression::BinaryOperation(_, _, _) => format!("node_{}", circuit.id),
    }
}

fn print_bfe(bfe: BFieldElement) -> String {
    format!("BFieldElement::new({})", bfe.value())
}

fn print_xfe(xfe: XFieldElement) -> String {
    format!(
        "XFieldElement::new([{}, {}, {}])",
        xfe.coefficients[0], xfe.coefficients[1], xfe.coefficients[2]
    )
}
