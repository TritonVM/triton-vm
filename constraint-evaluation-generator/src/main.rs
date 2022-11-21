use std::collections::HashSet;
use std::process::Command;

use heck::ToUpperCamelCase;
use itertools::Itertools;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;

use triton_vm::table::challenges::TableChallenges;
use triton_vm::table::constraint_circuit::{
    CircuitExpression, CircuitId, ConstraintCircuit, InputIndicator,
};
use triton_vm::table::hash_table::ExtHashTable;
use triton_vm::table::instruction_table::ExtInstructionTable;
use triton_vm::table::jump_stack_table::ExtJumpStackTable;
use triton_vm::table::op_stack_table::ExtOpStackTable;
use triton_vm::table::processor_table::ExtProcessorTable;
use triton_vm::table::program_table::ExtProgramTable;
use triton_vm::table::ram_table::ExtRamTable;

fn main() {
    println!("Generate those constraint evaluators!");

    let table_name_snake = "program_table";
    let source_code = gen(
        table_name_snake,
        &mut ExtProgramTable::ext_initial_constraints_as_circuits(),
        &mut ExtProgramTable::ext_consistency_constraints_as_circuits(),
        &mut ExtProgramTable::ext_transition_constraints_as_circuits(),
    );
    write(table_name_snake, source_code);

    let tabe_name_snake = "instruction_table";
    let source_code = gen(
        tabe_name_snake,
        &mut ExtInstructionTable::ext_initial_constraints_as_circuits(),
        &mut ExtInstructionTable::ext_consistency_constraints_as_circuits(),
        &mut ExtInstructionTable::ext_transition_constraints_as_circuits(),
    );
    write(tabe_name_snake, source_code);

    let table_name_snake = "processor_table";
    let source_code = gen(
        table_name_snake,
        &mut ExtProcessorTable::ext_initial_constraints_as_circuits(),
        &mut ExtProcessorTable::ext_consistency_constraints_as_circuits(),
        &mut ExtProcessorTable::ext_transition_constraints_as_circuits(),
    );
    write(table_name_snake, source_code);

    let table_name_snake = "op_stack_table";
    let source_code = gen(
        table_name_snake,
        &mut ExtOpStackTable::ext_initial_constraints_as_circuits(),
        &mut ExtOpStackTable::ext_consistency_constraints_as_circuits(),
        &mut ExtOpStackTable::ext_transition_constraints_as_circuits(),
    );
    write(table_name_snake, source_code);

    let table_name_snake = "ram_table";
    let source_code = gen(
        table_name_snake,
        &mut ExtRamTable::ext_initial_constraints_as_circuits(),
        &mut ExtRamTable::ext_consistency_constraints_as_circuits(),
        &mut ExtRamTable::ext_transition_constraints_as_circuits(),
    );
    write(table_name_snake, source_code);

    let table_name_snake = "jump_stack_table";
    let source_code = gen(
        table_name_snake,
        &mut ExtJumpStackTable::ext_initial_constraints_as_circuits(),
        &mut ExtJumpStackTable::ext_consistency_constraints_as_circuits(),
        &mut ExtJumpStackTable::ext_transition_constraints_as_circuits(),
    );
    write(table_name_snake, source_code);

    let table_name_snake = "hash_table";
    let source_code = gen(
        table_name_snake,
        &mut ExtHashTable::ext_initial_constraints_as_circuits(),
        &mut ExtHashTable::ext_consistency_constraints_as_circuits(),
        &mut ExtHashTable::ext_transition_constraints_as_circuits(),
    );
    write(table_name_snake, source_code);

    if let Err(fmt_failed) = Command::new("cargo").arg("fmt").output() {
        println!("cargo fmt failed: {}", fmt_failed);
    }
}

fn write(table_name_snake: &str, rust_source_code: String) {
    let output_filename =
        format!("triton-vm/src/table/constraints/{table_name_snake}_constraints.rs");

    std::fs::write(output_filename, rust_source_code).expect("Write Rust source code");
}

fn gen<T: TableChallenges, SII: InputIndicator, DII: InputIndicator>(
    table_name_snake: &str,
    initial_constraint_circuits: &mut [ConstraintCircuit<T, SII>],
    consistency_constraint_circuits: &mut [ConstraintCircuit<T, SII>],
    transition_constraint_circuits: &mut [ConstraintCircuit<T, DII>],
) -> String {
    let table_id_name = table_name_snake.to_upper_camel_case();
    let challenge_enum_name = format!("{table_id_name}ChallengeId");
    let table_mod_name = format!("Ext{table_id_name}");

    let initial_constraints_degrees =
        turn_circuits_into_degree_bounds_string(initial_constraint_circuits);
    let consistency_constraints_degrees =
        turn_circuits_into_degree_bounds_string(consistency_constraint_circuits);
    let transition_constraints_degrees =
        turn_circuits_into_degree_bounds_string(transition_constraint_circuits);

    let initial_constraint_strings = turn_circuits_into_string(initial_constraint_circuits);
    let consistency_constraint_strings = turn_circuits_into_string(consistency_constraint_circuits);
    let transition_constraint_strings = turn_circuits_into_string(transition_constraint_circuits);

    // maybe-prefixes to supress clippy's warnings for unused variables
    let initial_challenges_used = if initial_constraint_strings.contains("challenges") {
        ""
    } else {
        "_"
    };
    let consistency_challenges_used = if consistency_constraint_strings.contains("challenges") {
        ""
    } else {
        "_"
    };
    let consistency_constraints_exist = if consistency_constraints_degrees.is_empty() {
        "_"
    } else {
        ""
    };

    format!(
        "
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::shared_math::b_field_element::BFieldElement;

use crate::table::challenges::AllChallenges;
use crate::table::challenges::TableChallenges;
use crate::table::extension_table::Evaluable;
use crate::table::extension_table::Quotientable;
use crate::table::table_collection::interpolant_degree;
use crate::table::{table_name_snake}::{table_mod_name};
use crate::table::{table_name_snake}::{challenge_enum_name}::*;

// This file has been auto-generated. Any modifications _will_ be lost.
// To re-generate, execute:
// `cargo run --bin constraint-evaluation-generator`
impl Evaluable for {table_mod_name} {{
    #[inline]
    fn evaluate_initial_constraints(
        &self,
        row: &[XFieldElement],
        challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {{
        let {initial_challenges_used}challenges = &challenges.{table_name_snake}_challenges;
        {initial_constraint_strings}
    }}

    #[inline]
    fn evaluate_consistency_constraints(
        &self,
        {consistency_constraints_exist}row: &[XFieldElement],
        challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {{
        let {consistency_challenges_used}challenges = &challenges.{table_name_snake}_challenges;
        {consistency_constraint_strings}
    }}

    #[inline]
    fn evaluate_transition_constraints(
        &self,
        current_row: &[XFieldElement],
        next_row: &[XFieldElement],
        challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {{
        let challenges = &challenges.{table_name_snake}_challenges;
        {transition_constraint_strings}
    }}
}}

impl Quotientable for {table_mod_name} {{
    fn get_initial_quotient_degree_bounds(
        &self,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Vec<Degree> {{
        let zerofier_degree = 1 as Degree;
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);
        [{initial_constraints_degrees}].to_vec()
    }}

    fn get_consistency_quotient_degree_bounds(
        &self,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Vec<Degree> {{
        let {consistency_constraints_exist}zerofier_degree = padded_height as Degree;
        let {consistency_constraints_exist}interpolant_degree =
            interpolant_degree(padded_height, num_trace_randomizers);
        [{consistency_constraints_degrees}].to_vec()
    }}

    fn get_transition_quotient_degree_bounds(
        &self,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Vec<Degree> {{
        let zerofier_degree = padded_height as Degree - 1;
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);
        [{transition_constraints_degrees}].to_vec()
    }}
}}
"
    )
}

fn turn_circuits_into_degree_bounds_string<T: TableChallenges, II: InputIndicator>(
    transition_constraint_circuits: &[ConstraintCircuit<T, II>],
) -> String {
    transition_constraint_circuits
        .iter()
        .map(|circuit| circuit.degree())
        .map(|degree| format!("interpolant_degree * {degree} as Degree - zerofier_degree"))
        .join(",\n")
}

fn turn_circuits_into_string<T: TableChallenges, II: InputIndicator>(
    constraint_circuits: &mut [ConstraintCircuit<T, II>],
) -> String {
    // Delete redundant nodes
    ConstraintCircuit::constant_folding(&mut constraint_circuits.iter_mut().collect_vec());

    // Assert that all node IDs are unique (sanity check)
    ConstraintCircuit::assert_has_unique_ids(constraint_circuits);

    // Count number of times each node is visited
    ConstraintCircuit::traverse_multiple(constraint_circuits);

    // Get all values for the visited counters in the entire multitree
    let mut visited_counters = vec![];
    for constraint in constraint_circuits.iter() {
        visited_counters.append(&mut constraint.get_all_visited_counters());
    }

    visited_counters.sort_unstable();
    visited_counters.reverse();
    visited_counters.dedup();

    // Declare shared values
    // In the main function we predeclare all variables with a visit count of more than 1
    // These declarations must be made from the highest count number to the lowest, otherwise
    // the code will refer to bindings that have not yet been made
    let mut shared_evaluations: Vec<String> = vec![];
    for visited_counter in visited_counters {
        if visited_counter == 1 {
            continue;
        }
        shared_evaluations.push(evaluate_nodes_with_visit_count(
            visited_counter,
            constraint_circuits,
        ));
    }

    let shared_declarations = shared_evaluations.join("");

    let mut constraint_evaluation_expressions: Vec<String> = vec![];
    for constraint in constraint_circuits.iter() {
        // Build code for expressions that evaluate to the constraints
        let mut constraint_evaluation = String::default();
        let _dependent_symbols = evaluate_single_node(
            1,
            constraint,
            &HashSet::default(),
            &mut constraint_evaluation,
        );

        constraint_evaluation_expressions.push(constraint_evaluation);
    }

    let constraint_evaluations_joined = constraint_evaluation_expressions.join(",\n");

    format!("{shared_declarations}\n\nvec![{constraint_evaluations_joined}]")
}

/// Produce the code to evaluate code for all nodes that share a value number of
/// times visited. A value for all nodes with a higher count than the provided are assumed
/// to be in scope.
fn evaluate_nodes_with_visit_count<T: TableChallenges, II: InputIndicator>(
    visited_count: usize,
    circuits: &[ConstraintCircuit<T, II>],
) -> String {
    let mut in_scope: HashSet<CircuitId> = HashSet::new();
    let mut output = String::default();

    for circuit in circuits.iter() {
        declare_single_node_with_visit_count(visited_count, circuit, &mut in_scope, &mut output);
    }

    output
}

fn declare_single_node_with_visit_count<T: TableChallenges, II: InputIndicator>(
    requested_visited_count: usize,
    circuit: &ConstraintCircuit<T, II>,
    in_scope: &mut HashSet<CircuitId>,
    output: &mut String,
) {
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
    // then the node value *must* already be in scope. We should not redeclare it.
    // We also do not declare nodes that are e.g `row[3]` since they are already in scope
    // through the `points` input argument, and we do not declare constants.
    if circuit.visited_counter > requested_visited_count
        || in_scope.contains(&circuit.id)
        || matches!(circuit.expression, CircuitExpression::BConstant(_))
        || matches!(circuit.expression, CircuitExpression::XConstant(_))
        || matches!(circuit.expression, CircuitExpression::Challenge(_))
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
        output.push_str(";\n");

        let new_insertion = in_scope.insert(circuit.id.clone());
        assert!(new_insertion);
    }
}

/// Return a variable name for the node. Returns `point[n]` if node is just
/// a value from the codewords. Otherwise returns the ID of the circuit.
fn get_binding_name<T: TableChallenges, II: InputIndicator>(
    circuit: &ConstraintCircuit<T, II>,
) -> String {
    match &circuit.expression {
        CircuitExpression::XConstant(xfe) => print_xfe(xfe),
        CircuitExpression::BConstant(bfe) => print_bfe(bfe),
        CircuitExpression::Input(idx) => idx.to_string(),
        CircuitExpression::Challenge(challenge_id) => {
            format!("challenges.get_challenge({challenge_id})")
        }
        CircuitExpression::BinaryOperation(_, _, _) => format!("node_{}", circuit.id),
    }
}

/// Add to `output` the code for evaluating a single node.
/// Return a list of symbols that this evaluation depends on.
fn evaluate_single_node<T: TableChallenges, II: InputIndicator>(
    requested_visited_count: usize,
    circuit: &ConstraintCircuit<T, II>,
    in_scope: &HashSet<CircuitId>,
    output: &mut String,
) -> Vec<String> {
    // If this node has already been declared, or visit counter is higher than requested,
    // than the node value *must* be in scope, meaning that we can just reference it.
    if circuit.visited_counter > requested_visited_count || in_scope.contains(&circuit.id) {
        let binding_name = get_binding_name(circuit);
        output.push_str(&binding_name);
        return match &circuit.expression {
            CircuitExpression::BinaryOperation(_, _, _) => vec![binding_name],
            _ => vec![],
        };
    }

    // If variable is not already in scope, then we must generate the expression to
    // evaluate it.
    let mut ret = vec![];
    match &circuit.expression {
        CircuitExpression::BinaryOperation(binop, lhs, rhs) => {
            output.push('(');
            let lhs_symbols = evaluate_single_node(
                requested_visited_count,
                &lhs.as_ref().borrow(),
                in_scope,
                output,
            );
            output.push(')');
            output.push_str(&binop.to_string());
            output.push('(');
            let rhs_symbols = evaluate_single_node(
                requested_visited_count,
                &rhs.as_ref().borrow(),
                in_scope,
                output,
            );
            output.push(')');

            let ret_as_vec = vec![lhs_symbols, rhs_symbols].concat();
            let ret_as_hash_set: HashSet<String> = ret_as_vec.into_iter().collect();
            ret = ret_as_hash_set.into_iter().collect_vec()
        }
        _ => output.push_str(&get_binding_name(circuit)),
    }

    ret
}

fn print_bfe(bfe: &BFieldElement) -> String {
    format!("BFieldElement::new({})", bfe.value())
}

fn print_xfe(xfe: &XFieldElement) -> String {
    format!(
        "XFieldElement::new_u64([{}, {}, {}])",
        xfe.coefficients[0].value(),
        xfe.coefficients[1].value(),
        xfe.coefficients[2].value()
    )
}
