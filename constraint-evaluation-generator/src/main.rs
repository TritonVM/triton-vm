use itertools::Itertools;
use std::collections::HashSet;
use std::process::Command;

use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;

use triton_vm::table::challenges::TableChallenges;
use triton_vm::table::constraint_circuit::CircuitExpression;
use triton_vm::table::constraint_circuit::ConstraintCircuit;
use triton_vm::table::constraint_circuit::InputIndicator;
use triton_vm::table::hash_table::ExtHashTable;
use triton_vm::table::instruction_table::ExtInstructionTable;
use triton_vm::table::jump_stack_table::ExtJumpStackTable;
use triton_vm::table::op_stack_table::ExtOpStackTable;
use triton_vm::table::processor_table::ExtProcessorTable;
use triton_vm::table::program_table::ExtProgramTable;
use triton_vm::table::ram_table::ExtRamTable;

fn main() {
    println!("Generate those constraint evaluators!");

    let (table_name_snake, table_name_camel) = construct_needed_table_identifiers(&["program"]);
    let source_code = gen(
        &table_name_snake,
        &table_name_camel,
        &mut ExtProgramTable::ext_initial_constraints_as_circuits(),
        &mut ExtProgramTable::ext_consistency_constraints_as_circuits(),
        &mut ExtProgramTable::ext_transition_constraints_as_circuits(),
        &mut ExtProgramTable::ext_terminal_constraints_as_circuits(),
    );
    write(&table_name_snake, source_code);

    let (table_name_snake, table_name_camel) = construct_needed_table_identifiers(&["instruction"]);
    let source_code = gen(
        &table_name_snake,
        &table_name_camel,
        &mut ExtInstructionTable::ext_initial_constraints_as_circuits(),
        &mut ExtInstructionTable::ext_consistency_constraints_as_circuits(),
        &mut ExtInstructionTable::ext_transition_constraints_as_circuits(),
        &mut ExtInstructionTable::ext_terminal_constraints_as_circuits(),
    );
    write(&table_name_snake, source_code);

    let (table_name_snake, table_name_camel) = construct_needed_table_identifiers(&["processor"]);
    let source_code = gen(
        &table_name_snake,
        &table_name_camel,
        &mut ExtProcessorTable::ext_initial_constraints_as_circuits(),
        &mut ExtProcessorTable::ext_consistency_constraints_as_circuits(),
        &mut ExtProcessorTable::ext_transition_constraints_as_circuits(),
        &mut ExtProcessorTable::ext_terminal_constraints_as_circuits(),
    );
    write(&table_name_snake, source_code);

    let (table_name_snake, table_name_camel) = construct_needed_table_identifiers(&["op", "stack"]);
    let source_code = gen(
        &table_name_snake,
        &table_name_camel,
        &mut ExtOpStackTable::ext_initial_constraints_as_circuits(),
        &mut ExtOpStackTable::ext_consistency_constraints_as_circuits(),
        &mut ExtOpStackTable::ext_transition_constraints_as_circuits(),
        &mut ExtOpStackTable::ext_terminal_constraints_as_circuits(),
    );
    write(&table_name_snake, source_code);

    let (table_name_snake, table_name_camel) = construct_needed_table_identifiers(&["ram"]);
    let source_code = gen(
        &table_name_snake,
        &table_name_camel,
        &mut ExtRamTable::ext_initial_constraints_as_circuits(),
        &mut ExtRamTable::ext_consistency_constraints_as_circuits(),
        &mut ExtRamTable::ext_transition_constraints_as_circuits(),
        &mut ExtRamTable::ext_terminal_constraints_as_circuits(),
    );
    write(&table_name_snake, source_code);

    let (table_name_snake, table_name_camel) =
        construct_needed_table_identifiers(&["jump", "stack"]);
    let source_code = gen(
        &table_name_snake,
        &table_name_camel,
        &mut ExtJumpStackTable::ext_initial_constraints_as_circuits(),
        &mut ExtJumpStackTable::ext_consistency_constraints_as_circuits(),
        &mut ExtJumpStackTable::ext_transition_constraints_as_circuits(),
        &mut ExtJumpStackTable::ext_terminal_constraints_as_circuits(),
    );
    write(&table_name_snake, source_code);

    let (table_name_snake, table_name_camel) = construct_needed_table_identifiers(&["hash"]);
    let source_code = gen(
        &table_name_snake,
        &table_name_camel,
        &mut ExtHashTable::ext_initial_constraints_as_circuits(),
        &mut ExtHashTable::ext_consistency_constraints_as_circuits(),
        &mut ExtHashTable::ext_transition_constraints_as_circuits(),
        &mut ExtHashTable::ext_terminal_constraints_as_circuits(),
    );
    write(&table_name_snake, source_code);

    if let Err(fmt_failed) = Command::new("cargo").arg("fmt").output() {
        println!("cargo fmt failed: {}", fmt_failed);
    }
}

fn construct_needed_table_identifiers(table_name_constituents: &[&str]) -> (String, String) {
    let table_name_snake = format!("{}_table", table_name_constituents.join("_"));
    let title_case = table_name_constituents
        .iter()
        .map(|part| {
            let (first_char, rest) = part.split_at(1);
            let first_char_upper = first_char.to_uppercase();
            format!("{first_char_upper}{rest}")
        })
        .collect_vec();
    let table_name_camel = format!("{}Table", title_case.iter().join(""));
    (table_name_snake, table_name_camel)
}

fn write(table_name_snake: &str, rust_source_code: String) {
    let output_filename =
        format!("triton-vm/src/table/constraints/{table_name_snake}_constraints.rs");

    std::fs::write(output_filename, rust_source_code).expect("Write Rust source code");
}

fn gen<T: TableChallenges, SII: InputIndicator, DII: InputIndicator>(
    table_name_snake: &str,
    table_id_name: &str,
    initial_constraint_circuits: &mut [ConstraintCircuit<T, SII>],
    consistency_constraint_circuits: &mut [ConstraintCircuit<T, SII>],
    transition_constraint_circuits: &mut [ConstraintCircuit<T, DII>],
    terminal_constraint_circuits: &mut [ConstraintCircuit<T, SII>],
) -> String {
    let challenge_enum_name = format!("{table_id_name}ChallengeId");
    let table_mod_name = format!("Ext{table_id_name}");

    let num_initial_constraints = initial_constraint_circuits.len();
    let num_consistency_constraints = consistency_constraint_circuits.len();
    let num_transition_constraints = transition_constraint_circuits.len();
    let num_terminal_constraints = terminal_constraint_circuits.len();

    let initial_constraints_degrees =
        turn_circuits_into_degree_bounds_string(initial_constraint_circuits);
    let consistency_constraints_degrees =
        turn_circuits_into_degree_bounds_string(consistency_constraint_circuits);
    let transition_constraints_degrees =
        turn_circuits_into_degree_bounds_string(transition_constraint_circuits);
    let terminal_constraints_degrees =
        turn_circuits_into_degree_bounds_string(terminal_constraint_circuits);

    let initial_constraint_strings = turn_circuits_into_string(initial_constraint_circuits);
    let consistency_constraint_strings = turn_circuits_into_string(consistency_constraint_circuits);
    let transition_constraint_strings = turn_circuits_into_string(transition_constraint_circuits);
    let terminal_constraint_strings = turn_circuits_into_string(terminal_constraint_circuits);

    format!(
        "
use ndarray::ArrayView1;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::challenges::AllChallenges;
use crate::table::challenges::TableChallenges;
use crate::table::extension_table::Evaluable;
use crate::table::extension_table::Quotientable;
use crate::table::{table_name_snake}::{table_mod_name};
use crate::table::{table_name_snake}::{challenge_enum_name}::*;

// This file has been auto-generated. Any modifications _will_ be lost.
// To re-generate, execute:
// `cargo run --bin constraint-evaluation-generator`
impl Evaluable for {table_mod_name} {{
    #[inline]
    #[allow(unused_variables)]
    fn evaluate_initial_constraints(
        base_row: ArrayView1<BFieldElement>,
        ext_row: ArrayView1<XFieldElement>,
        challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {{
        let challenges = &challenges.{table_name_snake}_challenges;
        {initial_constraint_strings}
    }}

    #[inline]
    #[allow(unused_variables)]
    fn evaluate_consistency_constraints(
        base_row: ArrayView1<BFieldElement>,
        ext_row: ArrayView1<XFieldElement>,
        challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {{
        let challenges = &challenges.{table_name_snake}_challenges;
        {consistency_constraint_strings}
    }}

    #[inline]
    #[allow(unused_variables)]
    fn evaluate_transition_constraints(
        current_base_row: ArrayView1<BFieldElement>,
        current_ext_row: ArrayView1<XFieldElement>,
        next_base_row: ArrayView1<BFieldElement>,
        next_ext_row: ArrayView1<XFieldElement>,
        challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {{
        let challenges = &challenges.{table_name_snake}_challenges;
        {transition_constraint_strings}
    }}

    #[inline]
    #[allow(unused_variables)]
    fn evaluate_terminal_constraints(
        base_row: ArrayView1<BFieldElement>,
        ext_row: ArrayView1<XFieldElement>,
        challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {{
        let challenges = &challenges.{table_name_snake}_challenges;
        {terminal_constraint_strings}
    }}
}}

impl Quotientable for {table_mod_name} {{
    fn num_initial_quotients() -> usize {{
        {num_initial_constraints}
    }}

    fn num_consistency_quotients() -> usize {{
        {num_consistency_constraints}
    }}

    fn num_transition_quotients() -> usize {{
        {num_transition_constraints}
    }}

    fn num_terminal_quotients() -> usize {{
        {num_terminal_constraints}
    }}

    #[allow(unused_variables)]
    fn initial_quotient_degree_bounds(
        interpolant_degree: Degree,
    ) -> Vec<Degree> {{
        let zerofier_degree = 1 as Degree;
        [{initial_constraints_degrees}].to_vec()
    }}

    #[allow(unused_variables)]
    fn consistency_quotient_degree_bounds(
        interpolant_degree: Degree,
        padded_height: usize,
    ) -> Vec<Degree> {{
        let zerofier_degree = padded_height as Degree;
        [{consistency_constraints_degrees}].to_vec()
    }}

    #[allow(unused_variables)]
    fn transition_quotient_degree_bounds(
        interpolant_degree: Degree,
        padded_height: usize,
    ) -> Vec<Degree> {{
        let zerofier_degree = padded_height as Degree - 1;
        [{transition_constraints_degrees}].to_vec()
    }}

    #[allow(unused_variables)]
    fn terminal_quotient_degree_bounds(
        interpolant_degree: Degree,
    ) -> Vec<Degree> {{
        let zerofier_degree = 1 as Degree;
        [{terminal_constraints_degrees}].to_vec()
    }}
}}
"
    )
}

fn turn_circuits_into_degree_bounds_string<T: TableChallenges, II: InputIndicator>(
    constraint_circuits: &[ConstraintCircuit<T, II>],
) -> String {
    constraint_circuits
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

    // Get all values for the visited counters in the entire multi-circuit
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
        shared_evaluations.push(declare_nodes_with_visit_count(
            visited_counter,
            constraint_circuits,
        ));
    }

    let shared_declarations = shared_evaluations.join("");

    let mut base_constraint_evaluation_expressions: Vec<String> = vec![];
    let mut ext_constraint_evaluation_expressions: Vec<String> = vec![];
    for constraint in constraint_circuits.iter() {
        // Build code for expressions that evaluate to the constraints
        let (constraint_evaluation, _dependent_symbols) =
            evaluate_single_node(1, constraint, &HashSet::default());
        match is_bfield_element(constraint) {
            true => base_constraint_evaluation_expressions.push(constraint_evaluation),
            false => ext_constraint_evaluation_expressions.push(constraint_evaluation),
        }
    }

    let base_constraint_evaluations_joined = base_constraint_evaluation_expressions.join(",\n");
    let ext_constraint_evaluations_joined = ext_constraint_evaluation_expressions.join(",\n");

    format!(
        "{shared_declarations}

        let base_constraints = [{base_constraint_evaluations_joined}];
        let ext_constraints = [{ext_constraint_evaluations_joined}];

        base_constraints
            .map(|v| BFieldElement::lift(&v))
            .iter()
            .chain(ext_constraints.iter())
            .cloned()
            .collect()"
    )
}

/// Produce the code to evaluate code for all nodes that share a value number of
/// times visited. A value for all nodes with a higher count than the provided are assumed
/// to be in scope.
fn declare_nodes_with_visit_count<T: TableChallenges, II: InputIndicator>(
    requested_visited_count: usize,
    circuits: &[ConstraintCircuit<T, II>],
) -> String {
    let mut in_scope: HashSet<usize> = HashSet::new();
    let mut output = String::default();

    for circuit in circuits.iter() {
        declare_single_node_with_visit_count(
            requested_visited_count,
            circuit,
            &mut in_scope,
            &mut output,
        );
    }

    output
}

fn declare_single_node_with_visit_count<T: TableChallenges, II: InputIndicator>(
    requested_visited_count: usize,
    circuit: &ConstraintCircuit<T, II>,
    in_scope: &mut HashSet<usize>,
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
        || !matches!(
            circuit.expression,
            CircuitExpression::BinaryOperation(_, _, _)
        )
    {
        return;
    }

    // If this line is met, it means that the visit count is as requested, and that
    // the value is not in scope. So it must be added to the scope. We find the
    // expression for the value, and then put it into scope through a let expression
    if circuit.visited_counter == requested_visited_count && !in_scope.contains(&circuit.id) {
        let binding_name = get_binding_name(circuit);
        output.push_str(&format!("let {binding_name} =\n"));
        let (to_output, _) = evaluate_single_node(requested_visited_count, circuit, in_scope);
        output.push_str(&to_output);
        output.push_str(";\n");

        let new_insertion = in_scope.insert(circuit.id);
        // sanity check: don't declare same node multiple times
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

/// Recursively check whether a node is composed of only BFieldElements, i.e., only uses
/// (1) inputs from base rows, (2) constants from the B-field, and (3) binary operations on
/// BFieldElements.
fn is_bfield_element<T: TableChallenges, II: InputIndicator>(
    circuit: &ConstraintCircuit<T, II>,
) -> bool {
    match &circuit.expression {
        CircuitExpression::XConstant(_) => false,
        CircuitExpression::BConstant(_) => true,
        CircuitExpression::Input(indicator) => indicator.is_base_table_row(),
        CircuitExpression::Challenge(_) => false,
        CircuitExpression::BinaryOperation(_, lhs, rhs) => {
            is_bfield_element(&lhs.as_ref().borrow()) && is_bfield_element(&rhs.as_ref().borrow())
        }
    }
}

/// Return (1) the code for evaluating a single node and (2) a list of symbols that this evaluation
/// depends on.
fn evaluate_single_node<T: TableChallenges, II: InputIndicator>(
    requested_visited_count: usize,
    circuit: &ConstraintCircuit<T, II>,
    in_scope: &HashSet<usize>,
) -> (String, Vec<String>) {
    let mut output = String::default();
    // If this node has already been declared, or visit counter is higher than requested,
    // than the node value *must* be in scope, meaning that we can just reference it.
    if circuit.visited_counter > requested_visited_count || in_scope.contains(&circuit.id) {
        let binding_name = get_binding_name(circuit);
        output.push_str(&binding_name);
        return match &circuit.expression {
            CircuitExpression::BinaryOperation(_, _, _) => (output, vec![binding_name]),
            _ => (output, vec![]),
        };
    }

    // If variable is not already in scope, then we must generate the expression to evaluate it.
    let mut dependent_symbols = vec![];
    match &circuit.expression {
        CircuitExpression::BinaryOperation(binop, lhs, rhs) => {
            output.push('(');
            let (to_output, lhs_symbols) =
                evaluate_single_node(requested_visited_count, &lhs.as_ref().borrow(), in_scope);
            output.push_str(&to_output);
            output.push(')');
            output.push_str(&binop.to_string());
            output.push('(');
            let (to_output, rhs_symbols) =
                evaluate_single_node(requested_visited_count, &rhs.as_ref().borrow(), in_scope);
            output.push_str(&to_output);
            output.push(')');

            let ret_as_vec = vec![lhs_symbols, rhs_symbols].concat();
            let ret_as_hash_set: HashSet<String> = ret_as_vec.into_iter().collect();
            dependent_symbols = ret_as_hash_set.into_iter().collect_vec()
        }
        _ => output.push_str(&get_binding_name(circuit)),
    }

    (output, dependent_symbols)
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
