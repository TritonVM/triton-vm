use std::collections::HashSet;

use itertools::Itertools;
use triton_vm::table::base_table::InheritsFromTable;
use triton_vm::table::challenges::TableChallenges;
use triton_vm::table::constraint_circuit::{CircuitExpression, CircuitId, ConstraintCircuit};
use triton_vm::table::instruction_table::ExtInstructionTable;
use triton_vm::table::jump_stack_table::ExtJumpStackTable;
use triton_vm::table::op_stack_table::ExtOpStackTable;
use triton_vm::table::processor_table::ExtProcessorTable;
use triton_vm::table::program_table::ExtProgramTable;
use triton_vm::table::ram_table::ExtRamTable;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;

fn main() {
    println!("Generate those constraint evaluators!");

    // Program table
    let table = ExtProgramTable::default();
    let mut constraint_circuits = ExtProgramTable::ext_transition_constraints_as_circuits();
    gen(
        &table,
        "program_table",
        "ProgramTable",
        &mut constraint_circuits,
    );

    // Instruction table
    let table = ExtInstructionTable::default();
    let mut constraint_circuits = ExtInstructionTable::ext_transition_constraints_as_circuits();
    gen(
        &table,
        "instruction_table",
        "InstructionTable",
        &mut constraint_circuits,
    );

    // Processor table
    let table = ExtProcessorTable::default();
    let mut constraint_circuits = ExtProcessorTable::ext_transition_constraints_as_circuits();
    gen(
        &table,
        "processor_table",
        "ProcessorTable",
        &mut constraint_circuits,
    );

    // Opstack table
    let table = ExtOpStackTable::default();
    let mut constraint_circuits = ExtOpStackTable::ext_transition_constraints_as_circuits();
    gen(
        &table,
        "op_stack_table",
        "OpStackTable",
        &mut constraint_circuits,
    );

    // RAM table
    let table = ExtRamTable::default();
    let mut constraint_circuits = ExtRamTable::ext_transition_constraints_as_circuits();
    gen(&table, "ram_table", "RamTable", &mut constraint_circuits);

    // JumpStack table
    let table = ExtJumpStackTable::default();
    let mut constraint_circuits = ExtJumpStackTable::ext_transition_constraints_as_circuits();
    gen(
        &table,
        "jump_stack_table",
        "JumpStackTable",
        &mut constraint_circuits,
    );
}

fn gen<Table: InheritsFromTable<XFieldElement>, T: TableChallenges>(
    _table: &Table,
    table_name_snake: &str,
    table_id_name: &str,
    constraint_circuits: &mut [ConstraintCircuit<T>],
) {
    // Delete redundant nodes
    ConstraintCircuit::constant_folding(&mut constraint_circuits.iter_mut().collect_vec());

    // Assert that all node IDs are unique (sanity check)
    ConstraintCircuit::assert_has_unique_ids(constraint_circuits);

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
    let challenge_enum_name = format!("{table_id_name}ChallengeId");
    let mut shared_evaluations: Vec<String> = vec![];
    while requested_visited > 1 {
        shared_evaluations.push(evaluate_nodes_with_visit_count(
            requested_visited,
            constraint_circuits,
        ));
        requested_visited -= 1;
    }

    let filename = format!("triton-vm/src/table/constraints/{table_name_snake}_constraints.rs");
    let shared_declarations = shared_evaluations.join("");

    let mut constraint_evaluation_expressions: Vec<String> = vec![];
    for (_constraint_count, constraint) in constraint_circuits.iter().enumerate() {
        // Build code for expressions that evaluate to the transition constraints
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

    let root_evaluation_expressions = format!(
        "vec![
        {constraint_evaluations_joined}
    ]"
    );

    let table_mod_name = format!("Ext{table_id_name}");
    let template = format!(
        "
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::shared_math::b_field_element::BFieldElement;

use crate::table::challenges::AllChallenges;
use crate::table::challenges::TableChallenges;
use crate::table::extension_table::Evaluable;
use crate::table::{table_name_snake}::{table_mod_name};
use crate::table::{table_name_snake}::{challenge_enum_name}::*;

impl Evaluable for {table_mod_name} {{
    #[inline]
    fn evaluate_transition_constraints(
        &self,
        current_row: &[XFieldElement],
        next_row: &[XFieldElement],
        challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {{
        let challenges = &challenges.{table_name_snake}_challenges;
        {shared_declarations}

        {root_evaluation_expressions}
    }}
}}
"
    );

    println!("{template}");
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
fn get_binding_name<T: TableChallenges>(circuit: &ConstraintCircuit<T>) -> String {
    match &circuit.expression {
        CircuitExpression::XConstant(xfe) => print_xfe(*xfe),
        CircuitExpression::BConstant(bfe) => print_bfe(*bfe),
        CircuitExpression::Input(idx) => {
            if *idx >= circuit.var_count / 2 {
                let idx = idx - circuit.var_count / 2;
                format!("next_row[{idx}]")
            } else {
                format!("current_row[{idx}]")
            }
        }
        CircuitExpression::Challenge(challenge_id) => {
            format!("challenges.get_challenge({challenge_id})")
        }
        CircuitExpression::BinaryOperation(_, _, _) => format!("node_{}", circuit.id),
    }
}

/// Add to `output` the code for evaluating a single node.
/// Return a list of symbols that this evaluation depends on.
fn evaluate_single_node<T: TableChallenges>(
    requested_visited_count: usize,
    circuit: &ConstraintCircuit<T>,
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

fn print_bfe(bfe: BFieldElement) -> String {
    format!("BFieldElement::new({})", bfe.value())
}

fn print_xfe(xfe: XFieldElement) -> String {
    format!(
        "XFieldElement::new([{}, {}, {}])",
        xfe.coefficients[0].value(),
        xfe.coefficients[1].value(),
        xfe.coefficients[2].value()
    )
}
