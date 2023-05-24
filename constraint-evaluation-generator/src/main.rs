use std::collections::HashSet;
use std::process::Command;

use itertools::Itertools;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;

use triton_vm::table::cascade_table::ExtCascadeTable;
use triton_vm::table::constraint_circuit::BinOp;
use triton_vm::table::constraint_circuit::CircuitExpression;
use triton_vm::table::constraint_circuit::CircuitExpression::*;
use triton_vm::table::constraint_circuit::ConstraintCircuit;
use triton_vm::table::constraint_circuit::ConstraintCircuitBuilder;
use triton_vm::table::constraint_circuit::ConstraintCircuitMonad;
use triton_vm::table::constraint_circuit::DualRowIndicator;
use triton_vm::table::constraint_circuit::InputIndicator;
use triton_vm::table::constraint_circuit::SingleRowIndicator;
use triton_vm::table::cross_table_argument::GrandCrossTableArg;
use triton_vm::table::degree_lowering_table;
use triton_vm::table::hash_table::ExtHashTable;
use triton_vm::table::jump_stack_table::ExtJumpStackTable;
use triton_vm::table::lookup_table::ExtLookupTable;
use triton_vm::table::master_table;
use triton_vm::table::op_stack_table::ExtOpStackTable;
use triton_vm::table::processor_table::ExtProcessorTable;
use triton_vm::table::program_table::ExtProgramTable;
use triton_vm::table::ram_table::ExtRamTable;
use triton_vm::table::u32_table::ExtU32Table;

fn main() {
    let circuit_builder = ConstraintCircuitBuilder::new();
    let mut initial_constraints = vec![
        ExtProgramTable::initial_constraints(&circuit_builder),
        ExtProcessorTable::initial_constraints(&circuit_builder),
        ExtOpStackTable::initial_constraints(&circuit_builder),
        ExtRamTable::initial_constraints(&circuit_builder),
        ExtJumpStackTable::initial_constraints(&circuit_builder),
        ExtHashTable::initial_constraints(&circuit_builder),
        ExtCascadeTable::initial_constraints(&circuit_builder),
        ExtLookupTable::initial_constraints(&circuit_builder),
        ExtU32Table::initial_constraints(&circuit_builder),
        GrandCrossTableArg::initial_constraints(&circuit_builder),
    ]
    .concat();

    let circuit_builder = ConstraintCircuitBuilder::new();
    let mut consistency_constraints = vec![
        ExtProgramTable::consistency_constraints(&circuit_builder),
        ExtProcessorTable::consistency_constraints(&circuit_builder),
        ExtOpStackTable::consistency_constraints(&circuit_builder),
        ExtRamTable::consistency_constraints(&circuit_builder),
        ExtJumpStackTable::consistency_constraints(&circuit_builder),
        ExtHashTable::consistency_constraints(&circuit_builder),
        ExtCascadeTable::consistency_constraints(&circuit_builder),
        ExtLookupTable::consistency_constraints(&circuit_builder),
        ExtU32Table::consistency_constraints(&circuit_builder),
        GrandCrossTableArg::consistency_constraints(&circuit_builder),
    ]
    .concat();

    let circuit_builder = ConstraintCircuitBuilder::new();
    let mut transition_constraints = vec![
        ExtProgramTable::transition_constraints(&circuit_builder),
        ExtProcessorTable::transition_constraints(&circuit_builder),
        ExtOpStackTable::transition_constraints(&circuit_builder),
        ExtRamTable::transition_constraints(&circuit_builder),
        ExtJumpStackTable::transition_constraints(&circuit_builder),
        ExtHashTable::transition_constraints(&circuit_builder),
        ExtCascadeTable::transition_constraints(&circuit_builder),
        ExtLookupTable::transition_constraints(&circuit_builder),
        ExtU32Table::transition_constraints(&circuit_builder),
        GrandCrossTableArg::transition_constraints(&circuit_builder),
    ]
    .concat();

    let circuit_builder = ConstraintCircuitBuilder::new();
    let mut terminal_constraints = vec![
        ExtProgramTable::terminal_constraints(&circuit_builder),
        ExtProcessorTable::terminal_constraints(&circuit_builder),
        ExtOpStackTable::terminal_constraints(&circuit_builder),
        ExtRamTable::terminal_constraints(&circuit_builder),
        ExtJumpStackTable::terminal_constraints(&circuit_builder),
        ExtHashTable::terminal_constraints(&circuit_builder),
        ExtCascadeTable::terminal_constraints(&circuit_builder),
        ExtLookupTable::terminal_constraints(&circuit_builder),
        ExtU32Table::terminal_constraints(&circuit_builder),
        GrandCrossTableArg::terminal_constraints(&circuit_builder),
    ]
    .concat();

    ConstraintCircuitMonad::constant_folding(&mut initial_constraints);
    ConstraintCircuitMonad::constant_folding(&mut consistency_constraints);
    ConstraintCircuitMonad::constant_folding(&mut transition_constraints);
    ConstraintCircuitMonad::constant_folding(&mut terminal_constraints);

    // Subtract the degree lowering table's width from the total number of columns to guarantee
    // the same number of columns even for repeated runs of the constraint evaluation generator.
    let mut num_base_cols = master_table::NUM_BASE_COLUMNS - degree_lowering_table::BASE_WIDTH;
    let mut num_ext_cols = master_table::NUM_EXT_COLUMNS - degree_lowering_table::EXT_WIDTH;
    let (init_base_substitutions, init_ext_substitutions) = ConstraintCircuitMonad::lower_to_degree(
        &mut initial_constraints,
        master_table::AIR_TARGET_DEGREE,
        num_base_cols,
        num_ext_cols,
    );
    num_base_cols += init_base_substitutions.len();
    num_ext_cols += init_ext_substitutions.len();

    let (cons_base_substitutions, cons_ext_substitutions) = ConstraintCircuitMonad::lower_to_degree(
        &mut consistency_constraints,
        master_table::AIR_TARGET_DEGREE,
        num_base_cols,
        num_ext_cols,
    );
    num_base_cols += cons_base_substitutions.len();
    num_ext_cols += cons_ext_substitutions.len();

    let (tran_base_substitutions, tran_ext_substitutions) = ConstraintCircuitMonad::lower_to_degree(
        &mut transition_constraints,
        master_table::AIR_TARGET_DEGREE,
        num_base_cols,
        num_ext_cols,
    );
    num_base_cols += tran_base_substitutions.len();
    num_ext_cols += tran_ext_substitutions.len();

    let (term_base_substitutions, term_ext_substitutions) = ConstraintCircuitMonad::lower_to_degree(
        &mut terminal_constraints,
        master_table::AIR_TARGET_DEGREE,
        num_base_cols,
        num_ext_cols,
    );

    let table_code = generate_degree_lowering_table_code(
        &init_base_substitutions,
        &cons_base_substitutions,
        &tran_base_substitutions,
        &term_base_substitutions,
        &init_ext_substitutions,
        &cons_ext_substitutions,
        &tran_ext_substitutions,
        &term_ext_substitutions,
    );

    let initial_constraints = vec![
        initial_constraints,
        init_base_substitutions,
        init_ext_substitutions,
    ]
    .concat();
    let consistency_constraints = vec![
        consistency_constraints,
        cons_base_substitutions,
        cons_ext_substitutions,
    ]
    .concat();
    let transition_constraints = vec![
        transition_constraints,
        tran_base_substitutions,
        tran_ext_substitutions,
    ]
    .concat();
    let terminal_constraints = vec![
        terminal_constraints,
        term_base_substitutions,
        term_ext_substitutions,
    ]
    .concat();

    let mut initial_constraints = consume(initial_constraints);
    let mut consistency_constraints = consume(consistency_constraints);
    let mut transition_constraints = consume(transition_constraints);
    let mut terminal_constraints = consume(terminal_constraints);

    let constraint_code = generate_constraint_code(
        &mut initial_constraints,
        &mut consistency_constraints,
        &mut transition_constraints,
        &mut terminal_constraints,
    );

    std::fs::write("triton-vm/src/table/degree_lowering_table.rs", table_code)
        .expect("Writing to disk has failed.");
    std::fs::write("triton-vm/src/table/constraints.rs", constraint_code)
        .expect("Writing to disk has failed.");

    match Command::new("cargo")
        .arg("clippy")
        .arg("--workspace")
        .arg("--all-targets")
        .output()
    {
        Ok(_) => (),
        Err(err) => panic!("cargo clippy failed: {err}"),
    }
    match Command::new("cargo").arg("fmt").output() {
        Ok(_) => (),
        Err(err) => panic!("cargo fmt failed: {err}"),
    }
}

/// Consumes every `ConstraintCircuitMonad`, returning their corresponding `ConstraintCircuit`s.
fn consume<II: InputIndicator>(
    constraints: Vec<ConstraintCircuitMonad<II>>,
) -> Vec<ConstraintCircuit<II>> {
    constraints.into_iter().map(|c| c.consume()).collect()
}

fn generate_constraint_code<SII: InputIndicator, DII: InputIndicator>(
    initial_constraint_circuits: &mut [ConstraintCircuit<SII>],
    consistency_constraint_circuits: &mut [ConstraintCircuit<SII>],
    transition_constraint_circuits: &mut [ConstraintCircuit<DII>],
    terminal_constraint_circuits: &mut [ConstraintCircuit<SII>],
) -> String {
    let num_initial_constraints = initial_constraint_circuits.len();
    let num_consistency_constraints = consistency_constraint_circuits.len();
    let num_transition_constraints = transition_constraint_circuits.len();
    let num_terminal_constraints = terminal_constraint_circuits.len();

    let (
        initial_constraint_degrees,
        initial_constraint_strings_bfe,
        initial_constraint_strings_xfe,
    ) = turn_circuits_into_string(initial_constraint_circuits);
    let (
        consistency_constraint_degrees,
        consistency_constraint_strings_bfe,
        consistency_constraint_strings_xfe,
    ) = turn_circuits_into_string(consistency_constraint_circuits);
    let (
        transition_constraint_degrees,
        transition_constraint_strings_bfe,
        transition_constraint_strings_xfe,
    ) = turn_circuits_into_string(transition_constraint_circuits);
    let (
        terminal_constraint_degrees,
        terminal_constraint_strings_bfe,
        terminal_constraint_strings_xfe,
    ) = turn_circuits_into_string(terminal_constraint_circuits);

    format!(
        "
use ndarray::ArrayView1;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::challenges::Challenges;
use crate::table::challenges::ChallengeId::*;
use crate::table::extension_table::Evaluable;
use crate::table::extension_table::Quotientable;
use crate::table::master_table::MasterExtTable;

// This file has been auto-generated. Any modifications _will_ be lost.
// To re-generate, execute:
// `cargo run --bin constraint-evaluation-generator`
impl Evaluable<BFieldElement> for MasterExtTable {{
    #[inline]
    #[allow(unused_variables)]
    fn evaluate_initial_constraints(
        base_row: ArrayView1<BFieldElement>,
        ext_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement> {{
        {initial_constraint_strings_bfe}
    }}

    #[inline]
    #[allow(unused_variables)]
    fn evaluate_consistency_constraints(
        base_row: ArrayView1<BFieldElement>,
        ext_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement> {{
        {consistency_constraint_strings_bfe}
    }}

    #[inline]
    #[allow(unused_variables)]
    fn evaluate_transition_constraints(
        current_base_row: ArrayView1<BFieldElement>,
        current_ext_row: ArrayView1<XFieldElement>,
        next_base_row: ArrayView1<BFieldElement>,
        next_ext_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement> {{
        {transition_constraint_strings_bfe}
    }}

    #[inline]
    #[allow(unused_variables)]
    fn evaluate_terminal_constraints(
        base_row: ArrayView1<BFieldElement>,
        ext_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement> {{
        {terminal_constraint_strings_bfe}
    }}
}}

impl Evaluable<XFieldElement> for MasterExtTable {{
    #[inline]
    #[allow(unused_variables)]
    fn evaluate_initial_constraints(
        base_row: ArrayView1<XFieldElement>,
        ext_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement> {{
        {initial_constraint_strings_xfe}
    }}

    #[inline]
    #[allow(unused_variables)]
    fn evaluate_consistency_constraints(
        base_row: ArrayView1<XFieldElement>,
        ext_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement> {{
        {consistency_constraint_strings_xfe}
    }}

    #[inline]
    #[allow(unused_variables)]
    fn evaluate_transition_constraints(
        current_base_row: ArrayView1<XFieldElement>,
        current_ext_row: ArrayView1<XFieldElement>,
        next_base_row: ArrayView1<XFieldElement>,
        next_ext_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement> {{
        {transition_constraint_strings_xfe}
    }}

    #[inline]
    #[allow(unused_variables)]
    fn evaluate_terminal_constraints(
        base_row: ArrayView1<XFieldElement>,
        ext_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement> {{
        {terminal_constraint_strings_xfe}
    }}
}}

impl Quotientable for MasterExtTable {{
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
        let zerofier_degree = 1;
        [{initial_constraint_degrees}].to_vec()
    }}

    #[allow(unused_variables)]
    fn consistency_quotient_degree_bounds(
        interpolant_degree: Degree,
        padded_height: usize,
    ) -> Vec<Degree> {{
        let zerofier_degree = padded_height as Degree;
        [{consistency_constraint_degrees}].to_vec()
    }}

    #[allow(unused_variables)]
    fn transition_quotient_degree_bounds(
        interpolant_degree: Degree,
        padded_height: usize,
    ) -> Vec<Degree> {{
        let zerofier_degree = padded_height as Degree - 1;
        [{transition_constraint_degrees}].to_vec()
    }}

    #[allow(unused_variables)]
    fn terminal_quotient_degree_bounds(
        interpolant_degree: Degree,
    ) -> Vec<Degree> {{
        let zerofier_degree = 1;
        [{terminal_constraint_degrees}].to_vec()
    }}
}}
"
    )
}

/// Given a slice of constraint circuits, return a tuple of strings corresponding to code
/// evaluating these constraints as well as their degrees. In particular:
/// 1. The first string contains code that, when evaluated, produces the constraints' degrees,
/// 1. the second string contains code that, when evaluated, produces the constraints' values, with
///     the input type for the base row being `BFieldElement`, and
/// 1. the third string is like the second string, except that the input type for the base row is
///    `XFieldElement`.
fn turn_circuits_into_string<II: InputIndicator>(
    constraint_circuits: &mut [ConstraintCircuit<II>],
) -> (String, String, String) {
    if constraint_circuits.is_empty() {
        return ("".to_string(), "vec![]".to_string(), "vec![]".to_string());
    }

    // Sanity check: all node IDs must be unique.
    // This also ounts the number of times each node is referenced.
    ConstraintCircuit::assert_has_unique_ids(constraint_circuits);

    // Get all unique reference counts.
    let mut visited_counters = constraint_circuits
        .iter()
        .flat_map(|constraint| constraint.get_all_visited_counters())
        .collect_vec();
    visited_counters.sort_unstable();
    visited_counters.dedup();

    // Declare all shared variables, i.e., those with a visit count greater than 1.
    // These declarations must be made starting from the highest visit count.
    // Otherwise, the resulting code will refer to bindings that have not yet been made.
    let shared_declarations = visited_counters
        .into_iter()
        .filter(|&x| x > 1)
        .rev()
        .map(|visit_count| declare_nodes_with_visit_count(visit_count, constraint_circuits))
        .collect_vec()
        .join("");

    let (base_constraints, ext_constraints): (Vec<_>, Vec<_>) = constraint_circuits
        .iter()
        .partition(|constraint| constraint.evaluates_to_base_element());

    // The order of the constraints' degrees must match the order of the constraints.
    // Hence, listing the degrees is only possible after the partition into base and extension
    // constraints is known.
    let degree_bounds_string = base_constraints
        .iter()
        .chain(ext_constraints.iter())
        .map(|circuit| match circuit.degree() {
            d if d > 1 => format!("interpolant_degree * {d} - zerofier_degree"),
            d if d == 1 => "interpolant_degree - zerofier_degree".to_string(),
            _ => unreachable!("Constraint degree must be positive"),
        })
        .join(",\n");

    let build_constraint_evaluation_code = |constraints: &[&ConstraintCircuit<II>]| {
        constraints
            .iter()
            .map(|constraint| evaluate_single_node(1, constraint, &HashSet::default()))
            .collect_vec()
            .join(",\n")
    };
    let base_constraint_strings = build_constraint_evaluation_code(&base_constraints);
    let ext_constraint_strings = build_constraint_evaluation_code(&ext_constraints);

    // If there are no base constraints, the type needs to be explicitly declared.
    let base_constraint_bfe_type = match base_constraints.is_empty() {
        true => ": [BFieldElement; 0]",
        false => "",
    };

    let constraint_string_bfe = format!(
        "{shared_declarations}
        let base_constraints{base_constraint_bfe_type} = [{base_constraint_strings}];
        let ext_constraints = [{ext_constraint_strings}];
        base_constraints
            .into_iter()
            .map(|bfe| bfe.lift())
            .chain(ext_constraints.into_iter())
            .collect()"
    );

    let constraint_string_xfe = format!(
        "{shared_declarations}
        let base_constraints = [{base_constraint_strings}];
        let ext_constraints = [{ext_constraint_strings}];
        base_constraints
            .into_iter()
            .chain(ext_constraints.into_iter())
            .collect()"
    );

    (
        degree_bounds_string,
        constraint_string_bfe,
        constraint_string_xfe,
    )
}

/// Produce the code to evaluate code for all nodes that share a value number of
/// times visited. A value for all nodes with a higher count than the provided are assumed
/// to be in scope.
fn declare_nodes_with_visit_count<II: InputIndicator>(
    requested_visited_count: usize,
    circuits: &[ConstraintCircuit<II>],
) -> String {
    let mut scope: HashSet<usize> = HashSet::new();

    circuits
        .iter()
        .map(|circuit| {
            declare_single_node_with_visit_count(circuit, requested_visited_count, &mut scope)
        })
        .collect_vec()
        .join("")
}

fn declare_single_node_with_visit_count<II: InputIndicator>(
    circuit: &ConstraintCircuit<II>,
    requested_visited_count: usize,
    scope: &mut HashSet<usize>,
) -> String {
    // Don't declare a node twice.
    if scope.contains(&circuit.id) {
        return String::default();
    }

    // A higher-than-requested visit counter means the node is already in global scope, albeit not
    // necessarily in the passed-in scope.
    if circuit.visited_counter > requested_visited_count {
        return String::default();
    }

    let CircuitExpression::BinaryOperation(_, lhs, rhs) = &circuit.expression else {
        // Constants are already (or can be) trivially declared.
        return String::default();
    };

    // If the visited counter is not yet exact, start recursing on the BinaryOperation's children.
    if circuit.visited_counter < requested_visited_count {
        let out_left = declare_single_node_with_visit_count(
            &lhs.as_ref().borrow(),
            requested_visited_count,
            scope,
        );
        let out_right = declare_single_node_with_visit_count(
            &rhs.as_ref().borrow(),
            requested_visited_count,
            scope,
        );
        return [out_left, out_right].join("");
    }

    // Declare a new binding.
    assert_eq!(circuit.visited_counter, requested_visited_count);
    let binding_name = get_binding_name(circuit);
    let evaluation = evaluate_single_node(requested_visited_count, circuit, scope);

    let is_new_insertion = scope.insert(circuit.id);
    assert!(is_new_insertion);

    format!("let {binding_name} = {evaluation};\n")
}

/// Return a variable name for the node. Returns `point[n]` if node is just
/// a value from the codewords. Otherwise returns the ID of the circuit.
fn get_binding_name<II: InputIndicator>(circuit: &ConstraintCircuit<II>) -> String {
    match &circuit.expression {
        CircuitExpression::BConstant(bfe) => print_bfe(bfe),
        CircuitExpression::XConstant(xfe) => print_xfe(xfe),
        CircuitExpression::Input(idx) => idx.to_string(),
        CircuitExpression::Challenge(challenge_id) => {
            format!("challenges.get_challenge({challenge_id})")
        }
        CircuitExpression::BinaryOperation(_, _, _) => format!("node_{}", circuit.id),
    }
}

fn print_bfe(bfe: &BFieldElement) -> String {
    format!("BFieldElement::from_raw_u64({})", bfe.raw_u64())
}

fn print_xfe(xfe: &XFieldElement) -> String {
    let coeff_0 = print_bfe(&xfe.coefficients[0]);
    let coeff_1 = print_bfe(&xfe.coefficients[1]);
    let coeff_2 = print_bfe(&xfe.coefficients[2]);
    format!("XFieldElement::new([{coeff_0}, {coeff_1}, {coeff_2}])")
}

/// Recursively construct the code for evaluating a single node.
fn evaluate_single_node<II: InputIndicator>(
    requested_visited_count: usize,
    circuit: &ConstraintCircuit<II>,
    scope: &HashSet<usize>,
) -> String {
    let binding_name = get_binding_name(circuit);

    // Don't declare a node twice.
    if scope.contains(&circuit.id) {
        return binding_name;
    }

    // The binding must already be known.
    if circuit.visited_counter > requested_visited_count {
        return binding_name;
    }

    // Constants have trivial bindings.
    let CircuitExpression::BinaryOperation(binop, lhs, rhs) = &circuit.expression else {
        return binding_name;
    };

    let lhs = lhs.as_ref().borrow();
    let rhs = rhs.as_ref().borrow();
    let evaluated_lhs = evaluate_single_node(requested_visited_count, &lhs, scope);
    let evaluated_rhs = evaluate_single_node(requested_visited_count, &rhs, scope);
    let operation = binop.to_string();
    format!("({evaluated_lhs}) {operation} ({evaluated_rhs})")
}

/// Given a substitution rule, i.e., a `ConstraintCircuit` of the form `x - expr`, generate code
/// that evaluates `expr` on the appropriate base rows and sets `x` to the result.
fn substitution_rule_to_code<II: InputIndicator>(circuit: ConstraintCircuit<II>) -> String {
    let circuit_evaluates_to_base_element = circuit.evaluates_to_base_element();
    let BinaryOperation(BinOp::Sub, new_var, expr) = circuit.expression else {
        panic!("Substitution rule must be a subtraction.");
    };
    let Input(new_var) = new_var.as_ref().borrow().expression else {
        panic!("Substitution rule must be a simple substitution.");
    };
    let new_var_idx = match circuit_evaluates_to_base_element {
        true => new_var.base_col_index(),
        false => new_var.ext_col_index(),
    };

    let expr = expr.as_ref().borrow().to_owned();
    let expr = evaluate_single_node(usize::MAX, &expr, &HashSet::new());

    format!("deterministic_row[{new_var_idx} - deterministic_section_start] = {expr};")
}

/// Given all substitution rules, generate the code that evaluates them in order.
/// This includes generating the columns that are to be filled using the substitution rules.
#[allow(clippy::too_many_arguments)]
fn generate_degree_lowering_table_code(
    init_base_substitutions: &[ConstraintCircuitMonad<SingleRowIndicator>],
    cons_base_substitutions: &[ConstraintCircuitMonad<SingleRowIndicator>],
    tran_base_substitutions: &[ConstraintCircuitMonad<DualRowIndicator>],
    term_base_substitutions: &[ConstraintCircuitMonad<SingleRowIndicator>],
    init_ext_substitutions: &[ConstraintCircuitMonad<SingleRowIndicator>],
    cons_ext_substitutions: &[ConstraintCircuitMonad<SingleRowIndicator>],
    tran_ext_substitutions: &[ConstraintCircuitMonad<DualRowIndicator>],
    term_ext_substitutions: &[ConstraintCircuitMonad<SingleRowIndicator>],
) -> String {
    let num_new_base_cols = init_base_substitutions.len()
        + cons_base_substitutions.len()
        + tran_base_substitutions.len()
        + term_base_substitutions.len();
    let num_new_ext_cols = init_ext_substitutions.len()
        + cons_ext_substitutions.len()
        + tran_ext_substitutions.len()
        + term_ext_substitutions.len();

    // A zero-variant enum cannot be annotated with `repr(usize)`.
    let base_repr_usize = match num_new_base_cols == 0 {
        true => "",
        false => "#[repr(usize)]",
    };
    let ext_repr_usize = match num_new_ext_cols == 0 {
        true => "",
        false => "#[repr(usize)]",
    };

    let base_columns = (0..num_new_base_cols)
        .map(|i| format!("DegreeLoweringBaseCol{i}"))
        .collect_vec()
        .join(",\n");
    let ext_columns = (0..num_new_ext_cols)
        .map(|i| format!("DegreeLoweringExtCol{i}"))
        .collect_vec()
        .join(",\n");

    let fill_base_columns_code = generate_fill_base_columns_code(
        init_base_substitutions,
        cons_base_substitutions,
        tran_base_substitutions,
        term_base_substitutions,
    );
    let fill_ext_columns_code = generate_fill_ext_columns_code(
        init_ext_substitutions,
        cons_ext_substitutions,
        tran_ext_substitutions,
        term_ext_substitutions,
    );

    format!(
        "
use ndarray::s;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use ndarray::Axis;
use ndarray::Zip;
use strum::EnumCount;
use strum_macros::Display;
use strum_macros::EnumCount as EnumCountMacro;
use strum_macros::EnumIter;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::master_table::NUM_BASE_COLUMNS;
use crate::table::master_table::NUM_EXT_COLUMNS;

pub const BASE_WIDTH: usize = DegreeLoweringBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = DegreeLoweringExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

// This file has been auto-generated. Any modifications _will_ be lost.
// To re-generate, execute:
// `cargo run --bin constraint-evaluation-generator`

{base_repr_usize}
#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum DegreeLoweringBaseTableColumn {{
    {base_columns}
}}

{ext_repr_usize}
#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum DegreeLoweringExtTableColumn {{
    {ext_columns}
}}

#[derive(Debug, Clone)]
pub struct DegreeLoweringTable {{}}

impl DegreeLoweringTable {{
    {fill_base_columns_code}

    {fill_ext_columns_code}
}}"
    )
}

fn generate_fill_base_columns_code(
    init_base_substitutions: &[ConstraintCircuitMonad<SingleRowIndicator>],
    cons_base_substitutions: &[ConstraintCircuitMonad<SingleRowIndicator>],
    tran_base_substitutions: &[ConstraintCircuitMonad<DualRowIndicator>],
    term_base_substitutions: &[ConstraintCircuitMonad<SingleRowIndicator>],
) -> String {
    if init_base_substitutions.is_empty()
        && cons_base_substitutions.is_empty()
        && tran_base_substitutions.is_empty()
        && term_base_substitutions.is_empty()
    {
        return "pub fn fill_deterministic_base_columns(_: &mut ArrayViewMut2<BFieldElement>) {
            // prevent unused variable warning
            let _ = NUM_BASE_COLUMNS;
            // no substitutions
        }"
        .to_owned();
    }

    let single_row_substitutions = init_base_substitutions
        .iter()
        .chain(cons_base_substitutions.iter())
        .chain(term_base_substitutions.iter())
        .map(|c| substitution_rule_to_code(c.circuit.as_ref().borrow().to_owned()))
        .collect_vec()
        .join("\n");
    let single_row_substitutions = if single_row_substitutions.is_empty() {
        "".to_owned()
    } else {
        format!(
            "
        // For single-row constraints.
        Zip::from(main_trace_section.axis_iter(Axis(0)))
            .and(deterministic_section.axis_iter_mut(Axis(0)))
            .par_for_each(|base_row, mut deterministic_row| {{
                {single_row_substitutions}
            }});"
        )
    };

    let dual_row_substitutions = tran_base_substitutions
        .iter()
        .map(|c| substitution_rule_to_code(c.circuit.as_ref().borrow().to_owned()))
        .collect_vec()
        .join("\n");
    let dual_row_substitutions = if dual_row_substitutions.is_empty() {
        "".to_owned()
    } else {
        format!(
            "
        // For dual-row constraints.
        // The last row of the deterministic section for transition constraints is not used.
        let mut deterministic_section = deterministic_section.slice_mut(s![..-1, ..]);
        Zip::from(main_trace_section.axis_windows(Axis(0), 2))
            .and(deterministic_section.exact_chunks_mut((1, deterministic_section.ncols())))
            .par_for_each(|main_trace_chunk, mut deterministic_chunk| {{
                let current_base_row = main_trace_chunk.row(0);
                let next_base_row = main_trace_chunk.row(1);
                let mut deterministic_row = deterministic_chunk.row_mut(0);
                {dual_row_substitutions}
            }});"
        )
    };

    format!(
        "
    pub fn fill_deterministic_base_columns(master_base_table: &mut ArrayViewMut2<BFieldElement>) {{
        assert_eq!(NUM_BASE_COLUMNS, master_base_table.ncols());

        let main_trace_section_start = 0;
        let main_trace_section_end = main_trace_section_start + NUM_BASE_COLUMNS - BASE_WIDTH;
        let deterministic_section_start = main_trace_section_end;
        let deterministic_section_end = deterministic_section_start + BASE_WIDTH;

        let (main_trace_section, mut deterministic_section) = master_base_table.multi_slice_mut((
            s![.., main_trace_section_start..main_trace_section_end],
            s![.., deterministic_section_start..deterministic_section_end],
        ));

        {single_row_substitutions}

        {dual_row_substitutions}
    }}"
    )
}

fn generate_fill_ext_columns_code(
    init_ext_substitutions: &[ConstraintCircuitMonad<SingleRowIndicator>],
    cons_ext_substitutions: &[ConstraintCircuitMonad<SingleRowIndicator>],
    tran_ext_substitutions: &[ConstraintCircuitMonad<DualRowIndicator>],
    term_ext_substitutions: &[ConstraintCircuitMonad<SingleRowIndicator>],
) -> String {
    if init_ext_substitutions.is_empty()
        && cons_ext_substitutions.is_empty()
        && tran_ext_substitutions.is_empty()
        && term_ext_substitutions.is_empty()
    {
        return "pub fn fill_deterministic_ext_columns(
                _: ArrayView2<BFieldElement>,
                _: &mut ArrayViewMut2<XFieldElement>,
            ) {
                // prevent unused variable warning
                let _ = NUM_EXT_COLUMNS;
                // no substitutions
            }"
        .to_owned();
    }

    let single_row_substitutions = init_ext_substitutions
        .iter()
        .chain(cons_ext_substitutions.iter())
        .chain(term_ext_substitutions.iter())
        .map(|c| substitution_rule_to_code(c.circuit.as_ref().borrow().to_owned()))
        .collect_vec()
        .join("\n");
    let single_row_substitutions = if single_row_substitutions.is_empty() {
        "".to_owned()
    } else {
        format!(
            "
        // For single-row constraints.
        Zip::from(master_base_table.axis_iter(Axis(0)))
            .and(main_ext_section.axis_iter(Axis(0)))
            .and(deterministic_section.axis_iter_mut(Axis(0)))
            .par_for_each(|base_row, ext_row, mut deterministic_row| {{
                {single_row_substitutions}
            }});"
        )
    };

    let dual_row_substitutions = tran_ext_substitutions
        .iter()
        .map(|c| substitution_rule_to_code(c.circuit.as_ref().borrow().to_owned()))
        .collect_vec()
        .join("\n");
    let dual_row_substitutions = if dual_row_substitutions.is_empty() {
        "".to_owned()
    } else {
        format!(
            "
        // For dual-row constraints.
        // The last row of the deterministic section for transition constraints is not used.
        let mut deterministic_section = deterministic_section.slice_mut(s![..-1, ..]);
        Zip::from(master_base_table.axis_windows(Axis(0), 2))
            .and(main_ext_section.axis_windows(Axis(0), 2))
            .and(deterministic_section.exact_chunks_mut((1, deterministic_section.ncols())))
            .par_for_each(|base_chunk, main_ext_chunk, mut det_ext_chunk| {{
                let current_base_row = base_chunk.row(0);
                let next_base_row = base_chunk.row(1);
                let current_ext_row = main_ext_chunk.row(0);
                let next_ext_row = main_ext_chunk.row(1);
                let mut deterministic_row = det_ext_chunk.row_mut(0);
                {dual_row_substitutions}
            }});"
        )
    };

    format!(
        "
    pub fn fill_deterministic_ext_columns(
        master_base_table: ArrayView2<BFieldElement>,
        master_ext_table: &mut ArrayViewMut2<XFieldElement>,
    ) {{
        assert_eq!(NUM_BASE_COLUMNS, master_base_table.ncols());
        assert_eq!(NUM_EXT_COLUMNS, master_ext_table.ncols());

        let main_ext_section_start = 0;
        let main_ext_section_end = main_ext_section_start + NUM_EXT_COLUMNS - EXT_WIDTH;
        let det_ext_section_start = main_ext_section_end;
        let det_ext_section_end = det_ext_section_start + EXT_WIDTH;

        let (main_ext_section, mut deterministic_section) = master_ext_table.multi_slice_mut((
            s![.., main_ext_section_start..main_ext_section_end],
            s![.., det_ext_section_start..det_ext_section_end],
        ));

        {single_row_substitutions}

        {dual_row_substitutions}
    }}
"
    )
}
