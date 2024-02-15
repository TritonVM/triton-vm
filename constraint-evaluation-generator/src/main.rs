use std::collections::HashSet;
use std::fs::write;

use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::format_ident;
use quote::quote;
use twenty_first::prelude::*;

use triton_vm::table::cascade_table::ExtCascadeTable;
use triton_vm::table::constraint_circuit::BinOp;
use triton_vm::table::constraint_circuit::CircuitExpression;
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

mod asm;

struct AllSubstitutions {
    base: Substitutions,
    ext: Substitutions,
}

struct Substitutions {
    init: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
    cons: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
    tran: Vec<ConstraintCircuitMonad<DualRowIndicator>>,
    term: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
}

impl Substitutions {
    fn len(&self) -> usize {
        self.init.len() + self.cons.len() + self.tran.len() + self.term.len()
    }
}

struct Constraints {
    init: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
    cons: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
    tran: Vec<ConstraintCircuitMonad<DualRowIndicator>>,
    term: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
}

fn main() {
    let mut constraints = all_constraints();
    let substitutions = lower_to_target_degree_through_substitutions(&mut constraints);
    let degree_lowering_table_code = generate_degree_lowering_table_code(&substitutions);

    let constraints =
        combine_existing_and_substitution_induced_constraints(constraints, substitutions);
    let constraint_code = generate_constraint_code(&constraints);
    let triton_asm_constraint_code = asm::generate_constraint_code(&constraints);

    write_code_to_file(degree_lowering_table_code, "degree_lowering_table");
    write_code_to_file(constraint_code, "constraints");
    write_code_to_file(triton_asm_constraint_code, "asm_air_constraints");
}

fn all_constraints() -> Constraints {
    let mut constraints = Constraints {
        init: all_initial_constraints(),
        cons: all_consistency_constraints(),
        tran: all_transition_constraints(),
        term: all_terminal_constraints(),
    };
    constant_fold_all_constraints(&mut constraints);
    constraints
}

fn all_initial_constraints() -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
    let circuit_builder = ConstraintCircuitBuilder::new();
    vec![
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
    .concat()
}

fn all_consistency_constraints() -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
    let circuit_builder = ConstraintCircuitBuilder::new();
    vec![
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
    .concat()
}

fn all_transition_constraints() -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let circuit_builder = ConstraintCircuitBuilder::new();
    vec![
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
    .concat()
}

fn all_terminal_constraints() -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
    let circuit_builder = ConstraintCircuitBuilder::new();
    vec![
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
    .concat()
}

fn constant_fold_all_constraints(constraints: &mut Constraints) {
    ConstraintCircuitMonad::constant_folding(&mut constraints.init);
    ConstraintCircuitMonad::constant_folding(&mut constraints.cons);
    ConstraintCircuitMonad::constant_folding(&mut constraints.tran);
    ConstraintCircuitMonad::constant_folding(&mut constraints.term);
}

fn lower_to_target_degree_through_substitutions(
    all_constraints: &mut Constraints,
) -> AllSubstitutions {
    // Subtract the degree lowering table's width from the total number of columns to guarantee
    // the same number of columns even for repeated runs of the constraint evaluation generator.
    let mut num_base_cols = master_table::NUM_BASE_COLUMNS - degree_lowering_table::BASE_WIDTH;
    let mut num_ext_cols = master_table::NUM_EXT_COLUMNS - degree_lowering_table::EXT_WIDTH;
    let (init_base_substitutions, init_ext_substitutions) = ConstraintCircuitMonad::lower_to_degree(
        &mut all_constraints.init,
        master_table::AIR_TARGET_DEGREE,
        num_base_cols,
        num_ext_cols,
    );
    num_base_cols += init_base_substitutions.len();
    num_ext_cols += init_ext_substitutions.len();

    let (cons_base_substitutions, cons_ext_substitutions) = ConstraintCircuitMonad::lower_to_degree(
        &mut all_constraints.cons,
        master_table::AIR_TARGET_DEGREE,
        num_base_cols,
        num_ext_cols,
    );
    num_base_cols += cons_base_substitutions.len();
    num_ext_cols += cons_ext_substitutions.len();

    let (tran_base_substitutions, tran_ext_substitutions) = ConstraintCircuitMonad::lower_to_degree(
        &mut all_constraints.tran,
        master_table::AIR_TARGET_DEGREE,
        num_base_cols,
        num_ext_cols,
    );
    num_base_cols += tran_base_substitutions.len();
    num_ext_cols += tran_ext_substitutions.len();

    let (term_base_substitutions, term_ext_substitutions) = ConstraintCircuitMonad::lower_to_degree(
        &mut all_constraints.term,
        master_table::AIR_TARGET_DEGREE,
        num_base_cols,
        num_ext_cols,
    );

    AllSubstitutions {
        base: Substitutions {
            init: init_base_substitutions,
            cons: cons_base_substitutions,
            tran: tran_base_substitutions,
            term: term_base_substitutions,
        },
        ext: Substitutions {
            init: init_ext_substitutions,
            cons: cons_ext_substitutions,
            tran: tran_ext_substitutions,
            term: term_ext_substitutions,
        },
    }
}

fn combine_existing_and_substitution_induced_constraints(
    constraints: Constraints,
    substitutions: AllSubstitutions,
) -> Constraints {
    let init = [
        constraints.init,
        substitutions.base.init,
        substitutions.ext.init,
    ];
    let cons = [
        constraints.cons,
        substitutions.base.cons,
        substitutions.ext.cons,
    ];
    let tran = [
        constraints.tran,
        substitutions.base.tran,
        substitutions.ext.tran,
    ];
    let term = [
        constraints.term,
        substitutions.base.term,
        substitutions.ext.term,
    ];
    Constraints {
        init: init.concat(),
        cons: cons.concat(),
        tran: tran.concat(),
        term: term.concat(),
    }
}

fn generate_constraint_code(constraints: &Constraints) -> TokenStream {
    let num_init_constraints = constraints.init.len();
    let num_cons_constraints = constraints.cons.len();
    let num_tran_constraints = constraints.tran.len();
    let num_term_constraints = constraints.term.len();

    let mut init_constraint_circuits = consume(&constraints.init);
    let mut cons_constraint_circuits = consume(&constraints.cons);
    let mut tran_constraint_circuits = consume(&constraints.tran);
    let mut term_constraint_circuits = consume(&constraints.term);

    let (init_constraint_degrees, init_constraints_bfe, init_constraints_xfe) =
        tokenize_circuits(&mut init_constraint_circuits);
    let (cons_constraint_degrees, cons_constraints_bfe, cons_constraints_xfe) =
        tokenize_circuits(&mut cons_constraint_circuits);
    let (tran_constraint_degrees, tran_constraints_bfe, tran_constraints_xfe) =
        tokenize_circuits(&mut tran_constraint_circuits);
    let (term_constraint_degrees, term_constraints_bfe, term_constraints_xfe) =
        tokenize_circuits(&mut term_constraint_circuits);

    let imports = generate_imports();
    let evaluable_over_base_field = generate_evaluable_implementation_over_field(
        &init_constraints_bfe,
        &cons_constraints_bfe,
        &tran_constraints_bfe,
        &term_constraints_bfe,
        quote!(BFieldElement),
    );
    let evaluable_over_ext_field = generate_evaluable_implementation_over_field(
        &init_constraints_xfe,
        &cons_constraints_xfe,
        &tran_constraints_xfe,
        &term_constraints_xfe,
        quote!(XFieldElement),
    );

    let quotient_trait_impl = quote!(
    impl Quotientable for MasterExtTable {
        fn num_initial_quotients() -> usize { #num_init_constraints }
        fn num_consistency_quotients() -> usize { #num_cons_constraints }
        fn num_transition_quotients() -> usize { #num_tran_constraints }
        fn num_terminal_quotients() -> usize { #num_term_constraints }

        #[allow(unused_variables)]
        fn initial_quotient_degree_bounds(interpolant_degree: Degree) -> Vec<Degree> {
            let zerofier_degree = 1;
            [#init_constraint_degrees].to_vec()
        }

        #[allow(unused_variables)]
        fn consistency_quotient_degree_bounds(
            interpolant_degree: Degree,
            padded_height: usize,
        ) -> Vec<Degree> {
            let zerofier_degree = padded_height as Degree;
            [#cons_constraint_degrees].to_vec()
        }

        #[allow(unused_variables)]
        fn transition_quotient_degree_bounds(
            interpolant_degree: Degree,
            padded_height: usize,
        ) -> Vec<Degree> {
            let zerofier_degree = padded_height as Degree - 1;
            [#tran_constraint_degrees].to_vec()
        }

        #[allow(unused_variables)]
        fn terminal_quotient_degree_bounds(interpolant_degree: Degree) -> Vec<Degree> {
            let zerofier_degree = 1;
            [#term_constraint_degrees].to_vec()
        }
    }
    );

    quote!(
        // This file has been auto-generated. Any modifications _will_ be lost.
        // To re-generate, execute:
        // `cargo run --bin constraint-evaluation-generator`
        #imports
        #evaluable_over_base_field
        #evaluable_over_ext_field
        #quotient_trait_impl
    )
}

fn generate_imports() -> TokenStream {
    quote!(
        use ndarray::ArrayView1;
        use twenty_first::prelude::BFieldElement;
        use twenty_first::prelude::XFieldElement;
        use twenty_first::shared_math::mpolynomial::Degree;

        use crate::table::challenges::Challenges;
        use crate::table::challenges::ChallengeId::*;
        use crate::table::extension_table::Evaluable;
        use crate::table::extension_table::Quotientable;
        use crate::table::master_table::MasterExtTable;
    )
}

fn generate_evaluable_implementation_over_field(
    init_constraints: &TokenStream,
    cons_constraints: &TokenStream,
    tran_constraints: &TokenStream,
    term_constraints: &TokenStream,
    field: TokenStream,
) -> TokenStream {
    quote!(
    impl Evaluable<#field> for MasterExtTable {
        #[allow(unused_variables)]
        fn evaluate_initial_constraints(
            base_row: ArrayView1<#field>,
            ext_row: ArrayView1<XFieldElement>,
            challenges: &Challenges,
        ) -> Vec<XFieldElement> {
            #init_constraints
        }

        #[allow(unused_variables)]
        fn evaluate_consistency_constraints(
            base_row: ArrayView1<#field>,
            ext_row: ArrayView1<XFieldElement>,
            challenges: &Challenges,
        ) -> Vec<XFieldElement> {
            #cons_constraints
        }

        #[allow(unused_variables)]
        fn evaluate_transition_constraints(
            current_base_row: ArrayView1<#field>,
            current_ext_row: ArrayView1<XFieldElement>,
            next_base_row: ArrayView1<#field>,
            next_ext_row: ArrayView1<XFieldElement>,
            challenges: &Challenges,
        ) -> Vec<XFieldElement> {
            #tran_constraints
        }

        #[allow(unused_variables)]
        fn evaluate_terminal_constraints(
            base_row: ArrayView1<#field>,
            ext_row: ArrayView1<XFieldElement>,
            challenges: &Challenges,
        ) -> Vec<XFieldElement> {
            #term_constraints
        }
    }
    )
}

/// Consumes every [`ConstraintCircuitMonad`], returning their corresponding [`ConstraintCircuit`]s.
fn consume<II: InputIndicator>(
    constraints: &[ConstraintCircuitMonad<II>],
) -> Vec<ConstraintCircuit<II>> {
    constraints.iter().map(|c| c.consume()).collect()
}

/// Given a slice of constraint circuits, return a tuple of [`TokenStream`]s corresponding to code
/// evaluating these constraints as well as their degrees. In particular:
/// 1. The first stream contains code that, when evaluated, produces the constraints' degrees,
/// 1. the second stream contains code that, when evaluated, produces the constraints' values, with
///     the input type for the base row being `BFieldElement`, and
/// 1. the third stream is like the second, except that the input type for the base row is
///    `XFieldElement`.
fn tokenize_circuits<II: InputIndicator>(
    constraint_circuits: &mut [ConstraintCircuit<II>],
) -> (TokenStream, TokenStream, TokenStream) {
    if constraint_circuits.is_empty() {
        return (quote!(), quote!(vec![]), quote!(vec![]));
    }

    // Sanity check: all node IDs must be unique.
    // This also counts the number of times each node is referenced.
    ConstraintCircuit::assert_has_unique_ids(constraint_circuits);

    // Get all unique reference counts.
    let mut visited_counters = constraint_circuits
        .iter()
        .flat_map(ConstraintCircuit::all_visited_counters)
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
        .collect_vec();

    let (base_constraints, ext_constraints): (Vec<_>, Vec<_>) = constraint_circuits
        .iter()
        .partition(|constraint| constraint.evaluates_to_base_element());

    // The order of the constraints' degrees must match the order of the constraints.
    // Hence, listing the degrees is only possible after the partition into base and extension
    // constraints is known.
    let tokenized_degree_bounds = base_constraints
        .iter()
        .chain(ext_constraints.iter())
        .map(|circuit| match circuit.degree() {
            d if d > 1 => quote!(interpolant_degree * #d - zerofier_degree),
            1 => quote!(interpolant_degree - zerofier_degree),
            _ => unreachable!("Constraint degree must be positive"),
        })
        .collect_vec();
    let tokenized_degree_bounds = quote!(#(#tokenized_degree_bounds),*);

    let tokenize_constraint_evaluation = |constraints: &[&ConstraintCircuit<II>]| {
        constraints
            .iter()
            .map(|constraint| evaluate_single_node(1, constraint, &HashSet::default()))
            .collect_vec()
    };
    let tokenized_base_constraints = tokenize_constraint_evaluation(&base_constraints);
    let tokenized_ext_constraints = tokenize_constraint_evaluation(&ext_constraints);

    // If there are no base constraints, the type needs to be explicitly declared.
    let tokenized_bfe_base_constraints = match base_constraints.is_empty() {
        true => quote!(let base_constraints: [BFieldElement; 0] = []),
        false => quote!(let base_constraints = [#(#tokenized_base_constraints),*]),
    };
    let tokenized_bfe_constraints = quote!(
        #(#shared_declarations)*
        #tokenized_bfe_base_constraints;
        let ext_constraints = [#(#tokenized_ext_constraints),*];
        base_constraints
            .into_iter()
            .map(|bfe| bfe.lift())
            .chain(ext_constraints)
            .collect()
    );

    let tokenized_xfe_constraints = quote!(
        #(#shared_declarations)*
        let base_constraints = [#(#tokenized_base_constraints),*];
        let ext_constraints = [#(#tokenized_ext_constraints),*];
        base_constraints
            .into_iter()
            .chain(ext_constraints)
            .collect()
    );

    (
        tokenized_degree_bounds,
        tokenized_bfe_constraints,
        tokenized_xfe_constraints,
    )
}

/// Produce the code to evaluate code for all nodes that share a value number of
/// times visited. A value for all nodes with a higher count than the provided are assumed
/// to be in scope.
fn declare_nodes_with_visit_count<II: InputIndicator>(
    requested_visited_count: usize,
    circuits: &[ConstraintCircuit<II>],
) -> TokenStream {
    let mut scope: HashSet<usize> = HashSet::new();

    let tokenized_circuits = circuits
        .iter()
        .filter_map(|circuit| {
            declare_single_node_with_visit_count(circuit, requested_visited_count, &mut scope)
        })
        .collect_vec();
    quote!(#(#tokenized_circuits)*)
}

fn declare_single_node_with_visit_count<II: InputIndicator>(
    circuit: &ConstraintCircuit<II>,
    requested_visited_count: usize,
    scope: &mut HashSet<usize>,
) -> Option<TokenStream> {
    // Don't declare a node twice.
    if scope.contains(&circuit.id) {
        return None;
    }

    // A higher-than-requested visit counter means the node is already in global scope, albeit not
    // necessarily in the passed-in scope.
    if circuit.visited_counter > requested_visited_count {
        return None;
    }

    let CircuitExpression::BinaryOperation(_, lhs, rhs) = &circuit.expression else {
        // Constants are already (or can be) trivially declared.
        return None;
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
        return match (out_left, out_right) {
            (None, None) => None,
            (Some(l), None) => Some(l),
            (None, Some(r)) => Some(r),
            (Some(l), Some(r)) => Some(quote!(#l #r)),
        };
    }

    // Declare a new binding.
    assert_eq!(circuit.visited_counter, requested_visited_count);
    let binding_name = get_binding_name(circuit);
    let evaluation = evaluate_single_node(requested_visited_count, circuit, scope);

    let is_new_insertion = scope.insert(circuit.id);
    assert!(is_new_insertion);

    Some(quote!(let #binding_name = #evaluation;))
}

/// Return a variable name for the node. Returns `point[n]` if node is just
/// a value from the codewords. Otherwise, returns the ID of the circuit.
fn get_binding_name<II: InputIndicator>(circuit: &ConstraintCircuit<II>) -> TokenStream {
    match &circuit.expression {
        CircuitExpression::BConstant(bfe) => tokenize_bfe(*bfe),
        CircuitExpression::XConstant(xfe) => tokenize_xfe(*xfe),
        CircuitExpression::Input(idx) => quote!(#idx),
        CircuitExpression::Challenge(challenge) => {
            let challenge_ident = format_ident!("{challenge}");
            quote!(challenges[#challenge_ident])
        }
        CircuitExpression::BinaryOperation(_, _, _) => {
            let node_ident = format_ident!("node_{}", circuit.id);
            quote!(#node_ident)
        }
    }
}

fn tokenize_bfe(bfe: BFieldElement) -> TokenStream {
    let raw_u64 = bfe.raw_u64();
    quote!(BFieldElement::from_raw_u64(#raw_u64))
}

fn tokenize_xfe(xfe: XFieldElement) -> TokenStream {
    let [c_0, c_1, c_2] = xfe.coefficients.map(tokenize_bfe);
    quote!(XFieldElement::new([#c_0, #c_1, #c_2]))
}

/// Recursively construct the code for evaluating a single node.
fn evaluate_single_node<II: InputIndicator>(
    requested_visited_count: usize,
    circuit: &ConstraintCircuit<II>,
    scope: &HashSet<usize>,
) -> TokenStream {
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
    quote!((#evaluated_lhs) #binop (#evaluated_rhs))
}

/// Given all substitution rules, generate the code that evaluates them in order.
/// This includes generating the columns that are to be filled using the substitution rules.
fn generate_degree_lowering_table_code(substitutions: &AllSubstitutions) -> TokenStream {
    let num_new_base_cols = substitutions.base.len();
    let num_new_ext_cols = substitutions.ext.len();

    // A zero-variant enum cannot be annotated with `repr(usize)`.
    let base_repr_usize = match num_new_base_cols == 0 {
        true => quote!(),
        false => quote!(#[repr(usize)]),
    };
    let ext_repr_usize = match num_new_ext_cols == 0 {
        true => quote!(),
        false => quote!(#[repr(usize)]),
    };
    let use_challenge_ids = match num_new_ext_cols == 0 {
        true => quote!(),
        false => quote!(
            use crate::table::challenges::ChallengeId::*;
        ),
    };

    let base_columns = (0..num_new_base_cols)
        .map(|i| format_ident!("DegreeLoweringBaseCol{i}"))
        .map(|ident| quote!(#ident))
        .collect_vec();
    let ext_columns = (0..num_new_ext_cols)
        .map(|i| format_ident!("DegreeLoweringExtCol{i}"))
        .map(|ident| quote!(#ident))
        .collect_vec();

    let fill_base_columns_code = generate_fill_base_columns_code(&substitutions.base);
    let fill_ext_columns_code = generate_fill_ext_columns_code(&substitutions.ext);

    quote!(
        //! The degree lowering table contains the introduced variables that allow
        //! lowering the degree of the AIR. See
        //! [`crate::table::master_table::AIR_TARGET_DEGREE`]
        //! for additional information.
        //!
        //! This file has been auto-generated. Any modifications _will_ be lost.
        //! To re-generate, execute:
        //! `cargo run --bin constraint-evaluation-generator`

        use ndarray::s;
        use ndarray::ArrayView2;
        use ndarray::ArrayViewMut2;
        use strum::Display;
        use strum::EnumCount;
        use strum::EnumIter;
        use twenty_first::prelude::BFieldElement;
        use twenty_first::prelude::XFieldElement;

        #use_challenge_ids
        use crate::table::challenges::Challenges;
        use crate::table::master_table::NUM_BASE_COLUMNS;
        use crate::table::master_table::NUM_EXT_COLUMNS;

        pub const BASE_WIDTH: usize = DegreeLoweringBaseTableColumn::COUNT;
        pub const EXT_WIDTH: usize = DegreeLoweringExtTableColumn::COUNT;
        pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

        #base_repr_usize
        #[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
        pub enum DegreeLoweringBaseTableColumn {
            #(#base_columns),*
        }

        #ext_repr_usize
        #[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
        pub enum DegreeLoweringExtTableColumn {
            #(#ext_columns),*
        }

        #[derive(Debug, Copy, Clone, Eq, PartialEq)]
        pub struct DegreeLoweringTable;

        impl DegreeLoweringTable {
            #fill_base_columns_code
            #fill_ext_columns_code
        }
    )
}

fn generate_fill_base_columns_code(substitutions: &Substitutions) -> TokenStream {
    let derived_section_init_start =
        master_table::NUM_BASE_COLUMNS - degree_lowering_table::BASE_WIDTH;
    let derived_section_cons_start = derived_section_init_start + substitutions.init.len();
    let derived_section_tran_start = derived_section_cons_start + substitutions.cons.len();
    let derived_section_term_start = derived_section_tran_start + substitutions.tran.len();

    let init_col_indices = (0..substitutions.init.len())
        .map(|i| i + derived_section_init_start)
        .collect_vec();
    let cons_col_indices = (0..substitutions.cons.len())
        .map(|i| i + derived_section_cons_start)
        .collect_vec();
    let tran_col_indices = (0..substitutions.tran.len())
        .map(|i| i + derived_section_tran_start)
        .collect_vec();
    let term_col_indices = (0..substitutions.term.len())
        .map(|i| i + derived_section_term_start)
        .collect_vec();

    let init_substitutions = several_substitution_rules_to_code(&substitutions.init);
    let cons_substitutions = several_substitution_rules_to_code(&substitutions.cons);
    let tran_substitutions = several_substitution_rules_to_code(&substitutions.tran);
    let term_substitutions = several_substitution_rules_to_code(&substitutions.term);

    let init_substitutions = base_single_row_substitutions(&init_col_indices, &init_substitutions);
    let cons_substitutions = base_single_row_substitutions(&cons_col_indices, &cons_substitutions);
    let tran_substitutions = base_dual_row_substitutions(&tran_col_indices, &tran_substitutions);
    let term_substitutions = base_single_row_substitutions(&term_col_indices, &term_substitutions);

    quote!(
    #[allow(unused_variables)]
    pub fn fill_derived_base_columns(mut master_base_table: ArrayViewMut2<BFieldElement>) {
        assert_eq!(NUM_BASE_COLUMNS, master_base_table.ncols());
        #init_substitutions
        #cons_substitutions
        #tran_substitutions
        #term_substitutions
    }
    )
}

fn generate_fill_ext_columns_code(substitutions: &Substitutions) -> TokenStream {
    let derived_section_init_start =
        master_table::NUM_EXT_COLUMNS - degree_lowering_table::EXT_WIDTH;
    let derived_section_cons_start = derived_section_init_start + substitutions.init.len();
    let derived_section_tran_start = derived_section_cons_start + substitutions.cons.len();
    let derived_section_term_start = derived_section_tran_start + substitutions.tran.len();

    let init_col_indices = (0..substitutions.init.len())
        .map(|i| i + derived_section_init_start)
        .collect_vec();
    let cons_col_indices = (0..substitutions.cons.len())
        .map(|i| i + derived_section_cons_start)
        .collect_vec();
    let tran_col_indices = (0..substitutions.tran.len())
        .map(|i| i + derived_section_tran_start)
        .collect_vec();
    let term_col_indices = (0..substitutions.term.len())
        .map(|i| i + derived_section_term_start)
        .collect_vec();

    let init_substitutions = several_substitution_rules_to_code(&substitutions.init);
    let cons_substitutions = several_substitution_rules_to_code(&substitutions.cons);
    let tran_substitutions = several_substitution_rules_to_code(&substitutions.tran);
    let term_substitutions = several_substitution_rules_to_code(&substitutions.term);

    let init_substitutions = ext_single_row_substitutions(&init_col_indices, &init_substitutions);
    let cons_substitutions = ext_single_row_substitutions(&cons_col_indices, &cons_substitutions);
    let tran_substitutions = ext_dual_row_substitutions(&tran_col_indices, &tran_substitutions);
    let term_substitutions = ext_single_row_substitutions(&term_col_indices, &term_substitutions);

    quote!(
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        pub fn fill_derived_ext_columns(
            master_base_table: ArrayView2<BFieldElement>,
            mut master_ext_table: ArrayViewMut2<XFieldElement>,
            challenges: &Challenges,
        ) {
            assert_eq!(NUM_BASE_COLUMNS, master_base_table.ncols());
            assert_eq!(NUM_EXT_COLUMNS, master_ext_table.ncols());
            assert_eq!(master_base_table.nrows(), master_ext_table.nrows());
            #init_substitutions
            #cons_substitutions
            #tran_substitutions
            #term_substitutions
        }
    )
}

fn several_substitution_rules_to_code<II: InputIndicator>(
    substitution_rules: &[ConstraintCircuitMonad<II>],
) -> Vec<TokenStream> {
    substitution_rules
        .iter()
        .map(|c| substitution_rule_to_code(c.circuit.as_ref().borrow().to_owned()))
        .collect()
}

/// Given a substitution rule, i.e., a `ConstraintCircuit` of the form `x - expr`, generate code
/// that evaluates `expr`.
fn substitution_rule_to_code<II: InputIndicator>(circuit: ConstraintCircuit<II>) -> TokenStream {
    let CircuitExpression::BinaryOperation(BinOp::Sub, new_var, expr) = circuit.expression else {
        panic!("Substitution rule must be a subtraction.");
    };
    let CircuitExpression::Input(_) = new_var.as_ref().borrow().expression else {
        panic!("Substitution rule must be a simple substitution.");
    };

    let expr = expr.as_ref().borrow().to_owned();
    evaluate_single_node(usize::MAX, &expr, &HashSet::new())
}

fn base_single_row_substitutions(indices: &[usize], substitutions: &[TokenStream]) -> TokenStream {
    assert_eq!(indices.len(), substitutions.len());
    if indices.is_empty() {
        return quote!();
    }
    quote!(
        master_base_table.rows_mut().into_iter().for_each(|mut row| {
        #(
        let (base_row, mut det_col) =
            row.multi_slice_mut((s![..#indices],s![#indices..=#indices]));
        det_col[0] = #substitutions;
        )*
        });
    )
}

fn base_dual_row_substitutions(indices: &[usize], substitutions: &[TokenStream]) -> TokenStream {
    assert_eq!(indices.len(), substitutions.len());
    if indices.is_empty() {
        return quote!();
    }
    quote!(
        for curr_row_idx in 0..master_base_table.nrows() - 1 {
            let next_row_idx = curr_row_idx + 1;
            let (mut curr_base_row, next_base_row) = master_base_table.multi_slice_mut((
                s![curr_row_idx..=curr_row_idx, ..],
                s![next_row_idx..=next_row_idx, ..],
            ));
            let mut curr_base_row = curr_base_row.row_mut(0);
            let next_base_row = next_base_row.row(0);
            #(
            let (current_base_row, mut det_col) =
                curr_base_row.multi_slice_mut((s![..#indices], s![#indices..=#indices]));
            det_col[0] = #substitutions;
            )*
        }
    )
}

fn ext_single_row_substitutions(indices: &[usize], substitutions: &[TokenStream]) -> TokenStream {
    assert_eq!(indices.len(), substitutions.len());
    if indices.is_empty() {
        return quote!();
    }
    quote!(
        for row_idx in 0..master_base_table.nrows() - 1 {
            let base_row = master_base_table.row(row_idx);
            let mut extension_row = master_ext_table.row_mut(row_idx);
            #(
            let (ext_row, mut det_col) =
                extension_row.multi_slice_mut((s![..#indices],s![#indices..=#indices]));
            det_col[0] = #substitutions;
            )*
        }
    )
}

fn ext_dual_row_substitutions(indices: &[usize], substitutions: &[TokenStream]) -> TokenStream {
    assert_eq!(indices.len(), substitutions.len());
    if indices.is_empty() {
        return quote!();
    }
    quote!(
        for curr_row_idx in 0..master_base_table.nrows() - 1 {
            let next_row_idx = curr_row_idx + 1;
            let current_base_row = master_base_table.row(curr_row_idx);
            let next_base_row = master_base_table.row(next_row_idx);
            let (mut curr_ext_row, next_ext_row) = master_ext_table.multi_slice_mut((
                s![curr_row_idx..=curr_row_idx, ..],
                s![next_row_idx..=next_row_idx, ..],
            ));
            let mut curr_ext_row = curr_ext_row.row_mut(0);
            let next_ext_row = next_ext_row.row(0);
            #(
            let (current_ext_row, mut det_col) =
                curr_ext_row.multi_slice_mut((s![..#indices], s![#indices..=#indices]));
            det_col[0] = #substitutions;
            )*
        }
    )
}

fn write_code_to_file(code: TokenStream, file_name: &str) {
    let syntax_tree = syn::parse2(code).unwrap();
    let code = prettyplease::unparse(&syntax_tree);
    let path = format!("triton-vm/src/table/{file_name}.rs");
    write(path, code).unwrap();
}
