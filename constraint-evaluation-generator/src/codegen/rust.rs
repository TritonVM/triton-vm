use std::collections::HashSet;

use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::format_ident;
use quote::quote;

use triton_vm::table::constraint_circuit::CircuitExpression;
use triton_vm::table::constraint_circuit::ConstraintCircuit;
use triton_vm::table::constraint_circuit::InputIndicator;

use crate::codegen::Codegen;
use crate::codegen::RustBackend;
use crate::Constraints;

impl Codegen for RustBackend {
    fn constraint_evaluation_code(constraints: &Constraints) -> TokenStream {
        let num_init_constraints = constraints.init.len();
        let num_cons_constraints = constraints.cons.len();
        let num_tran_constraints = constraints.tran.len();
        let num_term_constraints = constraints.term.len();

        let (init_constraint_degrees, init_constraints_bfe, init_constraints_xfe) =
            Self::tokenize_circuits(&constraints.init());
        let (cons_constraint_degrees, cons_constraints_bfe, cons_constraints_xfe) =
            Self::tokenize_circuits(&constraints.cons());
        let (tran_constraint_degrees, tran_constraints_bfe, tran_constraints_xfe) =
            Self::tokenize_circuits(&constraints.tran());
        let (term_constraint_degrees, term_constraints_bfe, term_constraints_xfe) =
            Self::tokenize_circuits(&constraints.term());

        let uses = Self::uses();
        let evaluable_over_base_field = Self::generate_evaluable_implementation_over_field(
            &init_constraints_bfe,
            &cons_constraints_bfe,
            &tran_constraints_bfe,
            &term_constraints_bfe,
            quote!(BFieldElement),
        );
        let evaluable_over_ext_field = Self::generate_evaluable_implementation_over_field(
            &init_constraints_xfe,
            &cons_constraints_xfe,
            &tran_constraints_xfe,
            &term_constraints_xfe,
            quote!(XFieldElement),
        );

        let quotient_trait_impl = quote!(
        impl Quotientable for MasterExtTable {
            const NUM_INITIAL_CONSTRAINTS: usize = #num_init_constraints;
            const NUM_CONSISTENCY_CONSTRAINTS: usize = #num_cons_constraints;
            const NUM_TRANSITION_CONSTRAINTS: usize = #num_tran_constraints;
            const NUM_TERMINAL_CONSTRAINTS: usize = #num_term_constraints;

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
            #uses
            #evaluable_over_base_field
            #evaluable_over_ext_field
            #quotient_trait_impl
        )
    }
}

impl RustBackend {
    fn uses() -> TokenStream {
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

    /// Return a tuple of [`TokenStream`]s corresponding to code evaluating these constraints as
    /// well as their degrees. In particular:
    /// 1. The first stream contains code that, when evaluated, produces the constraints' degrees,
    /// 1. the second stream contains code that, when evaluated, produces the constraints' values,
    /// with the input type for the base row being `BFieldElement`, and
    /// 1. the third stream is like the second, except that the input type for the base row is
    ///    `XFieldElement`.
    fn tokenize_circuits<II: InputIndicator>(
        constraints: &[ConstraintCircuit<II>],
    ) -> (TokenStream, TokenStream, TokenStream) {
        if constraints.is_empty() {
            return (quote!(), quote!(vec![]), quote!(vec![]));
        }

        let ref_counters = constraints
            .iter()
            .flat_map(ConstraintCircuit::all_ref_counters)
            .sorted()
            .unique();

        // Declare all shared variables, i.e., those with a ref count greater than 1.
        // These declarations must be made starting from the highest ref count.
        // Otherwise, the resulting code will refer to bindings that have not yet been made.
        let shared_declarations = ref_counters
            .filter(|&x| x > 1)
            .rev()
            .map(|count| Self::declare_nodes_with_ref_count(count, constraints))
            .collect_vec();

        let (base_constraints, ext_constraints): (Vec<_>, Vec<_>) = constraints
            .iter()
            .partition(|constraint| constraint.evaluates_to_base_element());

        // The order of the constraints' degrees must match the order of the constraints.
        // Hence, listing the degrees is only possible after the partition into base and extension
        // constraints is known.
        let tokenized_degree_bounds = base_constraints
            .iter()
            .chain(&ext_constraints)
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
                .map(|constraint| Self::evaluate_single_node(1, constraint, &HashSet::default()))
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

    /// Produce the code to evaluate code for all nodes that share a ref count. A value for all
    /// nodes with a higher count (i.e., deeper in the tree) than the provided are assumed to be in
    /// scope.
    fn declare_nodes_with_ref_count<II: InputIndicator>(
        ref_count: usize,
        circuits: &[ConstraintCircuit<II>],
    ) -> TokenStream {
        let mut scope: HashSet<usize> = HashSet::new();
        let all_nodes_in_circuit =
            |circuit| Self::declare_single_node_with_ref_count(circuit, ref_count, &mut scope);
        let tokenized_circuits = circuits.iter().filter_map(all_nodes_in_circuit);
        quote!(#(#tokenized_circuits)*)
    }

    fn declare_single_node_with_ref_count<II: InputIndicator>(
        circuit: &ConstraintCircuit<II>,
        ref_count: usize,
        scope: &mut HashSet<usize>,
    ) -> Option<TokenStream> {
        // Don't declare a node twice.
        if scope.contains(&circuit.id) {
            return None;
        }

        // A higher-than-requested ref count means the node is already in global scope, albeit not
        // necessarily in the passed-in scope.
        if circuit.ref_count > ref_count {
            return None;
        }

        // Constants are already (or can be) trivially declared.
        let CircuitExpression::BinaryOperation(_, lhs, rhs) = &circuit.expression else {
            return None;
        };

        // If the ref count is not exact, recurse on the BinaryOperation's children.
        if circuit.ref_count < ref_count {
            let out_left =
                Self::declare_single_node_with_ref_count(&lhs.as_ref().borrow(), ref_count, scope);
            let out_right =
                Self::declare_single_node_with_ref_count(&rhs.as_ref().borrow(), ref_count, scope);
            return match (out_left, out_right) {
                (None, None) => None,
                (Some(l), None) => Some(l),
                (None, Some(r)) => Some(r),
                (Some(l), Some(r)) => Some(quote!(#l #r)),
            };
        }

        assert_eq!(circuit.ref_count, ref_count);
        let binding_name = Self::binding_name(circuit);
        let evaluation = Self::evaluate_single_node(ref_count, circuit, scope);
        let new_binding = quote!(let #binding_name = #evaluation;);

        let is_new_insertion = scope.insert(circuit.id);
        assert!(is_new_insertion);

        Some(new_binding)
    }

    /// Recursively construct the code for evaluating a single node.
    pub fn evaluate_single_node<II: InputIndicator>(
        ref_count: usize,
        circuit: &ConstraintCircuit<II>,
        scope: &HashSet<usize>,
    ) -> TokenStream {
        // Don't declare a node twice.
        if scope.contains(&circuit.id) {
            return Self::binding_name(circuit);
        }

        // The binding must already be known.
        if circuit.ref_count > ref_count {
            return Self::binding_name(circuit);
        }

        // Constants have trivial bindings.
        let CircuitExpression::BinaryOperation(binop, lhs, rhs) = &circuit.expression else {
            return Self::binding_name(circuit);
        };

        let lhs = lhs.as_ref().borrow();
        let rhs = rhs.as_ref().borrow();
        let lhs = Self::evaluate_single_node(ref_count, &lhs, scope);
        let rhs = Self::evaluate_single_node(ref_count, &rhs, scope);
        quote!((#lhs) #binop (#rhs))
    }

    fn binding_name<II: InputIndicator>(circuit: &ConstraintCircuit<II>) -> TokenStream {
        match &circuit.expression {
            CircuitExpression::BConstant(bfe) => Self::tokenize_bfe(*bfe),
            CircuitExpression::XConstant(xfe) => Self::tokenize_xfe(*xfe),
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
}
