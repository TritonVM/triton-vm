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
            fn initial_quotient_degree_bounds(interpolant_degree: isize) -> Vec<isize> {
                let zerofier_degree = 1;
                [#init_constraint_degrees].to_vec()
            }

            #[allow(unused_variables)]
            fn consistency_quotient_degree_bounds(
                interpolant_degree: isize,
                padded_height: usize,
            ) -> Vec<isize> {
                let zerofier_degree = padded_height as isize;
                [#cons_constraint_degrees].to_vec()
            }

            #[allow(unused_variables)]
            fn transition_quotient_degree_bounds(
                interpolant_degree: isize,
                padded_height: usize,
            ) -> Vec<isize> {
                let zerofier_degree = padded_height as isize - 1;
                [#tran_constraint_degrees].to_vec()
            }

            #[allow(unused_variables)]
            fn terminal_quotient_degree_bounds(interpolant_degree: isize) -> Vec<isize> {
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

            use crate::table::challenges::Challenges;
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

        let mut backend = Self::default();
        let shared_declarations = backend.declare_shared_nodes(constraints);
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
                .map(|constraint| backend.evaluate_single_node(constraint))
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

    /// Declare all shared variables, i.e., those with a ref count greater than 1.
    /// These declarations must be made starting from the highest ref count.
    /// Otherwise, the resulting code will refer to bindings that have not yet been made.
    fn declare_shared_nodes<II: InputIndicator>(
        &mut self,
        constraints: &[ConstraintCircuit<II>],
    ) -> Vec<TokenStream> {
        let constraints_iter = constraints.iter();
        let all_ref_counts = constraints_iter.flat_map(ConstraintCircuit::all_ref_counters);
        let relevant_ref_counts = all_ref_counts.unique().filter(|&x| x > 1);
        let ordered_ref_counts = relevant_ref_counts.sorted().rev();

        ordered_ref_counts
            .map(|count| self.declare_nodes_with_ref_count(constraints, count))
            .collect()
    }

    /// Produce the code to evaluate code for all nodes that share a ref count.
    fn declare_nodes_with_ref_count<II: InputIndicator>(
        &mut self,
        circuits: &[ConstraintCircuit<II>],
        ref_count: usize,
    ) -> TokenStream {
        let all_nodes_in_circuit =
            |circuit| self.declare_single_node_with_ref_count(circuit, ref_count);
        let tokenized_circuits = circuits.iter().filter_map(all_nodes_in_circuit);
        quote!(#(#tokenized_circuits)*)
    }

    fn declare_single_node_with_ref_count<II: InputIndicator>(
        &mut self,
        circuit: &ConstraintCircuit<II>,
        ref_count: usize,
    ) -> Option<TokenStream> {
        if self.scope.contains(&circuit.id) {
            return None;
        }

        // constants can be declared trivially
        let CircuitExpression::BinaryOperation(_, lhs, rhs) = &circuit.expression else {
            return None;
        };

        if circuit.ref_count < ref_count {
            let out_left = self.declare_single_node_with_ref_count(&lhs.borrow(), ref_count);
            let out_right = self.declare_single_node_with_ref_count(&rhs.borrow(), ref_count);
            return match (out_left, out_right) {
                (None, None) => None,
                (Some(l), None) => Some(l),
                (None, Some(r)) => Some(r),
                (Some(l), Some(r)) => Some(quote!(#l #r)),
            };
        }

        assert_eq!(circuit.ref_count, ref_count);
        let binding_name = Self::binding_name(circuit);
        let evaluation = self.evaluate_single_node(circuit);
        let new_binding = quote!(let #binding_name = #evaluation;);

        let is_new_insertion = self.scope.insert(circuit.id);
        assert!(is_new_insertion);

        Some(new_binding)
    }

    /// Recursively construct the code for evaluating a single node.
    pub fn evaluate_single_node<II: InputIndicator>(
        &self,
        circuit: &ConstraintCircuit<II>,
    ) -> TokenStream {
        if self.scope.contains(&circuit.id) {
            return Self::binding_name(circuit);
        }

        let CircuitExpression::BinaryOperation(binop, lhs, rhs) = &circuit.expression else {
            return Self::binding_name(circuit);
        };

        let lhs = self.evaluate_single_node(&lhs.borrow());
        let rhs = self.evaluate_single_node(&rhs.borrow());
        quote!((#lhs) #binop (#rhs))
    }

    fn binding_name<II: InputIndicator>(circuit: &ConstraintCircuit<II>) -> TokenStream {
        match &circuit.expression {
            CircuitExpression::BConstant(bfe) => Self::tokenize_bfe(*bfe),
            CircuitExpression::XConstant(xfe) => Self::tokenize_xfe(*xfe),
            CircuitExpression::Input(idx) => quote!(#idx),
            CircuitExpression::Challenge(challenge) => quote!(challenges[#challenge]),
            CircuitExpression::BinaryOperation(_, _, _) => {
                let node_ident = format_ident!("node_{}", circuit.id);
                quote!(#node_ident)
            }
        }
    }
}
