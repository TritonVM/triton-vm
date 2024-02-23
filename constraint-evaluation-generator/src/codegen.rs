use std::collections::HashSet;

use proc_macro2::TokenStream;
use quote::quote;
use twenty_first::prelude::BFieldElement;
use twenty_first::prelude::XFieldElement;

use triton_vm::table::constraint_circuit::BinOp;
use triton_vm::table::constraint_circuit::CircuitExpression;
use triton_vm::table::constraint_circuit::ConstraintCircuit;
use triton_vm::table::constraint_circuit::InputIndicator;

use crate::constraints::Constraints;

mod rust;
mod tasm;

pub(crate) trait Codegen {
    fn constraint_evaluation_code(constraints: &Constraints) -> TokenStream;

    /// Produce the code to evaluate code for all nodes that share a value number of
    /// times visited. A value for all nodes with a higher count (i.e., deeper in the tree) than the
    /// provided are assumed to be in scope.
    fn declare_nodes_with_visit_count<II: InputIndicator>(
        visit_count: usize,
        circuits: &[ConstraintCircuit<II>],
    ) -> TokenStream {
        let mut scope: HashSet<usize> = HashSet::new();
        let all_nodes_in_circuit =
            |circuit| Self::declare_single_node_with_visit_count(circuit, visit_count, &mut scope);
        let tokenized_circuits = circuits.iter().filter_map(all_nodes_in_circuit);
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

        // A higher-than-requested visit counter means the node is already in global scope, albeit
        // not necessarily in the passed-in scope.
        if circuit.visited_counter > requested_visited_count {
            return None;
        }

        // Constants are already (or can be) trivially declared.
        let CircuitExpression::BinaryOperation(_, lhs, rhs) = &circuit.expression else {
            return None;
        };

        // If the visited counter is not exact, recurse on the BinaryOperation's children.
        if circuit.visited_counter < requested_visited_count {
            let out_left = Self::declare_single_node_with_visit_count(
                &lhs.as_ref().borrow(),
                requested_visited_count,
                scope,
            );
            let out_right = Self::declare_single_node_with_visit_count(
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

        let new_binding = Self::declare_new_binding(circuit, requested_visited_count, scope);
        Some(new_binding)
    }

    /// Recursively construct the code for evaluating a single node.
    fn evaluate_single_node<II: InputIndicator>(
        requested_visited_count: usize,
        circuit: &ConstraintCircuit<II>,
        scope: &HashSet<usize>,
    ) -> TokenStream {
        // Don't declare a node twice.
        if scope.contains(&circuit.id) {
            return Self::load_node(circuit);
        }

        // The binding must already be known.
        if circuit.visited_counter > requested_visited_count {
            return Self::load_node(circuit);
        }

        // Constants have trivial bindings.
        let CircuitExpression::BinaryOperation(binop, lhs, rhs) = &circuit.expression else {
            return Self::load_node(circuit);
        };

        let lhs = lhs.as_ref().borrow();
        let rhs = rhs.as_ref().borrow();
        let evaluated_lhs = Self::evaluate_single_node(requested_visited_count, &lhs, scope);
        let evaluated_rhs = Self::evaluate_single_node(requested_visited_count, &rhs, scope);
        Self::perform_bin_op(*binop, evaluated_lhs, evaluated_rhs)
    }

    fn declare_new_binding<II: InputIndicator>(
        circuit: &ConstraintCircuit<II>,
        requested_visited_count: usize,
        scope: &mut HashSet<usize>,
    ) -> TokenStream;

    /// Return a variable name for the node. Returns `point[n]` if node is just
    /// a value from the codewords. Otherwise, returns the ID of the circuit.
    fn load_node<II: InputIndicator>(circuit: &ConstraintCircuit<II>) -> TokenStream;

    fn perform_bin_op(binop: BinOp, lhs: TokenStream, rhs: TokenStream) -> TokenStream;
}

fn tokenize_bfe(bfe: BFieldElement) -> TokenStream {
    let raw_u64 = bfe.raw_u64();
    quote!(BFieldElement::from_raw_u64(#raw_u64))
}

fn tokenize_xfe(xfe: XFieldElement) -> TokenStream {
    let [c_0, c_1, c_2] = xfe.coefficients.map(tokenize_bfe);
    quote!(XFieldElement::new([#c_0, #c_1, #c_2]))
}

pub(crate) struct RustBackend;

pub(crate) struct TasmBackend;
