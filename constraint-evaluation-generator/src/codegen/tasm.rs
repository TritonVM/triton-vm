use std::collections::HashSet;

use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::quote;
use quote::ToTokens;
use twenty_first::prelude::XFieldElement;

use triton_vm::table::constraint_circuit::BinOp;
use triton_vm::table::constraint_circuit::CircuitExpression;
use triton_vm::table::constraint_circuit::ConstraintCircuit;
use triton_vm::table::constraint_circuit::InputIndicator;

use crate::codegen::Codegen;
use crate::codegen::TasmBackend;
use crate::constraints::Constraints;

impl Codegen for TasmBackend {
    /// Emits a function that emits [Triton assembly][tasm] that evaluates Triton VM's AIR
    /// constraints over the [extension field][XFieldElement].
    ///
    /// [tasm]: triton_vm::prelude::triton_asm
    fn constraint_evaluation_code(constraints: &Constraints) -> TokenStream {
        let uses = Self::uses();

        let mut backend = Self::new();
        let init_constraints = backend.tokenize_circuits(&constraints.init());
        let cons_constraints = backend.tokenize_circuits(&constraints.cons());
        let tran_constraints = backend.tokenize_circuits(&constraints.tran());
        let term_constraints = backend.tokenize_circuits(&constraints.term());

        quote!(
            #uses
            impl TasmConstraintInstantiator {
                pub(super) fn instantiate_initial_constraints(
                    &mut self
                ) -> Vec<LabelledInstruction> {
                    [#(#init_constraints,)*].concat()
                }

                pub(super) fn instantiate_consistency_constraints(
                    &mut self
                ) -> Vec<LabelledInstruction> {
                    [#(#cons_constraints,)*].concat()
                }

                pub(super) fn instantiate_transition_constraints(
                    &mut self
                ) -> Vec<LabelledInstruction> {
                    [#(#tran_constraints,)*].concat()
                }

                pub(super) fn instantiate_terminal_constraints(
                    &mut self
                ) -> Vec<LabelledInstruction> {
                    [#(#term_constraints,)*].concat()
                }
            }
        )
    }
}

impl TasmBackend {
    fn new() -> Self {
        let scope = HashSet::new();
        Self { scope }
    }

    fn uses() -> TokenStream {
        quote!(
            use twenty_first::prelude::*;
            use crate::instruction::AnInstruction;
            use crate::instruction::LabelledInstruction;
            use crate::table::IOList;
            use crate::table::TasmConstraintInstantiator;
        )
    }

    fn tokenize_circuits<II: InputIndicator>(
        &mut self,
        constraints: &[ConstraintCircuit<II>],
    ) -> Vec<TokenStream> {
        self.reset_scope();
        let store_shared_nodes = self.store_all_shared_nodes(constraints);

        // to match the `RustBackend`, base constraints must be emitted first
        let (base_constraints, ext_constraints): (Vec<_>, Vec<_>) = constraints
            .iter()
            .partition(|constraint| constraint.evaluates_to_base_element());
        let sorted_constraints = base_constraints.into_iter().chain(ext_constraints);
        let write_to_output = sorted_constraints
            .flat_map(|c| self.write_evaluated_constraint_into_output_list(c))
            .collect_vec();

        [store_shared_nodes, write_to_output].concat()
    }

    fn reset_scope(&mut self) {
        self.scope = HashSet::new();
    }

    fn store_all_shared_nodes<II: InputIndicator>(
        &mut self,
        constraints: &[ConstraintCircuit<II>],
    ) -> Vec<TokenStream> {
        let ref_counts = constraints.iter().flat_map(|c| c.all_ref_counters());
        let relevant_ref_counts = ref_counts.sorted().unique().filter(|&c| c > 1).rev();
        relevant_ref_counts
            .map(|count| self.store_all_shared_nodes_of_ref_count(constraints, count))
            .concat()
    }

    fn store_all_shared_nodes_of_ref_count<II: InputIndicator>(
        &mut self,
        constraints: &[ConstraintCircuit<II>],
        count: usize,
    ) -> Vec<TokenStream> {
        constraints
            .iter()
            .flat_map(|c| self.store_single_shared_node_of_ref_count(c, count))
            .collect()
    }

    fn store_single_shared_node_of_ref_count<II: InputIndicator>(
        &mut self,
        constraint: &ConstraintCircuit<II>,
        ref_count: usize,
    ) -> Vec<TokenStream> {
        if self.scope.contains(&constraint.id) {
            return vec![];
        }

        let CircuitExpression::BinaryOperation(_, lhs, rhs) = &constraint.expression else {
            return vec![];
        };

        if constraint.ref_count < ref_count {
            let out_left = self.store_single_shared_node_of_ref_count(&lhs.borrow(), ref_count);
            let out_right = self.store_single_shared_node_of_ref_count(&rhs.borrow(), ref_count);
            return [out_left, out_right].concat();
        }

        assert_eq!(constraint.ref_count, ref_count);
        let evaluate = self.evaluate_single_node(constraint);
        let store = Self::store_ext_field_element(constraint.id);
        let is_new_insertion = self.scope.insert(constraint.id);
        assert!(is_new_insertion);

        [evaluate, store].concat()
    }

    fn evaluate_single_node<II: InputIndicator>(
        &self,
        constraint: &ConstraintCircuit<II>,
    ) -> Vec<TokenStream> {
        if self.scope.contains(&constraint.id) {
            return Self::load_node(constraint);
        }

        let CircuitExpression::BinaryOperation(binop, lhs, rhs) = &constraint.expression else {
            return Self::load_node(constraint);
        };

        let lhs = self.evaluate_single_node(&lhs.borrow());
        let rhs = self.evaluate_single_node(&rhs.borrow());
        let binop = Self::tokenize_bin_op(*binop);
        [lhs, rhs, binop].concat()
    }

    fn tokenize_bin_op(binop: BinOp) -> Vec<TokenStream> {
        let binop = match binop {
            BinOp::Add => quote!(AnInstruction::XxAdd),
            BinOp::Mul => quote!(AnInstruction::XxMul),
        };
        vec![quote!(vec![LabelledInstruction::Instruction(#binop)])]
    }

    fn write_evaluated_constraint_into_output_list<II: InputIndicator>(
        &self,
        constraint: &ConstraintCircuit<II>,
    ) -> Vec<TokenStream> {
        let mut evaluated_constraint = self.evaluate_single_node(constraint);
        evaluated_constraint.push(quote!(self.write_into_output_list()));
        evaluated_constraint
    }

    fn load_node<II: InputIndicator>(circuit: &ConstraintCircuit<II>) -> Vec<TokenStream> {
        match circuit.expression {
            CircuitExpression::BConstant(bfe) => Self::load_ext_field_constant(bfe.into()),
            CircuitExpression::XConstant(xfe) => Self::load_ext_field_constant(xfe),
            CircuitExpression::Input(input) => Self::load_input(input),
            CircuitExpression::Challenge(challenge_idx) => Self::load_challenge(challenge_idx),
            CircuitExpression::BinaryOperation(_, _, _) => Self::load_evaluated_bin_op(circuit.id),
        }
    }

    fn load_ext_field_constant(xfe: XFieldElement) -> Vec<TokenStream> {
        let xfe = Self::tokenize_xfe(xfe);
        vec![quote!(Self::load_ext_field_constant(#xfe))]
    }

    fn load_input<II: InputIndicator>(input: II) -> Vec<TokenStream> {
        let list = match (input.is_current_row(), input.is_base_table_column()) {
            (true, true) => IOList::CurrBaseRow,
            (true, false) => IOList::CurrExtRow,
            (false, true) => IOList::NextBaseRow,
            (false, false) => IOList::NextExtRow,
        };
        Self::load_ext_field_element_from_list(list, input.column())
    }

    fn load_challenge(challenge_idx: usize) -> Vec<TokenStream> {
        Self::load_ext_field_element_from_list(IOList::Challenges, challenge_idx)
    }

    fn load_evaluated_bin_op(node_id: usize) -> Vec<TokenStream> {
        Self::load_ext_field_element_from_list(IOList::FreeMemPage, node_id)
    }

    fn load_ext_field_element_from_list(list: IOList, element_index: usize) -> Vec<TokenStream> {
        vec![quote!(self.load_ext_field_element_from_list(#list, #element_index))]
    }

    fn store_ext_field_element(element_index: usize) -> Vec<TokenStream> {
        vec![quote!(self.store_ext_field_element(#element_index))]
    }
}

macro_rules! io_list {
    ($($variant:ident),*) => {
        #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
        enum IOList {
            $($variant,)*
        }

        impl ToTokens for IOList {
            fn to_tokens(&self, tokens: &mut TokenStream) {
                match self {
                    $(Self::$variant => tokens.extend(quote!(IOList::$variant)),)*
                }
            }
        }
    };
}

io_list!(
    FreeMemPage,
    CurrBaseRow,
    CurrExtRow,
    NextBaseRow,
    NextExtRow,
    Challenges
);

#[cfg(test)]
mod tests {
    use super::*;

    fn print_constraints_as_tasm(constraints: &Constraints) {
        let tasm = TasmBackend::constraint_evaluation_code(constraints);
        let syntax_tree = syn::parse2(tasm).unwrap();
        let code = prettyplease::unparse(&syntax_tree);
        println!("{code}");
    }

    #[test]
    fn print_mini_constraints_as_tasm() {
        print_constraints_as_tasm(&Constraints::mini_constraints());
    }

    #[test]
    fn print_test_constraints_as_tasm() {
        print_constraints_as_tasm(&Constraints::test_constraints());
    }
}
