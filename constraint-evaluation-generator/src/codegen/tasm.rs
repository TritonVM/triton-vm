use std::collections::HashSet;

use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::quote;
use quote::ToTokens;
use twenty_first::bfe;
use twenty_first::prelude::x_field_element::EXTENSION_DEGREE;
use twenty_first::prelude::BFieldElement;
use twenty_first::prelude::XFieldElement;

use triton_vm::table::challenges::ChallengeId;
use triton_vm::table::constraint_circuit::BinOp;
use triton_vm::table::constraint_circuit::CircuitExpression;
use triton_vm::table::constraint_circuit::ConstraintCircuit;
use triton_vm::table::constraint_circuit::InputIndicator;
use triton_vm::table::TasmConstraintEvaluationMemoryLayout;

use crate::codegen::Codegen;
use crate::codegen::TasmBackend;
use crate::constraints::Constraints;

/// An offset from the [memory layout][layout]'s `free_mem_page_ptr`, in number of
/// [`XFieldElement`]s. Indicates the start of the to-be-returned array.
///
/// [layout]: TasmConstraintEvaluationMemoryLayout
const OUT_ARRAY_OFFSET: usize = {
    let mem_page_size = TasmConstraintEvaluationMemoryLayout::MEM_PAGE_SIZE;
    let max_num_words_for_evaluated_constraints = 1 << 16; // magic!
    let out_array_offset_in_words = mem_page_size - max_num_words_for_evaluated_constraints;
    assert!(out_array_offset_in_words % EXTENSION_DEGREE == 0);
    out_array_offset_in_words / EXTENSION_DEGREE
};

impl Codegen for TasmBackend {
    /// Emits a function that emits [Triton assembly][tasm] that evaluates Triton VM's AIR
    /// constraints over the [extension field][XFieldElement].
    ///
    /// [tasm]: triton_vm::prelude::triton_asm
    fn constraint_evaluation_code(constraints: &Constraints) -> TokenStream {
        let uses = Self::uses();
        let doc_comment = Self::doc_comment(constraints);

        let mut backend = Self::new();
        let init_constraints = backend.tokenize_circuits(&constraints.init());
        let cons_constraints = backend.tokenize_circuits(&constraints.cons());
        let tran_constraints = backend.tokenize_circuits(&constraints.tran());
        let term_constraints = backend.tokenize_circuits(&constraints.term());

        let prepare_return_values = Self::prepare_return_values();

        quote!(
            #uses
            #[doc = #doc_comment]
            pub fn air_constraint_evaluation_tasm(
                mem_layout: TasmConstraintEvaluationMemoryLayout,
            ) -> Vec<LabelledInstruction> {
                let instructions = vec![
                    #init_constraints
                    #cons_constraints
                    #tran_constraints
                    #term_constraints
                    #prepare_return_values
                ];
                instructions.into_iter().map(LabelledInstruction::Instruction).collect()
            }
        )
    }
}

impl TasmBackend {
    fn new() -> Self {
        Self {
            scope: HashSet::new(),
            elements_written: 0,
        }
    }

    fn uses() -> TokenStream {
        quote!(
            use twenty_first::prelude::BFieldElement;
            use crate::instruction::AnInstruction;
            use crate::instruction::LabelledInstruction;
            use crate::op_stack::NumberOfWords;
            use crate::table::TasmConstraintEvaluationMemoryLayout;
        )
    }

    fn doc_comment(constraints: &Constraints) -> String {
        let num_init_constraints = constraints.init.len();
        let num_cons_constraints = constraints.cons.len();
        let num_tran_constraints = constraints.tran.len();
        let num_term_constraints = constraints.term.len();
        let num_total_constraints = constraints.len();

        format!(
            "
        The emitted Triton assembly has the following signature:

        # Signature

        ```text
        BEFORE: _
        AFTER:  _ *evaluated_constraints
        ```
        # Requirements

        In order for this method to emit Triton assembly, various memory regions need to be
        declared. This is done through [`TasmConstraintEvaluationMemoryLayout`]. The memory
        layout must be [integral].

        # Guarantees

        - The emitted code does not declare any labels.
        - The emitted code is “straight-line”, _i.e._, does not contain any of the instructions
            `call`, `return`, `recurse`, or `skiz`.
        - The emitted code does not contain instruction `halt`.
        - All memory write access of the emitted code is within the bounds of the memory region
            pointed to by `*free_memory_page`.
        - `*evaluated_constraints` points to an array of [`XFieldElement`][xfe]s of length \
            {num_total_constraints}. Of these,
            - the first {num_init_constraints} elements are the evaluated initial constraints,
            - the next {num_cons_constraints} elements are the evaluated consistency constraints,
            - the next {num_tran_constraints} elements are the evaluated transition constraints,
            - the last {num_term_constraints} elements are the evaluated terminal constraints.

            Above constants can be accessed programmatically through the methods
            [`num_quotients()`][num_quotients], as well as `num_initial_quotients()`,
            `num_consistency_quotients()`, `num_transition_quotients()`, and
            `num_terminal_quotients()` on the [`MasterExtTable`][master_ext_table].

        [integral]: crate::table::TasmConstraintEvaluationMemoryLayout::is_integral
        [bfe]: crate::prelude::BFieldElement
        [xfe]: crate::prelude::XFieldElement
        [num_base_cols]: crate::table::NUM_BASE_COLUMNS
        [num_ext_cols]: crate::table::NUM_EXT_COLUMNS
        [num_challenges]: crate::table::challenges::Challenges::count()
        [num_quotients]: crate::table::master_table::num_quotients
        [master_ext_table]: crate::table::master_table::MasterExtTable
            "
        )
    }

    fn tokenize_circuits<II: InputIndicator>(
        &mut self,
        constraints: &[ConstraintCircuit<II>],
    ) -> TokenStream {
        if constraints.is_empty() {
            return quote!();
        }

        self.reset_scope();
        let store_shared_nodes = self.store_all_shared_nodes(constraints);

        // to match the `RustBackend`, base constraints must be emitted first
        let (base_constraints, ext_constraints): (Vec<_>, Vec<_>) = constraints
            .iter()
            .partition(|constraint| constraint.evaluates_to_base_element());
        let sorted_constraints = base_constraints.into_iter().chain(ext_constraints);
        let write_to_output =
            sorted_constraints.map(|c| self.write_evaluated_constraint_into_output_list(c));

        quote!(#(#store_shared_nodes)* #(#write_to_output)*)
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
            .map(|c| self.store_single_shared_node_of_ref_count(c, count))
            .collect()
    }

    fn store_single_shared_node_of_ref_count<II: InputIndicator>(
        &mut self,
        constraint: &ConstraintCircuit<II>,
        ref_count: usize,
    ) -> TokenStream {
        if self.scope.contains(&constraint.id) {
            return quote!();
        }

        let CircuitExpression::BinaryOperation(_, lhs, rhs) = &constraint.expression else {
            return quote!();
        };

        assert!(
            constraint.ref_count <= ref_count,
            "Constraints with ref count greater than {ref_count} must be in scope."
        );

        if constraint.ref_count < ref_count {
            let out_left = self.store_single_shared_node_of_ref_count(&lhs.borrow(), ref_count);
            let out_right = self.store_single_shared_node_of_ref_count(&rhs.borrow(), ref_count);
            return quote!(#out_left #out_right);
        }

        let evaluate = self.evaluate_single_node(constraint);
        let store = Self::store_ext_field_element(constraint.id);
        let is_new_insertion = self.scope.insert(constraint.id);
        assert!(is_new_insertion);

        quote!(#evaluate #store)
    }

    fn evaluate_single_node<II: InputIndicator>(
        &self,
        constraint: &ConstraintCircuit<II>,
    ) -> TokenStream {
        if self.scope.contains(&constraint.id) {
            return Self::load_node(constraint);
        }

        let CircuitExpression::BinaryOperation(binop, lhs, rhs) = &constraint.expression else {
            return Self::load_node(constraint);
        };

        let lhs = self.evaluate_single_node(&lhs.borrow());
        let rhs = self.evaluate_single_node(&rhs.borrow());
        let binop = Self::tokenize_bin_op(*binop);
        quote!(#lhs #rhs #binop)
    }

    fn write_evaluated_constraint_into_output_list<II: InputIndicator>(
        &mut self,
        constraint: &ConstraintCircuit<II>,
    ) -> TokenStream {
        let evaluated_constraint = self.evaluate_single_node(constraint);
        let element_index = OUT_ARRAY_OFFSET + self.elements_written;
        let store_element = Self::store_ext_field_element(element_index);
        self.elements_written += 1;
        quote!(#evaluated_constraint #store_element)
    }

    fn load_node<II: InputIndicator>(circuit: &ConstraintCircuit<II>) -> TokenStream {
        match circuit.expression {
            CircuitExpression::BConstant(bfe) => Self::load_base_field_constant(bfe),
            CircuitExpression::XConstant(xfe) => Self::load_ext_field_constant(xfe),
            CircuitExpression::Input(input) => Self::load_input(input),
            CircuitExpression::Challenge(challenge) => Self::load_challenge(challenge),
            CircuitExpression::BinaryOperation(_, _, _) => Self::load_evaluated_bin_op(circuit.id),
        }
    }

    fn load_base_field_constant(bfe: BFieldElement) -> TokenStream {
        let zero = Self::tokenize_bfe(bfe!(0));
        let bfe = Self::tokenize_bfe(bfe);
        quote!(AnInstruction::Push(#zero), AnInstruction::Push(#zero), AnInstruction::Push(#bfe),)
    }

    fn load_ext_field_constant(xfe: XFieldElement) -> TokenStream {
        let [c0, c1, c2] = xfe.coefficients.map(Self::tokenize_bfe);
        quote!(AnInstruction::Push(#c2), AnInstruction::Push(#c1), AnInstruction::Push(#c0),)
    }

    fn load_input<II: InputIndicator>(input: II) -> TokenStream {
        let list = match (input.is_current_row(), input.is_base_table_column()) {
            (true, true) => IOList::CurrBaseRow,
            (true, false) => IOList::CurrExtRow,
            (false, true) => IOList::NextBaseRow,
            (false, false) => IOList::NextExtRow,
        };
        Self::load_ext_field_element_from_list(list, input.column())
    }

    fn load_challenge(challenge: ChallengeId) -> TokenStream {
        Self::load_ext_field_element_from_list(IOList::Challenges, challenge.index())
    }

    fn load_evaluated_bin_op(node_id: usize) -> TokenStream {
        Self::load_ext_field_element_from_list(IOList::FreeMemPage, node_id)
    }

    fn load_ext_field_element_from_list(list: IOList, element_index: usize) -> TokenStream {
        let word_offset = element_index * EXTENSION_DEGREE;
        let start_to_read_offset = EXTENSION_DEGREE - 1;
        let word_index = word_offset + start_to_read_offset;
        let word_index = bfe!(word_index as u64);
        let word_index = Self::tokenize_bfe(word_index);

        quote!(
            AnInstruction::Push(#list + #word_index),
            AnInstruction::ReadMem(NumberOfWords::N3),
            AnInstruction::Pop(NumberOfWords::N1),
        )
    }

    fn store_ext_field_element(element_index: usize) -> TokenStream {
        let free_mem_page = IOList::FreeMemPage;

        let word_offset = element_index * EXTENSION_DEGREE;
        let word_index = bfe!(word_offset as u64);
        let word_index = Self::tokenize_bfe(word_index);

        quote!(
            AnInstruction::Push(#free_mem_page + #word_index),
            AnInstruction::WriteMem(NumberOfWords::N3),
            AnInstruction::Pop(NumberOfWords::N1),
        )
    }

    fn tokenize_bin_op(binop: BinOp) -> TokenStream {
        match binop {
            BinOp::Add => quote!(AnInstruction::XxAdd,),
            BinOp::Mul => quote!(AnInstruction::XxMul,),
        }
    }

    fn prepare_return_values() -> TokenStream {
        let free_mem_page = IOList::FreeMemPage;
        let out_array_offset_in_num_bfes = OUT_ARRAY_OFFSET * EXTENSION_DEGREE;
        let out_array_offset = bfe!(out_array_offset_in_num_bfes as u64);
        let out_array_offset = Self::tokenize_bfe(out_array_offset);
        quote!(AnInstruction::Push(#free_mem_page + #out_array_offset),)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
enum IOList {
    FreeMemPage,
    CurrBaseRow,
    CurrExtRow,
    NextBaseRow,
    NextExtRow,
    Challenges,
}

impl ToTokens for IOList {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend(quote!(mem_layout.));
        match self {
            IOList::FreeMemPage => tokens.extend(quote!(free_mem_page_ptr)),
            IOList::CurrBaseRow => tokens.extend(quote!(curr_base_row_ptr)),
            IOList::CurrExtRow => tokens.extend(quote!(curr_ext_row_ptr)),
            IOList::NextBaseRow => tokens.extend(quote!(next_base_row_ptr)),
            IOList::NextExtRow => tokens.extend(quote!(next_ext_row_ptr)),
            IOList::Challenges => tokens.extend(quote!(challenges_ptr)),
        }
    }
}

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
