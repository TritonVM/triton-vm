use std::collections::HashSet;

use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::format_ident;
use quote::quote;
use twenty_first::prelude::x_field_element::EXTENSION_DEGREE;
use twenty_first::prelude::BFieldElement;
use twenty_first::prelude::XFieldElement;

use triton_vm::op_stack::OpStackElement;
use triton_vm::table::challenges::ChallengeId;
use triton_vm::table::constraint_circuit::BinOp;
use triton_vm::table::constraint_circuit::CircuitExpression;
use triton_vm::table::constraint_circuit::ConstraintCircuit;
use triton_vm::table::constraint_circuit::InputIndicator;

use crate::codegen;
use crate::codegen::Codegen;
use crate::codegen::TasmBackend;
use crate::constraints::Constraints;

// See content of [`TasmBackend::doc_comment`]
const ARG_POS_FREE_MEM_PAGE: OpStackElement = OpStackElement::ST0;
const ARG_POS_CURR_BASE_ROW: OpStackElement = OpStackElement::ST1;
const ARG_POS_CURR_EXT_ROW: OpStackElement = OpStackElement::ST2;
const ARG_POS_NEXT_BASE_ROW: OpStackElement = OpStackElement::ST3;
const ARG_POS_NEXT_EXT_ROW: OpStackElement = OpStackElement::ST4;
const ARG_POS_CHALLENGES: OpStackElement = OpStackElement::ST5;

impl Codegen for TasmBackend {
    /// Emits a function that emits [Triton assembly][tasm] that evaluates Triton VM's AIR
    /// constraints over the [extension field][XFieldElement].
    ///
    /// [tasm]: triton_vm::prelude::triton_asm
    fn constraint_evaluation_code(constraints: &Constraints) -> TokenStream {
        let uses = Self::uses();
        let doc_comment = Self::doc_comment(constraints);

        let init_constraints = Self::tokenize_circuits(&constraints.init());
        let cons_constraints = Self::tokenize_circuits(&constraints.cons());
        let tran_constraints = Self::tokenize_circuits(&constraints.tran());
        let term_constraints = Self::tokenize_circuits(&constraints.term());

        quote!(
            #uses
            #[doc = #doc_comment]
            pub fn air_constraint_evaluation_tasm() -> Box<[LabelledInstruction]> {
                Box::new([
                    #init_constraints
                    #cons_constraints
                    #tran_constraints
                    #term_constraints
                ])
            }
        )
    }

    fn declare_new_binding<II: InputIndicator>(
        circuit: &ConstraintCircuit<II>,
        requested_visited_count: usize,
        scope: &mut HashSet<usize>,
    ) -> TokenStream {
        assert_eq!(circuit.visited_counter, requested_visited_count);
        let evaluate_node_instructions =
            Self::evaluate_single_node(requested_visited_count, circuit, scope);
        let storage_instructions =
            Self::store_ext_field_element_in_list(circuit.id, ARG_POS_FREE_MEM_PAGE);
        quote!(#evaluate_node_instructions #storage_instructions)
    }

    fn load_node<II: InputIndicator>(circuit: &ConstraintCircuit<II>) -> TokenStream {
        match circuit.expression {
            CircuitExpression::BConstant(bfe) => Self::load_base_field_constant(bfe),
            CircuitExpression::XConstant(xfe) => Self::load_ext_field_constant(xfe),
            CircuitExpression::Input(input) => Self::load_input(input),
            CircuitExpression::Challenge(challenge) => Self::load_challenge(challenge),
            CircuitExpression::BinaryOperation(_, _, _) => Self::load_pre_declared_node(circuit.id),
        }
    }

    fn perform_bin_op(binop: BinOp, lhs: TokenStream, rhs: TokenStream) -> TokenStream {
        let binop = Self::tokenize_bin_op(binop);
        quote!(#lhs #rhs #binop)
    }
}

impl TasmBackend {
    fn uses() -> TokenStream {
        quote!(
            use num_traits::Zero;
            use num_traits::One;
            use crate::instruction::AnInstruction;
            use crate::instruction::LabelledInstruction;
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
        BEFORE: _ *challenges *next_ext_row *next_base_row \
                        *curr_ext_row *curr_base_row *free_memory_page
        AFTER:  _ *evaluated_constraints
        ```
        # Requirements

        - both `*curr_base_row` and `*next_base_row` are pointers to an array of
            [`XFieldElement`][xfe]s of length [`NUM_BASE_COLUMNS`][num_base_cols].
        - both `*curr_ext_row` and `*next_ext_row` are pointers to an array of
            [`XFieldElement`][xfe]s of length [`NUM_EXT_COLUMNS`][num_ext_cols].
        - `*challenges` is a pointer to an array of [`XFieldElement`][xfe]s of length
            [`NUM_CHALLENGES`][num_challenges].
        - `*free_memory_page` points to a region of memory that is reserved for the emitted
            code. The size of the that region is at least 2^32 [`BFieldElement`][bfe]s.

        # Guarantees

        - The emitted code does not declare any labels.
        - The emitted code does not contain any of the instructions `call`, `return`, `recurse`,
            or `halt`.
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

    fn tokenize_circuits<II: InputIndicator>(constraints: &[ConstraintCircuit<II>]) -> TokenStream {
        if constraints.is_empty() {
            return quote!();
        }

        let all_visited_counters = constraints
            .iter()
            .flat_map(ConstraintCircuit::all_visited_counters);
        let all_visited_counters = all_visited_counters.sorted().unique().collect_vec();
        let relevant_visited_counters = all_visited_counters.into_iter().filter(|&x| x > 1);
        let shared_declarations = relevant_visited_counters
            .rev()
            .map(|count| Self::declare_nodes_with_visit_count(count, constraints))
            .collect_vec();

        // to match the `RustBackend`, base constraints must be emitted first
        let (base_constraints, ext_constraints): (Vec<_>, Vec<_>) = constraints
            .iter()
            .partition(|constraint| constraint.evaluates_to_base_element());
        let constraints = base_constraints
            .into_iter()
            .chain(ext_constraints)
            .collect_vec();
        let constraint_evaluator = constraints
            .iter()
            .flat_map(|constraint| Self::evaluate_single_node(1, constraint, &HashSet::default()))
            .collect_vec();

        quote!(#(#shared_declarations)* #(#constraint_evaluator)*)
    }

    fn load_base_field_constant(bfe: BFieldElement) -> TokenStream {
        let bfe = codegen::tokenize_bfe(bfe);
        quote!(
            LabelledInstruction::Instruction(AnInstruction::Push(BFieldElement::zero())),
            LabelledInstruction::Instruction(AnInstruction::Push(BFieldElement::zero())),
            LabelledInstruction::Instruction(AnInstruction::Push(#bfe)),
        )
    }

    fn load_ext_field_constant(xfe: XFieldElement) -> TokenStream {
        let [c0, c1, c2] = xfe.coefficients.map(codegen::tokenize_bfe);
        quote!(
            LabelledInstruction::Instruction(AnInstruction::Push(#c2)),
            LabelledInstruction::Instruction(AnInstruction::Push(#c1)),
            LabelledInstruction::Instruction(AnInstruction::Push(#c0)),
        )
    }

    fn load_input<II: InputIndicator>(input: II) -> TokenStream {
        let arg_pos = match (input.is_curr_row(), input.is_base_table_column()) {
            (true, true) => ARG_POS_CURR_BASE_ROW,
            (true, false) => ARG_POS_CURR_EXT_ROW,
            (false, true) => ARG_POS_NEXT_BASE_ROW,
            (false, false) => ARG_POS_NEXT_EXT_ROW,
        };
        let input_index = match input.is_base_table_column() {
            true => input.base_col_index(),
            false => input.ext_col_index(),
        };

        Self::load_ext_field_element_from_list(input_index, arg_pos)
    }

    fn load_challenge(challenge: ChallengeId) -> TokenStream {
        Self::load_ext_field_element_from_list(challenge.index(), ARG_POS_CHALLENGES)
    }

    fn load_pre_declared_node(node_id: usize) -> TokenStream {
        Self::load_ext_field_element_from_list(node_id, ARG_POS_FREE_MEM_PAGE)
    }

    fn load_ext_field_element_from_list(
        element_index: usize,
        arg_pos: OpStackElement,
    ) -> TokenStream {
        let word_offset = element_index * EXTENSION_DEGREE;
        let start_to_read_offset = EXTENSION_DEGREE - 1;
        let word_index = word_offset + start_to_read_offset;
        let word_index = BFieldElement::from(word_index as u64);

        let arg_pos = Self::tokenize_op_stack_element(arg_pos);
        let word_index = codegen::tokenize_bfe(word_index);

        quote!(
            LabelledInstruction::Instruction(AnInstruction::Dup(#arg_pos)),
            LabelledInstruction::Instruction(AnInstruction::Push(#word_index)),
            LabelledInstruction::Instruction(AnInstruction::Add),
            LabelledInstruction::Instruction(AnInstruction::ReadMem(NumberOfWords::N3)),
            LabelledInstruction::Instruction(AnInstruction::Pop(NumberOfWords::N1)),
        )
    }

    fn store_ext_field_element_in_list(
        element_index: usize,
        arg_pos: OpStackElement,
    ) -> TokenStream {
        let word_offset = element_index * EXTENSION_DEGREE;
        let word_index = BFieldElement::from(word_offset as u64);

        let arg_pos = Self::tokenize_op_stack_element(arg_pos);
        let word_index = codegen::tokenize_bfe(word_index);

        quote!(
            LabelledInstruction::Instruction(AnInstruction::Dup(#arg_pos)),
            LabelledInstruction::Instruction(AnInstruction::Push(#word_index)),
            LabelledInstruction::Instruction(AnInstruction::Add),
            LabelledInstruction::Instruction(AnInstruction::WriteMem(NumberOfWords::N3)),
            LabelledInstruction::Instruction(AnInstruction::Pop(NumberOfWords::N1)),
        )
    }

    fn tokenize_op_stack_element(st: OpStackElement) -> TokenStream {
        let idx = st.index();
        let st = format_ident!("ST{idx}");
        quote!(OpStackElement::#st)
    }

    fn tokenize_bin_op(binop: BinOp) -> TokenStream {
        if binop == BinOp::Sub {
            let minus_one = Self::load_base_field_constant(-BFieldElement::new(1));
            return quote!(
                #minus_one
                LabelledInstruction::Instruction(AnInstruction::XxMul),
                LabelledInstruction::Instruction(AnInstruction::XxAdd),
            );
        }

        let op = match binop {
            BinOp::Add => quote!(XxAdd),
            BinOp::Sub => unreachable!(),
            BinOp::Mul => quote!(XxMul),
        };
        quote!(LabelledInstruction::Instruction(AnInstruction::#op),)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_test_constraints_as_tasm() {
        let constraints = Constraints::mini_constraints();
        let tasm = TasmBackend::constraint_evaluation_code(&constraints);
        let syntax_tree = syn::parse2(tasm).unwrap();
        let code = prettyplease::unparse(&syntax_tree);
        println!("{code}");
    }
}
