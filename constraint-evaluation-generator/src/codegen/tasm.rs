use itertools::Itertools;
use proc_macro2::TokenStream;
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

/// An offset from [`ARG_POS_FREE_MEM_PAGE`]'s value. Indicates the start of the to-be-returned
/// array.
///
/// The value is a bit magical; as per the contract declared in [`TasmBackend::doc_comment`], the
/// region of free memory must be at least (1 << 32) words big. The number of constraints is assumed
/// to stay smaller than (1 << 16). It is also assumed that the number of nodes in any of the
/// multicircuits does not grow beyond [`OUT_ARRAY_OFFSET`].
const OUT_ARRAY_OFFSET: usize = (1 << 32) - (1 << 16);

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

        let from_input_to_output_args = Self::from_input_args_to_output_args();

        quote!(
            #uses
            #[doc = #doc_comment]
            pub fn air_constraint_evaluation_tasm(
                layout: TasmConstraintEvaluationMemoryLayout,
            ) -> Box<[LabelledInstruction]> {
                Box::new([
                    #init_constraints
                    #cons_constraints
                    #tran_constraints
                    #term_constraints
                    #from_input_to_output_args
                ])
            }
        )
    }
}

impl TasmBackend {
    fn new() -> Self {
        Self {
            additional_stack_size: 0,
            elements_written: 0,
        }
    }

    fn uses() -> TokenStream {
        quote!(
            use num_traits::Zero;
            use twenty_first::prelude::BFieldElement;
            use crate::instruction::AnInstruction;
            use crate::instruction::LabelledInstruction;
            use crate::op_stack::NumberOfWords;
            use crate::op_stack::OpStackElement;
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

    fn tokenize_circuits<II: InputIndicator>(
        &mut self,
        constraints: &[ConstraintCircuit<II>],
    ) -> TokenStream {
        if constraints.is_empty() {
            return quote!();
        }

        let constraint_evaluator = self.evaluate_all_bin_ops(constraints);

        // to match the `RustBackend`, base constraints must be emitted first
        let (base_constraints, ext_constraints): (Vec<_>, Vec<_>) = constraints
            .iter()
            .partition(|constraint| constraint.evaluates_to_base_element());
        let output_list_writer = base_constraints
            .into_iter()
            .chain(ext_constraints)
            .map(|constraint| self.write_evaluated_constraint_into_output_list(constraint));

        let tokenized_circuits = quote!(#(#constraint_evaluator)* #(#output_list_writer)*);
        debug_assert_eq!(0, self.additional_stack_size);

        tokenized_circuits
    }

    fn evaluate_all_bin_ops<II: InputIndicator>(
        &mut self,
        constraints: &[ConstraintCircuit<II>],
    ) -> Vec<TokenStream> {
        let ref_counters = constraints
            .iter()
            .flat_map(ConstraintCircuit::all_ref_counters);

        let mut bin_op_evaluation_code = vec![];
        for ref_count in ref_counters.sorted().unique().rev() {
            for circuit in constraints.iter().filter(|c| c.ref_count == ref_count) {
                let CircuitExpression::BinaryOperation(op, ref lhs, ref rhs) = circuit.expression
                else {
                    continue;
                };
                let lhs = self.load_node(&lhs.borrow());
                let rhs = self.load_node(&rhs.borrow());
                let binop = self.tokenize_bin_op(op);
                let store = self.store_ext_field_element_in_list(circuit.id, ARG_POS_FREE_MEM_PAGE);
                bin_op_evaluation_code.push(quote!(#lhs #rhs #binop #store));
            }
        }

        bin_op_evaluation_code
    }

    fn write_evaluated_constraint_into_output_list<II: InputIndicator>(
        &mut self,
        constraint: &ConstraintCircuit<II>,
    ) -> TokenStream {
        let evaluated_constraint = self.load_node(constraint);
        let element_index = OUT_ARRAY_OFFSET + self.elements_written;
        let store_element =
            self.store_ext_field_element_in_list(element_index, ARG_POS_FREE_MEM_PAGE);
        quote!(#evaluated_constraint #store_element)
    }

    fn load_node<II: InputIndicator>(&mut self, circuit: &ConstraintCircuit<II>) -> TokenStream {
        match circuit.expression {
            CircuitExpression::BConstant(bfe) => self.load_base_field_constant(bfe),
            CircuitExpression::XConstant(xfe) => self.load_ext_field_constant(xfe),
            CircuitExpression::Input(input) => self.load_input(input),
            CircuitExpression::Challenge(challenge) => self.load_challenge(challenge),
            CircuitExpression::BinaryOperation(_, _, _) => self.load_evaluated_bin_op(circuit.id),
        }
    }

    fn load_base_field_constant(&mut self, bfe: BFieldElement) -> TokenStream {
        self.additional_stack_size += EXTENSION_DEGREE;
        let bfe = Self::tokenize_bfe(bfe);
        quote!(
            LabelledInstruction::Instruction(AnInstruction::Push(BFieldElement::zero())),
            LabelledInstruction::Instruction(AnInstruction::Push(BFieldElement::zero())),
            LabelledInstruction::Instruction(AnInstruction::Push(#bfe)),
        )
    }

    fn load_ext_field_constant(&mut self, xfe: XFieldElement) -> TokenStream {
        self.additional_stack_size += EXTENSION_DEGREE;
        let [c0, c1, c2] = xfe.coefficients.map(Self::tokenize_bfe);
        quote!(
            LabelledInstruction::Instruction(AnInstruction::Push(#c2)),
            LabelledInstruction::Instruction(AnInstruction::Push(#c1)),
            LabelledInstruction::Instruction(AnInstruction::Push(#c0)),
        )
    }

    fn load_input<II: InputIndicator>(&mut self, input: II) -> TokenStream {
        let arg_pos = match (input.is_current_row(), input.is_base_table_column()) {
            (true, true) => ARG_POS_CURR_BASE_ROW,
            (true, false) => ARG_POS_CURR_EXT_ROW,
            (false, true) => ARG_POS_NEXT_BASE_ROW,
            (false, false) => ARG_POS_NEXT_EXT_ROW,
        };

        self.load_ext_field_element_from_list(input.column(), arg_pos)
    }

    fn load_challenge(&mut self, challenge: ChallengeId) -> TokenStream {
        self.load_ext_field_element_from_list(challenge.index(), ARG_POS_CHALLENGES)
    }

    fn load_evaluated_bin_op(&mut self, node_id: usize) -> TokenStream {
        self.load_ext_field_element_from_list(node_id, ARG_POS_FREE_MEM_PAGE)
    }

    fn load_ext_field_element_from_list(
        &mut self,
        element_index: usize,
        list_pointer: OpStackElement,
    ) -> TokenStream {
        let word_offset = element_index * EXTENSION_DEGREE;
        let start_to_read_offset = EXTENSION_DEGREE - 1;
        let word_index = word_offset + start_to_read_offset;
        let word_index = BFieldElement::from(word_index as u64);

        let actual_list_pointer = usize::from(list_pointer) + self.additional_stack_size;
        let actual_list_pointer = OpStackElement::try_from(actual_list_pointer).unwrap();
        let actual_list_pointer = Self::tokenize_op_stack_element(actual_list_pointer);
        let word_index = Self::tokenize_bfe(word_index);

        self.additional_stack_size += EXTENSION_DEGREE;
        quote!(
            LabelledInstruction::Instruction(AnInstruction::Dup(#actual_list_pointer)),
            LabelledInstruction::Instruction(AnInstruction::Push(#word_index)),
            LabelledInstruction::Instruction(AnInstruction::Add),
            LabelledInstruction::Instruction(AnInstruction::ReadMem(NumberOfWords::N3)),
            LabelledInstruction::Instruction(AnInstruction::Pop(NumberOfWords::N1)),
        )
    }

    fn store_ext_field_element_in_list(
        &mut self,
        element_index: usize,
        list_pointer: OpStackElement,
    ) -> TokenStream {
        let word_offset = element_index * EXTENSION_DEGREE;
        let word_index = BFieldElement::from(word_offset as u64);

        let actual_list_pointer = usize::from(list_pointer) + self.additional_stack_size;
        let actual_list_pointer = OpStackElement::try_from(actual_list_pointer).unwrap();
        let actual_list_pointer = Self::tokenize_op_stack_element(actual_list_pointer);
        let word_index = Self::tokenize_bfe(word_index);

        self.additional_stack_size -= EXTENSION_DEGREE;
        quote!(
            LabelledInstruction::Instruction(AnInstruction::Dup(#actual_list_pointer)),
            LabelledInstruction::Instruction(AnInstruction::Push(#word_index)),
            LabelledInstruction::Instruction(AnInstruction::Add),
            LabelledInstruction::Instruction(AnInstruction::WriteMem(NumberOfWords::N3)),
            LabelledInstruction::Instruction(AnInstruction::Pop(NumberOfWords::N1)),
        )
    }

    fn tokenize_bin_op(&mut self, binop: BinOp) -> TokenStream {
        if binop == BinOp::Sub {
            let minus_one = self.load_base_field_constant(-BFieldElement::new(1));
            self.additional_stack_size -= 2 * EXTENSION_DEGREE;
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
        self.additional_stack_size -= EXTENSION_DEGREE;
        quote!(LabelledInstruction::Instruction(AnInstruction::#op),)
    }

    fn from_input_args_to_output_args() -> TokenStream {
        let out_array_offset = OUT_ARRAY_OFFSET as u64;
        let out_array_offset = quote!(BFieldElement::new(#out_array_offset));
        let out_array_offset = quote!(AnInstruction::Push(#out_array_offset));
        let out_array_offset = quote!(LabelledInstruction::Instruction(#out_array_offset),);

        let add = quote!(LabelledInstruction::Instruction(AnInstruction::Add),);
        let swap_5 = quote!(AnInstruction::Swap(OpStackElement::ST5));
        let swap_5 = quote!(LabelledInstruction::Instruction(#swap_5),);

        let pop_5 = quote!(AnInstruction::Pop(NumberOfWords::N5));
        let pop_5 = quote!(LabelledInstruction::Instruction(#pop_5));
        let from_input_to_output_args = quote!(#out_array_offset #add #swap_5 #pop_5);
        from_input_to_output_args
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
