//! The various tables' constraints are very inefficient to evaluate if they
//! live in RAM. Instead, the build script turns them into rust code, which is
//! then optimized by rustc.

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use constraint_circuit::BinOp;
use constraint_circuit::CircuitExpression;
use constraint_circuit::ConstraintCircuit;
use constraint_circuit::InputIndicator;
use isa::instruction::Instruction;
use isa::op_stack::NumberOfWords;
use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::ToTokens;
use quote::format_ident;
use quote::quote;
use twenty_first::prelude::x_field_element::EXTENSION_DEGREE;
use twenty_first::prelude::*;

use crate::Constraints;

pub trait Codegen {
    fn constraint_evaluation_code(constraints: &Constraints) -> TokenStream;

    fn tokenize_bfe(bfe: BFieldElement) -> TokenStream {
        let raw_u64 = bfe.raw_u64();
        quote!(BFieldElement::from_raw_u64(#raw_u64))
    }

    fn tokenize_xfe(xfe: XFieldElement) -> TokenStream {
        let [c_0, c_1, c_2] = xfe.coefficients.map(Self::tokenize_bfe);
        quote!(XFieldElement::new([#c_0, #c_1, #c_2]))
    }
}

#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub struct RustBackend {
    /// All [circuit][ConstraintCircuit] IDs known to be in scope.
    scope: HashSet<usize>,
}

#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub struct TasmBackend {
    /// All [circuit][ConstraintCircuit] IDs known to be processed and stored to
    /// memory.
    scope: HashSet<usize>,

    /// The number of elements written to the output list.
    elements_written: usize,

    /// Whether the code that is to be generated can assume statically provided
    /// addresses for the various input arrays.
    input_location_is_static: bool,
}

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
        impl MasterAuxTable {
            pub const NUM_INITIAL_CONSTRAINTS: usize = #num_init_constraints;
            pub const NUM_CONSISTENCY_CONSTRAINTS: usize = #num_cons_constraints;
            pub const NUM_TRANSITION_CONSTRAINTS: usize = #num_tran_constraints;
            pub const NUM_TERMINAL_CONSTRAINTS: usize = #num_term_constraints;
            pub const NUM_CONSTRAINTS: usize = Self::NUM_INITIAL_CONSTRAINTS
                + Self::NUM_CONSISTENCY_CONSTRAINTS
                + Self::NUM_TRANSITION_CONSTRAINTS
                + Self::NUM_TERMINAL_CONSTRAINTS;

            #[allow(unused_variables)]
            pub fn initial_quotient_degree_bounds(interpolant_degree: isize) -> Vec<isize> {
                let zerofier_degree = 1;
                [#init_constraint_degrees].to_vec()
            }

            #[allow(unused_variables)]
            pub fn consistency_quotient_degree_bounds(
                interpolant_degree: isize,
                padded_height: usize,
            ) -> Vec<isize> {
                let zerofier_degree = padded_height as isize;
                [#cons_constraint_degrees].to_vec()
            }

            #[allow(unused_variables)]
            pub fn transition_quotient_degree_bounds(
                interpolant_degree: isize,
                padded_height: usize,
            ) -> Vec<isize> {
                let zerofier_degree = padded_height as isize - 1;
                [#tran_constraint_degrees].to_vec()
            }

            #[allow(unused_variables)]
            pub fn terminal_quotient_degree_bounds(interpolant_degree: isize) -> Vec<isize> {
                let zerofier_degree = 1;
                [#term_constraint_degrees].to_vec()
            }
        }
        );

        quote!(
            #evaluable_over_base_field
            #evaluable_over_ext_field
            #quotient_trait_impl
        )
    }
}

impl RustBackend {
    fn generate_evaluable_implementation_over_field(
        init_constraints: &TokenStream,
        cons_constraints: &TokenStream,
        tran_constraints: &TokenStream,
        term_constraints: &TokenStream,
        field: TokenStream,
    ) -> TokenStream {
        quote!(
        impl Evaluable<#field> for MasterAuxTable {
            #[allow(unused_variables)]
            fn evaluate_initial_constraints(
                main_row: ArrayView1<#field>,
                aux_row: ArrayView1<XFieldElement>,
                challenges: &Challenges,
            ) -> Vec<XFieldElement> {
                #init_constraints
            }

            #[allow(unused_variables)]
            fn evaluate_consistency_constraints(
                main_row: ArrayView1<#field>,
                aux_row: ArrayView1<XFieldElement>,
                challenges: &Challenges,
            ) -> Vec<XFieldElement> {
                #cons_constraints
            }

            #[allow(unused_variables)]
            fn evaluate_transition_constraints(
                current_main_row: ArrayView1<#field>,
                current_aux_row: ArrayView1<XFieldElement>,
                next_main_row: ArrayView1<#field>,
                next_aux_row: ArrayView1<XFieldElement>,
                challenges: &Challenges,
            ) -> Vec<XFieldElement> {
                #tran_constraints
            }

            #[allow(unused_variables)]
            fn evaluate_terminal_constraints(
                main_row: ArrayView1<#field>,
                aux_row: ArrayView1<XFieldElement>,
                challenges: &Challenges,
            ) -> Vec<XFieldElement> {
                #term_constraints
            }
        }
        )
    }

    /// Return a tuple of [`TokenStream`]s corresponding to code evaluating
    /// these constraints as well as their degrees. In particular:
    /// 1. The first stream contains code that, when evaluated, produces the
    ///    constraints' degrees,
    /// 1. the second stream contains code that, when evaluated, produces the
    ///    constraints' values, with the input type for the main row being
    ///    `BFieldElement`, and
    /// 1. the third stream is like the second, except that the input type for
    ///    the main row is `XFieldElement`.
    fn tokenize_circuits<II: InputIndicator>(
        constraints: &[ConstraintCircuit<II>],
    ) -> (TokenStream, TokenStream, TokenStream) {
        if constraints.is_empty() {
            return (quote!(), quote!(vec![]), quote!(vec![]));
        }

        let mut backend = Self::default();
        let shared_declarations = backend.declare_shared_nodes(constraints);
        let (main_constraints, aux_constraints): (Vec<_>, Vec<_>) = constraints
            .iter()
            .partition(|constraint| constraint.evaluates_to_base_element());

        // The order of the constraints' degrees must match the order of the
        // constraints. Hence, listing the degrees is only possible after the
        // partition into main and auxiliary constraints is known.
        let tokenized_degree_bounds = main_constraints
            .iter()
            .chain(&aux_constraints)
            .map(|circuit| match circuit.degree() {
                d if d > 1 => quote!(interpolant_degree * #d - zerofier_degree),
                1 => quote!(interpolant_degree - zerofier_degree),
                _ => panic!("Constraint degree must be positive"),
            })
            .collect_vec();
        let tokenized_degree_bounds = quote!(#(#tokenized_degree_bounds),*);

        let tokenize_constraint_evaluation = |constraints: &[&ConstraintCircuit<II>]| {
            constraints
                .iter()
                .map(|constraint| backend.evaluate_single_node(constraint))
                .collect_vec()
        };
        let tokenized_main_constraints = tokenize_constraint_evaluation(&main_constraints);
        let tokenized_aux_constraints = tokenize_constraint_evaluation(&aux_constraints);

        // If there are no main constraints, the type needs to be explicitly
        // declared.
        let tokenized_bfe_main_constraints = match main_constraints.is_empty() {
            true => quote!(let main_constraints: [BFieldElement; 0] = []),
            false => quote!(let main_constraints = [#(#tokenized_main_constraints),*]),
        };
        let tokenized_bfe_constraints = quote!(
            #(#shared_declarations)*
            #tokenized_bfe_main_constraints;
            let aux_constraints = [#(#tokenized_aux_constraints),*];
            main_constraints
                .into_iter()
                .map(|bfe| bfe.lift())
                .chain(aux_constraints)
                .collect()
        );

        let tokenized_xfe_constraints = quote!(
            #(#shared_declarations)*
            let main_constraints = [#(#tokenized_main_constraints),*];
            let aux_constraints = [#(#tokenized_aux_constraints),*];
            main_constraints
                .into_iter()
                .chain(aux_constraints)
                .collect()
        );

        (
            tokenized_degree_bounds,
            tokenized_bfe_constraints,
            tokenized_xfe_constraints,
        )
    }

    /// Declare all shared variables, i.e., those with a reference count greater
    /// than 1. These declarations must be made starting from the highest ref
    /// count. Otherwise, the resulting code will refer to bindings that have
    /// not yet been made.
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
        let CircuitExpression::BinOp(_, lhs, rhs) = &circuit.expression else {
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

        let CircuitExpression::BinOp(binop, lhs, rhs) = &circuit.expression else {
            return Self::binding_name(circuit);
        };

        let lhs = self.evaluate_single_node(&lhs.borrow());
        let rhs = self.evaluate_single_node(&rhs.borrow());
        quote!((#lhs) #binop (#rhs))
    }

    fn binding_name<II: InputIndicator>(circuit: &ConstraintCircuit<II>) -> TokenStream {
        match &circuit.expression {
            CircuitExpression::BConst(bfe) => Self::tokenize_bfe(*bfe),
            CircuitExpression::XConst(xfe) => Self::tokenize_xfe(*xfe),
            CircuitExpression::Input(idx) => quote!(#idx),
            CircuitExpression::Challenge(challenge) => quote!(challenges[#challenge]),
            CircuitExpression::BinOp(_, _, _) => {
                let node_ident = format_ident!("node_{}", circuit.id);
                quote!(#node_ident)
            }
        }
    }
}

/// The minimal required size of a memory page in [`BFieldElement`]s.
pub const MEM_PAGE_SIZE: usize = 1 << 32;

/// An offset from the memory layout's `free_mem_page_ptr`, in number of
/// [`XFieldElement`]s. Indicates the start of the to-be-returned array.
///
/// See `IntegralMemoryLayout` in crate triton-vm for more context.
const OUT_ARRAY_OFFSET: usize = {
    let max_num_words_for_evaluated_constraints = 1 << 16; // magic!
    let out_array_offset_in_words = MEM_PAGE_SIZE - max_num_words_for_evaluated_constraints;
    assert!(out_array_offset_in_words.is_multiple_of(EXTENSION_DEGREE));
    out_array_offset_in_words / EXTENSION_DEGREE
};

/// Convenience macro to get raw opcodes of any [`Instruction`] variant,
/// including its argument if applicable.
///
/// [labelled]: isa::instruction::LabelledInstruction::Instruction
macro_rules! instr {
    ($($instr:tt)*) => {{
        let instr = Instruction::$($instr)*;
        let opcode = u64::from(instr.opcode());
        match instr.arg().map(|arg| arg.value()) {
            Some(arg) => vec![quote!(#opcode), quote!(#arg)],
            None => vec![quote!(#opcode)],
        }
    }};
}

/// Convenience macro to get raw opcode of a [`Push`][push] instruction
/// including its argument.
///
/// [push]: Instruction::Push
macro_rules! push {
    ($arg:ident) => {{
        let opcode = u64::from(Instruction::Push(BFieldElement::new(0)).opcode());
        let arg = u64::from($arg);
        vec![quote!(#opcode), quote!(#arg)]
    }};
    ($list:ident + $offset:expr) => {{
        let opcode = u64::from(Instruction::Push(BFieldElement::new(0)).opcode());
        let offset = u64::try_from($offset).unwrap();
        assert!(offset < u64::MAX - BFieldElement::P);
        // clippy will complain about the generated code if it contains `+ 0`
        if offset == 0 {
            vec![quote!(#opcode), quote!(#$list)]
        } else {
            vec![quote!(#opcode), quote!(#$list + #offset)]
        }
    }};
}

impl Codegen for TasmBackend {
    /// Emits a function that emits [Triton assembly][tasm] that evaluates
    /// Triton VM's AIR constraints over the [extension
    /// field][XFieldElement].
    ///
    /// [tasm]: isa::triton_asm
    fn constraint_evaluation_code(constraints: &Constraints) -> TokenStream {
        let doc_comment = Self::doc_comment_static_version();

        let mut backend = Self::statically_known_input_locations();
        let init_constraints = backend.tokenize_circuits(&constraints.init());
        let cons_constraints = backend.tokenize_circuits(&constraints.cons());
        let tran_constraints = backend.tokenize_circuits(&constraints.tran());
        let term_constraints = backend.tokenize_circuits(&constraints.term());
        let prepare_return_values = Self::prepare_return_values();
        let num_instructions = init_constraints.len()
            + cons_constraints.len()
            + tran_constraints.len()
            + term_constraints.len()
            + prepare_return_values.len();
        let num_instructions = u64::try_from(num_instructions).unwrap();

        let convert_and_decode_assembled_instructions = quote!(
            let raw_instructions = raw_instructions
                .into_iter()
                .map(BFieldElement::new)
                .collect::<Vec<_>>();
            let program = Program::decode(&raw_instructions).unwrap();

            let irrelevant_label = |_: &_| String::new();
            program
                .into_iter()
                .map(|instruction| instruction.map_call_address(irrelevant_label))
                .map(LabelledInstruction::Instruction)
                .collect()
        );

        let statically_known_input_locations = quote!(
            #[doc = #doc_comment]
            pub fn static_air_constraint_evaluation_tasm(
                mem_layout: StaticTasmConstraintEvaluationMemoryLayout,
            ) -> Vec<LabelledInstruction> {
                let free_mem_page_ptr = mem_layout.free_mem_page_ptr.value();
                let curr_main_row_ptr = mem_layout.curr_main_row_ptr.value();
                let curr_aux_row_ptr = mem_layout.curr_aux_row_ptr.value();
                let next_main_row_ptr = mem_layout.next_main_row_ptr.value();
                let next_aux_row_ptr = mem_layout.next_aux_row_ptr.value();
                let challenges_ptr = mem_layout.challenges_ptr.value();

                let raw_instructions = vec![
                    #num_instructions,
                    #(#init_constraints,)*
                    #(#cons_constraints,)*
                    #(#tran_constraints,)*
                    #(#term_constraints,)*
                    #(#prepare_return_values,)*
                ];
                #convert_and_decode_assembled_instructions
            }
        );

        let doc_comment = Self::doc_comment_dynamic_version();

        let mut backend = Self::dynamically_known_input_locations();
        let move_row_pointers = backend.write_row_pointers_to_ram();
        let init_constraints = backend.tokenize_circuits(&constraints.init());
        let cons_constraints = backend.tokenize_circuits(&constraints.cons());
        let tran_constraints = backend.tokenize_circuits(&constraints.tran());
        let term_constraints = backend.tokenize_circuits(&constraints.term());
        let prepare_return_values = Self::prepare_return_values();
        let num_instructions = move_row_pointers.len()
            + init_constraints.len()
            + cons_constraints.len()
            + tran_constraints.len()
            + term_constraints.len()
            + prepare_return_values.len();
        let num_instructions = u64::try_from(num_instructions).unwrap();

        let dynamically_known_input_locations = quote!(
            #[doc = #doc_comment]
            pub fn dynamic_air_constraint_evaluation_tasm(
                mem_layout: DynamicTasmConstraintEvaluationMemoryLayout,
            ) -> Vec<LabelledInstruction> {
                let num_pointer_pointers = 4;
                let free_mem_page_ptr = mem_layout.free_mem_page_ptr.value() + num_pointer_pointers;
                let curr_main_row_ptr = mem_layout.free_mem_page_ptr.value();
                let curr_aux_row_ptr = mem_layout.free_mem_page_ptr.value() + 1;
                let next_main_row_ptr = mem_layout.free_mem_page_ptr.value() + 2;
                let next_aux_row_ptr = mem_layout.free_mem_page_ptr.value() + 3;
                let challenges_ptr = mem_layout.challenges_ptr.value();

                let raw_instructions = vec![
                    #num_instructions,
                    #(#move_row_pointers,)*
                    #(#init_constraints,)*
                    #(#cons_constraints,)*
                    #(#tran_constraints,)*
                    #(#term_constraints,)*
                    #(#prepare_return_values,)*
                ];
                #convert_and_decode_assembled_instructions
            }
        );

        let uses = Self::uses();
        quote!(
            #uses
            #statically_known_input_locations
            #dynamically_known_input_locations
        )
    }
}

impl TasmBackend {
    fn statically_known_input_locations() -> Self {
        Self {
            scope: HashSet::new(),
            elements_written: 0,
            input_location_is_static: true,
        }
    }

    fn dynamically_known_input_locations() -> Self {
        Self {
            input_location_is_static: false,
            ..Self::statically_known_input_locations()
        }
    }

    fn uses() -> TokenStream {
        quote!(
            use twenty_first::prelude::BFieldCodec;
            use twenty_first::prelude::BFieldElement;
            use isa::instruction::LabelledInstruction;
            use isa::program::Program;
            use crate::memory_layout::StaticTasmConstraintEvaluationMemoryLayout;
            use crate::memory_layout::DynamicTasmConstraintEvaluationMemoryLayout;
        )
    }

    fn doc_comment_static_version() -> &'static str {
        "
         The emitted Triton assembly has the following signature:

         # Signature

         ```text
         BEFORE: _
         AFTER:  _ *evaluated_constraints
         ```
         # Requirements

         In order for this method to emit Triton assembly, various memory regions need to be
         declared. This is done through [`StaticTasmConstraintEvaluationMemoryLayout`]. The memory
         layout must be [integral].

         # Guarantees

         - The emitted code does not declare any labels.
         - The emitted code is “straight-line”, _i.e._, does not contain any of the instructions
           `call`, `return`, `recurse`, `recurse_or_return`, or `skiz`.
         - The emitted code does not contain instruction `halt`.
         - All memory write access of the emitted code is within the bounds of the memory region
           pointed to by `*free_memory_page`.
         - `*evaluated_constraints` points to an array of [`XFieldElement`][xfe]s of length
           [`NUM_CONSTRAINTS`][total]. Each element is the evaluation of one constraint. In
           particular, the disjoint sequence of slices containing
           [`NUM_INITIAL_CONSTRAINTS`][init], [`NUM_CONSISTENCY_CONSTRAINTS`][cons],
           [`NUM_TRANSITION_CONSTRAINTS`][tran], and [`NUM_TERMINAL_CONSTRAINTS`][term]
           (respectively and in this order) correspond to the evaluations of the initial,
           consistency, transition, and terminal constraints.

         [integral]: crate::memory_layout::IntegralMemoryLayout::is_integral
         [xfe]: twenty_first::prelude::XFieldElement
         [total]: crate::table::master_table::MasterAuxTable::NUM_CONSTRAINTS
         [init]: crate::table::master_table::MasterAuxTable::NUM_INITIAL_CONSTRAINTS
         [cons]: crate::table::master_table::MasterAuxTable::NUM_CONSISTENCY_CONSTRAINTS
         [tran]: crate::table::master_table::MasterAuxTable::NUM_TRANSITION_CONSTRAINTS
         [term]: crate::table::master_table::MasterAuxTable::NUM_TERMINAL_CONSTRAINTS
        "
    }

    fn doc_comment_dynamic_version() -> &'static str {
        "
         The emitted Triton assembly has the following signature:

         # Signature

         ```text
         BEFORE: _ *current_main_row *current_aux_row *next_main_row *next_aux_row
         AFTER:  _ *evaluated_constraints
         ```
         # Requirements

         In order for this method to emit Triton assembly, various memory regions need to be
         declared. This is done through [`DynamicTasmConstraintEvaluationMemoryLayout`]. The memory
         layout must be [integral].

         # Guarantees

         - The emitted code does not declare any labels.
         - The emitted code is “straight-line”, _i.e._, does not contain any of the instructions
           `call`, `return`, `recurse`, `recurse_or_return`, or `skiz`.
         - The emitted code does not contain instruction `halt`.
         - All memory write access of the emitted code is within the bounds of the memory region
           pointed to by `*free_memory_page`.
         - `*evaluated_constraints` points to an array of [`XFieldElement`][xfe]s of length
           [`NUM_CONSTRAINTS`][total]. Each element is the evaluation of one constraint. In
           particular, the disjoint sequence of slices containing
           [`NUM_INITIAL_CONSTRAINTS`][init], [`NUM_CONSISTENCY_CONSTRAINTS`][cons],
           [`NUM_TRANSITION_CONSTRAINTS`][tran], and [`NUM_TERMINAL_CONSTRAINTS`][term]
           (respectively and in this order) correspond to the evaluations of the initial,
           consistency, transition, and terminal constraints.

         [integral]: crate::memory_layout::IntegralMemoryLayout::is_integral
         [xfe]: twenty_first::prelude::XFieldElement
         [total]: crate::table::master_table::MasterAuxTable::NUM_CONSTRAINTS
         [init]: crate::table::master_table::MasterAuxTable::NUM_INITIAL_CONSTRAINTS
         [cons]: crate::table::master_table::MasterAuxTable::NUM_CONSISTENCY_CONSTRAINTS
         [tran]: crate::table::master_table::MasterAuxTable::NUM_TRANSITION_CONSTRAINTS
         [term]: crate::table::master_table::MasterAuxTable::NUM_TERMINAL_CONSTRAINTS
        "
    }

    /// Moves the dynamic arguments ({current, next} {main, aux} row pointers)
    /// to static addresses dedicated to them.
    fn write_row_pointers_to_ram(&self) -> Vec<TokenStream> {
        // BEFORE: _ *current_main_row *current_aux_row *next_main_row *next_aux_row
        // AFTER: _

        let write_pointer_to_ram = |list_id| {
            [
                push!(list_id + 0),
                instr!(WriteMem(NumberOfWords::N1)),
                instr!(Pop(NumberOfWords::N1)),
            ]
            .concat()
        };

        [
            IOList::NextAuxRow,
            IOList::NextMainRow,
            IOList::CurrAuxRow,
            IOList::CurrMainRow,
        ]
        .into_iter()
        .flat_map(write_pointer_to_ram)
        .collect()
    }

    fn tokenize_circuits<II: InputIndicator>(
        &mut self,
        constraints: &[ConstraintCircuit<II>],
    ) -> Vec<TokenStream> {
        self.scope = HashSet::new();
        let store_shared_nodes = self.store_all_shared_nodes(constraints);

        // to match the `RustBackend`, main constraints must be emitted first
        let (main_constraints, aux_constraints): (Vec<_>, Vec<_>) = constraints
            .iter()
            .partition(|constraint| constraint.evaluates_to_base_element());
        let sorted_constraints = main_constraints.into_iter().chain(aux_constraints);
        let write_to_output = sorted_constraints
            .map(|c| self.write_evaluated_constraint_into_output_list(c))
            .concat();

        [store_shared_nodes, write_to_output].concat()
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
            .concat()
    }

    fn store_single_shared_node_of_ref_count<II: InputIndicator>(
        &mut self,
        constraint: &ConstraintCircuit<II>,
        ref_count: usize,
    ) -> Vec<TokenStream> {
        if self.scope.contains(&constraint.id) {
            return vec![];
        }

        // Nodes that are not binary operations are already in scope as inputs
        // or challenges, or they are constants.
        let CircuitExpression::BinOp(_, lhs, rhs) = &constraint.expression else {
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
            return self.load_node(constraint);
        }

        let CircuitExpression::BinOp(binop, lhs, rhs) = &constraint.expression else {
            return self.load_node(constraint);
        };

        let tokenized_lhs = self.evaluate_single_node(&lhs.borrow());
        let tokenized_rhs = self.evaluate_single_node(&rhs.borrow());
        let tokenized_binop = match binop {
            BinOp::Add => instr!(XxAdd),
            BinOp::Mul => instr!(XxMul),
        };

        // Use more efficient instructions if one side is a base field element.
        // Applying domain-specific knowledge, `CircuitExpression::Input`s can
        // never be base field elements as the verifier only evaluates the
        // constraints on out-of-domain rows. The TASM backend is only intended
        // for verification.
        let extract_bfe_const =
            |circuit: &Rc<RefCell<ConstraintCircuit<II>>>| match circuit.borrow().expression {
                CircuitExpression::BConst(bfe) => Some(bfe),
                _ => None,
            };

        match (binop, extract_bfe_const(lhs), extract_bfe_const(rhs)) {
            (_, Some(_), Some(_)) => {
                panic!("Constant folding should have eliminated this binary operation")
            }
            (_, None, None) => [tokenized_lhs, tokenized_rhs, tokenized_binop].concat(),
            (BinOp::Add, None, Some(c)) => [tokenized_lhs, instr!(AddI(c))].concat(),
            (BinOp::Add, Some(c), None) => [tokenized_rhs, instr!(AddI(c))].concat(),
            (BinOp::Mul, None, Some(_)) => [tokenized_lhs, tokenized_rhs, instr!(XbMul)].concat(),
            (BinOp::Mul, Some(_), None) => [tokenized_rhs, tokenized_lhs, instr!(XbMul)].concat(),
        }
    }

    fn write_evaluated_constraint_into_output_list<II: InputIndicator>(
        &mut self,
        constraint: &ConstraintCircuit<II>,
    ) -> Vec<TokenStream> {
        let evaluated_constraint = self.evaluate_single_node(constraint);
        let element_index = OUT_ARRAY_OFFSET + self.elements_written;
        let store_element = Self::store_ext_field_element(element_index);
        self.elements_written += 1;
        [evaluated_constraint, store_element].concat()
    }

    fn load_node<II: InputIndicator>(&self, circuit: &ConstraintCircuit<II>) -> Vec<TokenStream> {
        match circuit.expression {
            CircuitExpression::BConst(bfe) => push!(bfe),
            CircuitExpression::XConst(xfe) => {
                let [c0, c1, c2] = xfe.coefficients.map(|c| push!(c));
                [c2, c1, c0].concat()
            }
            CircuitExpression::Input(input) => self.load_input(input),
            CircuitExpression::Challenge(challenge_idx) => Self::load_challenge(challenge_idx),
            CircuitExpression::BinOp(_, _, _) => Self::load_evaluated_bin_op(circuit.id),
        }
    }

    fn load_input<II: InputIndicator>(&self, input: II) -> Vec<TokenStream> {
        let list = match (input.is_current_row(), input.is_main_table_column()) {
            (true, true) => IOList::CurrMainRow,
            (true, false) => IOList::CurrAuxRow,
            (false, true) => IOList::NextMainRow,
            (false, false) => IOList::NextAuxRow,
        };
        if self.input_location_is_static {
            Self::load_ext_field_element_from_list(list, input.column())
        } else {
            Self::load_ext_field_element_from_pointed_to_list(list, input.column())
        }
    }

    fn load_challenge(challenge_idx: usize) -> Vec<TokenStream> {
        Self::load_ext_field_element_from_list(IOList::Challenges, challenge_idx)
    }

    fn load_evaluated_bin_op(node_id: usize) -> Vec<TokenStream> {
        Self::load_ext_field_element_from_list(IOList::FreeMemPage, node_id)
    }

    fn load_ext_field_element_from_list(list: IOList, element_index: usize) -> Vec<TokenStream> {
        let word_index = Self::element_index_to_word_index_for_reading(element_index);

        [
            push!(list + word_index),
            instr!(ReadMem(NumberOfWords::N3)),
            instr!(Pop(NumberOfWords::N1)),
        ]
        .concat()
    }

    fn load_ext_field_element_from_pointed_to_list(
        list: IOList,
        element_index: usize,
    ) -> Vec<TokenStream> {
        let word_index = Self::element_index_to_word_index_for_reading(element_index);

        [
            push!(list + 0),
            instr!(ReadMem(NumberOfWords::N1)),
            instr!(Pop(NumberOfWords::N1)),
            instr!(AddI(word_index)),
            instr!(ReadMem(NumberOfWords::N3)),
            instr!(Pop(NumberOfWords::N1)),
        ]
        .concat()
    }

    fn element_index_to_word_index_for_reading(element_index: usize) -> BFieldElement {
        let word_offset = element_index * EXTENSION_DEGREE;
        let start_to_read_offset = EXTENSION_DEGREE - 1;
        let word_index = word_offset + start_to_read_offset;
        bfe!(u64::try_from(word_index).unwrap())
    }

    fn store_ext_field_element(element_index: usize) -> Vec<TokenStream> {
        let free_mem_page = IOList::FreeMemPage;

        let word_offset = element_index * EXTENSION_DEGREE;
        let word_index = u64::try_from(word_offset).unwrap();

        let push_address = push!(free_mem_page + word_index);
        let write_mem = instr!(WriteMem(NumberOfWords::N3));
        let pop = instr!(Pop(NumberOfWords::N1));

        [push_address, write_mem, pop].concat()
    }

    fn prepare_return_values() -> Vec<TokenStream> {
        let free_mem_page = IOList::FreeMemPage;
        let out_array_offset_in_num_bfes = OUT_ARRAY_OFFSET * EXTENSION_DEGREE;
        let out_array_offset = u64::try_from(out_array_offset_in_num_bfes).unwrap();
        push!(free_mem_page + out_array_offset)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
enum IOList {
    FreeMemPage,
    CurrMainRow,
    CurrAuxRow,
    NextMainRow,
    NextAuxRow,
    Challenges,
}

impl ToTokens for IOList {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            IOList::FreeMemPage => tokens.extend(quote!(free_mem_page_ptr)),
            IOList::CurrMainRow => tokens.extend(quote!(curr_main_row_ptr)),
            IOList::CurrAuxRow => tokens.extend(quote!(curr_aux_row_ptr)),
            IOList::NextMainRow => tokens.extend(quote!(next_main_row_ptr)),
            IOList::NextAuxRow => tokens.extend(quote!(next_aux_row_ptr)),
            IOList::Challenges => tokens.extend(quote!(challenges_ptr)),
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use constraint_circuit::ConstraintCircuitBuilder;
    use constraint_circuit::SingleRowIndicator;
    use twenty_first::prelude::*;

    use super::*;

    pub(crate) fn mini_constraints() -> Constraints {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let challenge = |c: usize| circuit_builder.challenge(c);
        let constant = |c: u32| circuit_builder.x_constant(c);
        let main_row = |i| circuit_builder.input(SingleRowIndicator::Main(i));
        let aux_row = |i| circuit_builder.input(SingleRowIndicator::Aux(i));

        let constraint = main_row(0) * challenge(3) - aux_row(1) * constant(42);

        Constraints {
            init: vec![constraint],
            cons: vec![],
            tran: vec![],
            term: vec![],
        }
    }

    pub fn print_constraints<B: Codegen>(constraints: &Constraints) {
        let code = B::constraint_evaluation_code(constraints);
        let syntax_tree = syn::parse2(code).unwrap();
        let code = prettyplease::unparse(&syntax_tree);
        println!("{code}");
    }

    #[test]
    fn tokenizing_base_field_elements_produces_expected_result() {
        insta::assert_snapshot!(
            RustBackend::tokenize_bfe(bfe!(42)),
            @"BFieldElement :: from_raw_u64 (180388626390u64)",
        );
    }

    #[test]
    fn tokenizing_extension_field_elements_produces_expected_result() {
        insta::assert_snapshot!(
            RustBackend::tokenize_xfe(xfe!([42, 43, 44])),
            @"XFieldElement :: new ([\
                BFieldElement :: from_raw_u64 (180388626390u64) , \
                BFieldElement :: from_raw_u64 (184683593685u64) , \
                BFieldElement :: from_raw_u64 (188978560980u64)\
            ])",
        );
    }

    #[test]
    fn print_mini_constraints_rust() {
        print_constraints::<RustBackend>(&mini_constraints());
    }

    #[test]
    fn print_mini_constraints_tasm() {
        print_constraints::<TasmBackend>(&mini_constraints());
    }
}
