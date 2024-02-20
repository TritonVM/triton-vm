use proc_macro2::TokenStream;
use quote::quote;

use crate::Constraints;

impl Constraints {
    /// Emits a function that emits [Triton assembly][tasm] that evaluates Triton VM's AIR
    /// constraints over the [extension field][XFieldElement].
    ///
    /// [tasm]: triton_vm::prelude::triton_asm
    pub fn generate_triton_assembly(&self) -> TokenStream {
        let uses = Self::generate_asm_uses();
        let doc_comment = self.generate_doc_comment();

        quote!(
            #uses
            #[doc = #doc_comment]
            pub fn air_constraint_evaluation_tasm() -> Box<[LabelledInstruction]> {
                todo!()
            }
        )
    }

    fn generate_asm_uses() -> TokenStream {
        quote!(
            use crate::instruction::LabelledInstruction;
        )
    }

    fn generate_doc_comment(&self) -> String {
        let num_init_constraints = self.init.len();
        let num_cons_constraints = self.cons.len();
        let num_tran_constraints = self.tran.len();
        let num_term_constraints = self.term.len();
        let num_total_constraints = self.len();

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
}
