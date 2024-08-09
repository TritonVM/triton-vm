use std::collections::HashSet;

use proc_macro2::TokenStream;
use quote::quote;
use twenty_first::prelude::BFieldElement;
use twenty_first::prelude::XFieldElement;

use crate::constraints::Constraints;

mod rust;
mod static_tasm;

pub(crate) trait Codegen {
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
pub(crate) struct RustBackend {
    /// All [circuit] IDs known to be in scope.
    ///
    /// [circuit]: triton_vm::table::constraint_circuit::ConstraintCircuit
    scope: HashSet<usize>,
}

#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub(crate) struct StaticTasmBackend {
    /// All [circuit] IDs known to be processed and stored to memory.
    ///
    /// [circuit]: triton_vm::table::constraint_circuit::ConstraintCircuit
    scope: HashSet<usize>,

    /// The number of elements written to the output list.
    ///
    /// See [`TasmBackend::doc_comment`] for details.
    elements_written: usize,
}
