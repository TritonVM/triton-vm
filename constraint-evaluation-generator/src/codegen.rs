use proc_macro2::TokenStream;
use quote::format_ident;
use quote::quote;
use twenty_first::prelude::BFieldElement;
use twenty_first::prelude::XFieldElement;

use triton_vm::op_stack::OpStackElement;

use crate::constraints::Constraints;

mod rust;
mod tasm;

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

    fn tokenize_op_stack_element(st: OpStackElement) -> TokenStream {
        let idx = st.index();
        let st = format_ident!("ST{idx}");
        quote!(OpStackElement::#st)
    }
}

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) struct RustBackend;

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) struct TasmBackend {
    /// Tracks the number of additional words on the stack when execution reaches a given point.
    /// The input arguments listed in [`TasmBackend::doc_comment`] do not contribute.
    additional_stack_size: usize,

    /// The number of elements written to the output list.
    ///
    /// See [`TasmBackend::doc_comment`] for details.
    elements_written: usize,
}
