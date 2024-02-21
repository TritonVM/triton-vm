use proc_macro2::TokenStream;

use crate::constraints::Constraints;

mod rust;
mod tasm;

pub(crate) trait Codegen {
    fn constraint_evaluation_code(constraints: &Constraints) -> TokenStream;
}

pub(crate) struct RustBackend;

pub(crate) struct TasmBackend;
