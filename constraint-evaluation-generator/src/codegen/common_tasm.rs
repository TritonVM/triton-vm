use proc_macro2::TokenStream;
use quote::quote;

pub(crate) fn uses() -> TokenStream {
    quote!(
        use twenty_first::prelude::BFieldCodec;
        use twenty_first::prelude::BFieldElement;
        use crate::instruction::LabelledInstruction;
        use crate::Program;
        use crate::table::TasmConstraintEvaluationMemoryLayout;
        // for rustdoc – https://github.com/rust-lang/rust/issues/74563
        #[allow(unused_imports)]
        use crate::table::extension_table::Quotientable;
    )
}
