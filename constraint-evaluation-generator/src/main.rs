use std::fs::write;

use proc_macro2::TokenStream;

use crate::constraints::Constraints;

mod asm;
mod constraints;
mod rust;
mod substitution;

fn main() {
    let mut constraints = Constraints::all();
    constraints.fold_constants();
    let substitutions = constraints.lower_to_target_degree_through_substitutions();
    let degree_lowering_table_code = substitutions.generate_degree_lowering_table_code();

    let constraints = constraints.combine_with_substitution_induced_constraints(substitutions);
    let rust_code = constraints.generate_rust_code();
    let asm_code = constraints.generate_triton_assembly();

    write_code_to_file(degree_lowering_table_code, "degree_lowering_table");
    write_code_to_file(rust_code, "constraints");
    write_code_to_file(asm_code, "asm_air_constraints");
}

fn write_code_to_file(code: TokenStream, file_name: &str) {
    let syntax_tree = syn::parse2(code).unwrap();
    let code = prettyplease::unparse(&syntax_tree);
    let path = format!("triton-vm/src/table/{file_name}.rs");
    write(path, code).unwrap();
}
