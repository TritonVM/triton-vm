use std::path::Path;

use constraint_builder::Constraints;
use constraint_builder::codegen::Codegen;
use constraint_builder::codegen::RustBackend;
use constraint_builder::codegen::TasmBackend;
use proc_macro2::TokenStream;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    let mut constraints = Constraints::all();
    let degree_lowering_info = Constraints::default_degree_lowering_info();
    let substitutions =
        constraints.lower_to_target_degree_through_substitutions(degree_lowering_info);
    let deg_low_table = substitutions.generate_degree_lowering_table_code();

    let constraints = constraints.combine_with_substitution_induced_constraints(substitutions);
    let rust = RustBackend::constraint_evaluation_code(&constraints);
    let tasm = TasmBackend::constraint_evaluation_code(&constraints);

    write_code_to_file(deg_low_table, "degree_lowering_table.rs");
    write_code_to_file(rust, "evaluate_constraints.rs");
    write_code_to_file(tasm, "tasm_constraints.rs");
}

fn write_code_to_file(code: TokenStream, file_name: &str) {
    let syntax_tree = syn::parse2(code).unwrap();
    let code = prettyplease::unparse(&syntax_tree);

    let out_dir = std::env::var_os("OUT_DIR").unwrap();
    let file_path = Path::new(&out_dir).join(file_name);
    std::fs::write(file_path, code).unwrap();
}
