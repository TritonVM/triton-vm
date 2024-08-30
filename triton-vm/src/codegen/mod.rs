use constraint_circuit::ConstraintCircuit;
use constraint_circuit::ConstraintCircuitMonad;
use constraint_circuit::DegreeLoweringInfo;
use constraint_circuit::DualRowIndicator;
use constraint_circuit::InputIndicator;
use constraint_circuit::SingleRowIndicator;
use itertools::Itertools;
use proc_macro2::TokenStream;
use std::fs::write;

use crate::codegen::constraints::Codegen;
use crate::codegen::constraints::RustBackend;
use crate::codegen::constraints::TasmBackend;
use crate::codegen::substitutions::AllSubstitutions;
use crate::codegen::substitutions::Substitutions;

mod constraints;
mod substitutions;

pub fn gen(mut constraints: Constraints, info: DegreeLoweringInfo) {
    let substitutions = constraints.lower_to_target_degree_through_substitutions(info);
    let degree_lowering_table_code = substitutions.generate_degree_lowering_table_code();

    let constraints = constraints.combine_with_substitution_induced_constraints(substitutions);
    let rust = RustBackend::constraint_evaluation_code(&constraints);
    let tasm = TasmBackend::constraint_evaluation_code(&constraints);

    write_code_to_file(
        degree_lowering_table_code,
        "triton-vm/src/table/degree_lowering_table.rs",
    );
    write_code_to_file(rust, "triton-vm/src/table/constraints.rs");
    write_code_to_file(tasm, "triton-vm/src/air/tasm_air_constraints.rs");
}

fn write_code_to_file(code: TokenStream, file_name: &str) {
    let syntax_tree = syn::parse2(code).unwrap();
    let code = prettyplease::unparse(&syntax_tree);
    write(file_name, code).unwrap();
}

#[derive(Debug, Clone)]
pub(crate) struct Constraints {
    pub init: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
    pub cons: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
    pub tran: Vec<ConstraintCircuitMonad<DualRowIndicator>>,
    pub term: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
}

impl Constraints {
    pub fn lower_to_target_degree_through_substitutions(
        &mut self,
        mut info: DegreeLoweringInfo,
    ) -> AllSubstitutions {
        let lowering_info = info;

        let (init_base_substitutions, init_ext_substitutions) =
            ConstraintCircuitMonad::lower_to_degree(&mut self.init, info);
        info.num_base_cols += init_base_substitutions.len();
        info.num_ext_cols += init_ext_substitutions.len();

        let (cons_base_substitutions, cons_ext_substitutions) =
            ConstraintCircuitMonad::lower_to_degree(&mut self.cons, info);
        info.num_base_cols += cons_base_substitutions.len();
        info.num_ext_cols += cons_ext_substitutions.len();

        let (tran_base_substitutions, tran_ext_substitutions) =
            ConstraintCircuitMonad::lower_to_degree(&mut self.tran, info);
        info.num_base_cols += tran_base_substitutions.len();
        info.num_ext_cols += tran_ext_substitutions.len();

        let (term_base_substitutions, term_ext_substitutions) =
            ConstraintCircuitMonad::lower_to_degree(&mut self.term, info);

        AllSubstitutions {
            base: Substitutions {
                lowering_info,
                init: init_base_substitutions,
                cons: cons_base_substitutions,
                tran: tran_base_substitutions,
                term: term_base_substitutions,
            },
            ext: Substitutions {
                lowering_info,
                init: init_ext_substitutions,
                cons: cons_ext_substitutions,
                tran: tran_ext_substitutions,
                term: term_ext_substitutions,
            },
        }
    }

    #[must_use]
    pub fn combine_with_substitution_induced_constraints(
        self,
        AllSubstitutions { base, ext }: AllSubstitutions,
    ) -> Self {
        Self {
            init: [self.init, base.init, ext.init].concat(),
            cons: [self.cons, base.cons, ext.cons].concat(),
            tran: [self.tran, base.tran, ext.tran].concat(),
            term: [self.term, base.term, ext.term].concat(),
        }
    }

    pub fn init(&self) -> Vec<ConstraintCircuit<SingleRowIndicator>> {
        Self::consume(&self.init)
    }

    pub fn cons(&self) -> Vec<ConstraintCircuit<SingleRowIndicator>> {
        Self::consume(&self.cons)
    }

    pub fn tran(&self) -> Vec<ConstraintCircuit<DualRowIndicator>> {
        Self::consume(&self.tran)
    }

    pub fn term(&self) -> Vec<ConstraintCircuit<SingleRowIndicator>> {
        Self::consume(&self.term)
    }

    fn consume<II: InputIndicator>(
        constraints: &[ConstraintCircuitMonad<II>],
    ) -> Vec<ConstraintCircuit<II>> {
        let mut constraints = constraints.iter().map(|c| c.consume()).collect_vec();
        ConstraintCircuit::assert_unique_ids(&mut constraints);
        constraints
    }
}

#[cfg(test)]
mod tests {
    use constraint_circuit::ConstraintCircuitBuilder;
    use twenty_first::prelude::*;

    use crate::table;

    use super::*;

    #[repr(usize)]
    enum TestChallenges {
        Ch0,
        Ch1,
    }

    impl From<TestChallenges> for usize {
        fn from(challenge: TestChallenges) -> Self {
            challenge as usize
        }
    }

    fn degree_lowering_info() -> DegreeLoweringInfo {
        DegreeLoweringInfo {
            target_degree: 4,
            num_base_cols: 42,
            num_ext_cols: 13,
        }
    }

    #[test]
    fn test_constraints_can_be_fetched() {
        Constraints::test_constraints();
    }

    #[test]
    fn degree_lowering_tables_code_can_be_generated_for_test_constraints() {
        let mut constraints = Constraints::test_constraints();
        let substitutions =
            constraints.lower_to_target_degree_through_substitutions(degree_lowering_info());
        let _unused = substitutions.generate_degree_lowering_table_code();
    }

    #[test]
    fn degree_lowering_tables_code_can_be_generated_from_all_constraints() {
        let mut constraints = table::constraints();
        let substitutions =
            constraints.lower_to_target_degree_through_substitutions(degree_lowering_info());
        let _unused = substitutions.generate_degree_lowering_table_code();
    }

    #[test]
    fn constraints_and_substitutions_can_be_combined() {
        let mut constraints = Constraints::test_constraints();
        let substitutions =
            constraints.lower_to_target_degree_through_substitutions(degree_lowering_info());
        let _combined = constraints.combine_with_substitution_induced_constraints(substitutions);
    }

    impl Constraints {
        /// For testing purposes only. There is no meaning behind any of the constraints.
        pub(crate) fn test_constraints() -> Self {
            Self {
                init: Self::small_init_constraints(),
                cons: vec![],
                tran: Self::small_transition_constraints(),
                term: vec![],
            }
        }

        fn small_init_constraints() -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
            let circuit_builder = ConstraintCircuitBuilder::new();
            let challenge = |c| circuit_builder.challenge(c);
            let constant = |c: u32| circuit_builder.b_constant(bfe!(c));
            let input = |i| circuit_builder.input(SingleRowIndicator::BaseRow(i));
            let input_to_the_4th = |i| input(i) * input(i) * input(i) * input(i);

            vec![
                input(0) * input(1) - input(2),
                input_to_the_4th(0) - challenge(TestChallenges::Ch1) - constant(16),
                input(2) * input_to_the_4th(0) - input_to_the_4th(1),
            ]
        }

        fn small_transition_constraints() -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
            let circuit_builder = ConstraintCircuitBuilder::new();
            let challenge = |c| circuit_builder.challenge(c);
            let constant = |c: u32| circuit_builder.x_constant(c);

            let curr_b_row = |col| circuit_builder.input(DualRowIndicator::CurrentBaseRow(col));
            let next_b_row = |col| circuit_builder.input(DualRowIndicator::NextBaseRow(col));
            let curr_x_row = |col| circuit_builder.input(DualRowIndicator::CurrentExtRow(col));
            let next_x_row = |col| circuit_builder.input(DualRowIndicator::NextExtRow(col));

            vec![
                curr_b_row(0) * next_x_row(1) - next_b_row(1) * curr_x_row(0),
                curr_b_row(1) * next_x_row(2) - next_b_row(2) * curr_x_row(1),
                curr_b_row(2) * next_x_row(0) * next_x_row(1) * next_x_row(3) + constant(42),
                curr_b_row(0) * challenge(TestChallenges::Ch0) - challenge(TestChallenges::Ch1),
            ]
        }
    }
}
