// See the corresponding attribute in triton_vm/lib.rs
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use air::AIR;
use air::cross_table_argument::GrandCrossTableArg;
use air::table::cascade::CascadeTable;
use air::table::hash::HashTable;
use air::table::jump_stack::JumpStackTable;
use air::table::lookup::LookupTable;
use air::table::op_stack::OpStackTable;
use air::table::processor::ProcessorTable;
use air::table::program::ProgramTable;
use air::table::ram::RamTable;
use air::table::u32::U32Table;
use constraint_circuit::ConstraintCircuit;
use constraint_circuit::ConstraintCircuitBuilder;
use constraint_circuit::ConstraintCircuitMonad;
use constraint_circuit::DegreeLoweringInfo;
use constraint_circuit::DualRowIndicator;
use constraint_circuit::InputIndicator;
use constraint_circuit::SingleRowIndicator;
use itertools::Itertools;

use crate::substitutions::AllSubstitutions;
use crate::substitutions::Substitutions;

pub mod codegen;
mod substitutions;

#[derive(Debug, Clone)]
pub struct Constraints {
    pub init: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
    pub cons: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
    pub tran: Vec<ConstraintCircuitMonad<DualRowIndicator>>,
    pub term: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
}

impl Constraints {
    pub fn all() -> Constraints {
        Constraints {
            init: Self::initial_constraints(),
            cons: Self::consistency_constraints(),
            tran: Self::transition_constraints(),
            term: Self::terminal_constraints(),
        }
    }

    // Implementing Default for DegreeLoweringInfo is impossible because the
    // constants are defined in crate `air` but struct `DegreeLoweringInfo` is
    // defined in crate `triton-constraint-circuit`. Cfr. orphan rule.
    pub fn default_degree_lowering_info() -> DegreeLoweringInfo {
        constraint_circuit::DegreeLoweringInfo {
            target_degree: air::TARGET_DEGREE,
            num_main_cols: air::table::NUM_MAIN_COLUMNS,
            num_aux_cols: air::table::NUM_AUX_COLUMNS,
        }
    }

    pub fn initial_constraints() -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let circuit_builder = ConstraintCircuitBuilder::new();

        [
            ProgramTable::initial_constraints(&circuit_builder),
            ProcessorTable::initial_constraints(&circuit_builder),
            OpStackTable::initial_constraints(&circuit_builder),
            RamTable::initial_constraints(&circuit_builder),
            JumpStackTable::initial_constraints(&circuit_builder),
            HashTable::initial_constraints(&circuit_builder),
            CascadeTable::initial_constraints(&circuit_builder),
            LookupTable::initial_constraints(&circuit_builder),
            U32Table::initial_constraints(&circuit_builder),
            GrandCrossTableArg::initial_constraints(&circuit_builder),
        ]
        .concat()
    }

    pub fn consistency_constraints() -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let circuit_builder = ConstraintCircuitBuilder::new();

        [
            ProgramTable::consistency_constraints(&circuit_builder),
            ProcessorTable::consistency_constraints(&circuit_builder),
            OpStackTable::consistency_constraints(&circuit_builder),
            RamTable::consistency_constraints(&circuit_builder),
            JumpStackTable::consistency_constraints(&circuit_builder),
            HashTable::consistency_constraints(&circuit_builder),
            CascadeTable::consistency_constraints(&circuit_builder),
            LookupTable::consistency_constraints(&circuit_builder),
            U32Table::consistency_constraints(&circuit_builder),
            GrandCrossTableArg::consistency_constraints(&circuit_builder),
        ]
        .concat()
    }

    pub fn transition_constraints() -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let circuit_builder = ConstraintCircuitBuilder::new();

        [
            ProgramTable::transition_constraints(&circuit_builder),
            ProcessorTable::transition_constraints(&circuit_builder),
            OpStackTable::transition_constraints(&circuit_builder),
            RamTable::transition_constraints(&circuit_builder),
            JumpStackTable::transition_constraints(&circuit_builder),
            HashTable::transition_constraints(&circuit_builder),
            CascadeTable::transition_constraints(&circuit_builder),
            LookupTable::transition_constraints(&circuit_builder),
            U32Table::transition_constraints(&circuit_builder),
            GrandCrossTableArg::transition_constraints(&circuit_builder),
        ]
        .concat()
    }

    pub fn terminal_constraints() -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let circuit_builder = ConstraintCircuitBuilder::new();

        [
            ProgramTable::terminal_constraints(&circuit_builder),
            ProcessorTable::terminal_constraints(&circuit_builder),
            OpStackTable::terminal_constraints(&circuit_builder),
            RamTable::terminal_constraints(&circuit_builder),
            JumpStackTable::terminal_constraints(&circuit_builder),
            HashTable::terminal_constraints(&circuit_builder),
            CascadeTable::terminal_constraints(&circuit_builder),
            LookupTable::terminal_constraints(&circuit_builder),
            U32Table::terminal_constraints(&circuit_builder),
            GrandCrossTableArg::terminal_constraints(&circuit_builder),
        ]
        .concat()
    }

    pub fn lower_to_target_degree_through_substitutions(
        &mut self,
        lowering_info: DegreeLoweringInfo,
    ) -> AllSubstitutions {
        let mut info = lowering_info;

        let (init_main_substitutions, init_aux_substitutions) =
            ConstraintCircuitMonad::lower_to_degree(&mut self.init, info);
        info.num_main_cols += init_main_substitutions.len();
        info.num_aux_cols += init_aux_substitutions.len();

        let (cons_main_substitutions, cons_aux_substitutions) =
            ConstraintCircuitMonad::lower_to_degree(&mut self.cons, info);
        info.num_main_cols += cons_main_substitutions.len();
        info.num_aux_cols += cons_aux_substitutions.len();

        let (tran_main_substitutions, tran_aux_substitutions) =
            ConstraintCircuitMonad::lower_to_degree(&mut self.tran, info);
        info.num_main_cols += tran_main_substitutions.len();
        info.num_aux_cols += tran_aux_substitutions.len();

        let (term_main_substitutions, term_aux_substitutions) =
            ConstraintCircuitMonad::lower_to_degree(&mut self.term, info);

        AllSubstitutions {
            main: Substitutions {
                lowering_info,
                init: init_main_substitutions,
                cons: cons_main_substitutions,
                tran: tran_main_substitutions,
                term: term_main_substitutions,
            },
            aux: Substitutions {
                lowering_info,
                init: init_aux_substitutions,
                cons: cons_aux_substitutions,
                tran: tran_aux_substitutions,
                term: term_aux_substitutions,
            },
        }
    }

    #[must_use]
    pub fn combine_with_substitution_induced_constraints(
        self,
        AllSubstitutions { main, aux }: AllSubstitutions,
    ) -> Self {
        Self {
            init: [self.init, main.init, aux.init].concat(),
            cons: [self.cons, main.cons, aux.cons].concat(),
            tran: [self.tran, main.tran, aux.tran].concat(),
            term: [self.term, main.term, aux.term].concat(),
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
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use constraint_circuit::ConstraintCircuitBuilder;
    use twenty_first::prelude::*;

    use super::*;
    use crate::codegen::RustBackend;
    use crate::codegen::TasmBackend;

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
            num_main_cols: 42,
            num_aux_cols: 13,
        }
    }

    #[test]
    fn public_types_implement_usual_auto_traits() {
        fn implements_auto_traits<T: Sized + Send + Sync + Unpin>() {}

        implements_auto_traits::<RustBackend>();
        implements_auto_traits::<TasmBackend>();

        // maybe some day
        // implements_auto_traits::<Constraints>();
        // implements_auto_traits::<substitutions::Substitutions>();
        // implements_auto_traits::<substitutions::AllSubstitutions>();
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
        let mut constraints = Constraints::all();
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
        /// For testing purposes only. There is no meaning behind any of the
        /// constraints.
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
            let input = |i| circuit_builder.input(SingleRowIndicator::Main(i));
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

            let curr_b_row = |col| circuit_builder.input(DualRowIndicator::CurrentMain(col));
            let next_b_row = |col| circuit_builder.input(DualRowIndicator::NextMain(col));
            let curr_x_row = |col| circuit_builder.input(DualRowIndicator::CurrentAux(col));
            let next_x_row = |col| circuit_builder.input(DualRowIndicator::NextAux(col));

            vec![
                curr_b_row(0) * next_x_row(1) - next_b_row(1) * curr_x_row(0),
                curr_b_row(1) * next_x_row(2) - next_b_row(2) * curr_x_row(1),
                curr_b_row(2) * next_x_row(0) * next_x_row(1) * next_x_row(3) + constant(42),
                curr_b_row(0) * challenge(TestChallenges::Ch0) - challenge(TestChallenges::Ch1),
            ]
        }
    }
}
