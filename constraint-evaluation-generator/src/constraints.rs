use triton_vm::table;
use triton_vm::table::cascade_table::ExtCascadeTable;
use triton_vm::table::constraint_circuit::ConstraintCircuitBuilder;
use triton_vm::table::constraint_circuit::ConstraintCircuitMonad;
use triton_vm::table::constraint_circuit::DualRowIndicator;
use triton_vm::table::constraint_circuit::SingleRowIndicator;
use triton_vm::table::cross_table_argument::GrandCrossTableArg;
use triton_vm::table::degree_lowering_table;
use triton_vm::table::hash_table::ExtHashTable;
use triton_vm::table::jump_stack_table::ExtJumpStackTable;
use triton_vm::table::lookup_table::ExtLookupTable;
use triton_vm::table::master_table;
use triton_vm::table::op_stack_table::ExtOpStackTable;
use triton_vm::table::processor_table::ExtProcessorTable;
use triton_vm::table::program_table::ExtProgramTable;
use triton_vm::table::ram_table::ExtRamTable;
use triton_vm::table::u32_table::ExtU32Table;

use crate::substitution::AllSubstitutions;
use crate::substitution::Substitutions;

pub(crate) struct Constraints {
    pub init: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
    pub cons: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
    pub tran: Vec<ConstraintCircuitMonad<DualRowIndicator>>,
    pub term: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
}

impl Constraints {
    pub fn all() -> Self {
        Self {
            init: Self::initial_constraints(),
            cons: Self::consistency_constraints(),
            tran: Self::transition_constraints(),
            term: Self::terminal_constraints(),
        }
    }

    fn initial_constraints() -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let circuit_builder = ConstraintCircuitBuilder::new();
        vec![
            ExtProgramTable::initial_constraints(&circuit_builder),
            ExtProcessorTable::initial_constraints(&circuit_builder),
            ExtOpStackTable::initial_constraints(&circuit_builder),
            ExtRamTable::initial_constraints(&circuit_builder),
            ExtJumpStackTable::initial_constraints(&circuit_builder),
            ExtHashTable::initial_constraints(&circuit_builder),
            ExtCascadeTable::initial_constraints(&circuit_builder),
            ExtLookupTable::initial_constraints(&circuit_builder),
            ExtU32Table::initial_constraints(&circuit_builder),
            GrandCrossTableArg::initial_constraints(&circuit_builder),
        ]
        .concat()
    }

    fn consistency_constraints() -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let circuit_builder = ConstraintCircuitBuilder::new();
        vec![
            ExtProgramTable::consistency_constraints(&circuit_builder),
            ExtProcessorTable::consistency_constraints(&circuit_builder),
            ExtOpStackTable::consistency_constraints(&circuit_builder),
            ExtRamTable::consistency_constraints(&circuit_builder),
            ExtJumpStackTable::consistency_constraints(&circuit_builder),
            ExtHashTable::consistency_constraints(&circuit_builder),
            ExtCascadeTable::consistency_constraints(&circuit_builder),
            ExtLookupTable::consistency_constraints(&circuit_builder),
            ExtU32Table::consistency_constraints(&circuit_builder),
            GrandCrossTableArg::consistency_constraints(&circuit_builder),
        ]
        .concat()
    }

    fn transition_constraints() -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let circuit_builder = ConstraintCircuitBuilder::new();
        vec![
            ExtProgramTable::transition_constraints(&circuit_builder),
            ExtProcessorTable::transition_constraints(&circuit_builder),
            ExtOpStackTable::transition_constraints(&circuit_builder),
            ExtRamTable::transition_constraints(&circuit_builder),
            ExtJumpStackTable::transition_constraints(&circuit_builder),
            ExtHashTable::transition_constraints(&circuit_builder),
            ExtCascadeTable::transition_constraints(&circuit_builder),
            ExtLookupTable::transition_constraints(&circuit_builder),
            ExtU32Table::transition_constraints(&circuit_builder),
            GrandCrossTableArg::transition_constraints(&circuit_builder),
        ]
        .concat()
    }

    fn terminal_constraints() -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let circuit_builder = ConstraintCircuitBuilder::new();
        vec![
            ExtProgramTable::terminal_constraints(&circuit_builder),
            ExtProcessorTable::terminal_constraints(&circuit_builder),
            ExtOpStackTable::terminal_constraints(&circuit_builder),
            ExtRamTable::terminal_constraints(&circuit_builder),
            ExtJumpStackTable::terminal_constraints(&circuit_builder),
            ExtHashTable::terminal_constraints(&circuit_builder),
            ExtCascadeTable::terminal_constraints(&circuit_builder),
            ExtLookupTable::terminal_constraints(&circuit_builder),
            ExtU32Table::terminal_constraints(&circuit_builder),
            GrandCrossTableArg::terminal_constraints(&circuit_builder),
        ]
        .concat()
    }

    pub fn len(&self) -> usize {
        self.init.len() + self.cons.len() + self.tran.len() + self.term.len()
    }

    pub fn fold_constants(&mut self) {
        ConstraintCircuitMonad::constant_folding(&mut self.init);
        ConstraintCircuitMonad::constant_folding(&mut self.cons);
        ConstraintCircuitMonad::constant_folding(&mut self.tran);
        ConstraintCircuitMonad::constant_folding(&mut self.term);
    }

    pub fn lower_to_target_degree_through_substitutions(&mut self) -> AllSubstitutions {
        // Subtract the degree lowering table's width from the total number of columns to guarantee
        // the same number of columns even for repeated runs of the constraint evaluation generator.
        let mut num_base_cols = table::NUM_BASE_COLUMNS - degree_lowering_table::BASE_WIDTH;
        let mut num_ext_cols = table::NUM_EXT_COLUMNS - degree_lowering_table::EXT_WIDTH;
        let (init_base_substitutions, init_ext_substitutions) =
            ConstraintCircuitMonad::lower_to_degree(
                &mut self.init,
                master_table::AIR_TARGET_DEGREE,
                num_base_cols,
                num_ext_cols,
            );
        num_base_cols += init_base_substitutions.len();
        num_ext_cols += init_ext_substitutions.len();

        let (cons_base_substitutions, cons_ext_substitutions) =
            ConstraintCircuitMonad::lower_to_degree(
                &mut self.cons,
                master_table::AIR_TARGET_DEGREE,
                num_base_cols,
                num_ext_cols,
            );
        num_base_cols += cons_base_substitutions.len();
        num_ext_cols += cons_ext_substitutions.len();

        let (tran_base_substitutions, tran_ext_substitutions) =
            ConstraintCircuitMonad::lower_to_degree(
                &mut self.tran,
                master_table::AIR_TARGET_DEGREE,
                num_base_cols,
                num_ext_cols,
            );
        num_base_cols += tran_base_substitutions.len();
        num_ext_cols += tran_ext_substitutions.len();

        let (term_base_substitutions, term_ext_substitutions) =
            ConstraintCircuitMonad::lower_to_degree(
                &mut self.term,
                master_table::AIR_TARGET_DEGREE,
                num_base_cols,
                num_ext_cols,
            );

        AllSubstitutions {
            base: Substitutions {
                init: init_base_substitutions,
                cons: cons_base_substitutions,
                tran: tran_base_substitutions,
                term: term_base_substitutions,
            },
            ext: Substitutions {
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
}

#[cfg(test)]
pub(crate) mod tests {
    use triton_vm::prelude::BFieldElement;
    use triton_vm::prelude::XFieldElement;
    use triton_vm::table::challenges::ChallengeId;
    use triton_vm::table::constraint_circuit::DualRowIndicator;

    use super::*;

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
            let constant = |c: u32| circuit_builder.b_constant(BFieldElement::from(c));
            let input = |i| circuit_builder.input(SingleRowIndicator::BaseRow(i));

            let [x, y, z] = [0, 1, 2].map(input);

            let x_square = x.clone() * x.clone();
            let x_to_the_4th = x_square.clone() * x_square.clone();
            let y_square = y.clone() * y.clone();
            let y_to_the_4th = y_square.clone() * y_square.clone();

            let a = x.clone() * y.clone() - z.clone();
            let b = x_to_the_4th.clone() - challenge(ChallengeId::HashStateWeight3) - constant(16);
            let c = z - x_to_the_4th * y_to_the_4th;

            vec![a, b, c]
        }

        fn small_transition_constraints() -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
            let circuit_builder = ConstraintCircuitBuilder::new();
            let challenge = |c| circuit_builder.challenge(c);
            let constant = |c: u32| circuit_builder.x_constant(XFieldElement::from(c));

            let curr_base_row = |col| circuit_builder.input(DualRowIndicator::CurrentBaseRow(col));
            let next_base_row = |col| circuit_builder.input(DualRowIndicator::NextBaseRow(col));
            let curr_ext_row = |col| circuit_builder.input(DualRowIndicator::CurrentExtRow(col));
            let next_ext_row = |col| circuit_builder.input(DualRowIndicator::NextExtRow(col));

            let a = curr_base_row(0) * next_ext_row(1) - next_base_row(1) * curr_ext_row(0);
            let b = curr_base_row(1) * next_ext_row(2) - next_base_row(2) * curr_ext_row(1);
            let c = a.clone() * b.clone() - curr_base_row(2) * next_ext_row(3) + constant(42);
            let d = a.clone() * c.clone() - challenge(ChallengeId::HashStateWeight12);

            vec![a, b, c, d]
        }
    }
}
