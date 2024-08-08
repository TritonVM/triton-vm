pub mod dynamic_tasm;

use crate::table;
use crate::table::cascade_table::ExtCascadeTable;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::ConstraintCircuitMonad;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::cross_table_argument::GrandCrossTableArg;
use crate::table::degree_lowering_table;
use crate::table::hash_table::ExtHashTable;
use crate::table::jump_stack_table::ExtJumpStackTable;
use crate::table::lookup_table::ExtLookupTable;
use crate::table::op_stack_table::ExtOpStackTable;
use crate::table::processor_table::ExtProcessorTable;
use crate::table::program_table::ExtProgramTable;
use crate::table::ram_table::ExtRamTable;
use crate::table::u32_table::ExtU32Table;

#[derive(Debug, Clone)]
pub struct Air {
    pub init: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
    pub cons: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
    pub tran: Vec<ConstraintCircuitMonad<DualRowIndicator>>,
    pub term: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
}

macro_rules! constraints_without_degree_lowering {
    ($constraint_type: ident) => {{
        let circuit_builder = ConstraintCircuitBuilder::new();
        vec![
            ExtProgramTable::$constraint_type(&circuit_builder),
            ExtProcessorTable::$constraint_type(&circuit_builder),
            ExtOpStackTable::$constraint_type(&circuit_builder),
            ExtRamTable::$constraint_type(&circuit_builder),
            ExtJumpStackTable::$constraint_type(&circuit_builder),
            ExtHashTable::$constraint_type(&circuit_builder),
            ExtCascadeTable::$constraint_type(&circuit_builder),
            ExtLookupTable::$constraint_type(&circuit_builder),
            ExtU32Table::$constraint_type(&circuit_builder),
            GrandCrossTableArg::$constraint_type(&circuit_builder),
        ]
        .concat()
    }};
}

impl Default for Air {
    fn default() -> Self {
        Self::new()
    }
}

impl Air {
    pub fn new() -> Self {
        Self {
            init: Self::initial_constraints(),
            cons: Self::consistency_constraints(),
            tran: Self::transition_constraints(),
            term: Self::terminal_constraints(),
        }
    }

    pub fn initial_constraints() -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        constraints_without_degree_lowering!(initial_constraints)
    }

    pub fn consistency_constraints() -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        constraints_without_degree_lowering!(consistency_constraints)
    }

    pub fn transition_constraints() -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        constraints_without_degree_lowering!(transition_constraints)
    }

    pub fn terminal_constraints() -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        constraints_without_degree_lowering!(terminal_constraints)
    }

    pub fn lower_to_degree(&mut self, target_degree: isize) {
        // Subtract the degree lowering table's width from the total number of columns to guarantee
        // the same number of columns even for repeated runs of the constraint evaluation generator.
        let mut num_base_cols = table::NUM_BASE_COLUMNS - degree_lowering_table::BASE_WIDTH;
        let mut num_ext_cols = table::NUM_EXT_COLUMNS - degree_lowering_table::EXT_WIDTH;
        let (init_base_substitutions, init_ext_substitutions) =
            ConstraintCircuitMonad::lower_to_degree(
                &mut self.init,
                target_degree,
                num_base_cols,
                num_ext_cols,
            );
        num_base_cols += init_base_substitutions.len();
        num_ext_cols += init_ext_substitutions.len();

        let (cons_base_substitutions, cons_ext_substitutions) =
            ConstraintCircuitMonad::lower_to_degree(
                &mut self.cons,
                target_degree,
                num_base_cols,
                num_ext_cols,
            );
        num_base_cols += cons_base_substitutions.len();
        num_ext_cols += cons_ext_substitutions.len();

        let (tran_base_substitutions, tran_ext_substitutions) =
            ConstraintCircuitMonad::lower_to_degree(
                &mut self.tran,
                target_degree,
                num_base_cols,
                num_ext_cols,
            );
        num_base_cols += tran_base_substitutions.len();
        num_ext_cols += tran_ext_substitutions.len();

        let (term_base_substitutions, term_ext_substitutions) =
            ConstraintCircuitMonad::lower_to_degree(
                &mut self.term,
                target_degree,
                num_base_cols,
                num_ext_cols,
            );

        self.init
            .append(&mut [init_base_substitutions, init_ext_substitutions].concat());
        self.cons
            .append(&mut [cons_base_substitutions, cons_ext_substitutions].concat());
        self.tran
            .append(&mut [tran_base_substitutions, tran_ext_substitutions].concat());
        self.term
            .append(&mut [term_base_substitutions, term_ext_substitutions].concat());
    }
}
