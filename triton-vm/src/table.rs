use air::table::cascade::CascadeTable;
use air::table::hash::HashTable;
use air::table::jump_stack::JumpStackTable;
use air::AIR;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use strum::Display;
use strum::EnumCount;
use strum::EnumIter;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::challenges::Challenges;
pub use crate::stark::NUM_QUOTIENT_SEGMENTS;
use crate::table::master_table::MasterBaseTable;
use crate::table::master_table::MasterExtTable;
use crate::table::master_table::MasterTable;

pub mod degree_lowering;

pub mod cascade;
pub mod extension_table;
pub mod hash;
pub mod jump_stack;
pub mod lookup;
pub mod master_table;
pub mod op_stack;
pub mod processor;
pub mod program;
pub mod ram;
pub mod u32;

/// The total number of main columns across all tables.
/// The degree lowering columns _are_ included.
pub const NUM_MAIN_COLUMNS: usize =
    air::table::NUM_BASE_COLUMNS + degree_lowering::DegreeLoweringBaseTableColumn::COUNT;

/// The total number of auxiliary columns across all tables.
/// The degree lowering columns _are_ included,
/// randomizer polynomials are _not_ included.
pub const NUM_AUX_COLUMNS: usize =
    air::table::NUM_EXT_COLUMNS + degree_lowering::DegreeLoweringExtTableColumn::COUNT;

trait TraceTable: AIR {
    // a nicer design is in order
    type FillParam;
    type FillReturnInfo;

    fn fill(
        main_table: ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
        _: Self::FillParam,
    ) -> Self::FillReturnInfo;

    fn pad(main_table: ArrayViewMut2<BFieldElement>, table_length: usize);

    fn extend(
        base_table: ArrayView2<BFieldElement>,
        ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    );
}

#[derive(
    Debug, Display, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, EnumCount, EnumIter,
)]
pub enum ConstraintType {
    /// Pertains only to the first row of the execution trace.
    Initial,

    /// Pertains to each row of the execution trace.
    Consistency,

    /// Pertains to each pair of consecutive rows of the execution trace.
    Transition,

    /// Pertains only to the last row of the execution trace.
    Terminal,
}

/// A single row of a [`MasterBaseTable`].
///
/// Usually, the elements in the table are [`BFieldElement`]s. For out-of-domain rows, which is
/// relevant for “Domain Extension to Eliminate Pretenders” (DEEP), the elements are
/// [`XFieldElement`]s.
pub type BaseRow<T> = [T; MasterBaseTable::NUM_COLUMNS];

/// A single row of a [`MasterExtTable`].
pub type ExtensionRow = [XFieldElement; MasterExtTable::NUM_COLUMNS];

/// An element of the split-up quotient polynomial.
///
/// See also [`NUM_QUOTIENT_SEGMENTS`].
pub type QuotientSegments = [XFieldElement; NUM_QUOTIENT_SEGMENTS];

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use air::table::hash::HashTable;
    use air::table::lookup::LookupTable;
    use air::table::op_stack::OpStackTable;
    use air::table::processor::ProcessorTable;
    use air::table::program::ProgramTable;
    use air::table::ram::RamTable;
    use air::table::u32::U32Table;
    use air::table::CASCADE_TABLE_END;
    use air::table::EXT_CASCADE_TABLE_END;
    use air::table::EXT_HASH_TABLE_END;
    use air::table::EXT_JUMP_STACK_TABLE_END;
    use air::table::EXT_LOOKUP_TABLE_END;
    use air::table::EXT_OP_STACK_TABLE_END;
    use air::table::EXT_PROCESSOR_TABLE_END;
    use air::table::EXT_PROGRAM_TABLE_END;
    use air::table::EXT_RAM_TABLE_END;
    use air::table::EXT_U32_TABLE_END;
    use air::table::HASH_TABLE_END;
    use air::table::JUMP_STACK_TABLE_END;
    use air::table::LOOKUP_TABLE_END;
    use air::table::OP_STACK_TABLE_END;
    use air::table::PROCESSOR_TABLE_END;
    use air::table::PROGRAM_TABLE_END;
    use air::table::RAM_TABLE_END;
    use air::table::U32_TABLE_END;
    use constraint_circuit::BinOp;
    use constraint_circuit::CircuitExpression;
    use constraint_circuit::ConstraintCircuit;
    use constraint_circuit::ConstraintCircuitBuilder;
    use constraint_circuit::ConstraintCircuitMonad;
    use constraint_circuit::DegreeLoweringInfo;
    use constraint_circuit::InputIndicator;
    use itertools::Itertools;
    use ndarray::Array2;
    use ndarray::ArrayView2;
    use rand::prelude::StdRng;
    use rand::random;
    use rand::Rng;
    use rand_core::SeedableRng;
    use twenty_first::prelude::BFieldElement;

    use crate::challenges::Challenges;
    use crate::prelude::Claim;
    use crate::table::degree_lowering::DegreeLoweringTable;

    use super::*;

    /// Verify that all nodes evaluate to a unique value when given a randomized input.
    /// If this is not the case two nodes that are not equal evaluate to the same value.
    fn table_constraints_prop<II: InputIndicator>(
        constraints: &[ConstraintCircuit<II>],
        table_name: &str,
    ) {
        let seed = random();
        let mut rng = StdRng::seed_from_u64(seed);
        println!("seed: {seed}");

        let dummy_claim = Claim::default();
        let challenges: [XFieldElement; Challenges::SAMPLE_COUNT] = rng.gen();
        let challenges = challenges.to_vec();
        let challenges = Challenges::new(challenges, &dummy_claim);
        let challenges = &challenges.challenges;

        let num_rows = 2;
        let base_shape = [num_rows, NUM_MAIN_COLUMNS];
        let ext_shape = [num_rows, NUM_AUX_COLUMNS];
        let base_rows = Array2::from_shape_simple_fn(base_shape, || rng.gen::<BFieldElement>());
        let ext_rows = Array2::from_shape_simple_fn(ext_shape, || rng.gen::<XFieldElement>());
        let base_rows = base_rows.view();
        let ext_rows = ext_rows.view();

        let mut values = HashMap::new();
        for c in constraints {
            evaluate_assert_unique(c, challenges, base_rows, ext_rows, &mut values);
        }

        let circuit_degree = constraints.iter().map(|c| c.degree()).max().unwrap_or(-1);
        println!("Max degree constraint for {table_name} table: {circuit_degree}");
    }

    /// Recursively evaluates the given constraint circuit and its sub-circuits on the given
    /// base and extension table, and returns the result of the evaluation.
    /// At each recursive step, updates the given HashMap with the result of the evaluation.
    /// If the HashMap already contains the result of the evaluation, panics.
    /// This function is used to assert that the evaluation of a constraint circuit
    /// and its sub-circuits is unique.
    /// It is used to identify redundant constraints or sub-circuits.
    /// The employed method is the Schwartz-Zippel lemma.
    fn evaluate_assert_unique<II: InputIndicator>(
        constraint: &ConstraintCircuit<II>,
        challenges: &[XFieldElement],
        base_rows: ArrayView2<BFieldElement>,
        ext_rows: ArrayView2<XFieldElement>,
        values: &mut HashMap<XFieldElement, (usize, ConstraintCircuit<II>)>,
    ) -> XFieldElement {
        let value = match &constraint.expression {
            CircuitExpression::BinaryOperation(binop, lhs, rhs) => {
                let lhs = lhs.borrow();
                let rhs = rhs.borrow();
                let lhs = evaluate_assert_unique(&lhs, challenges, base_rows, ext_rows, values);
                let rhs = evaluate_assert_unique(&rhs, challenges, base_rows, ext_rows, values);
                binop.operation(lhs, rhs)
            }
            _ => constraint.evaluate(base_rows, ext_rows, challenges),
        };

        let own_id = constraint.id.to_owned();
        let maybe_entry = values.insert(value, (own_id, constraint.clone()));
        if let Some((other_id, other_circuit)) = maybe_entry {
            assert_eq!(
                own_id, other_id,
                "Circuit ID {other_id} and circuit ID {own_id} are not unique. \
                Collision on:\n\
                ID {other_id} – {other_circuit}\n\
                ID {own_id} – {constraint}\n\
                Both evaluate to {value}.",
            );
        }

        value
    }

    #[test]
    fn nodes_are_unique_for_all_constraints() {
        fn build_constraints<II: InputIndicator>(
            multicircuit_builder: &dyn Fn(
                &ConstraintCircuitBuilder<II>,
            ) -> Vec<ConstraintCircuitMonad<II>>,
        ) -> Vec<ConstraintCircuit<II>> {
            let circuit_builder = ConstraintCircuitBuilder::new();
            let multicircuit = multicircuit_builder(&circuit_builder);
            let mut constraints = multicircuit.into_iter().map(|c| c.consume()).collect_vec();
            ConstraintCircuit::assert_unique_ids(&mut constraints);
            constraints
        }

        macro_rules! assert_constraint_properties {
            ($table:ident) => {{
                let init = build_constraints(&$table::initial_constraints);
                let cons = build_constraints(&$table::consistency_constraints);
                let tran = build_constraints(&$table::transition_constraints);
                let term = build_constraints(&$table::terminal_constraints);
                table_constraints_prop(&init, concat!(stringify!($table), " init"));
                table_constraints_prop(&cons, concat!(stringify!($table), " cons"));
                table_constraints_prop(&tran, concat!(stringify!($table), " tran"));
                table_constraints_prop(&term, concat!(stringify!($table), " term"));
            }};
        }

        assert_constraint_properties!(ProcessorTable);
        assert_constraint_properties!(ProgramTable);
        assert_constraint_properties!(JumpStackTable);
        assert_constraint_properties!(OpStackTable);
        assert_constraint_properties!(RamTable);
        assert_constraint_properties!(HashTable);
        assert_constraint_properties!(U32Table);
        assert_constraint_properties!(CascadeTable);
        assert_constraint_properties!(LookupTable);
    }

    /// Like [`ConstraintCircuitMonad::lower_to_degree`] with additional assertion of expected
    /// properties. Also prints:
    /// - the given multicircuit prior to degree lowering
    /// - the multicircuit after degree lowering
    /// - the new base constraints
    /// - the new extension constraints
    /// - the numbers of original and new constraints
    fn lower_degree_and_assert_properties<II: InputIndicator>(
        multicircuit: &mut [ConstraintCircuitMonad<II>],
        info: DegreeLoweringInfo,
    ) -> (
        Vec<ConstraintCircuitMonad<II>>,
        Vec<ConstraintCircuitMonad<II>>,
    ) {
        let seed = random();
        let mut rng = StdRng::seed_from_u64(seed);
        println!("seed: {seed}");

        let num_constraints = multicircuit.len();
        println!("original multicircuit:");
        for circuit in multicircuit.iter() {
            println!("  {circuit}");
        }

        let (new_base_constraints, new_ext_constraints) =
            ConstraintCircuitMonad::lower_to_degree(multicircuit, info);

        assert_eq!(num_constraints, multicircuit.len());

        let target_deg = info.target_degree;
        assert!(ConstraintCircuitMonad::multicircuit_degree(multicircuit) <= target_deg);
        assert!(ConstraintCircuitMonad::multicircuit_degree(&new_base_constraints) <= target_deg);
        assert!(ConstraintCircuitMonad::multicircuit_degree(&new_ext_constraints) <= target_deg);

        // Check that the new constraints are simple substitutions.
        let mut substitution_rules = vec![];
        for (constraint_type, constraints) in [
            ("base", &new_base_constraints),
            ("ext", &new_ext_constraints),
        ] {
            for (i, constraint) in constraints.iter().enumerate() {
                let expression = constraint.circuit.borrow().expression.clone();
                let CircuitExpression::BinaryOperation(BinOp::Add, lhs, rhs) = expression else {
                    panic!("New {constraint_type} constraint {i} must be a subtraction.");
                };
                let CircuitExpression::Input(input_indicator) = lhs.borrow().expression.clone()
                else {
                    panic!("New {constraint_type} constraint {i} must be a simple substitution.");
                };
                let substitution_rule = rhs.borrow().clone();
                assert_substitution_rule_uses_legal_variables(input_indicator, &substitution_rule);
                substitution_rules.push(substitution_rule);
            }
        }

        // Use the Schwartz-Zippel lemma to check no two substitution rules are equal.
        let dummy_claim = Claim::default();
        let challenges: [XFieldElement; Challenges::SAMPLE_COUNT] = rng.gen();
        let challenges = challenges.to_vec();
        let challenges = Challenges::new(challenges, &dummy_claim);
        let challenges = &challenges.challenges;

        let num_rows = 2;
        let num_new_base_constraints = new_base_constraints.len();
        let num_new_ext_constraints = new_ext_constraints.len();
        let num_base_cols = NUM_MAIN_COLUMNS + num_new_base_constraints;
        let num_ext_cols = NUM_AUX_COLUMNS + num_new_ext_constraints;
        let base_shape = [num_rows, num_base_cols];
        let ext_shape = [num_rows, num_ext_cols];
        let base_rows = Array2::from_shape_simple_fn(base_shape, || rng.gen::<BFieldElement>());
        let ext_rows = Array2::from_shape_simple_fn(ext_shape, || rng.gen::<XFieldElement>());
        let base_rows = base_rows.view();
        let ext_rows = ext_rows.view();

        let evaluated_substitution_rules = substitution_rules
            .iter()
            .map(|c| c.evaluate(base_rows, ext_rows, challenges));

        let mut values_to_index = HashMap::new();
        for (idx, value) in evaluated_substitution_rules.enumerate() {
            if let Some(index) = values_to_index.get(&value) {
                panic!("Substitution {idx} must be distinct from substitution {index}.");
            } else {
                values_to_index.insert(value, idx);
            }
        }

        // Print the multicircuit and new constraints after degree lowering.
        println!("new multicircuit:");
        for circuit in multicircuit.iter() {
            println!("  {circuit}");
        }
        println!("new base constraints:");
        for constraint in &new_base_constraints {
            println!("  {constraint}");
        }
        println!("new ext constraints:");
        for constraint in &new_ext_constraints {
            println!("  {constraint}");
        }

        println!(
            "Started with {num_constraints} constraints. \
            Derived {num_new_base_constraints} new base, \
            {num_new_ext_constraints} new extension constraints."
        );

        (new_base_constraints, new_ext_constraints)
    }

    /// Panics if the given substitution rule uses variables with an index greater than (or equal)
    /// to the given index. In practice, this given index corresponds to a newly introduced
    /// variable.
    fn assert_substitution_rule_uses_legal_variables<II: InputIndicator>(
        new_var: II,
        substitution_rule: &ConstraintCircuit<II>,
    ) {
        match substitution_rule.expression.clone() {
            CircuitExpression::BinaryOperation(_, lhs, rhs) => {
                let lhs = lhs.borrow();
                let rhs = rhs.borrow();
                assert_substitution_rule_uses_legal_variables(new_var, &lhs);
                assert_substitution_rule_uses_legal_variables(new_var, &rhs);
            }
            CircuitExpression::Input(old_var) => {
                let new_var_is_base = new_var.is_base_table_column();
                let old_var_is_base = old_var.is_base_table_column();
                let legal_substitute = match (new_var_is_base, old_var_is_base) {
                    (true, false) => false,
                    (false, true) => true,
                    _ => old_var.column() < new_var.column(),
                };
                assert!(legal_substitute, "Cannot replace {old_var} with {new_var}.");
            }
            _ => (),
        };
    }

    #[test]
    fn degree_lowering_works_correctly_for_all_tables() {
        macro_rules! assert_degree_lowering {
            ($table:ident ($base_end:ident, $ext_end:ident)) => {{
                let degree_lowering_info = DegreeLoweringInfo {
                    target_degree: air::TARGET_DEGREE,
                    num_base_cols: $base_end,
                    num_ext_cols: $ext_end,
                };
                let circuit_builder = ConstraintCircuitBuilder::new();
                let mut init = $table::initial_constraints(&circuit_builder);
                lower_degree_and_assert_properties(&mut init, degree_lowering_info);

                let circuit_builder = ConstraintCircuitBuilder::new();
                let mut cons = $table::consistency_constraints(&circuit_builder);
                lower_degree_and_assert_properties(&mut cons, degree_lowering_info);

                let circuit_builder = ConstraintCircuitBuilder::new();
                let mut tran = $table::transition_constraints(&circuit_builder);
                lower_degree_and_assert_properties(&mut tran, degree_lowering_info);

                let circuit_builder = ConstraintCircuitBuilder::new();
                let mut term = $table::terminal_constraints(&circuit_builder);
                lower_degree_and_assert_properties(&mut term, degree_lowering_info);
            }};
        }

        assert_degree_lowering!(ProgramTable(PROGRAM_TABLE_END, EXT_PROGRAM_TABLE_END));
        assert_degree_lowering!(ProcessorTable(PROCESSOR_TABLE_END, EXT_PROCESSOR_TABLE_END));
        assert_degree_lowering!(OpStackTable(OP_STACK_TABLE_END, EXT_OP_STACK_TABLE_END));
        assert_degree_lowering!(RamTable(RAM_TABLE_END, EXT_RAM_TABLE_END));
        assert_degree_lowering!(JumpStackTable(
            JUMP_STACK_TABLE_END,
            EXT_JUMP_STACK_TABLE_END
        ));
        assert_degree_lowering!(HashTable(HASH_TABLE_END, EXT_HASH_TABLE_END));
        assert_degree_lowering!(CascadeTable(CASCADE_TABLE_END, EXT_CASCADE_TABLE_END));
        assert_degree_lowering!(LookupTable(LOOKUP_TABLE_END, EXT_LOOKUP_TABLE_END));
        assert_degree_lowering!(U32Table(U32_TABLE_END, EXT_U32_TABLE_END));
    }

    /// Fills the derived columns of the degree-lowering table using randomly generated rows and
    /// checks the resulting values for uniqueness. The described method corresponds to an
    /// application of the Schwartz-Zippel lemma to check uniqueness of the substitution rules
    /// generated during degree lowering.
    #[test]
    #[ignore = "(probably) requires normalization of circuit expressions"]
    fn substitution_rules_are_unique() {
        let challenges = Challenges::default();
        let mut base_table_rows = Array2::from_shape_fn((2, NUM_MAIN_COLUMNS), |_| random());
        let mut ext_table_rows = Array2::from_shape_fn((2, NUM_AUX_COLUMNS), |_| random());

        DegreeLoweringTable::fill_derived_base_columns(base_table_rows.view_mut());
        DegreeLoweringTable::fill_derived_ext_columns(
            base_table_rows.view(),
            ext_table_rows.view_mut(),
            &challenges,
        );

        let mut encountered_values = HashMap::new();
        for col_idx in 0..NUM_MAIN_COLUMNS {
            let val = base_table_rows[(0, col_idx)].lift();
            let other_entry = encountered_values.insert(val, col_idx);
            if let Some(other_idx) = other_entry {
                panic!("Duplicate value {val} in derived base column {other_idx} and {col_idx}.");
            }
        }
        println!("Now comparing extension columns…");
        for col_idx in 0..NUM_AUX_COLUMNS {
            let val = ext_table_rows[(0, col_idx)];
            let other_entry = encountered_values.insert(val, col_idx);
            if let Some(other_idx) = other_entry {
                panic!("Duplicate value {val} in derived ext column {other_idx} and {col_idx}.");
            }
        }
    }
}
