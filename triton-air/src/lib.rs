use constraint_circuit::ConstraintCircuitBuilder;
use constraint_circuit::ConstraintCircuitMonad;
use constraint_circuit::DualRowIndicator;
use constraint_circuit::SingleRowIndicator;
use strum::EnumCount;

use crate::table_column::MasterBaseTableColumn;
use crate::table_column::MasterExtTableColumn;

pub mod challenge_id;
pub mod cross_table_argument;
pub mod table;
pub mod table_column;

/// The degree of the AIR after the degree lowering step.
///
/// Using substitution and the introduction of new variables, the degree of the AIR as specified
/// in the respective tables
/// (e.g., in [`processor_table::ExtProcessorTable::transition_constraints`])
/// is lowered to this value.
/// For example, with a target degree of 2 and a (fictional) constraint of the form
/// `a = b²·c²·d`,
/// the degree lowering step could (as one among multiple possibilities)
/// - introduce new variables `e`, `f`, and `g`,
/// - introduce new constraints `e = b²`, `f = c²`, and `g = e·f`,
/// - replace the original constraint with `a = g·d`.
///
/// The degree lowering happens in the constraint evaluation generator.
/// It can be executed by running `cargo run --bin constraint-evaluation-generator`.
/// Executing the constraint evaluator is a prerequisite for running both the Stark prover
/// and the Stark verifier.
///
/// The new variables introduced by the degree lowering step are called “derived columns.”
/// They are added to the [`DegreeLoweringTable`], whose sole purpose is to store the values
/// of these derived columns.
pub const TARGET_DEGREE: isize = 4;

pub trait AIR {
    type MainColumn: MasterBaseTableColumn + EnumCount;
    type AuxColumn: MasterExtTableColumn + EnumCount;

    fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>>;

    fn consistency_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>>;

    fn transition_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>>;

    fn terminal_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_types_implement_usual_auto_traits() {
        fn implements_auto_traits<T: Sized + Send + Sync + Unpin>() {}

        implements_auto_traits::<challenge_id::ChallengeId>();
        implements_auto_traits::<cross_table_argument::PermArg>();
        implements_auto_traits::<cross_table_argument::EvalArg>();
        implements_auto_traits::<cross_table_argument::LookupArg>();
        implements_auto_traits::<cross_table_argument::GrandCrossTableArg>();
        implements_auto_traits::<table_column::ProgramBaseTableColumn>();
        implements_auto_traits::<table_column::ProgramExtTableColumn>();
        implements_auto_traits::<table_column::ProcessorBaseTableColumn>();
        implements_auto_traits::<table_column::ProcessorExtTableColumn>();
        implements_auto_traits::<table_column::OpStackBaseTableColumn>();
        implements_auto_traits::<table_column::OpStackExtTableColumn>();
        implements_auto_traits::<table_column::RamBaseTableColumn>();
        implements_auto_traits::<table_column::RamExtTableColumn>();
        implements_auto_traits::<table_column::JumpStackBaseTableColumn>();
        implements_auto_traits::<table_column::JumpStackExtTableColumn>();
        implements_auto_traits::<table_column::HashBaseTableColumn>();
        implements_auto_traits::<table_column::HashExtTableColumn>();
        implements_auto_traits::<table_column::CascadeBaseTableColumn>();
        implements_auto_traits::<table_column::CascadeExtTableColumn>();
        implements_auto_traits::<table_column::LookupBaseTableColumn>();
        implements_auto_traits::<table_column::LookupExtTableColumn>();
        implements_auto_traits::<table_column::U32BaseTableColumn>();
        implements_auto_traits::<table_column::U32ExtTableColumn>();

        implements_auto_traits::<table::TableId>();
        implements_auto_traits::<table::cascade::CascadeTable>();
        implements_auto_traits::<table::hash::HashTable>();
        implements_auto_traits::<table::hash::HashTableMode>();
        implements_auto_traits::<table::hash::PermutationTrace>();
        implements_auto_traits::<table::jump_stack::JumpStackTable>();
        implements_auto_traits::<table::lookup::LookupTable>();
        implements_auto_traits::<table::op_stack::OpStackTable>();
        implements_auto_traits::<table::processor::ProcessorTable>();
        implements_auto_traits::<table::program::ProgramTable>();
        implements_auto_traits::<table::ram::RamTable>();
        implements_auto_traits::<table::u32::U32Table>();
    }
}
