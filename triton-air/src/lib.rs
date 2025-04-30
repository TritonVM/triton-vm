// See the corresponding attribute in triton_vm/lib.rs
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use constraint_circuit::ConstraintCircuitBuilder;
use constraint_circuit::ConstraintCircuitMonad;
use constraint_circuit::DualRowIndicator;
use constraint_circuit::SingleRowIndicator;
use strum::EnumCount;

use crate::table_column::MasterAuxColumn;
use crate::table_column::MasterMainColumn;

pub mod challenge_id;
pub mod cross_table_argument;
pub mod table;
pub mod table_column;

mod private {
    /// A public but unnameable trait to seal trait [`AIR`][super::AIR].
    pub trait Seal {}
}

/// The degree of the AIR after the degree lowering step.
///
/// Using substitution and the introduction of new variables, the degree of the
/// AIR as specified in the respective tables
/// (e.g., in [`table::processor::ProcessorTable::transition_constraints`])
/// is lowered to this value.
/// For example, with a target degree of 2 and a (fictional) constraint of the
/// form `a = b²·c²·d`,
/// the degree lowering step could (as one among multiple possibilities)
/// - introduce new variables `e`, `f`, and `g`,
/// - introduce new constraints `e = b²`, `f = c²`, and `g = e·f`,
/// - replace the original constraint with `a = g·d`.
///
/// The degree lowering happens in Triton VM's build script, `build.rs`.
pub const TARGET_DEGREE: isize = 4;

/// The main trait for the [tables]' Arithmetic Intermediate Representation.
///
/// This is a _sealed_ trait. It is not intended (or possible) to implement this
/// trait outside the crate defining it.
///
/// [tables]: table::TableId
///
/// ### Dyn Compatibility
///
/// This trait is _not_ dyn-compatible.
pub trait AIR: private::Seal {
    type MainColumn: MasterMainColumn + EnumCount;
    type AuxColumn: MasterAuxColumn + EnumCount;

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
#[cfg_attr(coverage_nightly, coverage(off))]
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
        implements_auto_traits::<table_column::ProgramMainColumn>();
        implements_auto_traits::<table_column::ProgramAuxColumn>();
        implements_auto_traits::<table_column::ProcessorMainColumn>();
        implements_auto_traits::<table_column::ProcessorAuxColumn>();
        implements_auto_traits::<table_column::OpStackMainColumn>();
        implements_auto_traits::<table_column::OpStackAuxColumn>();
        implements_auto_traits::<table_column::RamMainColumn>();
        implements_auto_traits::<table_column::RamAuxColumn>();
        implements_auto_traits::<table_column::JumpStackMainColumn>();
        implements_auto_traits::<table_column::JumpStackAuxColumn>();
        implements_auto_traits::<table_column::HashMainColumn>();
        implements_auto_traits::<table_column::HashAuxColumn>();
        implements_auto_traits::<table_column::CascadeMainColumn>();
        implements_auto_traits::<table_column::CascadeAuxColumn>();
        implements_auto_traits::<table_column::LookupMainColumn>();
        implements_auto_traits::<table_column::LookupAuxColumn>();
        implements_auto_traits::<table_column::U32MainColumn>();
        implements_auto_traits::<table_column::U32AuxColumn>();

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
