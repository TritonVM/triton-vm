pub use crate::stark::NUM_QUOTIENT_SEGMENTS;
pub use crate::table::master_table::NUM_BASE_COLUMNS;
pub use crate::table::master_table::NUM_EXT_COLUMNS;

use arbitrary::Arbitrary;
use strum::Display;
use strum::EnumCount;
use strum::EnumIter;
use twenty_first::prelude::XFieldElement;

use crate::codegen::circuit::ConstraintCircuitBuilder;
use crate::codegen::circuit::ConstraintCircuitMonad;
use crate::codegen::circuit::DualRowIndicator;
use crate::codegen::circuit::SingleRowIndicator;
use crate::codegen::Constraints;
use crate::table::cascade_table::ExtCascadeTable;
use crate::table::cross_table_argument::GrandCrossTableArg;
use crate::table::hash_table::ExtHashTable;
use crate::table::jump_stack_table::ExtJumpStackTable;
use crate::table::lookup_table::ExtLookupTable;
use crate::table::op_stack_table::ExtOpStackTable;
use crate::table::processor_table::ExtProcessorTable;
use crate::table::program_table::ExtProgramTable;
use crate::table::ram_table::ExtRamTable;
use crate::table::u32_table::ExtU32Table;

pub mod cascade_table;
pub mod challenges;
#[rustfmt::skip]
pub mod constraints;
pub mod cross_table_argument;
#[rustfmt::skip]
pub mod degree_lowering_table;
pub mod extension_table;
pub mod hash_table;
pub mod jump_stack_table;
pub mod lookup_table;
pub mod master_table;
pub mod op_stack_table;
pub mod processor_table;
pub mod program_table;
pub mod ram_table;
pub mod table_column;
pub mod u32_table;

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

/// A single row of a [`MasterBaseTable`][table].
///
/// Usually, the elements in the table are [`BFieldElement`][bfe]s. For out-of-domain rows, which is
/// relevant for “Domain Extension to Eliminate Pretenders” (DEEP), the elements are
/// [`XFieldElement`]s.
///
/// [table]: master_table::MasterBaseTable
/// [bfe]: crate::prelude::BFieldElement
pub type BaseRow<T> = [T; NUM_BASE_COLUMNS];

/// A single row of a [`MasterExtensionTable`][table].
///
/// [table]: master_table::MasterExtTable
pub type ExtensionRow = [XFieldElement; NUM_EXT_COLUMNS];

/// An element of the split-up quotient polynomial.
///
/// See also [`NUM_QUOTIENT_SEGMENTS`].
pub type QuotientSegments = [XFieldElement; NUM_QUOTIENT_SEGMENTS];

pub fn constraints() -> Constraints {
    Constraints {
        init: initial_constraints(),
        cons: consistency_constraints(),
        tran: transition_constraints(),
        term: terminal_constraints(),
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
