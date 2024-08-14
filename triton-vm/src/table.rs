use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;

use arbitrary::Arbitrary;
use strum::EnumCount;
use strum::EnumIter;
use twenty_first::prelude::XFieldElement;

pub use crate::stark::NUM_QUOTIENT_SEGMENTS;
pub use crate::table::master_table::NUM_BASE_COLUMNS;
pub use crate::table::master_table::NUM_EXT_COLUMNS;

pub mod cascade_table;
pub mod challenges;
pub mod constraint_circuit;
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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, EnumCount, EnumIter)]
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

impl Display for ConstraintType {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            ConstraintType::Initial => write!(f, "initial"),
            ConstraintType::Consistency => write!(f, "consistency"),
            ConstraintType::Transition => write!(f, "transition"),
            ConstraintType::Terminal => write!(f, "terminal"),
        }
    }
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
