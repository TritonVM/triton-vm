//! This file is a placeholder for auto-generated code
//! Run `cargo run --bin constraint-evaluation-generator`
//! to fill in this file with optimized constraints.

use twenty_first::prelude::BFieldElement;
use twenty_first::prelude::XFieldElement;

use crate::table::extension_table::Evaluable;
use crate::table::extension_table::Quotientable;
use crate::table::master_table::MasterExtTable;

impl Evaluable<BFieldElement> for MasterExtTable {}

impl Evaluable<XFieldElement> for MasterExtTable {}

impl Quotientable for MasterExtTable {
    const NUM_INITIAL_CONSTRAINTS: usize = 0;
    const NUM_CONSISTENCY_CONSTRAINTS: usize = 0;
    const NUM_TRANSITION_CONSTRAINTS: usize = 0;
    const NUM_TERMINAL_CONSTRAINTS: usize = 0;
}
