use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::extension_table::Evaluable;
use crate::table::extension_table::Quotientable;
use crate::table::program_table::ExtProgramTable;

// This file is a placeholder for auto-generated code
// Run `cargo run --bin constraint-evaluation-generator`
// to fill in this file with optimized constraints.
impl Evaluable<BFieldElement> for ExtProgramTable {}
impl Evaluable<XFieldElement> for ExtProgramTable {}

impl Quotientable for ExtProgramTable {}
