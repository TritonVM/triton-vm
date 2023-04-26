use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::cascade_table::ExtCascadeTable;
use crate::table::extension_table::Evaluable;
use crate::table::extension_table::Quotientable;

// This file is a placeholder for auto-generated code
// Run `cargo run --bin constraint-evaluation-generator`
// to fill in this file with optimized constraints.
impl Evaluable<BFieldElement> for ExtCascadeTable {}
impl Evaluable<XFieldElement> for ExtCascadeTable {}

impl Quotientable for ExtCascadeTable {}
