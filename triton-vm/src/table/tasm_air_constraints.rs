//! This file is a placeholder for auto-generated code
//! Run `cargo run --bin constraint-evaluation-generator`
//! to fill in this file with optimized constraints.

use twenty_first::prelude::*;

use crate::instruction::LabelledInstruction;
use crate::table::constraints::ERROR_MESSAGE_GENERATE_CONSTRAINTS;
use crate::table::IOList;
use crate::table::TasmConstraintInstantiator;

impl TasmConstraintInstantiator {
    pub(super) fn instantiate_initial_constraints(&mut self) -> Vec<LabelledInstruction> {
        // suppress “unused code” warnings before code generation
        Self::load_ext_field_constant(xfe!(0));
        self.load_ext_field_element_from_list(IOList::CurrBaseRow, 0);
        self.load_ext_field_element_from_list(IOList::CurrExtRow, 0);
        self.load_ext_field_element_from_list(IOList::NextBaseRow, 0);
        self.load_ext_field_element_from_list(IOList::NextExtRow, 0);
        self.load_ext_field_element_from_list(IOList::Challenges, 0);
        self.store_ext_field_element(0);
        self.write_into_output_list();

        panic!("{ERROR_MESSAGE_GENERATE_CONSTRAINTS}")
    }

    pub(super) fn instantiate_consistency_constraints(&mut self) -> Vec<LabelledInstruction> {
        panic!("{ERROR_MESSAGE_GENERATE_CONSTRAINTS}")
    }

    pub(super) fn instantiate_transition_constraints(&mut self) -> Vec<LabelledInstruction> {
        panic!("{ERROR_MESSAGE_GENERATE_CONSTRAINTS}")
    }

    pub(super) fn instantiate_terminal_constraints(&mut self) -> Vec<LabelledInstruction> {
        panic!("{ERROR_MESSAGE_GENERATE_CONSTRAINTS}")
    }
}
