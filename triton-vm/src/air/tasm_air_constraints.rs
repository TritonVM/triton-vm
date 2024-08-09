//! This file is a placeholder for auto-generated code
//! Run `cargo run --bin constraint-evaluation-generator`
//! to fill in this file with optimized constraints.

use crate::air::memory_layout::DynamicTasmConstraintEvaluationMemoryLayout;
use crate::air::memory_layout::StaticTasmConstraintEvaluationMemoryLayout;
use crate::instruction::LabelledInstruction;
use crate::table::constraints::ERROR_MESSAGE_GENERATE_CONSTRAINTS;

pub fn static_air_constraint_evaluation_tasm(
    _: StaticTasmConstraintEvaluationMemoryLayout,
) -> Vec<LabelledInstruction> {
    panic!("{ERROR_MESSAGE_GENERATE_CONSTRAINTS}");
}

pub fn dynamic_air_constraint_evaluation_tasm(
    _: DynamicTasmConstraintEvaluationMemoryLayout,
) -> Vec<LabelledInstruction> {
    panic!("{ERROR_MESSAGE_GENERATE_CONSTRAINTS}");
}
