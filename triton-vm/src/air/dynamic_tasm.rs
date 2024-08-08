use crate::LabelledInstruction;

use super::Air;

/// Emit tasm code for evaluating the AIR constraints on an out-of-domain row
/// given pointers to:
///  - the challenges
///  - the {main,aux} {current,next} row
///  - the destination.
///
/// This is a dynamic counterpart of the automatically-generated
/// [`air_constraint_evaluation_tasm`], which emits tasm code relative to a
/// [memory layout] and without pointers (or pointer arithmetic).
///
/// [`air_constraint_evaluation_tasm`]: crate::table::tasm_air_constraints::air_constraint_evaluation_tasm
/// [memory layout]: crate::table::TasmConstraintEvaluationMemoryLayout
pub fn dynamic_air_evaluation_code_tasm(air: Air) -> Vec<LabelledInstruction> {
    todo!()
}
