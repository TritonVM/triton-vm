use std::cell::RefCell;
use std::rc::Rc;

use twenty_first::prelude::x_field_element::EXTENSION_DEGREE;

use crate::table::constraint_circuit::BinOp;
use crate::table::constraint_circuit::CircuitExpression;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::InputIndicator;
use crate::triton_asm;
use crate::LabelledInstruction;

use super::Air;

/// Emit tasm code for evaluating the AIR constraints on an out-of-domain row
/// given pointers to:
///  - the challenges
///  - the {main,aux} {current,next} row
///  - free memory (for shared nodes)
///  - the destination.
///
/// Specifically, the tasm code has the following signature:
///  - BEFORE: _ *challenges *main_current_row *main_next_row *aux_current_row *aux_next_row *free_mem *dest
///  - AFTER: _
///
/// This is a dynamic counterpart of the automatically-generated
/// [`air_constraint_evaluation_tasm`], which emits tasm code relative to a
/// [memory layout] and without pointers (or pointer arithmetic).
///
/// [`air_constraint_evaluation_tasm`]: crate::table::tasm_air_constraints::air_constraint_evaluation_tasm
/// [memory layout]: crate::table::TasmConstraintEvaluationMemoryLayout
pub fn dynamic_air_evaluation_code_tasm(air: Air) -> Vec<LabelledInstruction> {
    [
        compute_and_store_shared_nodes(&air.init()),
        compute_and_store_shared_nodes(&air.cons()),
        compute_and_store_shared_nodes(&air.tran()),
        compute_and_store_shared_nodes(&air.term()),
        compute_and_write_outputs(&air.init()),
        compute_and_write_outputs(&air.cons()),
        compute_and_write_outputs(&air.tran()),
        compute_and_write_outputs(&air.term()),
        clean_up_stack(),
    ]
    .concat()
}

fn compute_and_store_shared_nodes<II: InputIndicator>(
    constraints: &[ConstraintCircuit<II>],
) -> Vec<LabelledInstruction> {
    let mut nodes_already_stored = vec![];
    let mut instructions = vec![];
    for constraint in constraints {
        if let CircuitExpression::BinaryOperation(_bin_op, lhs, rhs) = &constraint.expression {
            instructions.append(&mut compute_and_store_all_ancestors_and_self(
                &mut nodes_already_stored,
                lhs,
            ));
            instructions.append(&mut compute_and_store_all_ancestors_and_self(
                &mut nodes_already_stored,
                rhs,
            ));
        }
    }
    instructions
}

fn compute_and_store_all_ancestors_and_self<II: InputIndicator>(
    nodes_already_stored: &mut Vec<usize>,
    node: &Rc<RefCell<ConstraintCircuit<II>>>,
) -> Vec<LabelledInstruction> {
    if nodes_already_stored.contains(&node.borrow().id) {
        return vec![];
    }

    let mut instructions = vec![];
    match &node.borrow().expression {
        CircuitExpression::BinaryOperation(bin_op, lhs, rhs) => {
            instructions.append(&mut compute_and_store_all_ancestors_and_self(
                nodes_already_stored,
                lhs,
            ));
            instructions.append(&mut compute_and_store_all_ancestors_and_self(
                nodes_already_stored,
                rhs,
            ));
            instructions.append(&mut load_first_operand(lhs));
            instructions.append(&mut load_second_operand(rhs));
            instructions.append(&mut compute_gate(*bin_op));
        }
        _ => {
            return vec![];
        }
    }
    instructions.append(&mut store_shared_node(node));
    nodes_already_stored.push(node.borrow().id);

    instructions
}

fn load_first_operand<II: InputIndicator>(
    node: &Rc<RefCell<ConstraintCircuit<II>>>,
) -> Vec<LabelledInstruction> {
    load_operand(node, 0)
}

fn load_second_operand<II: InputIndicator>(
    node: &Rc<RefCell<ConstraintCircuit<II>>>,
) -> Vec<LabelledInstruction> {
    load_operand(node, 3)
}

fn load_operand<II: InputIndicator>(
    node: &Rc<RefCell<ConstraintCircuit<II>>>,
    stack_offset: usize,
) -> Vec<LabelledInstruction> {
    let self_id = node.borrow().id;
    match node.borrow().expression {
        CircuitExpression::BConstant(b) => triton_asm! {
            push 0
            push 0
            push {b}
        },
        CircuitExpression::XConstant(x) => triton_asm! {
            push {x.coefficients[2]}
            push {x.coefficients[1]}
            push {x.coefficients[0]}
        },
        CircuitExpression::Input(i) => {
            let (stack_index, memory_offset) = if i.is_base_table_column() && i.is_current_row() {
                (5, i.column() * EXTENSION_DEGREE)
            } else if i.is_base_table_column() && !i.is_current_row() {
                (4, i.column() * EXTENSION_DEGREE)
            } else if !i.is_base_table_column() && i.is_current_row() {
                (3, i.column() * EXTENSION_DEGREE)
            } else if !i.is_base_table_column() && !i.is_current_row() {
                (2, i.column() * EXTENSION_DEGREE)
            } else {
                unreachable!()
            };
            triton_asm! {
                dup {stack_index + stack_offset}
                addi {memory_offset + EXTENSION_DEGREE - 1}
                read_mem {EXTENSION_DEGREE}
                pop 1
            }
        }
        CircuitExpression::Challenge(c) => triton_asm! {
            dup {6 + stack_offset}
            addi {c * EXTENSION_DEGREE + EXTENSION_DEGREE - 1}
            read_mem {EXTENSION_DEGREE}
            pop 1
        },
        CircuitExpression::BinaryOperation(_, _, _) => triton_asm! {
            dup {1 + stack_offset}
            addi {self_id * EXTENSION_DEGREE + EXTENSION_DEGREE - 1}
            read_mem {EXTENSION_DEGREE}
            pop 1
        },
    }
}

fn store_shared_node<II: InputIndicator>(
    node: &Rc<RefCell<ConstraintCircuit<II>>>,
) -> Vec<LabelledInstruction> {
    triton_asm! {
        dup 4
        addi {node.borrow().id * EXTENSION_DEGREE}
        write_mem 3
        pop 1
    }
}

fn compute_gate(bin_op: BinOp) -> Vec<LabelledInstruction> {
    match bin_op {
        BinOp::Add => triton_asm! {
            xxadd
        },
        BinOp::Mul => triton_asm! {
            xxmul
        },
    }
}

fn compute_and_write_outputs<II: InputIndicator>(
    constraints: &[ConstraintCircuit<II>],
) -> Vec<LabelledInstruction> {
    let mut instructions = vec![];
    for constraint in constraints {
        match &constraint.expression {
            CircuitExpression::BinaryOperation(bin_op, lhs, rhs) => {
                instructions.append(&mut load_first_operand(lhs));
                instructions.append(&mut load_second_operand(rhs));
                instructions.append(&mut compute_gate(*bin_op));
            }
            _ => unreachable!(),
        }
        instructions.append(&mut write_output());
    }
    instructions
}

fn write_output() -> Vec<LabelledInstruction> {
    //  _ *challenges *main_current_row *main_next_row *aux_current_row *aux_next_row *free_mem *dest [output]
    triton_asm! {
        dup 3
        write_mem {EXTENSION_DEGREE}
        swap 1
        pop 1
    }
}

fn clean_up_stack() -> Vec<LabelledInstruction> {
    triton_asm! {
        pop 4
        pop 3
    }
}
