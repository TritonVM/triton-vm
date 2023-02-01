use twenty_first::shared_math::b_field_element::BFieldElement;

use super::instruction::{AnInstruction::*, LabelledInstruction, LabelledInstruction::*};
use crate::ord_n::Ord16::*;

// OpStack manipulation
pub fn pop() -> LabelledInstruction<'static> {
    Instruction(Pop, "")
}

pub fn push(value: u64) -> LabelledInstruction<'static> {
    Instruction(Push(BFieldElement::new(value)), "")
}

pub fn divine() -> LabelledInstruction<'static> {
    Instruction(Divine(None), "")
}

pub fn dup0() -> LabelledInstruction<'static> {
    Instruction(Dup(ST0), "")
}

pub fn dup1() -> LabelledInstruction<'static> {
    Instruction(Dup(ST1), "")
}

pub fn dup2() -> LabelledInstruction<'static> {
    Instruction(Dup(ST2), "")
}

pub fn dup3() -> LabelledInstruction<'static> {
    Instruction(Dup(ST3), "")
}

pub fn dup4() -> LabelledInstruction<'static> {
    Instruction(Dup(ST4), "")
}

pub fn dup5() -> LabelledInstruction<'static> {
    Instruction(Dup(ST5), "")
}

pub fn dup6() -> LabelledInstruction<'static> {
    Instruction(Dup(ST6), "")
}

pub fn dup7() -> LabelledInstruction<'static> {
    Instruction(Dup(ST7), "")
}

pub fn dup8() -> LabelledInstruction<'static> {
    Instruction(Dup(ST8), "")
}

pub fn dup9() -> LabelledInstruction<'static> {
    Instruction(Dup(ST9), "")
}

pub fn dup10() -> LabelledInstruction<'static> {
    Instruction(Dup(ST10), "")
}

pub fn dup11() -> LabelledInstruction<'static> {
    Instruction(Dup(ST11), "")
}

pub fn dup12() -> LabelledInstruction<'static> {
    Instruction(Dup(ST12), "")
}

pub fn dup13() -> LabelledInstruction<'static> {
    Instruction(Dup(ST13), "")
}

pub fn dup14() -> LabelledInstruction<'static> {
    Instruction(Dup(ST14), "")
}

pub fn dup15() -> LabelledInstruction<'static> {
    Instruction(Dup(ST15), "")
}

// There is no swap0().

pub fn swap1() -> LabelledInstruction<'static> {
    Instruction(Swap(ST1), "")
}

pub fn swap2() -> LabelledInstruction<'static> {
    Instruction(Swap(ST2), "")
}

pub fn swap3() -> LabelledInstruction<'static> {
    Instruction(Swap(ST3), "")
}

pub fn swap4() -> LabelledInstruction<'static> {
    Instruction(Swap(ST4), "")
}

pub fn swap5() -> LabelledInstruction<'static> {
    Instruction(Swap(ST5), "")
}

pub fn swap6() -> LabelledInstruction<'static> {
    Instruction(Swap(ST6), "")
}

pub fn swap7() -> LabelledInstruction<'static> {
    Instruction(Swap(ST7), "")
}

pub fn swap8() -> LabelledInstruction<'static> {
    Instruction(Swap(ST8), "")
}

pub fn swap9() -> LabelledInstruction<'static> {
    Instruction(Swap(ST9), "")
}

pub fn swap10() -> LabelledInstruction<'static> {
    Instruction(Swap(ST10), "")
}

pub fn swap11() -> LabelledInstruction<'static> {
    Instruction(Swap(ST11), "")
}

pub fn swap12() -> LabelledInstruction<'static> {
    Instruction(Swap(ST12), "")
}

pub fn swap13() -> LabelledInstruction<'static> {
    Instruction(Swap(ST13), "")
}

pub fn swap14() -> LabelledInstruction<'static> {
    Instruction(Swap(ST14), "")
}

pub fn swap15() -> LabelledInstruction<'static> {
    Instruction(Swap(ST15), "")
}

// Control flow

pub fn nop() -> LabelledInstruction<'static> {
    Instruction(Nop, "")
}

pub fn skiz() -> LabelledInstruction<'static> {
    Instruction(Skiz, "")
}

pub fn call(label: String) -> LabelledInstruction<'static> {
    Instruction(Call(label), "")
}

pub fn return_() -> LabelledInstruction<'static> {
    Instruction(Return, "")
}

pub fn recurse() -> LabelledInstruction<'static> {
    Instruction(Recurse, "")
}

pub fn assert_() -> LabelledInstruction<'static> {
    Instruction(Assert, "")
}

pub fn halt() -> LabelledInstruction<'static> {
    Instruction(Halt, "")
}

// Memory access

pub fn read_mem() -> LabelledInstruction<'static> {
    Instruction(ReadMem, "")
}

pub fn write_mem() -> LabelledInstruction<'static> {
    Instruction(WriteMem, "")
}

// Hashing-related

pub fn hash() -> LabelledInstruction<'static> {
    Instruction(Hash, "")
}

pub fn divine_sibling() -> LabelledInstruction<'static> {
    Instruction(DivineSibling, "")
}

pub fn assert_vector() -> LabelledInstruction<'static> {
    Instruction(AssertVector, "")
}

pub fn absorb_init() -> LabelledInstruction<'static> {
    Instruction(AbsorbInit, "")
}

pub fn absorb() -> LabelledInstruction<'static> {
    Instruction(Absorb, "")
}

pub fn squeeze() -> LabelledInstruction<'static> {
    Instruction(Squeeze, "")
}

// Base field arithmetic on stack

pub fn add() -> LabelledInstruction<'static> {
    Instruction(Add, "")
}

pub fn mul() -> LabelledInstruction<'static> {
    Instruction(Mul, "")
}

pub fn invert() -> LabelledInstruction<'static> {
    Instruction(Invert, "")
}

pub fn eq() -> LabelledInstruction<'static> {
    Instruction(Eq, "")
}

// Bitwise arithmetic on stack

pub fn split() -> LabelledInstruction<'static> {
    Instruction(Split, "")
}

pub fn lt() -> LabelledInstruction<'static> {
    Instruction(Lt, "")
}

pub fn and() -> LabelledInstruction<'static> {
    Instruction(And, "")
}

pub fn xor() -> LabelledInstruction<'static> {
    Instruction(Xor, "")
}

pub fn log_2_floor() -> LabelledInstruction<'static> {
    Instruction(Log2Floor, "")
}

pub fn pow() -> LabelledInstruction<'static> {
    Instruction(Pow, "")
}

pub fn div() -> LabelledInstruction<'static> {
    Instruction(Div, "")
}

// Extension field arithmetic on stack

pub fn xxadd() -> LabelledInstruction<'static> {
    Instruction(XxAdd, "")
}

pub fn xxmul() -> LabelledInstruction<'static> {
    Instruction(XxMul, "")
}

pub fn xinvert() -> LabelledInstruction<'static> {
    Instruction(XInvert, "")
}

pub fn xbmul() -> LabelledInstruction<'static> {
    Instruction(XbMul, "")
}

// Read/write

pub fn read_io() -> LabelledInstruction<'static> {
    Instruction(ReadIo, "")
}

pub fn write_io() -> LabelledInstruction<'static> {
    Instruction(WriteIo, "")
}
