use twenty_first::shared_math::b_field_element::BFieldElement;

use super::instruction::{AnInstruction::*, LabelledInstruction, LabelledInstruction::*};
use crate::ord_n::Ord16::*;

// OpStack manipulation
pub fn pop() -> LabelledInstruction {
    Instruction(Pop)
}

pub fn push(value: u64) -> LabelledInstruction {
    Instruction(Push(BFieldElement::new(value)))
}

pub fn divine() -> LabelledInstruction {
    Instruction(Divine(None))
}

pub fn dup0() -> LabelledInstruction {
    Instruction(Dup(ST0))
}

pub fn dup1() -> LabelledInstruction {
    Instruction(Dup(ST1))
}

pub fn dup2() -> LabelledInstruction {
    Instruction(Dup(ST2))
}

pub fn dup3() -> LabelledInstruction {
    Instruction(Dup(ST3))
}

pub fn dup4() -> LabelledInstruction {
    Instruction(Dup(ST4))
}

pub fn dup5() -> LabelledInstruction {
    Instruction(Dup(ST5))
}

pub fn dup6() -> LabelledInstruction {
    Instruction(Dup(ST6))
}

pub fn dup7() -> LabelledInstruction {
    Instruction(Dup(ST7))
}

pub fn dup8() -> LabelledInstruction {
    Instruction(Dup(ST8))
}

pub fn dup9() -> LabelledInstruction {
    Instruction(Dup(ST9))
}

pub fn dup10() -> LabelledInstruction {
    Instruction(Dup(ST10))
}

pub fn dup11() -> LabelledInstruction {
    Instruction(Dup(ST11))
}

pub fn dup12() -> LabelledInstruction {
    Instruction(Dup(ST12))
}

pub fn dup13() -> LabelledInstruction {
    Instruction(Dup(ST13))
}

pub fn dup14() -> LabelledInstruction {
    Instruction(Dup(ST14))
}

pub fn dup15() -> LabelledInstruction {
    Instruction(Dup(ST15))
}

// There is no swap0().

pub fn swap1() -> LabelledInstruction {
    Instruction(Swap(ST1))
}

pub fn swap2() -> LabelledInstruction {
    Instruction(Swap(ST2))
}

pub fn swap3() -> LabelledInstruction {
    Instruction(Swap(ST3))
}

pub fn swap4() -> LabelledInstruction {
    Instruction(Swap(ST4))
}

pub fn swap5() -> LabelledInstruction {
    Instruction(Swap(ST5))
}

pub fn swap6() -> LabelledInstruction {
    Instruction(Swap(ST6))
}

pub fn swap7() -> LabelledInstruction {
    Instruction(Swap(ST7))
}

pub fn swap8() -> LabelledInstruction {
    Instruction(Swap(ST8))
}

pub fn swap9() -> LabelledInstruction {
    Instruction(Swap(ST9))
}

pub fn swap10() -> LabelledInstruction {
    Instruction(Swap(ST10))
}

pub fn swap11() -> LabelledInstruction {
    Instruction(Swap(ST11))
}

pub fn swap12() -> LabelledInstruction {
    Instruction(Swap(ST12))
}

pub fn swap13() -> LabelledInstruction {
    Instruction(Swap(ST13))
}

pub fn swap14() -> LabelledInstruction {
    Instruction(Swap(ST14))
}

pub fn swap15() -> LabelledInstruction {
    Instruction(Swap(ST15))
}

// Control flow

pub fn nop() -> LabelledInstruction {
    Instruction(Nop)
}

pub fn skiz() -> LabelledInstruction {
    Instruction(Skiz)
}

pub fn call(label: String) -> LabelledInstruction {
    Instruction(Call(label))
}

pub fn return_() -> LabelledInstruction {
    Instruction(Return)
}

pub fn recurse() -> LabelledInstruction {
    Instruction(Recurse)
}

pub fn assert_() -> LabelledInstruction {
    Instruction(Assert)
}

pub fn halt() -> LabelledInstruction {
    Instruction(Halt)
}

// Memory access

pub fn read_mem() -> LabelledInstruction {
    Instruction(ReadMem)
}

pub fn write_mem() -> LabelledInstruction {
    Instruction(WriteMem)
}

// Hashing-related

pub fn hash() -> LabelledInstruction {
    Instruction(Hash)
}

pub fn divine_sibling() -> LabelledInstruction {
    Instruction(DivineSibling)
}

pub fn assert_vector() -> LabelledInstruction {
    Instruction(AssertVector)
}

pub fn absorb_init() -> LabelledInstruction {
    Instruction(AbsorbInit)
}

pub fn absorb() -> LabelledInstruction {
    Instruction(Absorb)
}

pub fn squeeze() -> LabelledInstruction {
    Instruction(Squeeze)
}

// Base field arithmetic on stack

pub fn add() -> LabelledInstruction {
    Instruction(Add)
}

pub fn mul() -> LabelledInstruction {
    Instruction(Mul)
}

pub fn invert() -> LabelledInstruction {
    Instruction(Invert)
}

pub fn eq() -> LabelledInstruction {
    Instruction(Eq)
}

// Bitwise arithmetic on stack

pub fn split() -> LabelledInstruction {
    Instruction(Split)
}

pub fn lt() -> LabelledInstruction {
    Instruction(Lt)
}

pub fn and() -> LabelledInstruction {
    Instruction(And)
}

pub fn xor() -> LabelledInstruction {
    Instruction(Xor)
}

pub fn log_2_floor() -> LabelledInstruction {
    Instruction(Log2Floor)
}

pub fn pow() -> LabelledInstruction {
    Instruction(Pow)
}

pub fn div() -> LabelledInstruction {
    Instruction(Div)
}

// Extension field arithmetic on stack

pub fn xxadd() -> LabelledInstruction {
    Instruction(XxAdd)
}

pub fn xxmul() -> LabelledInstruction {
    Instruction(XxMul)
}

pub fn xinvert() -> LabelledInstruction {
    Instruction(XInvert)
}

pub fn xbmul() -> LabelledInstruction {
    Instruction(XbMul)
}

// Read/write

pub fn read_io() -> LabelledInstruction {
    Instruction(ReadIo)
}

pub fn write_io() -> LabelledInstruction {
    Instruction(WriteIo)
}
