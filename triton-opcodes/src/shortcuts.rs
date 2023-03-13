use twenty_first::shared_math::b_field_element::BFieldElement;

use super::instruction::AnInstruction::*;
use super::instruction::LabelledInstruction;
use super::instruction::LabelledInstruction::*;

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

pub fn dup(st: u64) -> LabelledInstruction {
    Instruction(Dup(st.try_into().unwrap()))
}

pub fn swap(st: u64) -> LabelledInstruction {
    assert_ne!(
        0, st,
        "Instruction `swap` cannot be used on stack register 0."
    );
    Instruction(Swap(st.try_into().unwrap()))
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
