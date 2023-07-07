use super::instruction::AnInstruction::*;
use super::instruction::LabelledInstruction;
use super::instruction::LabelledInstruction::*;

// OpStack manipulation
pub fn pop() -> LabelledInstruction {
    Instruction(Pop)
}

pub fn push(field_element: u64) -> LabelledInstruction {
    Instruction(Push(field_element.into()))
}

pub fn divine() -> LabelledInstruction {
    Instruction(Divine)
}

pub fn dup(stack_index: u64) -> LabelledInstruction {
    Instruction(Dup(stack_index.try_into().unwrap()))
}

pub fn swap(stack_index: u64) -> LabelledInstruction {
    assert_ne!(
        0, stack_index,
        "Instruction `swap` cannot be used on stack element 0."
    );
    Instruction(Swap(stack_index.try_into().unwrap()))
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

#[cfg(test)]
mod test {
    use crate::op_stack::OpStackElement::ST0;
    use crate::op_stack::OpStackElement::ST1;

    use super::*;

    #[test]
    fn shortcuts_correspond_to_expected_instructions() {
        let label = "label".to_string();
        assert_eq!(pop(), Instruction(Pop));
        assert_eq!(push(7), Instruction(Push(7_u64.into())));
        assert_eq!(divine(), Instruction(Divine));
        assert_eq!(dup(0), Instruction(Dup(ST0)));
        assert_eq!(swap(1), Instruction(Swap(ST1)));
        assert_eq!(nop(), Instruction(Nop));
        assert_eq!(skiz(), Instruction(Skiz));
        assert_eq!(call(label.clone()), Instruction(Call(label)));
        assert_eq!(return_(), Instruction(Return));
        assert_eq!(recurse(), Instruction(Recurse));
        assert_eq!(assert_(), Instruction(Assert));
        assert_eq!(halt(), Instruction(Halt));
        assert_eq!(read_mem(), Instruction(ReadMem));
        assert_eq!(write_mem(), Instruction(WriteMem));
        assert_eq!(hash(), Instruction(Hash));
        assert_eq!(divine_sibling(), Instruction(DivineSibling));
        assert_eq!(assert_vector(), Instruction(AssertVector));
        assert_eq!(absorb_init(), Instruction(AbsorbInit));
        assert_eq!(absorb(), Instruction(Absorb));
        assert_eq!(squeeze(), Instruction(Squeeze));
        assert_eq!(add(), Instruction(Add));
        assert_eq!(mul(), Instruction(Mul));
        assert_eq!(invert(), Instruction(Invert));
        assert_eq!(eq(), Instruction(Eq));
        assert_eq!(split(), Instruction(Split));
        assert_eq!(lt(), Instruction(Lt));
        assert_eq!(and(), Instruction(And));
        assert_eq!(xor(), Instruction(Xor));
        assert_eq!(log_2_floor(), Instruction(Log2Floor));
        assert_eq!(pow(), Instruction(Pow));
        assert_eq!(div(), Instruction(Div));
        assert_eq!(xxadd(), Instruction(XxAdd));
        assert_eq!(xxmul(), Instruction(XxMul));
        assert_eq!(xinvert(), Instruction(XInvert));
        assert_eq!(xbmul(), Instruction(XbMul));
        assert_eq!(read_io(), Instruction(ReadIo));
        assert_eq!(write_io(), Instruction(WriteIo));
    }

    #[test]
    #[should_panic(expected = "cannot be used")]
    fn swap_panics_on_zero() {
        swap(0);
    }
}
