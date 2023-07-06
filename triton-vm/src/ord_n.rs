use std::fmt::Display;

use get_size::GetSize;
use serde_derive::Deserialize;
use serde_derive::Serialize;
use strum_macros::EnumCount as EnumCountMacro;
use twenty_first::shared_math::b_field_element::BFieldElement;

use InstructionBit::*;
use OpStackElement::*;

/// Indicators for all the possible bits in an [`Instruction`](crate::instruction::Instruction).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, EnumCountMacro)]
pub enum InstructionBit {
    #[default]
    IB0,
    IB1,
    IB2,
    IB3,
    IB4,
    IB5,
    IB6,
    IB7,
}

impl Display for InstructionBit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bit_index = usize::from(*self);
        write!(f, "{bit_index}")
    }
}

impl From<InstructionBit> for usize {
    fn from(instruction_bit: InstructionBit) -> Self {
        match instruction_bit {
            IB0 => 0,
            IB1 => 1,
            IB2 => 2,
            IB3 => 3,
            IB4 => 4,
            IB5 => 5,
            IB6 => 6,
            IB7 => 7,
        }
    }
}

impl TryFrom<usize> for InstructionBit {
    type Error = String;

    fn try_from(bit_index: usize) -> Result<Self, Self::Error> {
        match bit_index {
            0 => Ok(IB0),
            1 => Ok(IB1),
            2 => Ok(IB2),
            3 => Ok(IB3),
            4 => Ok(IB4),
            5 => Ok(IB5),
            6 => Ok(IB6),
            7 => Ok(IB7),
            _ => Err(format!(
                "Index {bit_index} is out of range for `InstructionBit`."
            )),
        }
    }
}

impl From<InstructionBit> for BFieldElement {
    fn from(instruction_bit: InstructionBit) -> Self {
        let instruction_bit = usize::from(instruction_bit) as u64;
        instruction_bit.into()
    }
}

/// Represents numbers that are exactly 0 through 15, corresponding to those
/// [`OpStack`](crate::op_stack::OpStack) registers directly accessible by Triton VM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, GetSize, Serialize, Deserialize)]
pub enum OpStackElement {
    #[default]
    ST0,
    ST1,
    ST2,
    ST3,
    ST4,
    ST5,
    ST6,
    ST7,
    ST8,
    ST9,
    ST10,
    ST11,
    ST12,
    ST13,
    ST14,
    ST15,
}

impl Display for OpStackElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stack_index = u32::from(self);
        write!(f, "{stack_index}")
    }
}

impl From<OpStackElement> for u32 {
    fn from(stack_element: OpStackElement) -> Self {
        match stack_element {
            ST0 => 0,
            ST1 => 1,
            ST2 => 2,
            ST3 => 3,
            ST4 => 4,
            ST5 => 5,
            ST6 => 6,
            ST7 => 7,
            ST8 => 8,
            ST9 => 9,
            ST10 => 10,
            ST11 => 11,
            ST12 => 12,
            ST13 => 13,
            ST14 => 14,
            ST15 => 15,
        }
    }
}

impl From<&OpStackElement> for u32 {
    fn from(stack_element: &OpStackElement) -> Self {
        (*stack_element).into()
    }
}

impl TryFrom<u32> for OpStackElement {
    type Error = String;

    fn try_from(stack_index: u32) -> Result<Self, Self::Error> {
        match stack_index {
            0 => Ok(ST0),
            1 => Ok(ST1),
            2 => Ok(ST2),
            3 => Ok(ST3),
            4 => Ok(ST4),
            5 => Ok(ST5),
            6 => Ok(ST6),
            7 => Ok(ST7),
            8 => Ok(ST8),
            9 => Ok(ST9),
            10 => Ok(ST10),
            11 => Ok(ST11),
            12 => Ok(ST12),
            13 => Ok(ST13),
            14 => Ok(ST14),
            15 => Ok(ST15),
            _ => Err(format!(
                "Index {stack_index} is out of range for `OpStackElement`."
            )),
        }
    }
}

impl From<OpStackElement> for u64 {
    fn from(stack_element: OpStackElement) -> Self {
        u32::from(stack_element).into()
    }
}

impl TryFrom<u64> for OpStackElement {
    type Error = String;

    fn try_from(stack_index: u64) -> Result<Self, Self::Error> {
        let stack_index = u32::try_from(stack_index)
            .map_err(|_| format!("Index {stack_index} is out of range for `OpStackElement`."))?;
        stack_index.try_into()
    }
}

impl From<OpStackElement> for usize {
    fn from(stack_element: OpStackElement) -> Self {
        u32::from(stack_element) as usize
    }
}

impl From<&OpStackElement> for usize {
    fn from(stack_element: &OpStackElement) -> Self {
        (*stack_element).into()
    }
}

impl TryFrom<usize> for OpStackElement {
    type Error = String;

    fn try_from(stack_index: usize) -> Result<Self, Self::Error> {
        let stack_index =
            u32::try_from(stack_index).map_err(|_| "Cannot convert usize to u32.".to_string())?;
        stack_index.try_into()
    }
}

impl From<OpStackElement> for BFieldElement {
    fn from(stack_element: OpStackElement) -> Self {
        u32::from(stack_element).into()
    }
}

impl From<&OpStackElement> for BFieldElement {
    fn from(stack_element: &OpStackElement) -> Self {
        (*stack_element).into()
    }
}
