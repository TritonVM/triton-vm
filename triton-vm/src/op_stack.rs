use std::fmt::Display;
use std::result;

use anyhow::anyhow;
use anyhow::Result;
use get_size::GetSize;
use num_traits::Zero;
use serde_derive::Deserialize;
use serde_derive::Serialize;
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::digest::Digest;
use twenty_first::shared_math::tip5::DIGEST_LENGTH;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::op_stack::OpStackElement::*;

use super::error::InstructionError::*;

/// The number of registers dedicated to the top of the operational stack.
pub const NUM_OP_STACK_REGISTERS: usize = OpStackElement::COUNT;

/// The operational stack of Triton VM.
/// It always contains at least [`OpStackElement::COUNT`] elements. Initially, the bottom-most
/// [`DIGEST_LENGTH`] elements equal the digest of the program being executed.
/// The remaining elements are initially 0.
///
/// The OpStack is represented as one contiguous piece of memory, and Triton VM uses it as such.
/// For reasons of arithmetization, however, there is a distinction between the op-stack registers
/// and the op-stack underflow memory. The op-stack registers are the first
/// [`OpStackElement::COUNT`] elements of the op-stack, and the op-stack underflow memory is the
/// remaining elements.
#[derive(Debug, Clone)]
pub struct OpStack {
    pub stack: Vec<BFieldElement>,
}

impl OpStack {
    pub fn new(program_digest: Digest) -> Self {
        let mut stack = vec![BFieldElement::zero(); OpStackElement::COUNT];

        let reverse_digest = program_digest.reversed().values();
        stack[..DIGEST_LENGTH].copy_from_slice(&reverse_digest);

        Self { stack }
    }

    /// Push an element onto the op-stack.
    pub(crate) fn push(&mut self, element: BFieldElement) {
        self.stack.push(element);
    }

    /// Push an extension field element onto the op-stack.
    pub(crate) fn push_extension_field_element(&mut self, element: XFieldElement) {
        for coefficient in element.coefficients.into_iter().rev() {
            self.push(coefficient);
        }
    }

    /// Pop an element from the op-stack.
    pub(crate) fn pop(&mut self) -> Result<BFieldElement> {
        self.stack.pop().ok_or_else(|| anyhow!(OpStackTooShallow))
    }

    /// Pop an extension field element from the op-stack.
    pub(crate) fn pop_extension_field_element(&mut self) -> Result<XFieldElement> {
        let coefficients = self.pop_multiple()?;
        let element = XFieldElement::new(coefficients);
        Ok(element)
    }

    /// Pop a u32 from the op-stack.
    pub(crate) fn pop_u32(&mut self) -> Result<u32> {
        let element = self.pop()?;
        element
            .try_into()
            .map_err(|_| anyhow!(FailedU32Conversion(element)))
    }

    /// Pop multiple elements from the op-stack.
    pub(crate) fn pop_multiple<const N: usize>(&mut self) -> Result<[BFieldElement; N]> {
        let mut popped_elements = [BFieldElement::zero(); N];
        for element in popped_elements.iter_mut() {
            *element = self.pop()?;
        }
        Ok(popped_elements)
    }

    /// Fetches the indicated stack element without modifying the stack.
    pub(crate) fn peek_at(&self, stack_element: OpStackElement) -> BFieldElement {
        let stack_element_index = usize::from(stack_element);
        let top_of_stack_index = self.stack.len() - 1;
        self.stack[top_of_stack_index - stack_element_index]
    }

    /// Fetches the top-most extension field element without modifying the stack.
    pub(crate) fn peek_at_top_extension_field_element(&self) -> XFieldElement {
        let coefficients = [self.peek_at(ST0), self.peek_at(ST1), self.peek_at(ST2)];
        XFieldElement::new(coefficients)
    }

    /// Swaps the top of the stack with the indicated stack element.
    pub(crate) fn swap_top_with(&mut self, stack_element: OpStackElement) {
        let stack_element_index = usize::from(stack_element);
        let top_of_stack_index = self.stack.len() - 1;
        self.stack
            .swap(top_of_stack_index, top_of_stack_index - stack_element_index);
    }

    /// `true` if and only if the op-stack contains fewer elements than the number of
    /// op-stack registers, _i.e._, [`OpStackElement::COUNT`].
    pub(crate) fn is_too_shallow(&self) -> bool {
        self.stack.len() < OpStackElement::COUNT
    }

    /// The address of the next free address of the op-stack.
    /// Equivalent to the current length of the op-stack.
    pub(crate) fn op_stack_pointer(&self) -> BFieldElement {
        (self.stack.len() as u64).into()
    }

    /// The first element of the op-stack underflow memory, or 0 if the op-stack underflow memory
    /// is empty.
    pub(crate) fn op_stack_value(&self) -> BFieldElement {
        let top_of_stack_index = self.stack.len() - 1;
        if top_of_stack_index < OpStackElement::COUNT {
            return BFieldElement::zero();
        }
        let op_stack_value_index = top_of_stack_index - OpStackElement::COUNT;
        self.stack[op_stack_value_index]
    }
}

/// Represents the [`OpStack`] registers directly accessible by Triton VM.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, GetSize, Serialize, Deserialize, EnumCount,
)]
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

    fn try_from(stack_index: u32) -> result::Result<Self, Self::Error> {
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

    fn try_from(stack_index: u64) -> result::Result<Self, Self::Error> {
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

    fn try_from(stack_index: usize) -> result::Result<Self, Self::Error> {
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

#[cfg(test)]
mod tests {
    use twenty_first::shared_math::b_field_element::BFieldElement;

    use crate::op_stack::OpStack;
    use crate::op_stack::OpStackElement;

    #[test]
    fn sanity() {
        let digest = Default::default();
        let mut op_stack = OpStack::new(digest);

        // verify height
        assert_eq!(op_stack.stack.len(), 16);
        assert_eq!(
            op_stack.op_stack_pointer().value() as usize,
            op_stack.stack.len()
        );

        // push elements 1 thru 17
        for i in 1..=17 {
            op_stack.push(BFieldElement::new(i as u64));
        }

        // verify height
        assert_eq!(op_stack.stack.len(), 33);
        assert_eq!(
            op_stack.op_stack_pointer().value() as usize,
            op_stack.stack.len()
        );

        // verify that all accessible items are different
        let mut container = vec![
            op_stack.peek_at(OpStackElement::ST0),
            op_stack.peek_at(OpStackElement::ST1),
            op_stack.peek_at(OpStackElement::ST2),
            op_stack.peek_at(OpStackElement::ST3),
            op_stack.peek_at(OpStackElement::ST4),
            op_stack.peek_at(OpStackElement::ST5),
            op_stack.peek_at(OpStackElement::ST6),
            op_stack.peek_at(OpStackElement::ST7),
            op_stack.peek_at(OpStackElement::ST8),
            op_stack.peek_at(OpStackElement::ST9),
            op_stack.peek_at(OpStackElement::ST10),
            op_stack.peek_at(OpStackElement::ST11),
            op_stack.peek_at(OpStackElement::ST12),
            op_stack.peek_at(OpStackElement::ST13),
            op_stack.peek_at(OpStackElement::ST14),
            op_stack.peek_at(OpStackElement::ST15),
            op_stack.op_stack_value(),
        ];
        let len_before = container.len();
        container.sort_by_key(|a| a.value());
        container.dedup();
        let len_after = container.len();
        assert_eq!(len_before, len_after);

        // pop 11 elements
        for _ in 0..11 {
            op_stack.pop().expect("can't pop");
        }

        // verify height
        assert_eq!(op_stack.stack.len(), 22);
        assert_eq!(
            op_stack.op_stack_pointer().value() as usize,
            op_stack.stack.len()
        );

        // pop 2 XFieldElements
        op_stack.pop_extension_field_element().expect("can't pop");
        op_stack.pop_extension_field_element().expect("can't pop");

        // verify height
        assert_eq!(op_stack.stack.len(), 16);
        assert_eq!(
            op_stack.op_stack_pointer().value() as usize,
            op_stack.stack.len()
        );

        // verify underflow
        op_stack.pop().expect("can't pop");
        assert!(op_stack.is_too_shallow());
    }
}
