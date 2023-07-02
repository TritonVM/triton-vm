use anyhow::anyhow;
use anyhow::Result;
use num_traits::Zero;
use triton_opcodes::ord_n::Ord16;
use triton_opcodes::ord_n::Ord16::*;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::digest::Digest;
use twenty_first::shared_math::tip5::DIGEST_LENGTH;
use twenty_first::shared_math::x_field_element::XFieldElement;

use super::error::InstructionError::*;

#[derive(Debug, Clone)]
pub struct OpStack {
    pub stack: Vec<BFieldElement>,
}

/// The number of op-stack registers, and the internal index at which the op-stack underflow memory
/// has index 0. This offset is used to adjust for the fact that op-stack registers and the
/// op-stack underflow memory are stored in the same vector.
pub const NUM_OP_STACK_REGISTERS: usize = 16;

impl OpStack {
    pub fn new(program_digest: Digest) -> Self {
        let mut stack = vec![BFieldElement::zero(); NUM_OP_STACK_REGISTERS];

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
    pub(crate) fn peek_at(&self, stack_element: Ord16) -> BFieldElement {
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
    pub(crate) fn swap(&mut self, stack_element: Ord16) {
        let stack_element_index = usize::from(stack_element);
        let top_of_stack_index = self.stack.len() - 1;
        self.stack
            .swap(top_of_stack_index, top_of_stack_index - stack_element_index);
    }

    /// `true` if and only if the op-stack contains fewer elements than the number of
    /// op-stack registers, _i.e._, [`NUM_OP_STACK_REGISTERS`].
    pub(crate) fn is_too_shallow(&self) -> bool {
        self.stack.len() < NUM_OP_STACK_REGISTERS
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
        if top_of_stack_index < NUM_OP_STACK_REGISTERS {
            return BFieldElement::zero();
        }
        let op_stack_value_index = top_of_stack_index - NUM_OP_STACK_REGISTERS;
        self.stack[op_stack_value_index]
    }
}

#[cfg(test)]
mod op_stack_test {
    use triton_opcodes::ord_n::Ord16;
    use twenty_first::shared_math::b_field_element::BFieldElement;

    use crate::op_stack::OpStack;

    #[test]
    fn sanity_test() {
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
            op_stack.peek_at(Ord16::ST0),
            op_stack.peek_at(Ord16::ST1),
            op_stack.peek_at(Ord16::ST2),
            op_stack.peek_at(Ord16::ST3),
            op_stack.peek_at(Ord16::ST4),
            op_stack.peek_at(Ord16::ST5),
            op_stack.peek_at(Ord16::ST6),
            op_stack.peek_at(Ord16::ST7),
            op_stack.peek_at(Ord16::ST8),
            op_stack.peek_at(Ord16::ST9),
            op_stack.peek_at(Ord16::ST10),
            op_stack.peek_at(Ord16::ST11),
            op_stack.peek_at(Ord16::ST12),
            op_stack.peek_at(Ord16::ST13),
            op_stack.peek_at(Ord16::ST14),
            op_stack.peek_at(Ord16::ST15),
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
