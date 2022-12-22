use anyhow::Result;
use num_traits::Zero;
use triton_opcodes::ord_n::Ord16;
use triton_opcodes::ord_n::Ord16::*;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;

use super::error::vm_fail;
use super::error::InstructionError::*;

#[derive(Debug, Clone)]
pub struct OpStack {
    pub stack: Vec<BFieldElement>,
}

/// The number of op-stack registers, and the internal index at which the
/// op-stack memory has index 0. This offset is used to adjust for the fact
/// that op-stack registers are stored in the same way as op-stack memory.
pub const OP_STACK_REG_COUNT: usize = 16;

impl Default for OpStack {
    fn default() -> Self {
        Self {
            stack: vec![BFieldElement::zero(); OP_STACK_REG_COUNT],
        }
    }
}

impl OpStack {
    pub fn push(&mut self, elem: BFieldElement) {
        self.stack.push(elem);
    }

    pub fn push_x(&mut self, elem: XFieldElement) {
        self.push(elem.coefficients[2]);
        self.push(elem.coefficients[1]);
        self.push(elem.coefficients[0]);
    }

    pub fn pop(&mut self) -> Result<BFieldElement> {
        self.stack.pop().ok_or_else(|| vm_fail(OpStackTooShallow))
    }

    pub fn pop_x(&mut self) -> Result<XFieldElement> {
        Ok(XFieldElement::new([self.pop()?, self.pop()?, self.pop()?]))
    }

    pub fn pop_u32(&mut self) -> Result<u32> {
        let elem = self.pop()?;
        elem.try_into()
            .map_err(|_| vm_fail(FailedU32Conversion(elem)))
    }

    pub fn pop_n<const N: usize>(&mut self) -> Result<[BFieldElement; N]> {
        let mut buffer = [BFieldElement::zero(); N];
        for element in buffer.iter_mut() {
            *element = self.pop()?;
        }
        Ok(buffer)
    }

    pub fn safe_peek_x(&mut self) -> XFieldElement {
        XFieldElement::new([
            self.safe_peek(ST0),
            self.safe_peek(ST1),
            self.safe_peek(ST2),
        ])
    }

    pub fn safe_peek(&self, arg: Ord16) -> BFieldElement {
        let n: usize = arg.into();
        let top = self.stack.len() - 1;
        self.stack[top - n]
    }

    pub fn safe_swap(&mut self, arg: Ord16) {
        let n: usize = arg.into();
        let top = self.stack.len() - 1;
        self.stack.swap(top, top - n);
    }

    pub fn peek(&self, n: usize) -> Option<BFieldElement> {
        let top = self.stack.len() - 1;
        self.stack.get(top - n).copied()
    }

    pub fn height(&self) -> usize {
        self.stack.len()
    }

    pub fn is_too_shallow(&self) -> bool {
        self.stack.len() < OP_STACK_REG_COUNT
    }

    /// Get the arg'th op-stack register value
    pub fn st(&self, arg: Ord16) -> BFieldElement {
        let top = self.stack.len() - 1;
        let n: usize = arg.into();
        self.stack[top - n]
    }

    /// Operational stack pointer
    ///
    /// Contains address of next empty op-stack position.
    /// Equivalent to the current length of the op-stack.
    pub fn osp(&self) -> BFieldElement {
        BFieldElement::new(self.stack.len() as u64)
    }

    /// Operational stack value
    ///
    /// Has the value of the top-most op-stack value that does not have an st_ register.
    ///
    /// Assumed to be 0 when op-stack memory is empty.
    pub fn osv(&self) -> BFieldElement {
        if self.stack.len() <= OP_STACK_REG_COUNT {
            BFieldElement::zero()
        } else {
            let top = self.stack.len() - 1;
            let osv_index = top - OP_STACK_REG_COUNT;
            self.stack
                .get(osv_index)
                .copied()
                .expect("Cannot access OSV because stack is too shallow")
        }
    }
}

#[cfg(test)]
mod op_stack_test {
    use twenty_first::shared_math::b_field_element::BFieldElement;

    use crate::op_stack::OpStack;
    use triton_opcodes::ord_n::Ord16;

    #[test]
    fn test_sanity() {
        let mut op_stack = OpStack::default();

        // verify height
        assert_eq!(op_stack.height(), 16);
        assert_eq!(op_stack.osp().value() as usize, op_stack.height());

        // push elements 1 thru 17
        for i in 1..=17 {
            op_stack.push(BFieldElement::new(i as u64));
        }

        // verify height
        assert_eq!(op_stack.height(), 33);
        assert_eq!(op_stack.osp().value() as usize, op_stack.height());

        // verify that all accessible items are different
        let mut container = vec![
            op_stack.st(Ord16::ST0),
            op_stack.st(Ord16::ST1),
            op_stack.st(Ord16::ST2),
            op_stack.st(Ord16::ST3),
            op_stack.st(Ord16::ST4),
            op_stack.st(Ord16::ST5),
            op_stack.st(Ord16::ST6),
            op_stack.st(Ord16::ST7),
            op_stack.st(Ord16::ST8),
            op_stack.st(Ord16::ST9),
            op_stack.st(Ord16::ST10),
            op_stack.st(Ord16::ST11),
            op_stack.st(Ord16::ST12),
            op_stack.st(Ord16::ST13),
            op_stack.st(Ord16::ST14),
            op_stack.st(Ord16::ST15),
            op_stack.osv(),
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
        assert_eq!(op_stack.height(), 22);
        assert_eq!(op_stack.osp().value() as usize, op_stack.height());

        // pop 2 XFieldElements
        op_stack.pop_x().expect("can't pop");
        op_stack.pop_x().expect("can't pop");

        // verify height
        assert_eq!(op_stack.height(), 16);
        assert_eq!(op_stack.osp().value() as usize, op_stack.height());

        // verify underflow
        op_stack.pop().expect("can't pop");
        assert!(op_stack.is_too_shallow());
    }
}
