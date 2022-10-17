use super::error::{vm_fail, InstructionError::*};
use super::ord_n::{Ord16, Ord16::*};
use num_traits::Zero;
use std::error::Error;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;

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

    pub fn pop(&mut self) -> Result<BFieldElement, Box<dyn Error>> {
        self.stack.pop().ok_or_else(|| vm_fail(OpStackTooShallow))
    }

    pub fn pop_x(&mut self) -> Result<XFieldElement, Box<dyn Error>> {
        Ok(XFieldElement::new([self.pop()?, self.pop()?, self.pop()?]))
    }

    pub fn pop_u32(&mut self) -> Result<u32, Box<dyn Error>> {
        let elem = self.pop()?;
        elem.try_into()
            .map_err(|_| vm_fail(FailedU32Conversion(elem)))
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
