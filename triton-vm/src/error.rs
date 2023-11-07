use std::error::Error;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;

use twenty_first::shared_math::digest::DIGEST_LENGTH;

use InstructionError::*;

use crate::op_stack::OpStackElement;
use crate::BFieldElement;

#[derive(Debug, Clone)]
pub enum InstructionError {
    InstructionPointerOverflow(usize),
    OpStackTooShallow,
    JumpStackIsEmpty,
    AssertionFailed(usize, u32, BFieldElement),
    VectorAssertionFailed(usize, u32, OpStackElement, BFieldElement, BFieldElement),
    SwapST0,
    InverseOfZero,
    DivisionByZero,
    LogarithmOfZero,
    FailedU32Conversion(BFieldElement),
}

impl Display for InstructionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            InstructionPointerOverflow(ip) => {
                write!(f, "Instruction pointer {ip} points outside of program")
            }
            OpStackTooShallow => write!(f, "Operational stack is too shallow"),
            JumpStackIsEmpty => write!(f, "Jump stack is empty."),
            AssertionFailed(ip, clk, st0) => {
                write!(f, "Assertion failed: st0 must be 1. ")?;
                write!(f, "ip: {ip}, clk: {clk}, st0: {st0}")
            }
            VectorAssertionFailed(ip, clk, failing_position_lhs, lhs, rhs) => {
                let failing_index_lhs: usize = failing_position_lhs.into();
                let failing_index_rhs = failing_index_lhs + DIGEST_LENGTH;
                write!(f, "Vector assertion failed: ")?;
                write!(f, "op_stack[{failing_index_lhs}] == {lhs} != ")?;
                write!(f, "{rhs} == op_stack[{failing_index_rhs}]. ")?;
                write!(f, "ip: {ip}, clk: {clk}")
            }
            SwapST0 => write!(f, "Cannot swap stack element 0 with itself"),
            InverseOfZero => write!(f, "0 does not have a multiplicative inverse"),
            DivisionByZero => write!(f, "Division by 0 is impossible"),
            LogarithmOfZero => write!(f, "The logarithm of 0 does not exist"),
            FailedU32Conversion(bfe) => {
                let value = bfe.value();
                write!(f, "Failed to convert BFieldElement {value} into u32")
            }
        }
    }
}

impl Error for InstructionError {}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use crate::instruction::AnInstruction::*;
    use crate::instruction::LabelledInstruction;
    use crate::op_stack::OpStackElement::ST0;
    use crate::triton_program;
    use crate::Program;

    use super::*;

    #[test]
    #[should_panic(expected = "Instruction pointer 1 points outside of program")]
    fn vm_err() {
        let program = triton_program!(nop);
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    #[should_panic(expected = "Operational stack is too shallow")]
    fn shrink_op_stack_too_much() {
        let program = triton_program!(pop 3 halt);
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    #[should_panic(expected = "Jump stack is empty.")]
    fn return_without_call() {
        let program = triton_program!(return halt);
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    #[should_panic(expected = "Jump stack is empty.")]
    fn recurse_without_call() {
        let program = triton_program!(recurse halt);
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    #[should_panic(expected = "Assertion failed: st0 must be 1. ip: 2, clk: 1, st0: 0")]
    fn assert_false() {
        let program = triton_program!(push 0 assert halt);
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    #[should_panic(expected = "op_stack[1] == 10 != 1 == op_stack[6]")]
    fn print_unequal_vec_assert_error() {
        let program = triton_program! {
            push 4 push 3 push 2 push  1 push 0
            push 4 push 3 push 2 push 10 push 0
            assert_vector halt
        };
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    #[should_panic(expected = "swap stack element 0")]
    fn swap_st0() {
        // The parser rejects this program. Therefore, construct it manually.
        let swap_0 = LabelledInstruction::Instruction(Swap(ST0));
        let halt = LabelledInstruction::Instruction(Halt);
        let program = Program::new(&[swap_0, halt]);
        program.run([].into(), [].into()).unwrap();
    }

    #[proptest]
    fn assert_unequal_vec(
        #[strategy(arb())] test_vector: [BFieldElement; DIGEST_LENGTH],
        #[strategy(0..DIGEST_LENGTH)] disturbance_index: usize,
        #[strategy(arb())]
        #[filter(#test_vector[#disturbance_index] != #random_element)]
        random_element: BFieldElement,
    ) {
        let mut disturbed_vector = test_vector;
        disturbed_vector[disturbance_index] = random_element;

        let program = triton_program! {
            push {test_vector[4]}
            push {test_vector[3]}
            push {test_vector[2]}
            push {test_vector[1]}
            push {test_vector[0]}

            push {disturbed_vector[4]}
            push {disturbed_vector[3]}
            push {disturbed_vector[2]}
            push {disturbed_vector[1]}
            push {disturbed_vector[0]}

            assert_vector
            halt
        };

        let err = program.run([].into(), [].into()).unwrap_err();

        let err = err.downcast::<InstructionError>().unwrap();
        let VectorAssertionFailed(_, _, index, _, _) = err else {
            panic!("VM panicked with unexpected error {err}.")
        };
        let index: usize = index.into();
        prop_assert_eq!(disturbance_index, index);
    }

    #[test]
    #[should_panic(expected = "0 does not have a multiplicative inverse")]
    fn inverse_of_zero() {
        let program = triton_program!(push 0 invert halt);
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    #[should_panic(expected = "0 does not have a multiplicative inverse")]
    fn xfe_inverse_of_zero() {
        let program = triton_program!(push 0 push 0 push 0 xinvert halt);
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    #[should_panic(expected = "Division by 0 is impossible")]
    fn division_by_zero() {
        let program = triton_program!(push 0 push 5 div_mod halt);
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    #[should_panic(expected = "The logarithm of 0 does not exist")]
    fn log_of_zero() {
        let program = triton_program!(push 0 log_2_floor halt);
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    #[should_panic(expected = "Failed to convert BFieldElement 4294967297 into u32")]
    fn failed_u32_conversion() {
        let program = triton_program!(push 4294967297 push 1 and halt);
        program.run([].into(), [].into()).unwrap();
    }
}
