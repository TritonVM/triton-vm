use std::error::Error;
use std::fmt::Display;
use std::fmt::Formatter;

use twenty_first::shared_math::b_field_element::BFieldElement;

use InstructionError::*;

#[derive(Debug, Clone)]
pub enum InstructionError {
    InstructionPointerOverflow(usize),
    OpStackTooShallow,
    JumpStackIsEmpty,
    AssertionFailed(usize, u32, BFieldElement),
    InverseOfZero,
    DivisionByZero,
    LogarithmOfZero,
    FailedU32Conversion(BFieldElement),
}

impl Display for InstructionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            InstructionPointerOverflow(ip) => {
                write!(f, "Instruction pointer {ip} points outside of program")
            }

            OpStackTooShallow => {
                write!(f, "Operational stack is too shallow")
            }

            JumpStackIsEmpty => {
                write!(f, "Jump stack is empty.")
            }

            AssertionFailed(ip, clk, st0) => {
                write!(
                    f,
                    "Assertion failed: st0 must be 1. ip: {ip}, clk: {clk}, st0: {st0}"
                )
            }

            InverseOfZero => {
                write!(f, "0 does not have a multiplicative inverse")
            }

            DivisionByZero => {
                write!(f, "Division by 0 is impossible")
            }

            LogarithmOfZero => {
                write!(f, "The logarithm of 0 does not exist")
            }

            FailedU32Conversion(word) => {
                write!(
                    f,
                    "Failed to convert BFieldElement {} into u32",
                    word.value()
                )
            }
        }
    }
}

impl Error for InstructionError {}

#[cfg(test)]
mod tests {
    use triton_opcodes::program::Program;

    use crate::vm::run;

    #[test]
    #[should_panic(expected = "Instruction pointer 1 points outside of program")]
    fn test_vm_err() {
        let program = Program::from_code("nop").unwrap();
        run(&program, vec![], vec![]).unwrap();
    }

    #[test]
    #[should_panic(expected = "Operational stack is too shallow")]
    fn shrink_op_stack_too_much_test() {
        let program = Program::from_code("pop halt").unwrap();
        run(&program, vec![], vec![]).unwrap();
    }

    #[test]
    #[should_panic(expected = "Jump stack is empty.")]
    fn return_without_call_test() {
        let program = Program::from_code("return halt").unwrap();
        run(&program, vec![], vec![]).unwrap();
    }

    #[test]
    #[should_panic(expected = "Jump stack is empty.")]
    fn recurse_without_call_test() {
        let program = Program::from_code("recurse halt").unwrap();
        run(&program, vec![], vec![]).unwrap();
    }

    #[test]
    #[should_panic(expected = "Assertion failed: st0 must be 1. ip: 2, clk: 1, st0: 0")]
    fn assert_false_test() {
        let program = Program::from_code("push 0 assert halt").unwrap();
        run(&program, vec![], vec![]).unwrap();
    }

    #[test]
    #[should_panic(expected = "0 does not have a multiplicative inverse")]
    fn inverse_of_zero_test() {
        let program = Program::from_code("push 0 invert halt").unwrap();
        run(&program, vec![], vec![]).unwrap();
    }

    #[test]
    #[should_panic(expected = "Division by 0 is impossible")]
    fn division_by_zero_test() {
        let program = Program::from_code("push 0 push 5 div halt").unwrap();
        run(&program, vec![], vec![]).unwrap();
    }

    #[test]
    #[should_panic(expected = "The logarithm of 0 does not exist")]
    fn log_of_zero_test() {
        let program = Program::from_code("push 0 log_2_floor halt").unwrap();
        run(&program, vec![], vec![]).unwrap();
    }

    #[test]
    #[should_panic(expected = "Failed to convert BFieldElement 4294967297 into u32")]
    fn failed_u32_conversion_test() {
        let program = Program::from_code("push 4294967297 push 1 and halt").unwrap();
        run(&program, vec![], vec![]).unwrap();
    }
}
