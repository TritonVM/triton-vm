use thiserror::Error;
use twenty_first::shared_math::digest::DIGEST_LENGTH;

use crate::instruction::Instruction;
use crate::op_stack::NumberOfWords;
use crate::op_stack::OpStackElement;
use crate::vm::VMState;
use crate::BFieldElement;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub struct VMError<'pgm> {
    source: InstructionError,
    vm_state: VMState<'pgm>,
}

impl<'pgm> VMError<'pgm> {
    pub fn new(source: InstructionError, vm_state: VMState<'pgm>) -> Self {
        Self { source, vm_state }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum InstructionError {
    #[error("opcode {0} is invalid")]
    InvalidOpcode(u32),

    #[error("invalid argument {1} for instruction `{0}`")]
    IllegalArgument(Instruction, BFieldElement),

    #[error("instruction pointer points outside of program")]
    InstructionPointerOverflow,

    #[error("operational stack is too shallow")]
    OpStackTooShallow,

    #[error("jump stack is empty")]
    JumpStackIsEmpty,

    #[error("assertion failed: st0 must be 1")]
    AssertionFailed,

    #[error(
        "vector assertion failed: op_stack[{0}] != op_stack[{}]", usize::from(.0) + DIGEST_LENGTH
    )]
    VectorAssertionFailed(OpStackElement),

    #[error("cannot swap stack element 0 with itself")]
    SwapST0,

    #[error("0 does not have a multiplicative inverse")]
    InverseOfZero,

    #[error("division by 0 is impossible")]
    DivisionByZero,

    #[error("the logarithm of 0 does not exist")]
    LogarithmOfZero,

    #[error("failed to convert BFieldElement {0} into u32")]
    FailedU32Conversion(BFieldElement),

    #[error("instruction `read_io {0}`: public input buffer is empty after {1}")]
    EmptyPublicInput(NumberOfWords, usize),

    #[error("instruction `divine {0}`: secret input buffer is empty after {1}")]
    EmptySecretInput(NumberOfWords, usize),

    #[error("no more secret digests available")]
    EmptySecretDigestInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum ProofStreamError {
    #[error("queue must be non-empty in order to dequeue an item")]
    EmptyQueue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum FriValidationError {
    IncorrectNumberOfRevealedLeaves,
    BadMerkleAuthenticationPath,
    MismatchingLastCodeword,
    LastRoundPolynomialHasTooHighDegree,
    BadMerkleRootForLastCodeword,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ProgramDecodingError {
    EmptySequence,
    SequenceTooShort,
    SequenceTooLong,
    LengthMismatch,
    InvalidInstruction(usize, InstructionError),
    MissingArgument(usize, Instruction),
}

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
    #[should_panic(expected = "instruction pointer 1 points outside of program")]
    fn vm_err() {
        let program = triton_program!(nop);
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    #[should_panic(expected = "operational stack is too shallow")]
    fn shrink_op_stack_too_much() {
        let program = triton_program!(pop 3 halt);
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    #[should_panic(expected = "jump stack is empty")]
    fn return_without_call() {
        let program = triton_program!(return halt);
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    #[should_panic(expected = "jump stack is empty")]
    fn recurse_without_call() {
        let program = triton_program!(recurse halt);
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    #[should_panic(expected = "assertion failed: st0 must be 1")]
    fn assert_false() {
        let program = triton_program!(push 0 assert halt);
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    #[should_panic(expected = "op_stack[1] != op_stack[6]")]
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
        let InstructionError::VectorAssertionFailed(_, _, index, _, _) = err else {
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
    #[should_panic(expected = "division by 0 is impossible")]
    fn division_by_zero() {
        let program = triton_program!(push 0 push 5 div_mod halt);
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    #[should_panic(expected = "the logarithm of 0 does not exist")]
    fn log_of_zero() {
        let program = triton_program!(push 0 log_2_floor halt);
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    #[should_panic(expected = "failed to convert BFieldElement 4294967297 into u32")]
    fn failed_u32_conversion() {
        let program = triton_program!(push 4294967297 push 1 and halt);
        program.run([].into(), [].into()).unwrap();
    }
}
