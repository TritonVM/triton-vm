use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::num::TryFromIntError;

use thiserror::Error;
use twenty_first::shared_math::bfield_codec::BFieldCodec;
use twenty_first::shared_math::digest::DIGEST_LENGTH;

use crate::instruction::Instruction;
use crate::op_stack::OpStackElement;
use crate::proof_item::ProofItem;
use crate::proof_stream::ProofStream;
use crate::stark::StarkHasher;
use crate::vm::VMState;
use crate::BFieldElement;

/// Indicates a runtime error that resulted in a crash of Triton VM.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub struct VMError {
    /// The reason Triton VM crashed.
    pub source: InstructionError,

    /// The state of Triton VM at the time of the crash.
    pub vm_state: Box<VMState>,
}

impl VMError {
    pub fn new(source: InstructionError, vm_state: VMState) -> Self {
        let vm_state = Box::new(vm_state);
        Self { source, vm_state }
    }
}

impl Display for VMError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "VM error: {}", self.source)?;
        writeln!(f, "VM state:")?;
        writeln!(f, "{}", self.vm_state)
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum InstructionError {
    #[error("opcode {0} is invalid")]
    InvalidOpcode(u32),

    #[error("opcode is out of range: {0}")]
    OutOfRangeOpcode(#[from] TryFromIntError),

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

    #[error("vector assertion failed: stack[{0}] != stack[{}]", usize::from(.0) + DIGEST_LENGTH)]
    VectorAssertionFailed(OpStackElement),

    #[error("cannot swap stack element 0 with itself")]
    SwapST0,

    #[error("0 does not have a multiplicative inverse")]
    InverseOfZero,

    #[error("division by 0 is impossible")]
    DivisionByZero,

    #[error("the Sponge state must be initialized before it can be used")]
    SpongeNotInitialized,

    #[error("the logarithm of 0 does not exist")]
    LogarithmOfZero,

    #[error("failed to convert BFieldElement {0} into u32")]
    FailedU32Conversion(BFieldElement),

    #[error("public input buffer is empty after {0} reads")]
    EmptyPublicInput(usize),

    #[error("secret input buffer is empty after {0} reads")]
    EmptySecretInput(usize),

    #[error("no more secret digests available")]
    EmptySecretDigestInput,

    #[error("Triton VM has halted an cannot execute any further instructions")]
    MachineHalted,
}

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ProofStreamError {
    #[error("queue must be non-empty in order to dequeue an item")]
    EmptyQueue,

    #[error("expected {0}, but got {1}")]
    UnexpectedItem(&'static str, ProofItem),

    #[error("the proof stream must contain a log2_padded_height item")]
    NoLog2PaddedHeight,

    #[error("the proof stream must contain exactly one log2_padded_height item")]
    TooManyLog2PaddedHeights,

    #[error("decoding error: {0}")]
    DecodingError(#[from] <ProofStream<StarkHasher> as BFieldCodec>::Error),
}

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum FriValidationError {
    #[error("the number of revealed leaves does not match the number of collinearity checks")]
    IncorrectNumberOfRevealedLeaves,

    #[error("Merkle tree authentication failed")]
    BadMerkleAuthenticationPath,

    #[error("computed and received codeword of last round do not match")]
    LastCodewordMismatch,

    #[error("last round's polynomial has too high degree")]
    LastRoundPolynomialHasTooHighDegree,

    #[error("received codeword of last round does not correspond to its commitment")]
    BadMerkleRootForLastCodeword,

    #[error("proof stream error: {0}")]
    ProofStreamError(#[from] ProofStreamError),
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ProgramDecodingError {
    #[error("sequence to decode is empty")]
    EmptySequence,

    #[error("sequence to decode is too short")]
    SequenceTooShort,

    #[error("sequence to decode is too long")]
    SequenceTooLong,

    #[error("length of decoded program is unexpected")]
    LengthMismatch,

    #[error("sequence to decode contains invalid instruction at index {0}: {1}")]
    InvalidInstruction(usize, InstructionError),

    #[error("missing argument for instruction {1} at index {0}")]
    MissingArgument(usize, Instruction),
}

const CANONICAL_REPRESENTATION_ERROR_MESSAGE: &str =
    "must contain only elements in canonical representation, i.e., \
    elements smaller than the prime field's modulus 2^64 - 2^32 + 1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CanonicalRepresentationError {
    #[error("public input {CANONICAL_REPRESENTATION_ERROR_MESSAGE}")]
    PublicInput,

    #[error("secret input {CANONICAL_REPRESENTATION_ERROR_MESSAGE}")]
    NonDeterminismIndividualTokens,

    #[error("RAM addresses {CANONICAL_REPRESENTATION_ERROR_MESSAGE}")]
    NonDeterminismRamKeys,

    #[error("RAM values {CANONICAL_REPRESENTATION_ERROR_MESSAGE}")]
    NonDeterminismRamValues,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProvingError {
    #[error("claimed program digest does not match actual program digest")]
    ProgramDigestMismatch,

    #[error("claimed public output does not match actual public output")]
    PublicOutputMismatch,

    #[error("error while running Triton VM: {0}")]
    VMError(#[from] VMError),
}

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum VerificationError {
    #[error("received and computed out-of-domain quotient values don't match")]
    OutOfDomainQuotientValueMismatch,

    #[error("failed to verify authentication path for base codeword")]
    BaseCodewordAuthenticationFailure,

    #[error("failed to verify authentication path for extension codeword")]
    ExtensionCodewordAuthenticationFailure,

    #[error("failed to verify authentication path for combined quotient codeword")]
    QuotientCodewordAuthenticationFailure,

    #[error("received and computed combination codewords don't match")]
    CombinationCodewordMismatch,

    #[error("the number of received combination codeword indices does not match the parameters")]
    IncorrectNumberOfRowIndices,

    #[error("the number of received FRI codeword values does not match the parameters")]
    IncorrectNumberOfFRIValues,

    #[error("the number of received quotient segment elements does not match the parameters")]
    IncorrectNumberOfQuotientSegmentElements,

    #[error("the number of received base table rows does not match the parameters")]
    IncorrectNumberOfBaseTableRows,

    #[error("the number of received extension table rows does not match the parameters")]
    IncorrectNumberOfExtTableRows,

    #[error("proof stream error: {0}")]
    ProofStreamError(#[from] ProofStreamError),

    #[error("FRI validation error: {0}")]
    FriValidationError(#[from] FriValidationError),
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum OpStackElementError {
    #[error("index {0} is out of range for `OpStackElement`")]
    IndexOutOfBounds(u32),

    #[error("index is out of legal range: {0}")]
    FailedIntegerConversion(#[from] TryFromIntError),
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum NumberOfWordsError {
    #[error("index {0} is out of range for `NumberOfWords`")]
    IndexOutOfBounds(usize),

    #[error("index is out of legal range: {0}")]
    FailedIntegerConversion(#[from] TryFromIntError),
}

/// Errors that can occur when debugging programs written for Triton VM.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DebugError {
    #[error("debugging is already enabled")]
    DebuggingAlreadyEnabled,

    #[error("debugging is not enabled")]
    DebuggingNotEnabled,

    #[error("the op stack must be newly created to enable debugging")]
    OpStackAlreadyInUse,

    #[error("the op stack is not large enough to apply the given type hint")]
    TypeHintOutOfBounds,
}

#[cfg(test)]
mod tests {
    use assert2::assert;
    use assert2::let_assert;
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
    fn instruction_pointer_overflow() {
        let program = triton_program!(nop);
        let_assert!(Err(err) = program.run([].into(), [].into()));
        let_assert!(InstructionError::InstructionPointerOverflow = err.source);
    }

    #[test]
    fn shrink_op_stack_too_much() {
        let program = triton_program!(pop 3 halt);
        let_assert!(Err(err) = program.run([].into(), [].into()));
        let_assert!(InstructionError::OpStackTooShallow = err.source);
    }

    #[test]
    fn return_without_call() {
        let program = triton_program!(return halt);
        let_assert!(Err(err) = program.run([].into(), [].into()));
        let_assert!(InstructionError::JumpStackIsEmpty = err.source);
    }

    #[test]
    fn recurse_without_call() {
        let program = triton_program!(recurse halt);
        let_assert!(Err(err) = program.run([].into(), [].into()));
        let_assert!(InstructionError::JumpStackIsEmpty = err.source);
    }

    #[test]
    fn assert_false() {
        let program = triton_program!(push 0 assert halt);
        let_assert!(Err(err) = program.run([].into(), [].into()));
        let_assert!(InstructionError::AssertionFailed = err.source);
    }

    #[test]
    fn print_unequal_vec_assert_error() {
        let program = triton_program! {
            push 4 push 3 push 2 push  1 push 0
            push 4 push 3 push 2 push 10 push 0
            assert_vector halt
        };
        let_assert!(Err(err) = program.run([].into(), [].into()));
        let_assert!(InstructionError::VectorAssertionFailed(index) = err.source);
        assert!(1 == usize::from(index));
    }

    #[test]
    fn swap_st0() {
        // The parser rejects this program. Therefore, construct it manually.
        let swap_0 = LabelledInstruction::Instruction(Swap(ST0));
        let halt = LabelledInstruction::Instruction(Halt);
        let program = Program::new(&[swap_0, halt]);
        let_assert!(Err(err) = program.run([].into(), [].into()));
        let_assert!(InstructionError::SwapST0 = err.source);
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

        let_assert!(Err(err) = program.run([].into(), [].into()));
        let_assert!(InstructionError::VectorAssertionFailed(index) = err.source);
        prop_assert_eq!(disturbance_index, usize::from(index));
    }

    #[test]
    fn inverse_of_zero() {
        let program = triton_program!(push 0 invert halt);
        let_assert!(Err(err) = program.run([].into(), [].into()));
        let_assert!(InstructionError::InverseOfZero = err.source);
    }

    #[test]
    fn xfe_inverse_of_zero() {
        let program = triton_program!(push 0 push 0 push 0 xinvert halt);
        let_assert!(Err(err) = program.run([].into(), [].into()));
        let_assert!(InstructionError::InverseOfZero = err.source);
    }

    #[test]
    fn division_by_zero() {
        let program = triton_program!(push 0 push 5 div_mod halt);
        let_assert!(Err(err) = program.run([].into(), [].into()));
        let_assert!(InstructionError::DivisionByZero = err.source);
    }

    #[test]
    fn log_of_zero() {
        let program = triton_program!(push 0 log_2_floor halt);
        let_assert!(Err(err) = program.run([].into(), [].into()));
        let_assert!(InstructionError::LogarithmOfZero = err.source);
    }

    #[test]
    fn failed_u32_conversion() {
        let program = triton_program!(push 4294967297 push 1 and halt);
        let_assert!(Err(err) = program.run([].into(), [].into()));
        let_assert!(InstructionError::FailedU32Conversion(element) = err.source);
        assert!(4294967297 == element.value());
    }
}
