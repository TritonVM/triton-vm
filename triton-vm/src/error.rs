pub use isa::error::InstructionError;
pub use isa::error::NumberOfWordsError;
pub use isa::error::OpStackElementError;
pub use isa::error::OpStackError;
pub use isa::error::ParseError;
pub use isa::error::ProgramDecodingError;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use thiserror::Error;
use twenty_first::error::MerkleTreeError;
use twenty_first::prelude::*;

use crate::proof_item::ProofItem;
use crate::proof_item::ProofItemVariant;
use crate::proof_stream::ProofStream;
use crate::vm::VMState;

/// Indicates a runtime error that resulted in a crash of Triton VM.
#[derive(Debug, Clone, Eq, PartialEq, Error)]
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
#[derive(Debug, Copy, Clone, Eq, PartialEq, Error)]
pub enum ArithmeticDomainError {
    #[error("the domain's length must be a power of 2 but was {0}")]
    PrimitiveRootNotSupported(u64),

    #[error("the domain's length must be at least 2 to be halved, but it was {0}")]
    TooSmallForHalving(usize),
}

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ProofStreamError {
    #[error("queue must be non-empty in order to dequeue an item")]
    EmptyQueue,

    #[error("expected {expected}, got {got}")]
    UnexpectedItem {
        expected: ProofItemVariant,
        got: ProofItem,
    },

    #[error("the proof stream must contain a log2_padded_height item")]
    NoLog2PaddedHeight,

    #[error("the proof stream must contain exactly one log2_padded_height item")]
    TooManyLog2PaddedHeights,

    #[error(transparent)]
    DecodingError(#[from] <ProofStream as BFieldCodec>::Error),
}

#[non_exhaustive]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Error)]
pub enum FriSetupError {
    #[error("the expansion factor must be greater than 1")]
    ExpansionFactorTooSmall,

    #[error("the expansion factor must be a power of 2")]
    ExpansionFactorUnsupported,

    #[error("the expansion factor must be smaller than the domain length")]
    ExpansionFactorMismatch,

    #[error(transparent)]
    ArithmeticDomainError(#[from] ArithmeticDomainError),
}

#[non_exhaustive]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Error)]
pub enum FriProvingError {
    #[error(transparent)]
    MerkleTreeError(#[from] MerkleTreeError),

    #[error(transparent)]
    ArithmeticDomainError(#[from] ArithmeticDomainError),
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

    #[error("evaluations of last round's polynomial and last round codeword do not match")]
    LastRoundPolynomialEvaluationMismatch,

    #[error("last round's polynomial has too high degree")]
    LastRoundPolynomialHasTooHighDegree,

    #[error("received codeword of last round does not correspond to its commitment")]
    BadMerkleRootForLastCodeword,

    #[error(transparent)]
    ProofStreamError(#[from] ProofStreamError),

    #[error(transparent)]
    MerkleTreeError(#[from] MerkleTreeError),

    #[error(transparent)]
    ArithmeticDomainError(#[from] ArithmeticDomainError),
}

#[non_exhaustive]
#[derive(Debug, Clone, Eq, PartialEq, Error)]
pub enum ProvingError {
    #[error("claimed program digest does not match actual program digest")]
    ProgramDigestMismatch,

    #[error("claimed public output does not match actual public output")]
    PublicOutputMismatch,

    #[error("expected row of length {expected_len} but got {actual_len}")]
    TableRowConversionError {
        expected_len: usize,
        actual_len: usize,
    },

    #[error(transparent)]
    MerkleTreeError(#[from] MerkleTreeError),

    #[error(transparent)]
    ArithmeticDomainError(#[from] ArithmeticDomainError),

    #[error(transparent)]
    FriSetupError(#[from] FriSetupError),

    #[error(transparent)]
    FriProvingError(#[from] FriProvingError),

    #[error(transparent)]
    VMError(#[from] VMError),
}

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum VerificationError {
    #[error("received and computed out-of-domain quotient values don't match")]
    OutOfDomainQuotientValueMismatch,

    #[error("failed to verify authentication path for main codeword")]
    MainCodewordAuthenticationFailure,

    #[error("failed to verify authentication path for auxiliary codeword")]
    AuxiliaryCodewordAuthenticationFailure,

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

    #[error("the number of received main table rows does not match the parameters")]
    IncorrectNumberOfMainTableRows,

    #[error("the number of received auxiliary table rows does not match the parameters")]
    IncorrectNumberOfAuxTableRows,

    #[error(transparent)]
    ProofStreamError(#[from] ProofStreamError),

    #[error(transparent)]
    ArithmeticDomainError(#[from] ArithmeticDomainError),

    #[error(transparent)]
    FriSetupError(#[from] FriSetupError),

    #[error(transparent)]
    FriValidationError(#[from] FriValidationError),
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use assert2::assert;
    use assert2::let_assert;
    use isa::op_stack::OpStackError;
    use isa::triton_program;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use super::*;
    use crate::prelude::VM;

    #[test]
    fn instruction_pointer_overflow() {
        let program = triton_program!(nop);
        let_assert!(Err(err) = VM::run(program, [].into(), [].into()));
        let_assert!(InstructionError::InstructionPointerOverflow = err.source);
    }

    #[test]
    fn shrink_op_stack_too_much() {
        let program = triton_program!(pop 3 halt);
        let_assert!(Err(err) = VM::run(program, [].into(), [].into()));
        let_assert!(InstructionError::OpStackError(OpStackError::TooShallow) = err.source);
    }

    #[test]
    fn return_without_call() {
        let program = triton_program!(return halt);
        let_assert!(Err(err) = VM::run(program, [].into(), [].into()));
        let_assert!(InstructionError::JumpStackIsEmpty = err.source);
    }

    #[test]
    fn recurse_without_call() {
        let program = triton_program!(recurse halt);
        let_assert!(Err(err) = VM::run(program, [].into(), [].into()));
        let_assert!(InstructionError::JumpStackIsEmpty = err.source);
    }

    #[test]
    fn assert_false() {
        let program = triton_program!(push 0 assert halt);
        let_assert!(Err(err) = VM::run(program, [].into(), [].into()));
        let_assert!(InstructionError::AssertionFailed(error) = err.source);
        assert!(bfe!(1) == error.expected);
        assert!(bfe!(0) == error.actual);
        assert!(error.id.is_none());
    }

    #[test]
    fn assert_false_with_assertion_context() {
        let program = triton_program!(push 0 assert error_id 42 halt);
        let_assert!(Err(err) = VM::run(program, [].into(), [].into()));
        let_assert!(InstructionError::AssertionFailed(err) = err.source);
        assert!(bfe!(1) == err.expected);
        assert!(bfe!(0) == err.actual);
        assert!(Some(42) == err.id);
    }

    #[test]
    fn print_unequal_vec_assert_error() {
        let program = triton_program! {
            push 4 push 3 push 2 push  1 push 0
            push 4 push 3 push 2 push 10 push 0
            assert_vector halt
        };
        let_assert!(Err(err) = VM::run(program, [].into(), [].into()));
        let_assert!(InstructionError::VectorAssertionFailed(index, err) = err.source);
        assert!(1 == index);
        assert!(bfe!(10) == err.expected);
        assert!(bfe!(1) == err.actual);
        assert!(None == err.id);
    }

    #[proptest]
    fn assertion_context_error_id_is_propagated_correctly(
        #[filter(#actual != 1)] actual: i64,
        error_id: Option<i128>,
    ) {
        let program = if let Some(id) = error_id {
            triton_program! {push {actual} assert error_id {id}}
        } else {
            triton_program! {push {actual} assert}
        };
        let_assert!(Err(err) = VM::run(program, [].into(), [].into()));
        let_assert!(InstructionError::AssertionFailed(err) = err.source);
        prop_assert_eq!(bfe!(1), err.expected);
        prop_assert_eq!(bfe!(actual), err.actual);
        prop_assert_eq!(error_id, err.id);
    }

    #[proptest]
    fn triggering_assertion_failure_results_in_expected_error_id(
        #[strategy(0_usize..5)] failure_index: usize,
    ) {
        let mut almost_all_ones = [1; 5];
        almost_all_ones[failure_index] = 0;

        let program = triton_program! {
            push {almost_all_ones[0]} assert error_id 0
            push {almost_all_ones[1]} assert error_id 1
            push {almost_all_ones[2]} assert error_id 2
            push {almost_all_ones[3]} assert error_id 3
            push {almost_all_ones[4]} assert error_id 4
        };
        let_assert!(Err(err) = VM::run(program, [].into(), [].into()));
        let_assert!(InstructionError::AssertionFailed(err) = err.source);
        let expected_error_id = i128::try_from(failure_index)?;
        prop_assert_eq!(expected_error_id, err.id.unwrap());
    }

    #[proptest]
    fn assert_unequal_vec(
        #[strategy(arb())] test_vector: [BFieldElement; Digest::LEN],
        #[strategy(0..Digest::LEN)] disturbance_index: usize,
        #[strategy(arb())]
        #[filter(#test_vector[#disturbance_index] != #random_element)]
        random_element: BFieldElement,
        error_id: i128,
    ) {
        let mut disturbed_vector = test_vector;
        disturbed_vector[disturbance_index] = random_element;

        let program = triton_program! {
            push {disturbed_vector[4]}
            push {disturbed_vector[3]}
            push {disturbed_vector[2]}
            push {disturbed_vector[1]}
            push {disturbed_vector[0]}

            push {test_vector[4]}
            push {test_vector[3]}
            push {test_vector[2]}
            push {test_vector[1]}
            push {test_vector[0]}

            assert_vector error_id {error_id}
            halt
        };

        let_assert!(Err(err) = VM::run(program, [].into(), [].into()));
        let_assert!(InstructionError::VectorAssertionFailed(index, err) = err.source);
        prop_assert_eq!(disturbance_index, index);
        prop_assert_eq!(test_vector[index], err.expected, "unequal “expected”");
        prop_assert_eq!(disturbed_vector[index], err.actual, "unequal “actual”");
        prop_assert_eq!(Some(error_id), err.id);
    }

    #[test]
    fn inverse_of_zero() {
        let program = triton_program!(push 0 invert halt);
        let_assert!(Err(err) = VM::run(program, [].into(), [].into()));
        let_assert!(InstructionError::InverseOfZero = err.source);
    }

    #[test]
    fn xfe_inverse_of_zero() {
        let program = triton_program!(push 0 push 0 push 0 x_invert halt);
        let_assert!(Err(err) = VM::run(program, [].into(), [].into()));
        let_assert!(InstructionError::InverseOfZero = err.source);
    }

    #[test]
    fn division_by_zero() {
        let program = triton_program!(push 0 push 5 div_mod halt);
        let_assert!(Err(err) = VM::run(program, [].into(), [].into()));
        let_assert!(InstructionError::DivisionByZero = err.source);
    }

    #[test]
    fn log_of_zero() {
        let program = triton_program!(push 0 log_2_floor halt);
        let_assert!(Err(err) = VM::run(program, [].into(), [].into()));
        let_assert!(InstructionError::LogarithmOfZero = err.source);
    }

    #[test]
    fn failed_u32_conversion() {
        let program = triton_program!(push 4294967297 push 1 and halt);
        let_assert!(Err(err) = VM::run(program, [].into(), [].into()));
        let_assert!(InstructionError::OpStackError(err) = err.source);
        let_assert!(OpStackError::FailedU32Conversion(element) = err);
        assert!(4_294_967_297 == element.value());
    }
}
