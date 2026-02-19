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

pub(crate) const USIZE_TO_U64_ERR: &str =
    "internal error: type `usize` should have at most 64 bits";
pub(crate) const U32_TO_USIZE_ERR: &str =
    "internal error: type `usize` should have at least 32 bits";

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

    #[error("the exponent must be a power of 2, but it was {0}")]
    IllegalExponent(usize),
}

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ProofStreamError {
    #[error("queue must be non-empty in order to dequeue an item")]
    EmptyQueue,

    #[error("expected {expected}, got {got}")]
    UnexpectedItem {
        expected: ProofItemVariant,
        got: Box<ProofItem>,
    },

    #[error("the proof stream must contain a log2_padded_height item")]
    NoLog2PaddedHeight,

    #[error("the proof stream must contain exactly one log2_padded_height item")]
    TooManyLog2PaddedHeights,

    #[error(transparent)]
    DecodingError(#[from] <ProofStream as BFieldCodec>::Error),
}

/// Indicates the choice of an invalid combination of initial parameters for
/// the [low-degree test](crate::low_degree_test::LowDegreeTest).
#[non_exhaustive]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Error)]
pub enum LdtParameterError {
    #[error("the log₂ of the folding factor must be greater than or equal to 2, but is {0}")]
    TooSmallLog2FoldingFactor(usize),

    #[error("the log₂ of the folding factor must be less than 32, but is {0}")]
    TooBigLog2FoldingFactor(usize),

    #[error("the log₂ of the initial expansion factor must be greater than 0")]
    TooSmallInitialExpansionFactor,

    #[error("the log₂ of the initial expansion factor must be less than 32")]
    TooBigInitialExpansionFactor,

    #[error("the “high degree” threshold must be greater than or equal to the folding factor")]
    TooLowDegreeOfHighDegreePolynomials,

    #[error("the initial domain must be shorter than 2^32, but was 2^{0}")]
    InitialDomainTooBig(u64),
}

/// Indicates an error that occurred during
/// [proving](crate::low_degree_test::LowDegreeTest::prove) of the
/// [low-degree test](crate::low_degree_test::LowDegreeTest).
#[non_exhaustive]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Error)]
pub enum LdtProvingError {
    #[error("initial domain len ({domain_len}) must equal first codeword len ({codeword_len})")]
    InitialCodewordMismatch {
        domain_len: usize,
        codeword_len: usize,
    },
}

/// Indicates an error that occurred during
/// [verification](crate::low_degree_test::LowDegreeTest::verify) of the
/// [low-degree test](crate::low_degree_test::LowDegreeTest).
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum LdtVerificationError {
    #[error("the number of revealed leaves does not match the number of (in-domain) queries")]
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
    LdtParameterError(#[from] LdtParameterError),

    #[error(transparent)]
    LdtProvingError(#[from] LdtProvingError),

    #[error(transparent)]
    VMError(#[from] VMError),
}

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum VerificationError {
    #[error("received a log₂ padded height larger than (or equal to) 32")]
    Log2PaddedHeightTooLarge,

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

    #[error("the number of received low-degree test codeword values does not match the parameters")]
    IncorrectNumberOfLowDegTestValues,

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
    LdtParameterError(#[from] LdtParameterError),

    #[error(transparent)]
    LdtVerificationError(#[from] LdtVerificationError),
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use assert2::assert;
    use isa::op_stack::OpStackError;
    use isa::triton_program;
    use proptest::prelude::*;
    use proptest_arbitrary_adapter::arb;

    use super::*;
    use crate::prelude::VM;
    use crate::tests::proptest;
    use crate::tests::test;

    #[macro_rules_attr::apply(test)]
    fn instruction_pointer_overflow() {
        let program = triton_program!(nop);
        assert!(let Err(err) = VM::run(program, [].into(), [].into()));
        assert!(let InstructionError::InstructionPointerOverflow = err.source);
    }

    #[macro_rules_attr::apply(test)]
    fn shrink_op_stack_too_much() {
        let program = triton_program!(pop 3 halt);
        assert!(let Err(err) = VM::run(program, [].into(), [].into()));
        assert!(let InstructionError::OpStackError(OpStackError::TooShallow) = err.source);
    }

    #[macro_rules_attr::apply(test)]
    fn return_without_call() {
        let program = triton_program!(return halt);
        assert!(let Err(err) = VM::run(program, [].into(), [].into()));
        assert!(let InstructionError::JumpStackIsEmpty = err.source);
    }

    #[macro_rules_attr::apply(test)]
    fn recurse_without_call() {
        let program = triton_program!(recurse halt);
        assert!(let Err(err) = VM::run(program, [].into(), [].into()));
        assert!(let InstructionError::JumpStackIsEmpty = err.source);
    }

    #[macro_rules_attr::apply(test)]
    fn assert_false() {
        let program = triton_program!(push 0 assert halt);
        assert!(let Err(err) = VM::run(program, [].into(), [].into()));
        assert!(let InstructionError::AssertionFailed(error) = err.source);
        assert!(bfe!(1) == error.expected);
        assert!(bfe!(0) == error.actual);
        assert!(error.id.is_none());
    }

    #[macro_rules_attr::apply(test)]
    fn assert_false_with_assertion_context() {
        let program = triton_program!(push 0 assert error_id 42 halt);
        assert!(let Err(err) = VM::run(program, [].into(), [].into()));
        assert!(let InstructionError::AssertionFailed(err) = err.source);
        assert!(bfe!(1) == err.expected);
        assert!(bfe!(0) == err.actual);
        assert!(Some(42) == err.id);
    }

    #[macro_rules_attr::apply(test)]
    fn print_unequal_vec_assert_error() {
        let program = triton_program! {
            push 4 push 3 push 2 push  1 push 0
            push 4 push 3 push 2 push 10 push 0
            assert_vector halt
        };
        assert!(let Err(err) = VM::run(program, [].into(), [].into()));
        assert!(let InstructionError::VectorAssertionFailed(index, err) = err.source);
        assert!(1 == index);
        assert!(bfe!(10) == err.expected);
        assert!(bfe!(1) == err.actual);
        assert!(None == err.id);
    }

    #[macro_rules_attr::apply(proptest)]
    fn assertion_context_error_id_is_propagated_correctly(
        #[filter(#actual != 1)] actual: i64,
        error_id: Option<i128>,
    ) {
        let program = if let Some(id) = error_id {
            triton_program! {push {actual} assert error_id {id}}
        } else {
            triton_program! {push {actual} assert}
        };
        assert!(let Err(err) = VM::run(program, [].into(), [].into()));
        assert!(let InstructionError::AssertionFailed(err) = err.source);
        prop_assert_eq!(bfe!(1), err.expected);
        prop_assert_eq!(bfe!(actual), err.actual);
        prop_assert_eq!(error_id, err.id);
    }

    #[macro_rules_attr::apply(proptest)]
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
        assert!(let Err(err) = VM::run(program, [].into(), [].into()));
        assert!(let InstructionError::AssertionFailed(err) = err.source);
        let expected_error_id = i128::try_from(failure_index)?;
        prop_assert_eq!(expected_error_id, err.id.unwrap());
    }

    #[macro_rules_attr::apply(proptest)]
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

        assert!(let Err(err) = VM::run(program, [].into(), [].into()));
        assert!(let InstructionError::VectorAssertionFailed(index, err) = err.source);
        prop_assert_eq!(disturbance_index, index);
        prop_assert_eq!(test_vector[index], err.expected, "unequal “expected”");
        prop_assert_eq!(disturbed_vector[index], err.actual, "unequal “actual”");
        prop_assert_eq!(Some(error_id), err.id);
    }

    #[macro_rules_attr::apply(test)]
    fn inverse_of_zero() {
        let program = triton_program!(push 0 invert halt);
        assert!(let Err(err) = VM::run(program, [].into(), [].into()));
        assert!(let InstructionError::InverseOfZero = err.source);
    }

    #[macro_rules_attr::apply(test)]
    fn xfe_inverse_of_zero() {
        let program = triton_program!(push 0 push 0 push 0 x_invert halt);
        assert!(let Err(err) = VM::run(program, [].into(), [].into()));
        assert!(let InstructionError::InverseOfZero = err.source);
    }

    #[macro_rules_attr::apply(test)]
    fn division_by_zero() {
        let program = triton_program!(push 0 push 5 div_mod halt);
        assert!(let Err(err) = VM::run(program, [].into(), [].into()));
        assert!(let InstructionError::DivisionByZero = err.source);
    }

    #[macro_rules_attr::apply(test)]
    fn log_of_zero() {
        let program = triton_program!(push 0 log_2_floor halt);
        assert!(let Err(err) = VM::run(program, [].into(), [].into()));
        assert!(let InstructionError::LogarithmOfZero = err.source);
    }

    #[macro_rules_attr::apply(test)]
    fn failed_u32_conversion() {
        let program = triton_program!(push 4294967297 push 1 and halt);
        assert!(let Err(err) = VM::run(program, [].into(), [].into()));
        assert!(let InstructionError::OpStackError(err) = err.source);
        assert!(let OpStackError::FailedU32Conversion(element) = err);
        assert!(4_294_967_297 == element.value());
    }
}
