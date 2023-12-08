use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;

use arbitrary::Arbitrary;
use get_size::GetSize;
use itertools::Itertools;
use num_traits::Zero;
use serde_derive::Deserialize;
use serde_derive::Serialize;
use strum::EnumCount;
use strum::EnumIter;
use strum::IntoEnumIterator;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::digest::Digest;
use twenty_first::shared_math::tip5::DIGEST_LENGTH;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::error::InstructionError;
use crate::error::InstructionError::*;
use crate::error::NumberOfWordsError;
use crate::error::OpStackElementError;
use crate::op_stack::OpStackElement::*;

type Result<T> = std::result::Result<T, InstructionError>;
type OpStackElementResult<T> = std::result::Result<T, OpStackElementError>;
type NumWordsResult<T> = std::result::Result<T, NumberOfWordsError>;

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpStack {
    pub stack: Vec<BFieldElement>,

    /// A helper “shadow stack” mimicking the behavior of the actual stack to help debugging
    /// programs written for Triton VM. Usually `None`, unless the program is being debugged.
    pub debug_type_hints: Option<Vec<Option<ElementTypeHint>>>,

    underflow_io_sequence: Vec<UnderflowIO>,
}

impl OpStack {
    pub fn new(program_digest: Digest) -> Self {
        let mut stack = vec![BFieldElement::zero(); OpStackElement::COUNT];

        let reverse_digest = program_digest.reversed().values();
        stack[..DIGEST_LENGTH].copy_from_slice(&reverse_digest);

        Self {
            stack,
            debug_type_hints: None,
            underflow_io_sequence: vec![],
        }
    }

    pub fn new_for_debugging(program_digest: Digest) -> Self {
        let mut op_stack = Self::new(program_digest);
        op_stack.debug_type_hints = Some(Self::debug_type_hints_for_initial_stack());
        op_stack
    }

    fn debug_type_hints_for_initial_stack() -> Vec<Option<ElementTypeHint>> {
        let element_type_hint_template = ElementTypeHint {
            type_name: "Digest".to_string(),
            variable_name: "program_digest".to_string(),
            index: None,
        };

        let mut type_hints = vec![None; NUM_OP_STACK_REGISTERS];
        for (index, type_hint) in type_hints[0..DIGEST_LENGTH].iter_mut().rev().enumerate() {
            let mut element_type_hint = element_type_hint_template.clone();
            element_type_hint.index = Some(index);
            *type_hint = Some(element_type_hint);
        }
        type_hints
    }

    pub(crate) fn push(&mut self, element: BFieldElement) {
        self.stack.push(element);
        self.record_underflow_io(UnderflowIO::Write);
    }

    pub(crate) fn pop(&mut self) -> Result<BFieldElement> {
        self.record_underflow_io(UnderflowIO::Read);
        let element = self.stack.pop().ok_or(OpStackTooShallow)?;
        Ok(element)
    }

    fn record_underflow_io(&mut self, io_type: fn(BFieldElement) -> UnderflowIO) {
        let underflow_io = io_type(self.first_underflow_element());
        self.underflow_io_sequence.push(underflow_io);
    }

    pub(crate) fn start_recording_underflow_io_sequence(&mut self) {
        self.underflow_io_sequence.clear();
    }

    pub(crate) fn stop_recording_underflow_io_sequence(&mut self) -> Vec<UnderflowIO> {
        self.underflow_io_sequence.drain(..).collect()
    }

    pub(crate) fn push_extension_field_element(&mut self, element: XFieldElement) {
        for coefficient in element.coefficients.into_iter().rev() {
            self.push(coefficient);
        }
    }

    pub(crate) fn pop_extension_field_element(&mut self) -> Result<XFieldElement> {
        let coefficients = self.pop_multiple()?;
        let element = XFieldElement::new(coefficients);
        Ok(element)
    }

    pub(crate) fn assert_is_u32(&self, stack_element: OpStackElement) -> Result<()> {
        let element = self.peek_at(stack_element);
        match element.value() <= u32::MAX as u64 {
            true => Ok(()),
            false => Err(FailedU32Conversion(element)),
        }
    }

    pub(crate) fn pop_u32(&mut self) -> Result<u32> {
        let element = self.pop()?;
        let element = element
            .try_into()
            .map_err(|_| FailedU32Conversion(element))?;
        Ok(element)
    }

    pub(crate) fn pop_multiple<const N: usize>(&mut self) -> Result<[BFieldElement; N]> {
        let mut elements = vec![];
        for _ in 0..N {
            let element = self.pop()?;
            elements.push(element);
        }
        let elements = elements.try_into().unwrap();
        Ok(elements)
    }

    pub(crate) fn peek_at(&self, stack_element: OpStackElement) -> BFieldElement {
        let stack_element_index = usize::from(stack_element);
        let top_of_stack_index = self.stack.len() - 1;
        self.stack[top_of_stack_index - stack_element_index]
    }

    pub(crate) fn peek_at_top_extension_field_element(&self) -> XFieldElement {
        let coefficient_0 = self.peek_at(ST0);
        let coefficient_1 = self.peek_at(ST1);
        let coefficient_2 = self.peek_at(ST2);
        let coefficients = [coefficient_0, coefficient_1, coefficient_2];
        XFieldElement::new(coefficients)
    }

    pub(crate) fn swap_top_with(&mut self, stack_element: OpStackElement) {
        let stack_element_index = usize::from(stack_element);
        let top_of_stack_index = self.stack.len() - 1;
        self.stack
            .swap(top_of_stack_index, top_of_stack_index - stack_element_index);
    }

    pub(crate) fn would_be_too_shallow(&self, stack_delta: i32) -> bool {
        self.stack.len() as i32 + stack_delta < OpStackElement::COUNT as i32
    }

    /// The address of the next free address of the op-stack. Equivalent to the current length of
    /// the op-stack.
    pub(crate) fn pointer(&self) -> BFieldElement {
        (self.stack.len() as u64).into()
    }

    /// The first element of the op-stack underflow memory, or 0 if the op-stack underflow memory
    /// is empty.
    pub(crate) fn first_underflow_element(&self) -> BFieldElement {
        let default = BFieldElement::zero();
        let Some(top_of_stack_index) = self.stack.len().checked_sub(1) else {
            return default;
        };
        let Some(underflow_start) = top_of_stack_index.checked_sub(OpStackElement::COUNT) else {
            return default;
        };
        self.stack.get(underflow_start).copied().unwrap_or(default)
    }
}

/// Indicates changes to the op-stack underflow memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, GetSize, Serialize, Deserialize, Arbitrary)]
#[must_use = "The change to underflow memory should be handled."]
pub enum UnderflowIO {
    Read(BFieldElement),
    Write(BFieldElement),
}

impl UnderflowIO {
    /// Remove spurious read/write sequences arising from temporary stack changes.
    ///
    /// For example, the sequence `[Read(5), Write(5), Read(7)]` can be replaced with `[Read(7)]`.
    /// Similarly, the sequence `[Write(5), Write(3), Read(3), Read(5), Write(7)]` can be replaced
    /// with `[Write(7)]`.
    pub fn canonicalize_sequence(sequence: &mut Vec<Self>) {
        while let Some(index) = Self::index_of_dual_pair(sequence) {
            let _ = sequence.remove(index);
            let _ = sequence.remove(index);
        }
    }

    fn index_of_dual_pair(sequence: &[Self]) -> Option<usize> {
        sequence
            .iter()
            .tuple_windows()
            .find_position(|(&left, &right)| left.is_dual_to(right))
            .map(|(index, _)| index)
    }

    fn is_dual_to(&self, other: Self) -> bool {
        match (self, other) {
            (&Self::Read(read), Self::Write(write)) => read == write,
            (&Self::Write(write), Self::Read(read)) => read == write,
            _ => false,
        }
    }

    /// Whether the sequence of underflow IOs consists of either only reads or only writes.
    pub fn is_uniform_sequence(sequence: &[Self]) -> bool {
        sequence.iter().all(|io| io.is_same_type_as(sequence[0]))
    }

    fn is_same_type_as(&self, other: Self) -> bool {
        matches!(
            (self, other),
            (&Self::Read(_), Self::Read(_)) | (&Self::Write(_), Self::Write(_))
        )
    }

    /// Whether the sequence of underflow IOs consists of only writes.
    pub fn is_writing_sequence(sequence: &[Self]) -> bool {
        sequence.iter().all(|io| io.grows_stack())
    }

    /// Whether the sequence of underflow IOs consists of only reads.
    pub fn is_reading_sequence(sequence: &[Self]) -> bool {
        sequence.iter().all(|io| io.shrinks_stack())
    }

    pub fn shrinks_stack(&self) -> bool {
        match self {
            Self::Read(_) => true,
            Self::Write(_) => false,
        }
    }

    pub fn grows_stack(&self) -> bool {
        match self {
            Self::Read(_) => false,
            Self::Write(_) => true,
        }
    }

    pub fn payload(self) -> BFieldElement {
        match self {
            Self::Read(payload) => payload,
            Self::Write(payload) => payload,
        }
    }
}

/// Represents the [`OpStack`] registers directly accessible by Triton VM.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Default,
    GetSize,
    Serialize,
    Deserialize,
    EnumCount,
    EnumIter,
    Arbitrary,
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

impl OpStackElement {
    pub const fn index(self) -> u32 {
        match self {
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

impl Display for OpStackElement {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let index = self.index();
        write!(f, "{index}")
    }
}

impl From<OpStackElement> for u32 {
    fn from(stack_element: OpStackElement) -> Self {
        stack_element.index()
    }
}

impl From<&OpStackElement> for u32 {
    fn from(&stack_element: &OpStackElement) -> Self {
        stack_element.into()
    }
}

impl TryFrom<u32> for OpStackElement {
    type Error = OpStackElementError;

    fn try_from(stack_index: u32) -> OpStackElementResult<Self> {
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
            _ => Err(Self::Error::IndexOutOfBounds(stack_index)),
        }
    }
}

impl From<OpStackElement> for u64 {
    fn from(stack_element: OpStackElement) -> Self {
        u32::from(stack_element).into()
    }
}

impl TryFrom<u64> for OpStackElement {
    type Error = OpStackElementError;

    fn try_from(stack_index: u64) -> OpStackElementResult<Self> {
        u32::try_from(stack_index)?.try_into()
    }
}

impl From<OpStackElement> for usize {
    fn from(stack_element: OpStackElement) -> Self {
        u32::from(stack_element) as usize
    }
}

impl From<&OpStackElement> for usize {
    fn from(&stack_element: &OpStackElement) -> Self {
        stack_element.into()
    }
}

impl From<OpStackElement> for i32 {
    fn from(stack_element: OpStackElement) -> Self {
        u32::from(stack_element) as i32
    }
}

impl From<&OpStackElement> for i32 {
    fn from(&stack_element: &OpStackElement) -> Self {
        stack_element.into()
    }
}

impl TryFrom<i32> for OpStackElement {
    type Error = OpStackElementError;

    fn try_from(stack_index: i32) -> OpStackElementResult<Self> {
        u32::try_from(stack_index)?.try_into()
    }
}

impl TryFrom<usize> for OpStackElement {
    type Error = OpStackElementError;

    fn try_from(stack_index: usize) -> OpStackElementResult<Self> {
        u32::try_from(stack_index)?.try_into()
    }
}

impl From<OpStackElement> for BFieldElement {
    fn from(stack_element: OpStackElement) -> Self {
        u32::from(stack_element).into()
    }
}

impl From<&OpStackElement> for BFieldElement {
    fn from(&stack_element: &OpStackElement) -> Self {
        stack_element.into()
    }
}

impl TryFrom<BFieldElement> for OpStackElement {
    type Error = OpStackElementError;

    fn try_from(stack_index: BFieldElement) -> OpStackElementResult<Self> {
        u32::try_from(stack_index)?.try_into()
    }
}

/// Represents the argument, _i.e._, the `n`, for instructions like `pop n` or `read_io n`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Default,
    GetSize,
    Serialize,
    Deserialize,
    EnumCount,
    EnumIter,
    Arbitrary,
)]
pub enum NumberOfWords {
    #[default]
    N1,
    N2,
    N3,
    N4,
    N5,
}

impl NumberOfWords {
    pub const fn num_words(self) -> usize {
        match self {
            Self::N1 => 1,
            Self::N2 => 2,
            Self::N3 => 3,
            Self::N4 => 4,
            Self::N5 => 5,
        }
    }

    pub(crate) fn legal_values() -> [usize; Self::COUNT] {
        let legal_indices = Self::iter().map(|n| n.num_words()).collect_vec();
        legal_indices.try_into().unwrap()
    }

    pub(crate) fn illegal_values() -> [usize; OpStackElement::COUNT - Self::COUNT] {
        let all_values = OpStackElement::iter().map(|st| st.index() as usize);
        let illegal_values = all_values
            .filter(|i| !Self::legal_values().contains(i))
            .collect_vec();
        illegal_values.try_into().unwrap()
    }
}

impl Display for NumberOfWords {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.num_words())
    }
}

impl From<NumberOfWords> for usize {
    fn from(num_words: NumberOfWords) -> Self {
        num_words.num_words()
    }
}

impl From<&NumberOfWords> for usize {
    fn from(&num_words: &NumberOfWords) -> Self {
        num_words.into()
    }
}

impl From<NumberOfWords> for u32 {
    fn from(num_words: NumberOfWords) -> Self {
        num_words.num_words() as u32
    }
}

impl From<&NumberOfWords> for u32 {
    fn from(&num_words: &NumberOfWords) -> Self {
        num_words.into()
    }
}

impl From<NumberOfWords> for u64 {
    fn from(num_words: NumberOfWords) -> Self {
        u32::from(num_words).into()
    }
}

impl From<&NumberOfWords> for u64 {
    fn from(&num_words: &NumberOfWords) -> Self {
        num_words.into()
    }
}

impl From<NumberOfWords> for OpStackElement {
    fn from(num_words: NumberOfWords) -> Self {
        OpStackElement::try_from(num_words.num_words()).unwrap()
    }
}

impl From<&NumberOfWords> for OpStackElement {
    fn from(&num_words: &NumberOfWords) -> Self {
        num_words.into()
    }
}

impl From<NumberOfWords> for BFieldElement {
    fn from(num_words: NumberOfWords) -> Self {
        u32::from(num_words).into()
    }
}

impl From<&NumberOfWords> for BFieldElement {
    fn from(&num_words: &NumberOfWords) -> Self {
        num_words.into()
    }
}

impl TryFrom<usize> for NumberOfWords {
    type Error = NumberOfWordsError;

    fn try_from(index: usize) -> NumWordsResult<Self> {
        match index {
            1 => Ok(Self::N1),
            2 => Ok(Self::N2),
            3 => Ok(Self::N3),
            4 => Ok(Self::N4),
            5 => Ok(Self::N5),
            _ => Err(Self::Error::IndexOutOfBounds(index)),
        }
    }
}

impl TryFrom<u32> for NumberOfWords {
    type Error = NumberOfWordsError;

    fn try_from(index: u32) -> NumWordsResult<Self> {
        usize::try_from(index)?.try_into()
    }
}

impl TryFrom<OpStackElement> for NumberOfWords {
    type Error = NumberOfWordsError;

    fn try_from(index: OpStackElement) -> NumWordsResult<Self> {
        usize::from(index).try_into()
    }
}

impl TryFrom<u64> for NumberOfWords {
    type Error = NumberOfWordsError;

    fn try_from(index: u64) -> NumWordsResult<Self> {
        usize::try_from(index)?.try_into()
    }
}

impl TryFrom<BFieldElement> for NumberOfWords {
    type Error = NumberOfWordsError;

    fn try_from(index: BFieldElement) -> NumWordsResult<Self> {
        u32::try_from(index)?.try_into()
    }
}

impl TryFrom<&BFieldElement> for NumberOfWords {
    type Error = NumberOfWordsError;

    fn try_from(&index: &BFieldElement) -> NumWordsResult<Self> {
        index.try_into()
    }
}

/// A hint about the type of a single stack element. Helps debugging programs written for Triton VM.
/// **Does not enforce types.**
#[derive(Debug, Clone, PartialEq, Eq, Hash, Arbitrary)]
pub struct ElementTypeHint {
    /// The name of the type. See [`TypeHint`][type_hint] for details.
    ///
    /// [type_hint]: crate::instruction::TypeHint
    pub type_name: String,

    /// The name of the variable. See [`TypeHint`][type_hint] for details.
    ///
    /// [type_hint]: crate::instruction::TypeHint
    pub variable_name: String,

    /// The index of the element within the type. For example, if the type is `Digest`, then this
    /// could be `0` for the first element, `1` for the second element, and so on.
    ///
    /// Does not apply to types that are not composed of multiple [`BFieldElement`]s, like `u32` or
    /// [`BFieldElement`] itself.
    pub index: Option<usize>,
}

#[cfg(test)]
mod tests {
    use assert2::assert;
    use assert2::let_assert;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use strum::IntoEnumIterator;
    use test_strategy::proptest;
    use twenty_first::shared_math::b_field_element::BFieldElement;

    use crate::op_stack::NumberOfWords::N1;

    use super::*;

    #[test]
    fn sanity() {
        let digest = Default::default();
        let mut op_stack = OpStack::new(digest);

        // verify height
        assert!(op_stack.stack.len() == 16);
        assert!(op_stack.pointer().value() as usize == op_stack.stack.len());

        // push elements 1 thru 17
        for i in 1..=17 {
            op_stack.push(BFieldElement::new(i as u64));
        }

        // verify height
        assert!(op_stack.stack.len() == 33);
        assert!(op_stack.pointer().value() as usize == op_stack.stack.len());

        // verify that all accessible items are different
        let mut container = vec![
            op_stack.peek_at(ST0),
            op_stack.peek_at(ST1),
            op_stack.peek_at(ST2),
            op_stack.peek_at(ST3),
            op_stack.peek_at(ST4),
            op_stack.peek_at(ST5),
            op_stack.peek_at(ST6),
            op_stack.peek_at(ST7),
            op_stack.peek_at(ST8),
            op_stack.peek_at(ST9),
            op_stack.peek_at(ST10),
            op_stack.peek_at(ST11),
            op_stack.peek_at(ST12),
            op_stack.peek_at(ST13),
            op_stack.peek_at(ST14),
            op_stack.peek_at(ST15),
            op_stack.first_underflow_element(),
        ];
        let len_before = container.len();
        container.sort_by_key(|a| a.value());
        container.dedup();
        let len_after = container.len();
        assert!(len_before == len_after);

        // pop 11 elements
        for _ in 0..11 {
            let _ = op_stack.pop().expect("can't pop");
        }

        // verify height
        assert!(op_stack.stack.len() == 22);
        assert!(op_stack.pointer().value() as usize == op_stack.stack.len());

        // pop 2 XFieldElements
        let _ = op_stack.pop_extension_field_element().expect("can't pop");
        let _ = op_stack.pop_extension_field_element().expect("can't pop");

        // verify height
        assert!(op_stack.stack.len() == 16);
        assert!(op_stack.pointer().value() as usize == op_stack.stack.len());

        // verify underflow
        assert!(op_stack.would_be_too_shallow(-1));
    }

    #[test]
    fn trying_to_access_first_underflow_element_never_panics() {
        let mut op_stack = OpStack::new(Default::default());
        let way_too_long = 2 * op_stack.stack.len();
        for _ in 0..way_too_long {
            let _ = op_stack.pop();
            let _ = op_stack.first_underflow_element();
        }
    }

    #[test]
    fn conversion_from_stack_element_to_u32_and_back_is_identity() {
        for stack_element in OpStackElement::iter() {
            let stack_index = u32::from(stack_element);
            let_assert!(Ok(stack_element_again) = OpStackElement::try_from(stack_index));
            assert!(stack_element == stack_element_again);
        }
    }

    #[test]
    fn conversion_from_stack_element_to_i32_and_back_is_identity() {
        for stack_element in OpStackElement::iter() {
            let stack_index = i32::from(stack_element);
            let_assert!(Ok(stack_element_again) = OpStackElement::try_from(stack_index));
            assert!(stack_element == stack_element_again);
        }
    }

    #[test]
    fn canonicalize_empty_underflow_io_sequence() {
        let mut sequence = vec![];
        UnderflowIO::canonicalize_sequence(&mut sequence);

        let expected_sequence = Vec::<UnderflowIO>::new();
        assert!(expected_sequence == sequence);
    }

    #[test]
    fn canonicalize_simple_underflow_io_sequence() {
        let mut sequence = vec![
            UnderflowIO::Read(5_u64.into()),
            UnderflowIO::Write(5_u64.into()),
            UnderflowIO::Read(7_u64.into()),
        ];
        UnderflowIO::canonicalize_sequence(&mut sequence);

        let expected_sequence = vec![UnderflowIO::Read(7_u64.into())];
        assert!(expected_sequence == sequence);
    }

    #[test]
    fn canonicalize_medium_complex_underflow_io_sequence() {
        let mut sequence = vec![
            UnderflowIO::Write(5_u64.into()),
            UnderflowIO::Write(3_u64.into()),
            UnderflowIO::Read(3_u64.into()),
            UnderflowIO::Read(5_u64.into()),
            UnderflowIO::Write(7_u64.into()),
        ];
        UnderflowIO::canonicalize_sequence(&mut sequence);

        let expected_sequence = vec![UnderflowIO::Write(7_u64.into())];
        assert!(expected_sequence == sequence);
    }

    #[proptest]
    fn underflow_io_either_shrinks_stack_or_grows_stack(
        #[strategy(arb())] underflow_io: UnderflowIO,
    ) {
        let shrinks_stack = underflow_io.shrinks_stack();
        let grows_stack = underflow_io.grows_stack();
        assert!(shrinks_stack ^ grows_stack);
    }

    #[proptest]
    fn non_empty_uniform_underflow_io_sequence_is_either_reading_or_writing(
        #[strategy(vec(arb(), 1..OpStackElement::COUNT))] sequence: Vec<UnderflowIO>,
    ) {
        let is_reading_sequence = UnderflowIO::is_reading_sequence(&sequence);
        let is_writing_sequence = UnderflowIO::is_writing_sequence(&sequence);
        if UnderflowIO::is_uniform_sequence(&sequence) {
            prop_assert!(is_reading_sequence ^ is_writing_sequence);
        } else {
            prop_assert!(!is_reading_sequence);
            prop_assert!(!is_writing_sequence);
        }
    }

    #[test]
    fn conversion_from_number_of_words_to_usize_and_back_is_identity() {
        for num_words in NumberOfWords::iter() {
            let stack_index = usize::from(num_words);
            let_assert!(Ok(num_words_again) = NumberOfWords::try_from(stack_index));
            assert!(num_words == num_words_again);
        }
    }

    #[test]
    fn conversion_from_number_of_words_to_u64_and_back_is_identity() {
        for num_words in NumberOfWords::iter() {
            let stack_index = u64::from(num_words);
            let_assert!(Ok(num_words_again) = NumberOfWords::try_from(stack_index));
            assert!(num_words == num_words_again);
        }
    }

    #[test]
    fn conversion_from_number_of_words_to_op_stack_element_and_back_is_identity() {
        for num_words in NumberOfWords::iter() {
            let stack_element = OpStackElement::from(num_words);
            let_assert!(Ok(num_words_again) = NumberOfWords::try_from(stack_element));
            assert!(num_words == num_words_again);
        }
    }

    #[test]
    fn convert_from_various_primitive_types_to_op_stack_element() {
        assert!(let Ok(_) = OpStackElement::try_from(0_u32));
        assert!(let Ok(_) = OpStackElement::try_from(0_u64));
        assert!(let Ok(_) = OpStackElement::try_from(0_usize));
        assert!(let Ok(_) = OpStackElement::try_from(0_i32));
        assert!(let Ok(_) = OpStackElement::try_from(BFieldElement::zero()));
    }

    #[test]
    fn convert_from_various_primitive_types_to_number_of_words() {
        assert!(let Ok(_) = NumberOfWords::try_from(1_u32));
        assert!(let Ok(_) = NumberOfWords::try_from(1_u64));
        assert!(let Ok(_) = NumberOfWords::try_from(1_usize));
        assert!(let Ok(_) = NumberOfWords::try_from(BFieldElement::new(1)));
        assert!(let Ok(_) = NumberOfWords::try_from(ST1));
    }

    #[test]
    fn convert_from_op_stack_element_to_various_primitive_types() {
        let _ = u32::from(ST0);
        let _ = u64::from(ST0);
        let _ = usize::from(ST0);
        let _ = i32::from(ST0);
        let _ = BFieldElement::from(ST0);

        let _ = u32::from(&ST0);
        let _ = usize::from(&ST0);
        let _ = i32::from(&ST0);
        let _ = BFieldElement::from(&ST0);
    }

    #[test]
    fn convert_from_number_of_words_to_various_primitive_types() {
        let _ = u32::from(N1);
        let _ = u64::from(N1);
        let _ = usize::from(N1);
        let _ = BFieldElement::from(N1);
        let _ = OpStackElement::from(N1);

        let _ = u32::from(&N1);
        let _ = u64::from(&N1);
        let _ = usize::from(&N1);
        let _ = BFieldElement::from(&N1);
        let _ = OpStackElement::from(&N1);
    }

    #[proptest]
    fn out_of_range_op_stack_element_gives_error(
        #[strategy(arb())]
        #[filter(!OpStackElement::iter().map(|o| o.index()).contains(&(#index.value() as u32)))]
        index: BFieldElement,
    ) {
        assert!(let Err(_) = OpStackElement::try_from(index));
    }

    #[proptest]
    fn out_of_range_number_of_words_gives_error(
        #[strategy(arb())]
        #[filter(!NumberOfWords::legal_values().contains(&(#index.value() as usize)))]
        index: BFieldElement,
    ) {
        assert!(let Err(_) = NumberOfWords::try_from(&index));
    }

    #[test]
    fn number_of_words_to_b_field_element_gives_expected_range() {
        let computed_range = NumberOfWords::iter()
            .map(|num_words| BFieldElement::from(&num_words).value())
            .collect_vec();
        let expected_range = (1..=5).collect_vec();
        assert!(computed_range == expected_range);
    }

    #[test]
    fn number_of_legal_number_of_words_corresponds_to_distinct_number_of_number_of_words() {
        let legal_values = NumberOfWords::legal_values();
        let num_distinct_values = NumberOfWords::COUNT;
        assert!(num_distinct_values == legal_values.len());
    }

    #[test]
    fn compute_illegal_values_of_number_of_words() {
        let _ = NumberOfWords::illegal_values();
    }
}
