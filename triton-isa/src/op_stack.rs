use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::num::TryFromIntError;
use std::ops::Index;
use std::ops::IndexMut;

use arbitrary::Arbitrary;
use get_size2::GetSize;
use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;
use strum::EnumCount;
use strum::EnumIter;
use strum::IntoEnumIterator;
use thiserror::Error;
use twenty_first::prelude::*;

type Result<T> = std::result::Result<T, OpStackError>;
type OpStackElementResult<T> = std::result::Result<T, OpStackElementError>;
type NumWordsResult<T> = std::result::Result<T, NumberOfWordsError>;

/// The number of registers dedicated to the top of the operational stack.
pub const NUM_OP_STACK_REGISTERS: usize = OpStackElement::COUNT;

/// The operational stack of Triton VM.
/// It always contains at least [`OpStackElement::COUNT`] elements. Initially,
/// the bottom-most [`Digest::LEN`] elements equal the digest of the program
/// being executed. The remaining elements are initially 0.
///
/// The OpStack is represented as one contiguous piece of memory, and Triton VM
/// uses it as such. For reasons of arithmetization, however, there is a
/// distinction between the op-stack registers and the op-stack underflow
/// memory. The op-stack registers are the first [`OpStackElement::COUNT`]
/// elements of the op-stack, and the op-stack underflow memory is the remaining
/// elements.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize, Arbitrary)]
pub struct OpStack {
    /// The underlying, actual stack. When manually accessing, be aware of
    /// reversed indexing: while `op_stack[0]` is the top of the stack,
    /// `op_stack.stack[0]` is the lowest element in the stack.
    pub stack: Vec<BFieldElement>,

    underflow_io_sequence: Vec<UnderflowIO>,
}

#[non_exhaustive]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Error)]
pub enum OpStackError {
    #[error("operational stack is too shallow")]
    TooShallow,

    #[error("failed to convert BFieldElement {0} into u32")]
    FailedU32Conversion(BFieldElement),
}

impl OpStack {
    pub fn new(program_digest: Digest) -> Self {
        let mut stack = bfe_vec![0; OpStackElement::COUNT];

        let reverse_digest = program_digest.reversed().values();
        stack[..Digest::LEN].copy_from_slice(&reverse_digest);

        Self {
            stack,
            underflow_io_sequence: vec![],
        }
    }

    // If the op stack is empty, things have gone horribly wrong. Suppressing
    // this lint is preferred to implementing a basically useless `is_empty()`.
    #[expect(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    pub fn push(&mut self, element: BFieldElement) {
        self.stack.push(element);
        self.record_underflow_io(UnderflowIO::Write);
    }

    pub fn pop(&mut self) -> Result<BFieldElement> {
        self.record_underflow_io(UnderflowIO::Read);
        self.stack.pop().ok_or(OpStackError::TooShallow)
    }

    pub fn insert(&mut self, index: OpStackElement, element: BFieldElement) {
        let insertion_index = self.len() - usize::from(index);
        self.stack.insert(insertion_index, element);
        self.record_underflow_io(UnderflowIO::Write);
    }

    pub fn remove(&mut self, index: OpStackElement) -> BFieldElement {
        self.record_underflow_io(UnderflowIO::Read);
        let top_of_stack = self.len() - 1;
        let index = top_of_stack - usize::from(index);
        self.stack.remove(index)
    }

    fn record_underflow_io(&mut self, io_type: fn(BFieldElement) -> UnderflowIO) {
        let underflow_io = io_type(self.first_underflow_element());
        self.underflow_io_sequence.push(underflow_io);
    }

    pub fn start_recording_underflow_io_sequence(&mut self) {
        self.underflow_io_sequence.clear();
    }

    pub fn stop_recording_underflow_io_sequence(&mut self) -> Vec<UnderflowIO> {
        self.underflow_io_sequence.drain(..).collect()
    }

    pub fn push_extension_field_element(&mut self, element: XFieldElement) {
        for coefficient in element.coefficients.into_iter().rev() {
            self.push(coefficient);
        }
    }

    pub fn pop_extension_field_element(&mut self) -> Result<XFieldElement> {
        let coefficients = self.pop_multiple()?;
        Ok(xfe!(coefficients))
    }

    pub fn is_u32(&self, stack_element: OpStackElement) -> Result<()> {
        self.get_u32(stack_element).map(|_| ())
    }

    pub fn get_u32(&self, stack_element: OpStackElement) -> Result<u32> {
        let element = self[stack_element];
        element
            .try_into()
            .map_err(|_| OpStackError::FailedU32Conversion(element))
    }

    pub fn pop_u32(&mut self) -> Result<u32> {
        let element = self.pop()?;
        element
            .try_into()
            .map_err(|_| OpStackError::FailedU32Conversion(element))
    }

    pub fn pop_multiple<const N: usize>(&mut self) -> Result<[BFieldElement; N]> {
        let mut elements = bfe_array![0; N];
        for element in &mut elements {
            *element = self.pop()?;
        }
        Ok(elements)
    }

    pub fn peek_at_top_extension_field_element(&self) -> XFieldElement {
        xfe!([self[0], self[1], self[2]])
    }

    pub fn would_be_too_shallow(&self, stack_delta: i32) -> bool {
        self.len() as i32 + stack_delta < OpStackElement::COUNT as i32
    }

    /// The address of the next free address of the op-stack. Equivalent to the
    /// current length of the op-stack.
    pub fn pointer(&self) -> BFieldElement {
        u64::try_from(self.len()).unwrap().into()
    }

    /// The first element of the op-stack underflow memory, or 0 if the op-stack
    /// underflow memory is empty.
    pub(crate) fn first_underflow_element(&self) -> BFieldElement {
        let default = bfe!(0);
        let Some(top_of_stack_index) = self.len().checked_sub(1) else {
            return default;
        };
        let Some(underflow_start) = top_of_stack_index.checked_sub(OpStackElement::COUNT) else {
            return default;
        };
        self.stack.get(underflow_start).copied().unwrap_or(default)
    }
}

impl Index<usize> for OpStack {
    type Output = BFieldElement;

    fn index(&self, index: usize) -> &Self::Output {
        let top_of_stack = self.len() - 1;
        &self.stack[top_of_stack - index]
    }
}

impl IndexMut<usize> for OpStack {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let top_of_stack = self.len() - 1;
        &mut self.stack[top_of_stack - index]
    }
}

impl Index<OpStackElement> for OpStack {
    type Output = BFieldElement;

    fn index(&self, stack_element: OpStackElement) -> &Self::Output {
        &self[usize::from(stack_element)]
    }
}

impl IndexMut<OpStackElement> for OpStack {
    fn index_mut(&mut self, stack_element: OpStackElement) -> &mut Self::Output {
        &mut self[usize::from(stack_element)]
    }
}

impl IntoIterator for OpStack {
    type Item = BFieldElement;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let mut stack = self.stack;
        stack.reverse();
        stack.into_iter()
    }
}

/// Indicates changes to the op-stack underflow memory.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, GetSize, Arbitrary)]
#[must_use = "The change to underflow memory should be handled."]
pub enum UnderflowIO {
    Read(BFieldElement),
    Write(BFieldElement),
}

impl UnderflowIO {
    /// Remove spurious read/write sequences arising from temporary stack
    /// changes.
    ///
    /// For example, the sequence `[Read(5), Write(5), Read(7)]` can be replaced
    /// with `[Read(7)]`. Similarly, the sequence `[Write(5), Write(3),
    /// Read(3), Read(5), Write(7)]` can be replaced with `[Write(7)]`.
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
            .find_position(|&(&left, &right)| left.is_dual_to(right))
            .map(|(index, _)| index)
    }

    fn is_dual_to(&self, other: Self) -> bool {
        match (self, other) {
            (&Self::Read(read), Self::Write(write)) => read == write,
            (&Self::Write(write), Self::Read(read)) => read == write,
            _ => false,
        }
    }

    /// Whether the sequence of underflow IOs consists of either only reads or
    /// only writes.
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
    Default,
    Copy,
    Clone,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
    Serialize,
    Deserialize,
    EnumCount,
    EnumIter,
    GetSize,
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

#[non_exhaustive]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Error)]
pub enum OpStackElementError {
    #[error("index {0} is out of range for `OpStackElement`")]
    IndexOutOfBounds(u32),

    #[error(transparent)]
    FailedIntegerConversion(#[from] TryFromIntError),
}

impl OpStackElement {
    pub const fn index(self) -> u32 {
        match self {
            OpStackElement::ST0 => 0,
            OpStackElement::ST1 => 1,
            OpStackElement::ST2 => 2,
            OpStackElement::ST3 => 3,
            OpStackElement::ST4 => 4,
            OpStackElement::ST5 => 5,
            OpStackElement::ST6 => 6,
            OpStackElement::ST7 => 7,
            OpStackElement::ST8 => 8,
            OpStackElement::ST9 => 9,
            OpStackElement::ST10 => 10,
            OpStackElement::ST11 => 11,
            OpStackElement::ST12 => 12,
            OpStackElement::ST13 => 13,
            OpStackElement::ST14 => 14,
            OpStackElement::ST15 => 15,
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
            0 => Ok(OpStackElement::ST0),
            1 => Ok(OpStackElement::ST1),
            2 => Ok(OpStackElement::ST2),
            3 => Ok(OpStackElement::ST3),
            4 => Ok(OpStackElement::ST4),
            5 => Ok(OpStackElement::ST5),
            6 => Ok(OpStackElement::ST6),
            7 => Ok(OpStackElement::ST7),
            8 => Ok(OpStackElement::ST8),
            9 => Ok(OpStackElement::ST9),
            10 => Ok(OpStackElement::ST10),
            11 => Ok(OpStackElement::ST11),
            12 => Ok(OpStackElement::ST12),
            13 => Ok(OpStackElement::ST13),
            14 => Ok(OpStackElement::ST14),
            15 => Ok(OpStackElement::ST15),
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

/// Represents the argument, _i.e._, the `n`, for instructions like `pop n` or
/// `read_io n`.
#[derive(
    Debug,
    Default,
    Copy,
    Clone,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
    Serialize,
    Deserialize,
    EnumCount,
    EnumIter,
    GetSize,
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

#[non_exhaustive]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Error)]
pub enum NumberOfWordsError {
    #[error("index {0} is out of range for `NumberOfWords`")]
    IndexOutOfBounds(usize),

    #[error(transparent)]
    FailedIntegerConversion(#[from] TryFromIntError),
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

    pub fn legal_values() -> [usize; Self::COUNT] {
        let legal_indices = Self::iter().map(|n| n.num_words()).collect_vec();
        legal_indices.try_into().unwrap()
    }

    pub fn illegal_values() -> [usize; OpStackElement::COUNT - Self::COUNT] {
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

impl TryFrom<i32> for NumberOfWords {
    type Error = NumberOfWordsError;

    fn try_from(index: i32) -> NumWordsResult<Self> {
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use assert2::assert;
    use assert2::let_assert;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use strum::IntoEnumIterator;
    use test_strategy::proptest;

    use super::*;

    impl Default for OpStack {
        /// For testing purposes only.
        fn default() -> Self {
            OpStack::new(Digest::default())
        }
    }

    #[test]
    fn sanity() {
        let mut op_stack = OpStack::default();

        // verify height
        assert!(op_stack.len() == 16);
        assert!(op_stack.pointer().value() as usize == op_stack.len());

        // push elements 1 through 17
        for i in 1..=17 {
            op_stack.push(bfe!(i));
        }

        // verify height
        assert!(op_stack.len() == 33);
        assert!(op_stack.pointer().value() as usize == op_stack.len());

        let entire_stack = [
            op_stack[OpStackElement::ST0],
            op_stack[OpStackElement::ST1],
            op_stack[OpStackElement::ST2],
            op_stack[OpStackElement::ST3],
            op_stack[OpStackElement::ST4],
            op_stack[OpStackElement::ST5],
            op_stack[OpStackElement::ST6],
            op_stack[OpStackElement::ST7],
            op_stack[OpStackElement::ST8],
            op_stack[OpStackElement::ST9],
            op_stack[OpStackElement::ST10],
            op_stack[OpStackElement::ST11],
            op_stack[OpStackElement::ST12],
            op_stack[OpStackElement::ST13],
            op_stack[OpStackElement::ST14],
            op_stack[OpStackElement::ST15],
            op_stack.first_underflow_element(),
        ];
        assert!(entire_stack.into_iter().all_unique());

        // pop 11 elements
        for _ in 0..11 {
            let _ = op_stack.pop().unwrap();
        }

        // verify height
        assert!(op_stack.len() == 22);
        assert!(op_stack.pointer().value() as usize == op_stack.len());

        // pop 2 XFieldElements
        let _ = op_stack.pop_extension_field_element().unwrap();
        let _ = op_stack.pop_extension_field_element().unwrap();

        // verify height
        assert!(op_stack.len() == 16);
        assert!(op_stack.pointer().value() as usize == op_stack.len());

        // verify underflow
        assert!(op_stack.would_be_too_shallow(-1));
    }

    #[proptest]
    fn turning_op_stack_into_iterator_gives_top_element_first(
        #[strategy(arb())]
        #[filter(#op_stack.len() > 0)]
        op_stack: OpStack,
    ) {
        let top_element = op_stack[OpStackElement::ST0];
        let mut iterator = op_stack.into_iter();
        assert!(top_element == iterator.next().unwrap());
    }

    #[test]
    fn trying_to_access_first_underflow_element_never_panics() {
        let mut op_stack = OpStack::default();
        let way_too_long = 2 * op_stack.len();
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
            UnderflowIO::Read(bfe!(5)),
            UnderflowIO::Write(bfe!(5)),
            UnderflowIO::Read(bfe!(7)),
        ];
        UnderflowIO::canonicalize_sequence(&mut sequence);

        let expected_sequence = vec![UnderflowIO::Read(bfe!(7))];
        assert!(expected_sequence == sequence);
    }

    #[test]
    fn canonicalize_medium_complex_underflow_io_sequence() {
        let mut sequence = vec![
            UnderflowIO::Write(bfe!(5)),
            UnderflowIO::Write(bfe!(3)),
            UnderflowIO::Read(bfe!(3)),
            UnderflowIO::Read(bfe!(5)),
            UnderflowIO::Write(bfe!(7)),
        ];
        UnderflowIO::canonicalize_sequence(&mut sequence);

        let expected_sequence = vec![UnderflowIO::Write(bfe!(7))];
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

    #[test]
    fn empty_underflow_io_sequence_does_not_crash_uniformity_test() {
        assert!(UnderflowIO::is_uniform_sequence(&[]));
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
        assert!(let Ok(_) = OpStackElement::try_from(bfe!(0)));
    }

    #[test]
    fn convert_from_various_primitive_types_to_number_of_words() {
        assert!(let Ok(_) = NumberOfWords::try_from(1_u32));
        assert!(let Ok(_) = NumberOfWords::try_from(1_u64));
        assert!(let Ok(_) = NumberOfWords::try_from(1_usize));
        assert!(let Ok(_) = NumberOfWords::try_from(bfe!(1)));
        assert!(let Ok(_) = NumberOfWords::try_from(OpStackElement::ST1));
    }

    #[test]
    fn convert_from_op_stack_element_to_various_primitive_types() {
        let _ = u32::from(OpStackElement::ST0);
        let _ = u64::from(OpStackElement::ST0);
        let _ = usize::from(OpStackElement::ST0);
        let _ = i32::from(OpStackElement::ST0);
        let _ = BFieldElement::from(OpStackElement::ST0);
        let _ = bfe!(OpStackElement::ST0);

        let _ = u32::from(&OpStackElement::ST0);
        let _ = usize::from(&OpStackElement::ST0);
        let _ = i32::from(&OpStackElement::ST0);
        let _ = BFieldElement::from(&OpStackElement::ST0);
        let _ = bfe!(&OpStackElement::ST0);
    }

    #[test]
    fn convert_from_number_of_words_to_various_primitive_types() {
        let n1 = NumberOfWords::N1;

        let _ = u32::from(n1);
        let _ = u64::from(n1);
        let _ = usize::from(n1);
        let _ = BFieldElement::from(n1);
        let _ = OpStackElement::from(n1);
        let _ = bfe!(n1);

        let _ = u32::from(&n1);
        let _ = u64::from(&n1);
        let _ = usize::from(&n1);
        let _ = BFieldElement::from(&n1);
        let _ = OpStackElement::from(&n1);
        let _ = bfe!(&n1);
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
            .map(|num_words| bfe!(num_words).value())
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

    #[proptest]
    fn invalid_u32s_cannot_be_retrieved_as_u32s(
        #[strategy(arb())] st: OpStackElement,
        #[strategy(u64::from(u32::MAX) + 1..)] non_u32: u64,
    ) {
        let mut op_stack = OpStack::default();
        op_stack[st] = non_u32.into();

        assert!(let Err(_) = op_stack.is_u32(st));
        assert!(let Err(_) = op_stack.get_u32(st));
    }

    #[proptest]
    fn valid_u32s_can_be_retrieved_as_u32s(#[strategy(arb())] st: OpStackElement, valid_u32: u32) {
        let mut op_stack = OpStack::default();
        op_stack[st] = valid_u32.into();

        assert!(let Ok(()) = op_stack.is_u32(st));
        let_assert!(Ok(some_u32) = op_stack.get_u32(st));
        assert!(valid_u32 == some_u32);
    }

    #[proptest]
    fn inserting_an_element_into_the_stack_puts_it_at_the_correct_position(
        #[strategy(arb())] insertion_index: OpStackElement,
        #[strategy(arb())] insertion_element: BFieldElement,
    ) {
        let mut op_stack = OpStack::default();
        op_stack.insert(insertion_index, insertion_element);
        prop_assert_eq!(insertion_element, op_stack[insertion_index]);

        let expected_len = OpStackElement::COUNT + 1;
        prop_assert_eq!(expected_len, op_stack.len());
    }

    #[proptest]
    fn removing_an_element_from_the_stack_removes_the_correct_element(
        #[strategy(arb())] removal_index: OpStackElement,
    ) {
        let mut op_stack = OpStack::default();
        for i in (0..OpStackElement::COUNT as u64).rev() {
            op_stack.push(bfe!(i));
        }

        let expected_element = BFieldElement::from(removal_index);
        let removed_element = op_stack.remove(removal_index);
        prop_assert_eq!(expected_element, removed_element);

        let expected_len = 2 * OpStackElement::COUNT - 1;
        prop_assert_eq!(expected_len, op_stack.len());
    }
}
