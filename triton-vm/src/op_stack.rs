use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;

use anyhow::anyhow;
use anyhow::bail;
use anyhow::Result;
use arbitrary::Arbitrary;
use get_size::GetSize;
use itertools::Itertools;
use num_traits::Zero;
use serde_derive::Deserialize;
use serde_derive::Serialize;
use strum::EnumCount;
use strum::EnumIter;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::digest::Digest;
use twenty_first::shared_math::tip5::DIGEST_LENGTH;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::error::InstructionError::*;
use crate::op_stack::OpStackElement::*;

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
#[derive(Debug, Clone)]
pub struct OpStack {
    pub stack: Vec<BFieldElement>,
    underflow_io_sequence: Vec<UnderflowIO>,
}

impl OpStack {
    pub fn new(program_digest: Digest) -> Self {
        let mut stack = vec![BFieldElement::zero(); OpStackElement::COUNT];

        let reverse_digest = program_digest.reversed().values();
        stack[..DIGEST_LENGTH].copy_from_slice(&reverse_digest);

        Self {
            stack,
            underflow_io_sequence: vec![],
        }
    }

    pub(crate) fn push(&mut self, element: BFieldElement) {
        self.stack.push(element);
        self.record_underflow_io(UnderflowIO::Write);
    }

    pub(crate) fn pop(&mut self) -> Result<BFieldElement> {
        self.record_underflow_io(UnderflowIO::Read);
        let element = self.stack.pop().ok_or_else(|| anyhow!(OpStackTooShallow))?;
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

    pub(crate) fn pop_u32(&mut self) -> Result<u32> {
        let element = self.pop()?;
        let element = element
            .try_into()
            .map_err(|_| anyhow!(FailedU32Conversion(element)))?;
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

    pub(crate) fn swap_top_with(&mut self, stack_element: OpStackElement) {
        let stack_element_index = usize::from(stack_element);
        let top_of_stack_index = self.stack.len() - 1;
        self.stack
            .swap(top_of_stack_index, top_of_stack_index - stack_element_index);
    }

    pub(crate) fn is_too_shallow(&self) -> bool {
        self.stack.len() < OpStackElement::COUNT
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
    type Error = anyhow::Error;

    fn try_from(stack_index: u32) -> Result<Self> {
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
            _ => bail!("Index {stack_index} is out of range for `OpStackElement`."),
        }
    }
}

impl From<OpStackElement> for u64 {
    fn from(stack_element: OpStackElement) -> Self {
        u32::from(stack_element).into()
    }
}

impl TryFrom<u64> for OpStackElement {
    type Error = anyhow::Error;

    fn try_from(stack_index: u64) -> Result<Self> {
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
    type Error = anyhow::Error;

    fn try_from(stack_index: i32) -> Result<Self> {
        u32::try_from(stack_index)?.try_into()
    }
}

impl TryFrom<usize> for OpStackElement {
    type Error = anyhow::Error;

    fn try_from(stack_index: usize) -> Result<Self> {
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
)]
pub enum StackChangeArg {
    #[default]
    N1,
    N2,
    N3,
    N4,
    N5,
}

impl StackChangeArg {
    pub const fn index(self) -> usize {
        match self {
            Self::N1 => 1,
            Self::N2 => 2,
            Self::N3 => 3,
            Self::N4 => 4,
            Self::N5 => 5,
        }
    }
}

impl Display for StackChangeArg {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let index = self.index();
        write!(f, "{index}")
    }
}

impl From<StackChangeArg> for usize {
    fn from(stack_changing_argument: StackChangeArg) -> Self {
        stack_changing_argument.index()
    }
}

impl From<&StackChangeArg> for usize {
    fn from(&stack_changing_argument: &StackChangeArg) -> Self {
        stack_changing_argument.into()
    }
}

impl From<StackChangeArg> for u32 {
    fn from(stack_changing_argument: StackChangeArg) -> Self {
        stack_changing_argument.index() as u32
    }
}

impl From<&StackChangeArg> for u32 {
    fn from(&stack_changing_argument: &StackChangeArg) -> Self {
        stack_changing_argument.into()
    }
}

impl From<StackChangeArg> for u64 {
    fn from(stack_changing_argument: StackChangeArg) -> Self {
        u32::from(stack_changing_argument).into()
    }
}

impl From<&StackChangeArg> for u64 {
    fn from(&stack_changing_argument: &StackChangeArg) -> Self {
        stack_changing_argument.into()
    }
}

impl From<StackChangeArg> for OpStackElement {
    fn from(stack_changing_argument: StackChangeArg) -> Self {
        OpStackElement::try_from(stack_changing_argument.index()).unwrap()
    }
}

impl From<&StackChangeArg> for OpStackElement {
    fn from(&stack_changing_argument: &StackChangeArg) -> Self {
        stack_changing_argument.into()
    }
}

impl From<StackChangeArg> for BFieldElement {
    fn from(stack_changing_argument: StackChangeArg) -> Self {
        u32::from(stack_changing_argument).into()
    }
}

impl From<&StackChangeArg> for BFieldElement {
    fn from(&stack_changing_argument: &StackChangeArg) -> Self {
        stack_changing_argument.into()
    }
}

impl TryFrom<usize> for StackChangeArg {
    type Error = anyhow::Error;

    fn try_from(stack_changing_argument: usize) -> Result<Self> {
        match stack_changing_argument {
            1 => Ok(Self::N1),
            2 => Ok(Self::N2),
            3 => Ok(Self::N3),
            4 => Ok(Self::N4),
            5 => Ok(Self::N5),
            _ => bail!("Index {stack_changing_argument} is out of range for `StackChangeArg`."),
        }
    }
}

impl TryFrom<u32> for StackChangeArg {
    type Error = anyhow::Error;

    fn try_from(stack_changing_argument: u32) -> Result<Self> {
        usize::try_from(stack_changing_argument)?.try_into()
    }
}

impl TryFrom<OpStackElement> for StackChangeArg {
    type Error = anyhow::Error;

    fn try_from(stack_changing_argument: OpStackElement) -> Result<Self> {
        usize::try_from(stack_changing_argument)?.try_into()
    }
}

impl TryFrom<u64> for StackChangeArg {
    type Error = anyhow::Error;

    fn try_from(stack_changing_argument: u64) -> Result<Self> {
        usize::try_from(stack_changing_argument)?.try_into()
    }
}

impl TryFrom<BFieldElement> for StackChangeArg {
    type Error = anyhow::Error;

    fn try_from(stack_changing_argument: BFieldElement) -> Result<Self> {
        u32::try_from(stack_changing_argument)?.try_into()
    }
}

impl TryFrom<&BFieldElement> for StackChangeArg {
    type Error = anyhow::Error;

    fn try_from(&stack_changing_argument: &BFieldElement) -> Result<Self> {
        stack_changing_argument.try_into()
    }
}

#[cfg(test)]
mod tests {
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use strum::IntoEnumIterator;
    use test_strategy::proptest;
    use twenty_first::shared_math::b_field_element::BFieldElement;

    use super::*;

    #[test]
    fn sanity() {
        let digest = Default::default();
        let mut op_stack = OpStack::new(digest);

        // verify height
        assert_eq!(op_stack.stack.len(), 16);
        assert_eq!(op_stack.pointer().value() as usize, op_stack.stack.len());

        // push elements 1 thru 17
        for i in 1..=17 {
            op_stack.push(BFieldElement::new(i as u64));
        }

        // verify height
        assert_eq!(op_stack.stack.len(), 33);
        assert_eq!(op_stack.pointer().value() as usize, op_stack.stack.len());

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
        assert_eq!(len_before, len_after);

        // pop 11 elements
        for _ in 0..11 {
            let _ = op_stack.pop().expect("can't pop");
        }

        // verify height
        assert_eq!(op_stack.stack.len(), 22);
        assert_eq!(op_stack.pointer().value() as usize, op_stack.stack.len());

        // pop 2 XFieldElements
        let _ = op_stack.pop_extension_field_element().expect("can't pop");
        let _ = op_stack.pop_extension_field_element().expect("can't pop");

        // verify height
        assert_eq!(op_stack.stack.len(), 16);
        assert_eq!(op_stack.pointer().value() as usize, op_stack.stack.len());

        // verify underflow
        let _ = op_stack.pop().expect("can't pop");
        assert!(op_stack.is_too_shallow());
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
            let stack_index = u32::from(&stack_element);
            let stack_element_again = OpStackElement::try_from(stack_index).unwrap();
            assert_eq!(stack_element, stack_element_again);
        }
    }

    #[test]
    fn conversion_from_stack_element_to_i32_and_back_is_identity() {
        for stack_element in OpStackElement::iter() {
            let stack_index = i32::from(&stack_element);
            let stack_element_again = OpStackElement::try_from(stack_index).unwrap();
            assert_eq!(stack_element, stack_element_again);
        }
    }

    #[test]
    fn canonicalize_empty_underflow_io_sequence() {
        let mut sequence = vec![];
        UnderflowIO::canonicalize_sequence(&mut sequence);

        let expected_sequence = Vec::<UnderflowIO>::new();
        assert_eq!(expected_sequence, sequence);
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
        assert_eq!(expected_sequence, sequence);
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
        assert_eq!(expected_sequence, sequence);
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
    fn conversion_from_stack_changing_argument_to_usize_and_back_is_identity() {
        for stack_changing_argument in StackChangeArg::iter() {
            let stack_index = usize::from(&stack_changing_argument);
            let stack_changing_argument_again = StackChangeArg::try_from(stack_index).unwrap();
            assert_eq!(stack_changing_argument, stack_changing_argument_again);
        }
    }

    #[test]
    fn conversion_from_stack_changing_argument_to_u64_and_back_is_identity() {
        for stack_changing_argument in StackChangeArg::iter() {
            let stack_index = u64::from(&stack_changing_argument);
            let stack_changing_argument_again = StackChangeArg::try_from(stack_index).unwrap();
            assert_eq!(stack_changing_argument, stack_changing_argument_again);
        }
    }

    #[test]
    fn conversion_from_stack_changing_argument_to_op_stack_element_and_back_is_identity() {
        for stack_changing_argument in StackChangeArg::iter() {
            let stack_element = OpStackElement::from(&stack_changing_argument);
            let stack_changing_argument_again = StackChangeArg::try_from(stack_element).unwrap();
            assert_eq!(stack_changing_argument, stack_changing_argument_again);
        }
    }

    #[test]
    fn out_of_range_stack_changing_argument_gives_error() {
        let stack_changing_argument = StackChangeArg::iter().last().unwrap();
        let mut stack_index = BFieldElement::from(&stack_changing_argument);
        stack_index.increment();
        let maybe_stack_changing_argument = StackChangeArg::try_from(&stack_index);
        assert!(maybe_stack_changing_argument.is_err());
    }

    #[test]
    fn stack_change_arg_to_b_field_element_gives_expected_range() {
        let computed_range = StackChangeArg::iter()
            .map(|stack_change_arg| BFieldElement::from(&stack_change_arg).value())
            .collect_vec();
        let expected_range = (1..=5).collect_vec();
        assert_eq!(computed_range, expected_range);
    }
}
