use std::collections::HashMap;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::num::TryFromIntError;
use std::result;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use get_size2::GetSize;
use itertools::Itertools;
use lazy_static::lazy_static;
use num_traits::ConstZero;
use serde::Deserialize;
use serde::Serialize;
use strum::EnumCount;
use strum::EnumIter;
use strum::IntoEnumIterator;
use thiserror::Error;
use twenty_first::prelude::*;

use crate::op_stack::NumberOfWords;
use crate::op_stack::OpStackElement;
use crate::op_stack::OpStackError;

type Result<T> = result::Result<T, InstructionError>;

/// An `Instruction` has `call` addresses encoded as absolute integers.
pub type Instruction = AnInstruction<BFieldElement>;

pub const ALL_INSTRUCTIONS: [Instruction; Instruction::COUNT] = [
    Instruction::Pop(NumberOfWords::N1),
    Instruction::Push(BFieldElement::ZERO),
    Instruction::Divine(NumberOfWords::N1),
    Instruction::Pick(OpStackElement::ST0),
    Instruction::Place(OpStackElement::ST0),
    Instruction::Dup(OpStackElement::ST0),
    Instruction::Swap(OpStackElement::ST0),
    Instruction::Halt,
    Instruction::Nop,
    Instruction::Skiz,
    Instruction::Call(BFieldElement::ZERO),
    Instruction::Return,
    Instruction::Recurse,
    Instruction::RecurseOrReturn,
    Instruction::Assert,
    Instruction::ReadMem(NumberOfWords::N1),
    Instruction::WriteMem(NumberOfWords::N1),
    Instruction::Hash,
    Instruction::AssertVector,
    Instruction::SpongeInit,
    Instruction::SpongeAbsorb,
    Instruction::SpongeAbsorbMem,
    Instruction::SpongeSqueeze,
    Instruction::Add,
    Instruction::AddI(BFieldElement::ZERO),
    Instruction::Mul,
    Instruction::Invert,
    Instruction::Eq,
    Instruction::Split,
    Instruction::Lt,
    Instruction::And,
    Instruction::Xor,
    Instruction::Log2Floor,
    Instruction::Pow,
    Instruction::DivMod,
    Instruction::PopCount,
    Instruction::XxAdd,
    Instruction::XxMul,
    Instruction::XInvert,
    Instruction::XbMul,
    Instruction::ReadIo(NumberOfWords::N1),
    Instruction::WriteIo(NumberOfWords::N1),
    Instruction::MerkleStep,
    Instruction::MerkleStepMem,
    Instruction::XxDotStep,
    Instruction::XbDotStep,
];

pub const ALL_INSTRUCTION_NAMES: [&str; Instruction::COUNT] = {
    let mut names = [""; Instruction::COUNT];
    let mut i = 0;
    while i < Instruction::COUNT {
        names[i] = ALL_INSTRUCTIONS[i].name();
        i += 1;
    }
    names
};

lazy_static! {
    pub static ref OPCODE_TO_INSTRUCTION_MAP: HashMap<u32, Instruction> = {
        let mut opcode_to_instruction_map = HashMap::new();
        for instruction in Instruction::iter() {
            opcode_to_instruction_map.insert(instruction.opcode(), instruction);
        }
        opcode_to_instruction_map
    };
}

/// A `LabelledInstruction` has `call` addresses encoded as label names.
#[derive(Debug, Clone, Eq, PartialEq, Hash, EnumCount)]
pub enum LabelledInstruction {
    /// An instructions from the [instruction set architecture][isa].
    ///
    /// [isa]: https://triton-vm.org/spec/isa.html
    Instruction(AnInstruction<String>),

    /// Labels look like "`<name>:`" and are translated into absolute addresses.
    Label(String),

    Breakpoint,

    TypeHint(TypeHint),

    AssertionContext(AssertionContext),
}

/// A hint about a range of stack elements. Helps debugging programs written for
/// Triton VM. **Does not enforce types.**
///
/// Usually constructed by parsing special annotations in the assembly code, for
/// example:
///
/// ```tasm
/// hint variable_name: the_type = stack[0]
/// hint my_list = stack[1..4]
/// ```
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, GetSize)]
pub struct TypeHint {
    pub starting_index: usize,
    pub length: usize,

    /// The name of the type, _e.g._, `u32`, `list`, `Digest`, et cetera.
    pub type_name: Option<String>,

    /// The name of the variable.
    pub variable_name: String,
}

/// Context to help debugging failing instructions
/// [`assert`](Instruction::Assert) or
/// [`assert_vector`](Instruction::AssertVector).
#[non_exhaustive]
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, GetSize, Arbitrary)]
pub enum AssertionContext {
    ID(i128),
    // Message(String),
}

impl LabelledInstruction {
    pub const fn op_stack_size_influence(&self) -> i32 {
        match self {
            LabelledInstruction::Instruction(instruction) => instruction.op_stack_size_influence(),
            _ => 0,
        }
    }
}

impl Display for LabelledInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            LabelledInstruction::Instruction(instruction) => write!(f, "{instruction}"),
            LabelledInstruction::Label(label) => write!(f, "{label}:"),
            LabelledInstruction::Breakpoint => write!(f, "break"),
            LabelledInstruction::TypeHint(type_hint) => write!(f, "{type_hint}"),
            LabelledInstruction::AssertionContext(ctx) => write!(f, "{ctx}"),
        }
    }
}

impl Display for TypeHint {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let variable = &self.variable_name;

        let format_type = |t| format!(": {t}");
        let maybe_type = self.type_name.as_ref();
        let type_name = maybe_type.map(format_type).unwrap_or_default();

        let start = self.starting_index;
        let range = match self.length {
            1 => format!("{start}"),
            _ => format!("{start}..{end}", end = start + self.length),
        };

        write!(f, "hint {variable}{type_name} = stack[{range}]")
    }
}

impl Display for AssertionContext {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let Self::ID(id) = self;
        write!(f, "error_id {id}")
    }
}

impl<'a> Arbitrary<'a> for TypeHint {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let starting_index = u.arbitrary()?;
        let length = u.int_in_range(1..=500)?;
        let type_name = if u.arbitrary()? {
            Some(u.arbitrary::<TypeHintTypeName>()?.into())
        } else {
            None
        };
        let variable_name = u.arbitrary::<TypeHintVariableName>()?.into();

        let type_hint = Self {
            starting_index,
            length,
            type_name,
            variable_name,
        };
        Ok(type_hint)
    }
}

/// A Triton VM instruction. See the
/// [Instruction Set Architecture](https://triton-vm.org/spec/isa.html)
/// for more details.
///
/// The type parameter `Dest` describes the type of addresses (absolute or
/// labels).
#[derive(
    Debug,
    Copy,
    Clone,
    Eq,
    PartialEq,
    Hash,
    EnumCount,
    EnumIter,
    Serialize,
    Deserialize,
    GetSize,
    Arbitrary,
)]
pub enum AnInstruction<Dest: PartialEq + Default> {
    // OpStack manipulation
    Push(BFieldElement),
    Pop(NumberOfWords),
    Divine(NumberOfWords),
    Pick(OpStackElement),
    Place(OpStackElement),
    Dup(OpStackElement),
    Swap(OpStackElement),

    // Control flow
    Halt,
    Nop,
    Skiz,
    Call(Dest),
    Return,
    Recurse,
    RecurseOrReturn,
    Assert,

    // Memory access
    ReadMem(NumberOfWords),
    WriteMem(NumberOfWords),

    // Hashing-related
    Hash,
    AssertVector,
    SpongeInit,
    SpongeAbsorb,
    SpongeAbsorbMem,
    SpongeSqueeze,

    // Base field arithmetic on stack
    Add,
    AddI(BFieldElement),
    Mul,
    Invert,
    Eq,

    // Bitwise arithmetic on stack
    Split,
    Lt,
    And,
    Xor,
    Log2Floor,
    Pow,
    DivMod,
    PopCount,

    // Extension field arithmetic on stack
    XxAdd,
    XxMul,
    XInvert,
    XbMul,

    // Read/write
    ReadIo(NumberOfWords),
    WriteIo(NumberOfWords),

    // Many-in-One
    MerkleStep,
    MerkleStepMem,
    XxDotStep,
    XbDotStep,
}

impl<Dest: PartialEq + Default> AnInstruction<Dest> {
    /// Assign a unique positive integer to each `Instruction`.
    pub const fn opcode(&self) -> u32 {
        match self {
            AnInstruction::Pop(_) => 3,
            AnInstruction::Push(_) => 1,
            AnInstruction::Divine(_) => 9,
            AnInstruction::Pick(_) => 17,
            AnInstruction::Place(_) => 25,
            AnInstruction::Dup(_) => 33,
            AnInstruction::Swap(_) => 41,
            AnInstruction::Halt => 0,
            AnInstruction::Nop => 8,
            AnInstruction::Skiz => 2,
            AnInstruction::Call(_) => 49,
            AnInstruction::Return => 16,
            AnInstruction::Recurse => 24,
            AnInstruction::RecurseOrReturn => 32,
            AnInstruction::Assert => 10,
            AnInstruction::ReadMem(_) => 57,
            AnInstruction::WriteMem(_) => 11,
            AnInstruction::Hash => 18,
            AnInstruction::AssertVector => 26,
            AnInstruction::SpongeInit => 40,
            AnInstruction::SpongeAbsorb => 34,
            AnInstruction::SpongeAbsorbMem => 48,
            AnInstruction::SpongeSqueeze => 56,
            AnInstruction::Add => 42,
            AnInstruction::AddI(_) => 65,
            AnInstruction::Mul => 50,
            AnInstruction::Invert => 64,
            AnInstruction::Eq => 58,
            AnInstruction::Split => 4,
            AnInstruction::Lt => 6,
            AnInstruction::And => 14,
            AnInstruction::Xor => 22,
            AnInstruction::Log2Floor => 12,
            AnInstruction::Pow => 30,
            AnInstruction::DivMod => 20,
            AnInstruction::PopCount => 28,
            AnInstruction::XxAdd => 66,
            AnInstruction::XxMul => 74,
            AnInstruction::XInvert => 72,
            AnInstruction::XbMul => 82,
            AnInstruction::ReadIo(_) => 73,
            AnInstruction::WriteIo(_) => 19,
            AnInstruction::MerkleStep => 36,
            AnInstruction::MerkleStepMem => 44,
            AnInstruction::XxDotStep => 80,
            AnInstruction::XbDotStep => 88,
        }
    }

    pub const fn name(&self) -> &'static str {
        match self {
            AnInstruction::Pop(_) => "pop",
            AnInstruction::Push(_) => "push",
            AnInstruction::Divine(_) => "divine",
            AnInstruction::Pick(_) => "pick",
            AnInstruction::Place(_) => "place",
            AnInstruction::Dup(_) => "dup",
            AnInstruction::Swap(_) => "swap",
            AnInstruction::Halt => "halt",
            AnInstruction::Nop => "nop",
            AnInstruction::Skiz => "skiz",
            AnInstruction::Call(_) => "call",
            AnInstruction::Return => "return",
            AnInstruction::Recurse => "recurse",
            AnInstruction::RecurseOrReturn => "recurse_or_return",
            AnInstruction::Assert => "assert",
            AnInstruction::ReadMem(_) => "read_mem",
            AnInstruction::WriteMem(_) => "write_mem",
            AnInstruction::Hash => "hash",
            AnInstruction::AssertVector => "assert_vector",
            AnInstruction::SpongeInit => "sponge_init",
            AnInstruction::SpongeAbsorb => "sponge_absorb",
            AnInstruction::SpongeAbsorbMem => "sponge_absorb_mem",
            AnInstruction::SpongeSqueeze => "sponge_squeeze",
            AnInstruction::Add => "add",
            AnInstruction::AddI(_) => "addi",
            AnInstruction::Mul => "mul",
            AnInstruction::Invert => "invert",
            AnInstruction::Eq => "eq",
            AnInstruction::Split => "split",
            AnInstruction::Lt => "lt",
            AnInstruction::And => "and",
            AnInstruction::Xor => "xor",
            AnInstruction::Log2Floor => "log_2_floor",
            AnInstruction::Pow => "pow",
            AnInstruction::DivMod => "div_mod",
            AnInstruction::PopCount => "pop_count",
            AnInstruction::XxAdd => "xx_add",
            AnInstruction::XxMul => "xx_mul",
            AnInstruction::XInvert => "x_invert",
            AnInstruction::XbMul => "xb_mul",
            AnInstruction::ReadIo(_) => "read_io",
            AnInstruction::WriteIo(_) => "write_io",
            AnInstruction::MerkleStep => "merkle_step",
            AnInstruction::MerkleStepMem => "merkle_step_mem",
            AnInstruction::XxDotStep => "xx_dot_step",
            AnInstruction::XbDotStep => "xb_dot_step",
        }
    }

    pub const fn opcode_b(&self) -> BFieldElement {
        BFieldElement::new(self.opcode() as u64)
    }

    /// Number of words required to represent the instruction.
    pub fn size(&self) -> usize {
        match self {
            AnInstruction::Pop(_) | AnInstruction::Push(_) => 2,
            AnInstruction::Divine(_) => 2,
            AnInstruction::Pick(_) | AnInstruction::Place(_) => 2,
            AnInstruction::Dup(_) | AnInstruction::Swap(_) => 2,
            AnInstruction::Call(_) => 2,
            AnInstruction::ReadMem(_) | AnInstruction::WriteMem(_) => 2,
            AnInstruction::AddI(_) => 2,
            AnInstruction::ReadIo(_) | AnInstruction::WriteIo(_) => 2,
            _ => 1,
        }
    }

    /// Get the i'th instruction bit
    pub fn ib(&self, arg: InstructionBit) -> BFieldElement {
        bfe!((self.opcode() >> usize::from(arg)) & 1)
    }

    pub fn map_call_address<F, NewDest>(&self, f: F) -> AnInstruction<NewDest>
    where
        F: FnOnce(&Dest) -> NewDest,
        NewDest: PartialEq + Default,
    {
        match self {
            AnInstruction::Pop(x) => AnInstruction::Pop(*x),
            AnInstruction::Push(x) => AnInstruction::Push(*x),
            AnInstruction::Divine(x) => AnInstruction::Divine(*x),
            AnInstruction::Pick(x) => AnInstruction::Pick(*x),
            AnInstruction::Place(x) => AnInstruction::Place(*x),
            AnInstruction::Dup(x) => AnInstruction::Dup(*x),
            AnInstruction::Swap(x) => AnInstruction::Swap(*x),
            AnInstruction::Halt => AnInstruction::Halt,
            AnInstruction::Nop => AnInstruction::Nop,
            AnInstruction::Skiz => AnInstruction::Skiz,
            AnInstruction::Call(label) => AnInstruction::Call(f(label)),
            AnInstruction::Return => AnInstruction::Return,
            AnInstruction::Recurse => AnInstruction::Recurse,
            AnInstruction::RecurseOrReturn => AnInstruction::RecurseOrReturn,
            AnInstruction::Assert => AnInstruction::Assert,
            AnInstruction::ReadMem(x) => AnInstruction::ReadMem(*x),
            AnInstruction::WriteMem(x) => AnInstruction::WriteMem(*x),
            AnInstruction::Hash => AnInstruction::Hash,
            AnInstruction::AssertVector => AnInstruction::AssertVector,
            AnInstruction::SpongeInit => AnInstruction::SpongeInit,
            AnInstruction::SpongeAbsorb => AnInstruction::SpongeAbsorb,
            AnInstruction::SpongeAbsorbMem => AnInstruction::SpongeAbsorbMem,
            AnInstruction::SpongeSqueeze => AnInstruction::SpongeSqueeze,
            AnInstruction::Add => AnInstruction::Add,
            AnInstruction::AddI(x) => AnInstruction::AddI(*x),
            AnInstruction::Mul => AnInstruction::Mul,
            AnInstruction::Invert => AnInstruction::Invert,
            AnInstruction::Eq => AnInstruction::Eq,
            AnInstruction::Split => AnInstruction::Split,
            AnInstruction::Lt => AnInstruction::Lt,
            AnInstruction::And => AnInstruction::And,
            AnInstruction::Xor => AnInstruction::Xor,
            AnInstruction::Log2Floor => AnInstruction::Log2Floor,
            AnInstruction::Pow => AnInstruction::Pow,
            AnInstruction::DivMod => AnInstruction::DivMod,
            AnInstruction::PopCount => AnInstruction::PopCount,
            AnInstruction::XxAdd => AnInstruction::XxAdd,
            AnInstruction::XxMul => AnInstruction::XxMul,
            AnInstruction::XInvert => AnInstruction::XInvert,
            AnInstruction::XbMul => AnInstruction::XbMul,
            AnInstruction::ReadIo(x) => AnInstruction::ReadIo(*x),
            AnInstruction::WriteIo(x) => AnInstruction::WriteIo(*x),
            AnInstruction::MerkleStep => AnInstruction::MerkleStep,
            AnInstruction::MerkleStepMem => AnInstruction::MerkleStepMem,
            AnInstruction::XxDotStep => AnInstruction::XxDotStep,
            AnInstruction::XbDotStep => AnInstruction::XbDotStep,
        }
    }

    pub const fn op_stack_size_influence(&self) -> i32 {
        match self {
            AnInstruction::Pop(n) => -(n.num_words() as i32),
            AnInstruction::Push(_) => 1,
            AnInstruction::Divine(n) => n.num_words() as i32,
            AnInstruction::Pick(_) => 0,
            AnInstruction::Place(_) => 0,
            AnInstruction::Dup(_) => 1,
            AnInstruction::Swap(_) => 0,
            AnInstruction::Halt => 0,
            AnInstruction::Nop => 0,
            AnInstruction::Skiz => -1,
            AnInstruction::Call(_) => 0,
            AnInstruction::Return => 0,
            AnInstruction::Recurse => 0,
            AnInstruction::RecurseOrReturn => 0,
            AnInstruction::Assert => -1,
            AnInstruction::ReadMem(n) => n.num_words() as i32,
            AnInstruction::WriteMem(n) => -(n.num_words() as i32),
            AnInstruction::Hash => -5,
            AnInstruction::AssertVector => -5,
            AnInstruction::SpongeInit => 0,
            AnInstruction::SpongeAbsorb => -10,
            AnInstruction::SpongeAbsorbMem => 0,
            AnInstruction::SpongeSqueeze => 10,
            AnInstruction::Add => -1,
            AnInstruction::AddI(_) => 0,
            AnInstruction::Mul => -1,
            AnInstruction::Invert => 0,
            AnInstruction::Eq => -1,
            AnInstruction::Split => 1,
            AnInstruction::Lt => -1,
            AnInstruction::And => -1,
            AnInstruction::Xor => -1,
            AnInstruction::Log2Floor => 0,
            AnInstruction::Pow => -1,
            AnInstruction::DivMod => 0,
            AnInstruction::PopCount => 0,
            AnInstruction::XxAdd => -3,
            AnInstruction::XxMul => -3,
            AnInstruction::XInvert => 0,
            AnInstruction::XbMul => -1,
            AnInstruction::ReadIo(n) => n.num_words() as i32,
            AnInstruction::WriteIo(n) => -(n.num_words() as i32),
            AnInstruction::MerkleStep => 0,
            AnInstruction::MerkleStepMem => 0,
            AnInstruction::XxDotStep => 0,
            AnInstruction::XbDotStep => 0,
        }
    }

    /// Indicates whether the instruction operates on base field elements that
    /// are also u32s.
    pub fn is_u32_instruction(&self) -> bool {
        matches!(
            self,
            AnInstruction::Split
                | AnInstruction::Lt
                | AnInstruction::And
                | AnInstruction::Xor
                | AnInstruction::Log2Floor
                | AnInstruction::Pow
                | AnInstruction::DivMod
                | AnInstruction::PopCount
                | AnInstruction::MerkleStep
                | AnInstruction::MerkleStepMem
        )
    }
}

impl<Dest: Display + PartialEq + Default> Display for AnInstruction<Dest> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.name())?;
        match self {
            AnInstruction::Push(arg) => write!(f, " {arg}"),
            AnInstruction::Pick(arg) => write!(f, " {arg}"),
            AnInstruction::Place(arg) => write!(f, " {arg}"),
            AnInstruction::Pop(arg) => write!(f, " {arg}"),
            AnInstruction::Divine(arg) => write!(f, " {arg}"),
            AnInstruction::Dup(arg) => write!(f, " {arg}"),
            AnInstruction::Swap(arg) => write!(f, " {arg}"),
            AnInstruction::Call(arg) => write!(f, " {arg}"),
            AnInstruction::ReadMem(arg) => write!(f, " {arg}"),
            AnInstruction::WriteMem(arg) => write!(f, " {arg}"),
            AnInstruction::AddI(arg) => write!(f, " {arg}"),
            AnInstruction::ReadIo(arg) => write!(f, " {arg}"),
            AnInstruction::WriteIo(arg) => write!(f, " {arg}"),
            _ => Ok(()),
        }
    }
}

impl Instruction {
    /// Get the argument of the instruction, if it has one.
    pub fn arg(&self) -> Option<BFieldElement> {
        match self {
            AnInstruction::Push(arg) => Some(*arg),
            AnInstruction::Call(arg) => Some(*arg),
            AnInstruction::Pop(arg) => Some(arg.into()),
            AnInstruction::Divine(arg) => Some(arg.into()),
            AnInstruction::Pick(arg) => Some(arg.into()),
            AnInstruction::Place(arg) => Some(arg.into()),
            AnInstruction::Dup(arg) => Some(arg.into()),
            AnInstruction::Swap(arg) => Some(arg.into()),
            AnInstruction::ReadMem(arg) => Some(arg.into()),
            AnInstruction::WriteMem(arg) => Some(arg.into()),
            AnInstruction::AddI(arg) => Some(*arg),
            AnInstruction::ReadIo(arg) => Some(arg.into()),
            AnInstruction::WriteIo(arg) => Some(arg.into()),
            _ => None,
        }
    }

    /// Change the argument of the instruction, if it has one. Returns an `Err`
    /// if the instruction does not have an argument or if the argument is
    /// out of range.
    pub fn change_arg(self, new_arg: BFieldElement) -> Result<Self> {
        let illegal_argument_error = || InstructionError::IllegalArgument(self, new_arg);
        let num_words = new_arg.try_into().map_err(|_| illegal_argument_error());
        let op_stack_element = new_arg.try_into().map_err(|_| illegal_argument_error());

        let new_instruction = match self {
            AnInstruction::Pop(_) => AnInstruction::Pop(num_words?),
            AnInstruction::Push(_) => AnInstruction::Push(new_arg),
            AnInstruction::Divine(_) => AnInstruction::Divine(num_words?),
            AnInstruction::Pick(_) => AnInstruction::Pick(op_stack_element?),
            AnInstruction::Place(_) => AnInstruction::Place(op_stack_element?),
            AnInstruction::Dup(_) => AnInstruction::Dup(op_stack_element?),
            AnInstruction::Swap(_) => AnInstruction::Swap(op_stack_element?),
            AnInstruction::Call(_) => AnInstruction::Call(new_arg),
            AnInstruction::ReadMem(_) => AnInstruction::ReadMem(num_words?),
            AnInstruction::WriteMem(_) => AnInstruction::WriteMem(num_words?),
            AnInstruction::AddI(_) => AnInstruction::AddI(new_arg),
            AnInstruction::ReadIo(_) => AnInstruction::ReadIo(num_words?),
            AnInstruction::WriteIo(_) => AnInstruction::WriteIo(num_words?),
            _ => return Err(illegal_argument_error()),
        };

        Ok(new_instruction)
    }
}

impl TryFrom<u32> for Instruction {
    type Error = InstructionError;

    fn try_from(opcode: u32) -> Result<Self> {
        OPCODE_TO_INSTRUCTION_MAP
            .get(&opcode)
            .copied()
            .ok_or(InstructionError::InvalidOpcode(opcode))
    }
}

impl TryFrom<u64> for Instruction {
    type Error = InstructionError;

    fn try_from(opcode: u64) -> Result<Self> {
        let opcode = u32::try_from(opcode)?;
        opcode.try_into()
    }
}

impl TryFrom<usize> for Instruction {
    type Error = InstructionError;

    fn try_from(opcode: usize) -> Result<Self> {
        let opcode = u32::try_from(opcode)?;
        opcode.try_into()
    }
}

impl TryFrom<BFieldElement> for Instruction {
    type Error = InstructionError;

    fn try_from(opcode: BFieldElement) -> Result<Self> {
        let opcode = u32::try_from(opcode)?;
        opcode.try_into()
    }
}

/// Indicators for all the possible bits in an [`Instruction`].
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, EnumCount, EnumIter)]
pub enum InstructionBit {
    #[default]
    IB0,
    IB1,
    IB2,
    IB3,
    IB4,
    IB5,
    IB6,
}

impl Display for InstructionBit {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let bit_index = usize::from(*self);
        write!(f, "{bit_index}")
    }
}

impl From<InstructionBit> for usize {
    fn from(instruction_bit: InstructionBit) -> Self {
        match instruction_bit {
            InstructionBit::IB0 => 0,
            InstructionBit::IB1 => 1,
            InstructionBit::IB2 => 2,
            InstructionBit::IB3 => 3,
            InstructionBit::IB4 => 4,
            InstructionBit::IB5 => 5,
            InstructionBit::IB6 => 6,
        }
    }
}

impl TryFrom<usize> for InstructionBit {
    type Error = String;

    fn try_from(bit_index: usize) -> result::Result<Self, Self::Error> {
        match bit_index {
            0 => Ok(InstructionBit::IB0),
            1 => Ok(InstructionBit::IB1),
            2 => Ok(InstructionBit::IB2),
            3 => Ok(InstructionBit::IB3),
            4 => Ok(InstructionBit::IB4),
            5 => Ok(InstructionBit::IB5),
            6 => Ok(InstructionBit::IB6),
            _ => Err(format!(
                "Index {bit_index} is out of range for `InstructionBit`."
            )),
        }
    }
}

impl<'a> Arbitrary<'a> for LabelledInstruction {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let instruction = match u.choose_index(LabelledInstruction::COUNT)? {
            0 => u.arbitrary::<AnInstruction<String>>()?,
            1 => return Ok(Self::Label(u.arbitrary::<InstructionLabel>()?.into())),
            2 => return Ok(Self::Breakpoint),
            3 => return Ok(Self::TypeHint(u.arbitrary()?)),
            4 => return Ok(Self::AssertionContext(u.arbitrary()?)),
            _ => unreachable!(),
        };
        let legal_label = String::from(u.arbitrary::<InstructionLabel>()?);
        let instruction = instruction.map_call_address(|_| legal_label.clone());

        Ok(Self::Instruction(instruction))
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct InstructionLabel(String);

impl From<InstructionLabel> for String {
    fn from(label: InstructionLabel) -> Self {
        label.0
    }
}

impl<'a> Arbitrary<'a> for InstructionLabel {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let legal_start_characters = ('a'..='z').chain('A'..='Z').chain('_'..='_');
        let legal_characters = legal_start_characters
            .clone()
            .chain('0'..='9')
            .chain('-'..='-')
            .collect_vec();

        let mut label = u.choose(&legal_start_characters.collect_vec())?.to_string();
        for _ in 0..u.arbitrary_len::<char>()? {
            label.push(*u.choose(&legal_characters)?);
        }
        while ALL_INSTRUCTION_NAMES.contains(&label.as_str()) {
            label.push(*u.choose(&legal_characters)?);
        }
        Ok(Self(label))
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct TypeHintVariableName(String);

impl From<TypeHintVariableName> for String {
    fn from(label: TypeHintVariableName) -> Self {
        label.0
    }
}

impl<'a> Arbitrary<'a> for TypeHintVariableName {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let legal_start_characters = 'a'..='z';
        let legal_characters = legal_start_characters
            .clone()
            .chain('0'..='9')
            .chain('_'..='_')
            .collect_vec();

        let mut variable_name = u.choose(&legal_start_characters.collect_vec())?.to_string();
        for _ in 0..u.arbitrary_len::<char>()? {
            variable_name.push(*u.choose(&legal_characters)?);
        }
        Ok(Self(variable_name))
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct TypeHintTypeName(String);

impl From<TypeHintTypeName> for String {
    fn from(label: TypeHintTypeName) -> Self {
        label.0
    }
}

impl<'a> Arbitrary<'a> for TypeHintTypeName {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let mut type_name = String::new();
        for _ in 0..u.arbitrary_len::<char>()?.min(3) {
            type_name.push('*');
        }

        let legal_start_characters = ('a'..='z').chain('A'..='Z').chain(std::iter::once('_'));
        type_name.push(*u.choose(&legal_start_characters.clone().collect_vec())?);

        let legal_characters = legal_start_characters.chain('0'..='9').collect_vec();
        for _ in 0..u.arbitrary_len::<char>()?.min(10) {
            type_name.push(*u.choose(&legal_characters)?);
        }

        let mut generics = Vec::new();
        for _ in 0..u.arbitrary_len::<Self>()?.min(3) {
            let Self(generic) = u.arbitrary()?;
            generics.push(generic);
        }
        if !generics.is_empty() {
            type_name.push('<');
            type_name.push_str(&generics.into_iter().join(", "));
            type_name.push('>');
        }

        Ok(Self(type_name))
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Eq, PartialEq, Error)]
pub enum InstructionError {
    #[error("opcode {0} is invalid")]
    InvalidOpcode(u32),

    #[error("opcode is out of range: {0}")]
    OutOfRangeOpcode(#[from] TryFromIntError),

    #[error("invalid argument {1} for instruction `{0}`")]
    IllegalArgument(Instruction, BFieldElement),

    #[error("instruction pointer points outside of program")]
    InstructionPointerOverflow,

    #[error("jump stack is empty")]
    JumpStackIsEmpty,

    #[error("assertion failed: {0}")]
    AssertionFailed(AssertionError),

    #[error("vector assertion failed because stack[{0}] != stack[{r}]: {1}", r = .0 + Digest::LEN)]
    VectorAssertionFailed(usize, AssertionError),

    #[error("0 does not have a multiplicative inverse")]
    InverseOfZero,

    #[error("division by 0 is impossible")]
    DivisionByZero,

    #[error("the Sponge state must be initialized before it can be used")]
    SpongeNotInitialized,

    #[error("the logarithm of 0 does not exist")]
    LogarithmOfZero,

    #[error("public input buffer is empty after {0} reads")]
    EmptyPublicInput(usize),

    #[error("secret input buffer is empty after {0} reads")]
    EmptySecretInput(usize),

    #[error("no more secret digests available")]
    EmptySecretDigestInput,

    #[error("Triton VM has halted and cannot execute any further instructions")]
    MachineHalted,

    #[error(transparent)]
    OpStackError(#[from] OpStackError),
}

/// An error giving additional context to any failed assertion.
#[non_exhaustive]
#[derive(Debug, Clone, Eq, PartialEq, Error)]
pub struct AssertionError {
    /// The [element](BFieldElement) expected by the assertion.
    pub expected: BFieldElement,

    /// The actual [element](BFieldElement) encountered when executing the
    /// assertion.
    pub actual: BFieldElement,

    /// A user-defined error ID. Only has user-defined, no inherent, semantics.
    pub id: Option<i128>,
}

impl Display for AssertionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        if let Some(id) = self.id {
            write!(f, "[{id}] ")?;
        }
        write!(f, "expected {}, got {}", self.expected, self.actual)
    }
}

impl AssertionError {
    pub fn new(expected: impl Into<BFieldElement>, actual: impl Into<BFieldElement>) -> Self {
        Self {
            expected: expected.into(),
            actual: actual.into(),
            id: None,
        }
    }

    #[must_use]
    pub fn with_context(mut self, context: AssertionContext) -> Self {
        match context {
            AssertionContext::ID(id) => self.id = Some(id),
        };
        self
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub mod tests {
    use std::collections::HashMap;

    use assert2::assert;
    use itertools::Itertools;
    use num_traits::One;
    use num_traits::Zero;
    use rand::Rng;
    use strum::EnumCount;
    use strum::IntoEnumIterator;
    use strum::VariantNames;
    use twenty_first::prelude::*;

    use crate::triton_asm;
    use crate::triton_program;

    use super::*;

    #[derive(Debug, Copy, Clone, EnumCount, EnumIter, VariantNames)]
    pub enum InstructionBucket {
        HasArg,
        ShrinksStack,
        IsU32,
    }

    impl InstructionBucket {
        pub fn contains(self, instruction: Instruction) -> bool {
            match self {
                InstructionBucket::HasArg => instruction.arg().is_some(),
                InstructionBucket::ShrinksStack => instruction.op_stack_size_influence() < 0,
                InstructionBucket::IsU32 => instruction.is_u32_instruction(),
            }
        }

        pub fn flag(self) -> u32 {
            match self {
                InstructionBucket::HasArg => 1,
                InstructionBucket::ShrinksStack => 1 << 1,
                InstructionBucket::IsU32 => 1 << 2,
            }
        }
    }

    impl Instruction {
        pub fn flag_set(self) -> u32 {
            InstructionBucket::iter()
                .map(|bucket| u32::from(bucket.contains(self)) * bucket.flag())
                .fold(0, |acc, bit_flag| acc | bit_flag)
        }

        fn computed_opcode(self) -> u32 {
            let mut index_within_flag_set = 0;
            for other_instruction in Instruction::iter() {
                if other_instruction == self {
                    break;
                }
                if other_instruction.flag_set() == self.flag_set() {
                    index_within_flag_set += 1;
                }
            }

            index_within_flag_set * 2_u32.pow(InstructionBucket::COUNT as u32) + self.flag_set()
        }
    }

    #[test]
    fn computed_and_actual_opcodes_are_identical() {
        for instruction in Instruction::iter() {
            let opcode = instruction.computed_opcode();
            let name = instruction.name();
            println!("{opcode: >3} {name}");
        }

        for instruction in Instruction::iter() {
            let expected_opcode = instruction.computed_opcode();
            let opcode = instruction.opcode();
            assert!(expected_opcode == opcode, "{instruction}");
        }
    }

    #[test]
    fn opcodes_are_unique() {
        let mut opcodes_to_instruction_map = HashMap::new();
        for instruction in Instruction::iter() {
            let opcode = instruction.opcode();
            let maybe_entry = opcodes_to_instruction_map.insert(opcode, instruction);
            if let Some(other_instruction) = maybe_entry {
                panic!("{other_instruction} and {instruction} both have opcode {opcode}.");
            }
        }
    }

    #[test]
    fn number_of_instruction_bits_is_correct() {
        let all_opcodes = Instruction::iter().map(|instruction| instruction.opcode());
        let highest_opcode = all_opcodes.max().unwrap();
        let num_required_bits_for_highest_opcode = highest_opcode.ilog2() + 1;
        assert!(InstructionBit::COUNT == num_required_bits_for_highest_opcode as usize);
    }

    #[test]
    fn parse_push_pop() {
        let program = triton_program!(push 1 push 1 add pop 2);
        let instructions = program.into_iter().collect_vec();
        let expected = vec![
            Instruction::Push(bfe!(1)),
            Instruction::Push(bfe!(1)),
            Instruction::Add,
            Instruction::Pop(NumberOfWords::N2),
        ];

        assert!(expected == instructions);
    }

    #[test]
    #[should_panic(expected = "Duplicate label: foo")]
    fn fail_on_duplicate_labels() {
        triton_program!(
            push 2
            call foo
            bar: push 2
            foo: push 3
            foo: push 4
            halt
        );
    }

    #[test]
    fn ib_registers_are_binary() {
        for instruction in ALL_INSTRUCTIONS {
            for instruction_bit in InstructionBit::iter() {
                let ib_value = instruction.ib(instruction_bit);
                assert!(ib_value.is_zero() ^ ib_value.is_one());
            }
        }
    }

    #[test]
    fn instruction_to_opcode_to_instruction_is_consistent() {
        for instr in ALL_INSTRUCTIONS {
            assert!(instr == instr.opcode().try_into().unwrap());
        }
    }

    /// While the correct _number_ of instructions (respectively instruction
    /// names) is guaranteed at compile time, this test ensures the absence
    /// of repetitions.
    #[test]
    fn list_of_all_instructions_contains_unique_instructions() {
        assert!(ALL_INSTRUCTIONS.into_iter().all_unique());
        assert!(ALL_INSTRUCTION_NAMES.into_iter().all_unique());
    }

    #[test]
    fn convert_various_types_to_instructions() {
        let _push = Instruction::try_from(1_usize).unwrap();
        let _dup = Instruction::try_from(9_u64).unwrap();
        let _swap = Instruction::try_from(17_u32).unwrap();
        let _pop = Instruction::try_from(3_usize).unwrap();
    }

    #[test]
    fn change_arguments_of_various_instructions() {
        use NumberOfWords::N1;
        use NumberOfWords::N4;
        use OpStackElement::ST0;

        assert!(Instruction::Push(bfe!(0)).change_arg(bfe!(7)).is_ok());
        assert!(Instruction::Dup(ST0).change_arg(bfe!(1024)).is_err());
        assert!(Instruction::Swap(ST0).change_arg(bfe!(1337)).is_err());
        assert!(Instruction::Swap(ST0).change_arg(bfe!(0)).is_ok());
        assert!(Instruction::Swap(ST0).change_arg(bfe!(1)).is_ok());
        assert!(Instruction::Pop(N1).change_arg(bfe!(2)).is_ok());
        assert!(Instruction::Pop(N4).change_arg(bfe!(0)).is_err());
        assert!(Instruction::Nop.change_arg(bfe!(7)).is_err());
    }

    #[test]
    fn print_various_instructions() {
        println!("{:?}", Instruction::Push(bfe!(7)));
        println!("{}", Instruction::Assert);
        println!("{:?}", Instruction::Invert);
        println!("{}", Instruction::Dup(OpStackElement::ST14));
    }

    #[test]
    fn instruction_size_is_consistent_with_having_arguments() {
        for instruction in Instruction::iter() {
            if instruction.arg().is_some() {
                assert!(2 == instruction.size())
            } else {
                assert!(1 == instruction.size())
            }
        }
    }

    #[test]
    fn opcodes_are_consistent_with_argument_indication_bit() {
        let argument_indicator_bit_mask = 1;
        for instruction in Instruction::iter() {
            let opcode = instruction.opcode();
            println!("Testing instruction {instruction} with opcode {opcode}.");
            let has_arg = instruction.arg().is_some();
            assert!(has_arg == (opcode & argument_indicator_bit_mask != 0));
        }
    }

    #[test]
    fn opcodes_are_consistent_with_shrink_stack_indication_bit() {
        let shrink_stack_indicator_bit_mask = 2;
        for instruction in Instruction::iter() {
            let opcode = instruction.opcode();
            println!("Testing instruction {instruction} with opcode {opcode}.");
            let shrinks_stack = instruction.op_stack_size_influence() < 0;
            assert!(shrinks_stack == (opcode & shrink_stack_indicator_bit_mask != 0));
        }
    }

    #[test]
    fn opcodes_are_consistent_with_u32_indication_bit() {
        let u32_indicator_bit_mask = 4;
        for instruction in Instruction::iter() {
            let opcode = instruction.opcode();
            println!("Testing instruction {instruction} with opcode {opcode}.");
            assert!(instruction.is_u32_instruction() == (opcode & u32_indicator_bit_mask != 0));
        }
    }

    #[test]
    fn instruction_bits_are_consistent() {
        for instruction_bit in InstructionBit::iter() {
            println!("Testing instruction bit {instruction_bit}.");
            let bit_index = usize::from(instruction_bit);
            let recovered_instruction_bit = InstructionBit::try_from(bit_index).unwrap();
            assert!(instruction_bit == recovered_instruction_bit);
        }
    }

    #[test]
    fn instruction_bit_conversion_fails_for_invalid_bit_index() {
        let invalid_bit_index = rand::rng().random_range(InstructionBit::COUNT..=usize::MAX);
        let maybe_instruction_bit = InstructionBit::try_from(invalid_bit_index);
        assert!(maybe_instruction_bit.is_err());
    }

    #[test]
    fn stringify_some_instructions() {
        let instructions = triton_asm!(push 3 invert push 2 mul push 1 add write_io 1 halt);
        let code = instructions.iter().join("\n");
        println!("{code}");
    }

    #[test]
    fn labelled_instructions_act_on_op_stack_as_indicated() {
        for instruction in ALL_INSTRUCTIONS {
            let labelled_instruction = instruction.map_call_address(|_| "dummy_label".to_string());
            let labelled_instruction = LabelledInstruction::Instruction(labelled_instruction);

            assert!(
                instruction.op_stack_size_influence()
                    == labelled_instruction.op_stack_size_influence()
            );
        }
    }

    #[test]
    fn labels_indicate_no_change_to_op_stack() {
        let labelled_instruction = LabelledInstruction::Label("dummy_label".to_string());
        assert!(0 == labelled_instruction.op_stack_size_influence());
    }

    #[test]
    fn can_change_arg() {
        for instr in ALL_INSTRUCTIONS {
            if let Some(arg) = instr.arg() {
                let new_instr = instr.change_arg(arg + bfe!(1)).unwrap();
                assert_eq!(instr.opcode(), new_instr.opcode());
                assert_ne!(instr, new_instr);
            } else {
                assert!(instr.change_arg(bfe!(0)).is_err())
            }
        }
    }
}
