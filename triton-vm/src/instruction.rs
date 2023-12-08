use std::collections::HashMap;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::result;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use get_size::GetSize;
use itertools::Itertools;
use lazy_static::lazy_static;
use serde_derive::Deserialize;
use serde_derive::Serialize;
use strum::EnumCount;
use strum::EnumIter;
use strum::IntoEnumIterator;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::b_field_element::BFIELD_ZERO;

use AnInstruction::*;

use crate::error::InstructionError;
use crate::instruction::InstructionBit::*;
use crate::op_stack::NumberOfWords::*;
use crate::op_stack::OpStackElement::*;
use crate::op_stack::*;

type Result<T> = result::Result<T, InstructionError>;

/// An `Instruction` has `call` addresses encoded as absolute integers.
pub type Instruction = AnInstruction<BFieldElement>;

pub const ALL_INSTRUCTIONS: [Instruction; Instruction::COUNT] = all_instructions_without_args();
pub const ALL_INSTRUCTION_NAMES: [&str; Instruction::COUNT] = all_instruction_names();

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
#[derive(Debug, Clone, PartialEq, Eq, Hash, EnumCount)]
pub enum LabelledInstruction {
    /// An instructions from the [instruction set architecture][isa].
    ///
    /// [isa]: https://triton-vm.org/spec/isa.html
    Instruction(AnInstruction<String>),

    /// Labels look like "`<name>:`" and are translated into absolute addresses.
    Label(String),

    Breakpoint,

    TypeHint(TypeHint),
}

/// A hint about a range of stack elements. Helps debugging programs written for Triton VM.
/// **Does not enforce types.**
///
/// Usually constructed by parsing special annotations in the assembly code, for example:
/// ```tasm
/// hint variable_name: the_type = stack[0];
/// hint my_list = stack[1..4];
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, GetSize, Serialize, Deserialize)]
pub struct TypeHint {
    pub starting_index: usize,
    pub length: usize,

    /// The name of the type, _e.g._, `u32`, `list`, `Digest`, et cetera.
    pub type_name: Option<String>,

    /// The name of the variable.
    pub variable_name: String,
}

impl Display for TypeHint {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let variable = &self.variable_name;
        let type_name = match self.type_name {
            Some(ref type_name) => format!(": {type_name}"),
            None => "".to_string(),
        };

        let start = self.starting_index;
        let range = match self.length {
            1 => format!("{start}"),
            _ => format!("{start}..{end}", end = start + self.length),
        };

        write!(f, "hint {variable}{type_name} = stack[{range}];")
    }
}

impl<'a> Arbitrary<'a> for TypeHint {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let starting_index = u.arbitrary()?;
        let length = u.int_in_range(1..=500)?;
        let type_name = match u.arbitrary()? {
            true => Some(u.arbitrary::<TypeHintTypeName>()?.into()),
            false => None,
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

impl LabelledInstruction {
    pub const fn grows_op_stack(&self) -> bool {
        self.op_stack_size_influence() > 0
    }

    pub const fn changes_op_stack_size(&self) -> bool {
        self.op_stack_size_influence() != 0
    }

    pub const fn shrinks_op_stack(&self) -> bool {
        self.op_stack_size_influence() < 0
    }

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
        }
    }
}

/// Helps printing instructions with their labels.
pub fn stringify_instructions(instructions: &[LabelledInstruction]) -> String {
    instructions.iter().join("\n")
}

/// A Triton VM instruction. See the
/// [Instruction Set Architecture](https://triton-vm.org/spec/isa.html)
/// for more details.
///
/// The type parameter `Dest` describes the type of addresses (absolute or labels).
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    EnumCount,
    EnumIter,
    GetSize,
    Serialize,
    Deserialize,
    Arbitrary,
)]
pub enum AnInstruction<Dest: PartialEq + Default> {
    // OpStack manipulation
    Pop(NumberOfWords),
    Push(BFieldElement),
    Divine(NumberOfWords),
    Dup(OpStackElement),
    Swap(OpStackElement),

    // Control flow
    Halt,
    Nop,
    Skiz,
    Call(Dest),
    Return,
    Recurse,
    Assert,

    // Memory access
    ReadMem(NumberOfWords),
    WriteMem(NumberOfWords),

    // Hashing-related
    Hash,
    DivineSibling,
    AssertVector,
    SpongeInit,
    SpongeAbsorb,
    SpongeSqueeze,

    // Base field arithmetic on stack
    Add,
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
}

impl<Dest: PartialEq + Default> AnInstruction<Dest> {
    /// Assign a unique positive integer to each `Instruction`.
    pub const fn opcode(&self) -> u32 {
        match self {
            Pop(_) => 3,
            Push(_) => 1,
            Divine(_) => 9,
            Dup(_) => 17,
            Swap(_) => 25,
            Halt => 0,
            Nop => 8,
            Skiz => 2,
            Call(_) => 33,
            Return => 16,
            Recurse => 24,
            Assert => 10,
            ReadMem(_) => 41,
            WriteMem(_) => 11,
            Hash => 18,
            DivineSibling => 32,
            AssertVector => 26,
            SpongeInit => 40,
            SpongeAbsorb => 34,
            SpongeSqueeze => 48,
            Add => 42,
            Mul => 50,
            Invert => 56,
            Eq => 58,
            Split => 4,
            Lt => 6,
            And => 14,
            Xor => 22,
            Log2Floor => 12,
            Pow => 30,
            DivMod => 20,
            PopCount => 28,
            XxAdd => 66,
            XxMul => 74,
            XInvert => 64,
            XbMul => 82,
            ReadIo(_) => 49,
            WriteIo(_) => 19,
        }
    }

    pub(crate) const fn name(&self) -> &'static str {
        match self {
            Pop(_) => "pop",
            Push(_) => "push",
            Divine(_) => "divine",
            Dup(_) => "dup",
            Swap(_) => "swap",
            Halt => "halt",
            Nop => "nop",
            Skiz => "skiz",
            Call(_) => "call",
            Return => "return",
            Recurse => "recurse",
            Assert => "assert",
            ReadMem(_) => "read_mem",
            WriteMem(_) => "write_mem",
            Hash => "hash",
            DivineSibling => "divine_sibling",
            AssertVector => "assert_vector",
            SpongeInit => "sponge_init",
            SpongeAbsorb => "sponge_absorb",
            SpongeSqueeze => "sponge_squeeze",
            Add => "add",
            Mul => "mul",
            Invert => "invert",
            Eq => "eq",
            Split => "split",
            Lt => "lt",
            And => "and",
            Xor => "xor",
            Log2Floor => "log_2_floor",
            Pow => "pow",
            DivMod => "div_mod",
            PopCount => "pop_count",
            XxAdd => "xxadd",
            XxMul => "xxmul",
            XInvert => "xinvert",
            XbMul => "xbmul",
            ReadIo(_) => "read_io",
            WriteIo(_) => "write_io",
        }
    }

    pub fn opcode_b(&self) -> BFieldElement {
        self.opcode().into()
    }

    pub fn size(&self) -> usize {
        match self {
            Pop(_) | Push(_) => 2,
            Divine(_) => 2,
            Dup(_) | Swap(_) => 2,
            Call(_) => 2,
            ReadMem(_) | WriteMem(_) => 2,
            ReadIo(_) | WriteIo(_) => 2,
            _ => 1,
        }
    }

    /// Get the i'th instruction bit
    pub fn ib(&self, arg: InstructionBit) -> BFieldElement {
        let opcode = self.opcode();
        let bit_number: usize = arg.into();

        ((opcode >> bit_number) & 1).into()
    }

    pub(crate) fn map_call_address<F, NewDest>(&self, f: F) -> AnInstruction<NewDest>
    where
        F: Fn(&Dest) -> NewDest,
        NewDest: PartialEq + Default,
    {
        match self {
            Pop(x) => Pop(*x),
            Push(x) => Push(*x),
            Divine(x) => Divine(*x),
            Dup(x) => Dup(*x),
            Swap(x) => Swap(*x),
            Halt => Halt,
            Nop => Nop,
            Skiz => Skiz,
            Call(label) => Call(f(label)),
            Return => Return,
            Recurse => Recurse,
            Assert => Assert,
            ReadMem(x) => ReadMem(*x),
            WriteMem(x) => WriteMem(*x),
            Hash => Hash,
            DivineSibling => DivineSibling,
            AssertVector => AssertVector,
            SpongeInit => SpongeInit,
            SpongeAbsorb => SpongeAbsorb,
            SpongeSqueeze => SpongeSqueeze,
            Add => Add,
            Mul => Mul,
            Invert => Invert,
            Eq => Eq,
            Split => Split,
            Lt => Lt,
            And => And,
            Xor => Xor,
            Log2Floor => Log2Floor,
            Pow => Pow,
            DivMod => DivMod,
            PopCount => PopCount,
            XxAdd => XxAdd,
            XxMul => XxMul,
            XInvert => XInvert,
            XbMul => XbMul,
            ReadIo(x) => ReadIo(*x),
            WriteIo(x) => WriteIo(*x),
        }
    }

    pub const fn grows_op_stack(&self) -> bool {
        self.op_stack_size_influence() > 0
    }

    pub const fn changes_op_stack_size(&self) -> bool {
        self.op_stack_size_influence() != 0
    }

    pub const fn shrinks_op_stack(&self) -> bool {
        self.op_stack_size_influence() < 0
    }

    pub const fn op_stack_size_influence(&self) -> i32 {
        match self {
            Pop(n) => -(n.num_words() as i32),
            Push(_) => 1,
            Divine(n) => n.num_words() as i32,
            Dup(_) => 1,
            Swap(_) => 0,
            Halt => 0,
            Nop => 0,
            Skiz => -1,
            Call(_) => 0,
            Return => 0,
            Recurse => 0,
            Assert => -1,
            ReadMem(n) => n.num_words() as i32,
            WriteMem(n) => -(n.num_words() as i32),
            Hash => -5,
            DivineSibling => 5,
            AssertVector => -5,
            SpongeInit => 0,
            SpongeAbsorb => -10,
            SpongeSqueeze => 10,
            Add => -1,
            Mul => -1,
            Invert => 0,
            Eq => -1,
            Split => 1,
            Lt => -1,
            And => -1,
            Xor => -1,
            Log2Floor => 0,
            Pow => -1,
            DivMod => 0,
            PopCount => 0,
            XxAdd => -3,
            XxMul => -3,
            XInvert => 0,
            XbMul => -1,
            ReadIo(n) => n.num_words() as i32,
            WriteIo(n) => -(n.num_words() as i32),
        }
    }

    /// Indicates whether the instruction operates on base field elements that are also u32s.
    pub fn is_u32_instruction(&self) -> bool {
        matches!(
            self,
            Split | Lt | And | Xor | Log2Floor | Pow | DivMod | PopCount
        )
    }

    pub fn has_illegal_argument(&self) -> bool {
        matches!(self, Swap(ST0))
    }
}

impl<Dest: Display + PartialEq + Default> Display for AnInstruction<Dest> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.name())?;
        match self {
            Push(arg) => write!(f, " {arg}"),
            Pop(arg) | Divine(arg) => write!(f, " {arg}"),
            Dup(arg) | Swap(arg) => write!(f, " {arg}"),
            Call(arg) => write!(f, " {arg}"),
            ReadMem(arg) | WriteMem(arg) => write!(f, " {arg}"),
            ReadIo(arg) | WriteIo(arg) => write!(f, " {arg}"),
            _ => Ok(()),
        }
    }
}

impl Instruction {
    /// Get the argument of the instruction, if it has one.
    pub fn arg(&self) -> Option<BFieldElement> {
        match self {
            Push(arg) | Call(arg) => Some(*arg),
            Pop(arg) | Divine(arg) => Some(arg.into()),
            Dup(arg) | Swap(arg) => Some(arg.into()),
            ReadMem(arg) | WriteMem(arg) => Some(arg.into()),
            ReadIo(arg) | WriteIo(arg) => Some(arg.into()),
            _ => None,
        }
    }

    /// `true` iff the instruction has an argument.
    pub fn has_arg(&self) -> bool {
        self.arg().is_some()
    }

    /// Change the argument of the instruction, if it has one. Returns an `Err` if the instruction
    /// does not have an argument or if the argument is out of range.
    pub fn change_arg(self, new_arg: BFieldElement) -> Result<Self> {
        let illegal_argument_error = InstructionError::IllegalArgument(self, new_arg);
        let num_words = new_arg.try_into().map_err(|_| illegal_argument_error);
        let op_stack_element = new_arg.try_into().map_err(|_| illegal_argument_error);

        let new_instruction = match self {
            Pop(_) => Pop(num_words?),
            Push(_) => Push(new_arg),
            Divine(_) => Divine(num_words?),
            Dup(_) => Dup(op_stack_element?),
            Swap(_) => Swap(op_stack_element?),
            Call(_) => Call(new_arg),
            ReadMem(_) => ReadMem(num_words?),
            WriteMem(_) => WriteMem(num_words?),
            ReadIo(_) => ReadIo(num_words?),
            WriteIo(_) => WriteIo(num_words?),
            _ => return Err(illegal_argument_error),
        };

        match new_instruction.has_illegal_argument() {
            true => Err(illegal_argument_error),
            false => Ok(new_instruction),
        }
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

const fn all_instructions_without_args() -> [AnInstruction<BFieldElement>; Instruction::COUNT] {
    [
        Pop(N1),
        Push(BFIELD_ZERO),
        Divine(N1),
        Dup(ST0),
        Swap(ST0),
        Halt,
        Nop,
        Skiz,
        Call(BFIELD_ZERO),
        Return,
        Recurse,
        Assert,
        ReadMem(N1),
        WriteMem(N1),
        Hash,
        DivineSibling,
        AssertVector,
        SpongeInit,
        SpongeAbsorb,
        SpongeSqueeze,
        Add,
        Mul,
        Invert,
        Eq,
        Split,
        Lt,
        And,
        Xor,
        Log2Floor,
        Pow,
        DivMod,
        PopCount,
        XxAdd,
        XxMul,
        XInvert,
        XbMul,
        ReadIo(N1),
        WriteIo(N1),
    ]
}

const fn all_instruction_names() -> [&'static str; Instruction::COUNT] {
    let mut names = [""; Instruction::COUNT];
    let mut i = 0;
    while i < Instruction::COUNT {
        names[i] = ALL_INSTRUCTIONS[i].name();
        i += 1;
    }
    names
}

/// Indicators for all the possible bits in an [`Instruction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, EnumCount, EnumIter)]
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
            IB0 => 0,
            IB1 => 1,
            IB2 => 2,
            IB3 => 3,
            IB4 => 4,
            IB5 => 5,
            IB6 => 6,
        }
    }
}

impl TryFrom<usize> for InstructionBit {
    type Error = String;

    fn try_from(bit_index: usize) -> result::Result<Self, Self::Error> {
        match bit_index {
            0 => Ok(IB0),
            1 => Ok(IB1),
            2 => Ok(IB2),
            3 => Ok(IB3),
            4 => Ok(IB4),
            5 => Ok(IB5),
            6 => Ok(IB6),
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
            _ => unreachable!(),
        };
        let legal_label = String::from(u.arbitrary::<InstructionLabel>()?);
        let instruction = instruction.map_call_address(|_| legal_label.clone());

        if let Swap(ST0) = instruction {
            return Ok(Self::Instruction(Swap(ST1)));
        }

        Ok(Self::Instruction(instruction))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TypeHintTypeName(String);

impl From<TypeHintTypeName> for String {
    fn from(label: TypeHintTypeName) -> Self {
        label.0
    }
}

impl<'a> Arbitrary<'a> for TypeHintTypeName {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let legal_start_characters = ('a'..='z').chain('A'..='Z');
        let legal_characters = legal_start_characters
            .clone()
            .chain('0'..='9')
            .collect_vec();

        let mut type_name = u.choose(&legal_start_characters.collect_vec())?.to_string();
        for _ in 0..u.arbitrary_len::<char>()? {
            type_name.push(*u.choose(&legal_characters)?);
        }
        Ok(Self(type_name))
    }
}

#[cfg(test)]
mod tests {
    use assert2::assert;
    use assert2::let_assert;
    use std::collections::HashMap;

    use itertools::Itertools;
    use num_traits::One;
    use num_traits::Zero;
    use rand::thread_rng;
    use rand::Rng;
    use strum::EnumCount;
    use strum::IntoEnumIterator;
    use twenty_first::shared_math::b_field_element::BFieldElement;
    use twenty_first::shared_math::b_field_element::BFIELD_ZERO;
    use twenty_first::shared_math::digest::Digest;

    use crate::instruction::*;
    use crate::op_stack::NUM_OP_STACK_REGISTERS;
    use crate::triton_asm;
    use crate::triton_program;
    use crate::vm::tests::test_program_for_call_recurse_return;
    use crate::vm::VMState;
    use crate::NonDeterminism;
    use crate::Program;

    #[derive(Debug, Clone, Copy, EnumCount, EnumIter)]
    enum InstructionBucket {
        HasArg,
        ShrinksStack,
        IsU32,
    }

    impl InstructionBucket {
        fn contains(self, instruction: Instruction) -> bool {
            match self {
                InstructionBucket::HasArg => instruction.has_arg(),
                InstructionBucket::ShrinksStack => instruction.shrinks_op_stack(),
                InstructionBucket::IsU32 => instruction.is_u32_instruction(),
            }
        }
    }

    impl Instruction {
        #[must_use]
        fn replace_default_argument_if_illegal(self) -> Self {
            match self {
                Swap(ST0) => Swap(ST1),
                _ => self,
            }
        }

        fn flag_set(self) -> u32 {
            let instruction = self.replace_default_argument_if_illegal();
            InstructionBucket::iter()
                .map(|bucket| bucket.contains(instruction) as u32)
                .enumerate()
                .map(|(bucket_index, contains_self)| contains_self << bucket_index)
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
                panic!("{other_instruction} and {instruction} both have opcode {opcode}.",);
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
            Push(BFieldElement::one()),
            Push(BFieldElement::one()),
            Add,
            Pop(N2),
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

    #[test]
    /// Serves no other purpose than to increase code coverage results.
    fn run_constant_methods() {
        all_instructions_without_args();
        all_instruction_names();
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
        let push = Push(0_u64.into()).change_arg(7_u64.into());
        let dup = Dup(ST0).change_arg(1024_u64.into());
        let swap = Swap(ST0).change_arg(1337_u64.into());
        let swap_0 = Swap(ST0).change_arg(0_u64.into());
        let swap_1 = Swap(ST0).change_arg(1_u64.into());
        let pop_0 = Pop(N4).change_arg(0_u64.into());
        let pop_2 = Pop(N1).change_arg(2_u64.into());
        let nop = Nop.change_arg(7_u64.into());

        assert!(push.is_ok());
        assert!(dup.is_err());
        assert!(swap.is_err());
        assert!(swap_0.is_err());
        assert!(swap_1.is_ok());
        assert!(pop_0.is_err());
        assert!(pop_2.is_ok());
        assert!(nop.is_err());
    }

    #[test]
    fn print_various_instructions() {
        println!("instruction_push: {:?}", Instruction::Push(7_u64.into()));
        println!("instruction_assert: {}", Instruction::Assert);
        println!("instruction_invert: {:?}", Instruction::Invert);
        println!("instruction_dup: {}", Instruction::Dup(ST14));
    }

    #[test]
    fn instruction_size_is_consistent_with_having_arguments() {
        for instruction in Instruction::iter() {
            match instruction.has_arg() {
                true => assert!(2 == instruction.size()),
                false => assert!(1 == instruction.size()),
            }
        }
    }

    #[test]
    fn opcodes_are_consistent_with_argument_indication_bit() {
        let argument_indicator_bit_mask = 1;
        for instruction in Instruction::iter() {
            let opcode = instruction.opcode();
            println!("Testing instruction {instruction} with opcode {opcode}.");
            assert!(instruction.has_arg() == (opcode & argument_indicator_bit_mask != 0));
        }
    }

    #[test]
    fn opcodes_are_consistent_with_shrink_stack_indication_bit() {
        let shrink_stack_indicator_bit_mask = 2;
        for instruction in Instruction::iter() {
            let instruction = instruction.replace_default_argument_if_illegal();
            let opcode = instruction.opcode();
            println!("Testing instruction {instruction} with opcode {opcode}.");
            assert!(
                instruction.shrinks_op_stack() == (opcode & shrink_stack_indicator_bit_mask != 0)
            );
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
        let invalid_bit_index = thread_rng().gen_range(InstructionBit::COUNT..=usize::MAX);
        let maybe_instruction_bit = InstructionBit::try_from(invalid_bit_index);
        assert!(maybe_instruction_bit.is_err());
    }

    #[test]
    fn stringify_some_instructions() {
        let instructions = triton_asm!(push 3 invert push 2 mul push 1 add write_io 1 halt);
        let code = stringify_instructions(&instructions);
        println!("{code}");
    }

    #[test]
    fn instructions_act_on_op_stack_as_indicated() {
        for test_instruction in all_instructions_without_args() {
            let test_instruction = test_instruction.replace_default_argument_if_illegal();
            let (program, stack_size_before_test_instruction) =
                construct_test_program_for_instruction(test_instruction);
            let stack_size_after_test_instruction = terminal_op_stack_size_for_program(program);

            let stack_size_difference = (stack_size_after_test_instruction as i32)
                - (stack_size_before_test_instruction as i32);
            assert!(
                test_instruction.op_stack_size_influence() == stack_size_difference,
                "{test_instruction}"
            );
        }
    }

    fn construct_test_program_for_instruction(
        instruction: AnInstruction<BFieldElement>,
    ) -> (Program, usize) {
        match instruction_requires_jump_stack_setup(instruction) {
            true => program_with_jump_stack_setup(),
            false => program_without_jump_stack_setup_for_instruction(instruction),
        }
    }

    fn instruction_requires_jump_stack_setup(instruction: Instruction) -> bool {
        matches!(instruction, Call(_) | Return | Recurse)
    }

    fn program_with_jump_stack_setup() -> (Program, usize) {
        let program = test_program_for_call_recurse_return().program;
        let stack_size = NUM_OP_STACK_REGISTERS;
        (program, stack_size)
    }

    fn program_without_jump_stack_setup_for_instruction(
        test_instruction: AnInstruction<BFieldElement>,
    ) -> (Program, usize) {
        let num_push_instructions = 10;
        let push_instructions = triton_asm![push 1; num_push_instructions];
        let program = triton_program!(sponge_init {&push_instructions} {test_instruction} nop halt);

        let stack_size_when_reaching_test_instruction =
            NUM_OP_STACK_REGISTERS + num_push_instructions;
        (program, stack_size_when_reaching_test_instruction)
    }

    fn terminal_op_stack_size_for_program(program: Program) -> usize {
        let public_input = vec![BFIELD_ZERO].into();
        let mock_digests = vec![Digest::default()];
        let non_determinism: NonDeterminism<_> = vec![BFIELD_ZERO].into();
        let non_determinism = non_determinism.with_digests(mock_digests);

        let mut vm_state = VMState::new(&program, public_input, non_determinism);
        let_assert!(Ok(()) = vm_state.run());
        vm_state.op_stack.stack.len()
    }

    #[test]
    fn labelled_instructions_act_on_op_stack_as_indicated() {
        for instruction in all_instructions_without_args() {
            let labelled_instruction = instruction.map_call_address(|_| "dummy_label".to_string());
            let labelled_instruction = LabelledInstruction::Instruction(labelled_instruction);

            assert!(
                instruction.op_stack_size_influence()
                    == labelled_instruction.op_stack_size_influence()
            );
            assert!(instruction.grows_op_stack() == labelled_instruction.grows_op_stack());
            assert!(
                instruction.changes_op_stack_size() == labelled_instruction.changes_op_stack_size()
            );
            assert!(instruction.shrinks_op_stack() == labelled_instruction.shrinks_op_stack());
        }
    }

    #[test]
    fn labels_indicate_no_change_to_op_stack() {
        let labelled_instruction = LabelledInstruction::Label("dummy_label".to_string());
        assert!(0 == labelled_instruction.op_stack_size_influence());
        assert!(!labelled_instruction.grows_op_stack());
        assert!(!labelled_instruction.changes_op_stack_size());
        assert!(!labelled_instruction.shrinks_op_stack());
    }
}
