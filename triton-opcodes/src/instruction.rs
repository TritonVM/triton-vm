use std::collections::HashMap;
use std::fmt::Display;
use std::vec;

use anyhow::bail;
use anyhow::Result;
use strum::EnumCount;
use strum::IntoEnumIterator;
use strum_macros::Display as DisplayMacro;
use strum_macros::EnumCount as EnumCountMacro;
use strum_macros::EnumIter;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::b_field_element::BFIELD_ZERO;

use AnInstruction::*;

use crate::ord_n::Ord16;
use crate::ord_n::Ord16::*;
use crate::ord_n::Ord8;

/// An `Instruction` has `call` addresses encoded as absolute integers.
pub type Instruction = AnInstruction<BFieldElement>;
pub const ALL_INSTRUCTIONS: [Instruction; Instruction::COUNT] = all_instructions_without_args();
pub const ALL_INSTRUCTION_NAMES: [&str; Instruction::COUNT] = all_instruction_names();

/// A `LabelledInstruction` has `call` addresses encoded as label names.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LabelledInstruction {
    /// Instructions belong to the ISA:
    ///
    /// <https://triton-vm.org/spec/isa.html>
    Instruction(AnInstruction<String>),

    /// Labels look like "`<name>:`" and are translated into absolute addresses.
    Label(String),
}

impl Display for LabelledInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LabelledInstruction::Instruction(instr) => write!(f, "{instr}"),
            LabelledInstruction::Label(label_name) => write!(f, "{label_name}:"),
        }
    }
}

#[derive(Debug, DisplayMacro, Clone, Copy, PartialEq, Eq, Hash, EnumCountMacro)]
pub enum DivinationHint {}

/// A Triton VM instruction. See the
/// [Instruction Set Architecture](https://triton-vm.org/spec/isa.html)
/// for more details.
///
/// The type parameter `Dest` describes the type of addresses (absolute or labels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumCountMacro, EnumIter)]
pub enum AnInstruction<Dest: PartialEq + Default> {
    // OpStack manipulation
    Pop,
    Push(BFieldElement),
    Divine(Option<DivinationHint>),
    Dup(Ord16),
    Swap(Ord16),

    // Control flow
    Nop,
    Skiz,
    Call(Dest),
    Return,
    Recurse,
    Assert,
    Halt,

    // Memory access
    ReadMem,
    WriteMem,

    // Hashing-related
    Hash,
    DivineSibling,
    AssertVector,
    AbsorbInit,
    Absorb,
    Squeeze,

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
    Div,

    // Extension field arithmetic on stack
    XxAdd,
    XxMul,
    XInvert,
    XbMul,

    // Read/write
    ReadIo,
    WriteIo,
}

impl<Dest: PartialEq + Default> AnInstruction<Dest> {
    /// Drop the specific argument in favor of a default one.
    pub fn strip(&self) -> Self {
        match self {
            Push(_) => Push(Default::default()),
            Divine(_) => Divine(Default::default()),
            Dup(_) => Dup(Default::default()),
            Swap(_) => Swap(Default::default()),
            Call(_) => Call(Default::default()),
            Pop => Pop,
            Nop => Nop,
            Skiz => Skiz,
            Return => Return,
            Recurse => Recurse,
            Assert => Assert,
            Halt => Halt,
            ReadMem => ReadMem,
            WriteMem => WriteMem,
            Hash => Hash,
            DivineSibling => DivineSibling,
            AssertVector => AssertVector,
            AbsorbInit => AbsorbInit,
            Absorb => Absorb,
            Squeeze => Squeeze,
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
            Div => Div,
            XxAdd => XxAdd,
            XxMul => XxMul,
            XInvert => XInvert,
            XbMul => XbMul,
            ReadIo => ReadIo,
            WriteIo => WriteIo,
        }
    }

    /// Assign a unique positive integer to each `Instruction`.
    pub fn opcode(&self) -> u32 {
        match self {
            Pop => 2,
            Push(_) => 1,
            Divine(_) => 8,
            Dup(_) => 9,
            Swap(_) => 17,
            Nop => 16,
            Skiz => 10,
            Call(_) => 25,
            Return => 24,
            Recurse => 32,
            Assert => 18,
            Halt => 0,
            ReadMem => 40,
            WriteMem => 26,
            Hash => 48,
            DivineSibling => 56,
            AssertVector => 64,
            AbsorbInit => 72,
            Absorb => 80,
            Squeeze => 88,
            Add => 34,
            Mul => 42,
            Invert => 96,
            Eq => 50,
            Split => 4,
            Lt => 12,
            And => 20,
            Xor => 28,
            Log2Floor => 36,
            Pow => 44,
            Div => 52,
            XxAdd => 104,
            XxMul => 112,
            XInvert => 120,
            XbMul => 58,
            ReadIo => 128,
            WriteIo => 66,
        }
    }

    const fn name(&self) -> &'static str {
        match self {
            Pop => "pop",
            Push(_) => "push",
            Divine(_) => "divine",
            Dup(_) => "dup",
            Swap(_) => "swap",
            Nop => "nop",
            Skiz => "skiz",
            Call(_) => "call",
            Return => "return",
            Recurse => "recurse",
            Assert => "assert",
            Halt => "halt",
            ReadMem => "read_mem",
            WriteMem => "write_mem",
            Hash => "hash",
            DivineSibling => "divine_sibling",
            AssertVector => "assert_vector",
            AbsorbInit => "absorb_init",
            Absorb => "absorb",
            Squeeze => "squeeze",
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
            Div => "div",
            XxAdd => "xxadd",
            XxMul => "xxmul",
            XInvert => "xinvert",
            XbMul => "xbmul",
            ReadIo => "read_io",
            WriteIo => "write_io",
        }
    }

    pub fn opcode_b(&self) -> BFieldElement {
        self.opcode().into()
    }

    pub fn size(&self) -> usize {
        match matches!(self, Push(_) | Dup(_) | Swap(_) | Call(_)) {
            true => 2,
            false => 1,
        }
    }

    /// Get the i'th instruction bit
    pub fn ib(&self, arg: Ord8) -> BFieldElement {
        let opcode = self.opcode();
        let bit_number: usize = arg.into();

        ((opcode >> bit_number) & 1).into()
    }

    fn map_call_address<F, NewDest: PartialEq + Default>(&self, f: F) -> AnInstruction<NewDest>
    where
        F: Fn(&Dest) -> NewDest,
    {
        match self {
            Pop => Pop,
            Push(x) => Push(*x),
            Divine(x) => Divine(*x),
            Dup(x) => Dup(*x),
            Swap(x) => Swap(*x),
            Nop => Nop,
            Skiz => Skiz,
            Call(label) => Call(f(label)),
            Return => Return,
            Recurse => Recurse,
            Assert => Assert,
            Halt => Halt,
            ReadMem => ReadMem,
            WriteMem => WriteMem,
            Hash => Hash,
            DivineSibling => DivineSibling,
            AssertVector => AssertVector,
            AbsorbInit => AbsorbInit,
            Absorb => Absorb,
            Squeeze => Squeeze,
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
            Div => Div,
            XxAdd => XxAdd,
            XxMul => XxMul,
            XInvert => XInvert,
            XbMul => XbMul,
            ReadIo => ReadIo,
            WriteIo => WriteIo,
        }
    }
}

impl<Dest: Display + PartialEq + Default> Display for AnInstruction<Dest> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())?;
        match self {
            Push(arg) => write!(f, " {arg}"),
            Divine(Some(hint)) => write!(f, "_{}", format!("{hint}").to_ascii_lowercase()),
            Dup(arg) | Swap(arg) => write!(f, "{arg}"),
            Call(arg) => write!(f, " {arg}"),
            _ => Ok(()),
        }
    }
}

impl Instruction {
    pub fn arg(&self) -> Option<BFieldElement> {
        match self {
            // Double-word instructions (instructions that take arguments)
            Push(arg) => Some(*arg),
            Dup(arg) => Some(ord16_to_bfe(arg)),
            Swap(arg) => Some(ord16_to_bfe(arg)),
            Call(arg) => Some(*arg),
            _ => None,
        }
    }
}

impl TryFrom<u32> for Instruction {
    type Error = anyhow::Error;

    fn try_from(opcode: u32) -> Result<Self> {
        if let Some(instruction) =
            Instruction::iter().find(|instruction| instruction.opcode() == opcode)
        {
            Ok(instruction)
        } else {
            bail!("No instruction with opcode {opcode} exists.")
        }
    }
}

impl TryFrom<u64> for Instruction {
    type Error = anyhow::Error;

    fn try_from(opcode: u64) -> Result<Self> {
        (opcode as u32).try_into()
    }
}

impl TryFrom<usize> for Instruction {
    type Error = anyhow::Error;

    fn try_from(opcode: usize) -> Result<Self> {
        (opcode as u32).try_into()
    }
}

fn ord16_to_bfe(n: &Ord16) -> BFieldElement {
    let n: u32 = n.into();
    n.into()
}

/// Convert a program with labels to a program with absolute positions
pub fn convert_labels(program: &[LabelledInstruction]) -> Vec<Instruction> {
    let mut label_map = HashMap::<String, usize>::new();
    let mut instruction_pointer: usize = 0;

    // 1. Add all labels to a map
    for labelled_instruction in program.iter() {
        match labelled_instruction {
            LabelledInstruction::Label(label_name) => {
                label_map.insert(label_name.clone(), instruction_pointer);
            }

            LabelledInstruction::Instruction(instr) => {
                instruction_pointer += instr.size();
            }
        }
    }

    // 2. Convert every label to the lookup value of that map
    program
        .iter()
        .flat_map(|labelled_instruction| convert_labels_helper(labelled_instruction, &label_map))
        .collect()
}

fn convert_labels_helper(
    instruction: &LabelledInstruction,
    label_map: &HashMap<String, usize>,
) -> Vec<Instruction> {
    match instruction {
        LabelledInstruction::Label(_) => vec![],

        LabelledInstruction::Instruction(instr) => {
            let unlabelled_instruction: Instruction = instr.map_call_address(|label_name| {
                let label_not_found = format!("Label not found: {label_name}");
                let absolute_address = label_map.get(label_name).expect(&label_not_found);
                BFieldElement::new(*absolute_address as u64)
            });

            vec![unlabelled_instruction]
        }
    }
}

pub const fn all_instructions_without_args() -> [AnInstruction<BFieldElement>; Instruction::COUNT] {
    [
        Pop,
        Push(BFIELD_ZERO),
        Divine(None),
        Dup(ST0),
        Swap(ST0),
        Nop,
        Skiz,
        Call(BFIELD_ZERO),
        Return,
        Recurse,
        Assert,
        Halt,
        ReadMem,
        WriteMem,
        Hash,
        DivineSibling,
        AssertVector,
        AbsorbInit,
        Absorb,
        Squeeze,
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
        Div,
        XxAdd,
        XxMul,
        XInvert,
        XbMul,
        ReadIo,
        WriteIo,
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

pub mod sample_programs {
    pub const PUSH_PUSH_ADD_POP_S: &str = "
        push 1
        push 1
        add
        pop
    ";

    pub const EDGY_RAM_WRITES: &str = concat!();

    pub const READ_WRITE_X3: &str = "
        read_io
        read_io
        read_io
        write_io
        write_io
        write_io
    ";

    pub const READ_X3_WRITE_X14: &str = "
        read_io read_io read_io
        dup0 dup2 dup4
        dup0 dup2 dup4
        dup0 dup2 dup4
        dup0 dup2
        write_io write_io write_io write_io
        write_io write_io write_io write_io
        write_io write_io write_io write_io
        write_io write_io
    ";

    pub const HASH_HASH_HASH_HALT: &str = "
        hash
        hash
        hash
        halt
    ";
}

#[cfg(test)]
mod instruction_tests {
    use crate::instruction::ALL_INSTRUCTIONS;
    use itertools::Itertools;
    use num_traits::One;
    use num_traits::Zero;
    use strum::EnumCount;
    use strum::IntoEnumIterator;
    use twenty_first::shared_math::b_field_element::BFieldElement;

    use crate::ord_n::Ord8;
    use crate::program::Program;

    use super::all_instructions_without_args;
    use super::AnInstruction::{self, *};

    #[test]
    fn opcode_test() {
        // test for duplicates
        let mut opcodes = vec![];
        for instruction in AnInstruction::<BFieldElement>::iter() {
            assert!(
                !opcodes.contains(&instruction.opcode()),
                "Have different instructions with same opcode."
            );
            opcodes.push(instruction.opcode());
        }

        for opc in opcodes.iter() {
            println!(
                "opcode {} exists: {}",
                opc,
                AnInstruction::<BFieldElement>::try_from(*opc).unwrap()
            );
        }

        // assert size of list corresponds to number of opcodes
        assert_eq!(
            AnInstruction::<BFieldElement>::COUNT,
            opcodes.len(),
            "Mismatch in number of instructions!"
        );

        // assert iter method also covers push
        assert!(
            opcodes.contains(&AnInstruction::<BFieldElement>::Push(Default::default()).opcode()),
            "list of opcodes needs to contain push"
        );

        // test for width
        let max_opcode: u32 = AnInstruction::<BFieldElement>::iter()
            .map(|inst| inst.opcode())
            .max()
            .unwrap();
        let mut num_bits = 0;
        while (1 << num_bits) < max_opcode {
            num_bits += 1;
        }
        assert!(
            num_bits <= Ord8::COUNT,
            "Biggest instruction needs more than {} bits :(",
            Ord8::COUNT
        );

        // assert consistency
        for instruction in AnInstruction::<BFieldElement>::iter() {
            assert!(
                instruction == instruction.opcode().try_into().unwrap(),
                "instruction to opcode map must be consistent"
            );
        }
    }

    #[test]
    fn parse_push_pop_test() {
        let code = "
            push 1
            push 1
            add
            pop
        ";
        let program = Program::from_code(code).unwrap();
        let instructions = program.into_iter().collect_vec();
        let expected = vec![
            Push(BFieldElement::one()),
            Push(BFieldElement::one()),
            Add,
            Pop,
        ];

        assert_eq!(expected, instructions);
    }

    #[test]
    fn fail_on_duplicate_labels_test() {
        let code = "
            push 2
            call foo
            bar: push 2
            foo: push 3
            foo: push 4
            halt
        ";
        let program = Program::from_code(code);
        assert!(
            program.is_err(),
            "Duplicate labels should result in a parse error"
        );
    }

    #[test]
    fn ib_registers_are_binary_test() {
        use Ord8::*;

        for instruction in ALL_INSTRUCTIONS {
            let all_ibs: [Ord8; Ord8::COUNT] = [IB0, IB1, IB2, IB3, IB4, IB5, IB6, IB7];
            for ib in all_ibs {
                let ib_value = instruction.ib(ib);
                assert!(
                    ib_value.is_zero() || ib_value.is_one(),
                    "IB{ib} for instruction {instruction} is 0 or 1 ({ib_value})",
                );
            }
        }
    }

    #[test]
    fn instruction_to_opcode_to_instruction_is_consistent_test() {
        for instr in all_instructions_without_args() {
            assert_eq!(instr, instr.opcode().try_into().unwrap());
        }
    }

    #[test]
    fn print_all_instructions_and_opcodes() {
        for instr in all_instructions_without_args() {
            println!("{:>3} {: <10}", instr.opcode(), format!("{instr}"));
        }
    }
}
