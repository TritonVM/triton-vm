use std::collections::HashMap;
use std::error::Error;
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

use AnInstruction::*;
use TokenError::*;

use crate::ord_n::Ord16;
use crate::ord_n::Ord16::*;
use crate::ord_n::Ord8;

/// An `Instruction` has `call` addresses encoded as absolute integers.
pub type Instruction = AnInstruction<BFieldElement>;

/// A `LabelledInstruction` has `call` addresses encoded as label names.
#[derive(Debug, Clone, Eq, Hash)]
pub enum LabelledInstruction<'a> {
    /// Instructions belong to the ISA:
    ///
    /// https://triton-vm.org/spec/isa.html
    Instruction(AnInstruction<String>, &'a str),

    /// Labels look like "`<name>:`" and are translated into absolute addresses.
    Label(String, &'a str),
}

// FIXME: This can be replaced with `#[derive(PartialEq)]` once old parser is dead.
impl<'a> PartialEq for LabelledInstruction<'a> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Instruction(instr1, _), Self::Instruction(instr2, _)) => instr1 == instr2,
            (Self::Label(label1, _), Self::Label(label2, _)) => label1 == label2,
            _ => false,
        }
    }
}

impl<'a> Display for LabelledInstruction<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LabelledInstruction::Instruction(instr, _) => write!(f, "{}", instr),
            LabelledInstruction::Label(label_name, _) => write!(f, "{}:", label_name),
        }
    }
}

pub fn token_str<'a>(instruction: &LabelledInstruction<'a>) -> &'a str {
    match instruction {
        LabelledInstruction::Instruction(_, token_str) => token_str,
        LabelledInstruction::Label(_, token_str) => token_str,
    }
}

#[derive(Debug, DisplayMacro, Clone, Copy, PartialEq, Eq, Hash, EnumCountMacro)]
pub enum DivinationHint {}

/// A Triton VM instruction
///
/// The ISA is defined at:
///
/// https://triton-vm.org/spec/isa.html
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

impl<Dest: Display + PartialEq + Default> Display for AnInstruction<Dest> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // OpStack manipulation
            Pop => write!(f, "pop"),
            Push(arg) => write!(f, "push {}", arg),
            Divine(Some(hint)) => write!(f, "divine_{}", format!("{hint}").to_ascii_lowercase()),
            Divine(None) => write!(f, "divine"),
            Dup(arg) => write!(f, "dup{}", arg),
            Swap(arg) => write!(f, "swap{}", arg),

            // Control flow
            Nop => write!(f, "nop"),
            Skiz => write!(f, "skiz"),
            Call(arg) => write!(f, "call {}", arg),
            Return => write!(f, "return"),
            Recurse => write!(f, "recurse"),
            Assert => write!(f, "assert"),
            Halt => write!(f, "halt"),

            // Memory access
            ReadMem => write!(f, "read_mem"),
            WriteMem => write!(f, "write_mem"),

            // Hashing-related
            Hash => write!(f, "hash"),
            DivineSibling => write!(f, "divine_sibling"),
            AssertVector => write!(f, "assert_vector"),
            AbsorbInit => write!(f, "absorb_init"),
            Absorb => write!(f, "absorb"),
            Squeeze => write!(f, "squeeze"),

            // Base field arithmetic on stack
            Add => write!(f, "add"),
            Mul => write!(f, "mul"),
            Invert => write!(f, "invert"),
            Eq => write!(f, "eq"),

            // Bitwise arithmetic on stack
            Split => write!(f, "split"),
            Lt => write!(f, "lt"),
            And => write!(f, "and"),
            Xor => write!(f, "xor"),
            Log2Floor => write!(f, "log_2_floor"),
            Pow => write!(f, "pow"),
            Div => write!(f, "div"),

            // Extension field arithmetic on stack
            XxAdd => write!(f, "xxadd"),
            XxMul => write!(f, "xxmul"),
            XInvert => write!(f, "xinvert"),
            XbMul => write!(f, "xbmul"),

            // Read/write
            ReadIo => write!(f, "read_io"),
            WriteIo => write!(f, "write_io"),
        }
    }
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
            WriteMem => 48,
            Hash => 56,
            DivineSibling => 64,
            AssertVector => 72,
            AbsorbInit => 80,
            Absorb => 88,
            Squeeze => 96,
            Add => 26,
            Mul => 34,
            Invert => 104,
            Eq => 42,
            Split => 4,
            Lt => 12,
            And => 20,
            Xor => 28,
            Log2Floor => 36,
            Pow => 44,
            Div => 52,
            XxAdd => 112,
            XxMul => 120,
            XInvert => 128,
            XbMul => 50,
            ReadIo => 136,
            WriteIo => 58,
        }
    }

    /// Returns whether a given instruction modifies the op-stack.
    ///
    /// A modification involves any amount of pushing and/or popping.
    pub fn is_op_stack_instruction(&self) -> bool {
        !matches!(
            self,
            Nop | Call(_) | Return | Recurse | Halt | Hash | AssertVector
        )
    }

    pub fn opcode_b(&self) -> BFieldElement {
        self.opcode().into()
    }

    pub fn size(&self) -> usize {
        if matches!(self, Push(_) | Dup(_) | Swap(_) | Call(_)) {
            2
        } else {
            1
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
            bail!("No instruction with opcode {} exists.", opcode)
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

#[derive(Debug)]
pub enum TokenError {
    UnexpectedEndOfStream,
    UnknownInstruction(String),
}

impl Display for TokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnknownInstruction(s) => write!(f, "UnknownInstruction({})", s),
            UnexpectedEndOfStream => write!(f, "UnexpectedEndOfStream"),
        }
    }
}

impl Error for TokenError {}

/// Convert a program with labels to a program with absolute positions
pub fn convert_labels(program: &[LabelledInstruction]) -> Vec<Instruction> {
    let mut label_map = HashMap::<String, usize>::new();
    let mut instruction_pointer: usize = 0;

    // 1. Add all labels to a map
    for labelled_instruction in program.iter() {
        match labelled_instruction {
            LabelledInstruction::Label(label_name, _) => {
                label_map.insert(label_name.clone(), instruction_pointer);
            }

            LabelledInstruction::Instruction(instr, _) => {
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
        LabelledInstruction::Label(_, _) => vec![],

        LabelledInstruction::Instruction(instr, _) => {
            let unlabelled_instruction: Instruction = instr.map_call_address(|label_name| {
                let label_not_found = format!("Label not found: {}", label_name);
                let absolute_address = label_map.get(label_name).expect(&label_not_found);
                BFieldElement::new(*absolute_address as u64)
            });

            vec![unlabelled_instruction]
        }
    }
}

pub fn is_instruction_name(s: &str) -> bool {
    match s {
        "pop" => true,
        "push" => true,
        "divine" => true,
        "dup0" => true,
        "dup1" => true,
        "dup2" => true,
        "dup3" => true,
        "dup4" => true,
        "dup5" => true,
        "dup6" => true,
        "dup7" => true,
        "dup8" => true,
        "dup9" => true,
        "dup10" => true,
        "dup11" => true,
        "dup12" => true,
        "dup13" => true,
        "dup14" => true,
        "dup15" => true,
        "swap1" => true,
        "swap2" => true,
        "swap3" => true,
        "swap4" => true,
        "swap5" => true,
        "swap6" => true,
        "swap7" => true,
        "swap8" => true,
        "swap9" => true,
        "swap10" => true,
        "swap11" => true,
        "swap12" => true,
        "swap13" => true,
        "swap14" => true,
        "swap15" => true,

        // Control flow
        "nop" => true,
        "skiz" => true,
        "call" => true,
        "return" => true,
        "recurse" => true,
        "assert" => true,
        "halt" => true,

        // Memory access
        "read_mem" => true,
        "write_mem" => true,

        // Hashing-related instructions
        "hash" => true,
        "divine_sibling" => true,
        "assert_vector" => true,

        // Arithmetic on stack instructions
        "add" => true,
        "mul" => true,
        "invert" => true,
        "split" => true,
        "eq" => true,
        "xxadd" => true,
        "xxmul" => true,
        "xinvert" => true,
        "xbmul" => true,

        // Read/write
        "read_io" => true,
        "write_io" => true,

        _ => false,
    }
}

pub fn all_instructions_without_args() -> Vec<Instruction> {
    let all_instructions: [_; Instruction::COUNT] = [
        Pop,
        Push(Default::default()),
        Divine(None),
        Dup(Default::default()),
        Swap(Default::default()),
        Nop,
        Skiz,
        Call(Default::default()),
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
    ];
    all_instructions.to_vec()
}

pub fn all_labelled_instructions_with_args<'a>() -> Vec<LabelledInstruction<'a>> {
    vec![
        Pop,
        Push(BFieldElement::new(42)),
        Divine(None),
        Dup(ST0),
        Dup(ST1),
        Dup(ST2),
        Dup(ST3),
        Dup(ST4),
        Dup(ST5),
        Dup(ST6),
        Dup(ST7),
        Dup(ST8),
        Dup(ST9),
        Dup(ST10),
        Dup(ST11),
        Dup(ST12),
        Dup(ST13),
        Dup(ST14),
        Dup(ST15),
        Swap(ST1),
        Swap(ST2),
        Swap(ST3),
        Swap(ST4),
        Swap(ST5),
        Swap(ST6),
        Swap(ST7),
        Swap(ST8),
        Swap(ST9),
        Swap(ST10),
        Swap(ST11),
        Swap(ST12),
        Swap(ST13),
        Swap(ST14),
        Swap(ST15),
        Nop,
        Skiz,
        Call("foo".to_string()),
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
        Split,
        Eq,
        XxAdd,
        XxMul,
        XInvert,
        XbMul,
        ReadIo,
        WriteIo,
    ]
    .into_iter()
    .map(|instr| LabelledInstruction::Instruction(instr, ""))
    .collect()
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

    pub const ALL_INSTRUCTIONS: &str = "
        pop
        push 42
        divine

        dup0 dup1 dup2 dup3 dup4 dup5 dup6 dup7 dup8 dup9 dup10 dup11 dup12 dup13 dup14 dup15
        swap1 swap2 swap3 swap4 swap5 swap6 swap7 swap8 swap9 swap10 swap11 swap12 swap13 swap14 swap15

        nop
        skiz
        call foo

        return recurse assert halt read_mem write_mem hash divine_sibling assert_vector
        absorb_init absorb squeeze
        add mul invert split eq xxadd xxmul xinvert xbmul

        read_io write_io
    ";

    pub fn all_instructions_displayed() -> Vec<String> {
        vec![
            "pop",
            "push 42",
            "divine",
            "dup0",
            "dup1",
            "dup2",
            "dup3",
            "dup4",
            "dup5",
            "dup6",
            "dup7",
            "dup8",
            "dup9",
            "dup10",
            "dup11",
            "dup12",
            "dup13",
            "dup14",
            "dup15",
            "swap1",
            "swap2",
            "swap3",
            "swap4",
            "swap5",
            "swap6",
            "swap7",
            "swap8",
            "swap9",
            "swap10",
            "swap11",
            "swap12",
            "swap13",
            "swap14",
            "swap15",
            "nop",
            "skiz",
            "call foo",
            "return",
            "recurse",
            "assert",
            "halt",
            "read_mem",
            "write_mem",
            "hash",
            "divine_sibling",
            "assert_vector",
            "absorb_init",
            "absorb",
            "squeeze",
            "add",
            "mul",
            "invert",
            "split",
            "eq",
            "xxadd",
            "xxmul",
            "xinvert",
            "xbmul",
            "read_io",
            "write_io",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }
}

#[cfg(test)]
mod instruction_tests {
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

        for instruction in all_instructions_without_args() {
            let all_ibs: [Ord8; Ord8::COUNT] = [IB0, IB1, IB2, IB3, IB4, IB5, IB6, IB7];
            for ib in all_ibs {
                let ib_value = instruction.ib(ib);
                assert!(
                    ib_value.is_zero() || ib_value.is_one(),
                    "IB{} for instruction {} is 0 or 1 ({})",
                    ib,
                    instruction,
                    ib_value
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
