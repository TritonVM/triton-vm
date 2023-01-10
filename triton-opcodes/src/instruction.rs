use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;
use std::fmt::Display;
use std::ops::Neg;
use std::str::SplitWhitespace;
use std::vec;

use anyhow::bail;
use anyhow::Result;
use itertools::Itertools;
use num_traits::One;
use num_traits::Zero;
use regex::Regex;
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
use crate::ord_n::Ord7;

/// An `Instruction` has `call` addresses encoded as absolute integers.
pub type Instruction = AnInstruction<BFieldElement>;

/// A `LabelledInstruction` has `call` addresses encoded as label names.
///
/// A label name is a `String` that occurs as "`label_name`:".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LabelledInstruction {
    Instruction(AnInstruction<String>),
    Label(String),
}

impl Display for LabelledInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LabelledInstruction::Instruction(instr) => write!(f, "{}", instr),
            LabelledInstruction::Label(label_name) => write!(f, "{}:", label_name),
        }
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
            Add => 26,
            Mul => 34,
            Invert => 80,
            Eq => 42,
            Split => 4,
            Lt => 12,
            And => 20,
            Xor => 28,
            Log2Floor => 36,
            Pow => 44,
            Div => 52,
            XxAdd => 88,
            XxMul => 96,
            XInvert => 104,
            XbMul => 50,
            ReadIo => 112,
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
    pub fn ib(&self, arg: Ord7) -> BFieldElement {
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
                let label_not_found = format!("Label not found: {}", label_name);
                let absolute_address = label_map.get(label_name).expect(&label_not_found);
                BFieldElement::new(*absolute_address as u64)
            });

            vec![unlabelled_instruction]
        }
    }
}

pub fn parse(code_with_comments: &str) -> Result<Vec<LabelledInstruction>> {
    let remove_comments = Regex::new(r"//.*?(?:\n|$)").expect("a regex that matches comments");
    let code = remove_comments.replace_all(code_with_comments, "");
    let mut tokens = code.split_whitespace();
    let mut instructions = vec![];

    while let Some(token) = tokens.next() {
        let mut instruction = parse_token(token, &mut tokens)?;
        instructions.append(&mut instruction);
    }

    let all_labels: Vec<String> = instructions
        .iter()
        .flat_map(|instr| match instr {
            LabelledInstruction::Instruction(_) => vec![],
            LabelledInstruction::Label(label) => vec![label.clone()],
        })
        .collect();
    let mut seen_labels: HashSet<String> = HashSet::default();
    let mut duplicate_labels: HashSet<String> = HashSet::default();
    for label in all_labels {
        if !seen_labels.insert(label.clone()) {
            duplicate_labels.insert(label);
        }
    }

    if !duplicate_labels.is_empty() {
        bail!("Duplicate labels: {}", duplicate_labels.iter().join(", "));
    }

    Ok(instructions)
}

fn parse_token(token: &str, tokens: &mut SplitWhitespace) -> Result<Vec<LabelledInstruction>> {
    if let Some(label) = token.strip_suffix(':') {
        let label_name = label.to_string();
        return Ok(vec![LabelledInstruction::Label(label_name)]);
    }

    let instruction: Vec<AnInstruction<String>> = match token {
        // OpStack manipulation
        "pop" => vec![Pop],
        "push" => vec![Push(parse_elem(tokens)?)],
        "divine" => vec![Divine(None)],
        "dup0" => vec![Dup(ST0)],
        "dup1" => vec![Dup(ST1)],
        "dup2" => vec![Dup(ST2)],
        "dup3" => vec![Dup(ST3)],
        "dup4" => vec![Dup(ST4)],
        "dup5" => vec![Dup(ST5)],
        "dup6" => vec![Dup(ST6)],
        "dup7" => vec![Dup(ST7)],
        "dup8" => vec![Dup(ST8)],
        "dup9" => vec![Dup(ST9)],
        "dup10" => vec![Dup(ST10)],
        "dup11" => vec![Dup(ST11)],
        "dup12" => vec![Dup(ST12)],
        "dup13" => vec![Dup(ST13)],
        "dup14" => vec![Dup(ST14)],
        "dup15" => vec![Dup(ST15)],
        "swap1" => vec![Swap(ST1)],
        "swap2" => vec![Swap(ST2)],
        "swap3" => vec![Swap(ST3)],
        "swap4" => vec![Swap(ST4)],
        "swap5" => vec![Swap(ST5)],
        "swap6" => vec![Swap(ST6)],
        "swap7" => vec![Swap(ST7)],
        "swap8" => vec![Swap(ST8)],
        "swap9" => vec![Swap(ST9)],
        "swap10" => vec![Swap(ST10)],
        "swap11" => vec![Swap(ST11)],
        "swap12" => vec![Swap(ST12)],
        "swap13" => vec![Swap(ST13)],
        "swap14" => vec![Swap(ST14)],
        "swap15" => vec![Swap(ST15)],

        // Control flow
        "nop" => vec![Nop],
        "skiz" => vec![Skiz],
        "call" => vec![Call(parse_label(tokens)?)],
        "return" => vec![Return],
        "recurse" => vec![Recurse],
        "assert" => vec![Assert],
        "halt" => vec![Halt],

        // Memory access
        "read_mem" => vec![ReadMem],
        "write_mem" => vec![WriteMem],

        // Hashing-related
        "hash" => vec![Hash],
        "divine_sibling" => vec![DivineSibling],
        "assert_vector" => vec![AssertVector],

        // Base field arithmetic on stack
        "add" => vec![Add],
        "mul" => vec![Mul],
        "invert" => vec![Invert],
        "eq" => vec![Eq],

        // Bitwise arithmetic on stack
        "split" => vec![Split],
        "lt" => vec![Lt],
        "and" => vec![And],
        "xor" => vec![Xor],
        "log_2_floor" => vec![Log2Floor],
        "pow" => vec![Pow],
        "div" => vec![Div],

        // Extension field arithmetic on stack
        "xxadd" => vec![XxAdd],
        "xxmul" => vec![XxMul],
        "xinvert" => vec![XInvert],
        "xbmul" => vec![XbMul],

        // Read/write
        "read_io" => vec![ReadIo],
        "write_io" => vec![WriteIo],

        // pseudo instructions
        "neg" => vec![Push(BFieldElement::one().neg()), Mul],
        "sub" => vec![Swap(ST1), Push(BFieldElement::one().neg()), Mul, Add],
        "lsb" => pseudo_instruction_lsb(),
        "is_u32" => pseudo_instruction_is_u32(),

        _ => return Err(anyhow::Error::new(UnknownInstruction(token.to_string()))),
    };

    let labelled_instruction = instruction
        .into_iter()
        .map(LabelledInstruction::Instruction)
        .collect();

    Ok(labelled_instruction)
}

fn pseudo_instruction_lsb() -> Vec<AnInstruction<String>> {
    // input stack: _ a
    vec![
        Push(BFieldElement::new(2)), // _ a 2
        Swap(ST1),                   // _ 2 a
        Div,                         // _ a/2 a%2
    ]
}

fn pseudo_instruction_is_u32() -> Vec<AnInstruction<String>> {
    // input stack: _ a
    vec![
        Dup(ST0),                    // _ a a
        Split,                       // _ a lo hi
        Push(BFieldElement::zero()), // _ a lo hi 0
        Eq,                          // _ a lo (hi==0)
        Swap(ST2),                   // _ (hi==0) lo a
        Eq,                          // _ (hi==0) (lo==a)
        Mul,                         // _ (hi==0 & lo==a)
    ]
}

fn parse_elem(tokens: &mut SplitWhitespace) -> Result<BFieldElement> {
    let constant_s = tokens.next().ok_or(UnexpectedEndOfStream)?;

    let mut constant_n128: i128 = constant_s.parse::<i128>()?;
    if constant_n128 < 0 {
        constant_n128 += BFieldElement::QUOTIENT as i128;
    }
    let constant_n64: u64 = constant_n128.try_into()?;
    let constant_elem = BFieldElement::new(constant_n64);

    Ok(constant_elem)
}

fn parse_label(tokens: &mut SplitWhitespace) -> Result<String> {
    let label = tokens
        .next()
        .map(|s| s.to_string())
        .ok_or(UnexpectedEndOfStream)?;

    Ok(label)
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

pub fn all_labelled_instructions_with_args() -> Vec<LabelledInstruction> {
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
    .map(LabelledInstruction::Instruction)
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

    use crate::instruction::all_labelled_instructions_with_args;
    use crate::ord_n::Ord7;
    use crate::program::Program;

    use super::all_instructions_without_args;
    use super::parse;
    use super::sample_programs;
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
            num_bits <= Ord7::COUNT,
            "Biggest instruction needs more than {} bits :(",
            Ord7::COUNT
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
    fn parse_and_display_each_instruction_test() {
        let expected = all_labelled_instructions_with_args();
        let actual = parse(sample_programs::ALL_INSTRUCTIONS).unwrap();

        assert_eq!(expected, actual);

        for (actual, expected) in actual
            .iter()
            .map(|instr| format!("{}", instr))
            .zip_eq(sample_programs::all_instructions_displayed())
        {
            assert_eq!(expected, actual);
        }
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
        use Ord7::*;

        for instruction in all_instructions_without_args() {
            for ib in [IB0, IB1, IB2, IB3, IB4, IB5, IB6] {
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
