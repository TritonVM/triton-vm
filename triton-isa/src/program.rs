use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::hash_map::Entry;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::hash::Hash;
use std::io::Cursor;

use arbitrary::Arbitrary;
use get_size2::GetSize;
use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;
use thiserror::Error;
use twenty_first::prelude::*;

use crate::instruction::AnInstruction;
use crate::instruction::AssertionContext;
use crate::instruction::Instruction;
use crate::instruction::InstructionError;
use crate::instruction::LabelledInstruction;
use crate::instruction::TypeHint;
use crate::parser;
use crate::parser::ParseError;

/// A program for Triton VM. Triton VM can run and profile such programs,
/// and trace its execution in order to generate a proof of correct execution.
/// See there for details.
///
/// A program may contain debug information, such as label names and
/// breakpoints. Access this information through methods
/// [`label_for_address()`][label_for_address] and
/// [`is_breakpoint()`][is_breakpoint]. Some operations, most notably
/// [BField-encoding](BFieldCodec::encode), discard this debug information.
///
/// [program attestation]: https://triton-vm.org/spec/program-attestation.html
/// [label_for_address]: Program::label_for_address
/// [is_breakpoint]: Program::is_breakpoint
#[derive(Debug, Clone, Eq, Serialize, Deserialize, GetSize)]
pub struct Program {
    pub instructions: Vec<Instruction>,
    address_to_label: HashMap<u64, String>,
    debug_information: DebugInformation,
}

impl Display for Program {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        for instruction in self.labelled_instructions() {
            writeln!(f, "{instruction}")?;
        }
        Ok(())
    }
}

impl PartialEq for Program {
    fn eq(&self, other: &Program) -> bool {
        self.instructions.eq(&other.instructions)
    }
}

impl BFieldCodec for Program {
    type Error = ProgramDecodingError;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>, Self::Error> {
        if sequence.is_empty() {
            return Err(Self::Error::EmptySequence);
        }
        let program_length = sequence[0].value() as usize;
        let sequence = &sequence[1..];
        if sequence.len() < program_length {
            return Err(Self::Error::SequenceTooShort);
        }
        if sequence.len() > program_length {
            return Err(Self::Error::SequenceTooLong);
        }

        // instantiating with claimed capacity is a potential DOS vector
        let mut instructions = vec![];
        let mut read_idx = 0;
        while read_idx < program_length {
            let opcode = sequence[read_idx];
            let mut instruction = Instruction::try_from(opcode)
                .map_err(|err| Self::Error::InvalidInstruction(read_idx, err))?;
            let instruction_has_arg = instruction.arg().is_some();
            if instruction_has_arg && instructions.len() + instruction.size() > program_length {
                return Err(Self::Error::MissingArgument(read_idx, instruction));
            }
            if instruction_has_arg {
                let arg = sequence[read_idx + 1];
                instruction = instruction
                    .change_arg(arg)
                    .map_err(|err| Self::Error::InvalidInstruction(read_idx, err))?;
            }

            instructions.extend(vec![instruction; instruction.size()]);
            read_idx += instruction.size();
        }

        if read_idx != program_length {
            return Err(Self::Error::LengthMismatch);
        }
        if instructions.len() != program_length {
            return Err(Self::Error::LengthMismatch);
        }

        Ok(Box::new(Program {
            instructions,
            address_to_label: HashMap::default(),
            debug_information: DebugInformation::default(),
        }))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut sequence = Vec::with_capacity(self.len_bwords() + 1);
        sequence.push(bfe!(self.len_bwords() as u64));
        sequence.extend(self.to_bwords());
        sequence
    }

    fn static_length() -> Option<usize> {
        None
    }
}

impl<'a> Arbitrary<'a> for Program {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let contains_label = |labelled_instructions: &[_], maybe_label: &_| {
            let LabelledInstruction::Label(label) = maybe_label else {
                return false;
            };
            labelled_instructions
                .iter()
                .any(|labelled_instruction| match labelled_instruction {
                    LabelledInstruction::Label(l) => l == label,
                    _ => false,
                })
        };
        let is_assertion = |maybe_instruction: &_| {
            matches!(
                maybe_instruction,
                LabelledInstruction::Instruction(
                    AnInstruction::Assert | AnInstruction::AssertVector
                )
            )
        };

        let mut labelled_instructions = vec![];
        for _ in 0..u.arbitrary_len::<LabelledInstruction>()? {
            let labelled_instruction = u.arbitrary()?;
            if contains_label(&labelled_instructions, &labelled_instruction) {
                continue;
            }
            if let LabelledInstruction::AssertionContext(_) = labelled_instruction {
                // assertion context must come after an assertion
                continue;
            }

            let is_assertion = is_assertion(&labelled_instruction);
            labelled_instructions.push(labelled_instruction);

            if is_assertion && u.arbitrary()? {
                let assertion_context = LabelledInstruction::AssertionContext(u.arbitrary()?);
                labelled_instructions.push(assertion_context);
            }
        }

        let all_call_targets = labelled_instructions
            .iter()
            .filter_map(|instruction| match instruction {
                LabelledInstruction::Instruction(AnInstruction::Call(target)) => Some(target),
                _ => None,
            })
            .unique();
        let labels_that_are_called_but_not_declared = all_call_targets
            .map(|target| LabelledInstruction::Label(target.clone()))
            .filter(|label| !contains_label(&labelled_instructions, label))
            .collect_vec();

        for label in labels_that_are_called_but_not_declared {
            let insertion_index = u.choose_index(labelled_instructions.len() + 1)?;
            labelled_instructions.insert(insertion_index, label);
        }

        Ok(Program::new(&labelled_instructions))
    }
}

/// An `InstructionIter` loops the instructions of a `Program` by skipping
/// duplicate placeholders.
#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub struct InstructionIter {
    cursor: Cursor<Vec<Instruction>>,
}

impl Iterator for InstructionIter {
    type Item = Instruction;

    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.cursor.position() as usize;
        let instructions = self.cursor.get_ref();
        let instruction = *instructions.get(pos)?;
        self.cursor.set_position((pos + instruction.size()) as u64);

        Some(instruction)
    }
}

impl IntoIterator for Program {
    type Item = Instruction;

    type IntoIter = InstructionIter;

    fn into_iter(self) -> Self::IntoIter {
        let cursor = Cursor::new(self.instructions);
        InstructionIter { cursor }
    }
}

#[derive(Debug, Default, Clone, Eq, PartialEq, Serialize, Deserialize, Arbitrary, GetSize)]
struct DebugInformation {
    breakpoints: Vec<bool>,
    type_hints: HashMap<u64, Vec<TypeHint>>,
    assertion_context: HashMap<u64, AssertionContext>,
}

impl Program {
    pub fn new(labelled_instructions: &[LabelledInstruction]) -> Self {
        let label_to_address = parser::build_label_to_address_map(labelled_instructions);
        let instructions =
            parser::turn_labels_into_addresses(labelled_instructions, &label_to_address);
        let address_to_label = Self::flip_map(label_to_address);
        let debug_information = Self::extract_debug_information(labelled_instructions);

        debug_assert_eq!(instructions.len(), debug_information.breakpoints.len());
        Program {
            instructions,
            address_to_label,
            debug_information,
        }
    }

    fn flip_map<Key, Value: Eq + Hash>(map: HashMap<Key, Value>) -> HashMap<Value, Key> {
        map.into_iter().map(|(key, value)| (value, key)).collect()
    }

    fn extract_debug_information(
        labelled_instructions: &[LabelledInstruction],
    ) -> DebugInformation {
        let mut address = 0;
        let mut break_before_next_instruction = false;
        let mut debug_info = DebugInformation::default();
        for instruction in labelled_instructions {
            match instruction {
                LabelledInstruction::Instruction(instruction) => {
                    let new_breakpoints = vec![break_before_next_instruction; instruction.size()];
                    debug_info.breakpoints.extend(new_breakpoints);
                    break_before_next_instruction = false;
                    address += instruction.size() as u64;
                }
                LabelledInstruction::Label(_) => (),
                LabelledInstruction::Breakpoint => break_before_next_instruction = true,
                LabelledInstruction::TypeHint(hint) => match debug_info.type_hints.entry(address) {
                    Entry::Occupied(mut entry) => entry.get_mut().push(hint.clone()),
                    Entry::Vacant(entry) => entry.insert(vec![]).push(hint.clone()),
                },
                LabelledInstruction::AssertionContext(ctx) => {
                    let address_of_associated_assertion = address.saturating_sub(1);
                    debug_info
                        .assertion_context
                        .insert(address_of_associated_assertion, ctx.clone());
                }
            }
        }

        debug_info
    }

    /// Create a `Program` by parsing source code.
    pub fn from_code(code: &str) -> Result<Self, ParseError<'_>> {
        parser::parse(code)
            .map(|tokens| parser::to_labelled_instructions(&tokens))
            .map(|instructions| Program::new(&instructions))
    }

    pub fn labelled_instructions(&self) -> Vec<LabelledInstruction> {
        let call_targets = self.call_targets();
        let instructions_with_labels = self.instructions.iter().map(|instruction| {
            instruction.map_call_address(|&address| self.label_for_address(address.value()))
        });

        let mut labelled_instructions = vec![];
        let mut address = 0;
        let mut instruction_stream = instructions_with_labels.into_iter();
        while let Some(instruction) = instruction_stream.next() {
            let instruction_size = instruction.size() as u64;
            if call_targets.contains(&address) {
                let label = self.label_for_address(address);
                let label = LabelledInstruction::Label(label);
                labelled_instructions.push(label);
            }
            for type_hint in self.type_hints_at(address) {
                labelled_instructions.push(LabelledInstruction::TypeHint(type_hint));
            }
            if self.is_breakpoint(address) {
                labelled_instructions.push(LabelledInstruction::Breakpoint);
            }
            labelled_instructions.push(LabelledInstruction::Instruction(instruction));
            if let Some(context) = self.assertion_context_at(address) {
                labelled_instructions.push(LabelledInstruction::AssertionContext(context));
            }

            for _ in 1..instruction_size {
                instruction_stream.next();
            }
            address += instruction_size;
        }

        let leftover_labels = self
            .address_to_label
            .iter()
            .filter(|&(&labels_address, _)| labels_address >= address)
            .sorted();
        for (_, label) in leftover_labels {
            labelled_instructions.push(LabelledInstruction::Label(label.clone()));
        }

        labelled_instructions
    }

    fn call_targets(&self) -> HashSet<u64> {
        self.instructions
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::Call(address) => Some(address.value()),
                _ => None,
            })
            .collect()
    }

    pub fn is_breakpoint(&self, address: u64) -> bool {
        let address: usize = address.try_into().unwrap();
        self.debug_information
            .breakpoints
            .get(address)
            .copied()
            .unwrap_or_default()
    }

    pub fn type_hints_at(&self, address: u64) -> Vec<TypeHint> {
        self.debug_information
            .type_hints
            .get(&address)
            .cloned()
            .unwrap_or_default()
    }

    pub fn assertion_context_at(&self, address: u64) -> Option<AssertionContext> {
        self.debug_information
            .assertion_context
            .get(&address)
            .cloned()
    }

    /// Turn the program into a sequence of `BFieldElement`s. Each instruction
    /// is encoded as its opcode, followed by its argument (if any).
    ///
    /// **Note**: This is _almost_ (but not quite!) equivalent to
    /// [encoding](BFieldCodec::encode) the program. For that, use
    /// [`encode()`](Self::encode()) instead.
    pub fn to_bwords(&self) -> Vec<BFieldElement> {
        self.clone()
            .into_iter()
            .flat_map(|instruction| {
                let opcode = instruction.opcode_b();
                if let Some(arg) = instruction.arg() {
                    vec![opcode, arg]
                } else {
                    vec![opcode]
                }
            })
            .collect()
    }

    /// The total length of the program as `BFieldElement`s. Double-word
    /// instructions contribute two `BFieldElement`s.
    pub fn len_bwords(&self) -> usize {
        self.instructions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    /// Produces the program's canonical hash digest. Uses [`Tip5`], the
    /// canonical hash function for Triton VM.
    pub fn hash(&self) -> Digest {
        // not encoded using `BFieldCodec` because that would prepend the length
        Tip5::hash_varlen(&self.to_bwords())
    }

    /// The label for the given address, or a deterministic, unique substitute
    /// if no label is found.
    pub fn label_for_address(&self, address: u64) -> String {
        // Uniqueness of the label is relevant for printing and subsequent
        // parsing: Parsing fails on duplicate labels.
        self.address_to_label
            .get(&address)
            .cloned()
            .unwrap_or_else(|| format!("address_{address}"))
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Eq, PartialEq, Error)]
pub enum ProgramDecodingError {
    #[error("sequence to decode is empty")]
    EmptySequence,

    #[error("sequence to decode is too short")]
    SequenceTooShort,

    #[error("sequence to decode is too long")]
    SequenceTooLong,

    #[error("length of decoded program is unexpected")]
    LengthMismatch,

    #[error("sequence to decode contains invalid instruction at index {0}: {1}")]
    InvalidInstruction(usize, InstructionError),

    #[error("missing argument for instruction {1} at index {0}")]
    MissingArgument(usize, Instruction),
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use assert2::assert;
    use assert2::let_assert;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use rand::Rng;
    use test_strategy::proptest;

    use crate::triton_program;

    use super::*;

    #[proptest]
    fn random_program_encode_decode_equivalence(#[strategy(arb())] program: Program) {
        let encoding = program.encode();
        let decoding = *Program::decode(&encoding).unwrap();
        prop_assert_eq!(program, decoding);
    }

    #[test]
    fn decode_program_with_missing_argument_as_last_instruction() {
        let program = triton_program!(push 3 push 3 eq assert push 3);
        let program_length = program.len_bwords() as u64;
        let encoded = program.encode();

        let mut encoded = encoded[0..encoded.len() - 1].to_vec();
        encoded[0] = bfe!(program_length - 1);

        let_assert!(Err(err) = Program::decode(&encoded));
        let_assert!(ProgramDecodingError::MissingArgument(6, _) = err);
    }

    #[test]
    fn decode_program_with_shorter_than_indicated_sequence() {
        let program = triton_program!(nop nop hash push 0 skiz end: halt call end);
        let mut encoded = program.encode();
        encoded[0] += bfe!(1);
        let_assert!(Err(err) = Program::decode(&encoded));
        let_assert!(ProgramDecodingError::SequenceTooShort = err);
    }

    #[test]
    fn decode_program_with_longer_than_indicated_sequence() {
        let program = triton_program!(nop nop hash push 0 skiz end: halt call end);
        let mut encoded = program.encode();
        encoded[0] -= bfe!(1);
        let_assert!(Err(err) = Program::decode(&encoded));
        let_assert!(ProgramDecodingError::SequenceTooLong = err);
    }

    #[test]
    fn decode_program_from_empty_sequence() {
        let encoded = vec![];
        let_assert!(Err(err) = Program::decode(&encoded));
        let_assert!(ProgramDecodingError::EmptySequence = err);
    }

    #[test]
    fn hash_simple_program() {
        let program = triton_program!(halt);
        let digest = program.hash();

        let expected_digest = bfe_array![
            0x4338_de79_520b_3949_u64,
            0xe6a2_129b_2885_0dc9_u64,
            0xfd3c_d098_6a86_0450_u64,
            0x69fd_ba91_0ceb_a7bc_u64,
            0x7e5b_118c_9594_c062_u64,
        ];
        let expected_digest = Digest::new(expected_digest);

        assert!(expected_digest == digest);
    }

    #[test]
    fn empty_program_is_empty() {
        let program = triton_program!();
        assert!(program.is_empty());
    }

    #[test]
    fn create_program_from_code() {
        let element_3 = rand::rng().random_range(0..BFieldElement::P);
        let element_2 = 1337_usize;
        let element_1 = "17";
        let element_0 = bfe!(0);
        let instruction_push = Instruction::Push(bfe!(42));
        let dup_arg = 1;
        let label = "my_label".to_string();

        let source_code = format!(
            "push {element_3} push {element_2} push {element_1} push {element_0}
             call {label} halt
             {label}:
                {instruction_push}
                dup {dup_arg}
                skiz
                recurse
                return"
        );
        let program_from_code = Program::from_code(&source_code).unwrap();
        let program_from_macro = triton_program!({ source_code });
        assert!(program_from_code == program_from_macro);
    }

    #[test]
    fn parser_macro_with_interpolated_label_as_first_argument() {
        let label = "my_label";
        let _program = triton_program!(
            {label}: push 1 assert halt
        );
    }

    #[test]
    fn breakpoints_propagate_to_debug_information_as_expected() {
        let program = triton_program! {
            break push 1 push 2
            break break break break
            pop 2 hash halt
            break // no effect
        };

        assert!(program.is_breakpoint(0));
        assert!(program.is_breakpoint(1));
        assert!(!program.is_breakpoint(2));
        assert!(!program.is_breakpoint(3));
        assert!(program.is_breakpoint(4));
        assert!(program.is_breakpoint(5));
        assert!(!program.is_breakpoint(6));
        assert!(!program.is_breakpoint(7));

        // going beyond the length of the program must not break things
        assert!(!program.is_breakpoint(8));
        assert!(!program.is_breakpoint(9));
    }

    #[test]
    fn print_program_without_any_debug_information() {
        let program = triton_program! {
            call foo
            call bar
            call baz
            halt
            foo: nop nop return
            bar: call baz return
            baz: push 1 return
        };
        let encoding = program.encode();
        let program = Program::decode(&encoding).unwrap();
        println!("{program}");
    }

    #[proptest]
    fn printed_program_can_be_parsed_again(#[strategy(arb())] program: Program) {
        parser::parse(&program.to_string())?;
    }

    struct TypeHintTestCase {
        expected: TypeHint,
        input: &'static str,
    }

    impl TypeHintTestCase {
        fn run(&self) {
            let program = Program::from_code(self.input).unwrap();
            let [ref type_hint] = program.type_hints_at(0)[..] else {
                panic!("Expected a single type hint at address 0");
            };
            assert!(&self.expected == type_hint);
        }
    }

    #[test]
    fn parse_simple_type_hint() {
        let expected = TypeHint {
            starting_index: 0,
            length: 1,
            type_name: Some("Type".to_string()),
            variable_name: "foo".to_string(),
        };

        TypeHintTestCase {
            expected,
            input: "hint foo: Type = stack[0]",
        }
        .run();
    }

    #[test]
    fn parse_type_hint_with_range() {
        let expected = TypeHint {
            starting_index: 0,
            length: 5,
            type_name: Some("Digest".to_string()),
            variable_name: "foo".to_string(),
        };

        TypeHintTestCase {
            expected,
            input: "hint foo: Digest = stack[0..5]",
        }
        .run();
    }

    #[test]
    fn parse_type_hint_with_range_and_offset() {
        let expected = TypeHint {
            starting_index: 7,
            length: 3,
            type_name: Some("XFieldElement".to_string()),
            variable_name: "bar".to_string(),
        };

        TypeHintTestCase {
            expected,
            input: "hint bar: XFieldElement = stack[7..10]",
        }
        .run();
    }

    #[test]
    fn parse_type_hint_with_range_and_offset_and_weird_whitespace() {
        let expected = TypeHint {
            starting_index: 2,
            length: 12,
            type_name: Some("BigType".to_string()),
            variable_name: "bar".to_string(),
        };

        TypeHintTestCase {
            expected,
            input: " hint \t \t bar  :BigType=stack[ 2\t.. 14 ]\t \n",
        }
        .run();
    }

    #[test]
    fn parse_type_hint_with_no_type_only_variable_name() {
        let expected = TypeHint {
            starting_index: 0,
            length: 1,
            type_name: None,
            variable_name: "foo".to_string(),
        };

        TypeHintTestCase {
            expected,
            input: "hint foo = stack[0]",
        }
        .run();
    }

    #[test]
    fn parse_type_hint_with_no_type_only_variable_name_with_range() {
        let expected = TypeHint {
            starting_index: 2,
            length: 5,
            type_name: None,
            variable_name: "foo".to_string(),
        };

        TypeHintTestCase {
            expected,
            input: "hint foo = stack[2..7]",
        }
        .run();
    }

    #[test]
    fn assertion_context_is_propagated_into_debug_info() {
        let program = triton_program! {push 1000 assert error_id 17 halt};
        //                              ↑0   ↑1   ↑2

        let assertion_contexts = program.debug_information.assertion_context;
        assert!(1 == assertion_contexts.len());
        let_assert!(AssertionContext::ID(error_id) = &assertion_contexts[&2]);
        assert!(17 == *error_id);
    }

    #[test]
    fn printing_program_includes_debug_information() {
        let source_code = "\
            call foo\n\
            break\n\
            call bar\n\
            halt\n\
            foo:\n\
            break\n\
            call baz\n\
            push 1\n\
            nop\n\
            return\n\
            baz:\n\
            hash\n\
            hint my_digest: Digest = stack[0..5]\n\
            hint random_stuff = stack[17]\n\
            return\n\
            nop\n\
            pop 1\n\
            bar:\n\
            divine 1\n\
            hint got_insight: Magic = stack[0]\n\
            skiz\n\
            split\n\
            break\n\
            assert\n\
            error_id 1337\n\
            return\n\
        ";
        let program = Program::from_code(source_code).unwrap();
        let printed_program = format!("{program}");
        assert_eq!(source_code, &printed_program);
    }
}
