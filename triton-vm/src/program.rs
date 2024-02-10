use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::hash::Hash;
use std::io::Cursor;

use arbitrary::Arbitrary;
use get_size::GetSize;
use itertools::Itertools;
use serde_derive::Deserialize;
use serde_derive::Serialize;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::error::ProgramDecodingError;
use crate::error::VMError;
use crate::instruction::AnInstruction;
use crate::instruction::Instruction;
use crate::instruction::LabelledInstruction;
use crate::instruction::TypeHint;
use crate::parser::parse;
use crate::parser::to_labelled_instructions;
use crate::parser::ParseError;
use crate::vm::VMState;

type Result<T> = std::result::Result<T, VMError>;

/// A program for Triton VM.
/// It can be
/// [`run`](Program::run),
/// [`profiled`](Program::profile),
/// and its execution can be [`traced`](Program::trace_execution).
///
/// [`Hashing`](Program::hash) a program under [`Tip5`] yields a [`Digest`] that can be used
/// in a [`Claim`](crate::Claim), _i.e._, is consistent with Triton VM's [program attestation].
///
/// A program may contain debug information, such as label names and breakpoints.
/// Access this information through methods [`label_for_address()`][label_for_address] and
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
    breakpoints: Vec<bool>,
    type_hints: HashMap<u64, Vec<TypeHint>>,
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

    fn decode(sequence: &[BFieldElement]) -> std::result::Result<Box<Self>, Self::Error> {
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
            if instruction.has_arg() && instructions.len() + instruction.size() > program_length {
                return Err(Self::Error::MissingArgument(read_idx, instruction));
            }
            if instruction.has_arg() {
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
            breakpoints: vec![],
            type_hints: HashMap::default(),
        }))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut sequence = Vec::with_capacity(self.len_bwords() + 1);
        sequence.push(BFieldElement::new(self.len_bwords() as u64));
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

        let mut labelled_instructions = vec![];
        for _ in 0..u.arbitrary_len::<LabelledInstruction>()? {
            let labelled_instruction = u.arbitrary()?;
            if contains_label(&labelled_instructions, &labelled_instruction) {
                continue;
            }
            labelled_instructions.push(labelled_instruction);
        }

        let call_targets = labelled_instructions
            .iter()
            .filter_map(|instruction| match instruction {
                LabelledInstruction::Instruction(AnInstruction::Call(target)) => Some(target),
                _ => None,
            })
            .unique();
        let additional_labels = call_targets
            .map(|target| LabelledInstruction::Label(target.clone()))
            .collect_vec();

        for additional_label in additional_labels {
            if contains_label(&labelled_instructions, &additional_label) {
                continue;
            }
            let insertion_index = u.choose_index(labelled_instructions.len() + 1)?;
            labelled_instructions.insert(insertion_index, additional_label);
        }

        Ok(Program::new(&labelled_instructions))
    }
}

/// An `InstructionIter` loops the instructions of a `Program` by skipping duplicate placeholders.
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

impl Program {
    /// Create a `Program` from a slice of `LabelledInstruction`s.
    pub fn new(labelled_instructions: &[LabelledInstruction]) -> Self {
        let label_to_address = Self::build_label_to_address_map(labelled_instructions);
        let instructions =
            Self::turn_labels_into_addresses(labelled_instructions, &label_to_address);
        let address_to_label = Self::flip_map(label_to_address);
        let (breakpoints, type_hints) = Self::extract_debug_information(labelled_instructions);

        assert_eq!(instructions.len(), breakpoints.len());
        Program {
            instructions,
            address_to_label,
            breakpoints,
            type_hints,
        }
    }

    fn build_label_to_address_map(program: &[LabelledInstruction]) -> HashMap<String, u64> {
        use LabelledInstruction::*;

        let mut label_map = HashMap::new();
        let mut instruction_pointer = 0;

        for labelled_instruction in program {
            match labelled_instruction {
                Label(label) => match label_map.entry(label.clone()) {
                    Entry::Occupied(_) => panic!("Duplicate label: {label}"),
                    Entry::Vacant(entry) => _ = entry.insert(instruction_pointer),
                },
                Instruction(instruction) => instruction_pointer += instruction.size() as u64,
                _ => (),
            }
        }
        label_map
    }

    fn turn_labels_into_addresses(
        labelled_instructions: &[LabelledInstruction],
        label_to_address: &HashMap<String, u64>,
    ) -> Vec<Instruction> {
        labelled_instructions
            .iter()
            .flat_map(|instr| Self::turn_label_to_address_for_instruction(instr, label_to_address))
            .flat_map(|instr| vec![instr; instr.size()])
            .collect()
    }

    fn turn_label_to_address_for_instruction(
        labelled_instruction: &LabelledInstruction,
        label_map: &HashMap<String, u64>,
    ) -> Option<Instruction> {
        let LabelledInstruction::Instruction(instruction) = labelled_instruction else {
            return None;
        };

        let instruction_with_absolute_address = instruction.map_call_address(|label| {
            label_map
                .get(label)
                .map(|&address| BFieldElement::new(address))
                .unwrap_or_else(|| panic!("Label not found: {label}"))
        });
        Some(instruction_with_absolute_address)
    }

    fn flip_map<Key, Value: Eq + Hash>(map: HashMap<Key, Value>) -> HashMap<Value, Key> {
        map.into_iter().map(|(key, value)| (value, key)).collect()
    }

    fn extract_debug_information(
        labelled_instructions: &[LabelledInstruction],
    ) -> (Vec<bool>, HashMap<u64, Vec<TypeHint>>) {
        let mut breakpoints = vec![];
        let mut type_hints = HashMap::<_, Vec<_>>::new();
        let mut break_before_next_instruction = false;

        let mut address = 0;
        for instruction in labelled_instructions {
            match instruction {
                LabelledInstruction::Instruction(instruction) => {
                    breakpoints.extend(vec![break_before_next_instruction; instruction.size()]);
                    break_before_next_instruction = false;
                    address += instruction.size() as u64;
                }
                LabelledInstruction::Breakpoint => break_before_next_instruction = true,
                LabelledInstruction::TypeHint(type_hint) => match type_hints.entry(address) {
                    Entry::Occupied(mut entry) => entry.get_mut().push(type_hint.clone()),
                    Entry::Vacant(entry) => _ = entry.insert(vec![type_hint.clone()]),
                },
                _ => (),
            }
        }

        (breakpoints, type_hints)
    }

    /// Create a `Program` by parsing source code.
    pub fn from_code(code: &str) -> std::result::Result<Self, ParseError> {
        parse(code)
            .map(|tokens| to_labelled_instructions(&tokens))
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

            for _ in 1..instruction_size {
                instruction_stream.next();
            }
            address += instruction_size;
        }

        let leftover_labels = self
            .address_to_label
            .iter()
            .filter(|(&labels_address, _)| labels_address >= address)
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
        self.breakpoints.get(address).unwrap_or(&false).to_owned()
    }

    pub fn type_hints_at(&self, address: u64) -> Vec<TypeHint> {
        self.type_hints.get(&address).cloned().unwrap_or_default()
    }

    /// Turn the program into a sequence of `BFieldElement`s. Each instruction is encoded as its
    /// opcode, followed by its argument (if any).
    ///
    /// **Note**: This is _almost_ (but not quite!) equivalent to [encoding](BFieldCodec::encode)
    /// the program. For that, use [`encode()`](Self::encode()) instead.
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

    /// The total length of the program as `BFieldElement`s. Double-word instructions contribute
    /// two `BFieldElement`s.
    pub fn len_bwords(&self) -> usize {
        self.instructions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    /// Produces the program's canonical hash digest for the given `AlgebraicHasher`.
    pub fn hash<H: AlgebraicHasher>(&self) -> Digest {
        H::hash_varlen(&self.to_bwords())
    }

    /// Run Triton VM on the [`Program`] with the given public input and non-determinism.
    /// If an error is encountered, the returned [`VMError`] contains the [`VMState`] at the point
    /// of execution failure.
    ///
    /// See also [`trace_execution`][trace_execution] and [`profile`][profile].
    ///
    /// [trace_execution]: Self::trace_execution
    /// [profile]: Self::profile
    pub fn run(
        &self,
        public_input: PublicInput,
        non_determinism: NonDeterminism<BFieldElement>,
    ) -> Result<Vec<BFieldElement>> {
        let mut state = VMState::new(self, public_input, non_determinism);
        if let Err(err) = state.run() {
            return Err(VMError::new(err, state));
        }
        Ok(state.public_output)
    }

    /// Trace the execution of a [`Program`]. That is, [`run`][run] the [`Program`] and additionally
    /// record that part of every encountered state that is necessary for proving correct execution.
    /// If execution  succeeds, returns
    /// 1. an [`AlgebraicExecutionTrace`], and
    /// 1. the output of the program.
    ///
    /// See also [`run`][run] and [`profile`][profile].
    ///
    /// [run]: Self::run
    /// [profile]: Self::profile
    pub fn trace_execution(
        &self,
        public_input: PublicInput,
        non_determinism: NonDeterminism<BFieldElement>,
    ) -> Result<(AlgebraicExecutionTrace, Vec<BFieldElement>)> {
        let state = VMState::new(self, public_input, non_determinism);
        let (aet, terminal_state) = self.trace_execution_of_state(state)?;
        Ok((aet, terminal_state.public_output))
    }

    /// Trace the execution of a [`Program`] from a given [`VMState`]. Consider using
    /// [`trace_execution`][Self::trace_execution], unless you know this is what you want.
    ///
    /// Returns the [`AlgebraicExecutionTrace`] and the terminal [`VMState`] if execution succeeds.
    pub fn trace_execution_of_state(
        &self,
        mut state: VMState,
    ) -> Result<(AlgebraicExecutionTrace, VMState)> {
        let mut aet = AlgebraicExecutionTrace::new(self.clone());
        assert_eq!(self.instructions, state.program);
        assert_eq!(self.len_bwords(), aet.instruction_multiplicities.len());

        while !state.halting {
            if let Err(err) = aet.record_state(&state) {
                return Err(VMError::new(err, state));
            };
            let co_processor_calls = match state.step() {
                Ok(calls) => calls,
                Err(err) => return Err(VMError::new(err, state)),
            };
            for call in co_processor_calls {
                aet.record_co_processor_call(call);
            }
        }

        Ok((aet, state))
    }

    /// Run Triton VM with the given public and secret input, but record the number of cycles spent
    /// in each callable block of instructions. This function returns a Result wrapping a program
    /// profiler report, which is a Vec of [`ProfileLine`]s.
    ///
    /// See also [`run`][run] and [`trace_execution`][trace_execution].
    ///
    /// [run]: Self::run
    /// [trace_execution]: Self::trace_execution
    pub fn profile(
        &self,
        public_input: PublicInput,
        non_determinism: NonDeterminism<BFieldElement>,
    ) -> Result<(Vec<BFieldElement>, Vec<ProfileLine>)> {
        let mut profiler = VMProfiler::new();
        let mut state = VMState::new(self, public_input, non_determinism);
        while !state.halting {
            if let Ok(Instruction::Call(address)) = state.current_instruction() {
                let label = self.label_for_address(address.value());
                profiler.enter_span_with_label_at_cycle(label, state.cycle_count);
            }
            if let Ok(Instruction::Return) = state.current_instruction() {
                profiler.exit_span_at_cycle(state.cycle_count);
            }

            if let Err(err) = state.step() {
                return Err(VMError::new(err, state));
            };
        }

        let report = profiler.report_at_cycle(state.cycle_count);
        Ok((state.public_output, report))
    }

    /// The label for the given address, or a deterministic, unique substitute if no label is found.
    /// Uniqueness is relevant for printing and subsequent parsing, avoiding duplicate labels.
    pub fn label_for_address(&self, address: u64) -> String {
        let substitute_for_unknown_label = format!("address_{address}");
        self.address_to_label
            .get(&address)
            .unwrap_or(&substitute_for_unknown_label)
            .to_owned()
    }
}

#[derive(Debug, Default, Clone)]
struct VMProfiler {
    call_stack: Vec<usize>,
    profile: Vec<ProfileLine>,
}

impl VMProfiler {
    fn new() -> Self {
        VMProfiler {
            call_stack: vec![],
            profile: vec![],
        }
    }

    fn enter_span_with_label_at_cycle(&mut self, label: impl Into<String>, cycle: u32) {
        let call_stack_len = self.call_stack.len();
        let line_number = self.profile.len();

        let profile_line = ProfileLine::new(label, cycle).at_call_depth(call_stack_len);

        self.profile.push(profile_line);
        self.call_stack.push(line_number);
    }

    fn exit_span_at_cycle(&mut self, cycle: u32) {
        let maybe_top_of_call_stack = self.call_stack.pop();
        if let Some(line_number) = maybe_top_of_call_stack {
            self.profile[line_number].return_at_cycle(cycle);
        };
    }

    fn report_at_cycle(mut self, cycle: u32) -> Vec<ProfileLine> {
        self.stop_all_at_cycle(cycle);
        self.add_total(cycle);
        self.profile
    }

    fn stop_all_at_cycle(&mut self, cycle: u32) {
        for &line_number in &self.call_stack {
            self.profile[line_number].stop_at_cycle(cycle);
        }
    }

    fn add_total(&mut self, cycle: u32) {
        let mut line = ProfileLine::new("total", 0);
        line.return_at_cycle(cycle);
        self.profile.push(line);
    }
}

/// A single line in a profile report for profiling Triton Assembly programs.
#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub struct ProfileLine {
    pub label: String,
    pub call_depth: usize,
    pub start_cycle: u32,
    pub stop_cycle: u32,
    pub call_has_returned: bool,
}

impl ProfileLine {
    pub fn new(label: impl Into<String>, start_cycle: u32) -> Self {
        Self {
            label: label.into(),
            call_depth: 0,
            start_cycle,
            stop_cycle: 0,
            call_has_returned: false,
        }
    }

    pub fn at_call_depth(mut self, call_stack_depth: usize) -> Self {
        self.call_depth = call_stack_depth;
        self
    }

    pub fn return_at_cycle(&mut self, cycle: u32) {
        self.stop_at_cycle(cycle);
        self.call_has_returned = true;
    }

    pub fn stop_at_cycle(&mut self, cycle: u32) {
        self.stop_cycle = cycle;
    }

    pub fn cycle_count(&self) -> u32 {
        self.stop_cycle.saturating_sub(self.start_cycle)
    }
}

impl Display for ProfileLine {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let indentation = "  ".repeat(self.call_depth);
        let label = &self.label;
        let cycle_count = self.cycle_count();
        let open_indicator = match self.call_has_returned {
            true => "",
            false => " (open)",
        };
        write!(f, "{indentation}{label}{open_indicator}: {cycle_count}",)
    }
}

#[derive(Debug, Default, Clone, Eq, PartialEq, BFieldCodec, Arbitrary)]
pub struct PublicInput {
    pub individual_tokens: Vec<BFieldElement>,
}

impl From<Vec<BFieldElement>> for PublicInput {
    fn from(individual_tokens: Vec<BFieldElement>) -> Self {
        PublicInput { individual_tokens }
    }
}

impl From<&Vec<BFieldElement>> for PublicInput {
    fn from(tokens: &Vec<BFieldElement>) -> Self {
        PublicInput {
            individual_tokens: tokens.to_owned(),
        }
    }
}

impl From<&[BFieldElement]> for PublicInput {
    fn from(tokens: &[BFieldElement]) -> Self {
        PublicInput {
            individual_tokens: tokens.to_vec(),
        }
    }
}

impl From<Vec<u64>> for PublicInput {
    fn from(tokens: Vec<u64>) -> Self {
        PublicInput {
            individual_tokens: tokens.iter().map(|&element| element.into()).collect(),
        }
    }
}

impl From<[u64; 0]> for PublicInput {
    fn from(_tokens: [u64; 0]) -> Self {
        PublicInput {
            individual_tokens: vec![],
        }
    }
}

impl PublicInput {
    pub fn new(individual_tokens: Vec<BFieldElement>) -> Self {
        PublicInput { individual_tokens }
    }
}

/// All sources of non-determinism for a program. This includes elements that can be read using
/// instruction `divine`, digests that can be read using instruction `divine_sibling`,
/// and an initial state of random-access memory.
#[derive(Debug, Default, Clone, Eq, PartialEq, Serialize, Deserialize, Arbitrary)]
pub struct NonDeterminism<E>
where
    E: Into<BFieldElement> + Eq + Hash,
{
    pub individual_tokens: Vec<E>,
    pub digests: Vec<Digest>,
    pub ram: HashMap<E, E>,
}

impl From<Vec<BFieldElement>> for NonDeterminism<BFieldElement> {
    fn from(tokens: Vec<BFieldElement>) -> Self {
        NonDeterminism {
            individual_tokens: tokens,
            digests: vec![],
            ram: HashMap::new(),
        }
    }
}

impl From<&[BFieldElement]> for NonDeterminism<BFieldElement> {
    fn from(tokens: &[BFieldElement]) -> Self {
        NonDeterminism {
            individual_tokens: tokens.to_vec(),
            digests: vec![],
            ram: HashMap::new(),
        }
    }
}

impl From<Vec<u64>> for NonDeterminism<BFieldElement> {
    fn from(tokens: Vec<u64>) -> Self {
        NonDeterminism {
            individual_tokens: tokens.iter().map(|&element| element.into()).collect(),
            digests: vec![],
            ram: HashMap::new(),
        }
    }
}

impl From<Vec<u64>> for NonDeterminism<u64> {
    fn from(individual_tokens: Vec<u64>) -> Self {
        NonDeterminism {
            individual_tokens,
            digests: vec![],
            ram: HashMap::new(),
        }
    }
}

impl From<[u64; 0]> for NonDeterminism<BFieldElement> {
    fn from(_: [u64; 0]) -> Self {
        NonDeterminism {
            individual_tokens: vec![],
            digests: vec![],
            ram: HashMap::new(),
        }
    }
}

impl From<[u64; 0]> for NonDeterminism<u64> {
    fn from(_: [u64; 0]) -> Self {
        NonDeterminism {
            individual_tokens: vec![],
            digests: vec![],
            ram: HashMap::new(),
        }
    }
}

impl From<&NonDeterminism<u64>> for NonDeterminism<BFieldElement> {
    fn from(other: &NonDeterminism<u64>) -> Self {
        let individual_tokens = other
            .individual_tokens
            .iter()
            .map(|&element| element.into())
            .collect();
        let ram = other
            .ram
            .iter()
            .map(|(&key, &value)| (key.into(), value.into()))
            .collect();
        NonDeterminism {
            individual_tokens,
            digests: other.digests.clone(),
            ram,
        }
    }
}

impl<E> NonDeterminism<E>
where
    E: Into<BFieldElement> + Eq + Hash,
{
    pub fn new(individual_tokens: Vec<E>) -> Self {
        NonDeterminism {
            individual_tokens,
            digests: vec![],
            ram: HashMap::new(),
        }
    }

    pub fn with_digests(mut self, digests: Vec<Digest>) -> Self {
        self.digests = digests;
        self
    }

    pub fn with_ram(mut self, ram: HashMap<E, E>) -> Self {
        self.ram = ram;
        self
    }
}

#[cfg(test)]
mod tests {
    use assert2::assert;
    use assert2::let_assert;
    use itertools::Itertools;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use rand::thread_rng;
    use rand::Rng;
    use test_strategy::proptest;
    use twenty_first::prelude::Tip5;

    use crate::error::InstructionError;
    use crate::example_programs::CALCULATE_NEW_MMR_PEAKS_FROM_APPEND_WITH_SAFE_LISTS;
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
        encoded[0] = BFieldElement::new(program_length - 1);

        let_assert!(Err(err) = Program::decode(&encoded));
        let_assert!(ProgramDecodingError::MissingArgument(6, _) = err);
    }

    #[test]
    fn decode_program_with_shorter_than_indicated_sequence() {
        let program = triton_program!(nop nop hash push 0 skiz end: halt call end);
        let mut encoded = program.encode();
        encoded[0] += 1_u64.into();
        let_assert!(Err(err) = Program::decode(&encoded));
        let_assert!(ProgramDecodingError::SequenceTooShort = err);
    }

    #[test]
    fn decode_program_with_longer_than_indicated_sequence() {
        let program = triton_program!(nop nop hash push 0 skiz end: halt call end);
        let mut encoded = program.encode();
        encoded[0] -= 1_u64.into();
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
        let digest = program.hash::<Tip5>();

        let expected_digest = [
            4843866011885844809,
            16618866032559590857,
            18247689143239181392,
            7637465675240023996,
            9104890367162237026,
        ]
        .map(BFieldElement::new);
        let expected_digest = Digest::new(expected_digest);

        assert!(expected_digest == digest);
    }

    #[test]
    fn empty_program_is_empty() {
        let program = triton_program!();
        assert!(program.is_empty());
    }

    #[test]
    fn from_various_types_to_public_input() {
        let tokens = thread_rng().gen::<[BFieldElement; 12]>().to_vec();
        let public_input = PublicInput::new(tokens.clone());

        assert!(public_input == tokens.clone().into());
        assert!(public_input == (&tokens).into());
        assert!(public_input == tokens[..].into());
        assert!(public_input == (&tokens[..]).into());

        let tokens = tokens.into_iter().map(|e| e.value()).collect_vec();
        assert!(public_input == tokens.into());

        assert!(PublicInput::new(vec![]) == [].into());
    }

    #[test]
    fn from_various_types_to_non_determinism() {
        let tokens = thread_rng().gen::<[BFieldElement; 12]>().to_vec();
        let non_determinism = NonDeterminism::new(tokens.clone());

        assert!(non_determinism == tokens.clone().into());
        assert!(non_determinism == tokens[..].into());
        assert!(non_determinism == (&tokens[..]).into());

        let tokens = tokens.into_iter().map(|e| e.value()).collect_vec();
        assert!(non_determinism == tokens.into());

        assert!(NonDeterminism::<u64>::new(vec![]) == [].into());
        assert!(NonDeterminism::<BFieldElement>::new(vec![]) == [].into());
    }

    #[test]
    fn create_program_from_code() {
        let element_3 = thread_rng().gen_range(0_u64..BFieldElement::P);
        let element_2 = 1337_usize;
        let element_1 = "17";
        let element_0 = BFieldElement::new(0);
        let instruction_push = Instruction::Push(42_u64.into());
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
        let program = triton_program!(
            {label}: push 1 assert halt
        );
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    fn profile_can_be_created_and_agrees_with_regular_vm_run() {
        let program = CALCULATE_NEW_MMR_PEAKS_FROM_APPEND_WITH_SAFE_LISTS.clone();
        let (profile_output, profile) = program.profile([].into(), [].into()).unwrap();
        let mut vm_state = VMState::new(&program, [].into(), [].into());
        let_assert!(Ok(()) = vm_state.run());
        assert!(profile_output == vm_state.public_output);
        assert!(profile.last().unwrap().cycle_count() == vm_state.cycle_count);

        println!("Profile of Tasm Program:");
        for line in profile {
            println!("{line}");
        }
    }

    #[test]
    fn open_call_instructions_are_marked_in_its_profile() {
        let program = triton_program! {
            push 2 call outer_fn
            outer_fn:
                call inner_fn
                dup 0 skiz recurse halt
            inner_fn:
                push -1 add return
        };
        let (_, profile) = program.profile([].into(), [].into()).unwrap();

        println!();
        for line in &profile {
            println!("{line}");
        }

        let maybe_open_call = profile.iter().find(|line| !line.call_has_returned);
        assert!(maybe_open_call.is_some());
    }

    #[test]
    fn program_with_too_many_returns_crashes_vm_but_not_profiler() {
        let program = triton_program! {
            call foo return halt
            foo: return
        };
        let_assert!(Err(err) = program.profile([].into(), [].into()));
        let_assert!(InstructionError::JumpStackIsEmpty = err.source);
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
}
