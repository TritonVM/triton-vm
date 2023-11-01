use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::hash::Hash;
use std::io::Cursor;

use anyhow::anyhow;
use anyhow::bail;
use anyhow::Error;
use anyhow::Result;
use get_size::GetSize;
use serde_derive::Deserialize;
use serde_derive::Serialize;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::bfield_codec::BFieldCodec;
use twenty_first::shared_math::digest::Digest;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;

use crate::aet::AlgebraicExecutionTrace;
use crate::ensure_eq;
use crate::instruction::Instruction;
use crate::instruction::LabelledInstruction;
use crate::parser::parse;
use crate::parser::to_labelled_instructions;
use crate::vm::VMState;

/// A program for Triton VM.
/// It can be
/// [`run`](Program::run),
/// [`debugged`](Program::debug)
/// (also the RAM-friendlier [`debug_terminal_state`](Program::debug_terminal_state)),
/// [`profiled`](Program::profile),
/// and its execution can be [`traced`](Program::trace_execution).
///
/// [`Hashing`](Program::hash) a program under [`Tip5`][tip5] yields a [`Digest`] that can be used
/// in a [`Claim`](crate::Claim), _i.e._, is consistent with Triton VM's [program attestation].
///
/// A program may contain debug information, such as label names and breakpoints.
/// Access this information through methods [`label_for_address()`][label_for_address] and
/// [`is_breakpoint()`][is_breakpoint]. Some operations, most notably
/// [BField-encoding](BFieldCodec::encode), discard this debug information.
///
/// [program attestation]: https://triton-vm.org/spec/program-attestation.html
/// [tip5]: twenty_first::shared_math::tip5::Tip5
/// [label_for_address]: Program::label_for_address
/// [is_breakpoint]: Program::is_breakpoint
#[derive(Debug, Clone, Default, Eq, GetSize, Serialize, Deserialize)]
pub struct Program {
    pub instructions: Vec<Instruction>,
    address_to_label: HashMap<u64, String>,
    breakpoints: Vec<bool>,
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
    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if sequence.is_empty() {
            bail!("Sequence to decode must not be empty.");
        }
        let program_length = sequence[0].value() as usize;
        let sequence = &sequence[1..];
        ensure_eq!(program_length, sequence.len());

        let mut read_idx = 0;
        let mut instructions = Vec::with_capacity(program_length);
        while read_idx < program_length {
            let opcode = sequence[read_idx];
            let mut instruction: Instruction = opcode
                .try_into()
                .expect("Invalid opcode {opcode} at index {idx}.");
            if instruction.has_arg() && instructions.len() + instruction.size() > program_length {
                bail!("Missing argument for instruction {instruction} at index {read_idx}.");
            }
            if instruction.has_arg() {
                let arg = sequence[read_idx + 1];
                instruction = instruction
                    .change_arg(arg)
                    .expect("Invalid argument {arg} for instruction {instruction} at index {idx}.");
            }

            instructions.extend(vec![instruction; instruction.size()]);
            read_idx += instruction.size();
        }

        ensure_eq!(read_idx, program_length);
        ensure_eq!(instructions.len(), program_length);

        Ok(Box::new(Program {
            instructions,
            address_to_label: HashMap::new(),
            breakpoints: vec![],
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
        let breakpoints = Self::extract_breakpoints(labelled_instructions);

        assert_eq!(instructions.len(), breakpoints.len());
        Program {
            instructions,
            address_to_label,
            breakpoints,
        }
    }

    fn build_label_to_address_map(program: &[LabelledInstruction]) -> HashMap<String, u64> {
        use LabelledInstruction::*;

        let mut label_map = HashMap::new();
        let mut instruction_pointer = 0;

        for labelled_instruction in program.iter() {
            match labelled_instruction {
                Label(label) => match label_map.entry(label.clone()) {
                    Entry::Occupied(_) => panic!("Duplicate label: {label}"),
                    Entry::Vacant(entry) => {
                        entry.insert(instruction_pointer);
                    }
                },
                Instruction(instruction) => instruction_pointer += instruction.size() as u64,
                Breakpoint => (),
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

    fn extract_breakpoints(labelled_instructions: &[LabelledInstruction]) -> Vec<bool> {
        let mut breakpoints = vec![];
        let mut break_before_next_instruction = false;

        for instruction in labelled_instructions {
            match instruction {
                LabelledInstruction::Breakpoint => break_before_next_instruction = true,
                LabelledInstruction::Instruction(instruction) => {
                    breakpoints.extend(vec![break_before_next_instruction; instruction.size()]);
                    break_before_next_instruction = false;
                }
                _ => (),
            }
        }

        breakpoints
    }

    /// Create a `Program` by parsing source code.
    pub fn from_code(code: &str) -> Result<Self> {
        parse(code)
            .map(|tokens| to_labelled_instructions(&tokens))
            .map(|instructions| Program::new(&instructions))
            .map_err(|err| anyhow!("{err}"))
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
            if self.is_breakpoint(address) {
                labelled_instructions.push(LabelledInstruction::Breakpoint);
            }
            labelled_instructions.push(LabelledInstruction::Instruction(instruction));

            for _ in 1..instruction_size {
                instruction_stream.next();
            }
            address += instruction_size;
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

    /// Hash the program using the given `AlgebraicHasher`.
    pub fn hash<H: AlgebraicHasher>(&self) -> Digest {
        H::hash_varlen(&self.to_bwords())
    }

    /// Run Triton VM on the given [`Program`] with the given public and secret input.
    ///
    /// See also [`trace_execution`](Self::trace_execution) and [`debug`](Self::debug).
    pub fn run(
        &self,
        public_input: PublicInput,
        non_determinism: NonDeterminism<BFieldElement>,
    ) -> Result<Vec<BFieldElement>> {
        let mut state = VMState::new(self, public_input, non_determinism);
        while !state.halting {
            state.step()?;
        }
        Ok(state.public_output)
    }

    /// Trace the execution of a [`Program`]. That is, [`run`][run] the [`Program`] and additionally
    /// record that part of every encountered state that is necessary for proving correct execution.
    /// If execution  succeeds, returns
    /// 1. an [`AlgebraicExecutionTrace`], and
    /// 1. the output of the program.
    ///
    /// See also [`debug`](Self::debug) and [`run`][run].
    ///
    /// [run]: Self::run
    pub fn trace_execution(
        &self,
        public_input: PublicInput,
        non_determinism: NonDeterminism<BFieldElement>,
    ) -> Result<(AlgebraicExecutionTrace, Vec<BFieldElement>)> {
        let mut aet = AlgebraicExecutionTrace::new(self.clone());
        let mut state = VMState::new(self, public_input, non_determinism);
        assert_eq!(self.len_bwords(), aet.instruction_multiplicities.len());

        while !state.halting {
            aet.record_state(&state)?;
            let co_processor_calls = state.step()?;
            for call in co_processor_calls {
                aet.record_co_processor_call(call);
            }
        }

        Ok((aet, state.public_output))
    }

    /// Similar to [`run`](Self::run), but also returns a [`Vec`] of [`VMState`]s, one for each
    /// step of the VM. On premature termination of the VM, returns all [`VMState`]s up to the
    /// point of failure.
    ///
    /// The VM's initial state is either the provided `initial_state`, or a new [`VMState`] if
    /// `initial_state` is `None`. The initial state is included in the returned [`Vec`] of
    /// [`VMState`]s. If an initial state is provided, the `program`, `public_input` and
    /// `private_input` provided to this method are ignored and the initial state's program and
    /// inputs are used instead.
    ///
    /// If `num_cycles_to_execute` is `Some(number_of_cycles)`, the VM will execute at most
    /// `number_of_cycles` cycles. If `num_cycles_to_execute` is `None`, the VM will execute until
    /// it halts or the maximum number of cycles (2^{32}) is reached.
    ///
    /// See also [`debug_terminal_state`](Self::debug_terminal_state) and
    /// [`trace_execution`](Self::trace_execution).
    pub fn debug<'pgm>(
        &'pgm self,
        public_input: PublicInput,
        non_determinism: NonDeterminism<BFieldElement>,
        initial_state: Option<VMState<'pgm>>,
        num_cycles_to_execute: Option<u32>,
    ) -> (Vec<VMState<'pgm>>, Option<Error>) {
        let mut states = vec![];
        let mut state = match initial_state {
            Some(initial_state) => initial_state,
            None => VMState::new(self, public_input, non_determinism),
        };
        let max_cycles = Self::max_cycle_to_reach(&state, num_cycles_to_execute);

        while !state.halting && state.cycle_count < max_cycles {
            states.push(state.clone());
            if let Err(err) = state.step() {
                return (states, Some(err));
            }
        }

        states.push(state);
        (states, None)
    }

    /// Run Triton VM on the given [`Program`] with the given public and secret input, and return
    /// the final [`VMState`]. Requires substantially less RAM than [`debug`][debug] since no
    /// intermediate states are recorded.
    ///
    /// Parameters `initial_state` and `num_cycles_to_execute` are handled like in [`debug`][debug];
    /// see there for more details.
    ///
    /// If an error is encountered, returns the error and the [`VMState`] at the point of failure.
    ///
    /// See also [`trace_execution`](Self::trace_execution) and [`run`](Self::run).
    ///
    /// [debug]: Self::debug
    pub fn debug_terminal_state<'pgm>(
        &'pgm self,
        public_input: PublicInput,
        non_determinism: NonDeterminism<BFieldElement>,
        initial_state: Option<VMState<'pgm>>,
        num_cycles_to_execute: Option<u32>,
    ) -> Result<VMState<'pgm>, (Error, VMState<'pgm>)> {
        let mut state = match initial_state {
            Some(initial_state) => initial_state,
            None => VMState::new(self, public_input, non_determinism),
        };
        let max_cycles = Self::max_cycle_to_reach(&state, num_cycles_to_execute);

        while !state.halting && state.cycle_count < max_cycles {
            // [`VMState::step`] is not atomic – avoid returning an inconsistent state
            let consistent_state = state.clone();
            if let Err(err) = state.step() {
                return Err((err, consistent_state));
            }
        }
        Ok(state)
    }

    fn max_cycle_to_reach(state: &VMState, num_cycles_to_execute: Option<u32>) -> u32 {
        match num_cycles_to_execute {
            Some(number_of_cycles) => state.cycle_count + number_of_cycles,
            None => u32::MAX,
        }
    }

    /// Run Triton VM with the given public and secret input, but record the number of cycles spent
    /// in each callable block of instructions. This function returns a Result wrapping a program
    /// profiler report, which is a Vec of [`ProfileLine`]s.
    ///
    /// See also [`run`](Self::run), [`trace_execution`](Self::trace_execution) and
    /// [`debug`](Self::debug).
    pub fn profile(
        &self,
        public_input: PublicInput,
        non_determinism: NonDeterminism<BFieldElement>,
    ) -> Result<(Vec<BFieldElement>, Vec<ProfileLine>)> {
        let mut profiler = VMProfiler::new();
        let mut state = VMState::new(self, public_input, non_determinism);
        while !state.halting {
            if let Instruction::Call(address) = state.current_instruction()? {
                let label = self.label_for_address(address.value());
                profiler.enter_span_with_label_at_cycle(label, state.cycle_count);
            }
            if let Instruction::Return = state.current_instruction()? {
                profiler.exit_span_at_cycle(state.cycle_count);
            }

            state.step()?;
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

#[derive(Debug, Clone, Default)]
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
        for &line_number in self.call_stack.iter() {
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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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

#[derive(Clone, Default, Debug, PartialEq, Eq, BFieldCodec)]
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
/// and a initial state of random-access memory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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
    use itertools::Itertools;
    use rand::thread_rng;
    use rand::Rng;
    use twenty_first::shared_math::tip5::Tip5;

    use crate::example_programs::CALCULATE_NEW_MMR_PEAKS_FROM_APPEND_WITH_SAFE_LISTS;
    use crate::parser::tests::program_gen;
    use crate::triton_program;

    use super::*;

    #[test]
    fn random_program_encode_decode_equivalence() {
        let mut rng = thread_rng();
        for _ in 0..50 {
            let program_len = rng.gen_range(20..420);
            let source_code = program_gen(program_len);
            let program = triton_program!({ source_code });

            let encoded = program.encode();
            let decoded = *Program::decode(&encoded).unwrap();

            assert_eq!(program, decoded);
        }
    }

    #[test]
    fn decode_program_with_missing_argument_as_last_instruction() {
        let program = triton_program!(push 3 push 3 eq assert push 3);
        let program_length = program.len_bwords() as u64;
        let encoded = program.encode();

        let mut encoded = encoded[0..encoded.len() - 1].to_vec();
        encoded[0] = BFieldElement::new(program_length - 1);

        let err = Program::decode(&encoded).err().unwrap();
        assert_eq!(
            "Missing argument for instruction push 0 at index 6.",
            err.to_string(),
        );
    }

    #[test]
    #[should_panic(expected = "Expected `program_length` to equal `sequence.len()`.")]
    fn decode_program_with_length_mismatch() {
        let program = triton_program!(nop nop hash push 0 skiz end: halt call end);
        let mut encoded = program.encode();
        encoded[0] += 1_u64.into();
        Program::decode(&encoded).unwrap();
    }

    #[test]
    fn decode_program_from_empty_sequence() {
        let encoded = vec![];
        let err = Program::decode(&encoded).err().unwrap();
        assert_eq!("Sequence to decode must not be empty.", err.to_string(),);
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

        assert_eq!(expected_digest, digest);
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

        assert_eq!(public_input, tokens.clone().into());
        assert_eq!(public_input, (&tokens).into());
        assert_eq!(public_input, tokens[..].into());
        assert_eq!(public_input, (&tokens[..]).into());

        let tokens = tokens.into_iter().map(|e| e.value()).collect_vec();
        assert_eq!(public_input, tokens.into());

        assert_eq!(PublicInput::new(vec![]), [].into());
    }

    #[test]
    fn from_various_types_to_non_determinism() {
        let tokens = thread_rng().gen::<[BFieldElement; 12]>().to_vec();
        let non_determinism = NonDeterminism::new(tokens.clone());

        assert_eq!(non_determinism, tokens.clone().into());
        assert_eq!(non_determinism, tokens[..].into());
        assert_eq!(non_determinism, (&tokens[..]).into());

        let tokens = tokens.into_iter().map(|e| e.value()).collect_vec();
        assert_eq!(non_determinism, tokens.into());

        assert_eq!(NonDeterminism::<u64>::new(vec![]), [].into());
        assert_eq!(NonDeterminism::<BFieldElement>::new(vec![]), [].into());
    }

    #[test]
    fn test_creating_program_from_code() {
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
        assert_eq!(program_from_code, program_from_macro);
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
    fn test_profile() {
        let program = CALCULATE_NEW_MMR_PEAKS_FROM_APPEND_WITH_SAFE_LISTS.clone();
        let (profile_output, profile) = program.profile([].into(), [].into()).unwrap();
        let debug_terminal_state = program
            .debug_terminal_state([].into(), [].into(), None, None)
            .unwrap();
        assert_eq!(profile_output, debug_terminal_state.public_output);
        assert_eq!(
            profile.last().unwrap().cycle_count(),
            debug_terminal_state.cycle_count
        );

        println!("Profile of Tasm Program:");
        for line in profile {
            println!("{line}");
        }
    }

    #[test]
    fn test_profile_with_open_calls() {
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
        for line in profile.iter() {
            println!("{line}");
        }

        let maybe_open_call = profile.iter().find(|line| !line.call_has_returned);
        assert!(maybe_open_call.is_some());
    }

    #[test]
    #[should_panic(expected = "Jump stack is empty")]
    fn program_with_too_many_returns_crashes_vm_but_not_profiler() {
        let program = triton_program! {
            call foo return halt
            foo: return
        };
        let _ = program.profile([].into(), [].into()).unwrap();
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
