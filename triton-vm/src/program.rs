use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::hash::Hash;
use std::io::Cursor;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Sub;

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
use crate::profiler::profile_start;
use crate::profiler::profile_stop;
use crate::table::hash_table::PERMUTATION_TRACE_LENGTH;
use crate::table::u32_table::U32TableEntry;
use crate::vm::CoProcessorCall;
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
            breakpoints: vec![],
            type_hints: HashMap::default(),
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

impl Program {
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
        let mut label_map = HashMap::new();
        let mut instruction_pointer = 0;

        for labelled_instruction in program {
            if let LabelledInstruction::Instruction(instruction) = labelled_instruction {
                instruction_pointer += instruction.size() as u64;
                continue;
            }

            let LabelledInstruction::Label(label) = labelled_instruction else {
                continue;
            };
            let Entry::Vacant(new_label_map_entry) = label_map.entry(label.clone()) else {
                panic!("Duplicate label: {label}");
            };
            new_label_map_entry.insert(instruction_pointer);
        }

        label_map
    }

    fn turn_labels_into_addresses(
        labelled_instructions: &[LabelledInstruction],
        label_to_address: &HashMap<String, u64>,
    ) -> Vec<Instruction> {
        labelled_instructions
            .iter()
            .filter_map(|inst| Self::turn_label_to_address_for_instruction(inst, label_to_address))
            .flat_map(|inst| vec![inst; inst.size()])
            .collect()
    }

    fn turn_label_to_address_for_instruction(
        labelled_instruction: &LabelledInstruction,
        label_map: &HashMap<String, u64>,
    ) -> Option<Instruction> {
        let LabelledInstruction::Instruction(instruction) = labelled_instruction else {
            return None;
        };

        let instruction_with_absolute_address =
            instruction.map_call_address(|label| Self::address_for_label(label, label_map));
        Some(instruction_with_absolute_address)
    }

    fn address_for_label(label: &str, label_map: &HashMap<String, u64>) -> BFieldElement {
        let maybe_address = label_map.get(label).map(|&a| bfe!(a));
        maybe_address.unwrap_or_else(|| panic!("Label not found: {label}"))
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
                LabelledInstruction::Label(_) => (),
                LabelledInstruction::Breakpoint => break_before_next_instruction = true,
                LabelledInstruction::TypeHint(type_hint) => match type_hints.entry(address) {
                    Entry::Occupied(mut entry) => entry.get_mut().push(type_hint.clone()),
                    Entry::Vacant(entry) => _ = entry.insert(vec![type_hint.clone()]),
                },
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
        // not encoded using `BFieldCodec` because that would prepend the length
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
        non_determinism: NonDeterminism,
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
        non_determinism: NonDeterminism,
    ) -> Result<(AlgebraicExecutionTrace, Vec<BFieldElement>)> {
        profile_start!("trace execution", "gen");
        let state = VMState::new(self, public_input, non_determinism);
        let (aet, terminal_state) = self.trace_execution_of_state(state)?;
        profile_stop!("trace execution");
        Ok((aet, terminal_state.public_output))
    }

    /// Trace the execution of a [`Program`] from a given [`VMState`]. Consider
    /// using [`trace_execution`][Self::trace_execution], unless you know this is
    /// what you want.
    ///
    /// Returns the [`AlgebraicExecutionTrace`] and the terminal [`VMState`] if
    /// execution succeeds.
    ///
    /// # Panics
    ///
    /// - if the given [`VMState`] is not about to `self`
    /// - if the given [`VMState`] is incorrectly initialized
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

    /// Run Triton VM with the given public and secret input, recording the
    /// influence of a callable block of instructions on the
    /// [`AlgebraicExecutionTrace`]. For example, this can be used to identify the
    /// number of clock cycles spent in some block of instructions, or how many rows
    /// it contributes to the U32 Table.
    ///
    /// See also [`run`][run] and [`trace_execution`][trace_execution].
    ///
    /// [run]: Self::run
    /// [trace_execution]: Self::trace_execution
    pub fn profile(
        &self,
        public_input: PublicInput,
        non_determinism: NonDeterminism,
    ) -> Result<(Vec<BFieldElement>, ExecutionTraceProfile)> {
        let mut profiler = ExecutionTraceProfiler::new(self.instructions.len());
        let mut state = VMState::new(self, public_input, non_determinism);
        while !state.halting {
            if let Ok(Instruction::Call(address)) = state.current_instruction() {
                let label = self.label_for_address(address.value());
                profiler.enter_span(label);
            }
            if let Ok(Instruction::Return) = state.current_instruction() {
                profiler.exit_span();
            }
            match state.step() {
                Ok(calls) => profiler.handle_co_processor_calls(calls),
                Err(err) => return Err(VMError::new(err, state)),
            };
        }

        Ok((state.public_output, profiler.finish()))
    }

    /// The label for the given address, or a deterministic, unique substitute if no label is found.
    pub fn label_for_address(&self, address: u64) -> String {
        // Uniqueness of the label is relevant for printing and subsequent parsing:
        // Parsing fails on duplicate labels.
        self.address_to_label
            .get(&address)
            .cloned()
            .unwrap_or_else(|| format!("address_{address}"))
    }
}

#[derive(Debug, Default, Clone, Eq, PartialEq, Arbitrary)]
struct ExecutionTraceProfiler {
    call_stack: Vec<usize>,
    profile: Vec<ProfileLine>,
    table_heights: VMTableHeights,
    u32_table_entries: HashSet<U32TableEntry>,
}

/// A single line in a [profile report](ExecutionTraceProfile) for profiling
/// [Triton](crate) programs.
#[derive(Debug, Default, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub struct ProfileLine {
    pub label: String,
    pub call_depth: usize,

    /// Table heights at the start of this span, _i.e._, right before the corresponding
    /// [`call`](Instruction::Call) instruction was executed.
    pub table_heights_start: VMTableHeights,

    table_heights_stop: VMTableHeights,
}

/// A report for the completed execution of a [Triton](crate) program.
///
/// Offers a human-readable [`Display`] implementation and can be processed
/// programmatically.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub struct ExecutionTraceProfile {
    pub total: VMTableHeights,
    pub profile: Vec<ProfileLine>,
}

/// The heights of various [tables](AlgebraicExecutionTrace) relevant for
/// proving the correct execution in [Triton VM](crate).
#[non_exhaustive]
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub struct VMTableHeights {
    pub processor: u32,
    pub op_stack: u32,
    pub ram: u32,
    pub hash: u32,
    pub u32: u32,
}

impl ExecutionTraceProfiler {
    fn new(num_instructions: usize) -> Self {
        Self {
            call_stack: vec![],
            profile: vec![],
            table_heights: VMTableHeights::new(num_instructions),
            u32_table_entries: HashSet::default(),
        }
    }

    fn enter_span(&mut self, label: impl Into<String>) {
        let call_stack_len = self.call_stack.len();
        let line_number = self.profile.len();

        let profile_line = ProfileLine {
            label: label.into(),
            call_depth: call_stack_len,
            table_heights_start: self.table_heights,
            table_heights_stop: VMTableHeights::default(),
        };

        self.profile.push(profile_line);
        self.call_stack.push(line_number);
    }

    fn exit_span(&mut self) {
        if let Some(line_number) = self.call_stack.pop() {
            self.profile[line_number].table_heights_stop = self.table_heights;
        };
    }

    fn handle_co_processor_calls(&mut self, calls: Vec<CoProcessorCall>) {
        self.table_heights.processor += 1;
        for call in calls {
            match call {
                CoProcessorCall::SpongeStateReset => self.table_heights.hash += 1,
                CoProcessorCall::Tip5Trace(_, trace) => {
                    self.table_heights.hash += u32::try_from(trace.len()).unwrap();
                }
                CoProcessorCall::U32Call(c) => {
                    self.u32_table_entries.insert(c);
                    let contribution = U32TableEntry::table_height_contribution;
                    self.table_heights.u32 = self.u32_table_entries.iter().map(contribution).sum();
                }
                CoProcessorCall::OpStackCall(_) => self.table_heights.op_stack += 1,
                CoProcessorCall::RamCall(_) => self.table_heights.ram += 1,
            }
        }
    }

    fn finish(mut self) -> ExecutionTraceProfile {
        for &line_number in &self.call_stack {
            self.profile[line_number].table_heights_stop = self.table_heights;
        }

        ExecutionTraceProfile {
            total: self.table_heights,
            profile: self.profile,
        }
    }
}

impl VMTableHeights {
    fn new(num_instructions: usize) -> Self {
        let padded_program_len = (num_instructions + 1).next_multiple_of(Tip5::RATE);
        let num_absorbs = padded_program_len / Tip5::RATE;
        let initial_hash_table_len = num_absorbs * PERMUTATION_TRACE_LENGTH;

        Self {
            hash: initial_hash_table_len.try_into().unwrap(),
            ..Default::default()
        }
    }
}

impl Sub<Self> for VMTableHeights {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            processor: self.processor.saturating_sub(rhs.processor),
            op_stack: self.op_stack.saturating_sub(rhs.op_stack),
            ram: self.ram.saturating_sub(rhs.ram),
            hash: self.hash.saturating_sub(rhs.hash),
            u32: self.u32.saturating_sub(rhs.u32),
        }
    }
}

impl Add<Self> for VMTableHeights {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            processor: self.processor + rhs.processor,
            op_stack: self.op_stack + rhs.op_stack,
            ram: self.ram + rhs.ram,
            hash: self.hash + rhs.hash,
            u32: self.u32 + rhs.u32,
        }
    }
}

impl AddAssign<Self> for VMTableHeights {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl ProfileLine {
    fn table_height_contributions(&self) -> VMTableHeights {
        self.table_heights_stop - self.table_heights_start
    }
}

impl Display for ProfileLine {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let indentation = "  ".repeat(self.call_depth);
        let label = &self.label;
        let cycle_count = self.table_height_contributions().processor;
        write!(f, "{indentation}{label}: {cycle_count}")
    }
}

impl Display for ExecutionTraceProfile {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        struct AggregateLine {
            label: String,
            call_depth: usize,
            table_heights: VMTableHeights,
        }

        const COL_WIDTH: usize = 20;

        let mut aggregated: Vec<AggregateLine> = vec![];
        for line in &self.profile {
            if let Some(agg) = aggregated
                .iter_mut()
                .find(|agg| agg.label == line.label && agg.call_depth == line.call_depth)
            {
                agg.table_heights += line.table_height_contributions();
            } else {
                aggregated.push(AggregateLine {
                    label: line.label.clone(),
                    call_depth: line.call_depth,
                    table_heights: line.table_height_contributions(),
                });
            }
        }
        aggregated.push(AggregateLine {
            label: "Total".to_string(),
            call_depth: 0,
            table_heights: self.total,
        });

        let label = |line: &AggregateLine| "··".repeat(line.call_depth) + &line.label;
        let label_len = |line| label(line).len();

        let max_label_len = aggregated.iter().map(label_len).max();
        let max_label_len = max_label_len.unwrap_or_default().max(COL_WIDTH);

        let [soubroutine, processor, op_stack, ram, hash, u32_title] =
            ["Subroutine", "Processor", "Op Stack", "RAM", "Hash", "U32"];

        write!(f, "| {soubroutine:<max_label_len$} ")?;
        write!(f, "| {processor:>COL_WIDTH$} ")?;
        write!(f, "| {op_stack:>COL_WIDTH$} ")?;
        write!(f, "| {ram:>COL_WIDTH$} ")?;
        write!(f, "| {hash:>COL_WIDTH$} ")?;
        write!(f, "| {u32_title:>COL_WIDTH$} ")?;
        writeln!(f, "|")?;

        let dash = "-";
        write!(f, "|:{dash:-<max_label_len$}-")?;
        write!(f, "|-{dash:->COL_WIDTH$}:")?;
        write!(f, "|-{dash:->COL_WIDTH$}:")?;
        write!(f, "|-{dash:->COL_WIDTH$}:")?;
        write!(f, "|-{dash:->COL_WIDTH$}:")?;
        write!(f, "|-{dash:->COL_WIDTH$}:")?;
        writeln!(f, "|")?;

        for line in &aggregated {
            let rel_precision = 1;
            let rel_width = 3 + 1 + rel_precision; // eg '100.0'
            let abs_width = COL_WIDTH - rel_width - 4; // ' (' and '%)'

            let label = label(line);
            let proc_abs = line.table_heights.processor;
            let proc_rel = 100.0 * f64::from(proc_abs) / f64::from(self.total.processor);
            let proc_rel = format!("{proc_rel:.rel_precision$}");
            let stack_abs = line.table_heights.op_stack;
            let stack_rel = 100.0 * f64::from(stack_abs) / f64::from(self.total.op_stack);
            let stack_rel = format!("{stack_rel:.rel_precision$}");
            let ram_abs = line.table_heights.ram;
            let ram_rel = 100.0 * f64::from(ram_abs) / f64::from(self.total.ram);
            let ram_rel = format!("{ram_rel:.rel_precision$}");
            let hash_abs = line.table_heights.hash;
            let hash_rel = 100.0 * f64::from(hash_abs) / f64::from(self.total.hash);
            let hash_rel = format!("{hash_rel:.rel_precision$}");
            let u32_abs = line.table_heights.u32;
            let u32_rel = 100.0 * f64::from(u32_abs) / f64::from(self.total.u32);
            let u32_rel = format!("{u32_rel:.rel_precision$}");

            write!(f, "| {label:<max_label_len$} ")?;
            write!(f, "| {proc_abs:>abs_width$} ({proc_rel:>rel_width$}%) ")?;
            write!(f, "| {stack_abs:>abs_width$} ({stack_rel:>rel_width$}%) ")?;
            write!(f, "| {ram_abs:>abs_width$} ({ram_rel:>rel_width$}%) ")?;
            write!(f, "| {hash_abs:>abs_width$} ({hash_rel:>rel_width$}%) ")?;
            write!(f, "| {u32_abs:>abs_width$} ({u32_rel:>rel_width$}%) ")?;
            writeln!(f, "|")?;
        }

        Ok(())
    }
}

#[derive(Debug, Default, Clone, Eq, PartialEq, BFieldCodec, Arbitrary)]
pub struct PublicInput {
    pub individual_tokens: Vec<BFieldElement>,
}

impl From<Vec<BFieldElement>> for PublicInput {
    fn from(individual_tokens: Vec<BFieldElement>) -> Self {
        Self::new(individual_tokens)
    }
}

impl From<&Vec<BFieldElement>> for PublicInput {
    fn from(tokens: &Vec<BFieldElement>) -> Self {
        Self::new(tokens.to_owned())
    }
}

impl<const N: usize> From<[BFieldElement; N]> for PublicInput {
    fn from(tokens: [BFieldElement; N]) -> Self {
        Self::new(tokens.to_vec())
    }
}

impl From<&[BFieldElement]> for PublicInput {
    fn from(tokens: &[BFieldElement]) -> Self {
        Self::new(tokens.to_vec())
    }
}

impl PublicInput {
    pub fn new(individual_tokens: Vec<BFieldElement>) -> Self {
        Self { individual_tokens }
    }
}

/// All sources of non-determinism for a program. This includes elements that
/// can be read using instruction `divine`, digests that can be read using
/// instruction `merkle_step`, and an initial state of random-access memory.
#[derive(Debug, Default, Clone, Eq, PartialEq, Serialize, Deserialize, Arbitrary)]
pub struct NonDeterminism {
    pub individual_tokens: Vec<BFieldElement>,
    pub digests: Vec<Digest>,
    pub ram: HashMap<BFieldElement, BFieldElement>,
}

impl From<Vec<BFieldElement>> for NonDeterminism {
    fn from(tokens: Vec<BFieldElement>) -> Self {
        Self::new(tokens)
    }
}

impl From<&Vec<BFieldElement>> for NonDeterminism {
    fn from(tokens: &Vec<BFieldElement>) -> Self {
        Self::new(tokens.to_owned())
    }
}

impl<const N: usize> From<[BFieldElement; N]> for NonDeterminism {
    fn from(tokens: [BFieldElement; N]) -> Self {
        Self::new(tokens.to_vec())
    }
}

impl From<&[BFieldElement]> for NonDeterminism {
    fn from(tokens: &[BFieldElement]) -> Self {
        Self::new(tokens.to_vec())
    }
}

impl NonDeterminism {
    pub fn new<V: Into<Vec<BFieldElement>>>(individual_tokens: V) -> Self {
        Self {
            individual_tokens: individual_tokens.into(),
            digests: vec![],
            ram: HashMap::new(),
        }
    }

    #[must_use]
    pub fn with_digests<V: Into<Vec<Digest>>>(mut self, digests: V) -> Self {
        self.digests = digests.into();
        self
    }

    #[must_use]
    pub fn with_ram<H: Into<HashMap<BFieldElement, BFieldElement>>>(mut self, ram: H) -> Self {
        self.ram = ram.into();
        self
    }
}

#[cfg(test)]
mod tests {
    use assert2::assert;
    use assert2::let_assert;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use rand::thread_rng;
    use rand::Rng;
    use test_strategy::proptest;
    use twenty_first::prelude::Tip5;

    use crate::error::InstructionError;
    use crate::example_programs::CALCULATE_NEW_MMR_PEAKS_FROM_APPEND_WITH_SAFE_LISTS;
    use crate::table::master_table::TableId;
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

    #[proptest]
    fn from_various_types_to_public_input(#[strategy(arb())] tokens: Vec<BFieldElement>) {
        let public_input = PublicInput::new(tokens.clone());

        assert!(public_input == tokens.clone().into());
        assert!(public_input == (&tokens).into());
        assert!(public_input == tokens[..].into());
        assert!(public_input == (&tokens[..]).into());

        assert!(PublicInput::new(vec![]) == [].into());
    }

    #[proptest]
    fn from_various_types_to_non_determinism(#[strategy(arb())] tokens: Vec<BFieldElement>) {
        let non_determinism = NonDeterminism::new(tokens.clone());

        assert!(non_determinism == tokens.clone().into());
        assert!(non_determinism == tokens[..].into());
        assert!(non_determinism == (&tokens[..]).into());

        assert!(NonDeterminism::new(vec![]) == [].into());
    }

    #[test]
    fn create_program_from_code() {
        let element_3 = thread_rng().gen_range(0_u64..BFieldElement::P);
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
        assert!(profile.total.processor == vm_state.cycle_count);

        let_assert!(Ok((aet, trace_output)) = program.trace_execution([].into(), [].into()));
        assert!(profile_output == trace_output);
        let proc_height = u32::try_from(aet.height_of_table(TableId::Processor)).unwrap();
        assert!(proc_height == profile.total.processor);

        let op_stack_height = u32::try_from(aet.height_of_table(TableId::OpStack)).unwrap();
        assert!(op_stack_height == profile.total.op_stack);

        let ram_height = u32::try_from(aet.height_of_table(TableId::Ram)).unwrap();
        assert!(ram_height == profile.total.ram);

        let hash_height = u32::try_from(aet.height_of_table(TableId::Hash)).unwrap();
        assert!(hash_height == profile.total.hash);

        let u32_height = u32::try_from(aet.height_of_table(TableId::U32)).unwrap();
        assert!(u32_height == profile.total.u32);

        println!("{profile}");
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
