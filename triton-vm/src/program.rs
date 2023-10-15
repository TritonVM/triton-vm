use std::collections::hash_map::Entry;
use std::collections::HashMap;
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
use crate::error::InstructionError::InstructionPointerOverflow;
use crate::instruction::Instruction;
use crate::instruction::LabelledInstruction;
use crate::parser::parse;
use crate::parser::to_labelled_instructions;
use crate::vm::VMState;

/// A `Program` is a `Vec<Instruction>` that contains duplicate elements for instructions with a
/// size of 2. This means that the index in the vector corresponds to the VM's
/// `instruction_pointer`. These duplicate values should most often be skipped/ignored,
/// e.g. when pretty-printing.
#[derive(Debug, Clone, Default, PartialEq, Eq, GetSize, Serialize, Deserialize)]
pub struct Program {
    pub instructions: Vec<Instruction>,
    pub address_to_label: HashMap<u64, String>,
}

impl Display for Program {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let mut stream = self.instructions.iter();
        while let Some(instruction) = stream.next() {
            writeln!(f, "{instruction}")?;
            // 2-word instructions already print their arguments
            for _ in 1..instruction.size() {
                stream.next();
            }
        }
        Ok(())
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

        let mut idx = 0;
        let mut instructions = Vec::with_capacity(program_length);
        while idx < program_length {
            let opcode: u32 = match sequence[idx].value().try_into() {
                Ok(opcode) => opcode,
                Err(_) => bail!("Invalid opcode {} at index {idx}.", sequence[idx].value()),
            };
            let instruction: Instruction = opcode.try_into()?;
            if !instruction.has_arg() {
                instructions.push(instruction);
            } else if instructions.len() + 1 >= program_length {
                bail!("Missing argument for instruction {instruction} at index {idx}.");
            } else {
                let instruction = match instruction.change_arg(sequence[idx + 1]) {
                    Some(instruction) => instruction,
                    None => {
                        bail!("Invalid argument for instruction {instruction} at index {idx}.")
                    }
                };
                // Instructions with argument are recorded twice to align the `instruction_pointer`.
                instructions.push(instruction);
                instructions.push(instruction);
            }
            idx += instruction.size();
        }

        ensure_eq!(idx, program_length);

        Ok(Box::new(Program {
            instructions,
            address_to_label: HashMap::new(),
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
    /// Create a `Program` from a slice of `Instruction`.
    pub fn new(labelled_instructions: &[LabelledInstruction]) -> Self {
        let label_to_address = Self::build_label_to_address_map(labelled_instructions);
        let instructions = labelled_instructions
            .iter()
            .flat_map(|instr| Self::turn_label_to_address_for_instruction(instr, &label_to_address))
            .flat_map(|instr| vec![instr; instr.size()])
            .collect();
        let address_to_label = label_to_address
            .into_iter()
            .map(|(label, address)| (address, label))
            .collect();

        Program {
            instructions,
            address_to_label,
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

    /// Convert a single instruction with a labelled call target to an instruction with an absolute
    /// address as the call target. Discards all labels.
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

    /// Create a `Program` by parsing source code.
    pub fn from_code(code: &str) -> Result<Self> {
        parse(code)
            .map(|tokens| to_labelled_instructions(&tokens))
            .map(|instructions| Program::new(&instructions))
            .map_err(|err| anyhow!("{err}"))
    }

    /// Convert a `Program` to a `Vec<BFieldElement>`.
    ///
    /// Every single-word instruction is converted to a single word.
    ///
    /// Every double-word instruction is converted to two words.
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

            match state.instruction_pointer < aet.instruction_multiplicities.len() {
                true => aet.instruction_multiplicities[state.instruction_pointer] += 1,
                false => bail!(InstructionPointerOverflow(state.instruction_pointer)),
            }

            let maybe_co_processor_call = state.step()?;
            if let Some(co_processor_call) = maybe_co_processor_call {
                aet.record_co_processor_call(co_processor_call);
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
        let mut call_stack = vec![];
        let mut profile = vec![];

        let mut state = VMState::new(self, public_input, non_determinism);
        while !state.halting {
            if let Instruction::Call(address) = state.current_instruction()? {
                let label = self.label_for_address(address);
                let profile_line = ProfileLine::new(call_stack.len(), label, 0);
                let profile_line_number = profile.len();
                profile.push(profile_line);
                call_stack.push((state.cycle_count, profile_line_number));
            }

            if let Instruction::Return = state.current_instruction()? {
                let (clk_start, profile_line_number) = call_stack.pop().unwrap();
                profile[profile_line_number].cycle_count = state.cycle_count - clk_start;
            }

            state.step()?;
        }

        for (clk_start, profile_line_number) in call_stack {
            profile[profile_line_number].cycle_count = state.cycle_count - clk_start;
            profile[profile_line_number].label += " (open)";
        }
        profile.push(ProfileLine::new(0, "total".to_string(), state.cycle_count));

        Ok((state.public_output, profile))
    }

    fn label_for_address(&self, address: BFieldElement) -> String {
        let address = address.value();
        let substitute_for_unknown_label = format! {"address_{address}"};
        self.address_to_label
            .get(&address)
            .unwrap_or(&substitute_for_unknown_label)
            .to_owned()
    }
}

/// A single line in a profile report for profiling Triton Assembly programs.
pub struct ProfileLine {
    pub call_stack_depth: usize,
    pub label: String,
    pub cycle_count: u32,
}

impl ProfileLine {
    pub fn new(call_stack_depth: usize, label: String, cycle_count: u32) -> Self {
        ProfileLine {
            call_stack_depth,
            label,
            cycle_count,
        }
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
            profile.last().unwrap().cycle_count,
            debug_terminal_state.cycle_count
        );

        println!("Profile of Tasm Program:");
        for line in profile {
            let indentation = vec!["  "; line.call_stack_depth].join("");
            println!("{indentation} {}: {}", line.label, line.cycle_count);
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
            let indentation = vec!["  "; line.call_stack_depth].join("");
            println!("{indentation} {}: {}", line.label, line.cycle_count);
        }

        let maybe_open_call = profile.iter().find(|line| line.label.contains("(open)"));
        assert!(maybe_open_call.is_some());
    }
}
