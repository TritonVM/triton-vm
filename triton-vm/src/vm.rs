use std::collections::HashMap;
use std::collections::VecDeque;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::ops::Deref;
use std::ops::Range;

use air::table::hash::PermutationTrace;
use air::table::processor::NUM_HELPER_VARIABLE_REGISTERS;
use air::table_column::MasterMainColumn;
use air::table_column::ProcessorMainColumn;
use arbitrary::Arbitrary;
use isa::error::AssertionError;
use isa::error::InstructionError;
use isa::error::OpStackError;
use isa::instruction::Instruction;
use isa::op_stack::NumberOfWords;
use isa::op_stack::OpStack;
use isa::op_stack::OpStackElement;
use isa::op_stack::UnderflowIO;
use isa::program::Program;
use itertools::Itertools;
use ndarray::Array1;
use num_traits::ConstOne;
use num_traits::ConstZero;
use num_traits::Zero;
use serde::Deserialize;
use serde::Serialize;
use strum::EnumCount;
use twenty_first::math::x_field_element::EXTENSION_DEGREE;
use twenty_first::prelude::*;
use twenty_first::util_types::sponge;

use crate::aet::AlgebraicExecutionTrace;
use crate::error::VMError;
use crate::execution_trace_profiler::ExecutionTraceProfile;
use crate::execution_trace_profiler::ExecutionTraceProfiler;
use crate::profiler::profiler;
use crate::table::op_stack::OpStackTableEntry;
use crate::table::ram::RamTableCall;
use crate::table::u32::U32TableEntry;

type VMResult<T> = Result<T, VMError>;
type InstructionResult<T> = Result<T, InstructionError>;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize, Deserialize, Arbitrary)]
pub struct VM;

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize, Arbitrary)]
pub struct VMState {
    /// The **program memory** stores the instructions (and their arguments) of
    /// the program currently being executed by Triton VM. It is read-only.
    pub program: Program,

    /// A list of [`BFieldElement`]s the program can read from using instruction
    /// `read_io`.
    pub public_input: VecDeque<BFieldElement>,

    /// A list of [`BFieldElement`]s the program can write to using instruction
    /// `write_io`.
    pub public_output: Vec<BFieldElement>,

    /// A list of [`BFieldElement`]s the program can read from using instruction
    /// `divine`.
    pub secret_individual_tokens: VecDeque<BFieldElement>,

    /// A list of [`Digest`]s the program can use for instruction `merkle_step`.
    pub secret_digests: VecDeque<Digest>,

    /// The read-write **random-access memory** allows Triton VM to store
    /// arbitrary data.
    pub ram: HashMap<BFieldElement, BFieldElement>,

    ram_calls: Vec<RamTableCall>,

    /// The **Op-stack memory** stores Triton VM's entire operational stack.
    pub op_stack: OpStack,

    /// The **Jump-stack memory** stores the entire jump stack.
    pub jump_stack: Vec<(BFieldElement, BFieldElement)>,

    /// Number of cycles the program has been running for
    pub cycle_count: u32,

    /// Current instruction's address in program memory
    pub instruction_pointer: usize,

    /// The current state of the one, global [`Sponge`] that can be manipulated
    /// using instructions [`SpongeInit`][init], [`SpongeAbsorb`][absorb],
    /// [`SpongeAbsorbMem`][absorb_mem], and [`SpongeSqueeze`][squeeze].
    /// Instruction [`SpongeInit`][init] resets the Sponge.
    ///
    /// Note that this is the _full_ state, including capacity. The capacity
    /// should never be exposed outside the VM.
    ///
    /// [init]: Instruction::SpongeInit
    /// [absorb]: Instruction::SpongeAbsorb
    /// [absorb_mem]: Instruction::SpongeAbsorbMem
    /// [squeeze]: Instruction::SpongeSqueeze
    pub sponge: Option<Tip5>,

    /// Indicates whether the terminating instruction `halt` has been executed.
    pub halting: bool,
}

/// A call from the main processor to one of the coprocessors, including the
/// trace for that coprocessor or enough information to deduce the trace.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum CoProcessorCall {
    SpongeStateReset,

    /// Trace of the state registers for hash coprocessor table when executing
    /// instruction `hash` or one of the Sponge instructions `sponge_absorb`,
    /// `sponge_absorb_mem`, and `sponge_squeeze`.
    ///
    /// One row per round in the Tip5 permutation.
    Tip5Trace(Instruction, Box<PermutationTrace>),

    U32(U32TableEntry),

    OpStack(OpStackTableEntry),

    Ram(RamTableCall),
}

impl VM {
    /// Run Triton VM on the [`Program`] with the given public input and
    /// non-determinism. If an error is encountered, the returned
    /// [`VMError`] contains the [`VMState`] at the point of execution
    /// failure.
    ///
    /// See also [`trace_execution`][trace_execution] and [`profile`][profile].
    ///
    /// [trace_execution]: Self::trace_execution
    /// [profile]: Self::profile
    pub fn run(
        program: Program,
        public_input: PublicInput,
        non_determinism: NonDeterminism,
    ) -> VMResult<Vec<BFieldElement>> {
        let mut state = VMState::new(program, public_input, non_determinism);
        if let Err(err) = state.run() {
            return Err(VMError::new(err, state));
        }
        Ok(state.public_output)
    }

    /// Trace the execution of a [`Program`]. That is, [`run`][run] the
    /// [`Program`] and additionally record that part of every encountered
    /// state that is necessary for proving correct execution. If execution
    /// succeeds, returns
    /// 1. an [`AlgebraicExecutionTrace`], and
    /// 1. the output of the program.
    ///
    /// See also [`run`][run] and [`profile`][profile].
    ///
    /// [run]: Self::run
    /// [profile]: Self::profile
    pub fn trace_execution(
        program: Program,
        public_input: PublicInput,
        non_determinism: NonDeterminism,
    ) -> VMResult<(AlgebraicExecutionTrace, Vec<BFieldElement>)> {
        profiler!(start "trace execution" ("gen"));
        let state = VMState::new(program, public_input, non_determinism);
        let (aet, terminal_state) = Self::trace_execution_of_state(state)?;
        profiler!(stop "trace execution");
        Ok((aet, terminal_state.public_output))
    }

    /// Trace the execution of a [`Program`] from a given [`VMState`]. Consider
    /// using [`trace_execution`][Self::trace_execution], unless you know this
    /// is what you want.
    ///
    /// Returns the [`AlgebraicExecutionTrace`] and the terminal [`VMState`] if
    /// execution succeeds.
    pub fn trace_execution_of_state(
        mut state: VMState,
    ) -> VMResult<(AlgebraicExecutionTrace, VMState)> {
        let mut aet = AlgebraicExecutionTrace::new(state.program.clone());

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
    /// [`AlgebraicExecutionTrace`]. For example, this can be used to identify
    /// the number of clock cycles spent in some block of instructions, or
    /// how many rows that block of instructions contributes to the U32 Table.
    ///
    /// See also [`run`][run] and [`trace_execution`][trace_execution].
    ///
    /// [run]: Self::run
    /// [trace_execution]: Self::trace_execution
    pub fn profile(
        program: Program,
        public_input: PublicInput,
        non_determinism: NonDeterminism,
    ) -> VMResult<(Vec<BFieldElement>, ExecutionTraceProfile)> {
        let mut profiler = ExecutionTraceProfiler::new();
        let mut state = VMState::new(program.clone(), public_input, non_determinism);
        let mut previous_jump_stack_len = state.jump_stack.len();
        let mut aet = AlgebraicExecutionTrace::new(state.program.clone());
        while !state.halting {
            if let Err(err) = aet.record_state(&state) {
                return Err(VMError::new(err, state));
            };
            if let Ok(Instruction::Call(address)) = state.current_instruction() {
                let label = program.label_for_address(address.value());
                profiler.enter_span(label, &aet);
            }

            let co_processor_calls = match state.step() {
                Ok(calls) => calls,
                Err(err) => return Err(VMError::new(err, state)),
            };
            for call in co_processor_calls {
                aet.record_co_processor_call(call);
            }

            if state.jump_stack.len() < previous_jump_stack_len {
                profiler.exit_span(&aet);
            }
            previous_jump_stack_len = state.jump_stack.len();
        }

        Ok((state.public_output, profiler.finish(&aet)))
    }
}

impl VMState {
    /// Create initial `VMState` for a given `program`.
    pub fn new(
        program: Program,
        public_input: PublicInput,
        non_determinism: NonDeterminism,
    ) -> Self {
        let program_digest = program.hash();

        Self {
            program,
            public_input: public_input.individual_tokens.into(),
            public_output: vec![],
            secret_individual_tokens: non_determinism.individual_tokens.into(),
            secret_digests: non_determinism.digests.into(),
            ram: non_determinism.ram,
            ram_calls: vec![],
            op_stack: OpStack::new(program_digest),
            jump_stack: vec![],
            cycle_count: 0,
            instruction_pointer: 0,
            sponge: None,
            halting: false,
        }
    }

    pub fn derive_helper_variables(&self) -> [BFieldElement; NUM_HELPER_VARIABLE_REGISTERS] {
        let mut hvs = bfe_array![0; NUM_HELPER_VARIABLE_REGISTERS];
        let Ok(current_instruction) = self.current_instruction() else {
            return hvs;
        };

        let decompose_arg = |a: u64| bfe_array![a % 2, (a >> 1) % 2, (a >> 2) % 2, (a >> 3) % 2];
        let ram_read = |address| self.ram.get(&address).copied().unwrap_or_else(|| bfe!(0));

        match current_instruction {
            Instruction::Pop(_)
            | Instruction::Divine(_)
            | Instruction::Pick(_)
            | Instruction::Place(_)
            | Instruction::Dup(_)
            | Instruction::Swap(_)
            | Instruction::ReadMem(_)
            | Instruction::WriteMem(_)
            | Instruction::ReadIo(_)
            | Instruction::WriteIo(_) => {
                let arg = current_instruction.arg().unwrap().value();
                hvs[..4].copy_from_slice(&decompose_arg(arg));
            }
            Instruction::Skiz => {
                let st0 = self.op_stack[0];
                hvs[0] = st0.inverse_or_zero();
                let next_opcode = self.next_instruction_or_argument().value();
                let decomposition = Self::decompose_opcode_for_instruction_skiz(next_opcode);
                hvs[1..6].copy_from_slice(&decomposition);
            }
            Instruction::RecurseOrReturn => {
                hvs[0] = (self.op_stack[6] - self.op_stack[5]).inverse_or_zero()
            }
            Instruction::SpongeAbsorbMem => {
                hvs[0] = ram_read(self.op_stack[0] + bfe!(4));
                hvs[1] = ram_read(self.op_stack[0] + bfe!(5));
                hvs[2] = ram_read(self.op_stack[0] + bfe!(6));
                hvs[3] = ram_read(self.op_stack[0] + bfe!(7));
                hvs[4] = ram_read(self.op_stack[0] + bfe!(8));
                hvs[5] = ram_read(self.op_stack[0] + bfe!(9));
            }
            Instruction::MerkleStep => {
                let divined_digest = self.secret_digests.front().copied().unwrap_or_default();
                let node_index = self.op_stack[5].value();
                hvs[..5].copy_from_slice(&divined_digest.values());
                hvs[5] = bfe!(node_index % 2);
            }
            Instruction::MerkleStepMem => {
                let node_index = self.op_stack[5].value();
                let ram_pointer = self.op_stack[7];
                hvs[0] = ram_read(ram_pointer);
                hvs[1] = ram_read(ram_pointer + bfe!(1));
                hvs[2] = ram_read(ram_pointer + bfe!(2));
                hvs[3] = ram_read(ram_pointer + bfe!(3));
                hvs[4] = ram_read(ram_pointer + bfe!(4));
                hvs[5] = bfe!(node_index % 2);
            }
            Instruction::Split => {
                let top_of_stack = self.op_stack[0].value();
                let lo = bfe!(top_of_stack & 0xffff_ffff);
                let hi = bfe!(top_of_stack >> 32);
                if !lo.is_zero() {
                    let max_val_of_hi = bfe!(2_u64.pow(32) - 1);
                    hvs[0] = (hi - max_val_of_hi).inverse_or_zero();
                }
            }
            Instruction::Eq => hvs[0] = (self.op_stack[1] - self.op_stack[0]).inverse_or_zero(),
            Instruction::XxDotStep => {
                hvs[0] = ram_read(self.op_stack[0]);
                hvs[1] = ram_read(self.op_stack[0] + bfe!(1));
                hvs[2] = ram_read(self.op_stack[0] + bfe!(2));
                hvs[3] = ram_read(self.op_stack[1]);
                hvs[4] = ram_read(self.op_stack[1] + bfe!(1));
                hvs[5] = ram_read(self.op_stack[1] + bfe!(2));
            }
            Instruction::XbDotStep => {
                hvs[0] = ram_read(self.op_stack[0]);
                hvs[1] = ram_read(self.op_stack[1]);
                hvs[2] = ram_read(self.op_stack[1] + bfe!(1));
                hvs[3] = ram_read(self.op_stack[1] + bfe!(2));
            }
            _ => (),
        }

        hvs
    }

    fn decompose_opcode_for_instruction_skiz(opcode: u64) -> [BFieldElement; 5] {
        bfe_array![
            opcode % 2,
            (opcode >> 1) % 4,
            (opcode >> 3) % 4,
            (opcode >> 5) % 4,
            opcode >> 7,
        ]
    }

    /// Perform the state transition as a mutable operation on `self`.
    pub fn step(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        if self.halting {
            return Err(InstructionError::MachineHalted);
        }

        let current_instruction = self.current_instruction()?;
        let op_stack_delta = current_instruction.op_stack_size_influence();
        if self.op_stack.would_be_too_shallow(op_stack_delta) {
            return Err(InstructionError::OpStackError(OpStackError::TooShallow));
        }

        self.start_recording_op_stack_calls();
        let mut co_processor_calls = match current_instruction {
            Instruction::Pop(n) => self.pop(n)?,
            Instruction::Push(field_element) => self.push(field_element),
            Instruction::Divine(n) => self.divine(n)?,
            Instruction::Pick(stack_element) => self.pick(stack_element),
            Instruction::Place(stack_element) => self.place(stack_element)?,
            Instruction::Dup(stack_element) => self.dup(stack_element),
            Instruction::Swap(stack_element) => self.swap(stack_element),
            Instruction::Halt => self.halt(),
            Instruction::Nop => self.nop(),
            Instruction::Skiz => self.skiz()?,
            Instruction::Call(address) => self.call(address),
            Instruction::Return => self.return_from_call()?,
            Instruction::Recurse => self.recurse()?,
            Instruction::RecurseOrReturn => self.recurse_or_return()?,
            Instruction::Assert => self.assert()?,
            Instruction::ReadMem(n) => self.read_mem(n)?,
            Instruction::WriteMem(n) => self.write_mem(n)?,
            Instruction::Hash => self.hash()?,
            Instruction::SpongeInit => self.sponge_init(),
            Instruction::SpongeAbsorb => self.sponge_absorb()?,
            Instruction::SpongeAbsorbMem => self.sponge_absorb_mem()?,
            Instruction::SpongeSqueeze => self.sponge_squeeze()?,
            Instruction::AssertVector => self.assert_vector()?,
            Instruction::Add => self.add()?,
            Instruction::AddI(field_element) => self.addi(field_element),
            Instruction::Mul => self.mul()?,
            Instruction::Invert => self.invert()?,
            Instruction::Eq => self.eq()?,
            Instruction::Split => self.split()?,
            Instruction::Lt => self.lt()?,
            Instruction::And => self.and()?,
            Instruction::Xor => self.xor()?,
            Instruction::Log2Floor => self.log_2_floor()?,
            Instruction::Pow => self.pow()?,
            Instruction::DivMod => self.div_mod()?,
            Instruction::PopCount => self.pop_count()?,
            Instruction::XxAdd => self.xx_add()?,
            Instruction::XxMul => self.xx_mul()?,
            Instruction::XInvert => self.x_invert()?,
            Instruction::XbMul => self.xb_mul()?,
            Instruction::WriteIo(n) => self.write_io(n)?,
            Instruction::ReadIo(n) => self.read_io(n)?,
            Instruction::MerkleStep => self.merkle_step_non_determinism()?,
            Instruction::MerkleStepMem => self.merkle_step_mem()?,
            Instruction::XxDotStep => self.xx_dot_step()?,
            Instruction::XbDotStep => self.xb_dot_step()?,
        };
        let op_stack_calls = self.stop_recording_op_stack_calls();
        co_processor_calls.extend(op_stack_calls);

        self.cycle_count += 1;

        Ok(co_processor_calls)
    }

    fn start_recording_op_stack_calls(&mut self) {
        self.op_stack.start_recording_underflow_io_sequence();
    }

    fn stop_recording_op_stack_calls(&mut self) -> Vec<CoProcessorCall> {
        let sequence = self.op_stack.stop_recording_underflow_io_sequence();
        self.underflow_io_sequence_to_co_processor_calls(sequence)
    }

    fn underflow_io_sequence_to_co_processor_calls(
        &self,
        underflow_io_sequence: Vec<UnderflowIO>,
    ) -> Vec<CoProcessorCall> {
        let op_stack_table_entries = OpStackTableEntry::from_underflow_io_sequence(
            self.cycle_count,
            self.op_stack.pointer(),
            underflow_io_sequence,
        );
        op_stack_table_entries
            .into_iter()
            .map(CoProcessorCall::OpStack)
            .collect()
    }

    fn start_recording_ram_calls(&mut self) {
        self.ram_calls.clear();
    }

    fn stop_recording_ram_calls(&mut self) -> Vec<CoProcessorCall> {
        self.ram_calls.drain(..).map(CoProcessorCall::Ram).collect()
    }

    fn pop(&mut self, n: NumberOfWords) -> InstructionResult<Vec<CoProcessorCall>> {
        for _ in 0..n.num_words() {
            self.op_stack.pop()?;
        }

        self.instruction_pointer += 2;
        Ok(vec![])
    }

    fn push(&mut self, element: BFieldElement) -> Vec<CoProcessorCall> {
        self.op_stack.push(element);

        self.instruction_pointer += 2;
        vec![]
    }

    fn divine(&mut self, n: NumberOfWords) -> InstructionResult<Vec<CoProcessorCall>> {
        let input_len = self.secret_individual_tokens.len();
        if input_len < n.num_words() {
            return Err(InstructionError::EmptySecretInput(input_len));
        }
        for _ in 0..n.num_words() {
            let element = self.secret_individual_tokens.pop_front().unwrap();
            self.op_stack.push(element);
        }

        self.instruction_pointer += 2;
        Ok(vec![])
    }

    fn pick(&mut self, stack_register: OpStackElement) -> Vec<CoProcessorCall> {
        let element = self.op_stack.remove(stack_register);
        self.op_stack.push(element);

        self.instruction_pointer += 2;
        vec![]
    }

    fn place(&mut self, stack_register: OpStackElement) -> InstructionResult<Vec<CoProcessorCall>> {
        let element = self.op_stack.pop()?;
        self.op_stack.insert(stack_register, element);

        self.instruction_pointer += 2;
        Ok(vec![])
    }

    fn dup(&mut self, stack_register: OpStackElement) -> Vec<CoProcessorCall> {
        let element = self.op_stack[stack_register];
        self.op_stack.push(element);

        self.instruction_pointer += 2;
        vec![]
    }

    fn swap(&mut self, st: OpStackElement) -> Vec<CoProcessorCall> {
        (self.op_stack[0], self.op_stack[st]) = (self.op_stack[st], self.op_stack[0]);
        self.instruction_pointer += 2;
        vec![]
    }

    fn nop(&mut self) -> Vec<CoProcessorCall> {
        self.instruction_pointer += 1;
        vec![]
    }

    fn skiz(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        let top_of_stack = self.op_stack.pop()?;
        self.instruction_pointer += match top_of_stack.is_zero() {
            true => 1 + self.next_instruction()?.size(),
            false => 1,
        };
        Ok(vec![])
    }

    fn call(&mut self, call_destination: BFieldElement) -> Vec<CoProcessorCall> {
        let size_of_instruction_call = 2;
        let call_origin = (self.instruction_pointer as u32 + size_of_instruction_call).into();
        let jump_stack_entry = (call_origin, call_destination);
        self.jump_stack.push(jump_stack_entry);

        self.instruction_pointer = call_destination.value().try_into().unwrap();
        vec![]
    }

    fn return_from_call(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        let (call_origin, _) = self.jump_stack_pop()?;
        self.instruction_pointer = call_origin.value().try_into().unwrap();
        Ok(vec![])
    }

    fn recurse(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        let (_, call_destination) = self.jump_stack_peek()?;
        self.instruction_pointer = call_destination.value().try_into().unwrap();
        Ok(vec![])
    }

    fn recurse_or_return(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        if self.jump_stack.is_empty() {
            return Err(InstructionError::JumpStackIsEmpty);
        }

        let new_ip = if self.op_stack[5] == self.op_stack[6] {
            let (call_origin, _) = self.jump_stack_pop()?;
            call_origin
        } else {
            let (_, call_destination) = self.jump_stack_peek()?;
            call_destination
        };

        self.instruction_pointer = new_ip.value().try_into().unwrap();

        Ok(vec![])
    }

    fn assert(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        let actual = self.op_stack[0];
        let expected = BFieldElement::ONE;
        if actual != expected {
            let error = self.contextualized_assertion_error(expected, actual);
            return Err(InstructionError::AssertionFailed(error));
        }
        let _ = self.op_stack.pop()?;

        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn halt(&mut self) -> Vec<CoProcessorCall> {
        self.halting = true;
        self.instruction_pointer += 1;
        vec![]
    }

    fn read_mem(&mut self, n: NumberOfWords) -> InstructionResult<Vec<CoProcessorCall>> {
        self.start_recording_ram_calls();
        let mut ram_pointer = self.op_stack.pop()?;
        for _ in 0..n.num_words() {
            let ram_value = self.ram_read(ram_pointer);
            self.op_stack.push(ram_value);
            ram_pointer.decrement();
        }
        self.op_stack.push(ram_pointer);
        let ram_calls = self.stop_recording_ram_calls();

        self.instruction_pointer += 2;
        Ok(ram_calls)
    }

    fn write_mem(&mut self, n: NumberOfWords) -> InstructionResult<Vec<CoProcessorCall>> {
        self.start_recording_ram_calls();
        let mut ram_pointer = self.op_stack.pop()?;
        for _ in 0..n.num_words() {
            let ram_value = self.op_stack.pop()?;
            self.ram_write(ram_pointer, ram_value);
            ram_pointer.increment();
        }
        self.op_stack.push(ram_pointer);
        let ram_calls = self.stop_recording_ram_calls();

        self.instruction_pointer += 2;
        Ok(ram_calls)
    }

    fn ram_read(&mut self, ram_pointer: BFieldElement) -> BFieldElement {
        let ram_value = self
            .ram
            .get(&ram_pointer)
            .copied()
            .unwrap_or(BFieldElement::ZERO);

        let ram_table_call = RamTableCall {
            clk: self.cycle_count,
            ram_pointer,
            ram_value,
            is_write: false,
        };
        self.ram_calls.push(ram_table_call);

        ram_value
    }

    fn ram_write(&mut self, ram_pointer: BFieldElement, ram_value: BFieldElement) {
        let ram_table_call = RamTableCall {
            clk: self.cycle_count,
            ram_pointer,
            ram_value,
            is_write: true,
        };
        self.ram_calls.push(ram_table_call);

        self.ram.insert(ram_pointer, ram_value);
    }

    fn hash(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        let to_hash = self.op_stack.pop_multiple::<{ tip5::RATE }>()?;

        let mut hash_input = Tip5::new(sponge::Domain::FixedLength);
        hash_input.state[..tip5::RATE].copy_from_slice(&to_hash);
        let tip5_trace = hash_input.trace();
        let hash_output = &tip5_trace[tip5_trace.len() - 1][0..Digest::LEN];

        for i in (0..Digest::LEN).rev() {
            self.op_stack.push(hash_output[i]);
        }

        let co_processor_calls = vec![CoProcessorCall::Tip5Trace(
            Instruction::Hash,
            Box::new(tip5_trace),
        )];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn sponge_init(&mut self) -> Vec<CoProcessorCall> {
        self.sponge = Some(Tip5::init());
        self.instruction_pointer += 1;
        vec![CoProcessorCall::SpongeStateReset]
    }

    fn sponge_absorb(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        let Some(ref mut sponge) = self.sponge else {
            return Err(InstructionError::SpongeNotInitialized);
        };
        let to_absorb = self.op_stack.pop_multiple::<{ tip5::RATE }>()?;
        sponge.state[..tip5::RATE].copy_from_slice(&to_absorb);
        let tip5_trace = sponge.trace();

        let co_processor_calls = vec![CoProcessorCall::Tip5Trace(
            Instruction::SpongeAbsorb,
            Box::new(tip5_trace),
        )];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn sponge_absorb_mem(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        let Some(mut sponge) = self.sponge.take() else {
            return Err(InstructionError::SpongeNotInitialized);
        };

        self.start_recording_ram_calls();
        let mut mem_pointer = self.op_stack.pop()?;
        for i in 0..tip5::RATE {
            let element = self.ram_read(mem_pointer);
            mem_pointer.increment();
            sponge.state[i] = element;

            // not enough helper variables – overwrite part of the stack :(
            if i < tip5::RATE - NUM_HELPER_VARIABLE_REGISTERS {
                self.op_stack[i] = element;
            }
        }
        self.op_stack.push(mem_pointer);

        let tip5_trace = sponge.trace();
        self.sponge = Some(sponge);

        let mut co_processor_calls = self.stop_recording_ram_calls();
        co_processor_calls.push(CoProcessorCall::Tip5Trace(
            Instruction::SpongeAbsorb,
            Box::new(tip5_trace),
        ));

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn sponge_squeeze(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        let Some(ref mut sponge) = self.sponge else {
            return Err(InstructionError::SpongeNotInitialized);
        };
        for i in (0..tip5::RATE).rev() {
            self.op_stack.push(sponge.state[i]);
        }
        let tip5_trace = sponge.trace();

        let co_processor_calls = vec![CoProcessorCall::Tip5Trace(
            Instruction::SpongeSqueeze,
            Box::new(tip5_trace),
        )];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn assert_vector(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        for i in 0..Digest::LEN {
            let expected = self.op_stack[i];
            let actual = self.op_stack[i + Digest::LEN];
            if expected != actual {
                let error = self.contextualized_assertion_error(expected, actual);
                return Err(InstructionError::VectorAssertionFailed(i, error));
            }
        }
        self.op_stack.pop_multiple::<{ Digest::LEN }>()?;
        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn add(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        let lhs = self.op_stack.pop()?;
        let rhs = self.op_stack.pop()?;
        self.op_stack.push(lhs + rhs);

        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn addi(&mut self, i: BFieldElement) -> Vec<CoProcessorCall> {
        self.op_stack[0] += i;
        self.instruction_pointer += 2;
        vec![]
    }

    fn mul(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        let lhs = self.op_stack.pop()?;
        let rhs = self.op_stack.pop()?;
        self.op_stack.push(lhs * rhs);

        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn invert(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        let top_of_stack = self.op_stack[0];
        if top_of_stack.is_zero() {
            return Err(InstructionError::InverseOfZero);
        }
        let _ = self.op_stack.pop()?;
        self.op_stack.push(top_of_stack.inverse());
        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn eq(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        let lhs = self.op_stack.pop()?;
        let rhs = self.op_stack.pop()?;
        let eq: u32 = (lhs == rhs).into();
        self.op_stack.push(eq.into());

        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn split(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        let top_of_stack = self.op_stack.pop()?;
        let lo = bfe!(top_of_stack.value() & 0xffff_ffff);
        let hi = bfe!(top_of_stack.value() >> 32);
        self.op_stack.push(hi);
        self.op_stack.push(lo);

        let u32_table_entry = U32TableEntry::new(Instruction::Split, lo, hi);
        let co_processor_calls = vec![CoProcessorCall::U32(u32_table_entry)];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn lt(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        self.op_stack.is_u32(OpStackElement::ST0)?;
        self.op_stack.is_u32(OpStackElement::ST1)?;
        let lhs = self.op_stack.pop_u32()?;
        let rhs = self.op_stack.pop_u32()?;
        let lt: u32 = (lhs < rhs).into();
        self.op_stack.push(lt.into());

        let u32_table_entry = U32TableEntry::new(Instruction::Lt, lhs, rhs);
        let co_processor_calls = vec![CoProcessorCall::U32(u32_table_entry)];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn and(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        self.op_stack.is_u32(OpStackElement::ST0)?;
        self.op_stack.is_u32(OpStackElement::ST1)?;
        let lhs = self.op_stack.pop_u32()?;
        let rhs = self.op_stack.pop_u32()?;
        let and = lhs & rhs;
        self.op_stack.push(and.into());

        let u32_table_entry = U32TableEntry::new(Instruction::And, lhs, rhs);
        let co_processor_calls = vec![CoProcessorCall::U32(u32_table_entry)];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn xor(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        self.op_stack.is_u32(OpStackElement::ST0)?;
        self.op_stack.is_u32(OpStackElement::ST1)?;
        let lhs = self.op_stack.pop_u32()?;
        let rhs = self.op_stack.pop_u32()?;
        let xor = lhs ^ rhs;
        self.op_stack.push(xor.into());

        // Triton VM uses the following equality to compute the results of both
        // the `and` and `xor` instruction using the u32 coprocessor's `and`
        // capability: a ^ b = a + b - 2 · (a & b)
        let u32_table_entry = U32TableEntry::new(Instruction::And, lhs, rhs);
        let co_processor_calls = vec![CoProcessorCall::U32(u32_table_entry)];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn log_2_floor(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        self.op_stack.is_u32(OpStackElement::ST0)?;
        let top_of_stack = self.op_stack[0];
        if top_of_stack.is_zero() {
            return Err(InstructionError::LogarithmOfZero);
        }
        let top_of_stack = self.op_stack.pop_u32()?;
        let log_2_floor = top_of_stack.ilog2();
        self.op_stack.push(log_2_floor.into());

        let u32_table_entry = U32TableEntry::new(Instruction::Log2Floor, top_of_stack, 0);
        let co_processor_calls = vec![CoProcessorCall::U32(u32_table_entry)];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn pow(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        self.op_stack.is_u32(OpStackElement::ST1)?;
        let base = self.op_stack.pop()?;
        let exponent = self.op_stack.pop_u32()?;
        let base_pow_exponent = base.mod_pow(exponent.into());
        self.op_stack.push(base_pow_exponent);

        let u32_table_entry = U32TableEntry::new(Instruction::Pow, base, exponent);
        let co_processor_calls = vec![CoProcessorCall::U32(u32_table_entry)];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn div_mod(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        self.op_stack.is_u32(OpStackElement::ST0)?;
        self.op_stack.is_u32(OpStackElement::ST1)?;
        let denominator = self.op_stack[1];
        if denominator.is_zero() {
            return Err(InstructionError::DivisionByZero);
        }

        let numerator = self.op_stack.pop_u32()?;
        let denominator = self.op_stack.pop_u32()?;
        let quotient = numerator / denominator;
        let remainder = numerator % denominator;

        self.op_stack.push(quotient.into());
        self.op_stack.push(remainder.into());

        let remainder_is_less_than_denominator =
            U32TableEntry::new(Instruction::Lt, remainder, denominator);
        let numerator_and_quotient_range_check =
            U32TableEntry::new(Instruction::Split, numerator, quotient);
        let co_processor_calls = vec![
            CoProcessorCall::U32(remainder_is_less_than_denominator),
            CoProcessorCall::U32(numerator_and_quotient_range_check),
        ];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn pop_count(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        self.op_stack.is_u32(OpStackElement::ST0)?;
        let top_of_stack = self.op_stack.pop_u32()?;
        let pop_count = top_of_stack.count_ones();
        self.op_stack.push(pop_count.into());

        let u32_table_entry = U32TableEntry::new(Instruction::PopCount, top_of_stack, 0);
        let co_processor_calls = vec![CoProcessorCall::U32(u32_table_entry)];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn xx_add(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        let lhs = self.op_stack.pop_extension_field_element()?;
        let rhs = self.op_stack.pop_extension_field_element()?;
        self.op_stack.push_extension_field_element(lhs + rhs);
        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn xx_mul(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        let lhs = self.op_stack.pop_extension_field_element()?;
        let rhs = self.op_stack.pop_extension_field_element()?;
        self.op_stack.push_extension_field_element(lhs * rhs);
        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn x_invert(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        let top_of_stack = self.op_stack.peek_at_top_extension_field_element();
        if top_of_stack.is_zero() {
            return Err(InstructionError::InverseOfZero);
        }
        let inverse = top_of_stack.inverse();
        let _ = self.op_stack.pop_extension_field_element()?;
        self.op_stack.push_extension_field_element(inverse);
        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn xb_mul(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        let lhs = self.op_stack.pop()?;
        let rhs = self.op_stack.pop_extension_field_element()?;
        self.op_stack.push_extension_field_element(lhs.lift() * rhs);

        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn write_io(&mut self, n: NumberOfWords) -> InstructionResult<Vec<CoProcessorCall>> {
        for _ in 0..n.num_words() {
            let top_of_stack = self.op_stack.pop()?;
            self.public_output.push(top_of_stack);
        }

        self.instruction_pointer += 2;
        Ok(vec![])
    }

    fn read_io(&mut self, n: NumberOfWords) -> InstructionResult<Vec<CoProcessorCall>> {
        let input_len = self.public_input.len();
        if input_len < n.num_words() {
            return Err(InstructionError::EmptyPublicInput(input_len));
        }
        for _ in 0..n.num_words() {
            let read_element = self.public_input.pop_front().unwrap();
            self.op_stack.push(read_element);
        }

        self.instruction_pointer += 2;
        Ok(vec![])
    }

    fn merkle_step_non_determinism(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        self.op_stack.is_u32(OpStackElement::ST5)?;
        let sibling_digest = self.pop_secret_digest()?;
        self.merkle_step(sibling_digest)
    }

    fn merkle_step_mem(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        self.op_stack.is_u32(OpStackElement::ST5)?;
        self.start_recording_ram_calls();
        let mut ram_pointer = self.op_stack[7];
        let Digest(mut sibling_digest) = Digest::default();
        for digest_element in &mut sibling_digest {
            *digest_element = self.ram_read(ram_pointer);
            ram_pointer.increment();
        }
        self.op_stack[7] = ram_pointer;

        let mut co_processor_calls = self.merkle_step(sibling_digest)?;
        co_processor_calls.extend(self.stop_recording_ram_calls());
        Ok(co_processor_calls)
    }

    fn merkle_step(
        &mut self,
        sibling_digest: [BFieldElement; Digest::LEN],
    ) -> InstructionResult<Vec<CoProcessorCall>> {
        let node_index = self.op_stack.get_u32(OpStackElement::ST5)?;
        let parent_node_index = node_index / 2;

        let accumulator_digest = self.op_stack.pop_multiple::<{ Digest::LEN }>()?;
        let (left_sibling, right_sibling) = match node_index % 2 {
            0 => (accumulator_digest, sibling_digest),
            1 => (sibling_digest, accumulator_digest),
            _ => unreachable!(),
        };

        let mut tip5 = Tip5::new(sponge::Domain::FixedLength);
        tip5.state[..Digest::LEN].copy_from_slice(&left_sibling);
        tip5.state[Digest::LEN..2 * Digest::LEN].copy_from_slice(&right_sibling);
        let tip5_trace = tip5.trace();
        let accumulator_digest = &tip5_trace.last().unwrap()[0..Digest::LEN];

        for &digest_element in accumulator_digest.iter().rev() {
            self.op_stack.push(digest_element);
        }
        self.op_stack[5] = parent_node_index.into();

        self.instruction_pointer += 1;

        let hash_co_processor_call =
            CoProcessorCall::Tip5Trace(Instruction::Hash, Box::new(tip5_trace));
        let indices_are_u32 = CoProcessorCall::U32(U32TableEntry::new(
            Instruction::Split,
            node_index,
            parent_node_index,
        ));
        let co_processor_calls = vec![hash_co_processor_call, indices_are_u32];
        Ok(co_processor_calls)
    }

    fn xx_dot_step(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        self.start_recording_ram_calls();
        let mut rhs_address = self.op_stack.pop()?;
        let mut lhs_address = self.op_stack.pop()?;
        let mut rhs = xfe!(0);
        let mut lhs = xfe!(0);
        for i in 0..EXTENSION_DEGREE {
            rhs.coefficients[i] = self.ram_read(rhs_address);
            rhs_address.increment();
            lhs.coefficients[i] = self.ram_read(lhs_address);
            lhs_address.increment();
        }
        let accumulator = self.op_stack.pop_extension_field_element()? + rhs * lhs;
        self.op_stack.push_extension_field_element(accumulator);
        self.op_stack.push(lhs_address);
        self.op_stack.push(rhs_address);
        self.instruction_pointer += 1;
        let ram_calls = self.stop_recording_ram_calls();
        Ok(ram_calls)
    }

    fn xb_dot_step(&mut self) -> InstructionResult<Vec<CoProcessorCall>> {
        self.start_recording_ram_calls();
        let mut rhs_address = self.op_stack.pop()?;
        let mut lhs_address = self.op_stack.pop()?;
        let rhs = self.ram_read(rhs_address);
        rhs_address.increment();
        let mut lhs = xfe!(0);
        for i in 0..EXTENSION_DEGREE {
            lhs.coefficients[i] = self.ram_read(lhs_address);
            lhs_address.increment();
        }
        let accumulator = self.op_stack.pop_extension_field_element()? + rhs * lhs;
        self.op_stack.push_extension_field_element(accumulator);
        self.op_stack.push(lhs_address);
        self.op_stack.push(rhs_address);
        self.instruction_pointer += 1;
        let ram_calls = self.stop_recording_ram_calls();
        Ok(ram_calls)
    }

    pub fn to_processor_row(&self) -> Array1<BFieldElement> {
        use ProcessorMainColumn as Col;
        use isa::instruction::InstructionBit;

        assert!(
            self.op_stack.len() >= OpStackElement::COUNT,
            "unrecoverable internal error: Triton VM's stack is too shallow",
        );

        let mut row = Array1::zeros(Col::COUNT);
        row[Col::CLK.main_index()] = u64::from(self.cycle_count).into();
        row[Col::IP.main_index()] = (self.instruction_pointer as u32).into();

        let current_instruction = self.current_instruction().unwrap_or(Instruction::Nop);
        row[Col::CI.main_index()] = current_instruction.opcode_b();
        row[Col::NIA.main_index()] = self.next_instruction_or_argument();
        row[Col::IB0.main_index()] = current_instruction.ib(InstructionBit::IB0);
        row[Col::IB1.main_index()] = current_instruction.ib(InstructionBit::IB1);
        row[Col::IB2.main_index()] = current_instruction.ib(InstructionBit::IB2);
        row[Col::IB3.main_index()] = current_instruction.ib(InstructionBit::IB3);
        row[Col::IB4.main_index()] = current_instruction.ib(InstructionBit::IB4);
        row[Col::IB5.main_index()] = current_instruction.ib(InstructionBit::IB5);
        row[Col::IB6.main_index()] = current_instruction.ib(InstructionBit::IB6);

        row[Col::JSP.main_index()] = self.jump_stack_pointer();
        row[Col::JSO.main_index()] = self.jump_stack_origin();
        row[Col::JSD.main_index()] = self.jump_stack_destination();
        row[Col::ST0.main_index()] = self.op_stack[0];
        row[Col::ST1.main_index()] = self.op_stack[1];
        row[Col::ST2.main_index()] = self.op_stack[2];
        row[Col::ST3.main_index()] = self.op_stack[3];
        row[Col::ST4.main_index()] = self.op_stack[4];
        row[Col::ST5.main_index()] = self.op_stack[5];
        row[Col::ST6.main_index()] = self.op_stack[6];
        row[Col::ST7.main_index()] = self.op_stack[7];
        row[Col::ST8.main_index()] = self.op_stack[8];
        row[Col::ST9.main_index()] = self.op_stack[9];
        row[Col::ST10.main_index()] = self.op_stack[10];
        row[Col::ST11.main_index()] = self.op_stack[11];
        row[Col::ST12.main_index()] = self.op_stack[12];
        row[Col::ST13.main_index()] = self.op_stack[13];
        row[Col::ST14.main_index()] = self.op_stack[14];
        row[Col::ST15.main_index()] = self.op_stack[15];
        row[Col::OpStackPointer.main_index()] = self.op_stack.pointer();

        let helper_variables = self.derive_helper_variables();
        row[Col::HV0.main_index()] = helper_variables[0];
        row[Col::HV1.main_index()] = helper_variables[1];
        row[Col::HV2.main_index()] = helper_variables[2];
        row[Col::HV3.main_index()] = helper_variables[3];
        row[Col::HV4.main_index()] = helper_variables[4];
        row[Col::HV5.main_index()] = helper_variables[5];

        row
    }

    /// The “next instruction or argument” (NIA) is
    /// - the argument of the current instruction if it has one, or
    /// - the opcode of the next instruction otherwise.
    ///
    /// If the current instruction has no argument and there is no next
    /// instruction, the NIA is 1 to account for the hash-input padding
    /// separator of the program.
    ///
    /// If the instruction pointer is out of bounds, the returned NIA is 0.
    fn next_instruction_or_argument(&self) -> BFieldElement {
        let Ok(current_instruction) = self.current_instruction() else {
            return bfe!(0);
        };
        if let Some(argument) = current_instruction.arg() {
            return argument;
        }
        match self.next_instruction() {
            Ok(next_instruction) => next_instruction.opcode_b(),
            Err(_) => bfe!(1),
        }
    }

    fn jump_stack_pointer(&self) -> BFieldElement {
        (self.jump_stack.len() as u64).into()
    }

    fn jump_stack_origin(&self) -> BFieldElement {
        let maybe_origin = self.jump_stack.last().map(|(o, _d)| *o);
        maybe_origin.unwrap_or_else(BFieldElement::zero)
    }

    fn jump_stack_destination(&self) -> BFieldElement {
        let maybe_destination = self.jump_stack.last().map(|(_o, d)| *d);
        maybe_destination.unwrap_or_else(BFieldElement::zero)
    }

    pub fn current_instruction(&self) -> InstructionResult<Instruction> {
        let instructions = &self.program.instructions;
        let maybe_current_instruction = instructions.get(self.instruction_pointer).copied();
        maybe_current_instruction.ok_or(InstructionError::InstructionPointerOverflow)
    }

    /// Return the next instruction on the tape, skipping arguments.
    ///
    /// Note that this is not necessarily the next instruction to execute, since
    /// the current instruction could be a jump, but it is either
    /// `program.instructions[ip + 1]` or `program.instructions[ip + 2]`,
    /// depending on whether the current instruction takes an argument.
    pub fn next_instruction(&self) -> InstructionResult<Instruction> {
        let current_instruction = self.current_instruction()?;
        let next_instruction_pointer = self.instruction_pointer + current_instruction.size();
        let instructions = &self.program.instructions;
        let maybe_next_instruction = instructions.get(next_instruction_pointer).copied();
        maybe_next_instruction.ok_or(InstructionError::InstructionPointerOverflow)
    }

    fn jump_stack_pop(&mut self) -> InstructionResult<(BFieldElement, BFieldElement)> {
        self.jump_stack
            .pop()
            .ok_or(InstructionError::JumpStackIsEmpty)
    }

    fn jump_stack_peek(&mut self) -> InstructionResult<(BFieldElement, BFieldElement)> {
        self.jump_stack
            .last()
            .copied()
            .ok_or(InstructionError::JumpStackIsEmpty)
    }

    fn pop_secret_digest(&mut self) -> InstructionResult<[BFieldElement; Digest::LEN]> {
        let digest = self
            .secret_digests
            .pop_front()
            .ok_or(InstructionError::EmptySecretDigestInput)?;
        Ok(digest.values())
    }

    /// Run Triton VM on this state to completion, or until an error occurs.
    pub fn run(&mut self) -> InstructionResult<()> {
        while !self.halting {
            self.step()?;
        }
        Ok(())
    }

    fn contextualized_assertion_error(
        &self,
        expected: BFieldElement,
        actual: BFieldElement,
    ) -> AssertionError {
        let current_address =
            u64::try_from(self.instruction_pointer).expect("usize should fit in u64");

        let error = AssertionError::new(expected, actual);
        if let Some(context) = self.program.assertion_context_at(current_address) {
            error.with_context(context)
        } else {
            error
        }
    }
}

impl Display for VMState {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        use ProcessorMainColumn as ProcCol;

        let display_instruction = |instruction: Instruction| {
            if let Instruction::Call(address) = instruction {
                format!("call {}", self.program.label_for_address(address.value()))
            } else {
                instruction.to_string()
            }
        };

        let instruction = self
            .current_instruction()
            .map_or_else(|_| "--".to_string(), display_instruction)
            .chars()
            .take(54)
            .collect::<String>();

        let total_width = 103;
        let tab_width = instruction.chars().count().max(8);
        let clk_width = 17;
        let register_width = 20;
        let buffer_width = total_width - tab_width - clk_width - 7;

        let print_row = |f: &mut Formatter, s: String| writeln!(f, "│ {s: <total_width$} │");
        let print_blank_row = |f: &mut Formatter| print_row(f, String::new());
        let print_section_separator = |f: &mut Formatter| writeln!(f, "├─{:─<total_width$}─┤", "");

        let row = self.to_processor_row();

        let register =
            |col: ProcCol| format!("{:>register_width$}", row[col.main_index()].to_string());
        let multi_register = |regs: [_; 4]| regs.map(register).join(" | ");

        writeln!(f)?;
        writeln!(f, " ╭─{:─<tab_width$}─╮", "")?;
        writeln!(f, " │ {instruction: <tab_width$} │")?;
        writeln!(
            f,
            "╭┴─{:─<tab_width$}─┴─{:─<buffer_width$}─┬─{:─>clk_width$}─╮",
            "", "", ""
        )?;

        let ip = register(ProcCol::IP);
        let ci = register(ProcCol::CI);
        let nia = register(ProcCol::NIA);
        let jsp = register(ProcCol::JSP);
        let jso = register(ProcCol::JSO);
        let jsd = register(ProcCol::JSD);
        let osp = register(ProcCol::OpStackPointer);
        let clk = row[ProcCol::CLK.main_index()].to_string();
        let clk = clk.trim_start_matches('0');

        let first_line = format!("ip:   {ip} ╷ ci:   {ci} ╷ nia: {nia} │ {clk: >clk_width$}");
        print_row(f, first_line)?;
        writeln!(
            f,
            "│ jsp:  {jsp} │ jso:  {jso} │ jsd: {jsd} ╰─{:─>clk_width$}─┤",
            "",
        )?;
        print_row(f, format!("osp:  {osp} ╵"))?;
        print_blank_row(f)?;

        let st_00_03 = multi_register([ProcCol::ST0, ProcCol::ST1, ProcCol::ST2, ProcCol::ST3]);
        let st_04_07 = multi_register([ProcCol::ST4, ProcCol::ST5, ProcCol::ST6, ProcCol::ST7]);
        let st_08_11 = multi_register([ProcCol::ST8, ProcCol::ST9, ProcCol::ST10, ProcCol::ST11]);
        let st_12_15 = multi_register([ProcCol::ST12, ProcCol::ST13, ProcCol::ST14, ProcCol::ST15]);

        print_row(f, format!("st0-3:    [ {st_00_03} ]"))?;
        print_row(f, format!("st4-7:    [ {st_04_07} ]"))?;
        print_row(f, format!("st8-11:   [ {st_08_11} ]"))?;
        print_row(f, format!("st12-15:  [ {st_12_15} ]"))?;
        print_blank_row(f)?;

        let hv_00_03_line = format!(
            "hv0-3:    [ {} ]",
            multi_register([ProcCol::HV0, ProcCol::HV1, ProcCol::HV2, ProcCol::HV3])
        );
        let hv_04_05_line = format!(
            "hv4-5:    [ {} | {} ]",
            register(ProcCol::HV4),
            register(ProcCol::HV5),
        );
        print_row(f, hv_00_03_line)?;
        print_row(f, hv_04_05_line)?;

        let ib_registers = [
            ProcCol::IB6,
            ProcCol::IB5,
            ProcCol::IB4,
            ProcCol::IB3,
            ProcCol::IB2,
            ProcCol::IB1,
            ProcCol::IB0,
        ]
        .map(|reg| row[reg.main_index()])
        .map(|bfe| format!("{bfe:>2}"))
        .join(" | ");
        print_row(f, format!("ib6-0:    [ {ib_registers} ]"))?;

        print_section_separator(f)?;
        if let Some(ref sponge) = self.sponge {
            let sponge_state_slice = |idxs: Range<usize>| {
                idxs.map(|i| sponge.state[i].value())
                    .map(|ss| format!("{ss:>register_width$}"))
                    .join(" | ")
            };

            print_row(f, format!("sp0-3:    [ {} ]", sponge_state_slice(0..4)))?;
            print_row(f, format!("sp4-7:    [ {} ]", sponge_state_slice(4..8)))?;
            print_row(f, format!("sp8-11:   [ {} ]", sponge_state_slice(8..12)))?;
            print_row(f, format!("sp12-15:  [ {} ]", sponge_state_slice(12..16)))?;
        } else {
            let uninit_msg = format!("{:^total_width$}", "-- sponge is not initialized --");
            print_row(f, uninit_msg)?;
        };

        print_section_separator(f)?;
        if self.jump_stack.is_empty() {
            print_row(f, format!("{:^total_width$}", "-- jump stack is empty --"))?;
        } else {
            let idx_width = 3;
            let max_label_width = total_width - idx_width - 2; // for `: `

            for (idx, &(_, address)) in self.jump_stack.iter().rev().enumerate() {
                let label = self.program.label_for_address(address.value());
                let label = label.chars().take(max_label_width).collect::<String>();
                print_row(f, format!("{idx:>idx_width$}: {label}"))?;
                print_row(f, format!("        at {address}"))?;
            }
        }

        if self.halting {
            print_section_separator(f)?;
            print_row(f, format!("{:^total_width$}", "! halting !"))?;
        }

        writeln!(f, "╰─{:─<total_width$}─╯", "")
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

impl From<PublicInput> for Vec<BFieldElement> {
    fn from(value: PublicInput) -> Self {
        value.individual_tokens
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

impl Deref for PublicInput {
    type Target = [BFieldElement];

    fn deref(&self) -> &Self::Target {
        &self.individual_tokens
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
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) mod tests {
    use std::ops::BitAnd;
    use std::ops::BitXor;

    use assert2::assert;
    use assert2::let_assert;
    use isa::instruction::ALL_INSTRUCTIONS;
    use isa::instruction::AnInstruction;
    use isa::instruction::LabelledInstruction;
    use isa::op_stack::NUM_OP_STACK_REGISTERS;
    use isa::program::Program;
    use isa::triton_asm;
    use isa::triton_instr;
    use isa::triton_program;
    use itertools::izip;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use rand::Rng;
    use rand::RngCore;
    use rand::rngs::ThreadRng;
    use strum::EnumCount;
    use strum::EnumIter;
    use test_strategy::proptest;
    use twenty_first::math::other::random_elements;

    use crate::shared_tests::LeavedMerkleTreeTestData;
    use crate::shared_tests::TestableProgram;
    use crate::stark::Stark;
    use crate::stark::tests::program_executing_every_instruction;

    use super::*;

    #[test]
    fn instructions_act_on_op_stack_as_indicated() {
        for test_instruction in ALL_INSTRUCTIONS {
            let (program, stack_size_before_test_instruction) =
                construct_test_program_for_instruction(test_instruction);
            let public_input = PublicInput::from(bfe_array![0]);
            let mock_digests = [Digest::default()];
            let non_determinism = NonDeterminism::from(bfe_array![0]).with_digests(mock_digests);

            let mut vm_state = VMState::new(program, public_input, non_determinism);
            let_assert!(Ok(()) = vm_state.run());
            let stack_size_after_test_instruction = vm_state.op_stack.len();

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
        if matches!(
            instruction,
            Instruction::Call(_)
                | Instruction::Return
                | Instruction::Recurse
                | Instruction::RecurseOrReturn
        ) {
            // need jump stack setup
            let program = test_program_for_call_recurse_return().program;
            let stack_size = NUM_OP_STACK_REGISTERS;
            (program, stack_size)
        } else {
            let num_push_instructions = 10;
            let push_instructions = triton_asm![push 1; num_push_instructions];
            let program = triton_program!(sponge_init {&push_instructions} {instruction} nop halt);

            let stack_size_when_reaching_test_instruction =
                NUM_OP_STACK_REGISTERS + num_push_instructions;
            (program, stack_size_when_reaching_test_instruction)
        }
    }

    #[proptest]
    fn from_various_types_to_public_input(#[strategy(arb())] tokens: Vec<BFieldElement>) {
        let public_input = PublicInput::new(tokens.clone());

        assert!(public_input == tokens.clone().into());
        assert!(public_input == (&tokens).into());
        assert!(public_input == tokens[..].into());
        assert!(public_input == (&tokens[..]).into());
        assert!(tokens == <Vec<_>>::from(public_input));

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
    fn initialise_table() {
        let program = crate::example_programs::GREATEST_COMMON_DIVISOR.clone();
        let stdin = PublicInput::from([42, 56].map(|b| bfe!(b)));
        let secret_in = NonDeterminism::default();
        VM::trace_execution(program, stdin, secret_in).unwrap();
    }

    #[test]
    fn run_tvm_gcd() {
        let program = crate::example_programs::GREATEST_COMMON_DIVISOR.clone();
        let stdin = PublicInput::from([42, 56].map(|b| bfe!(b)));
        let secret_in = NonDeterminism::default();
        let_assert!(Ok(stdout) = VM::run(program, stdin, secret_in));

        let output = stdout.iter().map(|o| format!("{o}")).join(", ");
        println!("VM output: [{output}]");

        assert!(bfe!(14) == stdout[0]);
    }

    #[test]
    fn crash_triton_vm_and_print_vm_error() {
        let crashing_program = triton_program!(push 2 assert halt);
        let_assert!(Err(err) = VM::run(crashing_program, [].into(), [].into()));
        println!("{err}");
    }

    #[test]
    fn crash_triton_vm_with_non_empty_jump_stack_and_print_vm_error() {
        let crashing_program = triton_program! {
            call foo halt
            foo: call bar return
            bar: push 2 assert return
        };
        let_assert!(Err(err) = VM::run(crashing_program, [].into(), [].into()));
        let err_str = err.to_string();

        let_assert!(Some(bar_pos) = err_str.find("bar"));
        let_assert!(Some(foo_pos) = err_str.find("foo"));
        assert!(bar_pos < foo_pos, "deeper call must be listed higher");

        println!("{err_str}");
    }

    #[test]
    fn print_various_vm_states() {
        let TestableProgram {
            program,
            public_input,
            non_determinism,
            ..
        } = program_executing_every_instruction();
        let mut state = VMState::new(program, public_input, non_determinism);
        while !state.halting {
            println!("{state}");
            state.step().unwrap();
        }
    }

    #[test]
    fn print_vm_state_with_long_jump_stack() {
        let labels = [
            "astraldropper_",
            "bongoquest_",
            "cloudmother_",
            "daydream_",
            "essence_",
            "flowerflight_",
            "groovesister_",
            "highride_",
            "meadowdream_",
            "naturesounds_",
            "opaldancer_",
            "peacespirit_",
            "quyhrmfields_",
        ]
        .map(|s| s.repeat(15));

        let crashing_program = triton_program! {
            call {labels[0]}
            {labels[0]}:  call {labels[1]}
            {labels[1]}:  call {labels[2]}
            {labels[2]}:  call {labels[3]}
            {labels[3]}:  call {labels[4]}
            {labels[4]}:  call {labels[5]}
            {labels[5]}:  call {labels[6]}
            {labels[6]}:  call {labels[7]}
            {labels[7]}:  call {labels[8]}
            {labels[8]}:  call {labels[9]}
            {labels[9]}:  call {labels[10]}
            {labels[10]}: call {labels[11]}
            {labels[11]}: call {labels[12]}
            {labels[12]}: nop
        };

        let_assert!(Err(err) = VM::run(crashing_program, [].into(), [].into()));
        println!("{err}");
    }

    pub(crate) fn test_program_hash_nop_nop_lt() -> TestableProgram {
        let push_5_zeros = triton_asm![push 0; 5];
        let program = triton_program! {
            {&push_5_zeros} hash
            nop
            {&push_5_zeros} hash
            nop nop
            {&push_5_zeros} hash
            push 3 push 2 lt assert
            halt
        };
        TestableProgram::new(program)
    }

    pub(crate) fn test_program_for_halt() -> TestableProgram {
        TestableProgram::new(triton_program!(halt))
    }

    pub(crate) fn test_program_for_push_pop_dup_swap_nop() -> TestableProgram {
        TestableProgram::new(triton_program!(
            push 1 push 2 pop 1 assert
            push 1 dup  0 assert assert
            push 1 push 2 swap 1 assert pop 1
            nop nop nop halt
        ))
    }

    pub(crate) fn test_program_for_divine() -> TestableProgram {
        TestableProgram::new(triton_program!(divine 1 assert halt)).with_non_determinism([bfe!(1)])
    }

    pub(crate) fn test_program_for_skiz() -> TestableProgram {
        TestableProgram::new(triton_program!(push 1 skiz push 0 skiz assert push 1 skiz halt))
    }

    pub(crate) fn test_program_for_call_recurse_return() -> TestableProgram {
        TestableProgram::new(triton_program!(
            push 2
            call label
            pop 1 halt
            label:
                push -1 add dup 0
                skiz
                    recurse
                return
        ))
    }

    pub(crate) fn test_program_for_recurse_or_return() -> TestableProgram {
        TestableProgram::new(triton_program! {
            push 5 swap 5
            push 0 swap 5
            call label
            halt
            label:
                swap 5
                push 1 add
                swap 5
                recurse_or_return
        })
    }

    /// Test helper for property testing instruction `recurse_or_return`.
    ///
    /// The [assembled](Self::assemble) program
    /// - sets up a loop counter,
    /// - populates ST6 with some “iteration terminator”,
    /// - reads successive elements from standard input, and
    /// - compares them to the iteration terminator using `recurse_or_return`.
    ///
    /// The program halts after the loop has run for the expected number of
    /// iterations, crashing the VM if the number of iterations does not match
    /// expectations.
    #[derive(Debug, Clone, Eq, PartialEq, test_strategy::Arbitrary)]
    pub struct ProgramForRecurseOrReturn {
        #[strategy(arb())]
        iteration_terminator: BFieldElement,

        #[strategy(arb())]
        #[filter(#other_iterator_values.iter().all(|&v| v != #iteration_terminator))]
        other_iterator_values: Vec<BFieldElement>,
    }

    impl ProgramForRecurseOrReturn {
        pub fn assemble(self) -> TestableProgram {
            let expected_num_iterations = self.other_iterator_values.len() + 1;

            let program = triton_program! {
                // set up iteration counter
                push 0 hint iteration_counter = stack[0]

                // set up termination condition
                push {self.iteration_terminator}
                swap 6

                call iteration_loop

                // check iteration counter
                push {expected_num_iterations}
                eq assert
                halt

                iteration_loop:
                    // increment iteration counter
                    push 1 add

                    // check loop termination
                    swap 5
                    pop 1
                    read_io 1
                    swap 5
                    recurse_or_return
            };

            let mut input = self.other_iterator_values;
            input.push(self.iteration_terminator);
            TestableProgram::new(program).with_input(input)
        }
    }

    #[proptest]
    fn property_based_recurse_or_return_program_sanity_check(program: ProgramForRecurseOrReturn) {
        program.assemble().run()?;
    }

    #[test]
    fn vm_crashes_when_executing_recurse_or_return_with_empty_jump_stack() {
        let program = triton_program!(recurse_or_return halt);
        let_assert!(Err(err) = VM::run(program, PublicInput::default(), NonDeterminism::default()));
        assert!(InstructionError::JumpStackIsEmpty == err.source);
    }

    pub(crate) fn test_program_for_write_mem_read_mem() -> TestableProgram {
        TestableProgram::new(triton_program! {
            push 3 push 1 push 2    // _ 3 1 2
            push 7                  // _ 3 1 2 7
            write_mem 3             // _ 10
            push -1 add             // _ 9
            read_mem 2              // _ 3 1 7
            pop 1                   // _ 3 1
            assert halt             // _ 3
        })
    }

    pub(crate) fn test_program_for_hash() -> TestableProgram {
        let program = triton_program!(
            push 0 // filler to keep the OpStack large enough throughout the program
            push 0 push 0 push 1 push 2 push 3
            hash
            read_io 1 eq assert halt
        );
        let hash_input = bfe_array![3, 2, 1, 0, 0, 0, 0, 0, 0, 0];
        let digest = Tip5::hash_10(&hash_input);
        TestableProgram::new(program).with_input(&digest[..=0])
    }

    /// Helper function that returns code to push a digest to the top of the
    /// stack
    fn push_digest_to_stack_tasm(Digest([d0, d1, d2, d3, d4]): Digest) -> Vec<LabelledInstruction> {
        triton_asm!(push {d4} push {d3} push {d2} push {d1} push {d0})
    }

    pub(crate) fn test_program_for_merkle_step_right_sibling() -> TestableProgram {
        let accumulator_digest = Digest::new(bfe_array![2, 12, 22, 32, 42]);
        let divined_digest = Digest::new(bfe_array![10, 11, 12, 13, 14]);
        let expected_digest = Tip5::hash_pair(divined_digest, accumulator_digest);
        let merkle_tree_node_index = 3;
        let program = triton_program!(
            push {merkle_tree_node_index}
            {&push_digest_to_stack_tasm(accumulator_digest)}
            merkle_step

            {&push_digest_to_stack_tasm(expected_digest)}
            assert_vector pop 5
            assert halt
        );

        let non_determinism = NonDeterminism::default().with_digests(vec![divined_digest]);
        TestableProgram::new(program).with_non_determinism(non_determinism)
    }

    pub(crate) fn test_program_for_merkle_step_left_sibling() -> TestableProgram {
        let accumulator_digest = Digest::new(bfe_array![2, 12, 22, 32, 42]);
        let divined_digest = Digest::new(bfe_array![10, 11, 12, 13, 14]);
        let expected_digest = Tip5::hash_pair(accumulator_digest, divined_digest);
        let merkle_tree_node_index = 2;
        let program = triton_program!(
            push {merkle_tree_node_index}
            {&push_digest_to_stack_tasm(accumulator_digest)}
            merkle_step

            {&push_digest_to_stack_tasm(expected_digest)}
            assert_vector pop 5
            assert halt
        );

        let non_determinism = NonDeterminism::default().with_digests(vec![divined_digest]);
        TestableProgram::new(program).with_non_determinism(non_determinism)
    }

    pub(crate) fn test_program_for_merkle_step_mem_right_sibling() -> TestableProgram {
        let accumulator_digest = Digest::new(bfe_array![2, 12, 22, 32, 42]);
        let sibling_digest = Digest::new(bfe_array![10, 11, 12, 13, 14]);
        let expected_digest = Tip5::hash_pair(sibling_digest, accumulator_digest);
        let auth_path_address = 1337;
        let merkle_tree_node_index = 3;
        let program = triton_program!(
            push {auth_path_address}
            push 0 // dummy
            push {merkle_tree_node_index}
            {&push_digest_to_stack_tasm(accumulator_digest)}
            merkle_step_mem

            {&push_digest_to_stack_tasm(expected_digest)}
            assert_vector pop 5
            assert halt
        );

        let ram = (auth_path_address..)
            .map(BFieldElement::new)
            .zip(sibling_digest.values())
            .collect::<HashMap<_, _>>();
        let non_determinism = NonDeterminism::default().with_ram(ram);
        TestableProgram::new(program).with_non_determinism(non_determinism)
    }

    pub(crate) fn test_program_for_merkle_step_mem_left_sibling() -> TestableProgram {
        let accumulator_digest = Digest::new(bfe_array![2, 12, 22, 32, 42]);
        let sibling_digest = Digest::new(bfe_array![10, 11, 12, 13, 14]);
        let expected_digest = Tip5::hash_pair(accumulator_digest, sibling_digest);
        let auth_path_address = 1337;
        let merkle_tree_node_index = 2;
        let program = triton_program!(
            push {auth_path_address}
            push 0 // dummy
            push {merkle_tree_node_index}
            {&push_digest_to_stack_tasm(accumulator_digest)}
            merkle_step_mem

            {&push_digest_to_stack_tasm(expected_digest)}
            assert_vector pop 5
            assert halt
        );

        let ram = (auth_path_address..)
            .map(BFieldElement::new)
            .zip(sibling_digest.values())
            .collect::<HashMap<_, _>>();
        let non_determinism = NonDeterminism::default().with_ram(ram);
        TestableProgram::new(program).with_non_determinism(non_determinism)
    }

    /// Test helper for property-testing instruction `merkle_step_mem` in a
    /// meaningful context, namely, using
    /// [`MERKLE_TREE_UPDATE`](crate::example_programs::MERKLE_TREE_UPDATE).
    #[derive(Debug, Clone, test_strategy::Arbitrary)]
    pub struct ProgramForMerkleTreeUpdate {
        leaved_merkle_tree: LeavedMerkleTreeTestData,

        #[strategy(0..#leaved_merkle_tree.revealed_indices.len())]
        #[map(|i| #leaved_merkle_tree.revealed_indices[i])]
        revealed_leafs_index: usize,

        #[strategy(arb())]
        new_leaf: Digest,

        #[strategy(arb())]
        auth_path_address: BFieldElement,
    }

    impl ProgramForMerkleTreeUpdate {
        pub fn assemble(self) -> TestableProgram {
            let auth_path = self.authentication_path();
            let leaf_index = self.revealed_leafs_index;
            let merkle_tree = self.leaved_merkle_tree.merkle_tree;

            let ram = (self.auth_path_address.value()..)
                .map(BFieldElement::new)
                .zip(auth_path.iter().flat_map(|digest| digest.values()))
                .collect::<HashMap<_, _>>();
            let non_determinism = NonDeterminism::default().with_ram(ram);

            let old_leaf = Digest::from(self.leaved_merkle_tree.leaves[leaf_index]);
            let old_root = merkle_tree.root();
            let mut public_input =
                bfe_vec![self.auth_path_address, leaf_index, merkle_tree.height()];
            public_input.extend(old_leaf.reversed().values());
            public_input.extend(old_root.reversed().values());
            public_input.extend(self.new_leaf.reversed().values());

            TestableProgram::new(crate::example_programs::MERKLE_TREE_UPDATE.clone())
                .with_input(public_input)
                .with_non_determinism(non_determinism)
        }

        /// Checks whether the [`TestableProgram`] generated through
        /// [`Self::assemble`] can
        /// - be executed without crashing the VM, and
        /// - produces the correct output.
        #[must_use]
        pub fn is_integral(&self) -> bool {
            let inclusion_proof = MerkleTreeInclusionProof {
                tree_height: self.leaved_merkle_tree.merkle_tree.height(),
                indexed_leafs: vec![(self.revealed_leafs_index, self.new_leaf)],
                authentication_structure: self.authentication_path(),
            };

            let new_root = self.clone().assemble().run().unwrap();
            let new_root = Digest(new_root.try_into().unwrap());
            inclusion_proof.verify(new_root)
        }

        /// The authentication path for the leaf at `self.revealed_leafs_index`.
        /// Independent of the leaf's value, _i.e._, is up to date even one the
        /// leaf has been updated.
        fn authentication_path(&self) -> Vec<Digest> {
            self.leaved_merkle_tree
                .merkle_tree
                .authentication_structure(&[self.revealed_leafs_index])
                .unwrap()
        }
    }

    pub(crate) fn test_program_for_assert_vector() -> TestableProgram {
        TestableProgram::new(triton_program!(
            push 1 push 2 push 3 push 4 push 5
            push 1 push 2 push 3 push 4 push 5
            assert_vector halt
        ))
    }

    pub(crate) fn test_program_for_sponge_instructions() -> TestableProgram {
        let push_10_zeros = triton_asm![push 0; 10];
        TestableProgram::new(triton_program!(
            sponge_init
            {&push_10_zeros} sponge_absorb
            {&push_10_zeros} sponge_absorb
            sponge_squeeze halt
        ))
    }

    pub(crate) fn test_program_for_sponge_instructions_2() -> TestableProgram {
        let push_5_zeros = triton_asm![push 0; 5];
        let program = triton_program! {
            sponge_init
            sponge_squeeze sponge_absorb
            {&push_5_zeros} hash
            sponge_squeeze sponge_absorb
            halt
        };
        TestableProgram::new(program)
    }

    pub(crate) fn test_program_for_many_sponge_instructions() -> TestableProgram {
        let push_5_zeros = triton_asm![push 0; 5];
        let push_10_zeros = triton_asm![push 0; 10];
        let program = triton_program! {         //  elements on stack
            sponge_init                         //  0
            sponge_squeeze sponge_absorb        //  0
            {&push_10_zeros} sponge_absorb      //  0
            {&push_10_zeros} sponge_absorb      //  0
            sponge_squeeze sponge_squeeze       // 20
            sponge_squeeze sponge_absorb        // 20
            sponge_init sponge_init sponge_init // 20
            sponge_absorb                       // 10
            sponge_init                         // 10
            sponge_squeeze sponge_squeeze       // 30
            sponge_init sponge_squeeze          // 40
            {&push_5_zeros} hash sponge_absorb  // 30
            {&push_5_zeros} hash sponge_squeeze // 40
            {&push_5_zeros} hash sponge_absorb  // 30
            {&push_5_zeros} hash sponge_squeeze // 40
            sponge_init                         // 40
            sponge_absorb sponge_absorb         // 20
            sponge_absorb sponge_absorb         //  0
            {&push_10_zeros} sponge_absorb      //  0
            {&push_10_zeros} sponge_absorb      //  0
            {&push_10_zeros} sponge_absorb      //  0
            halt
        };
        TestableProgram::new(program)
    }

    pub(crate) fn property_based_test_program_for_assert_vector() -> TestableProgram {
        let mut rng = ThreadRng::default();
        let st: [BFieldElement; 5] = rng.random();

        let program = triton_program!(
            push {st[0]} push {st[1]} push {st[2]} push {st[3]} push {st[4]}
            read_io 5 assert_vector halt
        );

        TestableProgram::new(program).with_input(st)
    }

    /// Test helper for [`ProgramForSpongeAndHashInstructions`].
    #[derive(Debug, Copy, Clone, Eq, PartialEq, EnumCount, EnumIter, test_strategy::Arbitrary)]
    enum SpongeAndHashInstructions {
        SpongeInit,
        SpongeAbsorb(#[strategy(arb())] [BFieldElement; tip5::RATE]),
        SpongeAbsorbMem(#[strategy(arb())] BFieldElement),
        SpongeSqueeze,
        Hash(#[strategy(arb())] [BFieldElement; tip5::RATE]),
    }

    impl SpongeAndHashInstructions {
        fn to_vm_instruction_sequence(self) -> Vec<Instruction> {
            let push_array =
                |a: [_; tip5::RATE]| a.into_iter().rev().map(Instruction::Push).collect_vec();

            match self {
                Self::SpongeInit => vec![Instruction::SpongeInit],
                Self::SpongeAbsorb(input) => {
                    [push_array(input), vec![Instruction::SpongeAbsorb]].concat()
                }
                Self::SpongeAbsorbMem(ram_pointer) => {
                    vec![Instruction::Push(ram_pointer), Instruction::SpongeAbsorbMem]
                }
                Self::SpongeSqueeze => vec![Instruction::SpongeSqueeze],
                Self::Hash(input) => [push_array(input), vec![Instruction::Hash]].concat(),
            }
        }

        fn execute(self, sponge: &mut Tip5, ram: &HashMap<BFieldElement, BFieldElement>) {
            let ram_read = |p| ram.get(&p).copied().unwrap_or_else(|| bfe!(0));
            let ram_read_array = |p| {
                let ram_reads = (0..tip5::RATE as u32).map(|i| ram_read(p + bfe!(i)));
                ram_reads.collect_vec().try_into().unwrap()
            };

            match self {
                Self::SpongeInit => *sponge = Tip5::init(),
                Self::SpongeAbsorb(input) => sponge.absorb(input),
                Self::SpongeAbsorbMem(ram_pointer) => sponge.absorb(ram_read_array(ram_pointer)),
                Self::SpongeSqueeze => _ = sponge.squeeze(),
                Self::Hash(_) => (), // no effect on Sponge
            }
        }
    }

    /// Test helper for arbitrary sequences of hashing-related instructions.
    #[derive(Debug, Clone, Eq, PartialEq, test_strategy::Arbitrary)]
    pub struct ProgramForSpongeAndHashInstructions {
        instructions: Vec<SpongeAndHashInstructions>,

        #[strategy(arb())]
        ram: HashMap<BFieldElement, BFieldElement>,
    }

    impl ProgramForSpongeAndHashInstructions {
        pub fn assemble(self) -> TestableProgram {
            let mut sponge = Tip5::init();
            for instruction in &self.instructions {
                instruction.execute(&mut sponge, &self.ram);
            }
            let expected_rate = sponge.squeeze();
            let non_determinism = NonDeterminism::default().with_ram(self.ram);

            let sponge_and_hash_instructions = self
                .instructions
                .into_iter()
                .flat_map(|i| i.to_vm_instruction_sequence())
                .collect_vec();
            let output_equality_assertions =
                vec![triton_asm![read_io 1 eq assert]; tip5::RATE].concat();

            let program = triton_program! {
                sponge_init
                {&sponge_and_hash_instructions}
                sponge_squeeze
                {&output_equality_assertions}
                halt
            };

            TestableProgram::new(program)
                .with_input(expected_rate)
                .with_non_determinism(non_determinism)
        }
    }

    #[proptest]
    fn property_based_sponge_and_hash_instructions_program_sanity_check(
        program: ProgramForSpongeAndHashInstructions,
    ) {
        program.assemble().run()?;
    }

    pub(crate) fn test_program_for_add_mul_invert() -> TestableProgram {
        TestableProgram::new(triton_program!(
            push  2 push -1 add assert
            push -1 push -1 mul assert
            push  5 addi -4 assert
            push  3 dup 0 invert mul assert
            halt
        ))
    }

    pub(crate) fn property_based_test_program_for_split() -> TestableProgram {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u64() % BFieldElement::P;
        let hi = st0 >> 32;
        let lo = st0 & 0xffff_ffff;

        let program =
            triton_program!(push {st0} split read_io 1 eq assert read_io 1 eq assert halt);
        TestableProgram::new(program).with_input(bfe_array![lo, hi])
    }

    pub(crate) fn test_program_for_eq() -> TestableProgram {
        let input = bfe_array![42];
        TestableProgram::new(triton_program!(read_io 1 divine 1 eq assert halt))
            .with_input(input)
            .with_non_determinism(input)
    }

    pub(crate) fn property_based_test_program_for_eq() -> TestableProgram {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u64() % BFieldElement::P;
        let input = bfe_array![st0];

        let program =
            triton_program!(push {st0} dup 0 read_io 1 eq assert dup 0 divine 1 eq assert halt);
        TestableProgram::new(program)
            .with_input(input)
            .with_non_determinism(input)
    }

    pub(crate) fn test_program_for_lsb() -> TestableProgram {
        TestableProgram::new(triton_program!(
            push 3 call lsb assert assert halt
            lsb:
                push 2 swap 1 div_mod return
        ))
    }

    pub(crate) fn property_based_test_program_for_lsb() -> TestableProgram {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u32();
        let lsb = st0 % 2;
        let st0_shift_right = st0 >> 1;

        let program = triton_program!(
            push {st0} call lsb read_io 1 eq assert read_io 1 eq assert halt
            lsb:
                push 2 swap 1 div_mod return
        );
        TestableProgram::new(program).with_input([lsb, st0_shift_right].map(|b| bfe!(b)))
    }

    pub(crate) fn test_program_0_lt_0() -> TestableProgram {
        TestableProgram::new(triton_program!(push 0 push 0 lt halt))
    }

    pub(crate) fn test_program_for_lt() -> TestableProgram {
        TestableProgram::new(triton_program!(
            push 5 push 2 lt assert push 2 push 5 lt push 0 eq assert halt
        ))
    }

    pub(crate) fn property_based_test_program_for_lt() -> TestableProgram {
        let mut rng = ThreadRng::default();

        let st1_0 = rng.next_u32();
        let st0_0 = rng.next_u32();
        let result_0 = match st0_0 < st1_0 {
            true => 1,
            false => 0,
        };

        let st1_1 = rng.next_u32();
        let st0_1 = rng.next_u32();
        let result_1 = match st0_1 < st1_1 {
            true => 1,
            false => 0,
        };

        let program = triton_program!(
            push {st1_0} push {st0_0} lt read_io 1 eq assert
            push {st1_1} push {st0_1} lt read_io 1 eq assert halt
        );
        TestableProgram::new(program).with_input([result_0, result_1].map(|b| bfe!(b)))
    }

    pub(crate) fn test_program_for_and() -> TestableProgram {
        TestableProgram::new(triton_program!(
            push 5 push 3 and assert push 12 push 5 and push 4 eq assert halt
        ))
    }

    pub(crate) fn property_based_test_program_for_and() -> TestableProgram {
        let mut rng = ThreadRng::default();

        let st1_0 = rng.next_u32();
        let st0_0 = rng.next_u32();
        let result_0 = st0_0.bitand(st1_0);

        let st1_1 = rng.next_u32();
        let st0_1 = rng.next_u32();
        let result_1 = st0_1.bitand(st1_1);

        let program = triton_program!(
            push {st1_0} push {st0_0} and read_io 1 eq assert
            push {st1_1} push {st0_1} and read_io 1 eq assert halt
        );
        TestableProgram::new(program).with_input([result_0, result_1].map(|b| bfe!(b)))
    }

    pub(crate) fn test_program_for_xor() -> TestableProgram {
        TestableProgram::new(triton_program!(
            push 7 push 6 xor assert push 5 push 12 xor push 9 eq assert halt
        ))
    }

    pub(crate) fn property_based_test_program_for_xor() -> TestableProgram {
        let mut rng = ThreadRng::default();

        let st1_0 = rng.next_u32();
        let st0_0 = rng.next_u32();
        let result_0 = st0_0.bitxor(st1_0);

        let st1_1 = rng.next_u32();
        let st0_1 = rng.next_u32();
        let result_1 = st0_1.bitxor(st1_1);

        let program = triton_program!(
            push {st1_0} push {st0_0} xor read_io 1 eq assert
            push {st1_1} push {st0_1} xor read_io 1 eq assert halt
        );
        TestableProgram::new(program).with_input([result_0, result_1].map(|b| bfe!(b)))
    }

    pub(crate) fn test_program_for_log2floor() -> TestableProgram {
        TestableProgram::new(triton_program!(
            push  1 log_2_floor push  0 eq assert
            push  2 log_2_floor push  1 eq assert
            push  3 log_2_floor push  1 eq assert
            push  4 log_2_floor push  2 eq assert
            push  7 log_2_floor push  2 eq assert
            push  8 log_2_floor push  3 eq assert
            push 15 log_2_floor push  3 eq assert
            push 16 log_2_floor push  4 eq assert
            push 31 log_2_floor push  4 eq assert
            push 32 log_2_floor push  5 eq assert
            push 33 log_2_floor push  5 eq assert
            push 4294967295 log_2_floor push 31 eq assert halt
        ))
    }

    pub(crate) fn property_based_test_program_for_log2floor() -> TestableProgram {
        let mut rng = ThreadRng::default();

        let st0_0 = rng.next_u32();
        let l2f_0 = st0_0.ilog2();

        let st0_1 = rng.next_u32();
        let l2f_1 = st0_1.ilog2();

        let program = triton_program!(
            push {st0_0} log_2_floor read_io 1 eq assert
            push {st0_1} log_2_floor read_io 1 eq assert halt
        );
        TestableProgram::new(program).with_input([l2f_0, l2f_1].map(|b| bfe!(b)))
    }

    pub(crate) fn test_program_for_pow() -> TestableProgram {
        TestableProgram::new(triton_program!(
            // push [exponent] push [base] pow push [result] eq assert
            push  0 push 0 pow push    1 eq assert
            push  0 push 1 pow push    1 eq assert
            push  0 push 2 pow push    1 eq assert
            push  1 push 0 pow push    0 eq assert
            push  2 push 0 pow push    0 eq assert
            push  2 push 5 pow push   25 eq assert
            push  5 push 2 pow push   32 eq assert
            push 10 push 2 pow push 1024 eq assert
            push  3 push 3 pow push   27 eq assert
            push  3 push 5 pow push  125 eq assert
            push  9 push 7 pow push 40353607 eq assert
            push 3040597274 push 05218640216028681988 pow push 11160453713534536216 eq assert
            push 2378067562 push 13711477740065654150 pow push 06848017529532358230 eq assert
            push  129856251 push 00218966585049330803 pow push 08283208434666229347 eq assert
            push 1657936293 push 04999758396092641065 pow push 11426020017566937356 eq assert
            push 3474149688 push 05702231339458623568 pow push 02862889945380025510 eq assert
            push 2243935791 push 09059335263701504667 pow push 04293137302922963369 eq assert
            push 1783029319 push 00037306102533222534 pow push 10002149917806048099 eq assert
            push 3608140376 push 17716542154416103060 pow push 11885601801443303960 eq assert
            push 1220084884 push 07207865095616988291 pow push 05544378138345942897 eq assert
            push 3539668245 push 13491612301110950186 pow push 02612675697712040250 eq assert
            halt
        ))
    }

    pub(crate) fn property_based_test_program_for_pow() -> TestableProgram {
        let mut rng = ThreadRng::default();

        let base_0: BFieldElement = rng.random();
        let exp_0 = rng.next_u32();
        let result_0 = base_0.mod_pow_u32(exp_0);

        let base_1: BFieldElement = rng.random();
        let exp_1 = rng.next_u32();
        let result_1 = base_1.mod_pow_u32(exp_1);

        let program = triton_program!(
            push {exp_0} push {base_0} pow read_io 1 eq assert
            push {exp_1} push {base_1} pow read_io 1 eq assert halt
        );
        TestableProgram::new(program).with_input([result_0, result_1])
    }

    pub(crate) fn test_program_for_div_mod() -> TestableProgram {
        TestableProgram::new(triton_program!(push 2 push 3 div_mod assert assert halt))
    }

    pub(crate) fn property_based_test_program_for_div_mod() -> TestableProgram {
        let mut rng = ThreadRng::default();

        let denominator = rng.next_u32();
        let numerator = rng.next_u32();
        let quotient = numerator / denominator;
        let remainder = numerator % denominator;

        let program = triton_program!(
            push {denominator} push {numerator} div_mod read_io 1 eq assert read_io 1 eq assert halt
        );
        TestableProgram::new(program).with_input([remainder, quotient].map(|b| bfe!(b)))
    }

    pub(crate) fn test_program_for_starting_with_pop_count() -> TestableProgram {
        TestableProgram::new(triton_program!(pop_count dup 0 push 0 eq assert halt))
    }

    pub(crate) fn test_program_for_pop_count() -> TestableProgram {
        TestableProgram::new(triton_program!(push 10 pop_count push 2 eq assert halt))
    }

    pub(crate) fn property_based_test_program_for_pop_count() -> TestableProgram {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u32();
        let pop_count = st0.count_ones();
        let program = triton_program!(push {st0} pop_count read_io 1 eq assert halt);
        TestableProgram::new(program).with_input(bfe_array![pop_count])
    }

    pub(crate) fn property_based_test_program_for_is_u32() -> TestableProgram {
        let mut rng = ThreadRng::default();
        let st0_u32 = rng.next_u32();
        let st0_not_u32 = (u64::from(rng.next_u32()) << 32) + u64::from(rng.next_u32());
        let program = triton_program!(
            push {st0_u32} call is_u32 assert
            push {st0_not_u32} call is_u32 push 0 eq assert halt
            is_u32:
                 split pop 1 push 0 eq return
        );
        TestableProgram::new(program)
    }

    pub(crate) fn property_based_test_program_for_random_ram_access() -> TestableProgram {
        let mut rng = ThreadRng::default();
        let num_memory_accesses = rng.random_range(10..50);
        let memory_addresses: Vec<BFieldElement> = random_elements(num_memory_accesses);
        let mut memory_values: Vec<BFieldElement> = random_elements(num_memory_accesses);
        let mut instructions = vec![];

        // Read some memory before first write to ensure that the memory is
        // initialized with 0s. Not all addresses are read to have different
        // access patterns:
        // - Some addresses are read before written to.
        // - Other addresses are written to before read.
        for address in memory_addresses.iter().take(num_memory_accesses / 4) {
            instructions.extend(triton_asm!(push {address} read_mem 1 pop 1 push 0 eq assert));
        }

        // Write everything to RAM.
        for (address, value) in memory_addresses.iter().zip_eq(memory_values.iter()) {
            instructions.extend(triton_asm!(push {value} push {address} write_mem 1 pop 1));
        }

        // Read back in random order and check that the values did not change.
        let mut reading_permutation = (0..num_memory_accesses).collect_vec();
        for i in 0..num_memory_accesses {
            let j = rng.random_range(0..num_memory_accesses);
            reading_permutation.swap(i, j);
        }
        for idx in reading_permutation {
            let address = memory_addresses[idx];
            let value = memory_values[idx];
            instructions
                .extend(triton_asm!(push {address} read_mem 1 pop 1 push {value} eq assert));
        }

        // Overwrite half the values with new ones.
        let mut writing_permutation = (0..num_memory_accesses).collect_vec();
        for i in 0..num_memory_accesses {
            let j = rng.random_range(0..num_memory_accesses);
            writing_permutation.swap(i, j);
        }
        for idx in 0..num_memory_accesses / 2 {
            let address = memory_addresses[writing_permutation[idx]];
            let new_memory_value = rng.random();
            memory_values[writing_permutation[idx]] = new_memory_value;
            instructions
                .extend(triton_asm!(push {new_memory_value} push {address} write_mem 1 pop 1));
        }

        // Read back all, i.e., unchanged and overwritten values in (different
        // from before) random order and check that the values did not change.
        let mut reading_permutation = (0..num_memory_accesses).collect_vec();
        for i in 0..num_memory_accesses {
            let j = rng.random_range(0..num_memory_accesses);
            reading_permutation.swap(i, j);
        }
        for idx in reading_permutation {
            let address = memory_addresses[idx];
            let value = memory_values[idx];
            instructions
                .extend(triton_asm!(push {address} read_mem 1 pop 1 push {value} eq assert));
        }

        let program = triton_program! { { &instructions } halt };
        TestableProgram::new(program)
    }

    /// Sanity check for the relatively complex property-based test for random
    /// RAM access.
    #[test]
    fn run_dont_prove_property_based_test_for_random_ram_access() {
        let program = property_based_test_program_for_random_ram_access();
        program.run().unwrap();
    }

    #[test]
    fn can_compute_dot_product_from_uninitialized_ram() {
        let program = triton_program!(xx_dot_step xb_dot_step halt);
        VM::run(program, PublicInput::default(), NonDeterminism::default()).unwrap();
    }

    pub(crate) fn property_based_test_program_for_xx_dot_step() -> TestableProgram {
        let mut rng = ThreadRng::default();
        let n = rng.random_range(0..10);

        let push_xfe = |xfe: XFieldElement| {
            let [c_0, c_1, c_2] = xfe.coefficients;
            triton_asm! { push {c_2} push {c_1} push {c_0} }
        };
        let push_and_write_xfe = |xfe| {
            let push_xfe = push_xfe(xfe);
            triton_asm! {
                {&push_xfe}
                dup 3
                write_mem 3
                swap 1
                pop 1
            }
        };
        let into_write_instructions = |elements: Vec<_>| {
            elements
                .into_iter()
                .flat_map(push_and_write_xfe)
                .collect_vec()
        };

        let vector_one = (0..n).map(|_| rng.random()).collect_vec();
        let vector_two = (0..n).map(|_| rng.random()).collect_vec();
        let inner_product = vector_one
            .iter()
            .zip(&vector_two)
            .map(|(&a, &b)| a * b)
            .sum();
        let push_inner_product = push_xfe(inner_product);

        let push_and_write_vector_one = into_write_instructions(vector_one);
        let push_and_write_vector_two = into_write_instructions(vector_two);
        let many_dotsteps = (0..n).map(|_| triton_instr!(xx_dot_step)).collect_vec();

        let code = triton_program! {
            push 0
            {&push_and_write_vector_one}
            dup 0
            {&push_and_write_vector_two}
            pop 1
            push 0

            {&many_dotsteps}
            pop 2
            push 0 push 0

            {&push_inner_product}
            push 0 push 0
            assert_vector
            halt
        };
        TestableProgram::new(code)
    }

    /// Sanity check
    #[test]
    fn run_dont_prove_property_based_test_program_for_xx_dot_step() {
        let program = property_based_test_program_for_xx_dot_step();
        program.run().unwrap();
    }

    pub(crate) fn property_based_test_program_for_xb_dot_step() -> TestableProgram {
        let mut rng = ThreadRng::default();
        let n = rng.random_range(0..10);
        let push_xfe = |x: XFieldElement| {
            triton_asm! {
                push {x.coefficients[2]}
                push {x.coefficients[1]}
                push {x.coefficients[0]}
            }
        };
        let push_and_write_xfe = |x: XFieldElement| {
            triton_asm! {
                push {x.coefficients[2]}
                push {x.coefficients[1]}
                push {x.coefficients[0]}
                dup 3
                write_mem 3
                swap 1
                pop 1
            }
        };
        let push_and_write_bfe = |x: BFieldElement| {
            triton_asm! {
                push {x}
                dup 1
                write_mem 1
                swap 1
                pop 1
            }
        };
        let vector_one = (0..n).map(|_| rng.random::<XFieldElement>()).collect_vec();
        let vector_two = (0..n).map(|_| rng.random::<BFieldElement>()).collect_vec();
        let inner_product = vector_one
            .iter()
            .zip(vector_two.iter())
            .map(|(&a, &b)| a * b)
            .sum::<XFieldElement>();
        let push_and_write_vector_one = (0..n)
            .flat_map(|i| push_and_write_xfe(vector_one[i]))
            .collect_vec();
        let push_and_write_vector_two = (0..n)
            .flat_map(|i| push_and_write_bfe(vector_two[i]))
            .collect_vec();
        let push_inner_product = push_xfe(inner_product);
        let many_dotsteps = (0..n).map(|_| triton_instr!(xb_dot_step)).collect_vec();
        let code = triton_program! {
            push 0
            {&push_and_write_vector_one}
            dup 0
            {&push_and_write_vector_two}
            pop 1
            push 0
            swap 1
            {&many_dotsteps}
            pop 1
            pop 1
            push 0
            push 0
            {&push_inner_product}
            push 0
            push 0
            assert_vector
            halt
        };
        TestableProgram::new(code)
    }

    /// Sanity check
    #[test]
    fn run_dont_prove_property_based_test_program_for_xb_dot_step() {
        let program = property_based_test_program_for_xb_dot_step();
        program.run().unwrap();
    }

    #[proptest]
    fn negative_property_is_u32(
        #[strategy(arb())]
        #[filter(#st0.value() > u64::from(u32::MAX))]
        st0: BFieldElement,
    ) {
        let program = triton_program!(
            push {st0} call is_u32 assert halt
            is_u32:
                split pop 1 push 0 eq return
        );
        let program = TestableProgram::new(program);
        let_assert!(Err(err) = program.run());
        let_assert!(InstructionError::AssertionFailed(_) = err.source);
    }

    pub(crate) fn test_program_for_split() -> TestableProgram {
        TestableProgram::new(triton_program!(
            push -2 split push 4294967295 eq assert push 4294967294 eq assert
            push -1 split push 0 eq assert push 4294967295 eq assert
            push  0 split push 0 eq assert push 0 eq assert
            push  1 split push 1 eq assert push 0 eq assert
            push  2 split push 2 eq assert push 0 eq assert
            push 4294967297 split assert assert
            halt
        ))
    }

    pub(crate) fn test_program_for_xx_add() -> TestableProgram {
        TestableProgram::new(triton_program!(
            push 5 push 6 push 7 push 8 push 9 push 10 xx_add halt
        ))
    }

    pub(crate) fn test_program_for_xx_mul() -> TestableProgram {
        TestableProgram::new(triton_program!(
            push 5 push 6 push 7 push 8 push 9 push 10 xx_mul halt
        ))
    }

    pub(crate) fn test_program_for_x_invert() -> TestableProgram {
        TestableProgram::new(triton_program!(
            push 5 push 6 push 7 x_invert halt
        ))
    }

    pub(crate) fn test_program_for_xb_mul() -> TestableProgram {
        TestableProgram::new(triton_program!(
            push 5 push 6 push 7 push 8 xb_mul halt
        ))
    }

    pub(crate) fn test_program_for_read_io_write_io() -> TestableProgram {
        let program = triton_program!(
            read_io 1 assert read_io 2 dup 1 dup 1 add write_io 1 mul push 5 write_io 2 halt
        );
        TestableProgram::new(program).with_input([1, 3, 14].map(|b| bfe!(b)))
    }

    pub(crate) fn test_program_claim_in_ram_corresponds_to_currently_running_program()
    -> TestableProgram {
        let program = triton_program! {
            dup 15 dup 15 dup 15 dup 15 dup 15  // _ [own_digest]
            push 4 read_mem 5 pop 1             // _ [own_digest] [claim_digest]
            assert_vector                       // _ [own_digest]
            halt
        };

        let program_digest = program.hash();
        let enumerated_digest_elements = program_digest.values().into_iter().enumerate();
        let initial_ram = enumerated_digest_elements
            .map(|(address, d)| (bfe!(u64::try_from(address).unwrap()), d))
            .collect::<HashMap<_, _>>();
        let non_determinism = NonDeterminism::default().with_ram(initial_ram);

        TestableProgram::new(program).with_non_determinism(non_determinism)
    }

    #[proptest]
    fn xx_add(
        #[strategy(arb())] left_operand: XFieldElement,
        #[strategy(arb())] right_operand: XFieldElement,
    ) {
        let program = triton_program!(
            push {right_operand.coefficients[2]}
            push {right_operand.coefficients[1]}
            push {right_operand.coefficients[0]}
            push {left_operand.coefficients[2]}
            push {left_operand.coefficients[1]}
            push {left_operand.coefficients[0]}
            xx_add
            write_io 3
            halt
        );
        let actual_stdout = VM::run(program, [].into(), [].into())?;
        let expected_stdout = (left_operand + right_operand).coefficients.to_vec();
        prop_assert_eq!(expected_stdout, actual_stdout);
    }

    #[proptest]
    fn xx_mul(
        #[strategy(arb())] left_operand: XFieldElement,
        #[strategy(arb())] right_operand: XFieldElement,
    ) {
        let program = triton_program!(
            push {right_operand.coefficients[2]}
            push {right_operand.coefficients[1]}
            push {right_operand.coefficients[0]}
            push {left_operand.coefficients[2]}
            push {left_operand.coefficients[1]}
            push {left_operand.coefficients[0]}
            xx_mul
            write_io 3
            halt
        );
        let actual_stdout = VM::run(program, [].into(), [].into())?;
        let expected_stdout = (left_operand * right_operand).coefficients.to_vec();
        prop_assert_eq!(expected_stdout, actual_stdout);
    }

    #[proptest]
    fn xinv(
        #[strategy(arb())]
        #[filter(!#operand.is_zero())]
        operand: XFieldElement,
    ) {
        let program = triton_program!(
            push {operand.coefficients[2]}
            push {operand.coefficients[1]}
            push {operand.coefficients[0]}
            x_invert
            write_io 3
            halt
        );
        let actual_stdout = VM::run(program, [].into(), [].into())?;
        let expected_stdout = operand.inverse().coefficients.to_vec();
        prop_assert_eq!(expected_stdout, actual_stdout);
    }

    #[proptest]
    fn xb_mul(#[strategy(arb())] scalar: BFieldElement, #[strategy(arb())] operand: XFieldElement) {
        let program = triton_program!(
            push {operand.coefficients[2]}
            push {operand.coefficients[1]}
            push {operand.coefficients[0]}
            push {scalar}
            xb_mul
            write_io 3
            halt
        );
        let actual_stdout = VM::run(program, [].into(), [].into())?;
        let expected_stdout = (scalar * operand).coefficients.to_vec();
        prop_assert_eq!(expected_stdout, actual_stdout);
    }

    #[proptest]
    fn pseudo_sub(
        #[strategy(arb())] minuend: BFieldElement,
        #[strategy(arb())] subtrahend: BFieldElement,
    ) {
        let program = triton_program!(
            push {subtrahend} push {minuend} call sub write_io 1 halt
            sub:
                swap 1 push -1 mul add return
        );
        let actual_stdout = TestableProgram::new(program).run()?;
        let expected_stdout = vec![minuend - subtrahend];

        prop_assert_eq!(expected_stdout, actual_stdout);
    }

    // compile-time assertion
    const _OP_STACK_IS_BIG_ENOUGH: () = std::assert!(2 * Digest::LEN <= OpStackElement::COUNT);

    #[test]
    fn run_tvm_hello_world() {
        let program = triton_program!(
            push  10 // \n
            push  33 // !
            push 100 // d
            push 108 // l
            push 114 // r
            push 111 // o
            push  87 // W
            push  32 //
            push  44 // ,
            push 111 // o
            push 108 // l
            push 108 // l
            push 101 // e
            push  72 // H
            write_io 5 write_io 5 write_io 4
            halt
        );
        let mut vm_state = VMState::new(program, [].into(), [].into());
        let_assert!(Ok(()) = vm_state.run());
        assert!(BFieldElement::ZERO == vm_state.op_stack[0]);
    }

    #[test]
    fn run_tvm_merkle_step_right_sibling() {
        let program = test_program_for_merkle_step_right_sibling();
        let_assert!(Ok(_) = program.run());
    }

    #[test]
    fn run_tvm_merkle_step_left_sibling() {
        let program = test_program_for_merkle_step_left_sibling();
        let_assert!(Ok(_) = program.run());
    }

    #[test]
    fn run_tvm_merkle_step_mem_right_sibling() {
        let program = test_program_for_merkle_step_mem_right_sibling();
        let_assert!(Ok(_) = program.run());
    }

    #[test]
    fn run_tvm_merkle_step_mem_left_sibling() {
        let program = test_program_for_merkle_step_mem_left_sibling();
        let_assert!(Ok(_) = program.run());
    }

    #[test]
    fn run_tvm_halt_then_do_stuff() {
        let program = triton_program!(halt push 1 push 2 add invert write_io 5);
        let_assert!(Ok((aet, _)) = VM::trace_execution(program, [].into(), [].into()));

        let_assert!(Some(last_processor_row) = aet.processor_trace.rows().into_iter().next_back());
        let clk_count = last_processor_row[ProcessorMainColumn::CLK.main_index()];
        assert!(BFieldElement::ZERO == clk_count);

        let last_instruction = last_processor_row[ProcessorMainColumn::CI.main_index()];
        assert!(Instruction::Halt.opcode_b() == last_instruction);

        println!("{last_processor_row}");
    }

    #[test]
    fn run_tvm_basic_ram_read_write() {
        let program = triton_program!(
            push  8 push  5 write_mem 1 pop 1   // write  8 to address  5
            push 18 push 15 write_mem 1 pop 1   // write 18 to address 15
            push  5         read_mem  1 pop 2   // read from address  5
            push 15         read_mem  1 pop 2   // read from address 15
            push  7 push  5 write_mem 1 pop 1   // write  7 to address  5
            push 15         read_mem  1         // _ 18 14
            push  5         read_mem  1         // _ 18 14 7 4
            halt
        );

        let mut vm_state = VMState::new(program, [].into(), [].into());
        let_assert!(Ok(()) = vm_state.run());
        assert!(4 == vm_state.op_stack[0].value());
        assert!(7 == vm_state.op_stack[1].value());
        assert!(14 == vm_state.op_stack[2].value());
        assert!(18 == vm_state.op_stack[3].value());
    }

    #[test]
    fn run_tvm_edgy_ram_writes() {
        let program = triton_program!(
                        //       ┌ stack cannot shrink beyond this point
                        //       ↓
                        // _ 0 0 |
            push 0      // _ 0 0 | 0
            write_mem 1 // _ 0 1 |
            push 5      // _ 0 1 | 5
            swap 1      // _ 0 5 | 1
            push 3      // _ 0 5 | 1 3
            swap 1      // _ 0 5 | 3 1
            pop 1       // _ 0 5 | 3
            write_mem 1 // _ 0 4 |
            push -1 add // _ 0 3 |
            read_mem 1  // _ 0 5 | 2
            swap 2      // _ 2 5 | 0
            pop 1       // _ 2 5 |
            swap 1      // _ 5 2 |
            push 1      // _ 5 2 | 1
            add         // _ 5 3 |
            read_mem 1  // _ 5 5 | 2
            halt
        );

        let mut vm_state = VMState::new(program, [].into(), [].into());
        let_assert!(Ok(()) = vm_state.run());
        assert!(2_u64 == vm_state.op_stack[0].value());
        assert!(5_u64 == vm_state.op_stack[1].value());
        assert!(5_u64 == vm_state.op_stack[2].value());
    }

    #[proptest]
    fn triton_assembly_merkle_tree_authentication_path_verification(
        leaved_merkle_tree: LeavedMerkleTreeTestData,
    ) {
        let merkle_tree = &leaved_merkle_tree.merkle_tree;
        let auth_path = |&i| merkle_tree.authentication_structure(&[i]).unwrap();

        let revealed_indices = &leaved_merkle_tree.revealed_indices;
        let num_authentication_paths = revealed_indices.len();

        let auth_paths = revealed_indices.iter().flat_map(auth_path).collect_vec();
        let non_determinism = NonDeterminism::default().with_digests(auth_paths);

        let mut public_input = vec![(num_authentication_paths as u64).into()];
        public_input.extend(leaved_merkle_tree.root().reversed().values());
        for &leaf_index in revealed_indices {
            let node_index = (leaf_index + leaved_merkle_tree.num_leaves()) as u64;
            public_input.push(node_index.into());
            let revealed_leaf = leaved_merkle_tree.leaves_as_digests[leaf_index];
            public_input.extend(revealed_leaf.reversed().values());
        }

        let program = crate::example_programs::MERKLE_TREE_AUTHENTICATION_PATH_VERIFY.clone();
        assert!(let Ok(_) = VM::run(program, public_input.into(), non_determinism));
    }

    #[proptest]
    fn merkle_tree_updating_program_correctly_updates_a_merkle_tree(
        program_for_merkle_tree_update: ProgramForMerkleTreeUpdate,
    ) {
        prop_assert!(program_for_merkle_tree_update.is_integral());
    }

    #[proptest(cases = 10)]
    fn prove_verify_merkle_tree_update(
        program_for_merkle_tree_update: ProgramForMerkleTreeUpdate,
        #[strategy(1_usize..=4)] log_2_fri_expansion_factor: usize,
    ) {
        let stark = Stark::new(Stark::LOW_SECURITY_LEVEL, log_2_fri_expansion_factor);
        program_for_merkle_tree_update
            .assemble()
            .use_stark(stark)
            .prove_and_verify();
    }

    #[proptest]
    fn run_tvm_get_collinear_y(
        #[strategy(arb())] p0: (BFieldElement, BFieldElement),
        #[strategy(arb())]
        #[filter(#p0.0 != #p1.0)]
        p1: (BFieldElement, BFieldElement),
        #[strategy(arb())] p2_x: BFieldElement,
    ) {
        let p2_y = Polynomial::get_colinear_y(p0, p1, p2_x);

        let get_collinear_y_program = triton_program!(
            read_io 5                       // p0 p1 p2_x
            swap 3 push -1 mul dup 1 add    // dy = p0_y - p1_y
            dup 3 push -1 mul dup 5 add mul // dy·(p2_x - p0_x)
            dup 3 dup 3 push -1 mul add     // dx = p0_x - p1_x
            invert mul add                  // compute result
            swap 3 pop 3                    // leave a clean stack
            write_io 1 halt
        );

        let public_input = vec![p2_x, p1.1, p1.0, p0.1, p0.0];
        let_assert!(Ok(output) = VM::run(get_collinear_y_program, public_input.into(), [].into()));
        prop_assert_eq!(p2_y, output[0]);
    }

    #[test]
    fn run_tvm_countdown_from_10() {
        let countdown_program = triton_program!(
            push 10
            call loop

            loop:
                dup 0
                write_io 1
                push -1
                add
                dup 0
                skiz
                  recurse
                write_io 1
                halt
        );

        let_assert!(Ok(standard_out) = VM::run(countdown_program, [].into(), [].into()));
        let expected = (0..=10).map(BFieldElement::new).rev().collect_vec();
        assert!(expected == standard_out);
    }

    #[test]
    fn run_tvm_fibonacci() {
        for (input, expected_output) in [(0, 1), (7, 21), (11, 144)] {
            let program = TestableProgram::new(crate::example_programs::FIBONACCI_SEQUENCE.clone())
                .with_input(bfe_array![input]);
            let_assert!(Ok(output) = program.run());
            let_assert!(&[output] = &output[..]);
            assert!(expected_output == output.value(), "input was: {input}");
        }
    }

    #[test]
    fn run_tvm_swap() {
        let program = triton_program!(push 1 push 2 swap 1 assert write_io 1 halt);
        let_assert!(Ok(standard_out) = VM::run(program, [].into(), [].into()));
        assert!(bfe!(2) == standard_out[0]);
    }

    #[test]
    fn swap_st0_is_like_no_op() {
        let program = triton_program!(push 42 swap 0 write_io 1 halt);
        let_assert!(Ok(standard_out) = VM::run(program, [].into(), [].into()));
        assert!(bfe!(42) == standard_out[0]);
    }

    #[test]
    fn read_mem_uninitialized() {
        let program = triton_program!(read_mem 3 halt);
        let_assert!(Ok((aet, _)) = VM::trace_execution(program, [].into(), [].into()));
        assert!(2 == aet.processor_trace.nrows());
    }

    #[test]
    fn read_non_deterministically_initialized_ram_at_address_0() {
        let program = triton_program!(push 0 read_mem 1 pop 1 write_io 1 halt);

        let mut initial_ram = HashMap::new();
        initial_ram.insert(bfe!(0), bfe!(42));
        let non_determinism = NonDeterminism::default().with_ram(initial_ram);
        let program = TestableProgram::new(program).with_non_determinism(non_determinism);

        let_assert!(Ok(public_output) = program.clone().run());
        let_assert!(&[output] = &public_output[..]);
        assert!(42 == output.value());

        program.prove_and_verify();
    }

    #[proptest(cases = 10)]
    fn read_non_deterministically_initialized_ram_at_random_address(
        #[strategy(arb())] uninitialized_address: BFieldElement,
        #[strategy(arb())]
        #[filter(#uninitialized_address != #initialized_address)]
        initialized_address: BFieldElement,
        #[strategy(arb())] value: BFieldElement,
    ) {
        let program = triton_program!(
            push {uninitialized_address} read_mem 1 pop 1 write_io 1
            push {initialized_address} read_mem 1 pop 1 write_io 1
            halt
        );

        let mut initial_ram = HashMap::new();
        initial_ram.insert(initialized_address, value);
        let non_determinism = NonDeterminism::default().with_ram(initial_ram);
        let program = TestableProgram::new(program).with_non_determinism(non_determinism);

        let_assert!(Ok(public_output) = program.clone().run());
        let_assert!(&[uninit_value, init_value] = &public_output[..]);
        assert!(0 == uninit_value.value());
        assert!(value == init_value);

        program.prove_and_verify();
    }

    #[test]
    fn program_without_halt() {
        let program = triton_program!(nop);
        let_assert!(Err(err) = VM::trace_execution(program, [].into(), [].into()));
        let_assert!(InstructionError::InstructionPointerOverflow = err.source);
    }

    #[test]
    fn verify_sudoku() {
        let program = crate::example_programs::VERIFY_SUDOKU.clone();
        let sudoku = [
            8, 5, 9, /*  */ 7, 6, 1, /*  */ 4, 2, 3, //
            4, 2, 6, /*  */ 8, 5, 3, /*  */ 7, 9, 1, //
            7, 1, 3, /*  */ 9, 2, 4, /*  */ 8, 5, 6, //
            /************************************ */
            9, 6, 1, /*  */ 5, 3, 7, /*  */ 2, 8, 4, //
            2, 8, 7, /*  */ 4, 1, 9, /*  */ 6, 3, 5, //
            3, 4, 5, /*  */ 2, 8, 6, /*  */ 1, 7, 9, //
            /************************************ */
            5, 3, 4, /*  */ 6, 7, 8, /*  */ 9, 1, 2, //
            6, 7, 2, /*  */ 1, 9, 5, /*  */ 3, 4, 8, //
            1, 9, 8, /*  */ 3, 4, 2, /*  */ 5, 6, 7, //
        ];

        let std_in = PublicInput::from(sudoku.map(|b| bfe!(b)));
        let secret_in = NonDeterminism::default();
        assert!(let Ok(_) = VM::trace_execution(program.clone(), std_in, secret_in));

        // rows and columns adhere to Sudoku rules, boxes do not
        let bad_sudoku = [
            1, 2, 3, /*  */ 4, 5, 7, /*  */ 8, 9, 6, //
            4, 3, 1, /*  */ 5, 2, 9, /*  */ 6, 7, 8, //
            2, 7, 9, /*  */ 6, 1, 3, /*  */ 5, 8, 4, //
            /************************************ */
            7, 6, 5, /*  */ 3, 4, 8, /*  */ 9, 2, 1, //
            5, 1, 4, /*  */ 9, 8, 6, /*  */ 7, 3, 2, //
            6, 8, 2, /*  */ 7, 9, 4, /*  */ 1, 5, 3, //
            /************************************ */
            3, 5, 6, /*  */ 8, 7, 2, /*  */ 4, 1, 9, //
            9, 4, 8, /*  */ 1, 3, 5, /*  */ 2, 6, 7, //
            8, 9, 7, /*  */ 2, 6, 1, /*  */ 3, 4, 5, //
        ];
        let bad_std_in = PublicInput::from(bad_sudoku.map(|b| bfe!(b)));
        let secret_in = NonDeterminism::default();
        let_assert!(Err(err) = VM::trace_execution(program, bad_std_in, secret_in));
        let_assert!(InstructionError::AssertionFailed(_) = err.source);
    }

    fn instruction_does_not_change_vm_state_when_crashing_vm(
        program: TestableProgram,
        num_preparatory_steps: usize,
    ) {
        let mut vm_state = VMState::new(
            program.program,
            program.public_input,
            program.non_determinism,
        );
        for i in 0..num_preparatory_steps {
            assert!(let Ok(_) = vm_state.step(), "failed during step {i}");
        }
        let pre_crash_state = vm_state.clone();
        assert!(let Err(_) = vm_state.step());
        assert!(pre_crash_state == vm_state);
    }

    #[test]
    fn instruction_pop_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { push 0 pop 2 halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 1);
    }

    #[test]
    fn instruction_divine_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { divine 1 halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 0);
    }

    #[test]
    fn instruction_assert_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { push 0 assert halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 1);
    }

    #[test]
    fn instruction_merkle_step_does_not_change_vm_state_when_crashing_vm_invalid_node_index() {
        let non_u32 = u64::from(u32::MAX) + 1;
        let program = triton_program! { push {non_u32} swap 5 merkle_step halt };
        let nondeterminism = NonDeterminism::default().with_digests([Digest::default()]);
        let program = TestableProgram::new(program).with_non_determinism(nondeterminism);
        instruction_does_not_change_vm_state_when_crashing_vm(program, 2);
    }

    #[test]
    fn instruction_merkle_step_does_not_change_vm_state_when_crashing_vm_no_nd_digests() {
        let valid_u32 = u64::from(u32::MAX);
        let program = triton_program! { push {valid_u32} swap 5 merkle_step halt };
        let program = TestableProgram::new(program);
        instruction_does_not_change_vm_state_when_crashing_vm(program, 2);
    }

    #[test]
    fn instruction_merkle_step_mem_does_not_change_vm_state_when_crashing_vm_invalid_node_index() {
        let non_u32 = u64::from(u32::MAX) + 1;
        let program = triton_program! { push {non_u32} swap 5 merkle_step_mem halt };
        let program = TestableProgram::new(program);
        instruction_does_not_change_vm_state_when_crashing_vm(program, 2);
    }

    #[test]
    fn instruction_assert_vector_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { push 0 push 1 push 0 push 0 push 0 assert_vector halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 5);
    }

    #[test]
    fn instruction_sponge_absorb_does_not_change_vm_state_when_crashing_vm_sponge_uninit() {
        let ten_pushes = triton_asm![push 0; 10];
        let program = triton_program! { {&ten_pushes} sponge_absorb halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 10);
    }

    #[test]
    fn instruction_sponge_absorb_does_not_change_vm_state_when_crashing_vm_stack_too_small() {
        let program = triton_program! { sponge_init sponge_absorb halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 1);
    }

    #[test]
    fn instruction_sponge_absorb_mem_does_not_change_vm_state_when_crashing_vm_sponge_uninit() {
        let program = triton_program! { sponge_absorb_mem halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 0);
    }

    #[test]
    fn instruction_sponge_squeeze_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { sponge_squeeze halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 0);
    }

    #[test]
    fn instruction_invert_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { push 0 invert halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 1);
    }

    #[test]
    fn instruction_lt_does_not_change_vm_state_when_crashing_vm() {
        let non_u32 = u64::from(u32::MAX) + 1;
        let program = triton_program! { push {non_u32} lt halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 1);
    }

    #[test]
    fn instruction_and_does_not_change_vm_state_when_crashing_vm() {
        let non_u32 = u64::from(u32::MAX) + 1;
        let program = triton_program! { push {non_u32} and halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 1);
    }

    #[test]
    fn instruction_xor_does_not_change_vm_state_when_crashing_vm() {
        let non_u32 = u64::from(u32::MAX) + 1;
        let program = triton_program! { push {non_u32} xor halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 1);
    }

    #[test]
    fn instruction_log_2_floor_on_non_u32_operand_does_not_change_vm_state_when_crashing_vm() {
        let non_u32 = u64::from(u32::MAX) + 1;
        let program = triton_program! { push {non_u32} log_2_floor halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 1);
    }

    #[test]
    fn instruction_log_2_floor_on_operand_0_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { push 0 log_2_floor halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 1);
    }

    #[test]
    fn instruction_pow_does_not_change_vm_state_when_crashing_vm() {
        let non_u32 = u64::from(u32::MAX) + 1;
        let program = triton_program! { push {non_u32} push 0 pow halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 2);
    }

    #[test]
    fn instruction_div_mod_on_non_u32_operand_does_not_change_vm_state_when_crashing_vm() {
        let non_u32 = u64::from(u32::MAX) + 1;
        let program = triton_program! { push {non_u32} push 0 div_mod halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 2);
    }

    #[test]
    fn instruction_div_mod_on_denominator_0_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { push 0 push 1 div_mod halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 2);
    }

    #[test]
    fn instruction_pop_count_does_not_change_vm_state_when_crashing_vm() {
        let non_u32 = u64::from(u32::MAX) + 1;
        let program = triton_program! { push {non_u32} pop_count halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 1);
    }

    #[test]
    fn instruction_x_invert_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { x_invert halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 0);
    }

    #[test]
    fn instruction_read_io_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { read_io 1 halt };
        instruction_does_not_change_vm_state_when_crashing_vm(TestableProgram::new(program), 0);
    }

    #[proptest]
    fn serialize_deserialize_vm_state_to_and_from_json_is_identity(
        #[strategy(arb())] vm_state: VMState,
    ) {
        let serialized = serde_json::to_string(&vm_state).unwrap();
        let deserialized = serde_json::from_str(&serialized).unwrap();
        prop_assert_eq!(vm_state, deserialized);
    }

    #[proptest]
    fn xx_dot_step(
        #[strategy(0_usize..=25)] n: usize,
        #[strategy(vec(arb(), #n))] lhs: Vec<XFieldElement>,
        #[strategy(vec(arb(), #n))] rhs: Vec<XFieldElement>,
        #[strategy(arb())] lhs_address: BFieldElement,
        #[strategy(arb())] rhs_address: BFieldElement,
    ) {
        let mem_region_size = (n * EXTENSION_DEGREE) as u64;
        let mem_region = |addr: BFieldElement| addr.value()..addr.value() + mem_region_size;
        let right_in_left = mem_region(lhs_address).contains(&rhs_address.value());
        let left_in_right = mem_region(rhs_address).contains(&lhs_address.value());
        if right_in_left || left_in_right {
            let reason = "storing into overlapping regions would overwrite";
            return Err(TestCaseError::Reject(reason.into()));
        }

        let lhs_flat = lhs.iter().flat_map(|&xfe| xfe.coefficients).collect_vec();
        let rhs_flat = rhs.iter().flat_map(|&xfe| xfe.coefficients).collect_vec();
        let mut ram = HashMap::new();
        for (i, &l, &r) in izip!(0.., &lhs_flat, &rhs_flat) {
            ram.insert(lhs_address + bfe!(i), l);
            ram.insert(rhs_address + bfe!(i), r);
        }

        let public_input = PublicInput::default();
        let secret_input = NonDeterminism::default().with_ram(ram);
        let many_xx_dot_steps = triton_asm![xx_dot_step; n];
        let program = triton_program! {
            push 0 push 0 push 0

            push {lhs_address}
            push {rhs_address}

            {&many_xx_dot_steps}
            halt
        };

        let mut vmstate = VMState::new(program, public_input, secret_input);
        prop_assert!(vmstate.run().is_ok());

        prop_assert_eq!(
            vmstate.op_stack.pop().unwrap(),
            rhs_address + bfe!(rhs_flat.len() as u64)
        );
        prop_assert_eq!(
            vmstate.op_stack.pop().unwrap(),
            lhs_address + bfe!(lhs_flat.len() as u64)
        );

        let observed_dot_product = vmstate.op_stack.pop_extension_field_element().unwrap();
        let expected_dot_product = lhs
            .into_iter()
            .zip(rhs)
            .map(|(l, r)| l * r)
            .sum::<XFieldElement>();
        prop_assert_eq!(expected_dot_product, observed_dot_product);
    }

    #[proptest]
    fn xb_dot_step(
        #[strategy(0_usize..=25)] n: usize,
        #[strategy(vec(arb(), #n))] lhs: Vec<XFieldElement>,
        #[strategy(vec(arb(), #n))] rhs: Vec<BFieldElement>,
        #[strategy(arb())] lhs_address: BFieldElement,
        #[strategy(arb())] rhs_address: BFieldElement,
    ) {
        let mem_region_size_lhs = (n * EXTENSION_DEGREE) as u64;
        let mem_region_lhs = lhs_address.value()..lhs_address.value() + mem_region_size_lhs;
        let mem_region_rhs = rhs_address.value()..rhs_address.value() + n as u64;
        let right_in_left = mem_region_lhs.contains(&rhs_address.value());
        let left_in_right = mem_region_rhs.contains(&lhs_address.value());
        if right_in_left || left_in_right {
            let reason = "storing into overlapping regions would overwrite";
            return Err(TestCaseError::Reject(reason.into()));
        }

        let lhs_flat = lhs.iter().flat_map(|&xfe| xfe.coefficients).collect_vec();
        let mut ram = HashMap::new();
        for (i, &l) in (0..).zip(&lhs_flat) {
            ram.insert(lhs_address + bfe!(i), l);
        }
        for (i, &r) in (0..).zip(&rhs) {
            ram.insert(rhs_address + bfe!(i), r);
        }

        let public_input = PublicInput::default();
        let secret_input = NonDeterminism::default().with_ram(ram);
        let many_xb_dot_steps = triton_asm![xb_dot_step; n];
        let program = triton_program! {
            push 0 push 0 push 0

            push {lhs_address}
            push {rhs_address}

            {&many_xb_dot_steps}
            halt
        };

        let mut vmstate = VMState::new(program, public_input, secret_input);
        prop_assert!(vmstate.run().is_ok());

        prop_assert_eq!(
            vmstate.op_stack.pop().unwrap(),
            rhs_address + bfe!(rhs.len() as u64)
        );
        prop_assert_eq!(
            vmstate.op_stack.pop().unwrap(),
            lhs_address + bfe!(lhs_flat.len() as u64)
        );
        let observed_dot_product = vmstate.op_stack.pop_extension_field_element().unwrap();
        let expected_dot_product = lhs
            .into_iter()
            .zip(rhs)
            .map(|(l, r)| l * r)
            .sum::<XFieldElement>();
        prop_assert_eq!(expected_dot_product, observed_dot_product);
    }

    #[test]
    fn iterating_over_public_inputs_individual_tokens_is_easy() {
        let public_input = PublicInput::from(bfe_vec![1, 2, 3]);
        let actual = public_input.iter().join(", ");
        assert_eq!("1, 2, 3", actual);
    }
}
