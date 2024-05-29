use std::collections::HashMap;
use std::collections::VecDeque;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::ops::Range;

use arbitrary::Arbitrary;
use itertools::Itertools;
use ndarray::Array1;
use num_traits::One;
use num_traits::Zero;
use serde_derive::*;
use twenty_first::math::x_field_element::EXTENSION_DEGREE;
use twenty_first::prelude::*;
use twenty_first::util_types::algebraic_hasher::Domain;

use crate::error::InstructionError;
use crate::error::InstructionError::*;
use crate::instruction::AnInstruction::*;
use crate::instruction::Instruction;
use crate::op_stack::OpStackElement::*;
use crate::op_stack::*;
use crate::program::*;
use crate::table::hash_table::PermutationTrace;
use crate::table::op_stack_table::OpStackTableEntry;
use crate::table::processor_table;
use crate::table::ram_table::RamTableCall;
use crate::table::table_column::*;
use crate::table::u32_table::U32TableEntry;
use crate::vm::CoProcessorCall::*;

type Result<T> = std::result::Result<T, InstructionError>;

/// The number of helper variable registers
pub const NUM_HELPER_VARIABLE_REGISTERS: usize = 6;

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize, Arbitrary)]
pub struct VMState {
    /// The **program memory** stores the instructions (and their arguments) of the program
    /// currently being executed by Triton VM. It is read-only.
    pub program: Vec<Instruction>,

    /// A list of [`BFieldElement`]s the program can read from using instruction `read_io`.
    pub public_input: VecDeque<BFieldElement>,

    /// A list of [`BFieldElement`]s the program can write to using instruction `write_io`.
    pub public_output: Vec<BFieldElement>,

    /// A list of [`BFieldElement`]s the program can read from using instruction `divine`.
    pub secret_individual_tokens: VecDeque<BFieldElement>,

    /// A list of [`Digest`]s the program can use for instruction `merkle_step`.
    pub secret_digests: VecDeque<Digest>,

    /// The read-write **random-access memory** allows Triton VM to store arbitrary data.
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
    /// using instructions [`SpongeInit`], [`SpongeAbsorb`], [`SpongeAbsorbMem`],
    /// and [`SpongeSqueeze`]. Instruction [`SpongeInit`] resets the Sponge.
    ///
    /// Note that this is the _full_ state, including capacity. The capacity should never be
    /// exposed outside the VM.
    pub sponge: Option<Tip5>,

    /// Indicates whether the terminating instruction `halt` has been executed.
    pub halting: bool,
}

/// A call from the main processor to one of the co-processors, including the trace for that
/// co-processor or enough information to deduce the trace.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum CoProcessorCall {
    SpongeStateReset,

    /// Trace of the state registers for hash coprocessor table when executing
    /// instruction `hash` or one of the Sponge instructions `sponge_absorb`,
    /// `sponge_absorb_mem`, and `sponge_squeeze`.
    ///
    /// One row per round in the Tip5 permutation.
    Tip5Trace(Instruction, Box<PermutationTrace>),

    U32Call(U32TableEntry),

    OpStackCall(OpStackTableEntry),

    RamCall(RamTableCall),
}

impl VMState {
    /// Create initial `VMState` for a given `program`
    ///
    /// Since `program` is read-only across individual states, and multiple
    /// inner helper functions refer to it, a read-only reference is kept in
    /// the struct.
    pub fn new(
        program: &Program,
        public_input: PublicInput,
        non_determinism: NonDeterminism,
    ) -> Self {
        let program_digest = program.hash::<Tip5>();

        Self {
            program: program.instructions.clone(),
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
            Pop(_) | Divine(_) | Dup(_) | Swap(_) | ReadMem(_) | WriteMem(_) | ReadIo(_)
            | WriteIo(_) => {
                let arg = current_instruction.arg().unwrap().value();
                hvs[..4].copy_from_slice(&decompose_arg(arg));
            }
            Skiz => {
                let st0 = self.op_stack[ST0];
                hvs[0] = st0.inverse_or_zero();
                let next_opcode = self.next_instruction_or_argument().value();
                let decomposition = Self::decompose_opcode_for_instruction_skiz(next_opcode);
                let decomposition = decomposition.map(BFieldElement::new);
                hvs[1..6].copy_from_slice(&decomposition);
            }
            RecurseOrReturn => hvs[0] = (self.op_stack[ST6] - self.op_stack[ST5]).inverse_or_zero(),
            SpongeAbsorbMem => {
                hvs[0] = ram_read(self.op_stack[ST0] + bfe!(4));
                hvs[1] = ram_read(self.op_stack[ST0] + bfe!(5));
                hvs[2] = ram_read(self.op_stack[ST0] + bfe!(6));
                hvs[3] = ram_read(self.op_stack[ST0] + bfe!(7));
                hvs[4] = ram_read(self.op_stack[ST0] + bfe!(8));
                hvs[5] = ram_read(self.op_stack[ST0] + bfe!(9));
            }
            MerkleStep => {
                let divined_digest = self.secret_digests.front().copied().unwrap_or_default();
                let node_index = self.op_stack[ST5].value();
                hvs[..5].copy_from_slice(&divined_digest.values());
                hvs[5] = bfe!(node_index % 2);
            }
            Split => {
                let top_of_stack = self.op_stack[ST0].value();
                let lo = bfe!(top_of_stack & 0xffff_ffff);
                let hi = bfe!(top_of_stack >> 32);
                if !lo.is_zero() {
                    let max_val_of_hi = bfe!(2_u64.pow(32) - 1);
                    hvs[0] = (hi - max_val_of_hi).inverse_or_zero();
                }
            }
            Eq => hvs[0] = (self.op_stack[ST1] - self.op_stack[ST0]).inverse_or_zero(),
            XxDotStep => {
                hvs[0] = ram_read(self.op_stack[ST0]);
                hvs[1] = ram_read(self.op_stack[ST0] + bfe!(1));
                hvs[2] = ram_read(self.op_stack[ST0] + bfe!(2));
                hvs[3] = ram_read(self.op_stack[ST1]);
                hvs[4] = ram_read(self.op_stack[ST1] + bfe!(1));
                hvs[5] = ram_read(self.op_stack[ST1] + bfe!(2));
            }
            XbDotStep => {
                hvs[0] = ram_read(self.op_stack[ST0]);
                hvs[1] = ram_read(self.op_stack[ST1]);
                hvs[2] = ram_read(self.op_stack[ST1] + bfe!(1));
                hvs[3] = ram_read(self.op_stack[ST1] + bfe!(2));
            }
            _ => (),
        }

        hvs
    }

    fn decompose_opcode_for_instruction_skiz(opcode: u64) -> [u64; 5] {
        let mut decomposition = [0; 5];
        decomposition[0] = opcode % 2;
        decomposition[1] = (opcode >> 1) % 4;
        decomposition[2] = (opcode >> 3) % 4;
        decomposition[3] = (opcode >> 5) % 4;
        decomposition[4] = opcode >> 7;
        decomposition
    }

    /// Perform the state transition as a mutable operation on `self`.
    pub fn step(&mut self) -> Result<Vec<CoProcessorCall>> {
        if self.halting {
            return Err(MachineHalted);
        }

        let current_instruction = self.current_instruction()?;
        let op_stack_delta = current_instruction.op_stack_size_influence();
        if self.op_stack.would_be_too_shallow(op_stack_delta) {
            return Err(OpStackTooShallow);
        }

        self.start_recording_op_stack_calls();
        let mut co_processor_calls = match current_instruction {
            Pop(n) => self.pop(n)?,
            Push(field_element) => self.push(field_element),
            Divine(n) => self.divine(n)?,
            Dup(stack_element) => self.dup(stack_element),
            Swap(stack_element) => self.swap(stack_element)?,
            Halt => self.halt(),
            Nop => self.nop(),
            Skiz => self.skiz()?,
            Call(address) => self.call(address),
            Return => self.return_from_call()?,
            Recurse => self.recurse()?,
            RecurseOrReturn => self.recurse_or_return()?,
            Assert => self.assert()?,
            ReadMem(n) => self.read_mem(n)?,
            WriteMem(n) => self.write_mem(n)?,
            Hash => self.hash()?,
            SpongeInit => self.sponge_init(),
            SpongeAbsorb => self.sponge_absorb()?,
            SpongeAbsorbMem => self.sponge_absorb_mem()?,
            SpongeSqueeze => self.sponge_squeeze()?,
            AssertVector => self.assert_vector()?,
            Add => self.add()?,
            Mul => self.mul()?,
            Invert => self.invert()?,
            Eq => self.eq()?,
            Split => self.split()?,
            Lt => self.lt()?,
            And => self.and()?,
            Xor => self.xor()?,
            Log2Floor => self.log_2_floor()?,
            Pow => self.pow()?,
            DivMod => self.div_mod()?,
            PopCount => self.pop_count()?,
            XxAdd => self.xx_add()?,
            XxMul => self.xx_mul()?,
            XInvert => self.x_invert()?,
            XbMul => self.xb_mul()?,
            WriteIo(n) => self.write_io(n)?,
            ReadIo(n) => self.read_io(n)?,
            MerkleStep => self.merkle_step()?,
            XxDotStep => self.xx_dot_step()?,
            XbDotStep => self.xb_dot_step()?,
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
            .map(OpStackCall)
            .collect()
    }

    fn start_recording_ram_calls(&mut self) {
        self.ram_calls.clear();
    }

    fn stop_recording_ram_calls(&mut self) -> Vec<CoProcessorCall> {
        self.ram_calls.drain(..).map(RamCall).collect()
    }

    fn pop(&mut self, n: NumberOfWords) -> Result<Vec<CoProcessorCall>> {
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

    fn divine(&mut self, n: NumberOfWords) -> Result<Vec<CoProcessorCall>> {
        let input_len = self.secret_individual_tokens.len();
        if input_len < n.num_words() {
            return Err(EmptySecretInput(input_len));
        }
        for _ in 0..n.num_words() {
            let element = self.secret_individual_tokens.pop_front().unwrap();
            self.op_stack.push(element);
        }

        self.instruction_pointer += 2;
        Ok(vec![])
    }

    fn dup(&mut self, stack_register: OpStackElement) -> Vec<CoProcessorCall> {
        let element = self.op_stack[stack_register];
        self.op_stack.push(element);

        self.instruction_pointer += 2;
        vec![]
    }

    fn swap(&mut self, stack_register: OpStackElement) -> Result<Vec<CoProcessorCall>> {
        if stack_register == ST0 {
            return Err(SwapST0);
        }
        (self.op_stack[ST0], self.op_stack[stack_register]) =
            (self.op_stack[stack_register], self.op_stack[ST0]);
        self.instruction_pointer += 2;
        Ok(vec![])
    }

    fn nop(&mut self) -> Vec<CoProcessorCall> {
        self.instruction_pointer += 1;
        vec![]
    }

    fn skiz(&mut self) -> Result<Vec<CoProcessorCall>> {
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

    fn return_from_call(&mut self) -> Result<Vec<CoProcessorCall>> {
        let (call_origin, _) = self.jump_stack_pop()?;
        self.instruction_pointer = call_origin.value().try_into().unwrap();
        Ok(vec![])
    }

    fn recurse(&mut self) -> Result<Vec<CoProcessorCall>> {
        let (_, call_destination) = self.jump_stack_peek()?;
        self.instruction_pointer = call_destination.value().try_into().unwrap();
        Ok(vec![])
    }

    fn recurse_or_return(&mut self) -> Result<Vec<CoProcessorCall>> {
        if self.jump_stack.is_empty() {
            return Err(JumpStackIsEmpty);
        }

        let new_ip = if self.op_stack[ST5] == self.op_stack[ST6] {
            let (call_origin, _) = self.jump_stack_pop()?;
            call_origin
        } else {
            let (_, call_destination) = self.jump_stack_peek()?;
            call_destination
        };

        self.instruction_pointer = new_ip.value().try_into().unwrap();

        Ok(vec![])
    }

    fn assert(&mut self) -> Result<Vec<CoProcessorCall>> {
        if !self.op_stack[ST0].is_one() {
            return Err(AssertionFailed);
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

    fn read_mem(&mut self, n: NumberOfWords) -> Result<Vec<CoProcessorCall>> {
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

    fn write_mem(&mut self, n: NumberOfWords) -> Result<Vec<CoProcessorCall>> {
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
            .unwrap_or(b_field_element::BFIELD_ZERO);

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

    fn hash(&mut self) -> Result<Vec<CoProcessorCall>> {
        let to_hash = self.op_stack.pop_multiple::<{ tip5::RATE }>()?;

        let mut hash_input = Tip5::new(Domain::FixedLength);
        hash_input.state[..tip5::RATE].copy_from_slice(&to_hash);
        let tip5_trace = hash_input.trace();
        let hash_output = &tip5_trace[tip5_trace.len() - 1][0..tip5::DIGEST_LENGTH];

        for i in (0..tip5::DIGEST_LENGTH).rev() {
            self.op_stack.push(hash_output[i]);
        }

        let co_processor_calls = vec![Tip5Trace(Hash, Box::new(tip5_trace))];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn sponge_init(&mut self) -> Vec<CoProcessorCall> {
        self.sponge = Some(Tip5::init());
        self.instruction_pointer += 1;
        vec![SpongeStateReset]
    }

    fn sponge_absorb(&mut self) -> Result<Vec<CoProcessorCall>> {
        let Some(ref mut sponge) = self.sponge else {
            return Err(SpongeNotInitialized);
        };
        let to_absorb = self.op_stack.pop_multiple::<{ tip5::RATE }>()?;
        sponge.state[..tip5::RATE].copy_from_slice(&to_absorb);
        let tip5_trace = sponge.trace();

        let co_processor_calls = vec![Tip5Trace(SpongeAbsorb, Box::new(tip5_trace))];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn sponge_absorb_mem(&mut self) -> Result<Vec<CoProcessorCall>> {
        let Some(mut sponge) = self.sponge.take() else {
            return Err(SpongeNotInitialized);
        };

        self.start_recording_ram_calls();
        let mut mem_pointer = self.op_stack.pop()?;
        for i in 0..tip5::RATE {
            let element = self.ram_read(mem_pointer);
            mem_pointer.increment();
            sponge.state[i] = element;

            // there are not enough helper variables – overwrite part of the stack :(
            if i < tip5::RATE - NUM_HELPER_VARIABLE_REGISTERS {
                self.op_stack[i] = element;
            }
        }
        self.op_stack.push(mem_pointer);

        let tip5_trace = sponge.trace();
        self.sponge = Some(sponge);

        let mut co_processor_calls = self.stop_recording_ram_calls();
        co_processor_calls.push(Tip5Trace(SpongeAbsorb, Box::new(tip5_trace)));

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn sponge_squeeze(&mut self) -> Result<Vec<CoProcessorCall>> {
        let Some(ref mut sponge) = self.sponge else {
            return Err(SpongeNotInitialized);
        };
        for i in (0..tip5::RATE).rev() {
            self.op_stack.push(sponge.state[i]);
        }
        let tip5_trace = sponge.trace();

        let co_processor_calls = vec![Tip5Trace(SpongeSqueeze, Box::new(tip5_trace))];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn assert_vector(&mut self) -> Result<Vec<CoProcessorCall>> {
        for i in 0..tip5::DIGEST_LENGTH {
            if self.op_stack[i] != self.op_stack[i + tip5::DIGEST_LENGTH] {
                return Err(VectorAssertionFailed(i));
            }
        }
        let _: [_; tip5::DIGEST_LENGTH] = self.op_stack.pop_multiple()?;
        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn add(&mut self) -> Result<Vec<CoProcessorCall>> {
        let lhs = self.op_stack.pop()?;
        let rhs = self.op_stack.pop()?;
        self.op_stack.push(lhs + rhs);

        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn mul(&mut self) -> Result<Vec<CoProcessorCall>> {
        let lhs = self.op_stack.pop()?;
        let rhs = self.op_stack.pop()?;
        self.op_stack.push(lhs * rhs);

        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn invert(&mut self) -> Result<Vec<CoProcessorCall>> {
        let top_of_stack = self.op_stack[ST0];
        if top_of_stack.is_zero() {
            return Err(InverseOfZero);
        }
        let _ = self.op_stack.pop()?;
        self.op_stack.push(top_of_stack.inverse());
        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn eq(&mut self) -> Result<Vec<CoProcessorCall>> {
        let lhs = self.op_stack.pop()?;
        let rhs = self.op_stack.pop()?;
        let eq: u32 = (lhs == rhs).into();
        self.op_stack.push(eq.into());

        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn split(&mut self) -> Result<Vec<CoProcessorCall>> {
        let top_of_stack = self.op_stack.pop()?;
        let lo = bfe!(top_of_stack.value() & 0xffff_ffff);
        let hi = bfe!(top_of_stack.value() >> 32);
        self.op_stack.push(hi);
        self.op_stack.push(lo);

        let u32_table_entry = U32TableEntry::new(Split, lo, hi);
        let co_processor_calls = vec![U32Call(u32_table_entry)];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn lt(&mut self) -> Result<Vec<CoProcessorCall>> {
        self.op_stack.is_u32(ST0)?;
        self.op_stack.is_u32(ST1)?;
        let lhs = self.op_stack.pop_u32()?;
        let rhs = self.op_stack.pop_u32()?;
        let lt: u32 = (lhs < rhs).into();
        self.op_stack.push(lt.into());

        let u32_table_entry = U32TableEntry::new(Lt, lhs, rhs);
        let co_processor_calls = vec![U32Call(u32_table_entry)];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn and(&mut self) -> Result<Vec<CoProcessorCall>> {
        self.op_stack.is_u32(ST0)?;
        self.op_stack.is_u32(ST1)?;
        let lhs = self.op_stack.pop_u32()?;
        let rhs = self.op_stack.pop_u32()?;
        let and = lhs & rhs;
        self.op_stack.push(and.into());

        let u32_table_entry = U32TableEntry::new(And, lhs, rhs);
        let co_processor_calls = vec![U32Call(u32_table_entry)];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn xor(&mut self) -> Result<Vec<CoProcessorCall>> {
        self.op_stack.is_u32(ST0)?;
        self.op_stack.is_u32(ST1)?;
        let lhs = self.op_stack.pop_u32()?;
        let rhs = self.op_stack.pop_u32()?;
        let xor = lhs ^ rhs;
        self.op_stack.push(xor.into());

        // Triton VM uses the following equality to compute the results of both the `and`
        // and `xor` instruction using the u32 coprocessor's `and` capability:
        // a ^ b = a + b - 2 · (a & b)
        let u32_table_entry = U32TableEntry::new(And, lhs, rhs);
        let co_processor_calls = vec![U32Call(u32_table_entry)];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn log_2_floor(&mut self) -> Result<Vec<CoProcessorCall>> {
        self.op_stack.is_u32(ST0)?;
        let top_of_stack = self.op_stack[ST0];
        if top_of_stack.is_zero() {
            return Err(LogarithmOfZero);
        }
        let top_of_stack = self.op_stack.pop_u32()?;
        let log_2_floor = top_of_stack.ilog2();
        self.op_stack.push(log_2_floor.into());

        let u32_table_entry = U32TableEntry::new(Log2Floor, top_of_stack, 0);
        let co_processor_calls = vec![U32Call(u32_table_entry)];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn pow(&mut self) -> Result<Vec<CoProcessorCall>> {
        self.op_stack.is_u32(ST1)?;
        let base = self.op_stack.pop()?;
        let exponent = self.op_stack.pop_u32()?;
        let base_pow_exponent = base.mod_pow(exponent.into());
        self.op_stack.push(base_pow_exponent);

        let u32_table_entry = U32TableEntry::new(Pow, base, exponent);
        let co_processor_calls = vec![U32Call(u32_table_entry)];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn div_mod(&mut self) -> Result<Vec<CoProcessorCall>> {
        self.op_stack.is_u32(ST0)?;
        self.op_stack.is_u32(ST1)?;
        let denominator = self.op_stack[ST1];
        if denominator.is_zero() {
            return Err(DivisionByZero);
        }

        let numerator = self.op_stack.pop_u32()?;
        let denominator = self.op_stack.pop_u32()?;
        let quotient = numerator / denominator;
        let remainder = numerator % denominator;

        self.op_stack.push(quotient.into());
        self.op_stack.push(remainder.into());

        let remainder_is_less_than_denominator = U32TableEntry::new(Lt, remainder, denominator);
        let numerator_and_quotient_range_check = U32TableEntry::new(Split, numerator, quotient);
        let co_processor_calls = vec![
            U32Call(remainder_is_less_than_denominator),
            U32Call(numerator_and_quotient_range_check),
        ];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn pop_count(&mut self) -> Result<Vec<CoProcessorCall>> {
        self.op_stack.is_u32(ST0)?;
        let top_of_stack = self.op_stack.pop_u32()?;
        let pop_count = top_of_stack.count_ones();
        self.op_stack.push(pop_count.into());

        let u32_table_entry = U32TableEntry::new(PopCount, top_of_stack, 0);
        let co_processor_calls = vec![U32Call(u32_table_entry)];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn xx_add(&mut self) -> Result<Vec<CoProcessorCall>> {
        let lhs = self.op_stack.pop_extension_field_element()?;
        let rhs = self.op_stack.pop_extension_field_element()?;
        self.op_stack.push_extension_field_element(lhs + rhs);
        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn xx_mul(&mut self) -> Result<Vec<CoProcessorCall>> {
        let lhs = self.op_stack.pop_extension_field_element()?;
        let rhs = self.op_stack.pop_extension_field_element()?;
        self.op_stack.push_extension_field_element(lhs * rhs);
        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn x_invert(&mut self) -> Result<Vec<CoProcessorCall>> {
        let top_of_stack = self.op_stack.peek_at_top_extension_field_element();
        if top_of_stack.is_zero() {
            return Err(InverseOfZero);
        }
        let inverse = top_of_stack.inverse();
        let _ = self.op_stack.pop_extension_field_element()?;
        self.op_stack.push_extension_field_element(inverse);
        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn xb_mul(&mut self) -> Result<Vec<CoProcessorCall>> {
        let lhs = self.op_stack.pop()?;
        let rhs = self.op_stack.pop_extension_field_element()?;
        self.op_stack.push_extension_field_element(lhs.lift() * rhs);

        self.instruction_pointer += 1;
        Ok(vec![])
    }

    fn write_io(&mut self, n: NumberOfWords) -> Result<Vec<CoProcessorCall>> {
        for _ in 0..n.num_words() {
            let top_of_stack = self.op_stack.pop()?;
            self.public_output.push(top_of_stack);
        }

        self.instruction_pointer += 2;
        Ok(vec![])
    }

    fn read_io(&mut self, n: NumberOfWords) -> Result<Vec<CoProcessorCall>> {
        let input_len = self.public_input.len();
        if input_len < n.num_words() {
            return Err(EmptyPublicInput(input_len));
        }
        for _ in 0..n.num_words() {
            let read_element = self.public_input.pop_front().unwrap();
            self.op_stack.push(read_element);
        }

        self.instruction_pointer += 2;
        Ok(vec![])
    }

    fn merkle_step(&mut self) -> Result<Vec<CoProcessorCall>> {
        if self.secret_digests.is_empty() {
            return Err(EmptySecretDigestInput);
        }
        self.op_stack.is_u32(ST5)?;

        let accumulator_digest = self.op_stack.pop_multiple::<{ tip5::DIGEST_LENGTH }>()?;
        let node_index = self.op_stack.pop_u32()?;

        let parent_node_index = node_index / 2;
        self.op_stack.push(parent_node_index.into());

        let stack_contains_left_node = node_index % 2 == 0;
        let sibling_digest = self.pop_secret_digest()?;
        let mut hash_input = Tip5::new(Domain::FixedLength);
        if stack_contains_left_node {
            hash_input.state[..tip5::DIGEST_LENGTH].copy_from_slice(&accumulator_digest);
            hash_input.state[tip5::DIGEST_LENGTH..2 * tip5::DIGEST_LENGTH]
                .copy_from_slice(&sibling_digest);
        } else {
            hash_input.state[..tip5::DIGEST_LENGTH].copy_from_slice(&sibling_digest);
            hash_input.state[tip5::DIGEST_LENGTH..2 * tip5::DIGEST_LENGTH]
                .copy_from_slice(&accumulator_digest);
        }
        let tip5_trace = hash_input.trace();
        let accumulator_digest = &tip5_trace[tip5_trace.len() - 1][0..tip5::DIGEST_LENGTH];

        for i in (0..tip5::DIGEST_LENGTH).rev() {
            self.op_stack.push(accumulator_digest[i]);
        }

        let co_processor_calls = vec![Tip5Trace(Hash, Box::new(tip5_trace))];

        self.instruction_pointer += 1;
        Ok(co_processor_calls)
    }

    fn xx_dot_step(&mut self) -> Result<Vec<CoProcessorCall>> {
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

    fn xb_dot_step(&mut self) -> Result<Vec<CoProcessorCall>> {
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
        use crate::instruction::InstructionBit;
        use ProcessorBaseTableColumn::*;
        let mut processor_row = Array1::zeros(processor_table::BASE_WIDTH);

        let current_instruction = self.current_instruction().unwrap_or(Nop);
        let helper_variables = self.derive_helper_variables();

        processor_row[CLK.base_table_index()] = u64::from(self.cycle_count).into();
        processor_row[IP.base_table_index()] = (self.instruction_pointer as u32).into();
        processor_row[CI.base_table_index()] = current_instruction.opcode_b();
        processor_row[NIA.base_table_index()] = self.next_instruction_or_argument();
        processor_row[IB0.base_table_index()] = current_instruction.ib(InstructionBit::IB0);
        processor_row[IB1.base_table_index()] = current_instruction.ib(InstructionBit::IB1);
        processor_row[IB2.base_table_index()] = current_instruction.ib(InstructionBit::IB2);
        processor_row[IB3.base_table_index()] = current_instruction.ib(InstructionBit::IB3);
        processor_row[IB4.base_table_index()] = current_instruction.ib(InstructionBit::IB4);
        processor_row[IB5.base_table_index()] = current_instruction.ib(InstructionBit::IB5);
        processor_row[IB6.base_table_index()] = current_instruction.ib(InstructionBit::IB6);
        processor_row[JSP.base_table_index()] = self.jump_stack_pointer();
        processor_row[JSO.base_table_index()] = self.jump_stack_origin();
        processor_row[JSD.base_table_index()] = self.jump_stack_destination();
        processor_row[ST0.base_table_index()] = self.op_stack[OpStackElement::ST0];
        processor_row[ST1.base_table_index()] = self.op_stack[OpStackElement::ST1];
        processor_row[ST2.base_table_index()] = self.op_stack[OpStackElement::ST2];
        processor_row[ST3.base_table_index()] = self.op_stack[OpStackElement::ST3];
        processor_row[ST4.base_table_index()] = self.op_stack[OpStackElement::ST4];
        processor_row[ST5.base_table_index()] = self.op_stack[OpStackElement::ST5];
        processor_row[ST6.base_table_index()] = self.op_stack[OpStackElement::ST6];
        processor_row[ST7.base_table_index()] = self.op_stack[OpStackElement::ST7];
        processor_row[ST8.base_table_index()] = self.op_stack[OpStackElement::ST8];
        processor_row[ST9.base_table_index()] = self.op_stack[OpStackElement::ST9];
        processor_row[ST10.base_table_index()] = self.op_stack[OpStackElement::ST10];
        processor_row[ST11.base_table_index()] = self.op_stack[OpStackElement::ST11];
        processor_row[ST12.base_table_index()] = self.op_stack[OpStackElement::ST12];
        processor_row[ST13.base_table_index()] = self.op_stack[OpStackElement::ST13];
        processor_row[ST14.base_table_index()] = self.op_stack[OpStackElement::ST14];
        processor_row[ST15.base_table_index()] = self.op_stack[OpStackElement::ST15];
        processor_row[OpStackPointer.base_table_index()] = self.op_stack.pointer();
        processor_row[HV0.base_table_index()] = helper_variables[0];
        processor_row[HV1.base_table_index()] = helper_variables[1];
        processor_row[HV2.base_table_index()] = helper_variables[2];
        processor_row[HV3.base_table_index()] = helper_variables[3];
        processor_row[HV4.base_table_index()] = helper_variables[4];
        processor_row[HV5.base_table_index()] = helper_variables[5];

        processor_row
    }

    /// The “next instruction or argument” (NIA) is
    /// - the argument of the current instruction if it has one, or
    /// - the opcode of the next instruction otherwise.
    ///
    /// If the current instruction has no argument and there is no next instruction, the NIA is 1
    /// to account for the hash-input padding separator of the program.
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

    pub fn current_instruction(&self) -> Result<Instruction> {
        let maybe_current_instruction = self.program.get(self.instruction_pointer).copied();
        maybe_current_instruction.ok_or(InstructionPointerOverflow)
    }

    /// Return the next instruction on the tape, skipping arguments.
    ///
    /// Note that this is not necessarily the next instruction to execute, since the current
    /// instruction could be a jump, but it is either program[ip + 1] or program[ip + 2],
    /// depending on whether the current instruction takes an argument.
    pub fn next_instruction(&self) -> Result<Instruction> {
        let current_instruction = self.current_instruction()?;
        let next_instruction_pointer = self.instruction_pointer + current_instruction.size();
        let maybe_next_instruction = self.program.get(next_instruction_pointer).copied();
        maybe_next_instruction.ok_or(InstructionPointerOverflow)
    }

    fn jump_stack_pop(&mut self) -> Result<(BFieldElement, BFieldElement)> {
        self.jump_stack.pop().ok_or(JumpStackIsEmpty)
    }

    fn jump_stack_peek(&mut self) -> Result<(BFieldElement, BFieldElement)> {
        self.jump_stack.last().copied().ok_or(JumpStackIsEmpty)
    }

    fn pop_secret_digest(&mut self) -> Result<[BFieldElement; tip5::DIGEST_LENGTH]> {
        let digest = self
            .secret_digests
            .pop_front()
            .ok_or(EmptySecretDigestInput)?;
        Ok(digest.values())
    }

    /// Run Triton VM on this state to completion, or until an error occurs.
    pub fn run(&mut self) -> Result<()> {
        while !self.halting {
            self.step()?;
        }
        Ok(())
    }
}

impl Display for VMState {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        use ProcessorBaseTableColumn as ProcCol;

        let Ok(instruction) = self.current_instruction() else {
            return write!(f, "END-OF-FILE");
        };

        let total_width = 103;
        let tab_width = 54;
        let clk_width = 17;
        let register_width = 20;
        let buffer_width = total_width - tab_width - clk_width - 7;

        let print_row = |f: &mut Formatter, s: String| writeln!(f, "│ {s: <total_width$} │");
        let print_blank_row = |f: &mut Formatter| print_row(f, String::new());

        let row = self.to_processor_row();

        let register = |reg: ProcessorBaseTableColumn| {
            let reg_string = format!("{}", row[reg.base_table_index()]);
            format!("{reg_string:>register_width$}")
        };
        let multi_register = |regs: [_; 4]| regs.map(register).join(" | ");

        writeln!(f)?;
        writeln!(f, " ╭─{:─<tab_width$}─╮", "")?;
        writeln!(f, " │ {: <tab_width$} │", format!("{instruction}"))?;
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
        let clk = row[ProcCol::CLK.base_table_index()].to_string();
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
        .map(|reg| row[reg.base_table_index()])
        .map(|bfe| format!("{bfe:>2}"))
        .join(" | ");
        print_row(f, format!("ib6-0:    [ {ib_registers} ]",))?;

        let Some(ref sponge) = self.sponge else {
            return writeln!(f, "╰─{:─<total_width$}─╯", "");
        };

        let sponge_state_register = |i: usize| sponge.state[i].value();
        let sponge_state_slice = |idxs: Range<usize>| {
            idxs.map(sponge_state_register)
                .map(|ss| format!("{ss:>register_width$}"))
                .join(" | ")
        };

        let sponge_state_00_03 = sponge_state_slice(0..4);
        let sponge_state_04_07 = sponge_state_slice(4..8);
        let sponge_state_08_11 = sponge_state_slice(8..12);
        let sponge_state_12_15 = sponge_state_slice(12..16);

        writeln!(f, "├─{:─<total_width$}─┤", "")?;
        print_row(f, format!("sp0-3:    [ {sponge_state_00_03} ]"))?;
        print_row(f, format!("sp4-7:    [ {sponge_state_04_07} ]"))?;
        print_row(f, format!("sp8-11:   [ {sponge_state_08_11} ]"))?;
        print_row(f, format!("sp12-15:  [ {sponge_state_12_15} ]"))?;
        writeln!(f, "╰─{:─<total_width$}─╯", "")
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::ops::BitAnd;
    use std::ops::BitXor;

    use assert2::assert;
    use assert2::let_assert;
    use itertools::izip;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use rand::rngs::ThreadRng;
    use rand::Rng;
    use rand::RngCore;
    use strum::EnumCount;
    use strum::EnumIter;
    use test_strategy::proptest;
    use twenty_first::math::other::random_elements;

    use crate::example_programs::*;
    use crate::shared_tests::prove_with_low_security_level;
    use crate::shared_tests::LeavedMerkleTreeTestData;
    use crate::shared_tests::ProgramAndInput;
    use crate::triton_asm;
    use crate::triton_instr;
    use crate::triton_program;
    use crate::LabelledInstruction;

    use super::*;

    #[test]
    fn initialise_table() {
        let program = GREATEST_COMMON_DIVISOR.clone();
        let stdin = PublicInput::from([42, 56].map(|b| bfe!(b)));
        let secret_in = NonDeterminism::default();
        program.trace_execution(stdin, secret_in).unwrap();
    }

    #[test]
    fn run_tvm_gcd() {
        let program = GREATEST_COMMON_DIVISOR.clone();
        let stdin = PublicInput::from([42, 56].map(|b| bfe!(b)));
        let secret_in = NonDeterminism::default();
        let_assert!(Ok(stdout) = program.run(stdin, secret_in));

        let output = stdout.iter().map(|o| format!("{o}")).join(", ");
        println!("VM output: [{output}]");

        assert!(bfe!(14) == stdout[0]);
    }

    #[test]
    fn crash_triton_vm_and_print_vm_error() {
        let crashing_program = triton_program!(push 2 assert halt);
        let_assert!(Err(err) = crashing_program.run([].into(), [].into()));
        println!("{err}");
    }

    pub(crate) fn test_program_hash_nop_nop_lt() -> ProgramAndInput {
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
        ProgramAndInput::new(program)
    }

    pub(crate) fn test_program_for_halt() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(halt))
    }

    pub(crate) fn test_program_for_push_pop_dup_swap_nop() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(
            push 1 push 2 pop 1 assert
            push 1 dup  0 assert assert
            push 1 push 2 swap 1 assert pop 1
            nop nop nop halt
        ))
    }

    pub(crate) fn test_program_for_divine() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(divine 1 assert halt)).with_non_determinism([bfe!(1)])
    }

    pub(crate) fn test_program_for_skiz() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(push 1 skiz push 0 skiz assert push 1 skiz halt))
    }

    pub(crate) fn test_program_for_call_recurse_return() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(
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

    pub(crate) fn test_program_for_recurse_or_return() -> ProgramAndInput {
        ProgramAndInput::new(triton_program! {
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
        pub fn assemble(self) -> ProgramAndInput {
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
            ProgramAndInput::new(program).with_input(input)
        }
    }

    #[proptest]
    fn property_based_recurse_or_return_program_sanity_check(program: ProgramForRecurseOrReturn) {
        program.assemble().run()?;
    }

    pub(crate) fn test_program_for_write_mem_read_mem() -> ProgramAndInput {
        ProgramAndInput::new(triton_program! {
            push 3 push 1 push 2    // _ 3 1 2
            push 7                  // _ 3 1 2 7
            write_mem 3             // _ 10
            push -1 add             // _ 9
            read_mem 2              // _ 3 1 7
            pop 1                   // _ 3 1
            assert halt             // _ 3
        })
    }

    pub(crate) fn test_program_for_hash() -> ProgramAndInput {
        let program = triton_program!(
            push 0 // filler to keep the OpStack large enough throughout the program
            push 0 push 0 push 1 push 2 push 3
            hash
            read_io 1 eq assert halt
        );
        let hash_input = bfe_array![3, 2, 1, 0, 0, 0, 0, 0, 0, 0];
        let digest = Tip5::hash_10(&hash_input);
        ProgramAndInput::new(program).with_input(&digest[..=0])
    }

    /// Helper function that returns code to push a digest to the top of the stack
    fn push_digest_to_stack_tasm(Digest([d0, d1, d2, d3, d4]): Digest) -> Vec<LabelledInstruction> {
        triton_asm!(push {d4} push {d3} push {d2} push {d1} push {d0})
    }

    pub(crate) fn test_program_for_merkle_step_right_sibling() -> ProgramAndInput {
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
        ProgramAndInput::new(program).with_non_determinism(non_determinism)
    }

    pub(crate) fn test_program_for_merkle_step_left_sibling() -> ProgramAndInput {
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
        ProgramAndInput::new(program).with_non_determinism(non_determinism)
    }

    pub(crate) fn test_program_for_assert_vector() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(
            push 1 push 2 push 3 push 4 push 5
            push 1 push 2 push 3 push 4 push 5
            assert_vector halt
        ))
    }

    pub(crate) fn test_program_for_sponge_instructions() -> ProgramAndInput {
        let push_10_zeros = triton_asm![push 0; 10];
        ProgramAndInput::new(triton_program!(
            sponge_init
            {&push_10_zeros} sponge_absorb
            {&push_10_zeros} sponge_absorb
            sponge_squeeze halt
        ))
    }

    pub(crate) fn test_program_for_sponge_instructions_2() -> ProgramAndInput {
        let push_5_zeros = triton_asm![push 0; 5];
        let program = triton_program! {
            sponge_init
            sponge_squeeze sponge_absorb
            {&push_5_zeros} hash
            sponge_squeeze sponge_absorb
            halt
        };
        ProgramAndInput::new(program)
    }

    pub(crate) fn test_program_for_many_sponge_instructions() -> ProgramAndInput {
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
        ProgramAndInput::new(program)
    }

    pub(crate) fn property_based_test_program_for_assert_vector() -> ProgramAndInput {
        let mut rng = ThreadRng::default();
        let st: [BFieldElement; 5] = rng.gen();

        let program = triton_program!(
            push {st[0]} push {st[1]} push {st[2]} push {st[3]} push {st[4]}
            read_io 5 assert_vector halt
        );

        ProgramAndInput::new(program).with_input(st)
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
            let push_array = |a: [_; tip5::RATE]| a.into_iter().rev().map(Push).collect_vec();

            match self {
                Self::SpongeInit => vec![SpongeInit],
                Self::SpongeAbsorb(input) => [push_array(input), vec![SpongeAbsorb]].concat(),
                Self::SpongeAbsorbMem(ram_pointer) => vec![Push(ram_pointer), SpongeAbsorbMem],
                Self::SpongeSqueeze => vec![SpongeSqueeze],
                Self::Hash(input) => [push_array(input), vec![Hash]].concat(),
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
        pub fn assemble(self) -> ProgramAndInput {
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

            ProgramAndInput::new(program)
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

    pub(crate) fn test_program_for_add_mul_invert() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(
            push  2 push -1 add assert
            push -1 push -1 mul assert
            push  3 dup 0 invert mul assert
            halt
        ))
    }

    pub(crate) fn property_based_test_program_for_split() -> ProgramAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u64() % BFieldElement::P;
        let hi = st0 >> 32;
        let lo = st0 & 0xffff_ffff;

        let program =
            triton_program!(push {st0} split read_io 1 eq assert read_io 1 eq assert halt);
        ProgramAndInput::new(program).with_input([lo, hi].map(BFieldElement::new))
    }

    pub(crate) fn test_program_for_eq() -> ProgramAndInput {
        let input = bfe_array![42];
        ProgramAndInput::new(triton_program!(read_io 1 divine 1 eq assert halt))
            .with_input(input)
            .with_non_determinism(input)
    }

    pub(crate) fn property_based_test_program_for_eq() -> ProgramAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u64() % BFieldElement::P;
        let input = bfe_array![st0];

        let program =
            triton_program!(push {st0} dup 0 read_io 1 eq assert dup 0 divine 1 eq assert halt);
        ProgramAndInput::new(program)
            .with_input(input)
            .with_non_determinism(input)
    }

    pub(crate) fn test_program_for_lsb() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(
            push 3 call lsb assert assert halt
            lsb:
                push 2 swap 1 div_mod return
        ))
    }

    pub(crate) fn property_based_test_program_for_lsb() -> ProgramAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u32();
        let lsb = st0 % 2;
        let st0_shift_right = st0 >> 1;

        let program = triton_program!(
            push {st0} call lsb read_io 1 eq assert read_io 1 eq assert halt
            lsb:
                push 2 swap 1 div_mod return
        );
        ProgramAndInput::new(program).with_input([lsb, st0_shift_right].map(|b| bfe!(b)))
    }

    pub(crate) fn test_program_0_lt_0() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(push 0 push 0 lt halt))
    }

    pub(crate) fn test_program_for_lt() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(
            push 5 push 2 lt assert push 2 push 5 lt push 0 eq assert halt
        ))
    }

    pub(crate) fn property_based_test_program_for_lt() -> ProgramAndInput {
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
        ProgramAndInput::new(program).with_input([result_0, result_1].map(|b| bfe!(b)))
    }

    pub(crate) fn test_program_for_and() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(
            push 5 push 3 and assert push 12 push 5 and push 4 eq assert halt
        ))
    }

    pub(crate) fn property_based_test_program_for_and() -> ProgramAndInput {
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
        ProgramAndInput::new(program).with_input([result_0, result_1].map(|b| bfe!(b)))
    }

    pub(crate) fn test_program_for_xor() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(
            push 7 push 6 xor assert push 5 push 12 xor push 9 eq assert halt
        ))
    }

    pub(crate) fn property_based_test_program_for_xor() -> ProgramAndInput {
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
        ProgramAndInput::new(program).with_input([result_0, result_1].map(|b| bfe!(b)))
    }

    pub(crate) fn test_program_for_log2floor() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(
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

    pub(crate) fn property_based_test_program_for_log2floor() -> ProgramAndInput {
        let mut rng = ThreadRng::default();

        let st0_0 = rng.next_u32();
        let l2f_0 = st0_0.ilog2();

        let st0_1 = rng.next_u32();
        let l2f_1 = st0_1.ilog2();

        let program = triton_program!(
            push {st0_0} log_2_floor read_io 1 eq assert
            push {st0_1} log_2_floor read_io 1 eq assert halt
        );
        ProgramAndInput::new(program).with_input([l2f_0, l2f_1].map(|b| bfe!(b)))
    }

    pub(crate) fn test_program_for_pow() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(
            // push <exponent: u32> push <base: BFE> pow push <result: BFE> eq assert
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

    pub(crate) fn property_based_test_program_for_pow() -> ProgramAndInput {
        let mut rng = ThreadRng::default();

        let base_0: BFieldElement = rng.gen();
        let exp_0 = rng.next_u32();
        let result_0 = base_0.mod_pow_u32(exp_0);

        let base_1: BFieldElement = rng.gen();
        let exp_1 = rng.next_u32();
        let result_1 = base_1.mod_pow_u32(exp_1);

        let program = triton_program!(
            push {exp_0} push {base_0} pow read_io 1 eq assert
            push {exp_1} push {base_1} pow read_io 1 eq assert halt
        );
        ProgramAndInput::new(program).with_input([result_0, result_1])
    }

    pub(crate) fn test_program_for_div_mod() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(push 2 push 3 div_mod assert assert halt))
    }

    pub(crate) fn property_based_test_program_for_div_mod() -> ProgramAndInput {
        let mut rng = ThreadRng::default();

        let denominator = rng.next_u32();
        let numerator = rng.next_u32();
        let quotient = numerator / denominator;
        let remainder = numerator % denominator;

        let program = triton_program!(
            push {denominator} push {numerator} div_mod read_io 1 eq assert read_io 1 eq assert halt
        );
        ProgramAndInput::new(program).with_input([remainder, quotient].map(|b| bfe!(b)))
    }

    pub(crate) fn test_program_for_starting_with_pop_count() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(pop_count dup 0 push 0 eq assert halt))
    }

    pub(crate) fn test_program_for_pop_count() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(push 10 pop_count push 2 eq assert halt))
    }

    pub(crate) fn property_based_test_program_for_pop_count() -> ProgramAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u32();
        let pop_count = st0.count_ones();
        let program = triton_program!(push {st0} pop_count read_io 1 eq assert halt);
        ProgramAndInput::new(program).with_input(bfe_array![pop_count])
    }

    pub(crate) fn property_based_test_program_for_is_u32() -> ProgramAndInput {
        let mut rng = ThreadRng::default();
        let st0_u32 = rng.next_u32();
        let st0_not_u32 = (u64::from(rng.next_u32()) << 32) + u64::from(rng.next_u32());
        let program = triton_program!(
            push {st0_u32} call is_u32 assert
            push {st0_not_u32} call is_u32 push 0 eq assert halt
            is_u32:
                 split pop 1 push 0 eq return
        );
        ProgramAndInput::new(program)
    }

    pub(crate) fn property_based_test_program_for_random_ram_access() -> ProgramAndInput {
        let mut rng = ThreadRng::default();
        let num_memory_accesses = rng.gen_range(10..50);
        let memory_addresses: Vec<BFieldElement> = random_elements(num_memory_accesses);
        let mut memory_values: Vec<BFieldElement> = random_elements(num_memory_accesses);
        let mut instructions = vec![];

        // Read some memory before first write to ensure that the memory is initialized with 0s.
        // Not all addresses are read to have different access patterns:
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
            let j = rng.gen_range(0..num_memory_accesses);
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
            let j = rng.gen_range(0..num_memory_accesses);
            writing_permutation.swap(i, j);
        }
        for idx in 0..num_memory_accesses / 2 {
            let address = memory_addresses[writing_permutation[idx]];
            let new_memory_value = rng.gen();
            memory_values[writing_permutation[idx]] = new_memory_value;
            instructions
                .extend(triton_asm!(push {new_memory_value} push {address} write_mem 1 pop 1));
        }

        // Read back all, i.e., unchanged and overwritten values in (different from before) random
        // order and check that the values did not change.
        let mut reading_permutation = (0..num_memory_accesses).collect_vec();
        for i in 0..num_memory_accesses {
            let j = rng.gen_range(0..num_memory_accesses);
            reading_permutation.swap(i, j);
        }
        for idx in reading_permutation {
            let address = memory_addresses[idx];
            let value = memory_values[idx];
            instructions
                .extend(triton_asm!(push {address} read_mem 1 pop 1 push {value} eq assert));
        }

        let program = triton_program! { { &instructions } halt };
        ProgramAndInput::new(program)
    }

    /// Sanity check for the relatively complex property-based test for random RAM access.
    #[test]
    fn run_dont_prove_property_based_test_for_random_ram_access() {
        let source_code_and_input = property_based_test_program_for_random_ram_access();
        source_code_and_input.run().unwrap();
    }

    #[test]
    fn can_compute_dot_product_from_uninitialized_ram() {
        let program = triton_program!(xx_dot_step xb_dot_step halt);
        program
            .run(PublicInput::default(), NonDeterminism::default())
            .unwrap();
    }

    pub(crate) fn property_based_test_program_for_xx_dot_step() -> ProgramAndInput {
        let mut rng = ThreadRng::default();
        let n = rng.gen_range(0..10);

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

        let vector_one = (0..n).map(|_| rng.gen()).collect_vec();
        let vector_two = (0..n).map(|_| rng.gen()).collect_vec();
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
        ProgramAndInput::new(code)
    }

    /// Sanity check
    #[test]
    fn run_dont_prove_property_based_test_program_for_xx_dot_step() {
        let source_code_and_input = property_based_test_program_for_xx_dot_step();
        source_code_and_input.run().unwrap();
    }

    pub(crate) fn property_based_test_program_for_xb_dot_step() -> ProgramAndInput {
        let mut rng = ThreadRng::default();
        let n = rng.gen_range(0..10);
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
        let vector_one = (0..n).map(|_| rng.gen::<XFieldElement>()).collect_vec();
        let vector_two = (0..n).map(|_| rng.gen::<BFieldElement>()).collect_vec();
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
        ProgramAndInput::new(code)
    }

    /// Sanity check
    #[test]
    fn run_dont_prove_property_based_test_program_for_xb_dot_step() {
        let source_code_and_input = property_based_test_program_for_xb_dot_step();
        source_code_and_input.run().unwrap();
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
        let program_and_input = ProgramAndInput::new(program);
        let_assert!(Err(err) = program_and_input.run());
        let_assert!(AssertionFailed = err.source);
    }

    pub(crate) fn test_program_for_split() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(
            push -2 split push 4294967295 eq assert push 4294967294 eq assert
            push -1 split push 0 eq assert push 4294967295 eq assert
            push  0 split push 0 eq assert push 0 eq assert
            push  1 split push 1 eq assert push 0 eq assert
            push  2 split push 2 eq assert push 0 eq assert
            push 4294967297 split assert assert
            halt
        ))
    }

    pub(crate) fn test_program_for_xx_add() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(
            push 5 push 6 push 7 push 8 push 9 push 10 xx_add halt
        ))
    }

    pub(crate) fn test_program_for_xx_mul() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(
            push 5 push 6 push 7 push 8 push 9 push 10 xx_mul halt
        ))
    }

    pub(crate) fn test_program_for_x_invert() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(
            push 5 push 6 push 7 x_invert halt
        ))
    }

    pub(crate) fn test_program_for_xb_mul() -> ProgramAndInput {
        ProgramAndInput::new(triton_program!(
            push 5 push 6 push 7 push 8 xb_mul halt
        ))
    }

    pub(crate) fn test_program_for_read_io_write_io() -> ProgramAndInput {
        let program = triton_program!(
            read_io 1 assert read_io 2 dup 1 dup 1 add write_io 1 mul push 5 write_io 2 halt
        );
        ProgramAndInput::new(program).with_input([1, 3, 14].map(|b| bfe!(b)))
    }

    pub(crate) fn test_program_claim_in_ram_corresponds_to_currently_running_program(
    ) -> ProgramAndInput {
        let program = triton_program! {
            dup 15 dup 15 dup 15 dup 15 dup 15  // _ [own_digest]
            push 4 read_mem 5 pop 1             // _ [own_digest] [claim_digest]
            assert_vector                       // _ [own_digest]
            halt
        };

        let program_digest = program.hash::<Tip5>();
        let enumerated_digest_elements = program_digest.values().into_iter().enumerate();
        let initial_ram = enumerated_digest_elements
            .map(|(address, d)| (bfe!(u64::try_from(address).unwrap()), d))
            .collect::<HashMap<_, _>>();
        let non_determinism = NonDeterminism::default().with_ram(initial_ram);

        ProgramAndInput::new(program).with_non_determinism(non_determinism)
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
        let actual_stdout = program.run([].into(), [].into())?;
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
        let actual_stdout = program.run([].into(), [].into())?;
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
        let actual_stdout = program.run([].into(), [].into())?;
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
        let actual_stdout = program.run([].into(), [].into())?;
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
        let actual_stdout = ProgramAndInput::new(program).run()?;
        let expected_stdout = vec![minuend - subtrahend];

        prop_assert_eq!(expected_stdout, actual_stdout);
    }

    // compile-time assertion
    const _OP_STACK_IS_BIG_ENOUGH: () =
        std::assert!(2 * tip5::DIGEST_LENGTH <= OpStackElement::COUNT);

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
        let mut vm_state = VMState::new(&program, [].into(), [].into());
        let_assert!(Ok(()) = vm_state.run());
        assert!(b_field_element::BFIELD_ZERO == vm_state.op_stack[ST0]);
    }

    #[test]
    fn run_tvm_merkle_step_right_sibling() {
        let program_and_input = test_program_for_merkle_step_right_sibling();
        let_assert!(Ok(_) = program_and_input.run());
    }

    #[test]
    fn run_tvm_merkle_step_left_sibling() {
        let program_and_input = test_program_for_merkle_step_left_sibling();
        let_assert!(Ok(_) = program_and_input.run());
    }

    #[test]
    fn run_tvm_halt_then_do_stuff() {
        let program = triton_program!(halt push 1 push 2 add invert write_io 5);
        let_assert!(Ok((aet, _)) = program.trace_execution([].into(), [].into()));

        let_assert!(Some(last_processor_row) = aet.processor_trace.rows().into_iter().last());
        let clk_count = last_processor_row[ProcessorBaseTableColumn::CLK.base_table_index()];
        assert!(b_field_element::BFIELD_ZERO == clk_count);

        let last_instruction = last_processor_row[ProcessorBaseTableColumn::CI.base_table_index()];
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

        let mut vm_state = VMState::new(&program, [].into(), [].into());
        let_assert!(Ok(()) = vm_state.run());
        assert!(4 == vm_state.op_stack[ST0].value());
        assert!(7 == vm_state.op_stack[ST1].value());
        assert!(14 == vm_state.op_stack[ST2].value());
        assert!(18 == vm_state.op_stack[ST3].value());
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

        let mut vm_state = VMState::new(&program, [].into(), [].into());
        let_assert!(Ok(()) = vm_state.run());
        assert!(2_u64 == vm_state.op_stack[ST0].value());
        assert!(5_u64 == vm_state.op_stack[ST1].value());
        assert!(5_u64 == vm_state.op_stack[ST2].value());
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

        let program = MERKLE_TREE_AUTHENTICATION_PATH_VERIFY.clone();
        assert!(let Ok(_) = program.run(public_input.into(), non_determinism));
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
        let_assert!(Ok(output) = get_collinear_y_program.run(public_input.into(), [].into()));
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

        let_assert!(Ok(standard_out) = countdown_program.run([].into(), [].into()));
        let expected = (0..=10).map(BFieldElement::new).rev().collect_vec();
        assert!(expected == standard_out);
    }

    #[test]
    fn run_tvm_fibonacci_tvm() {
        let program = FIBONACCI_SEQUENCE.clone();
        let_assert!(Ok(standard_out) = program.run(bfe_array![7].into(), [].into()));
        assert!(bfe!(21) == standard_out[0]);
    }

    #[test]
    fn run_tvm_swap() {
        let program = triton_program!(push 1 push 2 swap 1 assert write_io 1 halt);
        let_assert!(Ok(standard_out) = program.run([].into(), [].into()));
        assert!(bfe!(2) == standard_out[0]);
    }

    #[test]
    fn read_mem_uninitialized() {
        let program = triton_program!(read_mem 3 halt);
        let_assert!(Ok((aet, _)) = program.trace_execution([].into(), [].into()));
        assert!(2 == aet.processor_trace.nrows());
    }

    #[test]
    fn read_non_deterministically_initialized_ram_at_address_0() {
        let program = triton_program!(push 0 read_mem 1 pop 1 write_io 1 halt);

        let mut initial_ram = HashMap::new();
        initial_ram.insert(bfe!(0), bfe!(42));

        let public_input = PublicInput::default();
        let secret_input = NonDeterminism::default().with_ram(initial_ram);

        let_assert!(Ok(public_output) = program.run(public_input.clone(), secret_input.clone()));
        assert!(42 == public_output[0].value());

        let log2_fri_expansion_factor = 2;
        prove_with_low_security_level(
            &program,
            public_input,
            secret_input,
            log2_fri_expansion_factor,
        );
    }

    #[proptest(cases = 10)]
    fn read_non_deterministically_initialized_ram_at_random_address(
        #[strategy(arb())] address: BFieldElement,
        #[strategy(arb())] value: BFieldElement,
    ) {
        let program = triton_program!(
            read_mem 1 swap 1 write_io 1
            push {address} read_mem 1 pop 1 write_io 1
            halt
        );

        let mut initial_ram = HashMap::new();
        initial_ram.insert(address, value);

        let public_input = PublicInput::default();
        let secret_input = NonDeterminism::default().with_ram(initial_ram);

        let_assert!(Ok(public_output) = program.run(public_input.clone(), secret_input.clone()));
        assert!(0 == public_output[0].value());
        assert!(value == public_output[1]);

        let log2_fri_expansion_factor = 2;
        prove_with_low_security_level(
            &program,
            public_input,
            secret_input,
            log2_fri_expansion_factor,
        );
    }

    #[test]
    fn program_without_halt() {
        let program = triton_program!(nop);
        let_assert!(Err(err) = program.trace_execution([].into(), [].into()));
        let_assert!(InstructionPointerOverflow = err.source);
    }

    #[test]
    fn verify_sudoku() {
        let program = VERIFY_SUDOKU.clone();
        let sudoku = [
            8, 5, 9, /**/ 7, 6, 1, /**/ 4, 2, 3, //
            4, 2, 6, /**/ 8, 5, 3, /**/ 7, 9, 1, //
            7, 1, 3, /**/ 9, 2, 4, /**/ 8, 5, 6, //
            /*************************************/
            9, 6, 1, /**/ 5, 3, 7, /**/ 2, 8, 4, //
            2, 8, 7, /**/ 4, 1, 9, /**/ 6, 3, 5, //
            3, 4, 5, /**/ 2, 8, 6, /**/ 1, 7, 9, //
            /*************************************/
            5, 3, 4, /**/ 6, 7, 8, /**/ 9, 1, 2, //
            6, 7, 2, /**/ 1, 9, 5, /**/ 3, 4, 8, //
            1, 9, 8, /**/ 3, 4, 2, /**/ 5, 6, 7, //
        ];

        let std_in = PublicInput::from(sudoku.map(|b| bfe!(b)));
        let secret_in = NonDeterminism::default();
        assert!(let Ok(_) = program.trace_execution(std_in, secret_in));

        // rows and columns adhere to Sudoku rules, boxes do not
        let bad_sudoku = [
            1, 2, 3, /**/ 4, 5, 7, /**/ 8, 9, 6, //
            4, 3, 1, /**/ 5, 2, 9, /**/ 6, 7, 8, //
            2, 7, 9, /**/ 6, 1, 3, /**/ 5, 8, 4, //
            /*************************************/
            7, 6, 5, /**/ 3, 4, 8, /**/ 9, 2, 1, //
            5, 1, 4, /**/ 9, 8, 6, /**/ 7, 3, 2, //
            6, 8, 2, /**/ 7, 9, 4, /**/ 1, 5, 3, //
            /*************************************/
            3, 5, 6, /**/ 8, 7, 2, /**/ 4, 1, 9, //
            9, 4, 8, /**/ 1, 3, 5, /**/ 2, 6, 7, //
            8, 9, 7, /**/ 2, 6, 1, /**/ 3, 4, 5, //
        ];
        let bad_std_in = PublicInput::from(bad_sudoku.map(|b| bfe!(b)));
        let secret_in = NonDeterminism::default();
        let_assert!(Err(err) = program.trace_execution(bad_std_in, secret_in));
        let_assert!(AssertionFailed = err.source);
    }

    fn instruction_does_not_change_vm_state_when_crashing_vm(
        program_and_input: ProgramAndInput,
        num_preparatory_steps: usize,
    ) {
        let mut vm_state = VMState::new(
            &program_and_input.program,
            program_and_input.public_input,
            program_and_input.non_determinism,
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
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 1);
    }

    #[test]
    fn instruction_divine_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { divine 1 halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 0);
    }

    #[test]
    fn instruction_assert_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { push 0 assert halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 1);
    }

    #[test]
    fn instruction_merkle_step_does_not_change_vm_state_when_crashing_vm_invalid_node_index() {
        let non_u32 = u64::from(u32::MAX) + 1;
        let program = triton_program! { push {non_u32} swap 5 merkle_step halt };
        let nondeterminism = NonDeterminism::default().with_digests([Digest::default()]);
        let program_and_input = ProgramAndInput::new(program).with_non_determinism(nondeterminism);
        instruction_does_not_change_vm_state_when_crashing_vm(program_and_input, 2);
    }

    #[test]
    fn instruction_merkle_step_does_not_change_vm_state_when_crashing_vm_no_nd_digests() {
        let valid_u32 = u64::from(u32::MAX);
        let program = triton_program! { push {valid_u32} swap 5 merkle_step halt };
        let program_and_input = ProgramAndInput::new(program);
        instruction_does_not_change_vm_state_when_crashing_vm(program_and_input, 2);
    }

    #[test]
    fn instruction_assert_vector_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { push 0 push 1 push 0 push 0 push 0 assert_vector halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 5);
    }

    #[test]
    fn instruction_sponge_absorb_does_not_change_vm_state_when_crashing_vm_sponge_uninit() {
        let ten_pushes = triton_asm![push 0; 10];
        let program = triton_program! { {&ten_pushes} sponge_absorb halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 10);
    }

    #[test]
    fn instruction_sponge_absorb_does_not_change_vm_state_when_crashing_vm_stack_too_small() {
        let program = triton_program! { sponge_init sponge_absorb halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 1);
    }

    #[test]
    fn instruction_sponge_absorb_mem_does_not_change_vm_state_when_crashing_vm_sponge_uninit() {
        let program = triton_program! { sponge_absorb_mem halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 0);
    }

    #[test]
    fn instruction_sponge_squeeze_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { sponge_squeeze halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 0);
    }

    #[test]
    fn instruction_invert_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { push 0 invert halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 1);
    }

    #[test]
    fn instruction_lt_does_not_change_vm_state_when_crashing_vm() {
        let non_u32 = u64::from(u32::MAX) + 1;
        let program = triton_program! { push {non_u32} lt halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 1);
    }

    #[test]
    fn instruction_and_does_not_change_vm_state_when_crashing_vm() {
        let non_u32 = u64::from(u32::MAX) + 1;
        let program = triton_program! { push {non_u32} and halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 1);
    }

    #[test]
    fn instruction_xor_does_not_change_vm_state_when_crashing_vm() {
        let non_u32 = u64::from(u32::MAX) + 1;
        let program = triton_program! { push {non_u32} xor halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 1);
    }

    #[test]
    fn instruction_log_2_floor_on_non_u32_operand_does_not_change_vm_state_when_crashing_vm() {
        let non_u32 = u64::from(u32::MAX) + 1;
        let program = triton_program! { push {non_u32} log_2_floor halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 1);
    }

    #[test]
    fn instruction_log_2_floor_on_operand_0_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { push 0 log_2_floor halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 1);
    }

    #[test]
    fn instruction_pow_does_not_change_vm_state_when_crashing_vm() {
        let non_u32 = u64::from(u32::MAX) + 1;
        let program = triton_program! { push {non_u32} push 0 pow halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 2);
    }

    #[test]
    fn instruction_div_mod_on_non_u32_operand_does_not_change_vm_state_when_crashing_vm() {
        let non_u32 = u64::from(u32::MAX) + 1;
        let program = triton_program! { push {non_u32} push 0 div_mod halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 2);
    }

    #[test]
    fn instruction_div_mod_on_denominator_0_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { push 0 push 1 div_mod halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 2);
    }

    #[test]
    fn instruction_pop_count_does_not_change_vm_state_when_crashing_vm() {
        let non_u32 = u64::from(u32::MAX) + 1;
        let program = triton_program! { push {non_u32} pop_count halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 1);
    }

    #[test]
    fn instruction_x_invert_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { x_invert halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 0);
    }

    #[test]
    fn instruction_read_io_does_not_change_vm_state_when_crashing_vm() {
        let program = triton_program! { read_io 1 halt };
        instruction_does_not_change_vm_state_when_crashing_vm(ProgramAndInput::new(program), 0);
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

        let mut vmstate = VMState::new(&program, public_input, secret_input);
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

        let mut vmstate = VMState::new(&program, public_input, secret_input);
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
}
