use std::collections::HashMap;
use std::collections::VecDeque;
use std::convert::TryInto;
use std::fmt::Display;

use anyhow::anyhow;
use anyhow::bail;
use anyhow::Result;
use ndarray::Array1;
use num_traits::One;
use num_traits::Zero;
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::b_field_element::BFIELD_ZERO;
use twenty_first::shared_math::tip5;
use twenty_first::shared_math::tip5::Tip5;
use twenty_first::shared_math::tip5::Tip5State;
use twenty_first::shared_math::tip5::DIGEST_LENGTH;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::algebraic_hasher::Domain;

use crate::error::InstructionError::*;
use crate::instruction::AnInstruction::*;
use crate::instruction::Instruction;
use crate::op_stack::OpStack;
use crate::op_stack::OpStackElement;
use crate::op_stack::OpStackElement::*;
use crate::program::Program;
use crate::program::PublicInput;
use crate::program::SecretInput;
use crate::stark::StarkHasher;
use crate::table::processor_table;
use crate::table::processor_table::ProcessorTraceRow;
use crate::table::table_column::MasterBaseTableColumn;
use crate::table::table_column::ProcessorBaseTableColumn;
use crate::vm::CoProcessorCall::*;

/// The number of helper variable registers
pub const NUM_HELPER_VARIABLE_REGISTERS: usize = 7;

#[derive(Debug, Clone)]
pub struct VMState<'pgm> {
    // Memory
    /// The **program memory** stores the instructions (and their arguments) of the program
    /// currently being executed by Triton VM. It is read-only.
    pub program: &'pgm [Instruction],

    /// A list of [`BFieldElement`]s the program can read from using instruction `read_io`.
    pub public_input_stream: VecDeque<BFieldElement>,

    /// A list of [`BFieldElement`]s the program can read from using instructions `divine`
    /// and `divine_sibling`.
    pub secret_input_stream: VecDeque<BFieldElement>,

    /// A list of [`BFieldElement`]s the program can write to using instruction `write_io`.
    pub public_output: Vec<BFieldElement>,

    /// The read-write **random-access memory** allows Triton VM to store arbitrary data.
    pub ram: HashMap<BFieldElement, BFieldElement>,

    /// The **Op-stack memory** stores Triton VM's entire operational stack.
    pub op_stack: OpStack,

    /// The **Jump-stack memory** stores the entire jump stack.
    pub jump_stack: Vec<(BFieldElement, BFieldElement)>,

    // Registers
    /// Number of cycles the program has been running for
    pub cycle_count: u32,

    /// Current instruction's address in program memory
    pub instruction_pointer: usize,

    /// The instruction that was executed last
    pub previous_instruction: BFieldElement,

    /// RAM pointer
    pub ramp: u64,

    /// The current state of the one, global Sponge that can be manipulated using instructions
    /// `AbsorbInit`, `Absorb`, and `Squeeze`. Instruction `AbsorbInit` resets the state prior to
    /// absorbing.
    /// Note that this is the _full_ state, including capacity. The capacity should never be
    /// exposed outside of the VM.
    pub sponge_state: [BFieldElement; tip5::STATE_SIZE],

    // Bookkeeping
    /// Indicates whether the terminating instruction `halt` has been executed.
    pub halting: bool,
}

/// A call from the main processor to one of the co-processors, including the trace for that
/// co-processor or enough information to deduce the trace.
#[derive(Debug, PartialEq, Eq)]
pub enum CoProcessorCall {
    /// Trace of the state registers for hash coprocessor table when executing instruction `hash`
    /// or any of the Sponge instructions `absorb_init`, `absorb`, `squeeze`.
    /// One row per round in the Tip5 permutation.
    Tip5Trace(
        Instruction,
        Box<[[BFieldElement; tip5::STATE_SIZE]; 1 + tip5::NUM_ROUNDS]>,
    ),

    /// Executed u32 instruction as well as its left-hand side and right-hand side
    U32TableEntries(Vec<(Instruction, BFieldElement, BFieldElement)>),
}

impl<'pgm> VMState<'pgm> {
    /// Create initial `VMState` for a given `program`
    ///
    /// Since `program` is read-only across individual states, and multiple
    /// inner helper functions refer to it, a read-only reference is kept in
    /// the struct.
    pub fn new(
        program: &'pgm Program,
        public_input: PublicInput,
        secret_input: SecretInput,
    ) -> Self {
        let program_digest = program.hash::<StarkHasher>();

        Self {
            program: &program.instructions,
            public_input_stream: public_input.stream.into(),
            secret_input_stream: secret_input.stream.into(),
            public_output: vec![],
            ram: secret_input.ram,
            op_stack: OpStack::new(program_digest),
            jump_stack: vec![],
            cycle_count: 0,
            instruction_pointer: 0,
            previous_instruction: Default::default(),
            ramp: 0,
            sponge_state: Default::default(),
            halting: false,
        }
    }

    pub fn derive_helper_variables(&self) -> [BFieldElement; NUM_HELPER_VARIABLE_REGISTERS] {
        let mut hvs = [BFieldElement::zero(); NUM_HELPER_VARIABLE_REGISTERS];
        let Ok(current_instruction) = self.current_instruction() else {
            return hvs;
        };

        if current_instruction.shrinks_op_stack() {
            let op_stack_pointer = self.op_stack.op_stack_pointer();
            let maximum_op_stack_pointer = BFieldElement::new(OpStackElement::COUNT as u64);
            let op_stack_pointer_minus_maximum = op_stack_pointer - maximum_op_stack_pointer;
            hvs[0] = op_stack_pointer_minus_maximum.inverse_or_zero();
        }

        match current_instruction {
            // For instructions making use of indicator polynomials, i.e., `dup` and `swap`.
            // Sets the corresponding helper variables to the bit-decomposed argument such that the
            // indicator polynomials of the AIR actually evaluate to 0.
            Dup(arg) | Swap(arg) => {
                let arg_val: u64 = arg.into();
                hvs[0] = BFieldElement::new(arg_val % 2);
                hvs[1] = BFieldElement::new((arg_val >> 1) % 2);
                hvs[2] = BFieldElement::new((arg_val >> 2) % 2);
                hvs[3] = BFieldElement::new((arg_val >> 3) % 2);
            }
            Skiz => {
                let st0 = self.op_stack.peek_at(ST0);
                hvs[1] = st0.inverse_or_zero();
                let next_opcode = self.next_instruction_or_argument().value();
                let decomposition = Self::decompose_opcode_for_instruction_skiz(next_opcode);
                let decomposition = decomposition.map(BFieldElement::new);
                hvs[2..7].copy_from_slice(&decomposition);
            }
            DivineSibling => {
                let node_index = self.op_stack.peek_at(ST10).value();
                // set hv0 register to least significant bit of st10
                hvs[0] = BFieldElement::new(node_index % 2);
            }
            Split => {
                let elem = self.op_stack.peek_at(ST0);
                let n: u64 = elem.value();
                let lo = BFieldElement::new(n & 0xffff_ffff);
                let hi = BFieldElement::new(n >> 32);
                if !lo.is_zero() {
                    let max_val_of_hi = BFieldElement::new(2_u64.pow(32) - 1);
                    hvs[0] = (hi - max_val_of_hi).inverse_or_zero();
                }
            }
            Eq => {
                let lhs = self.op_stack.peek_at(ST0);
                let rhs = self.op_stack.peek_at(ST1);
                hvs[1] = (rhs - lhs).inverse_or_zero();
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
    pub fn step(&mut self) -> Result<Option<CoProcessorCall>> {
        self.previous_instruction = match self.current_instruction() {
            Ok(instruction) => instruction.opcode_b(),
            // trying to read past the end of the program doesn't change the previous instruction
            Err(_) => self.previous_instruction,
        };

        let maybe_co_processor_trace = match self.current_instruction()? {
            Pop => self.instruction_pop()?,
            Push(field_element) => self.instruction_push(field_element),
            Divine => self.instruction_divine()?,
            Dup(stack_element) => self.instruction_dup(stack_element),
            Swap(stack_element) => self.instruction_swap(stack_element),
            Nop => self.instruction_nop(),
            Skiz => self.instruction_skiz()?,
            Call(address) => self.instruction_call(address),
            Return => self.instruction_return()?,
            Recurse => self.instruction_recurse()?,
            Assert => self.instruction_assert()?,
            Halt => self.instruction_halt(),
            ReadMem => self.instruction_read_mem(),
            WriteMem => self.instruction_write_mem()?,
            Hash => self.instruction_hash()?,
            AbsorbInit | Absorb => self.instruction_absorb()?,
            Squeeze => self.instruction_squeeze()?,
            DivineSibling => self.instruction_divine_sibling()?,
            AssertVector => self.instruction_assert_vector()?,
            Add => self.instruction_add()?,
            Mul => self.instruction_mul()?,
            Invert => self.instruction_invert()?,
            Eq => self.instruction_eq()?,
            Split => self.instruction_split()?,
            Lt => self.instruction_lt()?,
            And => self.instruction_and()?,
            Xor => self.instruction_xor()?,
            Log2Floor => self.instruction_log_2_floor()?,
            Pow => self.instruction_pow()?,
            Div => self.instruction_div()?,
            PopCount => self.instruction_pop_count()?,
            XxAdd => self.instruction_xx_add()?,
            XxMul => self.instruction_xx_mul()?,
            XInvert => self.instruction_x_invert()?,
            XbMul => self.instruction_xb_mul()?,
            WriteIo => self.instruction_write_io()?,
            ReadIo => self.instruction_read_io()?,
        };

        if self.op_stack.is_too_shallow() {
            bail!(OpStackTooShallow);
        }

        self.cycle_count += 1;
        Ok(maybe_co_processor_trace)
    }

    fn instruction_pop(&mut self) -> Result<Option<CoProcessorCall>> {
        self.op_stack.pop()?;
        self.instruction_pointer += 1;
        Ok(None)
    }

    fn instruction_push(&mut self, field_element: BFieldElement) -> Option<CoProcessorCall> {
        self.op_stack.push(field_element);
        self.instruction_pointer += 2;
        None
    }

    fn instruction_divine(&mut self) -> Result<Option<CoProcessorCall>> {
        let elem = self.secret_input_stream.pop_front().ok_or(anyhow!(
            "Instruction `divine`: secret input buffer is empty."
        ))?;
        self.op_stack.push(elem);
        self.instruction_pointer += 1;
        Ok(None)
    }

    fn instruction_dup(&mut self, stack_element: OpStackElement) -> Option<CoProcessorCall> {
        let stack_element = self.op_stack.peek_at(stack_element);
        self.op_stack.push(stack_element);
        self.instruction_pointer += 2;
        None
    }

    fn instruction_swap(&mut self, stack_element: OpStackElement) -> Option<CoProcessorCall> {
        self.op_stack.swap_top_with(stack_element);
        self.instruction_pointer += 2;
        None
    }

    fn instruction_nop(&mut self) -> Option<CoProcessorCall> {
        self.instruction_pointer += 1;
        None
    }

    fn instruction_skiz(&mut self) -> Result<Option<CoProcessorCall>> {
        let elem = self.op_stack.pop()?;
        self.instruction_pointer += match elem.is_zero() {
            true => 1 + self.next_instruction()?.size(),
            false => 1,
        };
        Ok(None)
    }

    fn instruction_call(&mut self, addr: BFieldElement) -> Option<CoProcessorCall> {
        let o_plus_2 = self.instruction_pointer as u32 + 2;
        let pair = (BFieldElement::new(o_plus_2 as u64), addr);
        self.jump_stack.push(pair);
        self.instruction_pointer = addr.value() as usize;
        None
    }

    fn instruction_return(&mut self) -> Result<Option<CoProcessorCall>> {
        let (orig_addr, _dest_addr) = self.jump_stack_pop()?;
        self.instruction_pointer = orig_addr.value() as usize;
        Ok(None)
    }

    fn instruction_recurse(&mut self) -> Result<Option<CoProcessorCall>> {
        let (_orig_addr, dest_addr) = self.jump_stack_peek()?;
        self.instruction_pointer = dest_addr.value() as usize;
        Ok(None)
    }

    fn instruction_assert(&mut self) -> Result<Option<CoProcessorCall>> {
        let elem = self.op_stack.pop()?;
        if !elem.is_one() {
            bail!(AssertionFailed(
                self.instruction_pointer,
                self.cycle_count,
                elem,
            ));
        }
        self.instruction_pointer += 1;
        Ok(None)
    }

    fn instruction_halt(&mut self) -> Option<CoProcessorCall> {
        self.halting = true;
        self.instruction_pointer += 1;
        None
    }

    fn instruction_read_mem(&mut self) -> Option<CoProcessorCall> {
        let ramp = self.op_stack.peek_at(ST0);
        let ramv = self.memory_get(&ramp);
        self.op_stack.push(ramv);
        self.ramp = ramp.value();
        self.instruction_pointer += 1;
        None
    }

    fn instruction_write_mem(&mut self) -> Result<Option<CoProcessorCall>> {
        let ramp = self.op_stack.peek_at(ST1);
        let ramv = self.op_stack.pop()?;
        self.ramp = ramp.value();
        self.ram.insert(ramp, ramv);
        self.instruction_pointer += 1;
        Ok(None)
    }

    fn instruction_hash(&mut self) -> Result<Option<CoProcessorCall>> {
        let to_hash = self.op_stack.pop_multiple::<{ tip5::RATE }>()?;
        let mut hash_input = Tip5State::new(Domain::FixedLength);
        hash_input.state[..tip5::RATE].copy_from_slice(&to_hash);
        let tip5_trace = Tip5::trace(&mut hash_input);
        let hash_output = &tip5_trace[tip5_trace.len() - 1][0..DIGEST_LENGTH];

        for i in (0..DIGEST_LENGTH).rev() {
            self.op_stack.push(hash_output[i]);
        }
        for _ in 0..DIGEST_LENGTH {
            self.op_stack.push(BFieldElement::zero());
        }
        let co_processor_trace = Some(Tip5Trace(Hash, Box::new(tip5_trace)));

        self.instruction_pointer += 1;
        Ok(co_processor_trace)
    }

    fn instruction_absorb(&mut self) -> Result<Option<CoProcessorCall>> {
        // fetch top elements but don't alter the stack
        let to_absorb = self.op_stack.pop_multiple::<{ tip5::RATE }>()?;
        for i in (0..tip5::RATE).rev() {
            self.op_stack.push(to_absorb[i]);
        }

        if self.current_instruction()? == AbsorbInit {
            self.sponge_state = Tip5State::new(Domain::VariableLength).state;
        }
        self.sponge_state[..tip5::RATE].copy_from_slice(&to_absorb);
        let tip5_trace = Tip5::trace(&mut Tip5State {
            state: self.sponge_state,
        });
        self.sponge_state = tip5_trace.last().unwrap().to_owned();
        let co_processor_trace = Some(Tip5Trace(self.current_instruction()?, Box::new(tip5_trace)));

        self.instruction_pointer += 1;
        Ok(co_processor_trace)
    }

    fn instruction_squeeze(&mut self) -> Result<Option<CoProcessorCall>> {
        let _ = self.op_stack.pop_multiple::<{ tip5::RATE }>()?;
        for i in (0..tip5::RATE).rev() {
            self.op_stack.push(self.sponge_state[i]);
        }
        let tip5_trace = Tip5::trace(&mut Tip5State {
            state: self.sponge_state,
        });
        self.sponge_state = tip5_trace.last().unwrap().to_owned();
        let co_processor_trace = Some(Tip5Trace(Squeeze, Box::new(tip5_trace)));

        self.instruction_pointer += 1;
        Ok(co_processor_trace)
    }

    fn instruction_divine_sibling(&mut self) -> Result<Option<CoProcessorCall>> {
        let _st0_through_st4 = self.op_stack.pop_multiple::<{ DIGEST_LENGTH }>()?;
        let known_digest = self.op_stack.pop_multiple()?;

        let node_index = self.op_stack.pop_u32()?;
        let parent_node_index = node_index / 2;
        self.op_stack.push((parent_node_index as u64).into());

        let mut sibling_digest = self.secret_input_pop_multiple()?;
        sibling_digest.reverse();
        let (left_digest, right_digest) =
            Self::put_known_digest_on_correct_side(node_index, known_digest, sibling_digest);

        for &digest_element in right_digest.iter().rev() {
            self.op_stack.push(digest_element);
        }
        for &digest_element in left_digest.iter().rev() {
            self.op_stack.push(digest_element);
        }
        self.instruction_pointer += 1;
        Ok(None)
    }

    fn instruction_assert_vector(&mut self) -> Result<Option<CoProcessorCall>> {
        if !self.assert_vector() {
            bail!(AssertionFailed(
                self.instruction_pointer,
                self.cycle_count,
                self.op_stack.peek_at(ST0)
            ));
        }
        self.instruction_pointer += 1;
        Ok(None)
    }

    fn instruction_add(&mut self) -> Result<Option<CoProcessorCall>> {
        let lhs = self.op_stack.pop()?;
        let rhs = self.op_stack.pop()?;
        self.op_stack.push(lhs + rhs);
        self.instruction_pointer += 1;
        Ok(None)
    }

    fn instruction_mul(&mut self) -> Result<Option<CoProcessorCall>> {
        let lhs = self.op_stack.pop()?;
        let rhs = self.op_stack.pop()?;
        self.op_stack.push(lhs * rhs);
        self.instruction_pointer += 1;
        Ok(None)
    }

    fn instruction_invert(&mut self) -> Result<Option<CoProcessorCall>> {
        let elem = self.op_stack.pop()?;
        if elem.is_zero() {
            bail!(InverseOfZero);
        }
        self.op_stack.push(elem.inverse());
        self.instruction_pointer += 1;
        Ok(None)
    }

    fn instruction_eq(&mut self) -> Result<Option<CoProcessorCall>> {
        let lhs = self.op_stack.pop()?;
        let rhs = self.op_stack.pop()?;
        self.op_stack.push(Self::eq(lhs, rhs));
        self.instruction_pointer += 1;
        Ok(None)
    }

    fn instruction_split(&mut self) -> Result<Option<CoProcessorCall>> {
        let elem = self.op_stack.pop()?;
        let lo = BFieldElement::new(elem.value() & 0xffff_ffff);
        let hi = BFieldElement::new(elem.value() >> 32);
        self.op_stack.push(hi);
        self.op_stack.push(lo);

        let u32_table_entry = (Split, lo, hi);
        let co_processor_trace = Some(U32TableEntries(vec![u32_table_entry]));

        self.instruction_pointer += 1;
        Ok(co_processor_trace)
    }

    fn instruction_lt(&mut self) -> Result<Option<CoProcessorCall>> {
        let lhs = self.op_stack.pop_u32()?;
        let rhs = self.op_stack.pop_u32()?;
        let lt = BFieldElement::new((lhs < rhs) as u64);
        self.op_stack.push(lt);

        let u32_table_entry = (Lt, (lhs as u64).into(), (rhs as u64).into());
        let co_processor_trace = Some(U32TableEntries(vec![u32_table_entry]));

        self.instruction_pointer += 1;
        Ok(co_processor_trace)
    }

    fn instruction_and(&mut self) -> Result<Option<CoProcessorCall>> {
        let lhs = self.op_stack.pop_u32()?;
        let rhs = self.op_stack.pop_u32()?;
        let and = BFieldElement::new((lhs & rhs) as u64);
        self.op_stack.push(and);

        let u32_table_entry = (And, (lhs as u64).into(), (rhs as u64).into());
        let co_processor_trace = Some(U32TableEntries(vec![u32_table_entry]));

        self.instruction_pointer += 1;
        Ok(co_processor_trace)
    }

    fn instruction_xor(&mut self) -> Result<Option<CoProcessorCall>> {
        let lhs = self.op_stack.pop_u32()?;
        let rhs = self.op_stack.pop_u32()?;
        let xor = BFieldElement::new((lhs ^ rhs) as u64);
        self.op_stack.push(xor);

        // Triton VM uses the following equality to compute the results of both the `and`
        // and `xor` instruction using the u32 coprocessor's `and` capability:
        // a ^ b = a + b - 2 · (a & b)
        let u32_table_entry = (And, (lhs as u64).into(), (rhs as u64).into());
        let co_processor_trace = Some(U32TableEntries(vec![u32_table_entry]));

        self.instruction_pointer += 1;
        Ok(co_processor_trace)
    }

    fn instruction_log_2_floor(&mut self) -> Result<Option<CoProcessorCall>> {
        let lhs = self.op_stack.pop_u32()?;
        if lhs.is_zero() {
            bail!(LogarithmOfZero);
        }
        let l2f = BFieldElement::new(lhs.ilog2().into());
        self.op_stack.push(l2f);

        let u32_table_entry = (Log2Floor, (lhs as u64).into(), BFIELD_ZERO);
        let co_processor_trace = Some(U32TableEntries(vec![u32_table_entry]));

        self.instruction_pointer += 1;
        Ok(co_processor_trace)
    }

    fn instruction_pow(&mut self) -> Result<Option<CoProcessorCall>> {
        let lhs = self.op_stack.pop()?;
        let rhs = self.op_stack.pop_u32()?;
        let pow = lhs.mod_pow(rhs as u64);
        self.op_stack.push(pow);

        let u32_table_entry = (Pow, lhs, (rhs as u64).into());
        let co_processor_trace = Some(U32TableEntries(vec![u32_table_entry]));

        self.instruction_pointer += 1;
        Ok(co_processor_trace)
    }

    fn instruction_div(&mut self) -> Result<Option<CoProcessorCall>> {
        let numerator = self.op_stack.pop_u32()?;
        let denominator = self.op_stack.pop_u32()?;
        if denominator.is_zero() {
            bail!(DivisionByZero);
        }
        let quotient = BFieldElement::new((numerator / denominator) as u64);
        let remainder = BFieldElement::new((numerator % denominator) as u64);
        self.op_stack.push(quotient);
        self.op_stack.push(remainder);

        let u32_table_entry_0 = (Lt, remainder, (denominator as u64).into());
        let u32_table_entry_1 = (Split, (numerator as u64).into(), quotient);
        let co_processor_trace = Some(U32TableEntries(vec![u32_table_entry_0, u32_table_entry_1]));

        self.instruction_pointer += 1;
        Ok(co_processor_trace)
    }

    fn instruction_pop_count(&mut self) -> Result<Option<CoProcessorCall>> {
        let lhs = self.op_stack.pop_u32()?;
        let pop_count = BFieldElement::new(lhs.count_ones() as u64);
        self.op_stack.push(pop_count);

        let u32_table_entry = (PopCount, (lhs as u64).into(), BFIELD_ZERO);
        let co_processor_trace = Some(U32TableEntries(vec![u32_table_entry]));

        self.instruction_pointer += 1;
        Ok(co_processor_trace)
    }

    fn instruction_xx_add(&mut self) -> Result<Option<CoProcessorCall>> {
        let lhs: XFieldElement = self.op_stack.pop_extension_field_element()?;
        let rhs: XFieldElement = self.op_stack.peek_at_top_extension_field_element();
        self.op_stack.push_extension_field_element(lhs + rhs);
        self.instruction_pointer += 1;
        Ok(None)
    }

    fn instruction_xx_mul(&mut self) -> Result<Option<CoProcessorCall>> {
        let lhs: XFieldElement = self.op_stack.pop_extension_field_element()?;
        let rhs: XFieldElement = self.op_stack.peek_at_top_extension_field_element();
        self.op_stack.push_extension_field_element(lhs * rhs);
        self.instruction_pointer += 1;
        Ok(None)
    }

    fn instruction_x_invert(&mut self) -> Result<Option<CoProcessorCall>> {
        let elem: XFieldElement = self.op_stack.pop_extension_field_element()?;
        if elem.is_zero() {
            bail!(InverseOfZero);
        }
        self.op_stack.push_extension_field_element(elem.inverse());
        self.instruction_pointer += 1;
        Ok(None)
    }

    fn instruction_xb_mul(&mut self) -> Result<Option<CoProcessorCall>> {
        let lhs: BFieldElement = self.op_stack.pop()?;
        let rhs: XFieldElement = self.op_stack.pop_extension_field_element()?;
        self.op_stack.push_extension_field_element(lhs.lift() * rhs);
        self.instruction_pointer += 1;
        Ok(None)
    }

    fn instruction_write_io(&mut self) -> Result<Option<CoProcessorCall>> {
        let elem_to_write = self.op_stack.pop()?;
        self.public_output.push(elem_to_write);
        self.instruction_pointer += 1;
        Ok(None)
    }

    fn instruction_read_io(&mut self) -> Result<Option<CoProcessorCall>> {
        let in_elem = self.public_input_stream.pop_front().ok_or(anyhow!(
            "Instruction `read_io`: public input buffer is empty."
        ))?;
        self.op_stack.push(in_elem);
        self.instruction_pointer += 1;
        Ok(None)
    }

    pub fn to_processor_row(&self) -> Array1<BFieldElement> {
        use crate::instruction::InstructionBit;
        use ProcessorBaseTableColumn::*;
        let mut processor_row = Array1::zeros(processor_table::BASE_WIDTH);

        let current_instruction = self.current_instruction().unwrap_or(Nop);
        let helper_variables = self.derive_helper_variables();
        let ram_pointer = self.ramp.into();

        processor_row[CLK.base_table_index()] = (self.cycle_count as u64).into();
        processor_row[PreviousInstruction.base_table_index()] = self.previous_instruction;
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
        processor_row[IB7.base_table_index()] = current_instruction.ib(InstructionBit::IB7);
        processor_row[JSP.base_table_index()] = self.jump_stack_pointer();
        processor_row[JSO.base_table_index()] = self.jump_stack_origin();
        processor_row[JSD.base_table_index()] = self.jump_stack_destination();
        processor_row[ST0.base_table_index()] = self.op_stack.peek_at(OpStackElement::ST0);
        processor_row[ST1.base_table_index()] = self.op_stack.peek_at(OpStackElement::ST1);
        processor_row[ST2.base_table_index()] = self.op_stack.peek_at(OpStackElement::ST2);
        processor_row[ST3.base_table_index()] = self.op_stack.peek_at(OpStackElement::ST3);
        processor_row[ST4.base_table_index()] = self.op_stack.peek_at(OpStackElement::ST4);
        processor_row[ST5.base_table_index()] = self.op_stack.peek_at(OpStackElement::ST5);
        processor_row[ST6.base_table_index()] = self.op_stack.peek_at(OpStackElement::ST6);
        processor_row[ST7.base_table_index()] = self.op_stack.peek_at(OpStackElement::ST7);
        processor_row[ST8.base_table_index()] = self.op_stack.peek_at(OpStackElement::ST8);
        processor_row[ST9.base_table_index()] = self.op_stack.peek_at(OpStackElement::ST9);
        processor_row[ST10.base_table_index()] = self.op_stack.peek_at(OpStackElement::ST10);
        processor_row[ST11.base_table_index()] = self.op_stack.peek_at(OpStackElement::ST11);
        processor_row[ST12.base_table_index()] = self.op_stack.peek_at(OpStackElement::ST12);
        processor_row[ST13.base_table_index()] = self.op_stack.peek_at(OpStackElement::ST13);
        processor_row[ST14.base_table_index()] = self.op_stack.peek_at(OpStackElement::ST14);
        processor_row[ST15.base_table_index()] = self.op_stack.peek_at(OpStackElement::ST15);
        processor_row[OSP.base_table_index()] = self.op_stack.op_stack_pointer();
        processor_row[OSV.base_table_index()] = self.op_stack.op_stack_value();
        processor_row[HV0.base_table_index()] = helper_variables[0];
        processor_row[HV1.base_table_index()] = helper_variables[1];
        processor_row[HV2.base_table_index()] = helper_variables[2];
        processor_row[HV3.base_table_index()] = helper_variables[3];
        processor_row[HV4.base_table_index()] = helper_variables[4];
        processor_row[HV5.base_table_index()] = helper_variables[5];
        processor_row[HV6.base_table_index()] = helper_variables[6];
        processor_row[RAMP.base_table_index()] = ram_pointer;
        processor_row[RAMV.base_table_index()] = self.memory_get(&ram_pointer);

        processor_row
    }

    fn eq(lhs: BFieldElement, rhs: BFieldElement) -> BFieldElement {
        match lhs == rhs {
            true => BFieldElement::one(),
            false => BFieldElement::zero(),
        }
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
            return BFieldElement::zero();
        };
        if let Some(argument) = current_instruction.arg() {
            return argument;
        }
        match self.next_instruction() {
            Ok(next_instruction) => next_instruction.opcode_b(),
            Err(_) => BFieldElement::one(),
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
        maybe_current_instruction.ok_or(anyhow!(InstructionPointerOverflow(
            self.instruction_pointer
        )))
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
        maybe_next_instruction.ok_or(anyhow!(InstructionPointerOverflow(
            next_instruction_pointer
        )))
    }

    fn jump_stack_pop(&mut self) -> Result<(BFieldElement, BFieldElement)> {
        let maybe_jump_stack_element = self.jump_stack.pop();
        maybe_jump_stack_element.ok_or(anyhow!(JumpStackIsEmpty))
    }

    fn jump_stack_peek(&mut self) -> Result<(BFieldElement, BFieldElement)> {
        let maybe_jump_stack_element = self.jump_stack.last().copied();
        maybe_jump_stack_element.ok_or(anyhow!(JumpStackIsEmpty))
    }

    fn memory_get(&self, memory_address: &BFieldElement) -> BFieldElement {
        let maybe_memory_value = self.ram.get(memory_address).copied();
        maybe_memory_value.unwrap_or_else(BFieldElement::zero)
    }

    fn assert_vector(&self) -> bool {
        for index in 0..DIGEST_LENGTH {
            // Safe as long as 2 * DIGEST_LEN <= OpStackElement::COUNT
            let lhs = index.try_into().unwrap();
            let rhs = (index + DIGEST_LENGTH).try_into().unwrap();
            if self.op_stack.peek_at(lhs) != self.op_stack.peek_at(rhs) {
                return false;
            }
        }
        true
    }

    fn secret_input_pop_multiple<const N: usize>(&mut self) -> Result<[BFieldElement; N]> {
        let mut popped_elements = [BFieldElement::zero(); N];
        for element in popped_elements.iter_mut() {
            *element = self
                .secret_input_stream
                .pop_front()
                .ok_or(anyhow!("Secret input buffer is empty."))?;
        }
        Ok(popped_elements)
    }

    /// If the given node index indicates a left node, puts the known digest to the left.
    /// Otherwise, puts the known digest to the right.
    /// Returns the left and right digests in that order.
    fn put_known_digest_on_correct_side(
        node_index: u32,
        known_digest: [BFieldElement; 5],
        sibling_digest: [BFieldElement; 5],
    ) -> ([BFieldElement; 5], [BFieldElement; 5]) {
        let is_left_node = node_index % 2 == 0;
        if is_left_node {
            (known_digest, sibling_digest)
        } else {
            (sibling_digest, known_digest)
        }
    }
}

impl<'pgm> Display for VMState<'pgm> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.current_instruction() {
            Ok(_) => {
                let row = self.to_processor_row();
                write!(f, "{}", ProcessorTraceRow { row: row.view() })
            }
            Err(_) => write!(f, "END-OF-FILE"),
        }
    }
}

#[cfg(test)]
pub mod triton_vm_tests {
    use std::ops::BitAnd;
    use std::ops::BitXor;

    use itertools::Itertools;
    use ndarray::Array1;
    use ndarray::ArrayView1;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;
    use rand::Rng;
    use rand::RngCore;
    use twenty_first::shared_math::b_field_element::BFIELD_ZERO;
    use twenty_first::shared_math::other::random_elements;
    use twenty_first::shared_math::other::random_elements_array;
    use twenty_first::shared_math::tip5::Tip5;
    use twenty_first::shared_math::traits::FiniteField;
    use twenty_first::shared_math::traits::ModPowU32;
    use twenty_first::util_types::algebraic_hasher::SpongeHasher;
    use twenty_first::util_types::merkle_tree::MerkleTree;
    use twenty_first::util_types::merkle_tree_maker::MerkleTreeMaker;

    use crate::error::InstructionError;
    use crate::example_programs::*;
    use crate::shared_tests::prove_with_low_security_level;
    use crate::shared_tests::ProgramAndInput;
    use crate::stark::MTMaker;
    use crate::table::processor_table::ProcessorTraceRow;
    use crate::triton_program;

    use super::*;

    fn pretty_print_array_view<FF: FiniteField>(array: ArrayView1<FF>) -> String {
        array
            .iter()
            .map(|ff| format!("{ff}"))
            .collect_vec()
            .join(", ")
    }

    #[test]
    fn initialise_table_test() {
        let program = GREATEST_COMMON_DIVISOR.clone();
        let stdin = vec![42, 56].into();
        let (aet, stdout) = program.trace_execution(stdin, [].into()).unwrap();

        println!(
            "VM output: [{}]",
            pretty_print_array_view(Array1::from(stdout).view())
        );
        for row in aet.processor_trace.rows() {
            println!("{}", ProcessorTraceRow { row });
        }
    }

    #[test]
    fn run_tvm_gcd_test() {
        let program = GREATEST_COMMON_DIVISOR.clone();
        let stdin = vec![42, 56].into();
        let stdout = program.run(stdin, [].into()).unwrap();

        let stdout = Array1::from(stdout);
        println!("VM output: [{}]", pretty_print_array_view(stdout.view()));

        assert_eq!(BFieldElement::new(14), stdout[0]);
    }

    pub(crate) fn test_hash_nop_nop_lt() -> ProgramAndInput {
        ProgramAndInput::without_input(
            triton_program!(hash nop hash nop nop hash push 3 push 2 lt assert halt),
        )
    }

    pub(crate) fn test_program_for_halt() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(halt))
    }

    pub(crate) fn test_program_for_push_pop_dup_swap_nop() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(
            push 1 push 2 pop assert
            push 1 dup  0 assert assert
            push 1 push 2 swap 1 assert pop
            nop nop nop halt
        ))
    }

    pub(crate) fn test_program_for_divine() -> ProgramAndInput {
        ProgramAndInput {
            program: triton_program!(divine assert halt),
            public_input: vec![],
            secret_input: vec![1],
        }
    }

    pub(crate) fn test_program_for_skiz() -> ProgramAndInput {
        ProgramAndInput::without_input(
            triton_program!(push 1 skiz push 0 skiz assert push 1 skiz halt),
        )
    }

    pub(crate) fn test_program_for_call_recurse_return() -> ProgramAndInput {
        ProgramAndInput::without_input(
            triton_program!(push 2 call label halt label: push -1 add dup 0 skiz recurse return),
        )
    }

    pub(crate) fn test_program_for_write_mem_read_mem() -> ProgramAndInput {
        ProgramAndInput::without_input(
            triton_program!(push 2 push 1 write_mem push 0 pop read_mem assert halt),
        )
    }

    pub(crate) fn test_program_for_hash() -> ProgramAndInput {
        let program = triton_program!(
            push 0 // filler to keep the OpStack large enough throughout the program
            push 0 push 0 push 1 push 2 push 3
            hash
            pop pop pop pop pop
            read_io eq assert halt
        );
        let hash_input = [3, 2, 1, 0, 0, 0, 0, 0, 0, 0].map(BFieldElement::new);
        let digest = Tip5::hash_10(&hash_input).map(|e| e.value());
        ProgramAndInput {
            program,
            public_input: vec![digest.to_vec()[0]],
            secret_input: vec![],
        }
    }

    pub(crate) fn test_program_for_divine_sibling_noswitch() -> ProgramAndInput {
        let program = triton_program!(
            push 3
            push 4 push 2 push 2 push 2 push 1
            push 5679457 push 1337 push 345887 push -234578456 push 23657565
            divine_sibling
            push 1 add assert assert assert assert assert
            assert
            push -1 add assert
            push -1 add assert
            push -1 add assert
            push -3 add assert
            assert halt
        );
        ProgramAndInput {
            program,
            public_input: vec![],
            secret_input: vec![1, 1, 1, 1, 0],
        }
    }

    pub(crate) fn test_program_for_divine_sibling_switch() -> ProgramAndInput {
        let program = triton_program!(
            push 2
            push 4 push 2 push 2 push 2 push 1
            push 5679457 push 1337 push 345887 push -234578456 push 23657565
            divine_sibling
            assert
            push -1 add assert
            push -1 add assert
            push -1 add assert
            push -3 add assert
            push 1 add assert assert assert assert assert
            assert halt
        );
        ProgramAndInput {
            program,
            public_input: vec![],
            secret_input: vec![1, 1, 1, 1, 0],
        }
    }

    pub(crate) fn test_program_for_assert_vector() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(
            push 1 push 2 push 3 push 4 push 5
            push 1 push 2 push 3 push 4 push 5
            assert_vector halt
        ))
    }

    pub(crate) fn test_program_for_sponge_instructions() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(
            absorb_init push 3 push 2 push 1 absorb absorb squeeze halt
        ))
    }

    pub(crate) fn test_program_for_sponge_instructions_2() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(
            hash absorb_init push 3 push 2 push 1 absorb absorb squeeze halt
        ))
    }

    pub(crate) fn test_program_for_many_sponge_instructions() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(
            absorb_init squeeze absorb absorb absorb squeeze squeeze squeeze absorb
            absorb_init absorb_init absorb_init absorb absorb_init squeeze squeeze
            absorb_init squeeze hash absorb hash squeeze hash absorb hash squeeze
            absorb_init absorb absorb absorb absorb absorb absorb absorb halt
        ))
    }

    pub(crate) fn property_based_test_program_for_assert_vector() -> ProgramAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.gen_range(0..BFieldElement::P);
        let st1 = rng.gen_range(0..BFieldElement::P);
        let st2 = rng.gen_range(0..BFieldElement::P);
        let st3 = rng.gen_range(0..BFieldElement::P);
        let st4 = rng.gen_range(0..BFieldElement::P);

        let program = triton_program!(
            push {st4} push {st3} push {st2} push {st1} push {st0}
            read_io read_io read_io read_io read_io assert_vector halt
        );

        ProgramAndInput {
            program,
            public_input: vec![st4, st3, st2, st1, st0],
            secret_input: vec![],
        }
    }

    pub(crate) fn property_based_test_program_for_sponge_instructions() -> ProgramAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.gen_range(0..BFieldElement::P);
        let st1 = rng.gen_range(0..BFieldElement::P);
        let st2 = rng.gen_range(0..BFieldElement::P);
        let st3 = rng.gen_range(0..BFieldElement::P);
        let st4 = rng.gen_range(0..BFieldElement::P);
        let st5 = rng.gen_range(0..BFieldElement::P);
        let st6 = rng.gen_range(0..BFieldElement::P);
        let st7 = rng.gen_range(0..BFieldElement::P);
        let st8 = rng.gen_range(0..BFieldElement::P);
        let st9 = rng.gen_range(0..BFieldElement::P);

        let sponge_input =
            [st0, st1, st2, st3, st4, st5, st6, st7, st8, st9].map(BFieldElement::new);

        let mut sponge = Tip5::init();
        Tip5::absorb(&mut sponge, &sponge_input);
        let sponge_output = Tip5::squeeze(&mut sponge);
        Tip5::absorb(&mut sponge, &sponge_output);
        Tip5::absorb(&mut sponge, &sponge_output);
        let sponge_output = Tip5::squeeze(&mut sponge);
        Tip5::absorb(&mut sponge, &sponge_output);
        Tip5::squeeze(&mut sponge);
        let sponge_output = Tip5::squeeze(&mut sponge);

        let program = triton_program!(
            push {st9} push {st8} push {st7} push {st6} push {st5}
            push {st4} push {st3} push {st2} push {st1} push {st0}
            absorb_init hash squeeze absorb absorb hash squeeze absorb squeeze squeeze
            read_io eq assert // st0
            read_io eq assert // st1
            read_io eq assert // st2
            read_io eq assert // st3
            read_io eq assert // st4
            read_io eq assert // st5
            read_io eq assert // st6
            read_io eq assert // st7
            read_io eq assert // st8
            read_io eq assert // st9
            halt
        );

        ProgramAndInput {
            program,
            public_input: sponge_output.map(|e| e.value()).to_vec(),
            secret_input: vec![],
        }
    }

    pub(crate) fn test_program_for_add_mul_invert() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(
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

        let program = triton_program!(push {st0} split read_io eq assert read_io eq assert halt);
        ProgramAndInput {
            program,
            public_input: vec![lo, hi],
            secret_input: vec![],
        }
    }

    pub(crate) fn test_program_for_eq() -> ProgramAndInput {
        ProgramAndInput {
            program: triton_program!(read_io divine eq assert halt),
            public_input: vec![42],
            secret_input: vec![42],
        }
    }

    pub(crate) fn property_based_test_program_for_eq() -> ProgramAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u64() % BFieldElement::P;

        let program =
            triton_program!(push {st0} dup 0 read_io eq assert dup 0 divine eq assert halt);
        ProgramAndInput {
            program,
            public_input: vec![st0],
            secret_input: vec![st0],
        }
    }

    pub(crate) fn test_program_for_lsb() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(
            push 3 call lsb assert assert halt
            lsb:
                push 2 swap 1 div return
        ))
    }

    pub(crate) fn property_based_test_program_for_lsb() -> ProgramAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u32();
        let lsb = st0 % 2;
        let st0_shift_right = st0 >> 1;

        let program = triton_program!(
            push {st0} call lsb read_io eq assert read_io eq assert halt
            lsb:
                push 2 swap 1 div return
        );
        ProgramAndInput {
            program,
            public_input: vec![lsb.into(), st0_shift_right.into()],
            secret_input: vec![],
        }
    }

    pub(crate) fn test_program_0_lt_0() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(push 0 push 0 lt halt))
    }

    pub(crate) fn test_program_for_lt() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(
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
            push {st1_0} push {st0_0} lt read_io eq assert
            push {st1_1} push {st0_1} lt read_io eq assert halt
        );
        ProgramAndInput {
            program,
            public_input: vec![result_0, result_1],
            secret_input: vec![],
        }
    }

    pub(crate) fn test_program_for_and() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(
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
            push {st1_0} push {st0_0} and read_io eq assert
            push {st1_1} push {st0_1} and read_io eq assert halt
        );
        ProgramAndInput {
            program,
            public_input: vec![result_0.into(), result_1.into()],
            secret_input: vec![],
        }
    }

    pub(crate) fn test_program_for_xor() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(
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
            push {st1_0} push {st0_0} xor read_io eq assert
            push {st1_1} push {st0_1} xor read_io eq assert halt
        );
        ProgramAndInput {
            program,
            public_input: vec![result_0.into(), result_1.into()],
            secret_input: vec![],
        }
    }

    pub(crate) fn test_program_for_log2floor() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(
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
            push {st0_0} log_2_floor read_io eq assert
            push {st0_1} log_2_floor read_io eq assert halt
        );
        ProgramAndInput {
            program,
            public_input: vec![l2f_0.into(), l2f_1.into()],
            secret_input: vec![],
        }
    }

    pub(crate) fn test_program_for_pow() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(
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
        let result_0 = base_0.mod_pow_u32(exp_0).value();

        let base_1: BFieldElement = rng.gen();
        let exp_1 = rng.next_u32();
        let result_1 = base_1.mod_pow_u32(exp_1).value();

        let program = triton_program!(
            push {exp_0} push {base_0} pow read_io eq assert
            push {exp_1} push {base_1} pow read_io eq assert halt
        );
        ProgramAndInput {
            program,
            public_input: vec![result_0, result_1],
            secret_input: vec![],
        }
    }

    pub(crate) fn test_program_for_div() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(push 2 push 3 div assert assert halt))
    }

    pub(crate) fn property_based_test_program_for_div() -> ProgramAndInput {
        let mut rng = ThreadRng::default();

        let denominator = rng.next_u32();
        let numerator = rng.next_u32();
        let quotient = numerator / denominator;
        let remainder = numerator % denominator;

        let program = triton_program!(
            push {denominator} push {numerator} div read_io eq assert read_io eq assert halt
        );
        ProgramAndInput {
            program,
            public_input: vec![remainder.into(), quotient.into()],
            secret_input: vec![],
        }
    }

    pub(crate) fn test_program_for_starting_with_pop_count() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(pop_count dup 0 push 0 eq assert halt))
    }

    pub(crate) fn test_program_for_pop_count() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(push 10 pop_count push 2 eq assert halt))
    }

    pub(crate) fn property_based_test_program_for_pop_count() -> ProgramAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u32();
        let pop_count = st0.count_ones();
        let program = triton_program!(push {st0} pop_count read_io eq assert halt);
        ProgramAndInput {
            program,
            public_input: vec![pop_count.into()],
            secret_input: vec![],
        }
    }

    fn property_based_test_program_for_is_u32() -> ProgramAndInput {
        let mut rng = ThreadRng::default();
        let st0_u32 = rng.next_u32();
        let st0_not_u32 = ((rng.next_u32() as u64) << 32) + (rng.next_u32() as u64);
        let program = triton_program!(
            push {st0_u32} call is_u32 assert
            push {st0_not_u32} call is_u32 push 0 eq assert halt
            is_u32:
                 split pop push 0 eq return
        );
        ProgramAndInput::without_input(program)
    }

    fn property_based_test_program_for_random_ram_access() -> ProgramAndInput {
        let mut rng = ThreadRng::default();
        let num_memory_accesses = rng.gen_range(10..50);
        let memory_addresses: Vec<BFieldElement> = random_elements(num_memory_accesses);
        let mut memory_values: Vec<BFieldElement> = random_elements(num_memory_accesses);
        let mut source_code = String::new();

        // Read some memory before first write to ensure that the memory is initialized with 0s.
        // Not all addresses are read to have different access patterns:
        // - Some addresses are read before written to.
        // - Other addresses are written to before read.
        for memory_address in memory_addresses.iter().take(num_memory_accesses / 4) {
            source_code.push_str(&format!(
                "push {memory_address} read_mem push 0 eq assert pop "
            ));
        }

        // Write everything to RAM.
        for (memory_address, memory_value) in memory_addresses.iter().zip_eq(memory_values.iter()) {
            source_code.push_str(&format!(
                "push {memory_address} push {memory_value} write_mem pop "
            ));
        }

        // Read back in random order and check that the values did not change.
        let mut reading_permutation = (0..num_memory_accesses).collect_vec();
        for i in 0..num_memory_accesses {
            let j = rng.gen_range(0..num_memory_accesses);
            reading_permutation.swap(i, j);
        }
        for idx in reading_permutation {
            let memory_address = memory_addresses[idx];
            let memory_value = memory_values[idx];
            source_code.push_str(&format!(
                "push {memory_address} read_mem push {memory_value} eq assert pop "
            ));
        }

        // Overwrite half the values with new ones.
        let mut writing_permutation = (0..num_memory_accesses).collect_vec();
        for i in 0..num_memory_accesses {
            let j = rng.gen_range(0..num_memory_accesses);
            writing_permutation.swap(i, j);
        }
        for idx in 0..num_memory_accesses / 2 {
            let memory_address = memory_addresses[writing_permutation[idx]];
            let new_memory_value = rng.gen();
            memory_values[writing_permutation[idx]] = new_memory_value;
            source_code.push_str(&format!(
                "push {memory_address} push {new_memory_value} write_mem pop "
            ));
        }

        // Read back all, i.e., unchanged and overwritten values in (different from before) random
        // order and check that the values did not change.
        let mut reading_permutation = (0..num_memory_accesses).collect_vec();
        for i in 0..num_memory_accesses {
            let j = rng.gen_range(0..num_memory_accesses);
            reading_permutation.swap(i, j);
        }
        for idx in reading_permutation {
            let memory_address = memory_addresses[idx];
            let memory_value = memory_values[idx];
            source_code.push_str(&format!(
                "push {memory_address} read_mem push {memory_value} eq assert pop "
            ));
        }

        source_code.push_str("halt");
        let program = triton_program!({ source_code });
        ProgramAndInput::without_input(program)
    }

    // Sanity check for the relatively complex property-based test for random RAM access.
    #[test]
    fn run_dont_prove_property_based_test_for_random_ram_access() {
        let source_code_and_input = property_based_test_program_for_random_ram_access();
        source_code_and_input.run().unwrap();
    }

    #[test]
    fn negative_property_is_u32_test() {
        let mut rng = ThreadRng::default();
        let st0 = (rng.next_u32() as u64) << 32;

        let program = triton_program!(
            push {st0} call is_u32 assert halt
            is_u32:
                split pop push 0 eq return
        );
        let program_and_input = ProgramAndInput::without_input(program);
        let err = program_and_input.run().err();
        let err = err.unwrap();
        let err = err.downcast::<InstructionError>().unwrap();
        let AssertionFailed(_, _, _) = err else {
            panic!("Non-u32 must not pass u32-ness test.");
        };
    }

    pub(crate) fn test_program_for_split() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(
            push -2 split push 4294967295 eq assert push 4294967294 eq assert
            push -1 split push 0 eq assert push 4294967295 eq assert
            push  0 split push 0 eq assert push 0 eq assert
            push  1 split push 1 eq assert push 0 eq assert
            push  2 split push 2 eq assert push 0 eq assert
            push 4294967297 split assert assert
            halt
        ))
    }

    pub(crate) fn test_program_for_xxadd() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(
            push 5 push 6 push 7 push 8 push 9 push 10 xxadd halt
        ))
    }

    pub(crate) fn test_program_for_xxmul() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(
            push 5 push 6 push 7 push 8 push 9 push 10 xxmul halt
        ))
    }

    pub(crate) fn test_program_for_xinvert() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(
            push 5 push 6 push 7 xinvert halt
        ))
    }

    pub(crate) fn test_program_for_xbmul() -> ProgramAndInput {
        ProgramAndInput::without_input(triton_program!(
            push 5 push 6 push 7 push 8 xbmul halt
        ))
    }

    pub(crate) fn test_program_for_read_io_write_io() -> ProgramAndInput {
        let program = triton_program!(
            read_io assert read_io read_io dup 1 dup 1 add write_io mul write_io halt
        );
        ProgramAndInput {
            program,
            public_input: vec![1, 3, 14],
            secret_input: vec![],
        }
    }

    pub(crate) fn small_tasm_test_programs() -> Vec<ProgramAndInput> {
        vec![
            test_program_for_halt(),
            test_hash_nop_nop_lt(),
            test_program_for_push_pop_dup_swap_nop(),
            test_program_for_divine(),
            test_program_for_skiz(),
            test_program_for_call_recurse_return(),
            test_program_for_write_mem_read_mem(),
            test_program_for_hash(),
            test_program_for_divine_sibling_noswitch(),
            test_program_for_divine_sibling_switch(),
            test_program_for_assert_vector(),
            test_program_for_sponge_instructions(),
            test_program_for_sponge_instructions_2(),
            test_program_for_many_sponge_instructions(),
            test_program_for_add_mul_invert(),
            test_program_for_eq(),
            test_program_for_lsb(),
            test_program_for_split(),
            test_program_0_lt_0(),
            test_program_for_lt(),
            test_program_for_and(),
            test_program_for_xor(),
            test_program_for_log2floor(),
            test_program_for_pow(),
            test_program_for_div(),
            test_program_for_starting_with_pop_count(),
            test_program_for_pop_count(),
            test_program_for_xxadd(),
            test_program_for_xxmul(),
            test_program_for_xinvert(),
            test_program_for_xbmul(),
            test_program_for_read_io_write_io(),
        ]
    }

    pub(crate) fn property_based_test_programs() -> Vec<ProgramAndInput> {
        vec![
            property_based_test_program_for_assert_vector(),
            property_based_test_program_for_sponge_instructions(),
            property_based_test_program_for_split(),
            property_based_test_program_for_eq(),
            property_based_test_program_for_lsb(),
            property_based_test_program_for_lt(),
            property_based_test_program_for_and(),
            property_based_test_program_for_xor(),
            property_based_test_program_for_log2floor(),
            property_based_test_program_for_pow(),
            property_based_test_program_for_div(),
            property_based_test_program_for_pop_count(),
            property_based_test_program_for_is_u32(),
            property_based_test_program_for_random_ram_access(),
        ]
    }

    #[test]
    fn xxadd_test() {
        let program = triton_program!(
            read_io read_io read_io
            read_io read_io read_io
            xxadd
            swap 2
            write_io write_io write_io
            halt
        );
        let program = ProgramAndInput {
            program,
            public_input: vec![2, 3, 5, 7, 11, 13],
            secret_input: vec![],
        };

        let actual_stdout = program.run().unwrap();
        let expected_stdout = [9, 14, 18].map(BFieldElement::new).to_vec();

        assert_eq!(expected_stdout, actual_stdout);
    }

    #[test]
    fn xxmul_test() {
        let program = triton_program!(
            read_io read_io read_io
            read_io read_io read_io
            xxmul
            swap 2
            write_io write_io write_io
            halt
        );
        let program = ProgramAndInput {
            program,
            public_input: vec![2, 3, 5, 7, 11, 13],
            secret_input: vec![],
        };

        let actual_stdout = program.run().unwrap();
        let expected_stdout = [108, 123, 22].map(BFieldElement::new).to_vec();

        assert_eq!(expected_stdout, actual_stdout);
    }

    #[test]
    fn xinv_test() {
        let program = triton_program!(
            read_io read_io read_io
            dup 2 dup 2 dup 2
            dup 2 dup 2 dup 2
            xinvert xxmul
            swap 2
            write_io write_io write_io
            xinvert
            swap 2
            write_io write_io write_io
            halt
        );
        let program = ProgramAndInput {
            program,
            public_input: vec![2, 3, 5],
            secret_input: vec![],
        };

        let actual_stdout = program.run().unwrap();
        let expected_stdout = [
            0,
            0,
            1,
            16360893149904808002,
            14209859389160351173,
            4432433203958274678,
        ]
        .map(BFieldElement::new)
        .to_vec();

        assert_eq!(expected_stdout, actual_stdout);
    }

    #[test]
    fn xbmul_test() {
        let program = triton_program!(
            read_io read_io read_io
            read_io
            xbmul
            swap 2
            write_io write_io write_io
            halt
        );
        let program = ProgramAndInput {
            program,
            public_input: vec![2, 3, 5, 7],
            secret_input: vec![],
        };

        let actual_stdout = program.run().unwrap();
        let expected_stdout = [14, 21, 35].map(BFieldElement::new).to_vec();

        assert_eq!(expected_stdout, actual_stdout);
    }

    #[test]
    fn pseudo_sub_test() {
        let program = triton_program!(
            push 7 push 19 call sub write_io halt
            sub:
                swap 1 push -1 mul add return
        );
        let actual_stdout = ProgramAndInput::without_input(program).run().unwrap();
        let expected_stdout = vec![BFieldElement::new(12)];

        assert_eq!(expected_stdout, actual_stdout);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    const fn op_stack_is_big_enough_test() {
        assert!(
            2 * DIGEST_LENGTH <= OpStackElement::COUNT,
            "The OpStack must be large enough to hold two digests."
        );
    }
    const _COMPILE_TIME_ASSERTION: () = op_stack_is_big_enough_test();

    #[test]
    fn run_tvm_hello_world_1_test() {
        let program = triton_program!(
            push  10 write_io
            push  33 write_io
            push 100 write_io
            push 108 write_io
            push 114 write_io
            push 111 write_io
            push  87 write_io
            push  32 write_io
            push  44 write_io
            push 111 write_io
            push 108 write_io
            push 108 write_io
            push 101 write_io
            push  72 write_io
            halt
        );
        let terminal_state = program
            .debug_terminal_state([].into(), [].into(), None, None)
            .unwrap();
        assert_eq!(BFIELD_ZERO, terminal_state.op_stack.peek_at(ST0));
    }

    #[test]
    fn run_tvm_halt_then_do_stuff_test() {
        let program = triton_program!(halt push 1 push 2 add invert write_io);
        let (aet, _) = program.trace_execution([].into(), [].into()).unwrap();

        let last_processor_row = aet.processor_trace.rows().into_iter().last().unwrap();
        let clk_count = last_processor_row[ProcessorBaseTableColumn::CLK.base_table_index()];
        assert_eq!(BFIELD_ZERO, clk_count);

        let last_instruction = last_processor_row[ProcessorBaseTableColumn::CI.base_table_index()];
        assert_eq!(Instruction::Halt.opcode_b(), last_instruction);

        println!("{last_processor_row}");
    }

    #[test]
    fn run_tvm_basic_ram_read_write_test() {
        let program = triton_program!(
            push  5 push  6 write_mem pop
            push 15 push 16 write_mem pop
            push  5         read_mem  pop pop
            push 15         read_mem  pop pop
            push  5 push  7 write_mem pop
            push 15         read_mem
            push  5         read_mem
            halt
        );

        let terminal_state = program
            .debug_terminal_state([].into(), [].into(), None, None)
            .unwrap();
        assert_eq!(BFieldElement::new(7), terminal_state.op_stack.peek_at(ST0));
        assert_eq!(BFieldElement::new(5), terminal_state.op_stack.peek_at(ST1));
        assert_eq!(BFieldElement::new(16), terminal_state.op_stack.peek_at(ST2));
        assert_eq!(BFieldElement::new(15), terminal_state.op_stack.peek_at(ST3));
    }

    #[test]
    fn run_tvm_edgy_ram_writes_test() {
        let program = triton_program!(
                        //       ┌ stack cannot shrink beyond this point
                        //       ↓
                        // _ 0 0 |
            push 0      // _ 0 0 | 0
            write_mem   // _ 0 0 |
            push 5      // _ 0 0 | 5
            swap 1      // _ 0 5 | 0
            push 3      // _ 0 5 | 0 3
            swap 1      // _ 0 5 | 3 0
            pop         // _ 0 5 | 3
            write_mem   // _ 0 5 |
            read_mem    // _ 0 5 | 3
            swap 2      // _ 3 5 | 0
            pop         // _ 3 5 |
            read_mem    // _ 3 5 | 3
            halt
        );

        let terminal_state = program
            .debug_terminal_state([].into(), [].into(), None, None)
            .unwrap();
        assert_eq!(BFieldElement::new(3), terminal_state.op_stack.peek_at(ST0));
        assert_eq!(BFieldElement::new(5), terminal_state.op_stack.peek_at(ST1));
        assert_eq!(BFieldElement::new(3), terminal_state.op_stack.peek_at(ST2));
    }

    #[test]
    fn run_tvm_sample_weights_test() {
        // sample weights for the recursive verifier
        // - input: seed, num_weights
        // - output: num_weights-many random weights
        let program = triton_program!(
            push 17 push 13 push 11        // get seed - should be an argument
            read_io                        // number of weights - should be argument
            sample_weights:                // proper program starts here
            call sample_weights_loop       // setup done, start sampling loop
            pop pop                        // clean up stack: RAM value & pointer
            pop pop pop pop                // clean up stack: seed & countdown
            halt                           // done - should be return

            sample_weights_loop:           // subroutine: loop until all weights are sampled
              dup 0 push 0 eq skiz return  // no weights left
              push -1 add                  // decrease number of weights to still sample
              push 0 push 0 push 0 push 0  // prepare for hashing
              push 0 push 0 push 0 push 0  // prepare for hashing
              dup 11 dup 11 dup 11 dup 11  // prepare for hashing
              hash                         // hash seed & countdown
              swap 13 swap 10 pop          // re-organize stack
              swap 13 swap 10 pop          // re-organize stack
              swap 13 swap 10 swap 7       // re-organize stack
              pop pop pop pop pop pop pop  // remove unnecessary remnants of digest
              recurse                      // repeat
        );
        let public_input = vec![11].into();
        program.run(public_input, [].into()).unwrap();
    }

    #[test]
    fn triton_assembly_merkle_tree_authentication_path_verification_test() {
        type H = Tip5;

        const TREE_HEIGHT: usize = 6;
        const NUM_LEAVES: usize = 1 << TREE_HEIGHT;
        let leaves: [_; NUM_LEAVES] = random_elements_array();
        let merkle_tree: MerkleTree<H> = MTMaker::from_digests(&leaves);
        let root = merkle_tree.get_root();

        let num_authentication_paths = 3;
        let selected_leaf_indices = (0..num_authentication_paths)
            .map(|_| thread_rng().gen_range(0..NUM_LEAVES))
            .collect_vec();

        let flat_authentication_path = |leaf_index| {
            let auth_path = merkle_tree.get_authentication_structure(&[leaf_index]);
            (0..TREE_HEIGHT)
                .flat_map(|i| auth_path[i].reversed().values())
                .collect_vec()
        };
        let secret_input = selected_leaf_indices
            .iter()
            .flat_map(|&leaf_index| flat_authentication_path(leaf_index))
            .collect_vec();

        let mut public_input = vec![(num_authentication_paths as u64).into()];
        public_input.append(&mut root.reversed().values().to_vec());
        for &leaf_index in &selected_leaf_indices {
            let node_index = (leaf_index + NUM_LEAVES) as u64;
            public_input.push(node_index.into());
            public_input.append(&mut leaves[leaf_index].reversed().values().to_vec());
        }

        let program = MERKLE_TREE_AUTHENTICATION_PATH_VERIFY.clone();
        program
            .run(public_input.into(), secret_input.into())
            .unwrap();
    }

    #[test]
    fn run_tvm_get_colinear_y_test() {
        // see also: get_colinear_y in src/shared_math/polynomial.rs
        let get_colinear_y_program = triton_program!(
            read_io                         // p2_x
            read_io read_io                 // p1_y p1_x
            read_io read_io                 // p0_y p0_x
            swap 3 push -1 mul dup 1 add    // dy = p0_y - p1_y
            dup 3 push -1 mul dup 5 add mul // dy·(p2_x - p0_x)
            dup 3 dup 3 push -1 mul add     // dx = p0_x - p1_x
            invert mul add                  // compute result
            swap 3 pop pop pop              // leave a clean stack
            write_io halt
        );

        println!("Successfully parsed the program.");
        let public_input = vec![7, 2, 1, 3, 4].into();
        let output = get_colinear_y_program.run(public_input, [].into()).unwrap();
        assert_eq!(BFieldElement::new(4), output[0]);
    }

    #[test]
    fn run_tvm_countdown_from_10_test() {
        let countdown_program = triton_program!(
            push 10
            call loop

            loop:
                dup 0
                write_io
                push -1
                add
                dup 0
                skiz
                  recurse
                write_io
                halt
        );

        let standard_out = countdown_program.run([].into(), [].into()).unwrap();
        let expected = (0..=10).map(BFieldElement::new).rev().collect_vec();
        assert_eq!(expected, standard_out);
    }

    #[test]
    fn run_tvm_fibonacci_tvm() {
        let program = FIBONACCI_SEQUENCE.clone();
        let standard_out = program.run(vec![7].into(), [].into()).unwrap();
        assert_eq!(BFieldElement::new(21), standard_out[0]);
    }

    #[test]
    fn run_tvm_swap_test() {
        let program = triton_program!(push 1 push 2 swap 1 assert write_io halt);
        let standard_out = program.run([].into(), [].into()).unwrap();
        assert_eq!(BFieldElement::new(2), standard_out[0]);
    }

    #[test]
    fn read_mem_unitialized() {
        let program = triton_program!(read_mem halt);
        let (aet, _) = program.trace_execution([].into(), [].into()).unwrap();
        assert_eq!(2, aet.processor_trace.nrows());
    }

    #[test]
    fn read_non_deterministically_initialized_ram_at_address_0() {
        let program = triton_program!(read_mem write_io halt);

        let mut initial_ram = HashMap::new();
        initial_ram.insert(0_u64.into(), 42_u64.into());

        let public_input = PublicInput::new(vec![]);
        let secret_input = SecretInput::new(vec![]).with_ram(initial_ram);

        let public_output = program
            .run(public_input.clone(), secret_input.clone())
            .unwrap();
        assert_eq!(42, public_output[0].value());

        prove_with_low_security_level(&program, public_input, secret_input, &mut None);
    }

    #[test]
    fn read_non_deterministically_initialized_ram_at_random_address() {
        let random_address = thread_rng().gen_range(1..2_u64.pow(16));
        let program = triton_program!(
            read_mem write_io
            push {random_address} read_mem write_io
            halt
        );

        let mut initial_ram = HashMap::new();
        initial_ram.insert(random_address.into(), 1337_u64.into());

        let public_input = PublicInput::new(vec![]);
        let secret_input = SecretInput::new(vec![]).with_ram(initial_ram);

        let public_output = program
            .run(public_input.clone(), secret_input.clone())
            .unwrap();
        assert_eq!(0, public_output[0].value());
        assert_eq!(1337, public_output[1].value());

        prove_with_low_security_level(&program, public_input, secret_input, &mut None);
    }

    #[test]
    fn program_without_halt_test() {
        let program = triton_program!(nop);
        let err = program.trace_execution([].into(), [].into()).err();
        let Some(err) = err else {
            panic!("Program without halt must fail.");
        };
        let Ok(err) = err.downcast::<InstructionError>() else {
            panic!("Program without halt must fail with InstructionError.");
        };
        let InstructionPointerOverflow(_) = err else {
            panic!("Program without halt must fail with InstructionPointerOverflow.");
        };
    }

    #[test]
    fn verify_sudoku_test() {
        let program = VERIFY_SUDOKU.clone();
        let stdin = vec![
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
        ]
        .into();
        let secret_in = [].into();
        if let Err(e) = program.trace_execution(stdin, secret_in) {
            panic!("The VM encountered an error: {e}");
        }

        // rows and columns adhere to Sudoku rules, boxes do not
        let bad_stdin = vec![
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
        ]
        .into();
        let secret_in = [].into();
        let err = program.trace_execution(bad_stdin, secret_in).err();
        let Some(err) = err else {
            panic!("Sudoku verifier must fail on bad Sudoku.");
        };
        let Ok(err) = err.downcast::<InstructionError>() else {
            panic!("Sudoku verifier must fail with InstructionError on bad Sudoku.");
        };
        let AssertionFailed(ip, _, _) = err else {
            panic!("Sudoku verifier must fail with AssertionFailed on bad Sudoku.");
        };
        assert_eq!(
            15, ip,
            "Sudoku verifier must fail on line 15 on bad Sudoku."
        );
    }
}
