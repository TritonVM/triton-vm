use std::collections::HashMap;
use std::convert::TryInto;
use std::error::Error;
use std::fmt::Display;

use itertools::Itertools;
use num_traits::{One, Zero};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::rescue_prime_regular::{
    RescuePrimeRegular, DIGEST_LENGTH, NUM_ROUNDS, ROUND_CONSTANTS, STATE_SIZE,
};
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::error::vm_err;
use crate::instruction::DivinationHint;
use crate::ord_n::{Ord16, Ord7};
use crate::table::base_matrix::ProcessorMatrixRow;
use crate::table::hash_table::{NUM_ROUND_CONSTANTS, TOTAL_NUM_CONSTANTS};
use crate::table::table_column::{
    InstructionBaseTableColumn, JumpStackBaseTableColumn, OpStackBaseTableColumn,
    ProcessorBaseTableColumn, RamBaseTableColumn,
};

use super::error::{vm_fail, InstructionError::*};
use super::instruction::{AnInstruction::*, Instruction};
use super::op_stack::OpStack;
use super::ord_n::{Ord16::*, Ord7::*};
use super::stdio::InputStream;
use super::table::{hash_table, instruction_table, jump_stack_table, op_stack_table};
use super::table::{processor_table, ram_table};
use super::vm::Program;

/// The number of state registers for hashing-specific instructions.
pub const STATE_REGISTER_COUNT: usize = 16;

/// The number of helper variable registers
pub const HV_REGISTER_COUNT: usize = 4;

#[derive(Debug, Default, Clone)]
pub struct VMState<'pgm> {
    ///
    /// Triton VM's four kinds of memory:
    ///
    /// 1. **Program memory**, from which the VM reads instructions
    program: &'pgm [Instruction],

    /// 2. **Random-access memory**, to which the VM can read and write field elements
    ram: HashMap<BFieldElement, BFieldElement>,

    /// 3. **Op-stack memory**, which stores the part of the operational stack
    ///    that is not represented explicitly by the operational stack registers
    ///
    ///    *(An implementation detail: We keep the entire stack in one `Vec<>`.)*
    op_stack: OpStack,

    /// 4. Jump-stack memory, which stores the entire jump stack
    jump_stack: Vec<(BFieldElement, BFieldElement)>,

    ///
    /// Registers
    ///
    /// Number of cycles the program has been running for
    pub cycle_count: u32,

    /// Current instruction's address in program memory
    pub instruction_pointer: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum VMOutput {
    /// Trace output from `write_io`
    WriteOutputSymbol(BFieldElement),

    /// Trace of state registers for hash coprocessor table
    ///
    /// One row per round in the XLIX permutation
    XlixTrace(Vec<[BFieldElement; hash_table::BASE_WIDTH]>),
}

#[allow(clippy::needless_range_loop)]
impl<'pgm> VMState<'pgm> {
    /// Create initial `VMState` for a given `program`
    ///
    /// Since `program` is read-only across individual states, and multiple
    /// inner helper functions refer to it, a read-only reference is kept in
    /// the struct.
    pub fn new(program: &'pgm Program) -> Self {
        let program = &program.instructions;
        Self {
            program,
            ..VMState::default()
        }
    }

    /// Determine if this is a final state.
    pub fn is_complete(&self) -> bool {
        match self.current_instruction() {
            Ok(Instruction::Halt) => true,
            _ => self.program.len() <= self.instruction_pointer,
        }
    }

    /// Given a state, compute `(next_state, vm_output)`.
    pub fn step<In>(
        &self,
        stdin: &mut In,
        secret_in: &mut In,
    ) -> Result<(VMState<'pgm>, Option<VMOutput>), Box<dyn Error>>
    where
        In: InputStream,
    {
        let mut next_state = self.clone();
        next_state
            .step_mut(stdin, secret_in)
            .map(|vm_output| (next_state, vm_output))
    }

    pub fn derive_helper_variables(&self) -> [BFieldElement; HV_REGISTER_COUNT] {
        let mut hvs = [BFieldElement::zero(); HV_REGISTER_COUNT];

        let current_instruction = self.current_instruction();
        if current_instruction.is_err() {
            return hvs;
        }
        let current_instruction = current_instruction.unwrap();

        // Helps preventing OpStack Underflow
        match current_instruction {
            Pop | Skiz | Assert | Add | Mul | Eq | XbMul | WriteIo => {
                if self.op_stack.osp() == BFieldElement::new(16) {
                    hvs[3] = BFieldElement::zero()
                } else {
                    hvs[3] = (self.op_stack.osp() - BFieldElement::new(16)).inverse()
                }
            }
            _ => (),
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
                let nia = self.nia().value();
                hvs[0] = BFieldElement::new(nia % 2);
                hvs[1] = BFieldElement::new(nia / 2);
                let st0 = self.op_stack.safe_peek(ST0);
                if !st0.is_zero() {
                    hvs[2] = st0.inverse();
                }
            }
            DivineSibling => {
                let node_index = self.op_stack.safe_peek(ST10).value();
                // set hv0 register to least significant bit of st10
                hvs[0] = BFieldElement::new(node_index as u64 % 2);
            }
            Split => {
                let elem = self.op_stack.safe_peek(ST0);
                let n: u64 = elem.value();
                let lo = BFieldElement::new(n & 0xffff_ffff);
                let hi = BFieldElement::new(n >> 32);
                if !lo.is_zero() {
                    let max_val_of_hi = BFieldElement::new(2_u64.pow(32) - 1);
                    hvs[0] = (hi - max_val_of_hi).inverse();
                }
            }
            Eq => {
                let lhs = self.op_stack.safe_peek(ST0);
                let rhs = self.op_stack.safe_peek(ST1);
                if !(rhs - lhs).is_zero() {
                    hvs[0] = (rhs - lhs).inverse();
                }
            }
            _ => (),
        }

        hvs
    }

    /// Perform the state transition as a mutable operation on `self`.
    pub fn step_mut<In>(
        &mut self,
        stdin: &mut In,
        secret_in: &mut In,
    ) -> Result<Option<VMOutput>, Box<dyn Error>>
    where
        In: InputStream,
    {
        // All instructions increase the cycle count
        self.cycle_count += 1;
        let mut vm_output = None;

        match self.current_instruction()? {
            Pop => {
                self.op_stack.pop()?;
                self.instruction_pointer += 1;
            }

            Push(arg) => {
                self.op_stack.push(arg);
                self.instruction_pointer += 2;
            }

            Divine(hint) => {
                use DivinationHint::*;

                let elem = if let Some(context) = hint {
                    match context {
                        Quotient => {
                            let numerator: u32 = self
                                .op_stack
                                .safe_peek(ST0)
                                .value()
                                .try_into()
                                .expect("Numerator uses more than 32 bits.");
                            let denominator: u32 = self
                                .op_stack
                                .safe_peek(ST1)
                                .value()
                                .try_into()
                                .expect("Denominator uses more than 32 bits.");
                            BFieldElement::new((numerator / denominator) as u64)
                        }
                    }
                } else {
                    secret_in.read_elem()?
                };
                self.op_stack.push(elem);
                self.instruction_pointer += 1;
            }

            Dup(arg) => {
                let elem = self.op_stack.safe_peek(arg);
                self.op_stack.push(elem);
                self.instruction_pointer += 2;
            }

            Swap(arg) => {
                // st[0] ... st[n] -> st[n] ... st[0]
                self.op_stack.safe_swap(arg);
                self.instruction_pointer += 2;
            }

            Nop => {
                self.instruction_pointer += 1;
            }

            Skiz => {
                let elem = self.op_stack.pop()?;
                self.instruction_pointer += if elem.is_zero() {
                    let next_instruction = self.next_instruction()?;
                    1 + next_instruction.size()
                } else {
                    1
                };
            }

            Call(addr) => {
                let o_plus_2 = self.instruction_pointer as u32 + 2;
                let pair = (BFieldElement::new(o_plus_2 as u64), addr);
                self.jump_stack.push(pair);
                self.instruction_pointer = addr.value() as usize;
            }

            Return => {
                let (orig_addr, _dest_addr) = self.jump_stack_pop()?;
                self.instruction_pointer = orig_addr.value() as usize;
            }

            Recurse => {
                let (_orig_addr, dest_addr) = self.jump_stack_peek()?;
                self.instruction_pointer = dest_addr.value() as usize;
            }

            Assert => {
                let elem = self.op_stack.pop()?;
                if !elem.is_one() {
                    return vm_err(AssertionFailed(
                        self.instruction_pointer,
                        self.cycle_count,
                        elem,
                    ));
                }
                self.instruction_pointer += 1;
            }

            Halt => {
                self.instruction_pointer = self.program.len();
            }

            ReadMem => {
                let ramp = self.op_stack.safe_peek(ST1);
                let ramv = self.memory_get(&ramp)?;
                self.op_stack.pop()?;
                self.op_stack.push(ramv);
                self.instruction_pointer += 1;
            }

            WriteMem => {
                let ramv = self.op_stack.safe_peek(ST0);
                let ramp = self.op_stack.safe_peek(ST1);
                self.ram.insert(ramp, ramv);
                self.instruction_pointer += 1;
            }

            Hash => {
                let mut hash_input = [BFieldElement::new(0); 2 * DIGEST_LENGTH];
                for i in 0..2 * DIGEST_LENGTH {
                    hash_input[i] = self.op_stack.pop()?;
                }

                let hash_trace = RescuePrimeRegular::trace(&hash_input);
                let hash_output = &hash_trace[hash_trace.len() - 1][0..DIGEST_LENGTH];
                let hash_trace_with_round_constants = Self::inprocess_hash_trace(&hash_trace);
                vm_output = Some(VMOutput::XlixTrace(hash_trace_with_round_constants));

                for i in (0..DIGEST_LENGTH).rev() {
                    self.op_stack.push(hash_output[i]);
                }

                for _ in 0..DIGEST_LENGTH {
                    self.op_stack.push(BFieldElement::zero());
                }

                self.instruction_pointer += 1;
            }

            DivineSibling => {
                self.divine_sibling::<In>(secret_in)?;
                self.instruction_pointer += 1;
            }

            AssertVector => {
                if !self.assert_vector() {
                    return vm_err(AssertionFailed(
                        self.instruction_pointer,
                        self.cycle_count,
                        self.op_stack
                            .peek(0)
                            .expect("Could not unwrap top of stack."),
                    ));
                }
                self.instruction_pointer += 1;
            }

            Add => {
                let lhs = self.op_stack.pop()?;
                let rhs = self.op_stack.pop()?;
                self.op_stack.push(lhs + rhs);
                self.instruction_pointer += 1;
            }

            Mul => {
                let lhs = self.op_stack.pop()?;
                let rhs = self.op_stack.pop()?;
                self.op_stack.push(lhs * rhs);
                self.instruction_pointer += 1;
            }

            Invert => {
                let elem = self.op_stack.pop()?;
                if elem.is_zero() {
                    return vm_err(InverseOfZero);
                }
                self.op_stack.push(elem.inverse());
                self.instruction_pointer += 1;
            }

            Split => {
                let elem = self.op_stack.pop()?;
                let n: u64 = elem.value();
                let lo = BFieldElement::new(n & 0xffff_ffff);
                let hi = BFieldElement::new(n >> 32);
                self.op_stack.push(lo);
                self.op_stack.push(hi);
                self.instruction_pointer += 1;
            }

            Eq => {
                let lhs = self.op_stack.pop()?;
                let rhs = self.op_stack.pop()?;
                self.op_stack.push(Self::eq(lhs, rhs));
                self.instruction_pointer += 1;
            }

            Lsb => {
                let top = self.op_stack.pop()?;
                let lsb = BFieldElement::new(top.value() & 1);
                self.op_stack.push(BFieldElement::new(top.value() >> 1));
                self.op_stack.push(lsb);
                self.instruction_pointer += 1;
            }

            XxAdd => {
                let lhs: XFieldElement = self.op_stack.pop_x()?;
                let rhs: XFieldElement = self.op_stack.safe_peek_x();
                self.op_stack.push_x(lhs + rhs);
                self.instruction_pointer += 1;
            }

            XxMul => {
                let lhs: XFieldElement = self.op_stack.pop_x()?;
                let rhs: XFieldElement = self.op_stack.safe_peek_x();
                self.op_stack.push_x(lhs * rhs);
                self.instruction_pointer += 1;
            }

            XInvert => {
                let elem: XFieldElement = self.op_stack.pop_x()?;
                self.op_stack.push_x(elem.inverse());
                self.instruction_pointer += 1;
            }

            XbMul => {
                let lhs: BFieldElement = self.op_stack.pop()?;
                let rhs: XFieldElement = self.op_stack.pop_x()?;
                self.op_stack.push_x(lhs.lift() * rhs);
                self.instruction_pointer += 1;
            }

            WriteIo => {
                vm_output = Some(VMOutput::WriteOutputSymbol(self.op_stack.pop()?));
                self.instruction_pointer += 1;
            }

            ReadIo => {
                let in_elem = stdin.read_elem()?;
                self.op_stack.push(in_elem);
                self.instruction_pointer += 1;
            }
        }

        // Check that no instruction left the OpStack with too few elements
        if self.op_stack.is_too_shallow() {
            return vm_err(OpStackTooShallow);
        }

        Ok(vm_output)
    }

    pub fn to_instruction_row(
        &self,
        current_instruction: Instruction,
    ) -> [BFieldElement; instruction_table::BASE_WIDTH] {
        use InstructionBaseTableColumn::*;
        let mut row = [BFieldElement::zero(); instruction_table::BASE_WIDTH];

        row[usize::from(Address)] = (self.instruction_pointer as u32).try_into().unwrap();
        row[usize::from(CI)] = current_instruction.opcode_b();
        row[usize::from(NIA)] = self.nia();

        row
    }

    pub fn to_processor_row(&self) -> [BFieldElement; processor_table::BASE_WIDTH] {
        use ProcessorBaseTableColumn::*;
        let mut row = [BFieldElement::zero(); processor_table::BASE_WIDTH];

        // todo: is `Instruction::Halt` a good default?
        let current_instruction = self.current_instruction().unwrap_or(Instruction::Halt);
        let hvs = self.derive_helper_variables();
        let ramp = self.op_stack.st(Ord16::ST1);

        row[usize::from(CLK)] = BFieldElement::new(self.cycle_count as u64);
        row[usize::from(IP)] = (self.instruction_pointer as u32).try_into().unwrap();
        row[usize::from(CI)] = current_instruction.opcode_b();
        row[usize::from(NIA)] = self.nia();
        row[usize::from(IB0)] = current_instruction.ib(Ord7::IB0);
        row[usize::from(IB1)] = current_instruction.ib(Ord7::IB1);
        row[usize::from(IB2)] = current_instruction.ib(Ord7::IB2);
        row[usize::from(IB3)] = current_instruction.ib(Ord7::IB3);
        row[usize::from(IB4)] = current_instruction.ib(Ord7::IB4);
        row[usize::from(IB5)] = current_instruction.ib(Ord7::IB5);
        row[usize::from(IB6)] = current_instruction.ib(Ord7::IB6);
        row[usize::from(JSP)] = self.jsp();
        row[usize::from(JSO)] = self.jso();
        row[usize::from(JSD)] = self.jsd();
        row[usize::from(ST0)] = self.op_stack.st(Ord16::ST0);
        row[usize::from(ST1)] = self.op_stack.st(Ord16::ST1);
        row[usize::from(ST2)] = self.op_stack.st(Ord16::ST2);
        row[usize::from(ST3)] = self.op_stack.st(Ord16::ST3);
        row[usize::from(ST4)] = self.op_stack.st(Ord16::ST4);
        row[usize::from(ST5)] = self.op_stack.st(Ord16::ST5);
        row[usize::from(ST6)] = self.op_stack.st(Ord16::ST6);
        row[usize::from(ST7)] = self.op_stack.st(Ord16::ST7);
        row[usize::from(ST8)] = self.op_stack.st(Ord16::ST8);
        row[usize::from(ST9)] = self.op_stack.st(Ord16::ST9);
        row[usize::from(ST10)] = self.op_stack.st(Ord16::ST10);
        row[usize::from(ST11)] = self.op_stack.st(Ord16::ST11);
        row[usize::from(ST12)] = self.op_stack.st(Ord16::ST12);
        row[usize::from(ST13)] = self.op_stack.st(Ord16::ST13);
        row[usize::from(ST14)] = self.op_stack.st(Ord16::ST14);
        row[usize::from(ST15)] = self.op_stack.st(Ord16::ST15);
        row[usize::from(OSP)] = self.op_stack.osp();
        row[usize::from(OSV)] = self.op_stack.osv();
        row[usize::from(HV0)] = hvs[0];
        row[usize::from(HV1)] = hvs[1];
        row[usize::from(HV2)] = hvs[2];
        row[usize::from(HV3)] = hvs[3];
        row[usize::from(RAMV)] = *self.ram.get(&ramp).unwrap_or(&BFieldElement::new(0));

        row
    }

    pub fn to_op_stack_row(
        &self,
        current_instruction: Instruction,
    ) -> [BFieldElement; op_stack_table::BASE_WIDTH] {
        use OpStackBaseTableColumn::*;
        let mut row = [BFieldElement::zero(); op_stack_table::BASE_WIDTH];

        row[usize::from(CLK)] = BFieldElement::new(self.cycle_count as u64);
        row[usize::from(IB1ShrinkStack)] = current_instruction.ib(IB1);
        row[usize::from(OSP)] = self.op_stack.osp();
        row[usize::from(OSV)] = self.op_stack.osv();

        row
    }

    pub fn to_ram_row(&self) -> [BFieldElement; ram_table::BASE_WIDTH] {
        use RamBaseTableColumn::*;
        let ramp = self.op_stack.st(ST1);

        let mut row = [BFieldElement::zero(); ram_table::BASE_WIDTH];

        row[usize::from(CLK)] = BFieldElement::new(self.cycle_count as u64);
        row[usize::from(RAMP)] = ramp;
        row[usize::from(RAMV)] = *self.ram.get(&ramp).unwrap_or(&BFieldElement::new(0));
        // value of InverseOfRampDifference is only known after sorting the RAM Table, thus not set

        row
    }

    pub fn to_jump_stack_row(
        &self,
        current_instruction: Instruction,
    ) -> [BFieldElement; jump_stack_table::BASE_WIDTH] {
        use JumpStackBaseTableColumn::*;
        let mut row = [BFieldElement::zero(); jump_stack_table::BASE_WIDTH];

        row[usize::from(CLK)] = BFieldElement::new(self.cycle_count as u64);
        row[usize::from(CI)] = current_instruction.opcode_b();
        row[usize::from(JSP)] = self.jsp();
        row[usize::from(JSO)] = self.jso();
        row[usize::from(JSD)] = self.jsd();

        row
    }

    fn eq(lhs: BFieldElement, rhs: BFieldElement) -> BFieldElement {
        if lhs == rhs {
            BFieldElement::one()
        } else {
            BFieldElement::zero()
        }
    }

    fn nia(&self) -> BFieldElement {
        self.current_instruction()
            .map(|curr_instr| {
                curr_instr.arg().unwrap_or_else(|| {
                    self.next_instruction()
                        .map(|next_instr| next_instr.opcode_b())
                        .unwrap_or_else(|_| BFieldElement::zero())
                })
            })
            .unwrap_or_else(|_| BFieldElement::zero())
    }

    /// Jump-stack pointer
    fn jsp(&self) -> BFieldElement {
        BFieldElement::new(self.jump_stack.len() as u64)
    }

    /// Jump-stack origin
    fn jso(&self) -> BFieldElement {
        self.jump_stack
            .last()
            .map(|(o, _d)| *o)
            .unwrap_or_else(BFieldElement::zero)
    }

    /// Jump-stack destination
    fn jsd(&self) -> BFieldElement {
        self.jump_stack
            .last()
            .map(|(_o, d)| *d)
            .unwrap_or_else(BFieldElement::zero)
    }

    pub fn current_instruction(&self) -> Result<Instruction, Box<dyn Error>> {
        self.program
            .get(self.instruction_pointer)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow(self.instruction_pointer)))
            .copied()
    }

    // Return the next instruction on the tape, skipping arguments
    //
    // Note that this is not necessarily the next instruction to execute,
    // since the current instruction could be a jump, but it is either
    // program[ip + 1] or program[ip + 2] depending on whether the current
    // instruction takes an argument or not.
    pub fn next_instruction(&self) -> Result<Instruction, Box<dyn Error>> {
        let ci = self.current_instruction()?;
        let ci_size = ci.size();
        let ni_pointer = self.instruction_pointer + ci_size;
        self.program
            .get(ni_pointer)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow(ni_pointer)))
            .copied()
    }

    fn _next_next_instruction(&self) -> Result<Instruction, Box<dyn Error>> {
        let cur_size = self.current_instruction()?.size();
        let next_size = self.next_instruction()?.size();
        self.program
            .get(self.instruction_pointer + cur_size + next_size)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow(self.instruction_pointer)))
            .copied()
    }

    fn jump_stack_pop(&mut self) -> Result<(BFieldElement, BFieldElement), Box<dyn Error>> {
        self.jump_stack
            .pop()
            .ok_or_else(|| vm_fail(JumpStackTooShallow))
    }

    fn jump_stack_peek(&mut self) -> Result<(BFieldElement, BFieldElement), Box<dyn Error>> {
        self.jump_stack
            .last()
            .copied()
            .ok_or_else(|| vm_fail(JumpStackTooShallow))
    }

    fn memory_get(&self, mem_addr: &BFieldElement) -> Result<BFieldElement, Box<dyn Error>> {
        self.ram
            .get(mem_addr)
            .copied()
            .ok_or_else(|| vm_fail(MemoryAddressNotFound))
    }

    fn assert_vector(&self) -> bool {
        for i in 0..DIGEST_LENGTH {
            // Safe as long as 2 * DIGEST_LEN <= OP_STACK_REG_COUNT
            let lhs = i.try_into().unwrap();
            let rhs = (i + DIGEST_LENGTH).try_into().unwrap();

            if self.op_stack.safe_peek(lhs) != self.op_stack.safe_peek(rhs) {
                return false;
            }
        }
        true
    }

    pub fn read_word(&self) -> Result<Option<BFieldElement>, Box<dyn Error>> {
        let current_instruction = self.current_instruction()?;
        if matches!(current_instruction, ReadIo) {
            Ok(Some(self.op_stack.safe_peek(ST0)))
        } else {
            Ok(None)
        }
    }

    fn divine_sibling<In: InputStream>(
        &mut self,
        secret_in: &mut In,
    ) -> Result<(), Box<dyn Error>> {
        // st0-st4
        for _ in 0..DIGEST_LENGTH {
            self.op_stack.pop()?;
        }

        // st5-st9
        let known_digest: [BFieldElement; DIGEST_LENGTH] = [
            self.op_stack.pop()?,
            self.op_stack.pop()?,
            self.op_stack.pop()?,
            self.op_stack.pop()?,
            self.op_stack.pop()?,
        ];

        // st10
        let node_index: u32 = self.op_stack.pop()?.try_into()?;

        // nondeterministic guess, flipped
        let mut sibling_digest = [BFieldElement::zero(); DIGEST_LENGTH];
        sibling_digest[4] = secret_in.read_elem()?;
        sibling_digest[3] = secret_in.read_elem()?;
        sibling_digest[2] = secret_in.read_elem()?;
        sibling_digest[1] = secret_in.read_elem()?;
        sibling_digest[0] = secret_in.read_elem()?;

        // least significant bit
        let hv0 = node_index % 2;

        // push new node index
        // st10
        self.op_stack
            .push(BFieldElement::new(node_index as u64 >> 1));

        // push 2 digests, in correct order
        // Correct order means the following:
        //
        // | sponge | stack | digest element | hv0 == 0 | hv0 == 1 |
        // |--------|-------|----------------|----------|----------|
        // | r0     | st0   | left0          | known0   | sibling0 |
        // | r1     | st1   | left1          | known1   | sibling1 |
        // | r2     | st2   | left2          | known2   | sibling2 |
        // | r3     | st3   | left3          | known3   | sibling3 |
        // | r4     | st4   | left4          | known4   | sibling4 |
        // | r5     | st5   | right0         | sibling0 | known0   |
        // | r6     | st6   | right1         | sibling1 | known1   |
        // | r7     | st7   | right2         | sibling2 | known2   |
        // | r8     | st8   | right3         | sibling3 | known3   |
        // | r9     | st9   | right4         | sibling4 | known4   |

        let (top_digest, runner_up) = if hv0 == 0 {
            (known_digest, sibling_digest)
        } else {
            (sibling_digest, known_digest)
        };

        for digest_element in runner_up.iter().rev() {
            self.op_stack.push(*digest_element);
        }

        for digest_element in top_digest.iter().rev() {
            self.op_stack.push(*digest_element);
        }

        Ok(())
    }

    fn inprocess_hash_trace(
        hash_trace: &[[BFieldElement;
              hash_table::BASE_WIDTH - hash_table::NUM_ROUND_CONSTANTS - 1]],
    ) -> Vec<[BFieldElement; hash_table::BASE_WIDTH]> {
        let mut hash_trace_with_constants = vec![];
        for (index, trace_row) in hash_trace.iter().enumerate() {
            let round_number = index + 1;
            let round_constants = Self::rescue_xlix_round_constants_by_round_number(round_number);
            let mut new_trace_row = [BFieldElement::zero(); hash_table::BASE_WIDTH];
            let mut offset = 0;
            new_trace_row[offset] = BFieldElement::new(round_number as u64);
            offset += 1;
            new_trace_row[offset..offset + STATE_SIZE].copy_from_slice(trace_row);
            offset += STATE_SIZE;
            new_trace_row[offset..].copy_from_slice(&round_constants);
            hash_trace_with_constants.push(new_trace_row)
        }
        hash_trace_with_constants
    }

    /// rescue_xlix_round_constants_by_round_number
    /// returns the 2m round constant for round `round_number`.
    /// This counter starts at 1; round number 0 indicates padding;
    /// and round number 9 indicates a transition to a new hash so
    /// the round constants will be all zeros.
    fn rescue_xlix_round_constants_by_round_number(
        round_number: usize,
    ) -> [BFieldElement; NUM_ROUND_CONSTANTS] {
        let round_constants: [BFieldElement; TOTAL_NUM_CONSTANTS] = ROUND_CONSTANTS
            .iter()
            .map(|&x| BFieldElement::new(x))
            .collect_vec()
            .try_into()
            .unwrap();

        match round_number {
            0 => [BFieldElement::zero(); hash_table::NUM_ROUND_CONSTANTS],
            i if i <= NUM_ROUNDS => round_constants
                [NUM_ROUND_CONSTANTS * (i - 1)..NUM_ROUND_CONSTANTS * i]
                .try_into()
                .unwrap(),
            i if i == NUM_ROUNDS + 1 => [BFieldElement::zero(); hash_table::NUM_ROUND_CONSTANTS],
            _ => panic!("Round with number {round_number} does not have round constants."),
        }
    }
}

impl<'pgm> Display for VMState<'pgm> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.current_instruction() {
            Ok(_) => {
                let row = self.to_processor_row();
                write!(f, "{}", ProcessorMatrixRow { row })
            }
            Err(_) => write!(f, "END-OF-FILE"),
        }
    }
}

#[cfg(test)]
mod vm_state_tests {
    use rand::thread_rng;
    use twenty_first::shared_math::traits::GetRandomElements;
    use twenty_first::util_types::merkle_tree::MerkleTree;
    use twenty_first::util_types::simple_hasher::Hasher;

    use crate::instruction::sample_programs;
    use crate::op_stack::OP_STACK_REG_COUNT;

    use super::*;

    // Property: All instructions increase the cycle count by 1.
    // Property: Most instructions increase the instruction pointer by 1.

    #[test]
    fn tvm_op_stack_big_enough_test() {
        assert!(
            DIGEST_LENGTH <= OP_STACK_REG_COUNT,
            "The OpStack must be large enough to hold a single Rescue-Prime digest"
        );
    }

    #[test]
    fn run_tvm_parse_pop_p_test() {
        let program = sample_programs::push_push_add_pop_p();
        let (trace, _out, _err) = program.run_with_input(&[], &[]);

        for state in trace.iter() {
            println!("{}", state);
        }
    }

    #[test]
    fn run_tvm_hello_world_1_test() {
        let code = sample_programs::HELLO_WORLD_1;
        let program = Program::from_code(code).unwrap();
        let (trace, _out, _err) = program.run_with_input(&[], &[]);

        let last_state = trace.last().unwrap();
        assert_eq!(BFieldElement::zero(), last_state.op_stack.safe_peek(ST0));

        println!("{}", last_state);
    }

    #[test]
    fn run_tvm_halt_then_do_stuff_test() {
        let halt_then_do_stuff = "halt push 1 push 2 add invert write_io";
        let program = Program::from_code(halt_then_do_stuff).unwrap();
        let (trace, _out, err) = program.run_with_input(&[], &[]);

        for state in trace.iter() {
            println!("{}", state);
        }
        if let Some(e) = err {
            println!("Error: {}", e);
        }

        // check for graceful termination
        let last_state = trace.get(trace.len() - 1).unwrap();
        assert_eq!(last_state.current_instruction().unwrap(), Halt);
    }

    #[test]
    fn run_tvm_basic_ram_read_write_test() {
        let program = Program::from_code(sample_programs::BASIC_RAM_READ_WRITE).unwrap();
        let (trace, _out, err) = program.run_with_input(&[], &[]);

        for state in trace.iter() {
            println!("{}", state);
        }
        if let Some(e) = err {
            println!("Error: {}", e);
        }

        let last_state = trace.last().expect("Execution seems to have failed.");
        let five = BFieldElement::new(5);
        let seven = BFieldElement::new(7);
        let fifteen = BFieldElement::new(15);
        let sixteen = BFieldElement::new(16);
        assert_eq!(seven, last_state.op_stack.st(ST0));
        assert_eq!(five, last_state.op_stack.st(ST1));
        assert_eq!(sixteen, last_state.op_stack.st(ST2));
        assert_eq!(fifteen, last_state.op_stack.st(ST3));
        assert_eq!(last_state.ram[&five], seven);
        assert_eq!(last_state.ram[&fifteen], sixteen);
    }

    #[test]
    fn run_tvm_edgy_ram_writes_test() {
        let program = Program::from_code(sample_programs::EDGY_RAM_WRITES).unwrap();
        let (trace, _out, err) = program.run_with_input(&[], &[]);

        for state in trace.iter() {
            println!("{}", state);
        }
        if let Some(e) = err {
            println!("Error: {}", e);
        }

        let last_state = trace.last().expect("Execution seems to have failed.");
        let zero = BFieldElement::new(0);
        let three = BFieldElement::new(3);
        let five = BFieldElement::new(5);
        assert_eq!(three, last_state.op_stack.st(ST0));
        assert_eq!(five, last_state.op_stack.st(ST1));
        assert_eq!(three, last_state.op_stack.st(ST2));
        assert_eq!(last_state.ram[&zero], zero);
        assert_eq!(last_state.ram[&five], three);
    }

    #[test]
    fn run_tvm_sample_weights_test() {
        let program = Program::from_code(sample_programs::SAMPLE_WEIGHTS).unwrap();
        println!("Successfully parsed the program.");
        let (trace, _out, err) = program.run_with_input(&[BFieldElement::new(11)], &[]);

        for state in trace.iter() {
            println!("{}", state);
        }
        if let Some(e) = err {
            panic!("The VM encountered an error: {}", e);
        }

        // check for graceful termination
        let last_state = trace.get(trace.len() - 1).unwrap();
        assert_eq!(last_state.current_instruction().unwrap(), Halt);
    }

    #[test]
    fn run_tvm_mt_ap_verify_test() {
        // generate merkle tree
        type H = RescuePrimeRegular;
        type Digest = <H as Hasher>::Digest;
        let hasher = H::new();
        let mut rng = thread_rng();
        const NUM_LEAFS: usize = 64;
        let leafs: [Digest; NUM_LEAFS] = (0..NUM_LEAFS)
            .map(|_| {
                BFieldElement::random_elements(DIGEST_LENGTH, &mut rng)
                    .try_into()
                    .unwrap()
                // hasher.hash_sequence(&BFieldElement::new(i as u64).to_sequence())
            })
            .collect_vec()
            .try_into()
            .unwrap();
        let zero_padding = [BFieldElement::zero(); DIGEST_LENGTH];
        let digests = leafs
            .iter()
            .map(|l| hasher.hash_pair(&zero_padding, l))
            .collect_vec();
        let merkle_tree = MerkleTree::<H>::from_digests(&digests);
        let root = merkle_tree.get_root();

        // generate program
        let program = Program::from_code(sample_programs::MT_AP_VERIFY).unwrap();
        let order: Vec<usize> = (0..5).rev().collect();
        let (trace, _out, err) = program.run_with_input(
            &[
                // number of path tests
                BFieldElement::new(3),
                // Merkle root
                root[order[0]],
                root[order[1]],
                root[order[2]],
                root[order[3]],
                root[order[4]],
                // node index 64, leaf index 0
                BFieldElement::new(64),
                // value of leaf with index 0
                leafs[0][order[0]],
                leafs[0][order[1]],
                leafs[0][order[2]],
                leafs[0][order[3]],
                leafs[0][order[4]],
                // node index 92, leaf index 28
                // 92 = 1011100_2
                // 28 =   11100_2
                BFieldElement::new(92),
                // value of leaf with index 28
                leafs[28][order[0]],
                leafs[28][order[1]],
                leafs[28][order[2]],
                leafs[28][order[3]],
                leafs[28][order[4]],
                // node index 119, leaf index 55
                BFieldElement::new(119),
                // 119 = 1110111_2
                // 55  =  110111_2
                // value of leaf with node 55
                leafs[55][order[0]],
                leafs[55][order[1]],
                leafs[55][order[2]],
                leafs[55][order[3]],
                leafs[55][order[4]],
            ],
            &[
                // Merkle Authentication Path 0
                // Merkle Authentication Path 0 Element 0
                merkle_tree.get_authentication_path(0)[0][order[0]],
                merkle_tree.get_authentication_path(0)[0][order[1]],
                merkle_tree.get_authentication_path(0)[0][order[2]],
                merkle_tree.get_authentication_path(0)[0][order[3]],
                merkle_tree.get_authentication_path(0)[0][order[4]],
                // Merkle Authentication Path 0 Element 1
                merkle_tree.get_authentication_path(0)[1][order[0]],
                merkle_tree.get_authentication_path(0)[1][order[1]],
                merkle_tree.get_authentication_path(0)[1][order[2]],
                merkle_tree.get_authentication_path(0)[1][order[3]],
                merkle_tree.get_authentication_path(0)[1][order[4]],
                // Merkle Authentication Path 0 Element 2
                merkle_tree.get_authentication_path(0)[2][order[0]],
                merkle_tree.get_authentication_path(0)[2][order[1]],
                merkle_tree.get_authentication_path(0)[2][order[2]],
                merkle_tree.get_authentication_path(0)[2][order[3]],
                merkle_tree.get_authentication_path(0)[2][order[4]],
                // Merkle Authentication Path 0 Element 3
                merkle_tree.get_authentication_path(0)[3][order[0]],
                merkle_tree.get_authentication_path(0)[3][order[1]],
                merkle_tree.get_authentication_path(0)[3][order[2]],
                merkle_tree.get_authentication_path(0)[3][order[3]],
                merkle_tree.get_authentication_path(0)[3][order[4]],
                // Merkle Authentication Path 0 Element 4
                merkle_tree.get_authentication_path(0)[4][order[0]],
                merkle_tree.get_authentication_path(0)[4][order[1]],
                merkle_tree.get_authentication_path(0)[4][order[2]],
                merkle_tree.get_authentication_path(0)[4][order[3]],
                merkle_tree.get_authentication_path(0)[4][order[4]],
                // Merkle Authentication Path 0 Element 5
                merkle_tree.get_authentication_path(0)[5][order[0]],
                merkle_tree.get_authentication_path(0)[5][order[1]],
                merkle_tree.get_authentication_path(0)[5][order[2]],
                merkle_tree.get_authentication_path(0)[5][order[3]],
                merkle_tree.get_authentication_path(0)[5][order[4]],
                // Merkle Authentication Path 1
                // Merkle Authentication Path 1 Element 0
                merkle_tree.get_authentication_path(28)[0][order[0]],
                merkle_tree.get_authentication_path(28)[0][order[1]],
                merkle_tree.get_authentication_path(28)[0][order[2]],
                merkle_tree.get_authentication_path(28)[0][order[3]],
                merkle_tree.get_authentication_path(28)[0][order[4]],
                // Merkle Authentication Path 1 Element 1
                merkle_tree.get_authentication_path(28)[1][order[0]],
                merkle_tree.get_authentication_path(28)[1][order[1]],
                merkle_tree.get_authentication_path(28)[1][order[2]],
                merkle_tree.get_authentication_path(28)[1][order[3]],
                merkle_tree.get_authentication_path(28)[1][order[4]],
                // Merkle Authentication Path 1 Element 2
                merkle_tree.get_authentication_path(28)[2][order[0]],
                merkle_tree.get_authentication_path(28)[2][order[1]],
                merkle_tree.get_authentication_path(28)[2][order[2]],
                merkle_tree.get_authentication_path(28)[2][order[3]],
                merkle_tree.get_authentication_path(28)[2][order[4]],
                // Merkle Authentication Path 1 Element 3
                merkle_tree.get_authentication_path(28)[3][order[0]],
                merkle_tree.get_authentication_path(28)[3][order[1]],
                merkle_tree.get_authentication_path(28)[3][order[2]],
                merkle_tree.get_authentication_path(28)[3][order[3]],
                merkle_tree.get_authentication_path(28)[3][order[4]],
                // Merkle Authentication Path 1 Element 4
                merkle_tree.get_authentication_path(28)[4][order[0]],
                merkle_tree.get_authentication_path(28)[4][order[1]],
                merkle_tree.get_authentication_path(28)[4][order[2]],
                merkle_tree.get_authentication_path(28)[4][order[3]],
                merkle_tree.get_authentication_path(28)[4][order[4]],
                // Merkle Authentication Path 1 Element 5
                merkle_tree.get_authentication_path(28)[5][order[0]],
                merkle_tree.get_authentication_path(28)[5][order[1]],
                merkle_tree.get_authentication_path(28)[5][order[2]],
                merkle_tree.get_authentication_path(28)[5][order[3]],
                merkle_tree.get_authentication_path(28)[5][order[4]],
                // Merkle Authentication Path 2
                // Merkle Authentication Path 2 Element 0
                merkle_tree.get_authentication_path(55)[0][order[0]],
                merkle_tree.get_authentication_path(55)[0][order[1]],
                merkle_tree.get_authentication_path(55)[0][order[2]],
                merkle_tree.get_authentication_path(55)[0][order[3]],
                merkle_tree.get_authentication_path(55)[0][order[4]],
                // Merkle Authentication Path 2 Element 1
                merkle_tree.get_authentication_path(55)[1][order[0]],
                merkle_tree.get_authentication_path(55)[1][order[1]],
                merkle_tree.get_authentication_path(55)[1][order[2]],
                merkle_tree.get_authentication_path(55)[1][order[3]],
                merkle_tree.get_authentication_path(55)[1][order[4]],
                // Merkle Authentication Path 2 Element 2
                merkle_tree.get_authentication_path(55)[2][order[0]],
                merkle_tree.get_authentication_path(55)[2][order[1]],
                merkle_tree.get_authentication_path(55)[2][order[2]],
                merkle_tree.get_authentication_path(55)[2][order[3]],
                merkle_tree.get_authentication_path(55)[2][order[4]],
                // Merkle Authentication Path 2 Element 3
                merkle_tree.get_authentication_path(55)[3][order[0]],
                merkle_tree.get_authentication_path(55)[3][order[1]],
                merkle_tree.get_authentication_path(55)[3][order[2]],
                merkle_tree.get_authentication_path(55)[3][order[3]],
                merkle_tree.get_authentication_path(55)[3][order[4]],
                // Merkle Authentication Path 2 Element 4
                merkle_tree.get_authentication_path(55)[4][order[0]],
                merkle_tree.get_authentication_path(55)[4][order[1]],
                merkle_tree.get_authentication_path(55)[4][order[2]],
                merkle_tree.get_authentication_path(55)[4][order[3]],
                merkle_tree.get_authentication_path(55)[4][order[4]],
                // Merkle Authentication Path 2 Element 5
                merkle_tree.get_authentication_path(55)[5][order[0]],
                merkle_tree.get_authentication_path(55)[5][order[1]],
                merkle_tree.get_authentication_path(55)[5][order[2]],
                merkle_tree.get_authentication_path(55)[5][order[3]],
                merkle_tree.get_authentication_path(55)[5][order[4]],
            ],
        );

        for state in trace.iter() {
            println!("{}", state);
        }
        if let Some(e) = err {
            panic!("The VM encountered an error: {}", e);
        }

        // check for graceful termination
        let last_state = trace.get(trace.len() - 1).unwrap();
        assert_eq!(last_state.current_instruction().unwrap(), Halt);
    }

    #[test]
    fn run_tvm_get_colinear_y_test() {
        let program = Program::from_code(sample_programs::GET_COLINEAR_Y).unwrap();
        println!("Successfully parsed the program.");
        let (trace, out, err) = program.run_with_input(
            &[
                BFieldElement::new(7),
                BFieldElement::new(2),
                BFieldElement::new(1),
                BFieldElement::new(3),
                BFieldElement::new(4),
            ],
            &[],
        );
        assert_eq!(out[0], BFieldElement::new(4));
        for state in trace.iter() {
            println!("{}", state);
        }
        if let Some(e) = err {
            panic!("The VM encountered an error: {}", e);
        }

        // check for graceful termination
        let last_state = trace.get(trace.len() - 1).unwrap();
        assert_eq!(last_state.current_instruction().unwrap(), Halt);
    }

    #[test]
    fn run_tvm_countdown_from_10_test() {
        let code = sample_programs::COUNTDOWN_FROM_10;
        let program = Program::from_code(code).unwrap();
        let (trace, _out, _err) = program.run_with_input(&[], &[]);

        println!("{}", program);
        for state in trace.iter() {
            println!("{}", state);
        }

        let last_state = trace.last().unwrap();
        assert_eq!(BFieldElement::zero(), last_state.op_stack.st(ST0));
    }

    #[test]
    fn run_tvm_fibonacci_vit_tvm() {
        let code = sample_programs::FIBONACCI_VIT;
        let program = Program::from_code(code).unwrap();

        let (trace, out, err) = program.run_with_input(&[7_u64.into()], &[]);

        for state in trace.iter() {
            println!("{}", state);
        }
        if let Some(e) = err {
            panic!("The VM encountered an error: {e}");
        }

        assert_eq!(Some(&BFieldElement::new(21)), out.get(0));
    }

    #[test]
    fn run_tvm_fibonacci_lt_test() {
        let code = sample_programs::FIBONACCI_LT;
        let program = Program::from_code(code).unwrap();
        let (trace, _out, _err) = program.run_with_input(&[], &[]);

        println!("{}", program);
        for state in trace.iter() {
            println!("{}", state);
        }

        let last_state = trace.last().unwrap();
        assert_eq!(BFieldElement::new(21), last_state.op_stack.st(ST0));
    }

    #[test]
    fn run_tvm_gcd_test() {
        let code = sample_programs::GCD_X_Y;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);
        let (trace, out, _err) = program.run_with_input(&[42_u64.into(), 56_u64.into()], &[]);

        println!("{}", program);
        for state in trace.iter() {
            println!("{}", state);
        }

        let expected = BFieldElement::new(14);
        let actual = *out.last().unwrap();
        assert_eq!(expected, actual);
    }

    #[test]
    #[ignore = "assembly not up to date"]
    fn run_tvm_xgcd_test() {
        // The XGCD program is work in progress.
        let code = sample_programs::XGCD;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);
        let (trace, _out, _err) = program.run_with_input(&[], &[]);

        println!("{}", program);
        for state in trace.iter() {
            println!("{}", state);
        }

        let _last_state = trace.last().unwrap();

        let _expected = BFieldElement::new(14);
        let _actual = _last_state.op_stack.st(ST0);

        //assert_eq!(expected, actual);
    }

    #[test]
    fn run_tvm_swap_test() {
        let code = "push 1 push 2 swap1 halt";
        let program = Program::from_code(code).unwrap();
        let (trace, _out, _err) = program.run_with_input(&[], &[]);

        for state in trace.iter() {
            println!("{}", state);
        }
    }
}
