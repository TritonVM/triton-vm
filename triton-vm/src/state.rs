use std::collections::HashMap;
use std::convert::TryInto;
use std::error::Error;
use std::fmt::Display;

use itertools::Itertools;
use num_traits::{One, Zero};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::other;
use twenty_first::shared_math::rescue_prime_regular::{
    RescuePrimeRegular, DIGEST_LENGTH, NUM_ROUNDS, ROUND_CONSTANTS, STATE_SIZE,
};
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::error::vm_err;
use crate::table::base_matrix::ProcessorMatrixRow;
use crate::table::hash_table::{NUM_ROUND_CONSTANTS, TOTAL_NUM_CONSTANTS};

use super::error::{vm_fail, InstructionError::*};
use super::instruction::{AnInstruction::*, Instruction};
use super::op_stack::OpStack;
use super::ord_n::{Ord16::*, Ord7::*};
use super::stdio::InputStream;
use super::table::{hash_table, instruction_table, jump_stack_table, op_stack_table, u32_op_table};
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

    /// Trace of u32 operations for u32 op table
    ///
    /// One row per defined bit
    U32OpTrace(Vec<[BFieldElement; u32_op_table::BASE_WIDTH]>),
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
        self.program.len() <= self.instruction_pointer
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
            Pop | Skiz | Assert | Add | Mul | Eq | Lt | And | Xor | XbMul | WriteIo => {
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
            Div => {
                hvs[0] = BFieldElement::one();
                let st0 = self.op_stack.safe_peek(ST0);
                if !st0.is_zero() {
                    hvs[2] = st0.inverse();
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

            Divine => {
                let elem = secret_in.read_elem()?;
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
                    return vm_err(AssertionFailed);
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
                    return vm_err(AssertionFailed);
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

            Lt => {
                let lhs: u32 = self.op_stack.pop_u32()?;
                let rhs: u32 = self.op_stack.pop_u32()?;
                self.op_stack.push(Self::lt(lhs, rhs));
                let trace = self.u32_op_trace(lhs, rhs);
                vm_output = Some(VMOutput::U32OpTrace(trace));
                self.instruction_pointer += 1;
            }

            And => {
                let lhs: u32 = self.op_stack.pop_u32()?;
                let rhs: u32 = self.op_stack.pop_u32()?;
                self.op_stack.push(BFieldElement::new((lhs & rhs) as u64));
                let trace = self.u32_op_trace(lhs, rhs);
                vm_output = Some(VMOutput::U32OpTrace(trace));
                self.instruction_pointer += 1;
            }

            Xor => {
                let lhs: u32 = self.op_stack.pop_u32()?;
                let rhs: u32 = self.op_stack.pop_u32()?;
                self.op_stack.push(BFieldElement::new((lhs ^ rhs) as u64));
                let trace = self.u32_op_trace(lhs, rhs);
                vm_output = Some(VMOutput::U32OpTrace(trace));
                self.instruction_pointer += 1;
            }

            Reverse => {
                let elem: u32 = self.op_stack.pop_u32()?;
                self.op_stack
                    .push(BFieldElement::new(elem.reverse_bits() as u64));

                // for instruction `reverse`, the Uint32 Table's RHS is (arbitrarily) set to 0
                let trace = self.u32_op_trace(elem, 0);
                vm_output = Some(VMOutput::U32OpTrace(trace));
                self.instruction_pointer += 1;
            }

            Div => {
                let denom: u32 = self.op_stack.pop_u32()?;
                let numerator: u32 = self.op_stack.pop_u32()?;
                let (quot, rem) = other::div_rem(numerator, denom);
                self.op_stack.push(BFieldElement::new(quot as u64));
                self.op_stack.push(BFieldElement::new(rem as u64));
                let trace = self.u32_op_trace(denom, numerator);
                vm_output = Some(VMOutput::U32OpTrace(trace));
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
        let ip = (self.instruction_pointer as u32).try_into().unwrap();
        let ci = current_instruction.opcode_b();
        let nia = self.nia();

        [ip, ci, nia]
    }

    pub fn to_processor_row(
        &self,
        current_instruction: Instruction,
    ) -> [BFieldElement; processor_table::BASE_WIDTH] {
        let clk = BFieldElement::new(self.cycle_count as u64);
        let ip = (self.instruction_pointer as u32).try_into().unwrap();
        // FIXME either have `nia()` use the argument `current_instruction` or derive `ci` from `ip`
        let ci = current_instruction.opcode_b();
        let nia = self.nia();
        let ib0 = current_instruction.ib(IB0);
        let ib1 = current_instruction.ib(IB1);
        let ib2 = current_instruction.ib(IB2);
        let ib3 = current_instruction.ib(IB3);
        let ib4 = current_instruction.ib(IB4);
        let ib5 = current_instruction.ib(IB5);
        let ib6 = current_instruction.ib(IB6);
        let st0 = self.op_stack.st(ST0);
        let st1 = self.op_stack.st(ST1);
        let st2 = self.op_stack.st(ST2);
        let st3 = self.op_stack.st(ST3);
        let st4 = self.op_stack.st(ST4);
        let st5 = self.op_stack.st(ST5);
        let st6 = self.op_stack.st(ST6);
        let st7 = self.op_stack.st(ST7);
        let st8 = self.op_stack.st(ST8);
        let st9 = self.op_stack.st(ST9);
        let st10 = self.op_stack.st(ST10);
        let st11 = self.op_stack.st(ST11);
        let st12 = self.op_stack.st(ST12);
        let st13 = self.op_stack.st(ST13);
        let st14 = self.op_stack.st(ST14);
        let st15 = self.op_stack.st(ST15);
        let osp = self.op_stack.osp();
        let osv = self.op_stack.osv();

        let hvs = self.derive_helper_variables();

        [
            clk,
            ip,
            ci,
            nia,
            ib0,
            ib1,
            ib2,
            ib3,
            ib4,
            ib5,
            ib6,
            self.jsp(),
            self.jso(),
            self.jsd(),
            st0,
            st1,
            st2,
            st3,
            st4,
            st5,
            st6,
            st7,
            st8,
            st9,
            st10,
            st11,
            st12,
            st13,
            st14,
            st15,
            osp,
            osv,
            hvs[0],
            hvs[1],
            hvs[2],
            hvs[3],
            *self.ram.get(&st1).unwrap_or(&BFieldElement::new(0)),
        ]
    }

    pub fn to_op_stack_row(
        &self,
        current_instruction: Instruction,
    ) -> [BFieldElement; op_stack_table::BASE_WIDTH] {
        let clk = BFieldElement::new(self.cycle_count as u64);
        let ib1_shrink_stack = current_instruction.ib(IB1);
        let osp = self.op_stack.osp();
        let osv = self.op_stack.osv();

        [clk, ib1_shrink_stack, osv, osp]
    }

    pub fn to_ram_row(&self) -> [BFieldElement; ram_table::BASE_WIDTH] {
        let clk = BFieldElement::new(self.cycle_count as u64);
        let ramp = self.op_stack.st(ST1);
        let ramv = *self.ram.get(&ramp).unwrap_or(&BFieldElement::new(0));

        // placeholder value – actual value only known after sorting the RAM Table
        let inverse_of_ramp_diff = BFieldElement::new(0);

        [clk, ramp, ramv, inverse_of_ramp_diff]
    }

    pub fn to_jump_stack_row(
        &self,
        current_instruction: Instruction,
    ) -> [BFieldElement; jump_stack_table::BASE_WIDTH] {
        let clk = BFieldElement::new(self.cycle_count as u64);
        let ci = current_instruction.opcode_b();

        [clk, ci, self.jsp(), self.jso(), self.jsd()]
    }

    pub fn u32_op_trace(
        &self,
        mut lhs: u32,
        mut rhs: u32,
    ) -> Vec<[BFieldElement; u32_op_table::BASE_WIDTH]> {
        let inverse_or_zero = |bfe: BFieldElement| {
            if bfe.is_zero() {
                bfe
            } else {
                bfe.inverse()
            }
        };

        let mut idc = 1;
        let mut bits = 0;
        let ci = self
            .current_instruction()
            .expect("U32 trace can only be generated with an instruction.")
            .opcode_b();

        let thirty_three = BFieldElement::new(33);
        let row = |idc: u32, bits: u32, lhs: u32, rhs: u32| {
            [
                BFieldElement::new(idc as u64),
                BFieldElement::new(bits as u64),
                inverse_or_zero(thirty_three - BFieldElement::new(bits as u64)),
                ci,
                BFieldElement::new(lhs as u64),
                BFieldElement::new(rhs as u64),
                Self::possibly_unclear_lt(idc, lhs, rhs),
                BFieldElement::new((lhs & rhs) as u64),
                BFieldElement::new((lhs ^ rhs) as u64),
                BFieldElement::new(lhs.reverse_bits() as u64),
                inverse_or_zero(BFieldElement::new(lhs as u64)),
                inverse_or_zero(BFieldElement::new(rhs as u64)),
            ]
        };

        let mut rows = vec![];
        let mut write_rows = true;
        while write_rows {
            rows.push(row(idc, bits, lhs, rhs));
            idc = 0;
            bits += 1;
            write_rows = lhs != 0 || rhs != 0;
            lhs >>= 1;
            rhs >>= 1;
        }

        rows
    }

    fn possibly_unclear_lt(idc: u32, lhs: u32, rhs: u32) -> BFieldElement {
        if idc == 0 && lhs == rhs {
            BFieldElement::new(2)
        } else {
            Self::lt(lhs, rhs)
        }
    }

    fn lt(lhs: u32, rhs: u32) -> BFieldElement {
        if lhs < rhs {
            BFieldElement::one()
        } else {
            BFieldElement::zero()
        }
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
        let known_digest: [BFieldElement; DIGEST_LENGTH] = [
            self.op_stack.pop()?,
            self.op_stack.pop()?,
            self.op_stack.pop()?,
            self.op_stack.pop()?,
            self.op_stack.pop()?,
        ];

        // st5-st9
        for _ in 0..DIGEST_LENGTH {
            self.op_stack.pop()?;
        }

        // st10
        let node_index: u32 = self.op_stack.pop()?.try_into()?;

        let sibling_digest = [
            secret_in.read_elem()?,
            secret_in.read_elem()?,
            secret_in.read_elem()?,
            secret_in.read_elem()?,
            secret_in.read_elem()?,
        ];

        // least significant bit
        let hv0 = node_index % 2;

        // push new node index
        // st10
        self.op_stack
            .push(BFieldElement::new(node_index as u64 >> 1));

        // push 2 digests, in correct order
        // Correct order means the following:
        //
        // | sponge | stack | digest element | hv0 == 0 |
        // |--------|-------|----------------|----------|
        // | r0     | st0   | left0          | known0   |
        // | r1     | st1   | left1          | known1   |
        // | r2     | st2   | left2          | known2   |
        // | r3     | st3   | left3          | known3   |
        // | r4     | st4   | left4          | known4   |
        // | r5     | st5   | right0         | sibling0 |
        // | r6     | st6   | right1         | sibling1 |
        // | r7     | st7   | right2         | sibling2 |
        // | r8     | st8   | right3         | sibling3 |
        // | r9     | st9   | right4         | sibling4 |

        if hv0 == 0 {
            for sibling_element in sibling_digest.iter().rev() {
                self.op_stack.push(*sibling_element);
            }

            for known_digest_element in known_digest.iter().rev() {
                self.op_stack.push(*known_digest_element);
            }
        } else {
            for known_digest_element in known_digest.iter().rev() {
                self.op_stack.push(*known_digest_element);
            }

            for sibling_element in sibling_digest.iter().rev() {
                self.op_stack.push(*sibling_element);
            }
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
            Ok(instruction) => {
                let row = self.to_processor_row(instruction);
                write!(f, "{}", ProcessorMatrixRow { row })
            }
            Err(_) => write!(f, "END-OF-FILE"),
        }
    }
}

#[cfg(test)]
mod vm_state_tests {
    use super::*;
    use crate::instruction::sample_programs;
    use crate::op_stack::OP_STACK_REG_COUNT;

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
        let last_state = trace.get(trace.len() - 2).unwrap();
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
        let last_state = trace.get(trace.len() - 2).unwrap();
        assert_eq!(last_state.current_instruction().unwrap(), Halt);
    }

    #[test]
    #[ignore]
    fn run_tvm_mt_ap_verify_test() {
        let program = Program::from_code(sample_programs::MT_AP_VERIFY).unwrap();
        println!("Successfully parsed the program.");
        let (trace, _out, err) = program.run_with_input(
            &[
                // Merkle root
                BFieldElement::new(2661879877493968030),
                BFieldElement::new(8411897398996365015),
                BFieldElement::new(11724741215505059774),
                BFieldElement::new(10869446635029787183),
                BFieldElement::new(3194712170375950680),
                // node index 64, leaf index 0
                BFieldElement::new(64),
                // value of leaf with index 0
                BFieldElement::new(17),
                BFieldElement::new(22),
                BFieldElement::new(19),
                // node index 92, leaf index 28
                BFieldElement::new(92),
                // value of leaf with index 28
                BFieldElement::new(45),
                BFieldElement::new(50),
                BFieldElement::new(47),
                // node index 119, leaf index 55
                BFieldElement::new(119),
                // value of leaf with node 55
                BFieldElement::new(72),
                BFieldElement::new(77),
                BFieldElement::new(74),
            ],
            &[
                // Merkle Authentication Path 0
                // Merkle Authentication Path 0 Element 0
                BFieldElement::new(7433611961471031299),
                BFieldElement::new(10663067815302282105),
                BFieldElement::new(11189271637150912214),
                BFieldElement::new(6731301558776007763),
                BFieldElement::new(12404371806864851196),
                // Merkle Authentication Path 0 Element 1
                BFieldElement::new(15447170459020364568),
                BFieldElement::new(13311771520545451802),
                BFieldElement::new(4832613912751814227),
                BFieldElement::new(16118512681346800136),
                BFieldElement::new(11903034542985100612),
                // Merkle Authentication Path 0 Element 2
                BFieldElement::new(927166763011592563),
                BFieldElement::new(1017721141586418898),
                BFieldElement::new(14149577177119432718),
                BFieldElement::new(11112535232426569259),
                BFieldElement::new(6770923340167310082),
                // Merkle Authentication Path 0 Element 3
                BFieldElement::new(11997402720255929816),
                BFieldElement::new(7083119985125877931),
                BFieldElement::new(3583918993470398367),
                BFieldElement::new(12665589384229632447),
                BFieldElement::new(4869924221127107207),
                // Merkle Authentication Path 0 Element 4
                BFieldElement::new(4108830855587634814),
                BFieldElement::new(11363551275926927759),
                BFieldElement::new(8897943612193465442),
                BFieldElement::new(18175199505544299571),
                BFieldElement::new(5933081913383911549),
                // Merkle Authentication Path 0 Element 5
                BFieldElement::new(239086846863014618),
                BFieldElement::new(18353654918351264251),
                BFieldElement::new(1162413056004073118),
                BFieldElement::new(63172233802162855),
                BFieldElement::new(15287652336563130555),
                // Merkle Authentication Path 1
                // Merkle Authentication Path 1 Element 0
                BFieldElement::new(9199975892950715767),
                BFieldElement::new(18392437377232084500),
                BFieldElement::new(7389509101855274876),
                BFieldElement::new(13193152724141987884),
                BFieldElement::new(12764531673520060724),
                // Merkle Authentication Path 1 Element 1
                BFieldElement::new(13265185672483741593),
                BFieldElement::new(4801722111881156327),
                BFieldElement::new(297253697970945484),
                BFieldElement::new(8955967409623509220),
                BFieldElement::new(10440367450900769517),
                // Merkle Authentication Path 1 Element 2
                BFieldElement::new(3378320220263195325),
                BFieldElement::new(17709073937843856976),
                BFieldElement::new(3737595776877974498),
                BFieldElement::new(1050267233733511018),
                BFieldElement::new(18417031760560110797),
                // Merkle Authentication Path 1 Element 3
                BFieldElement::new(11029368221459961736),
                BFieldElement::new(2601431810170510531),
                BFieldElement::new(3845091993529784163),
                BFieldElement::new(18440963282863373173),
                BFieldElement::new(15782363319704900162),
                // Merkle Authentication Path 1 Element 4
                BFieldElement::new(10193657868364591231),
                BFieldElement::new(10099674955292945516),
                BFieldElement::new(11861368391420694868),
                BFieldElement::new(12281343418175235418),
                BFieldElement::new(4979963636183136673),
                // Merkle Authentication Path 1 Element 5
                BFieldElement::new(239086846863014618),
                BFieldElement::new(18353654918351264251),
                BFieldElement::new(1162413056004073118),
                BFieldElement::new(63172233802162855),
                BFieldElement::new(15287652336563130555),
                // Merkle Authentication Path 2
                // Merkle Authentication Path 2 Element 0
                BFieldElement::new(4481571126490316833),
                BFieldElement::new(8911895412157369567),
                BFieldElement::new(5835492500982839536),
                BFieldElement::new(7582358620718112504),
                BFieldElement::new(17844368221186872833),
                // Merkle Authentication Path 2 Element 1
                BFieldElement::new(14881877338661058963),
                BFieldElement::new(13193566745649419854),
                BFieldElement::new(6162692737252551562),
                BFieldElement::new(11371203176785325596),
                BFieldElement::new(9217246242682535563),
                // Merkle Authentication Path 2 Element 2
                BFieldElement::new(13364374456634379783),
                BFieldElement::new(11904780360815341732),
                BFieldElement::new(13838542444368435771),
                BFieldElement::new(3920552087776628004),
                BFieldElement::new(11527431398195960804),
                // Merkle Authentication Path 2 Element 3
                BFieldElement::new(1435791031511559365),
                BFieldElement::new(15545210664684920678),
                BFieldElement::new(3431133792584929176),
                BFieldElement::new(8726944733794952298),
                BFieldElement::new(16054902179813715844),
                // Merkle Authentication Path 2 Element 4
                BFieldElement::new(6120454613508763402),
                BFieldElement::new(13046522894794631380),
                BFieldElement::new(12811925518855679797),
                BFieldElement::new(17271969057789657726),
                BFieldElement::new(9660251638518579939),
                // Merkle Authentication Path 2 Element 5
                BFieldElement::new(15982248888191274947),
                BFieldElement::new(16924250102716460133),
                BFieldElement::new(10777256019074274502),
                BFieldElement::new(5171550821485636583),
                BFieldElement::new(1372154037340399671),
            ],
        );

        for state in trace.iter() {
            println!("{}", state);
        }
        if let Some(e) = err {
            panic!("The VM encountered an error: {}", e);
        }

        // check for graceful termination
        let last_state = trace.get(trace.len() - 2).unwrap();
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
        let last_state = trace.get(trace.len() - 2).unwrap();
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

        let (trace, _out, _err) = program.run_with_input(&[], &[]);

        println!("{}", program);
        for state in trace.iter() {
            println!("{}", state);
        }

        let last_state = trace.last().unwrap();
        assert_eq!(BFieldElement::new(21), last_state.op_stack.st(ST0));
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
        let (trace, out, _err) =
            program.run_with_input(&[BFieldElement::new(42), BFieldElement::new(56)], &[]);

        println!("{}", program);
        for state in trace.iter() {
            println!("{}", state);
        }

        let expected = BFieldElement::new(14);
        let actual = *out.last().unwrap();
        assert_eq!(expected, actual);
    }

    #[test]
    #[ignore]
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
