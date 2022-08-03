use super::error::{vm_fail, InstructionError::*};
use super::instruction::{AnInstruction::*, Instruction};
use super::op_stack::OpStack;
use super::ord_n::{Ord16::*, Ord6::*};
use super::stdio::InputStream;
use super::table::{hash_table, instruction_table, jump_stack_table, op_stack_table, u32_op_table};
use super::table::{processor_table, ram_table};
use super::vm::Program;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::other;
use crate::shared_math::rescue_prime_xlix::RescuePrimeXlix;
use crate::shared_math::stark::triton::error::vm_err;
use crate::shared_math::stark::triton::table::base_matrix::ProcessorMatrixRow;
use crate::shared_math::traits::{IdentityValues, Inverse};
use crate::shared_math::x_field_element::XFieldElement;
use std::collections::HashMap;
use std::convert::TryInto;
use std::error::Error;
use std::fmt::Display;

type BWord = BFieldElement;
type XWord = XFieldElement;

/// The number of `BFieldElement`s in a Rescue-Prime digest for Triton VM.
pub const DIGEST_LEN: usize = 6;

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
    ram: HashMap<BWord, BWord>,

    /// 3. **Op-stack memory**, which stores the part of the operational stack
    ///    that is not represented explicitly by the operational stack registers
    ///
    ///    *(An implementation detail: We keep the entire stack in one `Vec<>`.)*
    op_stack: OpStack,

    /// 4. Jump-stack memory, which stores the entire jump stack
    jump_stack: Vec<(BWord, BWord)>,

    ///
    /// Registers
    ///
    /// Number of cycles the program has been running for
    pub cycle_count: u32,

    /// Current instruction's address in program memory
    pub instruction_pointer: usize,
}

#[derive(Debug, PartialEq)]
pub enum VMOutput {
    /// Trace output from `write_io`
    WriteOutputSymbol(BWord),

    /// Trace of state registers for hash coprocessor table
    ///
    /// One row per round in the XLIX permutation
    XlixTrace(Vec<[BWord; hash_table::BASE_WIDTH]>),

    /// Trace of u32 operations for u32 op table
    ///
    /// One row per defined bit
    U32OpTrace(Vec<[BWord; u32_op_table::BASE_WIDTH]>),
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
        rescue_prime: &RescuePrimeXlix<STATE_REGISTER_COUNT>,
    ) -> Result<(VMState<'pgm>, Option<VMOutput>), Box<dyn Error>>
    where
        In: InputStream,
    {
        let mut next_state = self.clone();
        next_state
            .step_mut(stdin, secret_in, rescue_prime)
            .map(|vm_output| (next_state, vm_output))
    }

    pub fn derive_helper_variables(&self) -> [BWord; HV_REGISTER_COUNT] {
        let mut hvs = [0.into(); HV_REGISTER_COUNT];

        let current_instruction = self.current_instruction();
        if current_instruction.is_err() {
            return hvs;
        }
        let current_instruction = current_instruction.unwrap();

        // Helps preventing OpStack Underflow
        match current_instruction {
            Pop | Skiz | Assert | Add | Mul | Eq | Lt | And | Xor | XbMul | WriteIo => {
                hvs[3] = (self.op_stack.osp() - 16.into()).inverse()
            }
            _ => (),
        }

        match current_instruction {
            // For instructions making use of indicator polynomials, i.e., `dup` and `swap`.
            // Sets the corresponding helper variables to the bit-decomposed argument such that the
            // indicator polynomials of the AIR actually evaluate to 0.
            Dup(arg) | Swap(arg) => {
                let arg_val: u64 = arg.into();
                hvs[0] = BWord::new(arg_val % 2);
                hvs[1] = BWord::new((arg_val >> 1) % 2);
                hvs[2] = BWord::new((arg_val >> 2) % 2);
                hvs[3] = BWord::new((arg_val >> 3) % 2);
            }
            Skiz => {
                let nia = self.nia().value();
                hvs[0] = BWord::new(nia % 2);
                hvs[1] = BWord::new(nia / 2);
                let st0 = self.op_stack.safe_peek(ST0);
                if !st0.is_zero() {
                    hvs[2] = st0.inverse();
                }
            }
            DivineSibling => {
                let node_index = self.op_stack.safe_peek(ST12).value();
                // lsb = least significant bit
                let node_index_lsb = node_index % 2;
                let node_index_msbs = node_index / 2;
                // set hv registers to correct decomposition of node_index
                hvs[0] = BWord::new(node_index_lsb);
                hvs[1] = BWord::new(node_index_msbs);
            }
            Split => {
                let elem = self.op_stack.safe_peek(ST0);
                let n: u64 = elem.value();
                let lo = BWord::new(n & 0xffff_ffff);
                let hi = BWord::new(n >> 32);
                if !lo.is_zero() {
                    let max_val_of_hi = BWord::new(2_u64.pow(32) - 1);
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
                hvs[0] = 1.into();
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
        rescue_prime: &RescuePrimeXlix<STATE_REGISTER_COUNT>,
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
                let pair = (o_plus_2.into(), addr);
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
                let mut state = [BWord::new(0); STATE_REGISTER_COUNT];
                for i in 0..2 * DIGEST_LEN {
                    state[i] = self.op_stack.pop()?;
                }

                let hash_trace = rescue_prime.rescue_xlix_permutation_trace(&mut state);
                vm_output = Some(VMOutput::XlixTrace(hash_trace));

                for _ in 0..DIGEST_LEN {
                    self.op_stack.push(0.into());
                }

                for i in (0..DIGEST_LEN).rev() {
                    self.op_stack.push(state[i]);
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
                let lo = BWord::new(n & 0xffff_ffff);
                let hi = BWord::new(n >> 32);
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
                self.op_stack.push((lhs & rhs).into());
                let trace = self.u32_op_trace(lhs, rhs);
                vm_output = Some(VMOutput::U32OpTrace(trace));
                self.instruction_pointer += 1;
            }

            Xor => {
                let lhs: u32 = self.op_stack.pop_u32()?;
                let rhs: u32 = self.op_stack.pop_u32()?;
                self.op_stack.push((lhs ^ rhs).into());
                let trace = self.u32_op_trace(lhs, rhs);
                vm_output = Some(VMOutput::U32OpTrace(trace));
                self.instruction_pointer += 1;
            }

            Reverse => {
                let elem: u32 = self.op_stack.pop_u32()?;
                self.op_stack.push(elem.reverse_bits().into());

                // for instruction `reverse`, the Uint32 Table's RHS is (arbitrarily) set to 0
                let trace = self.u32_op_trace(elem, 0);
                vm_output = Some(VMOutput::U32OpTrace(trace));
                self.instruction_pointer += 1;
            }

            Div => {
                let denom: u32 = self.op_stack.pop_u32()?;
                let numerator: u32 = self.op_stack.pop_u32()?;
                let (quot, rem) = other::div_rem(numerator, denom);
                self.op_stack.push(quot.into());
                self.op_stack.push(rem.into());
                let trace = self.u32_op_trace(denom, numerator);
                vm_output = Some(VMOutput::U32OpTrace(trace));
                self.instruction_pointer += 1;
            }

            XxAdd => {
                let lhs: XWord = self.op_stack.pop_x()?;
                let rhs: XWord = self.op_stack.safe_peek_x();
                self.op_stack.push_x(lhs + rhs);
                self.instruction_pointer += 1;
            }

            XxMul => {
                let lhs: XWord = self.op_stack.pop_x()?;
                let rhs: XWord = self.op_stack.safe_peek_x();
                self.op_stack.push_x(lhs * rhs);
                self.instruction_pointer += 1;
            }

            XInvert => {
                let elem: XWord = self.op_stack.pop_x()?;
                self.op_stack.push_x(elem.inverse());
                self.instruction_pointer += 1;
            }

            XbMul => {
                let lhs: BWord = self.op_stack.pop()?;
                let rhs: XWord = self.op_stack.pop_x()?;
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
        let clk = self.cycle_count.into();
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
            *self.ram.get(&st1).unwrap_or(&BWord::new(0)),
        ]
    }

    pub fn to_op_stack_row(
        &self,
        current_instruction: Instruction,
    ) -> [BFieldElement; op_stack_table::BASE_WIDTH] {
        let clk = self.cycle_count.into();
        let ci = current_instruction.opcode_b();
        let osp = self.op_stack.osp();
        let osv = self.op_stack.osv();

        // clk, ci, osv, osp
        [clk, ci, osv, osp]
    }

    pub fn to_ram_row(&self) -> [BFieldElement; ram_table::BASE_WIDTH] {
        let clk = self.cycle_count.into();
        let ramp = self.op_stack.st(ST1);
        let ramv = *self.ram.get(&ramp).unwrap_or(&BWord::new(0));

        // placeholder value – actual value only known after sorting the RAM Table
        let inverse_of_ramp_diff = BWord::new(0);

        [clk, ramp, ramv, inverse_of_ramp_diff]
    }

    pub fn to_jump_stack_row(
        &self,
        current_instruction: Instruction,
    ) -> [BFieldElement; jump_stack_table::BASE_WIDTH] {
        let clk = self.cycle_count.into();
        let ci = current_instruction.opcode_b();

        [clk, ci, self.jsp(), self.jso(), self.jsd()]
    }

    pub fn u32_op_trace(
        &self,
        mut lhs: u32,
        mut rhs: u32,
    ) -> Vec<[BFieldElement; u32_op_table::BASE_WIDTH]> {
        let mut rows = vec![];
        let mut idc = 1.into();
        let zero = 0.into();

        while lhs > 0 || rhs > 0 {
            let row = [
                idc,
                lhs.into(),
                rhs.into(),
                Self::lt(lhs, rhs),
                (lhs & rhs).into(),
                (lhs ^ rhs).into(),
                lhs.reverse_bits().into(),
            ];
            rows.push(row);
            lhs >>= 1;
            rhs >>= 1;
            idc = zero;
        }

        rows
    }

    fn lt(lhs: u32, rhs: u32) -> BWord {
        if rhs < lhs {
            1.into()
        } else {
            0.into()
        }
    }

    fn eq(lhs: BWord, rhs: BWord) -> BWord {
        if lhs == rhs {
            1.into()
        } else {
            0.into()
        }
    }

    fn nia(&self) -> BWord {
        self.current_instruction()
            .map(|curr_instr| {
                curr_instr.arg().unwrap_or_else(|| {
                    self.next_instruction()
                        .map(|next_instr| next_instr.opcode_b())
                        .unwrap_or_else(|_| 0.into())
                })
            })
            .unwrap_or_else(|_| 0.into())
    }

    /// Jump-stack pointer
    fn jsp(&self) -> BWord {
        BWord::new(self.jump_stack.len() as u64)
    }

    /// Jump-stack origin
    fn jso(&self) -> BWord {
        self.jump_stack
            .last()
            .map(|(o, _d)| *o)
            .unwrap_or_else(|| 0.into())
    }

    /// Jump-stack destination
    fn jsd(&self) -> BWord {
        self.jump_stack
            .last()
            .map(|(_o, d)| *d)
            .unwrap_or_else(|| 0.into())
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

    fn jump_stack_pop(&mut self) -> Result<(BWord, BWord), Box<dyn Error>> {
        self.jump_stack
            .pop()
            .ok_or_else(|| vm_fail(JumpStackTooShallow))
    }

    fn jump_stack_peek(&mut self) -> Result<(BWord, BWord), Box<dyn Error>> {
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
        for i in 0..DIGEST_LEN {
            // Safe as long as 2 * DIGEST_LEN <= OP_STACK_REG_COUNT
            let lhs = i.try_into().unwrap();
            let rhs = (i + DIGEST_LEN).try_into().unwrap();

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
        let known_digest: [BWord; DIGEST_LEN] = [
            self.op_stack.pop()?,
            self.op_stack.pop()?,
            self.op_stack.pop()?,
            self.op_stack.pop()?,
            self.op_stack.pop()?,
            self.op_stack.pop()?,
        ];

        for _ in 0..DIGEST_LEN {
            self.op_stack.pop()?;
        }

        let node_index: u32 = self.op_stack.pop()?.try_into()?;

        let sibling_digest = [
            secret_in.read_elem()?,
            secret_in.read_elem()?,
            secret_in.read_elem()?,
            secret_in.read_elem()?,
            secret_in.read_elem()?,
            secret_in.read_elem()?,
        ];

        // lsb = least significant bit
        let node_index_lsb = node_index % 2;
        let node_index_msbs = node_index / 2;

        let known_digest_is_left_node = node_index_lsb == 0;
        let (first, second) = if known_digest_is_left_node {
            // move sibling digest to rhs
            (known_digest, sibling_digest)
        } else {
            // move sibling digest to lhs
            (sibling_digest, known_digest)
        };

        // Push the new node index and 2 digests to stack

        self.op_stack.push(node_index_msbs.into());

        for word in second.iter().rev() {
            self.op_stack.push(*word);
        }

        for word in first.iter().rev() {
            self.op_stack.push(*word);
        }

        Ok(())
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
    use super::super::op_stack::OP_STACK_REG_COUNT;
    use super::*;
    use crate::shared_math::stark::triton::instruction::sample_programs;

    // Property: All instructions increase the cycle count by 1.
    // Property: Most instructions increase the instruction pointer by 1.

    #[test]
    fn tvm_op_stack_big_enough_test() {
        assert!(
            DIGEST_LEN <= OP_STACK_REG_COUNT,
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
        assert_eq!(BWord::ring_zero(), last_state.op_stack.safe_peek(ST0));

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
        let (trace, _out, err) = program.run_with_input(&[11.into()], &[]);

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
    fn run_tvm_mt_ap_verify_test() {
        let program = Program::from_code(sample_programs::MT_AP_VERIFY).unwrap();
        println!("Successfully parsed the program.");
        let (trace, _out, err) = program.run_with_input(
            &[
                // Merkle root
                BWord::new(2661879877493968030),
                BWord::new(8411897398996365015),
                BWord::new(11724741215505059774),
                BWord::new(10869446635029787183),
                BWord::new(3194712170375950680),
                BWord::new(5350293309391779043),
                // node index 64, leaf index 0
                BWord::new(64),
                // value of leaf with index 0
                BWord::new(17),
                BWord::new(22),
                BWord::new(19),
                // node index 92, leaf index 28
                BWord::new(92),
                // value of leaf with index 28
                BWord::new(45),
                BWord::new(50),
                BWord::new(47),
                // node index 119, leaf index 55
                BWord::new(119),
                // value of leaf with node 55
                BWord::new(72),
                BWord::new(77),
                BWord::new(74),
            ],
            &[
                // Merkle Authentication Path 0
                // Merkle Authentication Path 0 Element 0
                BWord::new(7433611961471031299),
                BWord::new(10663067815302282105),
                BWord::new(11189271637150912214),
                BWord::new(6731301558776007763),
                BWord::new(12404371806864851196),
                BWord::new(4001338418445453888),
                // Merkle Authentication Path 0 Element 1
                BWord::new(15447170459020364568),
                BWord::new(13311771520545451802),
                BWord::new(4832613912751814227),
                BWord::new(16118512681346800136),
                BWord::new(11903034542985100612),
                BWord::new(8722502554058837902),
                // Merkle Authentication Path 0 Element 2
                BWord::new(927166763011592563),
                BWord::new(1017721141586418898),
                BWord::new(14149577177119432718),
                BWord::new(11112535232426569259),
                BWord::new(6770923340167310082),
                BWord::new(6635263622554958787),
                // Merkle Authentication Path 0 Element 3
                BWord::new(11997402720255929816),
                BWord::new(7083119985125877931),
                BWord::new(3583918993470398367),
                BWord::new(12665589384229632447),
                BWord::new(4869924221127107207),
                BWord::new(2205377658620204174),
                // Merkle Authentication Path 0 Element 4
                BWord::new(4108830855587634814),
                BWord::new(11363551275926927759),
                BWord::new(8897943612193465442),
                BWord::new(18175199505544299571),
                BWord::new(5933081913383911549),
                BWord::new(11963915458141697161),
                // Merkle Authentication Path 0 Element 5
                BWord::new(239086846863014618),
                BWord::new(18353654918351264251),
                BWord::new(1162413056004073118),
                BWord::new(63172233802162855),
                BWord::new(15287652336563130555),
                BWord::new(6615623432715966135),
                // Merkle Authentication Path 1
                // Merkle Authentication Path 1 Element 0
                BWord::new(9199975892950715767),
                BWord::new(18392437377232084500),
                BWord::new(7389509101855274876),
                BWord::new(13193152724141987884),
                BWord::new(12764531673520060724),
                BWord::new(16294749329463136349),
                // Merkle Authentication Path 1 Element 1
                BWord::new(13265185672483741593),
                BWord::new(4801722111881156327),
                BWord::new(297253697970945484),
                BWord::new(8955967409623509220),
                BWord::new(10440367450900769517),
                BWord::new(10816277785135288164),
                // Merkle Authentication Path 1 Element 2
                BWord::new(3378320220263195325),
                BWord::new(17709073937843856976),
                BWord::new(3737595776877974498),
                BWord::new(1050267233733511018),
                BWord::new(18417031760560110797),
                BWord::new(13081044610877517462),
                // Merkle Authentication Path 1 Element 3
                BWord::new(11029368221459961736),
                BWord::new(2601431810170510531),
                BWord::new(3845091993529784163),
                BWord::new(18440963282863373173),
                BWord::new(15782363319704900162),
                BWord::new(5649168943621408804),
                // Merkle Authentication Path 1 Element 4
                BWord::new(10193657868364591231),
                BWord::new(10099674955292945516),
                BWord::new(11861368391420694868),
                BWord::new(12281343418175235418),
                BWord::new(4979963636183136673),
                BWord::new(18369998622044683261),
                // Merkle Authentication Path 1 Element 5
                BWord::new(239086846863014618),
                BWord::new(18353654918351264251),
                BWord::new(1162413056004073118),
                BWord::new(63172233802162855),
                BWord::new(15287652336563130555),
                BWord::new(6615623432715966135),
                // Merkle Authentication Path 2
                // Merkle Authentication Path 2 Element 0
                BWord::new(4481571126490316833),
                BWord::new(8911895412157369567),
                BWord::new(5835492500982839536),
                BWord::new(7582358620718112504),
                BWord::new(17844368221186872833),
                BWord::new(17133435817149957052),
                // Merkle Authentication Path 2 Element 1
                BWord::new(14881877338661058963),
                BWord::new(13193566745649419854),
                BWord::new(6162692737252551562),
                BWord::new(11371203176785325596),
                BWord::new(9217246242682535563),
                BWord::new(6324769433519982629),
                // Merkle Authentication Path 2 Element 2
                BWord::new(13364374456634379783),
                BWord::new(11904780360815341732),
                BWord::new(13838542444368435771),
                BWord::new(3920552087776628004),
                BWord::new(11527431398195960804),
                BWord::new(9866681327490442483),
                // Merkle Authentication Path 2 Element 3
                BWord::new(1435791031511559365),
                BWord::new(15545210664684920678),
                BWord::new(3431133792584929176),
                BWord::new(8726944733794952298),
                BWord::new(16054902179813715844),
                BWord::new(4961109724612073173),
                // Merkle Authentication Path 2 Element 4
                BWord::new(6120454613508763402),
                BWord::new(13046522894794631380),
                BWord::new(12811925518855679797),
                BWord::new(17271969057789657726),
                BWord::new(9660251638518579939),
                BWord::new(4505728643179916723),
                // Merkle Authentication Path 2 Element 5
                BWord::new(15982248888191274947),
                BWord::new(16924250102716460133),
                BWord::new(10777256019074274502),
                BWord::new(5171550821485636583),
                BWord::new(1372154037340399671),
                BWord::new(13169355684141832888),
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
        let (trace, out, err) =
            program.run_with_input(&[7.into(), 2.into(), 1.into(), 3.into(), 4.into()], &[]);
        assert_eq!(out[0], 4.into());
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
        assert_eq!(BWord::ring_zero(), last_state.op_stack.st(ST0));
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
        assert_eq!(BWord::new(21), last_state.op_stack.st(ST0));
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
        assert_eq!(BWord::new(21), last_state.op_stack.st(ST0));
    }

    #[test]
    fn run_tvm_gcd_test() {
        let code = sample_programs::GCD_X_Y;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);
        let (trace, out, _err) = program.run_with_input(&[42.into(), 56.into()], &[]);

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
