use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt::Display;

use anyhow::Result;
use itertools::Itertools;
use ndarray::Array1;
use num_traits::One;
use num_traits::Zero;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::other::log_2_floor;
use twenty_first::shared_math::rescue_prime_digest::DIGEST_LENGTH;
use twenty_first::shared_math::rescue_prime_regular::RescuePrimeRegular;
use twenty_first::shared_math::rescue_prime_regular::RescuePrimeRegularState;
use twenty_first::shared_math::rescue_prime_regular::NUM_ROUNDS;
use twenty_first::shared_math::rescue_prime_regular::RATE;
use twenty_first::shared_math::rescue_prime_regular::STATE_SIZE;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::algebraic_hasher::Domain;

use triton_opcodes::instruction::AnInstruction::*;
use triton_opcodes::instruction::Instruction;
use triton_opcodes::ord_n::Ord16;
use triton_opcodes::ord_n::Ord16::*;
use triton_opcodes::ord_n::Ord8;
use triton_opcodes::program::Program;

use crate::error::vm_err;
use crate::error::vm_fail;
use crate::error::InstructionError::*;
use crate::op_stack::OpStack;
use crate::table::processor_table;
use crate::table::processor_table::ProcessorTraceRow;
use crate::table::table_column::BaseTableColumn;
use crate::table::table_column::ProcessorBaseTableColumn;

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
    pub program: &'pgm [Instruction],

    /// 2. **Random-access memory**, to which the VM can read and write field elements
    pub ram: HashMap<BFieldElement, BFieldElement>,

    /// 3. **Op-stack memory**, which stores the part of the operational stack
    ///    that is not represented explicitly by the operational stack registers
    ///
    ///    *(An implementation detail: We keep the entire stack in one `Vec<>`.)*
    pub op_stack: OpStack,

    /// 4. Jump-stack memory, which stores the entire jump stack
    pub jump_stack: Vec<(BFieldElement, BFieldElement)>,

    ///
    /// Registers
    ///
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
    pub sponge_state: [BFieldElement; STATE_SIZE],
}

#[derive(Debug, PartialEq, Eq)]
pub enum VMOutput {
    /// Trace output from `write_io`
    WriteOutputSymbol(BFieldElement),

    /// Trace of the state registers for hash coprocessor table when executing instruction `hash`
    /// or any of the Sponge instructions `absorb_init`, `absorb`, `squeeze`.
    /// One row per round in the XLIX permutation.
    XlixTrace(
        Instruction,
        Box<[[BFieldElement; STATE_SIZE]; 1 + NUM_ROUNDS]>,
    ),

    /// Executed u32 instruction as well as its left-hand side and right-hand side
    U32TableEntries(Vec<(Instruction, BFieldElement, BFieldElement)>),
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
    pub fn step(
        &self,
        stdin: &mut Vec<BFieldElement>,
        secret_in: &mut Vec<BFieldElement>,
    ) -> Result<(VMState<'pgm>, Option<VMOutput>)> {
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

        // if current instruction shrinks the stack
        if matches!(
            current_instruction,
            Pop | Skiz | Assert | WriteIo | Add | Mul | Eq | XbMul | Lt | And | Xor | Pow
        ) {
            hvs[3] = (self.op_stack.osp() - BFieldElement::new(16)).inverse_or_zero();
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
                hvs[2] = st0.inverse_or_zero();
            }
            DivineSibling => {
                let node_index = self.op_stack.safe_peek(ST10).value();
                // set hv0 register to least significant bit of st10
                hvs[0] = BFieldElement::new(node_index % 2);
            }
            Split => {
                let elem = self.op_stack.safe_peek(ST0);
                let n: u64 = elem.value();
                let lo = BFieldElement::new(n & 0xffff_ffff);
                let hi = BFieldElement::new(n >> 32);
                if !lo.is_zero() {
                    let max_val_of_hi = BFieldElement::new(2_u64.pow(32) - 1);
                    hvs[0] = (hi - max_val_of_hi).inverse_or_zero();
                }
            }
            Eq => {
                let lhs = self.op_stack.safe_peek(ST0);
                let rhs = self.op_stack.safe_peek(ST1);
                hvs[0] = (rhs - lhs).inverse_or_zero();
            }
            _ => (),
        }

        hvs
    }

    /// Perform the state transition as a mutable operation on `self`.
    pub fn step_mut(
        &mut self,
        stdin: &mut Vec<BFieldElement>,
        secret_in: &mut Vec<BFieldElement>,
    ) -> Result<Option<VMOutput>> {
        // All instructions increase the cycle count
        self.cycle_count += 1;
        let mut vm_output = None;
        self.previous_instruction = match self.current_instruction() {
            Ok(instruction) => instruction.opcode_b(),
            // trying to read past the end of the program doesn't change the previous instruction
            Err(_) => self.previous_instruction,
        };

        match self.current_instruction()? {
            Pop => {
                self.op_stack.pop()?;
                self.instruction_pointer += 1;
            }

            Push(arg) => {
                self.op_stack.push(arg);
                self.instruction_pointer += 2;
            }

            Divine(_) => {
                let elem = secret_in.remove(0);
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
                self.instruction_pointer += 1;
            }

            ReadMem => {
                let ramp = self.op_stack.safe_peek(ST1);
                let ramv = self.memory_get(&ramp);
                self.op_stack.pop()?;
                self.op_stack.push(ramv);
                self.ramp = ramp.value();
                self.instruction_pointer += 1;
            }

            WriteMem => {
                let ramp = self.op_stack.safe_peek(ST1);
                let ramv = self.op_stack.safe_peek(ST0);
                self.ramp = ramp.value();
                self.ram.insert(ramp, ramv);
                self.instruction_pointer += 1;
            }

            Hash => {
                let to_hash = self.op_stack.pop_n::<{ 2 * DIGEST_LENGTH }>()?;
                let mut hash_input = RescuePrimeRegularState::new(Domain::FixedLength).state;
                hash_input[..2 * DIGEST_LENGTH].copy_from_slice(&to_hash);
                let xlix_trace = RescuePrimeRegular::trace(hash_input);
                let hash_output = &xlix_trace[xlix_trace.len() - 1][0..DIGEST_LENGTH];

                for i in (0..DIGEST_LENGTH).rev() {
                    self.op_stack.push(hash_output[i]);
                }
                for _ in 0..DIGEST_LENGTH {
                    self.op_stack.push(BFieldElement::zero());
                }

                vm_output = Some(VMOutput::XlixTrace(Hash, Box::new(xlix_trace)));
                self.instruction_pointer += 1;
            }

            AbsorbInit | Absorb => {
                // fetch top elements but don't alter the stack
                let to_absorb = self.op_stack.pop_n::<{ RATE }>()?;
                for i in (0..RATE).rev() {
                    self.op_stack.push(to_absorb[i]);
                }

                if self.current_instruction()? == AbsorbInit {
                    self.sponge_state = RescuePrimeRegularState::new(Domain::VariableLength).state;
                }
                self.sponge_state[..RATE]
                    .iter_mut()
                    .zip_eq(to_absorb.iter())
                    .for_each(|(sponge_state_element, &to_absorb_element)| {
                        *sponge_state_element += to_absorb_element;
                    });
                let xlix_trace = RescuePrimeRegular::trace(self.sponge_state);
                self.sponge_state = xlix_trace.last().unwrap().to_owned();

                vm_output = Some(VMOutput::XlixTrace(
                    self.current_instruction()?,
                    Box::new(xlix_trace),
                ));
                self.instruction_pointer += 1;
            }

            Squeeze => {
                let _ = self.op_stack.pop_n::<{ RATE }>()?;
                for i in (0..RATE).rev() {
                    self.op_stack.push(self.sponge_state[i]);
                }
                let xlix_trace = RescuePrimeRegular::trace(self.sponge_state);
                self.sponge_state = xlix_trace.last().unwrap().to_owned();

                vm_output = Some(VMOutput::XlixTrace(Squeeze, Box::new(xlix_trace)));
                self.instruction_pointer += 1;
            }

            DivineSibling => {
                self.divine_sibling(secret_in)?;
                self.instruction_pointer += 1;
            }

            SwapDigest => {
                let digest_upper = self.op_stack.pop_n::<{ DIGEST_LENGTH }>()?;
                let digest_lower = self.op_stack.pop_n::<{ DIGEST_LENGTH }>()?;
                for i in (0..DIGEST_LENGTH).rev() {
                    self.op_stack.push(digest_upper[i]);
                }
                for i in (0..DIGEST_LENGTH).rev() {
                    self.op_stack.push(digest_lower[i]);
                }
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

            Eq => {
                let lhs = self.op_stack.pop()?;
                let rhs = self.op_stack.pop()?;
                self.op_stack.push(Self::eq(lhs, rhs));
                self.instruction_pointer += 1;
            }

            Split => {
                let elem = self.op_stack.pop()?;
                let lo = BFieldElement::new(elem.value() & 0xffff_ffff);
                let hi = BFieldElement::new(elem.value() >> 32);
                self.op_stack.push(hi);
                self.op_stack.push(lo);
                self.instruction_pointer += 1;
                let u32_table_entry = (Instruction::Split, lo, hi);
                vm_output = Some(VMOutput::U32TableEntries(vec![u32_table_entry]));
            }

            Lt => {
                let lhs = self.op_stack.pop_u32()?;
                let rhs = self.op_stack.pop_u32()?;
                let lt = BFieldElement::new((lhs < rhs) as u64);
                self.op_stack.push(lt);
                self.instruction_pointer += 1;
                let u32_table_entry = (
                    Instruction::Lt,
                    BFieldElement::new(lhs as u64),
                    BFieldElement::new(rhs as u64),
                );
                vm_output = Some(VMOutput::U32TableEntries(vec![u32_table_entry]));
            }

            And => {
                let lhs = self.op_stack.pop_u32()?;
                let rhs = self.op_stack.pop_u32()?;
                let and = BFieldElement::new((lhs & rhs) as u64);
                self.op_stack.push(and);
                self.instruction_pointer += 1;
                let u32_table_entry = (
                    Instruction::And,
                    BFieldElement::new(lhs as u64),
                    BFieldElement::new(rhs as u64),
                );
                vm_output = Some(VMOutput::U32TableEntries(vec![u32_table_entry]));
            }

            Xor => {
                let lhs = self.op_stack.pop_u32()?;
                let rhs = self.op_stack.pop_u32()?;
                let xor = BFieldElement::new((lhs ^ rhs) as u64);
                self.op_stack.push(xor);
                self.instruction_pointer += 1;
                let u32_table_entry = (
                    Instruction::Xor,
                    BFieldElement::new(lhs as u64),
                    BFieldElement::new(rhs as u64),
                );
                vm_output = Some(VMOutput::U32TableEntries(vec![u32_table_entry]));
            }

            Log2Floor => {
                let lhs = self.op_stack.pop_u32()?;
                if lhs.is_zero() {
                    return vm_err(LogarithmOfZero);
                }
                let l2f = BFieldElement::new(log_2_floor(lhs as u128));
                self.op_stack.push(l2f);
                self.instruction_pointer += 1;
                let u32_table_entry = (
                    Instruction::Log2Floor,
                    BFieldElement::new(lhs as u64),
                    BFieldElement::zero(),
                );
                vm_output = Some(VMOutput::U32TableEntries(vec![u32_table_entry]));
            }

            Pow => {
                let lhs = self.op_stack.pop_u32()?;
                let rhs = self.op_stack.pop_u32()?;
                let pow = BFieldElement::new(lhs as u64).mod_pow(rhs as u64);
                self.op_stack.push(pow);
                self.instruction_pointer += 1;
                let u32_table_entry = (
                    Instruction::Pow,
                    BFieldElement::new(lhs as u64),
                    BFieldElement::new(rhs as u64),
                );
                vm_output = Some(VMOutput::U32TableEntries(vec![u32_table_entry]));
            }

            Div => {
                let numer = self.op_stack.pop_u32()?;
                let denom = self.op_stack.pop_u32()?;
                if denom.is_zero() {
                    return vm_err(DivisionByZero);
                }
                let quot = BFieldElement::new((numer / denom) as u64);
                let rem = BFieldElement::new((numer % denom) as u64);
                self.op_stack.push(quot);
                self.op_stack.push(rem);
                self.instruction_pointer += 1;
                let u32_table_entry_0 = (Instruction::Lt, rem, BFieldElement::new(denom as u64));
                let u32_table_entry_1 =
                    (Instruction::Split, BFieldElement::new(numer as u64), quot);
                vm_output = Some(VMOutput::U32TableEntries(vec![
                    u32_table_entry_0,
                    u32_table_entry_1,
                ]));
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
                if elem.is_zero() {
                    return vm_err(InverseOfZero);
                }
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
                let in_elem = stdin.remove(0);
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

    pub fn to_processor_row(&self) -> Array1<BFieldElement> {
        use ProcessorBaseTableColumn::*;
        let mut row = Array1::zeros(processor_table::BASE_WIDTH);

        let current_instruction = self.current_instruction().unwrap_or(Nop);
        let hvs = self.derive_helper_variables();
        let ramp = self.ramp.into();

        row[CLK.base_table_index()] = BFieldElement::new(self.cycle_count as u64);
        row[PreviousInstruction.base_table_index()] = self.previous_instruction;
        row[IP.base_table_index()] = (self.instruction_pointer as u32).into();
        row[CI.base_table_index()] = current_instruction.opcode_b();
        row[NIA.base_table_index()] = self.nia();
        row[IB0.base_table_index()] = current_instruction.ib(Ord8::IB0);
        row[IB1.base_table_index()] = current_instruction.ib(Ord8::IB1);
        row[IB2.base_table_index()] = current_instruction.ib(Ord8::IB2);
        row[IB3.base_table_index()] = current_instruction.ib(Ord8::IB3);
        row[IB4.base_table_index()] = current_instruction.ib(Ord8::IB4);
        row[IB5.base_table_index()] = current_instruction.ib(Ord8::IB5);
        row[IB6.base_table_index()] = current_instruction.ib(Ord8::IB6);
        row[IB7.base_table_index()] = current_instruction.ib(Ord8::IB7);
        row[JSP.base_table_index()] = self.jsp();
        row[JSO.base_table_index()] = self.jso();
        row[JSD.base_table_index()] = self.jsd();
        row[ST0.base_table_index()] = self.op_stack.st(Ord16::ST0);
        row[ST1.base_table_index()] = self.op_stack.st(Ord16::ST1);
        row[ST2.base_table_index()] = self.op_stack.st(Ord16::ST2);
        row[ST3.base_table_index()] = self.op_stack.st(Ord16::ST3);
        row[ST4.base_table_index()] = self.op_stack.st(Ord16::ST4);
        row[ST5.base_table_index()] = self.op_stack.st(Ord16::ST5);
        row[ST6.base_table_index()] = self.op_stack.st(Ord16::ST6);
        row[ST7.base_table_index()] = self.op_stack.st(Ord16::ST7);
        row[ST8.base_table_index()] = self.op_stack.st(Ord16::ST8);
        row[ST9.base_table_index()] = self.op_stack.st(Ord16::ST9);
        row[ST10.base_table_index()] = self.op_stack.st(Ord16::ST10);
        row[ST11.base_table_index()] = self.op_stack.st(Ord16::ST11);
        row[ST12.base_table_index()] = self.op_stack.st(Ord16::ST12);
        row[ST13.base_table_index()] = self.op_stack.st(Ord16::ST13);
        row[ST14.base_table_index()] = self.op_stack.st(Ord16::ST14);
        row[ST15.base_table_index()] = self.op_stack.st(Ord16::ST15);
        row[OSP.base_table_index()] = self.op_stack.osp();
        row[OSV.base_table_index()] = self.op_stack.osv();
        row[HV0.base_table_index()] = hvs[0];
        row[HV1.base_table_index()] = hvs[1];
        row[HV2.base_table_index()] = hvs[2];
        row[HV3.base_table_index()] = hvs[3];
        row[RAMP.base_table_index()] = ramp;
        row[RAMV.base_table_index()] = self.memory_get(&ramp);

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

    pub fn current_instruction(&self) -> Result<Instruction> {
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
    pub fn next_instruction(&self) -> Result<Instruction> {
        let ci = self.current_instruction()?;
        let ci_size = ci.size();
        let ni_pointer = self.instruction_pointer + ci_size;
        self.program
            .get(ni_pointer)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow(ni_pointer)))
            .copied()
    }

    fn _next_next_instruction(&self) -> Result<Instruction> {
        let cur_size = self.current_instruction()?.size();
        let next_size = self.next_instruction()?.size();
        self.program
            .get(self.instruction_pointer + cur_size + next_size)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow(self.instruction_pointer)))
            .copied()
    }

    fn jump_stack_pop(&mut self) -> Result<(BFieldElement, BFieldElement)> {
        self.jump_stack
            .pop()
            .ok_or_else(|| vm_fail(JumpStackTooShallow))
    }

    fn jump_stack_peek(&mut self) -> Result<(BFieldElement, BFieldElement)> {
        self.jump_stack
            .last()
            .copied()
            .ok_or_else(|| vm_fail(JumpStackTooShallow))
    }

    fn memory_get(&self, mem_addr: &BFieldElement) -> BFieldElement {
        self.ram
            .get(mem_addr)
            .copied()
            .unwrap_or_else(BFieldElement::zero)
    }

    fn assert_vector(&self) -> bool {
        for i in 0..DIGEST_LENGTH {
            // Safe as long as 2 * DIGEST_LEN <= OP_STACK_REG_COUNT
            let lhs = i.try_into().expect("Digest element position (lhs)");
            let rhs = (i + DIGEST_LENGTH)
                .try_into()
                .expect("Digest element position (rhs)");

            if self.op_stack.safe_peek(lhs) != self.op_stack.safe_peek(rhs) {
                return false;
            }
        }
        true
    }

    pub fn read_word(&self) -> Result<Option<BFieldElement>> {
        let current_instruction = self.current_instruction()?;
        if matches!(current_instruction, ReadIo) {
            Ok(Some(self.op_stack.safe_peek(ST0)))
        } else {
            Ok(None)
        }
    }

    fn divine_sibling(&mut self, secret_in: &mut Vec<BFieldElement>) -> Result<()> {
        // st0-st4
        let _ = self.op_stack.pop_n::<{ DIGEST_LENGTH }>()?;

        // st5-st9
        let known_digest = self.op_stack.pop_n::<{ DIGEST_LENGTH }>()?;

        // st10
        let node_index_elem: BFieldElement = self.op_stack.pop()?;
        let node_index: u32 = node_index_elem
            .try_into()
            .unwrap_or_else(|_| panic!("{:?} is not a u32", node_index_elem));

        // nondeterministic guess, flipped
        let sibling_digest: [BFieldElement; DIGEST_LENGTH] = {
            let mut tmp = [
                secret_in.remove(0),
                secret_in.remove(0),
                secret_in.remove(0),
                secret_in.remove(0),
                secret_in.remove(0),
            ];
            tmp.reverse();
            tmp
        };

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
mod vm_state_tests {
    use itertools::Itertools;
    use twenty_first::shared_math::other::random_elements_array;
    use twenty_first::shared_math::rescue_prime_digest::Digest;
    use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;
    use twenty_first::util_types::merkle_tree::MerkleTree;
    use twenty_first::util_types::merkle_tree_maker::MerkleTreeMaker;

    use crate::op_stack::OP_STACK_REG_COUNT;
    use crate::shared_tests::{FIBONACCI_VIT, FIB_FIXED_7_LT};
    use crate::stark::Maker;
    use crate::vm::run;
    use crate::vm::triton_vm_tests::GCD_X_Y;

    use super::*;

    // Property: All instructions increase the cycle count by 1.
    // Property: Most instructions increase the instruction pointer by 1.

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn tvm_op_stack_big_enough_test() {
        assert!(
            DIGEST_LENGTH <= OP_STACK_REG_COUNT,
            "The OpStack must be large enough to hold a single Rescue-Prime digest"
        );
    }

    #[test]
    fn run_tvm_parse_pop_p_test() {
        let program = Program::from_code("push 1 push 1 add pop").unwrap();
        let (trace, _out, _err) = run(&program, vec![], vec![]);

        for state in trace.iter() {
            println!("{}", state);
        }
    }

    #[test]
    fn run_tvm_hello_world_1_test() {
        let code = "
            push 10
            push 33
            push 100
            push 108
            push 114
            push 111
            push 87
            push 32
            push 44
            push 111
            push 108
            push 108
            push 101
            push 72
        
            write_io write_io write_io write_io write_io write_io write_io
            write_io write_io write_io write_io write_io write_io write_io
        ";
        let program = Program::from_code(code).unwrap();
        let (trace, _out, _err) = run(&program, vec![], vec![]);

        let last_state = trace.last().unwrap();
        assert_eq!(BFieldElement::zero(), last_state.op_stack.safe_peek(ST0));

        println!("{}", last_state);
    }

    #[test]
    fn run_tvm_halt_then_do_stuff_test() {
        let halt_then_do_stuff = "halt push 1 push 2 add invert write_io";
        let program = Program::from_code(halt_then_do_stuff).unwrap();
        let (trace, _out, err) = run(&program, vec![], vec![]);

        for state in trace.iter() {
            println!("{}", state);
        }
        if let Some(e) = err {
            println!("Error: {}", e);
        }

        // check for graceful termination
        let last_state = trace.last().unwrap();
        assert_eq!(last_state.current_instruction().unwrap(), Halt);
    }

    #[test]
    fn run_tvm_basic_ram_read_write_test() {
        let basic_ram_read_write_code = "
            push  5 push  6 write_mem pop pop 
            push 15 push 16 write_mem pop pop 
            push  5 push  0 read_mem  pop pop 
            push 15 push  0 read_mem  pop pop 
            push  5 push  7 write_mem pop pop 
            push 15 push  0 read_mem 
            push  5 push  0 read_mem 
            halt
            ";
        let program = Program::from_code(basic_ram_read_write_code).unwrap();
        let (trace, _out, err) = run(&program, vec![], vec![]);
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
        let edgy_ram_writes_code = "
            write_mem                          // this should write 0 to address 0
            push 5 swap2 push 3 swap2 pop pop  // stack is now of length 16 again
            write_mem                          // this should write 3 to address 5
            swap2 read_mem                     // stack's top should now be 3, 5, 3, 0, 0, …
            halt
        ";
        let program = Program::from_code(edgy_ram_writes_code).unwrap();
        let (trace, _out, err) = run(&program, vec![], vec![]);
        if let Some(e) = err {
            println!("Error: {}", e);
        }

        let last_state = trace.last().expect("Execution seems to have failed.");
        let zero = BFieldElement::zero();
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
        // TVM assembly to sample weights for the recursive verifier
        //
        // input: seed, num_weights
        //
        // output: num_weights-many random weights
        let sample_weights_code = "
            push 17 push 13 push 11        // get seed - should be an argument
            read_io                        // number of weights - should be argument
            sample_weights:                // proper program starts here
            call sample_weights_loop       // setup done, start sampling loop
            pop pop                        // clean up stack: RAM value & pointer
            pop pop pop pop                // clean up stack: seed & countdown
            halt                           // done - should be return

            sample_weights_loop:           // subroutine: loop until all weights are sampled
              dup0 push 0 eq skiz return   // no weights left
              push -1 add                  // decrease number of weights to still sample
              push 0 push 0 push 0 push 0  // prepare for hashing
              push 0 push 0 push 0 push 0  // prepare for hashing
              dup11 dup11 dup11 dup11      // prepare for hashing
              hash                         // hash seed & countdown
              swap13 swap10 pop            // re-organize stack
              swap13 swap10 pop            // re-organize stack
              swap13 swap10 swap7          // re-organize stack
              pop pop pop pop pop pop pop  // remove unnecessary remnants of digest
              recurse                      // repeat
        ";
        let program = Program::from_code(sample_weights_code).unwrap();
        println!("Successfully parsed the program.");
        let input_symbols = vec![BFieldElement::new(11)];
        let (trace, _out, err) = run(&program, input_symbols, vec![]);

        for state in trace.iter() {
            println!("{}", state);
        }
        if let Some(e) = err {
            panic!("The VM encountered an error: {}", e);
        }

        // check for graceful termination
        let last_state = trace.last().unwrap();
        assert_eq!(last_state.current_instruction().unwrap(), Halt);
    }

    /// TVM assembly to verify Merkle authentication paths
    ///
    /// input: merkle root, number of leafs, leaf values, APs
    ///
    /// output: Result<(), VMFail>
    const MT_AP_VERIFY: &str = concat!(
        "read_io ",                                 // number of authentication paths to test
        "",                                         // stack: [num]
        "mt_ap_verify: ",                           // proper program starts here
        "push 0 swap1 write_mem pop pop ",          // store number of APs at RAM address 0
        "",                                         // stack: []
        "read_io read_io read_io read_io read_io ", // read Merkle root
        "",                                         // stack: [r4 r3 r2 r1 r0]
        "call check_aps ",                          //
        "pop pop pop pop pop ",                     // leave clean stack: Merkle root
        "",                                         // stack: []
        "halt ",                                    // done – should be “return”
        "",
        "",                               // subroutine: check AP one at a time
        "",                               // stack before: [* r4 r3 r2 r1 r0]
        "",                               // stack after: [* r4 r3 r2 r1 r0]
        "check_aps: ",                    // start function description:
        "push 0 push 0 read_mem dup0 ",   // get number of APs left to check
        "",                               // stack: [* r4 r3 r2 r1 r0 0 num_left num_left]
        "push 0 eq ",                     // see if there are authentication paths left
        "",                               // stack: [* r4 r3 r2 r1 r0 0 num_left num_left==0]
        "skiz return ",                   // return if no authentication paths left
        "push -1 add write_mem pop pop ", // decrease number of authentication paths left to check
        "",                               // stack: [* r4 r3 r2 r1 r0]
        "call get_idx_and_hash_leaf ",    //
        "",                               // stack: [* r4 r3 r2 r1 r0 idx d4 d3 d2 d1 d0 0 0 0 0 0]
        "call traverse_tree ",            //
        "",                               // stack: [* r4 r3 r2 r1 r0 idx>>2 - - - - - - - - - -]
        "call assert_tree_top ",          //
        // stack: [* r4 r3 r2 r1 r0]
        "recurse ", // check next AP
        "",
        "",                                         // subroutine: read index & hash leaf
        "",                                         // stack before: [*]
        "",                        // stack afterwards: [* idx d4 d3 d2 d1 d0 0 0 0 0 0]
        "get_idx_and_hash_leaf: ", // start function description:
        "read_io ",                // read node index
        "read_io read_io read_io read_io read_io ", // read leaf's value
        "push 0 push 0 push 0 push 0 push 0 ", // pad before fixed-length hash
        "hash return ",            // compute leaf's digest
        "",
        "",                             // subroutine: go up tree
        "",                             // stack before: [* idx - - - - - - - - - -]
        "",                             // stack after: [* idx>>2 - - - - - - - - - -]
        "traverse_tree: ",              // start function description:
        "dup10 push 1 eq skiz return ", // break loop if node index is 1
        "divine_sibling hash recurse ", // move up one level in the Merkle tree
        "",
        "",                     // subroutine: compare digests
        "",                     // stack before: [* r4 r3 r2 r1 r0 idx a b c d e - - - - -]
        "",                     // stack after: [* r4 r3 r2 r1 r0]
        "assert_tree_top: ",    // start function description:
        "pop pop pop pop pop ", // remove unnecessary “0”s from hashing
        "",                     // stack: [* r4 r3 r2 r1 r0 idx a b c d e]
        "swap1 swap2 swap3 swap4 swap5 ",
        "",                     // stack: [* r4 r3 r2 r1 r0 a b c d e idx]
        "assert ",              //
        "",                     // stack: [* r4 r3 r2 r1 r0 a b c d e]
        "assert_vector ",       // actually compare to root of tree
        "pop pop pop pop pop ", // clean up stack, leave only one root
        "return ",              //
    );

    #[test]
    fn run_tvm_mt_ap_verify_test() {
        // generate merkle tree
        type H = RescuePrimeRegular;

        const NUM_LEAFS: usize = 64;
        let leafs: [Digest; NUM_LEAFS] = random_elements_array();
        let zero_padding: Digest = Digest::new([BFieldElement::zero(); DIGEST_LENGTH]);
        let digests = leafs
            .iter()
            .map(|leaf| H::hash_pair(&zero_padding, leaf))
            .collect_vec();
        let merkle_tree: MerkleTree<H, Maker> = Maker::from_digests(&digests);
        let root: Digest = merkle_tree.get_root();

        // generate program
        let program = Program::from_code(MT_AP_VERIFY).unwrap();
        let order: Vec<usize> = (0..5).rev().collect();

        let selected_leaf_indices = [0, 28, 55];

        let secret_input = selected_leaf_indices
            .iter()
            .flat_map(|leaf_index| {
                let auth_path = merkle_tree.get_authentication_path(*leaf_index);
                let selected_values: Vec<_> = (0..6)
                    .flat_map(|i| {
                        let values = auth_path[i].values();
                        let reordered_values: Vec<BFieldElement> =
                            order.iter().map(|ord| values[*ord]).collect();
                        reordered_values
                    })
                    .collect();
                selected_values
            })
            .collect_vec();

        let input = vec![
            // number of path tests
            BFieldElement::new(3),
            // Merkle root
            root.values()[order[0]],
            root.values()[order[1]],
            root.values()[order[2]],
            root.values()[order[3]],
            root.values()[order[4]],
            // node index 64, leaf index 0
            BFieldElement::new(64),
            // value of leaf with index 0
            leafs[0].values()[order[0]],
            leafs[0].values()[order[1]],
            leafs[0].values()[order[2]],
            leafs[0].values()[order[3]],
            leafs[0].values()[order[4]],
            // node index 92, leaf index 28
            // 92 = 1011100_2
            // 28 =   11100_2
            BFieldElement::new(92),
            // value of leaf with index 28
            leafs[28].values()[order[0]],
            leafs[28].values()[order[1]],
            leafs[28].values()[order[2]],
            leafs[28].values()[order[3]],
            leafs[28].values()[order[4]],
            // node index 119, leaf index 55
            BFieldElement::new(119),
            // 119 = 1110111_2
            // 55  =  110111_2
            // value of leaf with node 55
            leafs[55].values()[order[0]],
            leafs[55].values()[order[1]],
            leafs[55].values()[order[2]],
            leafs[55].values()[order[3]],
            leafs[55].values()[order[4]],
        ];

        let (trace, _out, err) = run(&program, input, secret_input);

        for state in trace.iter() {
            println!("{}", state);
        }
        if let Some(e) = err {
            panic!("The VM encountered an error: {}", e);
        }

        // check for graceful termination
        let last_state = trace.last().unwrap();
        assert_eq!(last_state.current_instruction().unwrap(), Halt);
    }

    #[test]
    fn run_tvm_get_colinear_y_test() {
        // see also: get_colinear_y in src/shared_math/polynomial.rs
        let get_colinear_y_code = "
            read_io                       // p2_x
            read_io read_io               // p1_y p1_x
            read_io read_io               // p0_y p0_x
            swap3 push -1 mul dup1 add    // dy = p0_y - p1_y
            dup3 push -1 mul dup5 add mul // dy·(p2_x - p0_x)
            dup3 dup3 push -1 mul add     // dx = p0_x - p1_x
            invert mul add                // compute result
            swap3 pop pop pop             // leave a clean stack
            write_io halt
        ";

        let program = Program::from_code(get_colinear_y_code).unwrap();
        println!("Successfully parsed the program.");
        let input_symbols = [7, 2, 1, 3, 4].map(BFieldElement::new).to_vec();
        let (trace, out, err) = run(&program, input_symbols, vec![]);
        assert_eq!(out[0], BFieldElement::new(4));
        for state in trace.iter() {
            println!("{}", state);
        }
        if let Some(e) = err {
            panic!("The VM encountered an error: {}", e);
        }

        // check for graceful termination
        let last_state = trace.last().unwrap();
        assert_eq!(last_state.current_instruction().unwrap(), Halt);
    }

    #[test]
    fn run_tvm_countdown_from_10_test() {
        let countdown_code = "
            push 10
            call loop
            
            loop:
                dup0
                write_io
                push -1
                add
                dup0
                skiz
                  recurse
                write_io
                halt
            ";

        let program = Program::from_code(countdown_code).unwrap();
        let (trace, out, err) = run(&program, vec![], vec![]);

        println!("{}", program);
        for state in trace.iter() {
            println!("{}", state);
        }

        if let Some(e) = err {
            panic!("The VM encountered an error: {e}");
        }

        let expected = (0..=10).map(BFieldElement::new).rev().collect_vec();
        assert_eq!(expected, out);
    }

    #[test]
    fn run_tvm_fibonacci_vit_tvm() {
        let code = FIBONACCI_VIT;
        let program = Program::from_code(code).unwrap();
        let (_trace, out, err) = run(&program, vec![7_u64.into()], vec![]);
        if let Some(e) = err {
            panic!("The VM encountered an error: {e}");
        }

        assert_eq!(Some(&BFieldElement::new(21)), out.get(0));
    }

    #[test]
    fn run_tvm_fibonacci_lt_test() {
        let code = FIB_FIXED_7_LT;
        let program = Program::from_code(code).unwrap();
        let (trace, _out, _err) = run(&program, vec![], vec![]);
        let last_state = trace.last().unwrap();
        assert_eq!(BFieldElement::new(21), last_state.op_stack.st(ST0));
    }

    #[test]
    fn run_tvm_gcd_test() {
        let code = GCD_X_Y;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);
        let (trace, out, _err) = run(&program, vec![42_u64.into(), 56_u64.into()], vec![]);

        println!("{}", program);
        for state in trace.iter() {
            println!("{}", state);
        }

        let expected = BFieldElement::new(14);
        let actual = *out.last().unwrap();
        assert_eq!(expected, actual);
    }

    #[test]
    fn run_tvm_swap_test() {
        let code = "push 1 push 2 swap1 halt";
        let program = Program::from_code(code).unwrap();
        let (_trace, _out, _err) = run(&program, vec![], vec![]);
    }

    #[test]
    fn read_mem_unitialized() {
        let program = Program::from_code("read_mem halt").unwrap();
        let (trace, _out, err) = run(&program, vec![], vec![]);
        assert!(err.is_none(), "Reading from uninitialized memory address");
        assert_eq!(2, trace.len());
    }
}
