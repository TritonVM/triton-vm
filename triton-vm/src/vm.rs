use std::collections::hash_map::Entry::*;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::convert::TryInto;
use std::fmt::Display;

use anyhow::anyhow;
use anyhow::bail;
use anyhow::Error;
use anyhow::Result;
use ndarray::s;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use num_traits::One;
use num_traits::Zero;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::b_field_element::BFIELD_ZERO;
use twenty_first::shared_math::tip5;
use twenty_first::shared_math::tip5::Tip5;
use twenty_first::shared_math::tip5::Tip5State;
use twenty_first::shared_math::tip5::DIGEST_LENGTH;
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
use crate::error::InstructionError::InstructionPointerOverflow;
use crate::error::InstructionError::*;
use crate::op_stack::OpStack;
use crate::table::hash_table;
use crate::table::hash_table::HashTable;
use crate::table::processor_table;
use crate::table::processor_table::ProcessorTraceRow;
use crate::table::table_column::HashBaseTableColumn::*;
use crate::table::table_column::MasterBaseTableColumn;
use crate::table::table_column::ProcessorBaseTableColumn;
use crate::vm::CoProcessorCall::*;

/// The number of helper variable registers
pub const HV_REGISTER_COUNT: usize = 4;

#[derive(Debug, Default, Clone)]
pub struct VMState<'pgm> {
    // Memory
    /// The **program memory** stores the instructions (and their arguments) of the program
    /// currently being executed by Triton VM. It is read-only.
    pub program: &'pgm [Instruction],

    /// A list of [`BFieldElement`]s the program can read from using instruction `read_io`.
    pub public_input: VecDeque<BFieldElement>,

    /// A list of [`BFieldElement`]s the program can read from using instructions `divine`
    /// and `divine_sibling`.
    pub secret_input: VecDeque<BFieldElement>,

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
        public_input: Vec<BFieldElement>,
        secret_input: Vec<BFieldElement>,
    ) -> Self {
        Self {
            program: &program.instructions,
            public_input: public_input.into(),
            secret_input: secret_input.into(),
            ..VMState::default()
        }
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
            Pop | Skiz
                | Assert
                | WriteMem
                | WriteIo
                | Add
                | Mul
                | Eq
                | XbMul
                | Lt
                | And
                | Xor
                | Pow
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
    pub fn step(&mut self) -> Result<Option<CoProcessorCall>> {
        // All instructions increase the cycle count
        self.cycle_count += 1;
        let mut co_processor_trace = None;
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
                let elem = self.secret_input.pop_front().ok_or(anyhow!(
                    "Instruction `divine`: secret input buffer is empty."
                ))?;
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
                self.halting = true;
                self.instruction_pointer += 1;
            }

            ReadMem => {
                let ramp = self.op_stack.safe_peek(ST0);
                let ramv = self.memory_get(&ramp);
                self.op_stack.push(ramv);
                self.ramp = ramp.value();
                self.instruction_pointer += 1;
            }

            WriteMem => {
                let ramp = self.op_stack.safe_peek(ST1);
                let ramv = self.op_stack.pop()?;
                self.ramp = ramp.value();
                self.ram.insert(ramp, ramv);
                self.instruction_pointer += 1;
            }

            Hash => {
                let to_hash = self.op_stack.pop_n::<{ tip5::RATE }>()?;
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

                co_processor_trace = Some(Tip5Trace(Hash, Box::new(tip5_trace)));
                self.instruction_pointer += 1;
            }

            AbsorbInit | Absorb => {
                // fetch top elements but don't alter the stack
                let to_absorb = self.op_stack.pop_n::<{ tip5::RATE }>()?;
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

                co_processor_trace =
                    Some(Tip5Trace(self.current_instruction()?, Box::new(tip5_trace)));
                self.instruction_pointer += 1;
            }

            Squeeze => {
                let _ = self.op_stack.pop_n::<{ tip5::RATE }>()?;
                for i in (0..tip5::RATE).rev() {
                    self.op_stack.push(self.sponge_state[i]);
                }
                let tip5_trace = Tip5::trace(&mut Tip5State {
                    state: self.sponge_state,
                });
                self.sponge_state = tip5_trace.last().unwrap().to_owned();

                co_processor_trace = Some(Tip5Trace(Squeeze, Box::new(tip5_trace)));
                self.instruction_pointer += 1;
            }

            DivineSibling => {
                self.divine_sibling()?;
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
                let u32_table_entry = (Split, lo, hi);
                co_processor_trace = Some(U32TableEntries(vec![u32_table_entry]));
            }

            Lt => {
                let lhs = self.op_stack.pop_u32()?;
                let rhs = self.op_stack.pop_u32()?;
                let lt = BFieldElement::new((lhs < rhs) as u64);
                self.op_stack.push(lt);
                self.instruction_pointer += 1;
                let u32_table_entry = (Lt, (lhs as u64).into(), (rhs as u64).into());
                co_processor_trace = Some(U32TableEntries(vec![u32_table_entry]));
            }

            And => {
                let lhs = self.op_stack.pop_u32()?;
                let rhs = self.op_stack.pop_u32()?;
                let and = BFieldElement::new((lhs & rhs) as u64);
                self.op_stack.push(and);
                self.instruction_pointer += 1;
                let u32_table_entry = (And, (lhs as u64).into(), (rhs as u64).into());
                co_processor_trace = Some(U32TableEntries(vec![u32_table_entry]));
            }

            Xor => {
                let lhs = self.op_stack.pop_u32()?;
                let rhs = self.op_stack.pop_u32()?;
                let xor = BFieldElement::new((lhs ^ rhs) as u64);
                self.op_stack.push(xor);
                self.instruction_pointer += 1;
                // Triton VM uses the following equality to compute the results of both the `and`
                // and `xor` instruction using the u32 coprocessor's `and` capability:
                // a ^ b = a + b - 2 · (a & b)
                let u32_table_entry = (And, (lhs as u64).into(), (rhs as u64).into());
                co_processor_trace = Some(U32TableEntries(vec![u32_table_entry]));
            }

            Log2Floor => {
                let lhs = self.op_stack.pop_u32()?;
                if lhs.is_zero() {
                    return vm_err(LogarithmOfZero);
                }
                let l2f = BFieldElement::new(lhs.ilog2().into());
                self.op_stack.push(l2f);
                self.instruction_pointer += 1;
                let u32_table_entry = (Log2Floor, (lhs as u64).into(), BFIELD_ZERO);
                co_processor_trace = Some(U32TableEntries(vec![u32_table_entry]));
            }

            Pow => {
                let lhs = self.op_stack.pop()?;
                let rhs = self.op_stack.pop_u32()?;
                let pow = lhs.mod_pow(rhs as u64);
                self.op_stack.push(pow);
                self.instruction_pointer += 1;
                let u32_table_entry = (Pow, lhs, (rhs as u64).into());
                co_processor_trace = Some(U32TableEntries(vec![u32_table_entry]));
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
                let u32_table_entry_0 = (Lt, rem, (denom as u64).into());
                let u32_table_entry_1 = (Split, (numer as u64).into(), quot);
                co_processor_trace =
                    Some(U32TableEntries(vec![u32_table_entry_0, u32_table_entry_1]));
            }

            PopCount => {
                let lhs = self.op_stack.pop_u32()?;
                let pop_count = BFieldElement::new(lhs.count_ones() as u64);
                self.op_stack.push(pop_count);
                self.instruction_pointer += 1;
                let u32_table_entry = (PopCount, (lhs as u64).into(), BFIELD_ZERO);
                co_processor_trace = Some(U32TableEntries(vec![u32_table_entry]));
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
                let elem_to_write = self.op_stack.pop()?;
                self.public_output.push(elem_to_write);
                self.instruction_pointer += 1;
            }

            ReadIo => {
                let in_elem = self.public_input.pop_front().ok_or(anyhow!(
                    "Instruction `read_io`: public input buffer is empty."
                ))?;
                self.op_stack.push(in_elem);
                self.instruction_pointer += 1;
            }
        }

        // Check that no instruction left the OpStack with too few elements
        if self.op_stack.is_too_shallow() {
            return vm_err(OpStackTooShallow);
        }

        Ok(co_processor_trace)
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

    /// Return the next instruction on the tape, skipping arguments
    ///
    /// Note that this is not necessarily the next instruction to execute,
    /// since the current instruction could be a jump, but it is either
    /// program[ip + 1] or program[ip + 2] depending on whether the current
    /// instruction takes an argument or not.
    pub fn next_instruction(&self) -> Result<Instruction> {
        let ci = self.current_instruction()?;
        let ci_size = ci.size();
        let ni_pointer = self.instruction_pointer + ci_size;
        self.program
            .get(ni_pointer)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow(ni_pointer)))
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

    fn divine_sibling(&mut self) -> Result<()> {
        // st0-st4
        let _ = self.op_stack.pop_n::<{ DIGEST_LENGTH }>()?;

        // st5-st9
        let known_digest = self.op_stack.pop_n::<{ DIGEST_LENGTH }>()?;

        // st10
        let node_index_elem: BFieldElement = self.op_stack.pop()?;
        let node_index: u32 = node_index_elem
            .try_into()
            .unwrap_or_else(|_| panic!("{node_index_elem:?} is not a u32"));

        // nondeterministic guess, flipped
        let sibling_elem_0 = self.secret_input.pop_front().ok_or(anyhow!(
            "Instruction `divine_sibling`: secret input buffer is empty (0)."
        ))?;
        let sibling_elem_1 = self.secret_input.pop_front().ok_or(anyhow!(
            "Instruction `divine_sibling`: secret input buffer is empty (1)."
        ))?;
        let sibling_elem_2 = self.secret_input.pop_front().ok_or(anyhow!(
            "Instruction `divine_sibling`: secret input buffer is empty (2)."
        ))?;
        let sibling_elem_3 = self.secret_input.pop_front().ok_or(anyhow!(
            "Instruction `divine_sibling`: secret input buffer is empty (3)."
        ))?;
        let sibling_elem_4 = self.secret_input.pop_front().ok_or(anyhow!(
            "Instruction `divine_sibling`: secret input buffer is empty (4)."
        ))?;
        let sibling_digest: [BFieldElement; DIGEST_LENGTH] = [
            sibling_elem_4,
            sibling_elem_3,
            sibling_elem_2,
            sibling_elem_1,
            sibling_elem_0,
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

/// Simulate, _i.e._, execute a [`Program`]. If execution succeeds, returns
/// 1. an [`AlgebraicExecutionTrace`], and
/// 1. the output of the program.
///
/// See also [`debug`] and [`run`].
pub fn simulate(
    program: &Program,
    public_input: Vec<BFieldElement>,
    secret_input: Vec<BFieldElement>,
) -> Result<(AlgebraicExecutionTrace, Vec<BFieldElement>)> {
    let mut aet = AlgebraicExecutionTrace::new(program.clone());
    let mut state = VMState::new(program, public_input, secret_input);
    assert_eq!(program.len_bwords(), aet.instruction_multiplicities.len());

    while !state.halting {
        aet.processor_trace
            .push_row(state.to_processor_row().view())
            .expect("shapes must be identical");

        if state.instruction_pointer < aet.instruction_multiplicities.len() {
            aet.instruction_multiplicities[state.instruction_pointer] += 1;
        } else {
            bail!(InstructionPointerOverflow(state.instruction_pointer));
        }

        let co_processor_trace = state.step()?;
        match co_processor_trace {
            Some(Tip5Trace(Hash, tip5_trace)) => aet.append_hash_trace(*tip5_trace),
            Some(Tip5Trace(instruction, tip5_trace)) => {
                aet.append_sponge_trace(instruction, *tip5_trace)
            }
            Some(U32TableEntries(u32_entries)) => {
                for u32_entry in u32_entries {
                    aet.u32_entries
                        .entry(u32_entry)
                        .and_modify(|multiplicity| *multiplicity += 1)
                        .or_insert(1);
                }
            }
            None => (),
        }
    }

    Ok((aet, state.public_output))
}

/// Similar to [`run`], but also returns a [`Vec`] of [`VMState`]s, one for each step of the VM.
/// On premature termination of the VM, returns all [`VMState`]s up to the point of failure.
///
/// The VM's initial state is either the provided `initial_state`, or a new [`VMState`] if
/// `initial_state` is `None`. The initial state is included in the returned [`Vec`] of
/// [`VMState`]s. If an initial state is provided, the `program`, `public_input` and `private_input`
/// provided to this method are ignored and the initial state's program and inputs are used instead.
///
/// If `num_cycles_to_execute` is `Some(number_of_cycles)`, the VM will execute at most
/// `number_of_cycles` cycles. If `num_cycles_to_execute` is `None`, the VM will execute until
/// it halts or the maximum number of cycles (2^{32}) is reached.
///
/// See also [`debug_terminal_state`] and [`simulate`].
pub fn debug<'pgm>(
    program: &'pgm Program,
    public_input: Vec<BFieldElement>,
    secret_input: Vec<BFieldElement>,
    initial_state: Option<VMState<'pgm>>,
    num_cycles_to_execute: Option<u32>,
) -> (Vec<VMState<'pgm>>, Option<Error>) {
    let mut states = vec![];
    let mut state = match initial_state {
        Some(initial_state) => initial_state,
        None => VMState::new(program, public_input, secret_input),
    };

    let max_cycles = match num_cycles_to_execute {
        Some(number_of_cycles) => state.cycle_count + number_of_cycles,
        None => u32::MAX,
    };

    while !state.halting && state.cycle_count < max_cycles {
        states.push(state.clone());
        if let Err(err) = state.step() {
            return (states, Some(err));
        }
    }

    states.push(state);
    (states, None)
}

/// Run Triton VM on the given [`Program`] with the given public and secret input, and return the
/// final [`VMState`]. Requires substantially less RAM than [`debug`] since no intermediate states
/// are recorded.
///
/// Parameters `initial_state` and `num_cycles_to_execute` are handled like in [`debug`];
/// see there for more details.
///
/// If an error is encountered, returns the error and the [`VMState`] at the point of failure.
///
/// See also [`simulate`] and [`run`].
pub fn debug_terminal_state<'pgm>(
    program: &'pgm Program,
    public_input: Vec<BFieldElement>,
    secret_input: Vec<BFieldElement>,
    initial_state: Option<VMState<'pgm>>,
    num_cycles_to_execute: Option<u32>,
) -> Result<VMState<'pgm>, (Error, VMState<'pgm>)> {
    let mut state = match initial_state {
        Some(initial_state) => initial_state,
        None => VMState::new(program, public_input, secret_input),
    };

    let max_cycles = match num_cycles_to_execute {
        Some(number_of_cycles) => state.cycle_count + number_of_cycles,
        None => u32::MAX,
    };

    while !state.halting && state.cycle_count < max_cycles {
        // The internal state transition method [`VMState::step`] is not atomic.
        // To avoid returning an inconsistent state in case of a failed transition, the last
        // known-to-be-consistent state is returned.
        let previous_state = state.clone();
        if let Err(err) = state.step() {
            return Err((err, previous_state));
        }
    }
    Ok(state)
}

/// Run Triton VM on the given [`Program`] with the given public and secret input.
///
/// See also [`simulate`] and [`debug`].
pub fn run(
    program: &Program,
    public_input: Vec<BFieldElement>,
    secret_input: Vec<BFieldElement>,
) -> Result<Vec<BFieldElement>> {
    let mut state = VMState::new(program, public_input, secret_input);
    while !state.halting {
        state.step()?;
    }
    Ok(state.public_output)
}

/// An Algebraic Execution Trace (AET) is the primary witness required for proof generation. It
/// holds every intermediate state of the processor and all co-processors, alongside additional
/// witness information, such as the number of times each instruction has been looked up
/// (equivalently, how often each instruction has been executed).
#[derive(Debug, Clone)]
pub struct AlgebraicExecutionTrace {
    /// The program that was executed in order to generate the trace.
    pub program: Program,

    /// The number of times each instruction has been executed.
    ///
    /// Each instruction in the `program` has one associated entry in `instruction_multiplicities`,
    /// counting the number of times this specific instruction at that location in the program
    /// memory has been executed.
    pub instruction_multiplicities: Vec<u32>,

    /// Records the state of the processor after each instruction.
    pub processor_trace: Array2<BFieldElement>,

    /// For the `hash` instruction, the hash trace records the internal state of the Tip5
    /// permutation for each round.
    pub hash_trace: Array2<BFieldElement>,

    /// For the Sponge instructions, i.e., `absorb_init`, `absorb`, and `squeeze`, the Sponge
    /// trace records the internal state of the Tip5 permutation for each round.
    pub sponge_trace: Array2<BFieldElement>,

    /// The u32 entries hold all pairs of BFieldElements that were written to the U32 Table,
    /// alongside the u32 instruction that was executed at the time. Additionally, it records how
    /// often the instruction was executed with these arguments.
    pub u32_entries: HashMap<(Instruction, BFieldElement, BFieldElement), u64>,

    /// Records how often each entry in the cascade table was looked up.
    pub cascade_table_lookup_multiplicities: HashMap<u16, u64>,

    /// Records how often each entry in the lookup table was looked up.
    pub lookup_table_lookup_multiplicities: [u64; 1 << 8],
}

impl AlgebraicExecutionTrace {
    pub fn new(program: Program) -> Self {
        let instruction_multiplicities = vec![0_u32; program.len_bwords()];
        Self {
            program,
            instruction_multiplicities,
            processor_trace: Array2::default([0, processor_table::BASE_WIDTH]),
            hash_trace: Array2::default([0, hash_table::BASE_WIDTH]),
            sponge_trace: Array2::default([0, hash_table::BASE_WIDTH]),
            u32_entries: HashMap::new(),
            cascade_table_lookup_multiplicities: HashMap::new(),
            lookup_table_lookup_multiplicities: [0; 1 << 8],
        }
    }

    pub fn append_hash_trace(
        &mut self,
        hash_permutation_trace: [[BFieldElement; tip5::STATE_SIZE]; tip5::NUM_ROUNDS + 1],
    ) {
        self.increase_lookup_multiplicities(hash_permutation_trace);
        let mut hash_trace_addendum = HashTable::convert_to_hash_table_rows(hash_permutation_trace);
        hash_trace_addendum
            .slice_mut(s![.., CI.base_table_index()])
            .fill(Instruction::Hash.opcode_b());
        self.hash_trace
            .append(Axis(0), hash_trace_addendum.view())
            .expect("shapes must be identical");
    }

    pub fn append_sponge_trace(
        &mut self,
        instruction: Instruction,
        hash_permutation_trace: [[BFieldElement; tip5::STATE_SIZE]; tip5::NUM_ROUNDS + 1],
    ) {
        assert!(matches!(
            instruction,
            Instruction::AbsorbInit | Instruction::Absorb | Instruction::Squeeze
        ));
        self.increase_lookup_multiplicities(hash_permutation_trace);
        let mut sponge_trace_addendum =
            HashTable::convert_to_hash_table_rows(hash_permutation_trace);
        sponge_trace_addendum
            .slice_mut(s![.., CI.base_table_index()])
            .fill(instruction.opcode_b());
        self.sponge_trace
            .append(Axis(0), sponge_trace_addendum.view())
            .expect("shapes must be identical");
    }

    /// Given a trace of the hash function's permutation, determines how often each entry in the
    /// - cascade table was looked up, and
    /// - lookup table was looked up
    /// and increases the multiplicities accordingly
    fn increase_lookup_multiplicities(
        &mut self,
        hash_permutation_trace: [[BFieldElement; tip5::STATE_SIZE]; tip5::NUM_ROUNDS + 1],
    ) {
        // The last row in the trace is the permutation's result, meaning that no lookups are
        // performed on it. Therefore, we skip it.
        for row in hash_permutation_trace.iter().rev().skip(1) {
            for &state_element in row[0..tip5::NUM_SPLIT_AND_LOOKUP].iter() {
                for limb in HashTable::base_field_element_into_16_bit_limbs(state_element) {
                    match self.cascade_table_lookup_multiplicities.entry(limb) {
                        Occupied(mut cascade_table_entry) => *cascade_table_entry.get_mut() += 1,
                        Vacant(cascade_table_entry) => {
                            cascade_table_entry.insert(1);
                            let limb_lo = limb & 0xff;
                            let limb_hi = (limb >> 8) & 0xff;
                            self.lookup_table_lookup_multiplicities[limb_lo as usize] += 1;
                            self.lookup_table_lookup_multiplicities[limb_hi as usize] += 1;
                        }
                    }
                }
            }
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
    use num_traits::One;
    use num_traits::Zero;
    use rand::rngs::ThreadRng;
    use rand::Rng;
    use rand::RngCore;
    use twenty_first::shared_math::b_field_element::BFIELD_ZERO;
    use twenty_first::shared_math::other::random_elements;
    use twenty_first::shared_math::other::random_elements_array;
    use twenty_first::shared_math::tip5::Digest;
    use twenty_first::shared_math::tip5::Tip5;
    use twenty_first::shared_math::traits::FiniteField;
    use twenty_first::shared_math::traits::ModPowU32;
    use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;
    use twenty_first::util_types::algebraic_hasher::SpongeHasher;
    use twenty_first::util_types::merkle_tree::MerkleTree;
    use twenty_first::util_types::merkle_tree_maker::MerkleTreeMaker;

    use crate::error::InstructionError;
    use crate::op_stack::OP_STACK_REG_COUNT;
    use crate::shared_tests::SourceCodeAndInput;
    use crate::shared_tests::FIBONACCI_SEQUENCE;
    use crate::shared_tests::VERIFY_SUDOKU;
    use crate::stark::MTMaker;
    use crate::table::processor_table::ProcessorTraceRow;

    use super::*;

    fn pretty_print_array_view<FF: FiniteField>(array: ArrayView1<FF>) -> String {
        array
            .iter()
            .map(|ff| format!("{ff}"))
            .collect_vec()
            .join(", ")
    }

    pub const GCD_X_Y: &str = "
        read_io  // _ a
        read_io  // _ a b
        dup 1    // _ a b a
        dup 1    // _ a b a b
        lt       // _ a b b<a
        skiz     // _ a b
            swap 1   // _ d n where n > d

        // ---
        loop_cond:
        dup 1
        push 0 
        eq 
        skiz 
            call terminate  // _ d n where d != 0
        dup 1  // _ d n d
        dup 1  // _ d n d n
        div    // _ d n q r
        swap 2 // _ d r q n
        pop    // _ d r q
        pop    // _ d r
        swap 1 // _ r d
        call loop_cond
        // ---
        
        terminate:
            // _ d n where d == 0
            write_io // _ d
            halt
        ";

    #[test]
    fn initialise_table_test() {
        let code = GCD_X_Y;
        let program = Program::from_code(code).unwrap();
        let stdin = vec![BFieldElement::new(42), BFieldElement::new(56)];
        let (aet, stdout) = simulate(&program, stdin, vec![]).unwrap();

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
        let program = Program::from_code(GCD_X_Y).unwrap();
        let stdin = vec![42_u64.into(), 56_u64.into()];
        let stdout = run(&program, stdin, vec![]).unwrap();

        let stdout = Array1::from(stdout);
        println!("VM output: [{}]", pretty_print_array_view(stdout.view()));

        assert_eq!(BFieldElement::new(14), stdout[0]);
    }

    pub fn test_hash_nop_nop_lt() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("hash nop hash nop nop hash push 3 push 2 lt assert halt")
    }

    pub fn test_program_for_halt() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("halt")
    }

    pub fn test_program_for_push_pop_dup_swap_nop() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "push 1 push 2 pop assert \
            push 1 dup 0 assert assert \
            push 1 push 2 swap 1 assert pop \
            nop nop nop halt",
        )
    }

    pub fn test_program_for_divine() -> SourceCodeAndInput {
        SourceCodeAndInput {
            source_code: "divine assert halt".to_string(),
            input: vec![],
            secret_input: vec![1],
        }
    }

    pub fn test_program_for_skiz() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 1 skiz push 0 skiz assert push 1 skiz halt")
    }

    pub fn test_program_for_call_recurse_return() -> SourceCodeAndInput {
        let source_code = "push 2 call label halt label: push -1 add dup 0 skiz recurse return";
        SourceCodeAndInput::without_input(source_code)
    }

    pub fn test_program_for_write_mem_read_mem() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 2 push 1 write_mem push 0 pop read_mem assert halt")
    }

    pub fn test_program_for_hash() -> SourceCodeAndInput {
        let source_code = "push 0 push 0 push 0 push 1 push 2 push 3 hash \
            pop pop pop pop pop read_io eq assert halt";
        let hash_input = [3, 2, 1, 0, 0, 0, 0, 0, 0, 0].map(BFieldElement::new);
        let digest = Tip5::hash_10(&hash_input).map(|e| e.value());
        SourceCodeAndInput {
            source_code: source_code.to_string(),
            input: vec![digest.to_vec()[0]],
            secret_input: vec![],
        }
    }

    pub fn test_program_for_divine_sibling_noswitch() -> SourceCodeAndInput {
        let source_code = "
            push 3 \
            push 4 push 2 push 2 push 2 push 1 \
            push 5679457 push 1337 push 345887 push -234578456 push 23657565 \
            divine_sibling \
            push 1 add assert assert assert assert assert \
            assert \
            push -1 add assert \
            push -1 add assert \
            push -1 add assert \
            push -3 add assert \
            assert halt ";
        SourceCodeAndInput {
            source_code: source_code.to_string(),
            input: vec![],
            secret_input: vec![1, 1, 1, 1, 0],
        }
    }

    pub fn test_program_for_divine_sibling_switch() -> SourceCodeAndInput {
        let source_code = "
            push 2 \
            push 4 push 2 push 2 push 2 push 1 \
            push 5679457 push 1337 push 345887 push -234578456 push 23657565 \
            divine_sibling \
            assert \
            push -1 add assert \
            push -1 add assert \
            push -1 add assert \
            push -3 add assert \
            push 1 add assert assert assert assert assert \
            assert halt ";
        SourceCodeAndInput {
            source_code: source_code.to_string(),
            input: vec![],
            secret_input: vec![1, 1, 1, 1, 0],
        }
    }

    pub fn test_program_for_assert_vector() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "push 1 push 2 push 3 push 4 push 5 \
             push 1 push 2 push 3 push 4 push 5 \
             assert_vector halt",
        )
    }

    pub fn test_program_for_sponge_instructions() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "absorb_init push 3 push 2 push 1 absorb absorb squeeze halt",
        )
    }

    pub fn test_program_for_sponge_instructions_2() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "hash absorb_init push 3 push 2 push 1 absorb absorb squeeze halt",
        )
    }

    pub fn test_program_for_many_sponge_instructions() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "absorb_init squeeze absorb absorb absorb squeeze squeeze squeeze absorb \
            absorb_init absorb_init absorb_init absorb absorb_init squeeze squeeze \
            absorb_init squeeze hash absorb hash squeeze hash absorb hash squeeze \
            absorb_init absorb absorb absorb absorb absorb absorb absorb halt",
        )
    }

    pub fn property_based_test_program_for_assert_vector() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.gen_range(0..BFieldElement::P);
        let st1 = rng.gen_range(0..BFieldElement::P);
        let st2 = rng.gen_range(0..BFieldElement::P);
        let st3 = rng.gen_range(0..BFieldElement::P);
        let st4 = rng.gen_range(0..BFieldElement::P);

        let source_code = format!(
            "push {st4} push {st3} push {st2} push {st1} push {st0} \
            read_io read_io read_io read_io read_io assert_vector halt",
        );

        SourceCodeAndInput {
            source_code,
            input: vec![st4, st3, st2, st1, st0],
            secret_input: vec![],
        }
    }

    pub fn property_based_test_program_for_sponge_instructions() -> SourceCodeAndInput {
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

        let source_code = format!(
            "
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
            ",
        );

        SourceCodeAndInput {
            source_code,
            input: sponge_output.map(|e| e.value()).to_vec(),
            secret_input: vec![],
        }
    }

    pub fn test_program_for_add_mul_invert() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "push 2 push -1 add assert \
            push -1 push -1 mul assert \
            push 3 dup 0 invert mul assert \
            halt",
        )
    }

    pub fn property_based_test_program_for_split() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u64() % BFieldElement::P;
        let hi = st0 >> 32;
        let lo = st0 & 0xffff_ffff;

        let source_code = format!("push {st0} split read_io eq assert read_io eq assert halt");
        SourceCodeAndInput {
            source_code,
            input: vec![lo, hi],
            secret_input: vec![],
        }
    }

    pub fn test_program_for_eq() -> SourceCodeAndInput {
        SourceCodeAndInput {
            source_code: "read_io divine eq assert halt".to_string(),
            input: vec![42],
            secret_input: vec![42],
        }
    }

    pub fn property_based_test_program_for_eq() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u64() % BFieldElement::P;

        let source_code = format!("push {st0} dup 0 read_io eq assert dup 0 divine eq assert halt");
        SourceCodeAndInput {
            source_code,
            input: vec![st0],
            secret_input: vec![st0],
        }
    }

    pub fn test_program_for_lsb() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "
            push 3 call lsb assert assert halt
            lsb:
                push 2 swap 1 div return
            ",
        )
    }

    pub fn property_based_test_program_for_lsb() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u32();
        let lsb = st0 % 2;
        let st0_shift_right = st0 >> 1;

        let source_code = format!(
            "
            push {st0} call lsb read_io eq assert read_io eq assert halt
            lsb:
                push 2 swap 1 div return
            "
        );
        SourceCodeAndInput {
            source_code,
            input: vec![lsb.into(), st0_shift_right.into()],
            secret_input: vec![],
        }
    }

    pub fn test_program_0_lt_0() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 0 push 0 lt halt")
    }

    pub fn test_program_for_lt() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "push 5 push 2 lt assert push 2 push 5 lt push 0 eq assert halt",
        )
    }

    pub fn property_based_test_program_for_lt() -> SourceCodeAndInput {
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

        let source_code = format!(
            "push {st1_0} push {st0_0} lt read_io eq assert \
             push {st1_1} push {st0_1} lt read_io eq assert halt"
        );
        SourceCodeAndInput {
            source_code,
            input: vec![result_0, result_1],
            secret_input: vec![],
        }
    }

    pub fn test_program_for_and() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "push 5 push 3 and assert push 12 push 5 and push 4 eq assert halt",
        )
    }

    pub fn property_based_test_program_for_and() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();

        let st1_0 = rng.next_u32();
        let st0_0 = rng.next_u32();
        let result_0 = st0_0.bitand(st1_0);

        let st1_1 = rng.next_u32();
        let st0_1 = rng.next_u32();
        let result_1 = st0_1.bitand(st1_1);

        let source_code = format!(
            "push {st1_0} push {st0_0} and read_io eq assert \
             push {st1_1} push {st0_1} and read_io eq assert halt"
        );
        SourceCodeAndInput {
            source_code,
            input: vec![result_0.into(), result_1.into()],
            secret_input: vec![],
        }
    }

    pub fn test_program_for_xor() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "push 7 push 6 xor assert push 5 push 12 xor push 9 eq assert halt",
        )
    }

    pub fn property_based_test_program_for_xor() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();

        let st1_0 = rng.next_u32();
        let st0_0 = rng.next_u32();
        let result_0 = st0_0.bitxor(st1_0);

        let st1_1 = rng.next_u32();
        let st0_1 = rng.next_u32();
        let result_1 = st0_1.bitxor(st1_1);

        let source_code = format!(
            "push {st1_0} push {st0_0} xor read_io eq assert \
             push {st1_1} push {st0_1} xor read_io eq assert halt"
        );
        SourceCodeAndInput {
            source_code,
            input: vec![result_0.into(), result_1.into()],
            secret_input: vec![],
        }
    }

    pub fn test_program_for_log2floor() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "push  1 log_2_floor push  0 eq assert \
             push  2 log_2_floor push  1 eq assert \
             push  3 log_2_floor push  1 eq assert \
             push  4 log_2_floor push  2 eq assert \
             push  7 log_2_floor push  2 eq assert \
             push  8 log_2_floor push  3 eq assert \
             push 15 log_2_floor push  3 eq assert \
             push 16 log_2_floor push  4 eq assert \
             push 31 log_2_floor push  4 eq assert \
             push 32 log_2_floor push  5 eq assert \
             push 33 log_2_floor push  5 eq assert \
             push 4294967295 log_2_floor push 31 eq assert halt",
        )
    }

    pub fn property_based_test_program_for_log2floor() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();

        let st0_0 = rng.next_u32();
        let l2f_0 = st0_0.ilog2();

        let st0_1 = rng.next_u32();
        let l2f_1 = st0_1.ilog2();

        let source_code = format!(
            "push {st0_0} log_2_floor read_io eq assert \
             push {st0_1} log_2_floor read_io eq assert halt"
        );
        SourceCodeAndInput {
            source_code,
            input: vec![l2f_0.into(), l2f_1.into()],
            secret_input: vec![],
        }
    }

    pub fn test_program_for_pow() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            // push <exponent: u32> push <base: BFE> pow push <result: BFE> eq assert
            "push  0 push 0 pow push    1 eq assert \
             push  0 push 1 pow push    1 eq assert \
             push  0 push 2 pow push    1 eq assert \
             push  1 push 0 pow push    0 eq assert \
             push  2 push 0 pow push    0 eq assert \
             push  2 push 5 pow push   25 eq assert \
             push  5 push 2 pow push   32 eq assert \
             push 10 push 2 pow push 1024 eq assert \
             push  3 push 3 pow push   27 eq assert \
             push  3 push 5 pow push  125 eq assert \
             push  9 push 7 pow push 40353607 eq assert \
             push 3040597274 push 05218640216028681988 pow push 11160453713534536216 eq assert \
             push 2378067562 push 13711477740065654150 pow push 06848017529532358230 eq assert \
             push  129856251 push 00218966585049330803 pow push 08283208434666229347 eq assert \
             push 1657936293 push 04999758396092641065 pow push 11426020017566937356 eq assert \
             push 3474149688 push 05702231339458623568 pow push 02862889945380025510 eq assert \
             push 2243935791 push 09059335263701504667 pow push 04293137302922963369 eq assert \
             push 1783029319 push 00037306102533222534 pow push 10002149917806048099 eq assert \
             push 3608140376 push 17716542154416103060 pow push 11885601801443303960 eq assert \
             push 1220084884 push 07207865095616988291 pow push 05544378138345942897 eq assert \
             push 3539668245 push 13491612301110950186 pow push 02612675697712040250 eq assert \
             halt",
        )
    }

    pub fn property_based_test_program_for_pow() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();

        let base_0: BFieldElement = rng.gen();
        let exp_0 = rng.next_u32();
        let result_0 = base_0.mod_pow_u32(exp_0).value();

        let base_1: BFieldElement = rng.gen();
        let exp_1 = rng.next_u32();
        let result_1 = base_1.mod_pow_u32(exp_1).value();

        let source_code = format!(
            "push {exp_0} push {base_0} pow read_io eq assert \
             push {exp_1} push {base_1} pow read_io eq assert halt"
        );
        SourceCodeAndInput {
            source_code,
            input: vec![result_0, result_1],
            secret_input: vec![],
        }
    }

    pub fn test_program_for_div() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 2 push 3 div assert assert halt")
    }

    pub fn property_based_test_program_for_div() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();

        let denominator = rng.next_u32();
        let numerator = rng.next_u32();
        let quotient = numerator / denominator;
        let remainder = numerator % denominator;

        let source_code = format!(
            "push {denominator} push {numerator} div read_io eq assert read_io eq assert halt"
        );
        SourceCodeAndInput {
            source_code,
            input: vec![remainder.into(), quotient.into()],
            secret_input: vec![],
        }
    }

    pub fn test_program_for_starting_with_pop_count() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("pop_count dup 0 push 0 eq assert halt")
    }

    pub fn test_program_for_pop_count() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 10 pop_count push 2 eq assert halt")
    }

    pub fn property_based_test_program_for_pop_count() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u32();
        let pop_count = st0.count_ones();
        let source_code = format!("push {st0} pop_count read_io eq assert halt",);
        SourceCodeAndInput {
            source_code,
            input: vec![pop_count.into()],
            secret_input: vec![],
        }
    }

    pub fn property_based_test_program_for_is_u32() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();
        let st0_u32 = rng.next_u32();
        let st0_not_u32 = ((rng.next_u32() as u64) << 32) + (rng.next_u32() as u64);
        SourceCodeAndInput::without_input(&format!(
            "push {st0_u32} call is_u32 assert \
             push {st0_not_u32} call is_u32 push 0 eq assert halt
             is_u32:
                 split pop push 0 eq return"
        ))
    }

    pub fn property_based_test_program_for_random_ram_access() -> SourceCodeAndInput {
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
        SourceCodeAndInput::without_input(&source_code)
    }

    #[test]
    // Sanity check for the relatively complex property-based test for random RAM access.
    fn run_dont_prove_property_based_test_for_random_ram_access() {
        let source_code_and_input = property_based_test_program_for_random_ram_access();
        source_code_and_input.simulate().unwrap();
    }

    #[test]
    pub fn negative_property_is_u32_test() {
        let mut rng = ThreadRng::default();
        let st0 = (rng.next_u32() as u64) << 32;

        let program = SourceCodeAndInput::without_input(&format!(
            "
            push {st0} call is_u32 assert halt
            is_u32:
                split pop push 0 eq return
            "
        ));
        let err = program.simulate().err();
        let err = err.unwrap();
        let err = err.downcast::<InstructionError>().unwrap();
        let AssertionFailed(_, _, _) = err else {
            panic!("Non-u32 must not pass u32-ness test.");
        };
    }

    pub fn test_program_for_split() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "push -2 split push 4294967295 eq assert push 4294967294 eq assert \
             push -1 split push 0 eq assert push 4294967295 eq assert \
             push  0 split push 0 eq assert push 0 eq assert \
             push  1 split push 1 eq assert push 0 eq assert \
             push  2 split push 2 eq assert push 0 eq assert \
             push 4294967297 split assert assert \
             halt",
        )
    }

    pub fn test_program_for_xxadd() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 5 push 6 push 7 push 8 push 9 push 10 xxadd halt")
    }

    pub fn test_program_for_xxmul() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 5 push 6 push 7 push 8 push 9 push 10 xxmul halt")
    }

    pub fn test_program_for_xinvert() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 5 push 6 push 7 xinvert halt")
    }

    pub fn test_program_for_xbmul() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 5 push 6 push 7 push 8 xbmul halt")
    }

    pub fn test_program_for_read_io_write_io() -> SourceCodeAndInput {
        SourceCodeAndInput {
            source_code:
                "read_io assert read_io read_io dup 1 dup 1 add write_io mul write_io halt"
                    .to_string(),
            input: vec![1, 3, 14],
            secret_input: vec![],
        }
    }

    pub fn small_tasm_test_programs() -> Vec<SourceCodeAndInput> {
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

    pub fn property_based_test_programs() -> Vec<SourceCodeAndInput> {
        vec![
            property_based_test_program_for_assert_vector(),
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
        let stdin_words = vec![2, 3, 5, 7, 11, 13];
        let xxadd_code = "
            read_io read_io read_io
            read_io read_io read_io
            xxadd
            swap 2
            write_io write_io write_io
            halt
        ";
        let program = SourceCodeAndInput {
            source_code: xxadd_code.to_string(),
            input: stdin_words,
            secret_input: vec![],
        };

        let (_, actual_stdout) = program.simulate().unwrap();
        let expected_stdout = [9, 14, 18].map(BFieldElement::new).to_vec();

        assert_eq!(expected_stdout, actual_stdout);
    }

    #[test]
    fn xxmul_test() {
        let stdin_words = vec![2, 3, 5, 7, 11, 13];
        let xxmul_code = "
            read_io read_io read_io
            read_io read_io read_io
            xxmul
            swap 2
            write_io write_io write_io
            halt
        ";
        let program = SourceCodeAndInput {
            source_code: xxmul_code.to_string(),
            input: stdin_words,
            secret_input: vec![],
        };

        let (_, actual_stdout) = program.simulate().unwrap();
        let expected_stdout = [108, 123, 22].map(BFieldElement::new).to_vec();

        assert_eq!(expected_stdout, actual_stdout);
    }

    #[test]
    fn xinv_test() {
        let stdin_words = vec![2, 3, 5];
        let xinv_code = "
            read_io read_io read_io
            dup 2 dup 2 dup 2
            dup 2 dup 2 dup 2
            xinvert xxmul
            swap 2
            write_io write_io write_io
            xinvert
            swap 2
            write_io write_io write_io
            halt";
        let program = SourceCodeAndInput {
            source_code: xinv_code.to_string(),
            input: stdin_words,
            secret_input: vec![],
        };

        let (_, actual_stdout) = program.simulate().unwrap();
        let expected_stdout = vec![
            BFieldElement::zero(),
            BFieldElement::zero(),
            BFieldElement::one(),
            BFieldElement::new(16360893149904808002),
            BFieldElement::new(14209859389160351173),
            BFieldElement::new(4432433203958274678),
        ];

        assert_eq!(expected_stdout, actual_stdout);
    }

    #[test]
    fn xbmul_test() {
        let stdin_words = vec![2, 3, 5, 7];
        let xbmul_code: &str = "
            read_io read_io read_io
            read_io
            xbmul
            swap 2
            write_io write_io write_io
            halt";
        let program = SourceCodeAndInput {
            source_code: xbmul_code.to_string(),
            input: stdin_words,
            secret_input: vec![],
        };

        let (_, actual_stdout) = program.simulate().unwrap();
        let expected_stdout = [14, 21, 35].map(BFieldElement::new).to_vec();

        assert_eq!(expected_stdout, actual_stdout);
    }

    #[test]
    fn pseudo_sub_test() {
        let (_, actual_stdout) = SourceCodeAndInput::without_input(
            "
            push 7 push 19 call sub write_io halt
            sub:
                swap 1 push -1 mul add return
            ",
        )
        .simulate()
        .unwrap();
        let expected_stdout = vec![BFieldElement::new(12)];

        assert_eq!(expected_stdout, actual_stdout);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn tvm_op_stack_big_enough_test() {
        assert!(
            DIGEST_LENGTH <= OP_STACK_REG_COUNT,
            "The OpStack must be large enough to hold a single digest"
        );
    }

    #[test]
    fn run_tvm_hello_world_1_test() {
        let code = "
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
        ";
        let program = Program::from_code(code).unwrap();
        let (aet, _) = simulate(&program, vec![], vec![]).unwrap();
        let last_processor_row = aet.processor_trace.rows().into_iter().last().unwrap();
        let st0 = last_processor_row[ProcessorBaseTableColumn::ST0.base_table_index()];
        assert_eq!(BFIELD_ZERO, st0);

        println!("{last_processor_row}");
    }

    #[test]
    fn run_tvm_halt_then_do_stuff_test() {
        let halt_then_do_stuff = "halt push 1 push 2 add invert write_io";
        let program = Program::from_code(halt_then_do_stuff).unwrap();
        let (aet, _) = simulate(&program, vec![], vec![]).unwrap();

        let last_processor_row = aet.processor_trace.rows().into_iter().last().unwrap();
        let clk_count = last_processor_row[ProcessorBaseTableColumn::CLK.base_table_index()];
        assert_eq!(BFIELD_ZERO, clk_count);

        let last_instruction = last_processor_row[ProcessorBaseTableColumn::CI.base_table_index()];
        assert_eq!(Instruction::Halt.opcode_b(), last_instruction);
    }

    #[test]
    fn run_tvm_basic_ram_read_write_test() {
        let basic_ram_read_write_code = "
            push  5 push  6 write_mem pop
            push 15 push 16 write_mem pop
            push  5         read_mem  pop pop
            push 15         read_mem  pop pop
            push  5 push  7 write_mem pop
            push 15         read_mem
            push  5         read_mem
            halt
            ";
        let program = Program::from_code(basic_ram_read_write_code).unwrap();
        let (aet, _) = simulate(&program, vec![], vec![]).unwrap();

        let st0_idx = ProcessorBaseTableColumn::ST0.base_table_index();
        let st1_idx = ProcessorBaseTableColumn::ST1.base_table_index();
        let st2_idx = ProcessorBaseTableColumn::ST2.base_table_index();
        let st3_idx = ProcessorBaseTableColumn::ST3.base_table_index();
        let last_processor_row = aet.processor_trace.rows().into_iter().last().unwrap();
        assert_eq!(BFieldElement::new(7), last_processor_row[st0_idx]);
        assert_eq!(BFieldElement::new(5), last_processor_row[st1_idx]);
        assert_eq!(BFieldElement::new(16), last_processor_row[st2_idx]);
        assert_eq!(BFieldElement::new(15), last_processor_row[st3_idx]);
    }

    #[test]
    fn run_tvm_edgy_ram_writes_test() {
        let edgy_ram_writes_code = "
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
        ";
        let program = Program::from_code(edgy_ram_writes_code).unwrap();
        let (aet, _) = simulate(&program, vec![], vec![]).unwrap();

        let st0_idx = ProcessorBaseTableColumn::ST0.base_table_index();
        let st1_idx = ProcessorBaseTableColumn::ST1.base_table_index();
        let st2_idx = ProcessorBaseTableColumn::ST2.base_table_index();
        let last_processor_row = aet.processor_trace.rows().into_iter().last().unwrap();
        assert_eq!(BFieldElement::new(3), last_processor_row[st0_idx]);
        assert_eq!(BFieldElement::new(5), last_processor_row[st1_idx]);
        assert_eq!(BFieldElement::new(3), last_processor_row[st2_idx]);
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
        ";
        let program = Program::from_code(sample_weights_code).unwrap();
        println!("Successfully parsed the program.");
        let public_input = vec![BFieldElement::new(11)];
        let (aet, _) = simulate(&program, public_input, vec![]).unwrap();

        let last_processor_row = aet.processor_trace.rows().into_iter().last().unwrap();
        let last_instruction = last_processor_row[ProcessorBaseTableColumn::CI.base_table_index()];
        assert_eq!(Instruction::Halt.opcode_b(), last_instruction);
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
        "push 0 swap 1 write_mem pop ",             // store number of APs at RAM address 0
        "",                                         // stack: []
        "read_io read_io read_io read_io read_io ", // read Merkle root
        "",                                         // stack: [r4 r3 r2 r1 r0]
        "call check_aps ",                          //
        "pop pop pop pop pop ",                     // leave clean stack: Merkle root
        "",                                         // stack: []
        "halt ",                                    // done – should be “return”
        "",
        "",                            // subroutine: check AP one at a time
        "",                            // stack before: [* r4 r3 r2 r1 r0]
        "",                            // stack after: [* r4 r3 r2 r1 r0]
        "check_aps: ",                 // start function description:
        "push 0 read_mem dup 0 ",      // get number of APs left to check
        "",                            // stack: [* r4 r3 r2 r1 r0 0 num_left num_left]
        "push 0 eq ",                  // see if there are authentication paths left
        "",                            // stack: [* r4 r3 r2 r1 r0 0 num_left num_left==0]
        "skiz return ",                // return if no authentication paths left
        "push -1 add write_mem pop ",  // decrease number of authentication paths left to check
        "",                            // stack: [* r4 r3 r2 r1 r0]
        "call get_idx_and_hash_leaf ", //
        "",                            // stack: [* r4 r3 r2 r1 r0 idx d4 d3 d2 d1 d0 0 0 0 0 0]
        "call traverse_tree ",         //
        "",                            // stack: [* r4 r3 r2 r1 r0 idx>>2 - - - - - - - - - -]
        "call assert_tree_top ",       //
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
        "",                              // subroutine: go up tree
        "",                              // stack before: [* idx - - - - - - - - - -]
        "",                              // stack after: [* idx>>2 - - - - - - - - - -]
        "traverse_tree: ",               // start function description:
        "dup 10 push 1 eq skiz return ", // break loop if node index is 1
        "divine_sibling hash recurse ",  // move up one level in the Merkle tree
        "",
        "",                     // subroutine: compare digests
        "",                     // stack before: [* r4 r3 r2 r1 r0 idx a b c d e - - - - -]
        "",                     // stack after: [* r4 r3 r2 r1 r0]
        "assert_tree_top: ",    // start function description:
        "pop pop pop pop pop ", // remove unnecessary “0”s from hashing
        "",                     // stack: [* r4 r3 r2 r1 r0 idx a b c d e]
        "swap 1 swap 2 swap 3 swap 4 swap 5 ",
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
        type H = Tip5;

        const NUM_LEAFS: usize = 64;
        let leafs: [Digest; NUM_LEAFS] = random_elements_array();
        let zero_padding: Digest = Digest::new([BFieldElement::zero(); DIGEST_LENGTH]);
        let digests = leafs
            .iter()
            .map(|leaf| H::hash_pair(&zero_padding, leaf))
            .collect_vec();
        let merkle_tree: MerkleTree<H> = MTMaker::from_digests(&digests);
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

        let public_input = vec![
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
        let (aet, _) = simulate(&program, public_input, secret_input).unwrap();

        let last_processor_row = aet.processor_trace.rows().into_iter().last().unwrap();
        let last_instruction = last_processor_row[ProcessorBaseTableColumn::CI.base_table_index()];
        assert_eq!(Instruction::Halt.opcode_b(), last_instruction);
    }

    #[test]
    fn run_tvm_get_colinear_y_test() {
        // see also: get_colinear_y in src/shared_math/polynomial.rs
        let get_colinear_y_code = "
            read_io                         // p2_x
            read_io read_io                 // p1_y p1_x
            read_io read_io                 // p0_y p0_x
            swap 3 push -1 mul dup 1 add    // dy = p0_y - p1_y
            dup 3 push -1 mul dup 5 add mul // dy·(p2_x - p0_x)
            dup 3 dup 3 push -1 mul add     // dx = p0_x - p1_x
            invert mul add                  // compute result
            swap 3 pop pop pop              // leave a clean stack
            write_io halt
        ";

        let program = Program::from_code(get_colinear_y_code).unwrap();
        println!("Successfully parsed the program.");
        let public_input = [7, 2, 1, 3, 4].map(BFieldElement::new).to_vec();
        let (aet, out) = simulate(&program, public_input, vec![]).unwrap();
        assert_eq!(BFieldElement::new(4), out[0]);

        let last_processor_row = aet.processor_trace.rows().into_iter().last().unwrap();
        let last_instruction = last_processor_row[ProcessorBaseTableColumn::CI.base_table_index()];
        assert_eq!(Instruction::Halt.opcode_b(), last_instruction);
    }

    #[test]
    fn run_tvm_countdown_from_10_test() {
        let countdown_code = "
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
            ";

        let program = Program::from_code(countdown_code).unwrap();
        let standard_out = run(&program, vec![], vec![]).unwrap();
        let expected = (0..=10).map(BFieldElement::new).rev().collect_vec();
        assert_eq!(expected, standard_out);
    }

    #[test]
    fn run_tvm_fibonacci_tvm() {
        let code = FIBONACCI_SEQUENCE;
        let program = Program::from_code(code).unwrap();
        let standard_out = run(&program, vec![7_u64.into()], vec![]).unwrap();
        assert_eq!(Some(&BFieldElement::new(21)), standard_out.get(0));
    }

    #[test]
    fn run_tvm_swap_test() {
        let code = "push 1 push 2 swap 1 assert write_io halt";
        let program = Program::from_code(code).unwrap();
        let standard_out = run(&program, vec![], vec![]).unwrap();
        assert_eq!(BFieldElement::new(2), standard_out[0]);
    }

    #[test]
    fn read_mem_unitialized() {
        let program = Program::from_code("read_mem halt").unwrap();
        let (aet, _) = simulate(&program, vec![], vec![]).unwrap();
        assert_eq!(2, aet.processor_trace.nrows());
    }

    #[test]
    fn program_without_halt_test() {
        let program = Program::from_code("nop").unwrap();
        let err = simulate(&program, vec![], vec![]).err();
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
        let program = Program::from_code(VERIFY_SUDOKU).unwrap();
        let stdin = [
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
        .map(BFieldElement::new)
        .to_vec();
        let secret_in = vec![];
        if let Err(e) = simulate(&program, stdin, secret_in) {
            panic!("The VM encountered an error: {e}");
        }

        // rows and columns adhere to Sudoku rules, boxes do not
        let bad_stdin = [
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
        .map(BFieldElement::new)
        .to_vec();
        let secret_in = vec![];
        let err = simulate(&program, bad_stdin, secret_in).err();
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
