use std::ops::AddAssign;

use air::AIR;
use air::table::TableId;
use air::table::hash::HashTable;
use air::table::hash::PermutationTrace;
use air::table::op_stack;
use air::table::processor;
use air::table::ram;
use air::table_column::HashMainColumn::CI;
use air::table_column::MasterMainColumn;
use arbitrary::Arbitrary;
use indexmap::IndexMap;
use indexmap::map::Entry::Occupied;
use indexmap::map::Entry::Vacant;
use isa::error::InstructionError;
use isa::error::InstructionError::InstructionPointerOverflow;
use isa::instruction::Instruction;
use isa::program::Program;
use itertools::Itertools;
use ndarray::Array2;
use ndarray::s;
use strum::EnumCount;
use strum::IntoEnumIterator;
use twenty_first::prelude::*;

use crate::ndarray_helper::ROW_AXIS;
use crate::table;
use crate::table::op_stack::OpStackTableEntry;
use crate::table::ram::RamTableCall;
use crate::table::u32::U32TableEntry;
use crate::vm::CoProcessorCall;
use crate::vm::VMState;

/// An Algebraic Execution Trace (AET) is the primary witness required for proof
/// generation. It holds every intermediate state of the processor and all
/// co-processors, alongside additional witness information, such as the number
/// of times each instruction has been looked up (equivalently, how often each
/// instruction has been executed).
#[derive(Debug, Clone)]
pub struct AlgebraicExecutionTrace {
    /// The program that was executed in order to generate the trace.
    pub program: Program,

    /// The number of times each instruction has been executed.
    ///
    /// Each instruction in the `program` has one associated entry in
    /// `instruction_multiplicities`, counting the number of times this
    /// specific instruction at that location in the program memory has been
    /// executed.
    pub instruction_multiplicities: Vec<u32>,

    /// Records the state of the processor after each instruction.
    pub processor_trace: Array2<BFieldElement>,

    pub op_stack_underflow_trace: Array2<BFieldElement>,

    pub ram_trace: Array2<BFieldElement>,

    /// The trace of hashing the program whose execution generated this
    /// `AlgebraicExecutionTrace`. The resulting digest
    /// 1. ties a [`Proof`](crate::proof::Proof) to the program it was produced
    ///    from, and
    /// 1. is accessible to the program being executed.
    pub program_hash_trace: Array2<BFieldElement>,

    /// For the `hash` instruction, the hash trace records the internal state of
    /// the Tip5 permutation for each round.
    pub hash_trace: Array2<BFieldElement>,

    /// For the Sponge instructions, i.e., `sponge_init`, `sponge_absorb`,
    /// `sponge_absorb_mem`, and `sponge_squeeze`, the Sponge trace records the
    /// internal state of the Tip5 permutation for each round.
    pub sponge_trace: Array2<BFieldElement>,

    /// The u32 entries hold all pairs of BFieldElements that were written to
    /// the U32 Table, alongside the u32 instruction that was executed at
    /// the time. Additionally, it records how often the instruction was
    /// executed with these arguments.
    //
    // `IndexMap` over `HashMap` for deterministic iteration order. This is not
    // needed for correctness of the STARK.
    pub u32_entries: IndexMap<U32TableEntry, u64>,

    /// Records how often each entry in the cascade table was looked up.
    //
    // `IndexMap` over `HashMap` for the same reasons as for field `u32_entries`
    pub cascade_table_lookup_multiplicities: IndexMap<u16, u64>,

    /// Records how often each entry in the lookup table was looked up.
    pub lookup_table_lookup_multiplicities: [u64; AlgebraicExecutionTrace::LOOKUP_TABLE_HEIGHT],
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub struct TableHeight {
    pub table: TableId,
    pub height: usize,
}

impl AlgebraicExecutionTrace {
    pub(crate) const LOOKUP_TABLE_HEIGHT: usize = 1 << 8;

    pub fn new(program: Program) -> Self {
        const PROCESSOR_WIDTH: usize = <processor::ProcessorTable as AIR>::MainColumn::COUNT;
        const OP_STACK_WIDTH: usize = <op_stack::OpStackTable as AIR>::MainColumn::COUNT;
        const RAM_WIDTH: usize = <ram::RamTable as AIR>::MainColumn::COUNT;
        const HASH_WIDTH: usize = <HashTable as AIR>::MainColumn::COUNT;

        let program_len = program.len_bwords();

        let mut aet = Self {
            program,
            instruction_multiplicities: vec![0_u32; program_len],
            processor_trace: Array2::default([0, PROCESSOR_WIDTH]),
            op_stack_underflow_trace: Array2::default([0, OP_STACK_WIDTH]),
            ram_trace: Array2::default([0, RAM_WIDTH]),
            program_hash_trace: Array2::default([0, HASH_WIDTH]),
            hash_trace: Array2::default([0, HASH_WIDTH]),
            sponge_trace: Array2::default([0, HASH_WIDTH]),
            u32_entries: IndexMap::new(),
            cascade_table_lookup_multiplicities: IndexMap::new(),
            lookup_table_lookup_multiplicities: [0; Self::LOOKUP_TABLE_HEIGHT],
        };
        aet.fill_program_hash_trace();
        aet
    }

    /// The height of the [AET](AlgebraicExecutionTrace) after [padding][pad].
    ///
    /// Guaranteed to be a power of two.
    ///
    /// [pad]: table::master_table::MasterMainTable::pad
    pub fn padded_height(&self) -> usize {
        self.height().height.next_power_of_two()
    }

    /// The height of the [AET](AlgebraicExecutionTrace) before [padding][pad].
    /// Corresponds to the height of the longest table.
    ///
    /// [pad]: table::master_table::MasterMainTable::pad
    pub fn height(&self) -> TableHeight {
        TableId::iter()
            .map(|t| TableHeight::new(t, self.height_of_table(t)))
            .max()
            .unwrap()
    }

    pub fn height_of_table(&self, table: TableId) -> usize {
        let hash_table_height = || {
            self.sponge_trace.nrows() + self.hash_trace.nrows() + self.program_hash_trace.nrows()
        };

        match table {
            TableId::Program => Self::padded_program_length(&self.program),
            TableId::Processor => self.processor_trace.nrows(),
            TableId::OpStack => self.op_stack_underflow_trace.nrows(),
            TableId::Ram => self.ram_trace.nrows(),
            TableId::JumpStack => self.processor_trace.nrows(),
            TableId::Hash => hash_table_height(),
            TableId::Cascade => self.cascade_table_lookup_multiplicities.len(),
            TableId::Lookup => Self::LOOKUP_TABLE_HEIGHT,
            TableId::U32 => self.u32_table_height(),
        }
    }

    /// # Panics
    ///
    /// - if the table height exceeds [`u32::MAX`]
    /// - if the table height exceeds [`usize::MAX`]
    fn u32_table_height(&self) -> usize {
        let entry_len = U32TableEntry::table_height_contribution;
        let height = self.u32_entries.keys().map(entry_len).sum::<u32>();
        height.try_into().unwrap()
    }

    fn padded_program_length(program: &Program) -> usize {
        // Padding is at least one 1.
        // Also note that the Program Table's side of the instruction lookup
        // argument requires at least one padding row to account for the
        // processor's “next instruction or argument.” Both of these are
        // captured by the “+ 1” in the following line.
        (program.len_bwords() + 1).next_multiple_of(Tip5::RATE)
    }

    /// Hash the program and record the entire Sponge's trace for program
    /// attestation.
    fn fill_program_hash_trace(&mut self) {
        let padded_program = Self::hash_input_pad_program(&self.program);
        let mut program_sponge = Tip5::init();
        for chunk_to_absorb in padded_program.chunks(Tip5::RATE) {
            program_sponge.state[..Tip5::RATE]
                .iter_mut()
                .zip_eq(chunk_to_absorb)
                .for_each(|(sponge_state_elem, &absorb_elem)| *sponge_state_elem = absorb_elem);
            let hash_trace = program_sponge.trace();
            let trace_addendum = table::hash::trace_to_table_rows(hash_trace);

            self.increase_lookup_multiplicities(hash_trace);
            self.program_hash_trace
                .append(ROW_AXIS, trace_addendum.view())
                .expect("shapes must be identical");
        }

        let instruction_column_index = CI.main_index();
        let mut instruction_column = self.program_hash_trace.column_mut(instruction_column_index);
        instruction_column.fill(Instruction::Hash.opcode_b());

        // consistency check
        let program_digest = program_sponge.state[..Digest::LEN].try_into().unwrap();
        let program_digest = Digest::new(program_digest);
        let expected_digest = self.program.hash();
        assert_eq!(expected_digest, program_digest);
    }

    fn hash_input_pad_program(program: &Program) -> Vec<BFieldElement> {
        let padded_program_length = Self::padded_program_length(program);

        // padding is one 1, then as many zeros as necessary: [1, 0, 0, …]
        let program_iter = program.to_bwords().into_iter();
        let one = bfe_array![1];
        let zeros = bfe_array![0; tip5::RATE];
        program_iter
            .chain(one)
            .chain(zeros)
            .take(padded_program_length)
            .collect()
    }

    pub(crate) fn record_state(&mut self, state: &VMState) -> Result<(), InstructionError> {
        self.record_instruction_lookup(state.instruction_pointer)?;
        self.append_state_to_processor_trace(state);
        Ok(())
    }

    fn record_instruction_lookup(
        &mut self,
        instruction_pointer: usize,
    ) -> Result<(), InstructionError> {
        if instruction_pointer >= self.instruction_multiplicities.len() {
            return Err(InstructionPointerOverflow);
        }
        self.instruction_multiplicities[instruction_pointer] += 1;
        Ok(())
    }

    fn append_state_to_processor_trace(&mut self, state: &VMState) {
        self.processor_trace
            .push_row(state.to_processor_row().view())
            .unwrap()
    }

    pub(crate) fn record_co_processor_call(&mut self, co_processor_call: CoProcessorCall) {
        match co_processor_call {
            CoProcessorCall::Tip5Trace(Instruction::Hash, trace) => self.append_hash_trace(*trace),
            CoProcessorCall::SpongeStateReset => self.append_initial_sponge_state(),
            CoProcessorCall::Tip5Trace(instruction, trace) => {
                self.append_sponge_trace(instruction, *trace)
            }
            CoProcessorCall::U32(u32_entry) => self.record_u32_table_entry(u32_entry),
            CoProcessorCall::OpStack(op_stack_entry) => self.record_op_stack_entry(op_stack_entry),
            CoProcessorCall::Ram(ram_call) => self.record_ram_call(ram_call),
        }
    }

    fn append_hash_trace(&mut self, trace: PermutationTrace) {
        self.increase_lookup_multiplicities(trace);
        let mut hash_trace_addendum = table::hash::trace_to_table_rows(trace);
        hash_trace_addendum
            .slice_mut(s![.., CI.main_index()])
            .fill(Instruction::Hash.opcode_b());
        self.hash_trace
            .append(ROW_AXIS, hash_trace_addendum.view())
            .expect("shapes must be identical");
    }

    fn append_initial_sponge_state(&mut self) {
        let round_number = 0;
        let initial_state = Tip5::init().state;
        let mut hash_table_row = table::hash::trace_row_to_table_row(initial_state, round_number);
        hash_table_row[CI.main_index()] = Instruction::SpongeInit.opcode_b();
        self.sponge_trace.push_row(hash_table_row.view()).unwrap();
    }

    fn append_sponge_trace(&mut self, instruction: Instruction, trace: PermutationTrace) {
        assert!(matches!(
            instruction,
            Instruction::SpongeAbsorb | Instruction::SpongeSqueeze
        ));
        self.increase_lookup_multiplicities(trace);
        let mut sponge_trace_addendum = table::hash::trace_to_table_rows(trace);
        sponge_trace_addendum
            .slice_mut(s![.., CI.main_index()])
            .fill(instruction.opcode_b());
        self.sponge_trace
            .append(ROW_AXIS, sponge_trace_addendum.view())
            .expect("shapes must be identical");
    }

    /// Given a trace of the hash function's permutation, determines how often
    /// each entry in the
    /// - cascade table was looked up, and
    /// - lookup table was looked up;
    ///
    /// and increases the multiplicities accordingly
    fn increase_lookup_multiplicities(&mut self, trace: PermutationTrace) {
        // The last row in the trace is the permutation's result: no lookups are
        // performed for it.
        let rows_for_which_lookups_are_performed = trace.iter().dropping_back(1);
        for row in rows_for_which_lookups_are_performed {
            self.increase_lookup_multiplicities_for_row(row);
        }
    }

    /// Given one row of the hash function's permutation trace, increase the
    /// multiplicities of the relevant entries in the cascade table and/or
    /// the lookup table.
    fn increase_lookup_multiplicities_for_row(&mut self, row: &[BFieldElement; tip5::STATE_SIZE]) {
        for &state_element in &row[0..tip5::NUM_SPLIT_AND_LOOKUP] {
            self.increase_lookup_multiplicities_for_state_element(state_element);
        }
    }

    /// Given one state element, increase the multiplicities of the
    /// corresponding entries in the cascade table and/or the lookup table.
    fn increase_lookup_multiplicities_for_state_element(&mut self, state_element: BFieldElement) {
        for limb in table::hash::base_field_element_into_16_bit_limbs(state_element) {
            match self.cascade_table_lookup_multiplicities.entry(limb) {
                Occupied(mut cascade_table_entry) => *cascade_table_entry.get_mut() += 1,
                Vacant(cascade_table_entry) => {
                    cascade_table_entry.insert(1);
                    self.increase_lookup_table_multiplicities_for_limb(limb);
                }
            }
        }
    }

    /// Given one 16-bit limb, increase the multiplicities of the corresponding
    /// entries in the lookup table.
    fn increase_lookup_table_multiplicities_for_limb(&mut self, limb: u16) {
        let limb_lo = limb & 0xff;
        let limb_hi = (limb >> 8) & 0xff;
        self.lookup_table_lookup_multiplicities[limb_lo as usize] += 1;
        self.lookup_table_lookup_multiplicities[limb_hi as usize] += 1;
    }

    fn record_u32_table_entry(&mut self, u32_entry: U32TableEntry) {
        self.u32_entries.entry(u32_entry).or_insert(0).add_assign(1)
    }

    fn record_op_stack_entry(&mut self, op_stack_entry: OpStackTableEntry) {
        let op_stack_table_row = op_stack_entry.to_main_table_row();
        self.op_stack_underflow_trace
            .push_row(op_stack_table_row.view())
            .unwrap();
    }

    fn record_ram_call(&mut self, ram_call: RamTableCall) {
        self.ram_trace
            .push_row(ram_call.to_table_row().view())
            .unwrap();
    }
}

impl TableHeight {
    fn new(table: TableId, height: usize) -> Self {
        Self { table, height }
    }
}

impl PartialOrd for TableHeight {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TableHeight {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.height.cmp(&other.height)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use assert2::assert;
    use isa::triton_asm;
    use isa::triton_program;

    use super::*;
    use crate::prelude::*;

    #[test]
    fn pad_program_requiring_no_padding_zeros() {
        let eight_nops = triton_asm![nop; 8];
        let program = triton_program!({&eight_nops} halt);
        let padded_program = AlgebraicExecutionTrace::hash_input_pad_program(&program);

        let expected = [program.to_bwords(), bfe_vec![1]].concat();
        assert!(expected == padded_program);
    }

    #[test]
    fn height_of_any_table_can_be_computed() {
        let program = triton_program!(halt);
        let (aet, _) =
            VM::trace_execution(program, PublicInput::default(), NonDeterminism::default())
                .unwrap();

        let _ = aet.height();
        for table in TableId::iter() {
            let _ = aet.height_of_table(table);
        }
    }
}
