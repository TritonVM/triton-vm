use std::cmp::max;
use std::ops::Mul;

use itertools::Itertools;
use ndarray::parallel::prelude::*;
use ndarray::*;
use num_traits::One;
use num_traits::Zero;
use strum::EnumCount;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::instruction::AnInstruction::*;
use crate::instruction::Instruction;
use crate::instruction::InstructionBit;
use crate::instruction::ALL_INSTRUCTIONS;
use crate::op_stack::NumberOfWords;
use crate::op_stack::OpStackElement;
use crate::table::challenges::ChallengeId;
use crate::table::challenges::ChallengeId::*;
use crate::table::challenges::Challenges;
use crate::table::constraint_circuit::DualRowIndicator::*;
use crate::table::constraint_circuit::SingleRowIndicator::*;
use crate::table::constraint_circuit::*;
use crate::table::cross_table_argument::*;
use crate::table::ram_table;
use crate::table::table_column::ProcessorBaseTableColumn::*;
use crate::table::table_column::ProcessorExtTableColumn::*;
use crate::table::table_column::*;

pub const BASE_WIDTH: usize = ProcessorBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = ProcessorExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ProcessorTable;

impl ProcessorTable {
    pub fn fill_trace(
        processor_table: &mut ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
        clk_jump_diffs_op_stack: &[BFieldElement],
        clk_jump_diffs_ram: &[BFieldElement],
        clk_jump_diffs_jump_stack: &[BFieldElement],
    ) {
        let num_rows = aet.processor_trace.nrows();
        let mut clk_jump_diff_multiplicities = Array1::zeros([num_rows]);
        for clk_jump_diff in clk_jump_diffs_op_stack
            .iter()
            .chain(clk_jump_diffs_ram)
            .chain(clk_jump_diffs_jump_stack)
        {
            let clk = clk_jump_diff.value() as usize;
            clk_jump_diff_multiplicities[clk] += b_field_element::BFIELD_ONE;
        }

        let mut processor_table = processor_table.slice_mut(s![0..num_rows, ..]);
        processor_table.assign(&aet.processor_trace);
        processor_table
            .column_mut(ClockJumpDifferenceLookupMultiplicity.base_table_index())
            .assign(&clk_jump_diff_multiplicities);
    }

    pub fn pad_trace(
        mut processor_table: ArrayViewMut2<BFieldElement>,
        processor_table_len: usize,
    ) {
        assert!(
            processor_table_len > 0,
            "Processor Table must have at least one row."
        );
        let mut padding_template = processor_table.row(processor_table_len - 1).to_owned();
        padding_template[IsPadding.base_table_index()] = bfe!(1);
        padding_template[ClockJumpDifferenceLookupMultiplicity.base_table_index()] = bfe!(0);
        processor_table
            .slice_mut(s![processor_table_len.., ..])
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| row.assign(&padding_template));

        let clk_range = processor_table_len..processor_table.nrows();
        let clk_col = Array1::from_iter(clk_range.map(|a| bfe!(a as u64)));
        clk_col.move_into(
            processor_table.slice_mut(s![processor_table_len.., CLK.base_table_index()]),
        );

        // The Jump Stack Table does not have a padding indicator. Hence, clock jump differences are
        // being looked up in its padding sections. The clock jump differences in that section are
        // always 1. The lookup multiplicities of clock value 1 must be increased accordingly: one
        // per padding row.
        let num_padding_rows = processor_table.nrows() - processor_table_len;
        let num_padding_rows = bfe!(num_padding_rows as u64);
        let mut row_1 = processor_table.row_mut(1);
        row_1[ClockJumpDifferenceLookupMultiplicity.base_table_index()] += num_padding_rows;
    }

    pub fn extend(
        base_table: ArrayView2<BFieldElement>,
        mut ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());
        let mut input_table_running_evaluation = EvalArg::default_initial();
        let mut output_table_running_evaluation = EvalArg::default_initial();
        let mut instruction_lookup_log_derivative = LookupArg::default_initial();
        let mut op_stack_table_running_product = PermArg::default_initial();
        let mut ram_table_running_product = PermArg::default_initial();
        let mut jump_stack_running_product = PermArg::default_initial();
        let mut hash_input_running_evaluation = EvalArg::default_initial();
        let mut hash_digest_running_evaluation = EvalArg::default_initial();
        let mut sponge_running_evaluation = EvalArg::default_initial();
        let mut u32_table_running_sum_log_derivative = LookupArg::default_initial();
        let mut clock_jump_diff_lookup_op_stack_log_derivative = LookupArg::default_initial();

        let mut previous_row: Option<ArrayView1<BFieldElement>> = None;
        for row_idx in 0..base_table.nrows() {
            let current_row = base_table.row(row_idx);

            // Standard Input & Output
            if let Some(prev_row) = previous_row {
                let previous_instruction = Self::instruction_from_row(prev_row);
                if let Some(Instruction::ReadIo(st)) = previous_instruction {
                    for i in (0..st.num_words()).rev() {
                        let input_symbol_column = Self::op_stack_column_by_index(i);
                        let input_symbol = current_row[input_symbol_column.base_table_index()];
                        input_table_running_evaluation = input_table_running_evaluation
                            * challenges[StandardInputIndeterminate]
                            + input_symbol;
                    }
                }
                if let Some(Instruction::WriteIo(st)) = previous_instruction {
                    for i in 0..st.num_words() {
                        let output_symbol_column = Self::op_stack_column_by_index(i);
                        let output_symbol = prev_row[output_symbol_column.base_table_index()];
                        output_table_running_evaluation = output_table_running_evaluation
                            * challenges[StandardOutputIndeterminate]
                            + output_symbol;
                    }
                }
            }

            // Program table
            if current_row[IsPadding.base_table_index()].is_zero() {
                let ip = current_row[IP.base_table_index()];
                let ci = current_row[CI.base_table_index()];
                let nia = current_row[NIA.base_table_index()];
                let compressed_row_for_instruction_lookup = ip * challenges[ProgramAddressWeight]
                    + ci * challenges[ProgramInstructionWeight]
                    + nia * challenges[ProgramNextInstructionWeight];
                instruction_lookup_log_derivative += (challenges[InstructionLookupIndeterminate]
                    - compressed_row_for_instruction_lookup)
                    .inverse();
            }

            op_stack_table_running_product *= Self::factor_for_op_stack_table_running_product(
                previous_row,
                current_row,
                challenges,
            );

            if let Some(factor) =
                Self::factor_for_ram_table_running_product(previous_row, current_row, challenges)
            {
                ram_table_running_product *= factor;
            };

            // JumpStack Table
            let clk = current_row[CLK.base_table_index()];
            let ci = current_row[CI.base_table_index()];
            let jsp = current_row[JSP.base_table_index()];
            let jso = current_row[JSO.base_table_index()];
            let jsd = current_row[JSD.base_table_index()];
            let compressed_row_for_jump_stack_table = clk * challenges[JumpStackClkWeight]
                + ci * challenges[JumpStackCiWeight]
                + jsp * challenges[JumpStackJspWeight]
                + jso * challenges[JumpStackJsoWeight]
                + jsd * challenges[JumpStackJsdWeight];
            jump_stack_running_product *=
                challenges[JumpStackIndeterminate] - compressed_row_for_jump_stack_table;

            // Hash Table – Hash's input from Processor to Hash Coprocessor
            let st_0_through_9 = [ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7, ST8, ST9];
            let hash_state_weights = &challenges[HashStateWeight0..HashStateWeight10];
            let compressed_row_for_hash_input_and_sponge_squeeze: XFieldElement = st_0_through_9
                .map(|st| current_row[st.base_table_index()])
                .into_iter()
                .zip_eq(hash_state_weights.iter())
                .map(|(st, &weight)| weight * st)
                .sum();
            let hash_digest_weights = &challenges[HashStateWeight0..HashStateWeight5];
            let compressed_row_for_hash_digest: XFieldElement = st_0_through_9
                [0..tip5::DIGEST_LENGTH]
                .iter()
                .map(|st| current_row[st.base_table_index()])
                .zip_eq(hash_digest_weights.iter())
                .map(|(st, &weight)| weight * st)
                .sum();

            if current_row[CI.base_table_index()] == Instruction::Hash.opcode_b() {
                hash_input_running_evaluation = hash_input_running_evaluation
                    * challenges[HashInputIndeterminate]
                    + compressed_row_for_hash_input_and_sponge_squeeze;
            }

            // Hash Table – Hash's output from Hash Coprocessor to Processor
            if let Some(prev_row) = previous_row {
                if prev_row[CI.base_table_index()] == Instruction::Hash.opcode_b() {
                    hash_digest_running_evaluation = hash_digest_running_evaluation
                        * challenges[HashDigestIndeterminate]
                        + compressed_row_for_hash_digest;
                }
            }

            // Hash Table – Sponge
            if let Some(prev_row) = previous_row {
                if prev_row[CI.base_table_index()] == Instruction::SpongeInit.opcode_b() {
                    sponge_running_evaluation = sponge_running_evaluation
                        * challenges[SpongeIndeterminate]
                        + challenges[HashCIWeight] * Instruction::SpongeInit.opcode_b();
                }

                if prev_row[CI.base_table_index()] == Instruction::SpongeAbsorb.opcode_b() {
                    let compressed_row: XFieldElement = st_0_through_9
                        .map(|st| prev_row[st.base_table_index()])
                        .into_iter()
                        .zip_eq(hash_state_weights.iter())
                        .map(|(st, &weight)| weight * st)
                        .sum();
                    sponge_running_evaluation = sponge_running_evaluation
                        * challenges[SpongeIndeterminate]
                        + challenges[HashCIWeight] * Instruction::SpongeAbsorb.opcode_b()
                        + compressed_row;
                }

                if prev_row[CI.base_table_index()] == Instruction::SpongeSqueeze.opcode_b() {
                    sponge_running_evaluation = sponge_running_evaluation
                        * challenges[SpongeIndeterminate]
                        + challenges[HashCIWeight] * Instruction::SpongeSqueeze.opcode_b()
                        + compressed_row_for_hash_input_and_sponge_squeeze;
                }
            }

            // U32 Table
            if let Some(prev_row) = previous_row {
                let previously_current_instruction = prev_row[CI.base_table_index()];
                if previously_current_instruction == Instruction::Split.opcode_b() {
                    let compressed_row = current_row[ST0.base_table_index()]
                        * challenges[U32LhsWeight]
                        + current_row[ST1.base_table_index()] * challenges[U32RhsWeight]
                        + prev_row[CI.base_table_index()] * challenges[U32CiWeight];
                    u32_table_running_sum_log_derivative +=
                        (challenges[U32Indeterminate] - compressed_row).inverse();
                }
                if previously_current_instruction == Instruction::Lt.opcode_b()
                    || previously_current_instruction == Instruction::And.opcode_b()
                    || previously_current_instruction == Instruction::Pow.opcode_b()
                {
                    let compressed_row = prev_row[ST0.base_table_index()]
                        * challenges[U32LhsWeight]
                        + prev_row[ST1.base_table_index()] * challenges[U32RhsWeight]
                        + prev_row[CI.base_table_index()] * challenges[U32CiWeight]
                        + current_row[ST0.base_table_index()] * challenges[U32ResultWeight];
                    u32_table_running_sum_log_derivative +=
                        (challenges[U32Indeterminate] - compressed_row).inverse();
                }
                if previously_current_instruction == Instruction::Xor.opcode_b() {
                    // Triton VM uses the following equality to compute the results of both the
                    // `and` and `xor` instruction using the u32 coprocessor's `and` capability:
                    //     a ^ b = a + b - 2 · (a & b)
                    // <=> a & b = (a + b - a ^ b) / 2
                    let st0_prev = prev_row[ST0.base_table_index()];
                    let st1_prev = prev_row[ST1.base_table_index()];
                    let st0 = current_row[ST0.base_table_index()];
                    let from_xor_in_processor_to_and_in_u32_coprocessor =
                        (st0_prev + st1_prev - st0) / bfe!(2);
                    let compressed_row = st0_prev * challenges[U32LhsWeight]
                        + st1_prev * challenges[U32RhsWeight]
                        + Instruction::And.opcode_b() * challenges[U32CiWeight]
                        + from_xor_in_processor_to_and_in_u32_coprocessor
                            * challenges[U32ResultWeight];
                    u32_table_running_sum_log_derivative +=
                        (challenges[U32Indeterminate] - compressed_row).inverse();
                }
                if previously_current_instruction == Instruction::Log2Floor.opcode_b()
                    || previously_current_instruction == Instruction::PopCount.opcode_b()
                {
                    let compressed_row = prev_row[ST0.base_table_index()]
                        * challenges[U32LhsWeight]
                        + prev_row[CI.base_table_index()] * challenges[U32CiWeight]
                        + current_row[ST0.base_table_index()] * challenges[U32ResultWeight];
                    u32_table_running_sum_log_derivative +=
                        (challenges[U32Indeterminate] - compressed_row).inverse();
                }
                if previously_current_instruction == Instruction::DivMod.opcode_b() {
                    let compressed_row_for_lt_check = current_row[ST0.base_table_index()]
                        * challenges[U32LhsWeight]
                        + prev_row[ST1.base_table_index()] * challenges[U32RhsWeight]
                        + Instruction::Lt.opcode_b() * challenges[U32CiWeight]
                        + bfe!(1) * challenges[U32ResultWeight];
                    let compressed_row_for_range_check = prev_row[ST0.base_table_index()]
                        * challenges[U32LhsWeight]
                        + current_row[ST1.base_table_index()] * challenges[U32RhsWeight]
                        + Instruction::Split.opcode_b() * challenges[U32CiWeight];
                    u32_table_running_sum_log_derivative +=
                        (challenges[U32Indeterminate] - compressed_row_for_lt_check).inverse();
                    u32_table_running_sum_log_derivative +=
                        (challenges[U32Indeterminate] - compressed_row_for_range_check).inverse();
                }
            }

            // Lookup Argument for clock jump differences
            let lookup_multiplicity =
                current_row[ClockJumpDifferenceLookupMultiplicity.base_table_index()];
            clock_jump_diff_lookup_op_stack_log_derivative +=
                (challenges[ClockJumpDifferenceLookupIndeterminate] - clk).inverse()
                    * lookup_multiplicity;

            let mut extension_row = ext_table.row_mut(row_idx);
            extension_row[InputTableEvalArg.ext_table_index()] = input_table_running_evaluation;
            extension_row[OutputTableEvalArg.ext_table_index()] = output_table_running_evaluation;
            extension_row[InstructionLookupClientLogDerivative.ext_table_index()] =
                instruction_lookup_log_derivative;
            extension_row[OpStackTablePermArg.ext_table_index()] = op_stack_table_running_product;
            extension_row[RamTablePermArg.ext_table_index()] = ram_table_running_product;
            extension_row[JumpStackTablePermArg.ext_table_index()] = jump_stack_running_product;
            extension_row[HashInputEvalArg.ext_table_index()] = hash_input_running_evaluation;
            extension_row[HashDigestEvalArg.ext_table_index()] = hash_digest_running_evaluation;
            extension_row[SpongeEvalArg.ext_table_index()] = sponge_running_evaluation;
            extension_row[U32LookupClientLogDerivative.ext_table_index()] =
                u32_table_running_sum_log_derivative;
            extension_row[ClockJumpDifferenceLookupServerLogDerivative.ext_table_index()] =
                clock_jump_diff_lookup_op_stack_log_derivative;
            previous_row = Some(current_row);
        }
    }

    fn factor_for_op_stack_table_running_product(
        maybe_previous_row: Option<ArrayView1<BFieldElement>>,
        current_row: ArrayView1<BFieldElement>,
        challenges: &Challenges,
    ) -> XFieldElement {
        let default_factor = xfe!(1);

        let is_padding_row = current_row[IsPadding.base_table_index()].is_one();
        if is_padding_row {
            return default_factor;
        }

        let Some(previous_row) = maybe_previous_row else {
            return default_factor;
        };

        let Some(previous_instruction) = Self::instruction_from_row(previous_row) else {
            return default_factor;
        };

        // shorter stack means relevant information is on top of stack, i.e., in stack registers
        let row_with_shorter_stack = if previous_instruction.op_stack_size_influence() > 0 {
            previous_row.view()
        } else {
            current_row.view()
        };
        let op_stack_delta = previous_instruction
            .op_stack_size_influence()
            .unsigned_abs() as usize;

        let mut factor = default_factor;
        for op_stack_pointer_offset in 0..op_stack_delta {
            let max_stack_element_index = OpStackElement::COUNT - 1;
            let stack_element_index = max_stack_element_index - op_stack_pointer_offset;
            let stack_element_column = Self::op_stack_column_by_index(stack_element_index);
            let underflow_element = row_with_shorter_stack[stack_element_column.base_table_index()];

            let op_stack_pointer = row_with_shorter_stack[OpStackPointer.base_table_index()];
            let offset = bfe!(op_stack_pointer_offset as u64);
            let offset_op_stack_pointer = op_stack_pointer + offset;

            let clk = previous_row[CLK.base_table_index()];
            let ib1_shrink_stack = previous_row[IB1.base_table_index()];
            let compressed_row = clk * challenges[OpStackClkWeight]
                + ib1_shrink_stack * challenges[OpStackIb1Weight]
                + offset_op_stack_pointer * challenges[OpStackPointerWeight]
                + underflow_element * challenges[OpStackFirstUnderflowElementWeight];
            factor *= challenges[OpStackIndeterminate] - compressed_row;
        }
        factor
    }

    fn factor_for_ram_table_running_product(
        maybe_previous_row: Option<ArrayView1<BFieldElement>>,
        current_row: ArrayView1<BFieldElement>,
        challenges: &Challenges,
    ) -> Option<XFieldElement> {
        let is_padding_row = current_row[IsPadding.base_table_index()].is_one();
        if is_padding_row {
            return None;
        }

        let previous_row = maybe_previous_row?;
        let instruction = Self::instruction_from_row(previous_row)?;

        let instruction_type = match instruction {
            ReadMem(_) => ram_table::INSTRUCTION_TYPE_READ,
            WriteMem(_) => ram_table::INSTRUCTION_TYPE_WRITE,
            _ => return None,
        };

        // longer stack means relevant information is on top of stack, i.e., in stack registers
        let row_with_longer_stack = match instruction {
            ReadMem(_) => current_row.view(),
            WriteMem(_) => previous_row.view(),
            _ => unreachable!(),
        };
        let op_stack_delta = instruction.op_stack_size_influence().unsigned_abs() as usize;

        let mut factor = xfe!(1);
        for ram_pointer_offset in 0..op_stack_delta {
            let num_ram_pointers = 1;
            let ram_value_index = ram_pointer_offset + num_ram_pointers;
            let ram_value_column = Self::op_stack_column_by_index(ram_value_index);
            let ram_value = row_with_longer_stack[ram_value_column.base_table_index()];
            let offset_ram_pointer =
                Self::offset_ram_pointer(instruction, row_with_longer_stack, ram_pointer_offset);

            let clk = previous_row[CLK.base_table_index()];
            let compressed_row = clk * challenges[RamClkWeight]
                + instruction_type * challenges[RamInstructionTypeWeight]
                + offset_ram_pointer * challenges[RamPointerWeight]
                + ram_value * challenges[RamValueWeight];
            factor *= challenges[RamIndeterminate] - compressed_row;
        }
        Some(factor)
    }

    fn offset_ram_pointer(
        instruction: Instruction,
        row_with_longer_stack: ArrayView1<BFieldElement>,
        ram_pointer_offset: usize,
    ) -> BFieldElement {
        let ram_pointer = row_with_longer_stack[ST0.base_table_index()];
        let offset = bfe!(ram_pointer_offset as u64);

        match instruction {
            // adjust for ram_pointer pointing in front of last-read address:
            // `push 0 read_mem 1` leaves stack as `_ a -1` where `a` was read from address 0.
            ReadMem(_) => ram_pointer + offset + bfe!(1),
            WriteMem(_) => ram_pointer + offset,
            _ => unreachable!(),
        }
    }

    fn instruction_from_row(row: ArrayView1<BFieldElement>) -> Option<Instruction> {
        let opcode = row[CI.base_table_index()];
        let instruction = Instruction::try_from(opcode).ok()?;

        if instruction.arg().is_some() {
            let arg = row[NIA.base_table_index()];
            return instruction.change_arg(arg).ok();
        }

        Some(instruction)
    }

    fn op_stack_column_by_index(index: usize) -> ProcessorBaseTableColumn {
        match index {
            0 => ST0,
            1 => ST1,
            2 => ST2,
            3 => ST3,
            4 => ST4,
            5 => ST5,
            6 => ST6,
            7 => ST7,
            8 => ST8,
            9 => ST9,
            10 => ST10,
            11 => ST11,
            12 => ST12,
            13 => ST13,
            14 => ST14,
            15 => ST15,
            i => panic!("Op Stack column index must be in [0, 15], not {i}."),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ExtProcessorTable;

impl ExtProcessorTable {
    pub fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let x_constant = |x| circuit_builder.x_constant(x);
        let challenge = |c| circuit_builder.challenge(c);
        let base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(BaseRow(col.master_base_table_index()))
        };
        let ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(ExtRow(col.master_ext_table_index()))
        };

        let clk_is_0 = base_row(CLK);
        let ip_is_0 = base_row(IP);
        let jsp_is_0 = base_row(JSP);
        let jso_is_0 = base_row(JSO);
        let jsd_is_0 = base_row(JSD);
        let st0_is_0 = base_row(ST0);
        let st1_is_0 = base_row(ST1);
        let st2_is_0 = base_row(ST2);
        let st3_is_0 = base_row(ST3);
        let st4_is_0 = base_row(ST4);
        let st5_is_0 = base_row(ST5);
        let st6_is_0 = base_row(ST6);
        let st7_is_0 = base_row(ST7);
        let st8_is_0 = base_row(ST8);
        let st9_is_0 = base_row(ST9);
        let st10_is_0 = base_row(ST10);
        let op_stack_pointer_is_16 = base_row(OpStackPointer) - constant(16);

        // Compress the program digest using an Evaluation Argument.
        // Lowest index in the digest corresponds to lowest index on the stack.
        let program_digest: [_; tip5::DIGEST_LENGTH] = [
            base_row(ST11),
            base_row(ST12),
            base_row(ST13),
            base_row(ST14),
            base_row(ST15),
        ];
        let compressed_program_digest = program_digest.into_iter().fold(
            circuit_builder.x_constant(EvalArg::default_initial()),
            |acc, digest_element| {
                acc * challenge(CompressProgramDigestIndeterminate) + digest_element
            },
        );
        let compressed_program_digest_is_expected_program_digest =
            compressed_program_digest - challenge(CompressedProgramDigest);

        // Permutation and Evaluation Arguments with all tables the Processor Table relates to

        // standard input
        let running_evaluation_for_standard_input_is_initialized_correctly =
            ext_row(InputTableEvalArg) - x_constant(EvalArg::default_initial());

        // program table
        let instruction_lookup_indeterminate = challenge(InstructionLookupIndeterminate);
        let instruction_ci_weight = challenge(ProgramInstructionWeight);
        let instruction_nia_weight = challenge(ProgramNextInstructionWeight);
        let compressed_row_for_instruction_lookup =
            instruction_ci_weight * base_row(CI) + instruction_nia_weight * base_row(NIA);
        let instruction_lookup_log_derivative_is_initialized_correctly =
            (ext_row(InstructionLookupClientLogDerivative)
                - x_constant(LookupArg::default_initial()))
                * (instruction_lookup_indeterminate - compressed_row_for_instruction_lookup)
                - constant(1);

        // standard output
        let running_evaluation_for_standard_output_is_initialized_correctly =
            ext_row(OutputTableEvalArg) - x_constant(EvalArg::default_initial());

        let running_product_for_op_stack_table_is_initialized_correctly =
            ext_row(OpStackTablePermArg) - x_constant(PermArg::default_initial());

        // ram table
        let running_product_for_ram_table_is_initialized_correctly =
            ext_row(RamTablePermArg) - x_constant(PermArg::default_initial());

        // jump-stack table
        let jump_stack_indeterminate = challenge(JumpStackIndeterminate);
        let jump_stack_ci_weight = challenge(JumpStackCiWeight);
        // note: `clk`, `jsp`, `jso`, and `jsd` are already constrained to be 0.
        let compressed_row_for_jump_stack_table = jump_stack_ci_weight * base_row(CI);
        let running_product_for_jump_stack_table_is_initialized_correctly =
            ext_row(JumpStackTablePermArg)
                - x_constant(PermArg::default_initial())
                    * (jump_stack_indeterminate - compressed_row_for_jump_stack_table);

        // clock jump difference lookup argument
        // A clock jump difference of 0 is illegal. Hence, the initial is recorded.
        let clock_jump_diff_lookup_log_derivative_is_initialized_correctly =
            ext_row(ClockJumpDifferenceLookupServerLogDerivative)
                - x_constant(LookupArg::default_initial());

        // from processor to hash table
        let hash_selector = base_row(CI) - constant(Instruction::Hash.opcode());
        let hash_deselector =
            Self::instruction_deselector_single_row(circuit_builder, Instruction::Hash);
        let hash_input_indeterminate = challenge(HashInputIndeterminate);
        // the opStack is guaranteed to be initialized to 0 by virtue of other initial constraints
        let compressed_row = constant(0);
        let running_evaluation_hash_input_has_absorbed_first_row = ext_row(HashInputEvalArg)
            - hash_input_indeterminate * x_constant(EvalArg::default_initial())
            - compressed_row;
        let running_evaluation_hash_input_is_default_initial =
            ext_row(HashInputEvalArg) - x_constant(EvalArg::default_initial());
        let running_evaluation_hash_input_is_initialized_correctly = hash_selector
            * running_evaluation_hash_input_is_default_initial
            + hash_deselector * running_evaluation_hash_input_has_absorbed_first_row;

        // from hash table to processor
        let running_evaluation_hash_digest_is_initialized_correctly =
            ext_row(HashDigestEvalArg) - x_constant(EvalArg::default_initial());

        // Hash Table – Sponge
        let running_evaluation_sponge_absorb_is_initialized_correctly =
            ext_row(SpongeEvalArg) - x_constant(EvalArg::default_initial());

        // u32 table
        let running_sum_log_derivative_for_u32_table_is_initialized_correctly =
            ext_row(U32LookupClientLogDerivative) - x_constant(LookupArg::default_initial());

        vec![
            clk_is_0,
            ip_is_0,
            jsp_is_0,
            jso_is_0,
            jsd_is_0,
            st0_is_0,
            st1_is_0,
            st2_is_0,
            st3_is_0,
            st4_is_0,
            st5_is_0,
            st6_is_0,
            st7_is_0,
            st8_is_0,
            st9_is_0,
            st10_is_0,
            compressed_program_digest_is_expected_program_digest,
            op_stack_pointer_is_16,
            running_evaluation_for_standard_input_is_initialized_correctly,
            instruction_lookup_log_derivative_is_initialized_correctly,
            running_evaluation_for_standard_output_is_initialized_correctly,
            running_product_for_op_stack_table_is_initialized_correctly,
            running_product_for_ram_table_is_initialized_correctly,
            running_product_for_jump_stack_table_is_initialized_correctly,
            clock_jump_diff_lookup_log_derivative_is_initialized_correctly,
            running_evaluation_hash_input_is_initialized_correctly,
            running_evaluation_hash_digest_is_initialized_correctly,
            running_evaluation_sponge_absorb_is_initialized_correctly,
            running_sum_log_derivative_for_u32_table_is_initialized_correctly,
        ]
    }

    pub fn consistency_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(BaseRow(col.master_base_table_index()))
        };

        // The composition of instruction bits ib0-ib7 corresponds the current instruction ci.
        let ib_composition = base_row(IB0)
            + constant(1 << 1) * base_row(IB1)
            + constant(1 << 2) * base_row(IB2)
            + constant(1 << 3) * base_row(IB3)
            + constant(1 << 4) * base_row(IB4)
            + constant(1 << 5) * base_row(IB5)
            + constant(1 << 6) * base_row(IB6);
        let ci_corresponds_to_ib0_thru_ib7 = base_row(CI) - ib_composition;

        let ib0_is_bit = base_row(IB0) * (base_row(IB0) - constant(1));
        let ib1_is_bit = base_row(IB1) * (base_row(IB1) - constant(1));
        let ib2_is_bit = base_row(IB2) * (base_row(IB2) - constant(1));
        let ib3_is_bit = base_row(IB3) * (base_row(IB3) - constant(1));
        let ib4_is_bit = base_row(IB4) * (base_row(IB4) - constant(1));
        let ib5_is_bit = base_row(IB5) * (base_row(IB5) - constant(1));
        let ib6_is_bit = base_row(IB6) * (base_row(IB6) - constant(1));
        let is_padding_is_bit = base_row(IsPadding) * (base_row(IsPadding) - constant(1));

        // In padding rows, the clock jump difference lookup multiplicity is 0. The one row
        // exempt from this rule is the row wth CLK == 1: since the memory-like tables don't have
        // an “awareness” of padding rows, they keep looking up clock jump differences of
        // magnitude 1.
        let clock_jump_diff_lookup_multiplicity_is_0_in_padding_rows = base_row(IsPadding)
            * (base_row(CLK) - constant(1))
            * base_row(ClockJumpDifferenceLookupMultiplicity);

        vec![
            ib0_is_bit,
            ib1_is_bit,
            ib2_is_bit,
            ib3_is_bit,
            ib4_is_bit,
            ib5_is_bit,
            ib6_is_bit,
            is_padding_is_bit,
            ci_corresponds_to_ib0_thru_ib7,
            clock_jump_diff_lookup_multiplicity_is_0_in_padding_rows,
        ]
    }

    fn indicator_polynomial(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        index: usize,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let one = || circuit_builder.b_constant(1);
        let hv = |idx| Self::helper_variable(circuit_builder, idx);

        match index {
            0 => (one() - hv(3)) * (one() - hv(2)) * (one() - hv(1)) * (one() - hv(0)),
            1 => (one() - hv(3)) * (one() - hv(2)) * (one() - hv(1)) * hv(0),
            2 => (one() - hv(3)) * (one() - hv(2)) * hv(1) * (one() - hv(0)),
            3 => (one() - hv(3)) * (one() - hv(2)) * hv(1) * hv(0),
            4 => (one() - hv(3)) * hv(2) * (one() - hv(1)) * (one() - hv(0)),
            5 => (one() - hv(3)) * hv(2) * (one() - hv(1)) * hv(0),
            6 => (one() - hv(3)) * hv(2) * hv(1) * (one() - hv(0)),
            7 => (one() - hv(3)) * hv(2) * hv(1) * hv(0),
            8 => hv(3) * (one() - hv(2)) * (one() - hv(1)) * (one() - hv(0)),
            9 => hv(3) * (one() - hv(2)) * (one() - hv(1)) * hv(0),
            10 => hv(3) * (one() - hv(2)) * hv(1) * (one() - hv(0)),
            11 => hv(3) * (one() - hv(2)) * hv(1) * hv(0),
            12 => hv(3) * hv(2) * (one() - hv(1)) * (one() - hv(0)),
            13 => hv(3) * hv(2) * (one() - hv(1)) * hv(0),
            14 => hv(3) * hv(2) * hv(1) * (one() - hv(0)),
            15 => hv(3) * hv(2) * hv(1) * hv(0),
            i => panic!("indicator polynomial index {i} out of bounds"),
        }
    }

    fn helper_variable(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        index: usize,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        match index {
            0 => circuit_builder.input(CurrentBaseRow(HV0.master_base_table_index())),
            1 => circuit_builder.input(CurrentBaseRow(HV1.master_base_table_index())),
            2 => circuit_builder.input(CurrentBaseRow(HV2.master_base_table_index())),
            3 => circuit_builder.input(CurrentBaseRow(HV3.master_base_table_index())),
            4 => circuit_builder.input(CurrentBaseRow(HV4.master_base_table_index())),
            5 => circuit_builder.input(CurrentBaseRow(HV5.master_base_table_index())),
            i => unimplemented!("Helper variable index {i} out of bounds."),
        }
    }

    /// Instruction-specific transition constraints are combined with deselectors in such a way
    /// that arbitrary sets of mutually exclusive combinations are summed, i.e.,
    ///
    /// ```py
    /// [ deselector_pop * tc_pop_0 + deselector_push * tc_push_0 + ...,
    ///   deselector_pop * tc_pop_1 + deselector_push * tc_push_1 + ...,
    ///   ...,
    ///   deselector_pop * tc_pop_i + deselector_push * tc_push_i + ...,
    ///   deselector_pop * 0        + deselector_push * tc_push_{i+1} + ...,
    ///   ...,
    /// ]
    /// ```
    /// For instructions that have fewer transition constraints than the maximal number of
    /// transition constraints among all instructions, the deselector is multiplied with a zero,
    /// causing no additional terms in the final sets of combined transition constraint polynomials.
    fn combine_instruction_constraints_with_deselectors(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        instr_tc_polys_tuples: [(Instruction, Vec<ConstraintCircuitMonad<DualRowIndicator>>);
            Instruction::COUNT],
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let (all_instructions, all_tc_polys_for_all_instructions): (Vec<_>, Vec<_>) =
            instr_tc_polys_tuples.into_iter().unzip();

        let all_instruction_deselectors = all_instructions
            .into_iter()
            .map(|instr| Self::instruction_deselector_current_row(circuit_builder, instr))
            .collect_vec();

        let max_number_of_constraints = all_tc_polys_for_all_instructions
            .iter()
            .map(|tc_polys_for_instr| tc_polys_for_instr.len())
            .max()
            .unwrap();

        let zero_poly = circuit_builder.b_constant(0);
        let all_tc_polys_for_all_instructions_transposed = (0..max_number_of_constraints)
            .map(|idx| {
                all_tc_polys_for_all_instructions
                    .iter()
                    .map(|tc_polys_for_instr| tc_polys_for_instr.get(idx).unwrap_or(&zero_poly))
                    .collect_vec()
            })
            .collect_vec();

        all_tc_polys_for_all_instructions_transposed
            .into_iter()
            .map(|row| {
                all_instruction_deselectors
                    .clone()
                    .into_iter()
                    .zip(row)
                    .map(|(deselector, instruction_tc)| deselector * instruction_tc.to_owned())
                    .sum()
            })
            .collect_vec()
    }

    fn combine_transition_constraints_with_padding_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        instruction_transition_constraints: Vec<ConstraintCircuitMonad<DualRowIndicator>>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u64| circuit_builder.b_constant(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let padding_row_transition_constraints = [
            vec![
                next_base_row(IP) - curr_base_row(IP),
                next_base_row(CI) - curr_base_row(CI),
                next_base_row(NIA) - curr_base_row(NIA),
            ],
            Self::instruction_group_keep_jump_stack(circuit_builder),
            Self::instruction_group_keep_op_stack(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat();

        let padding_row_deselector = constant(1) - next_base_row(IsPadding);
        let padding_row_selector = next_base_row(IsPadding);

        let max_number_of_constraints = max(
            instruction_transition_constraints.len(),
            padding_row_transition_constraints.len(),
        );

        (0..max_number_of_constraints)
            .map(|idx| {
                let instruction_constraint = instruction_transition_constraints
                    .get(idx)
                    .unwrap_or(&constant(0))
                    .to_owned();
                let padding_constraint = padding_row_transition_constraints
                    .get(idx)
                    .unwrap_or(&constant(0))
                    .to_owned();

                instruction_constraint * padding_row_deselector.clone()
                    + padding_constraint * padding_row_selector.clone()
            })
            .collect_vec()
    }

    fn instruction_group_decompose_arg(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };

        let hv0_is_a_bit = curr_base_row(HV0) * (curr_base_row(HV0) - constant(1));
        let hv1_is_a_bit = curr_base_row(HV1) * (curr_base_row(HV1) - constant(1));
        let hv2_is_a_bit = curr_base_row(HV2) * (curr_base_row(HV2) - constant(1));
        let hv3_is_a_bit = curr_base_row(HV3) * (curr_base_row(HV3) - constant(1));

        let helper_variables_are_binary_decomposition_of_nia = curr_base_row(NIA)
            - constant(8) * curr_base_row(HV3)
            - constant(4) * curr_base_row(HV2)
            - constant(2) * curr_base_row(HV1)
            - curr_base_row(HV0);

        vec![
            hv0_is_a_bit,
            hv1_is_a_bit,
            hv2_is_a_bit,
            hv3_is_a_bit,
            helper_variables_are_binary_decomposition_of_nia,
        ]
    }

    fn instruction_group_keep_ram(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };

        vec![next_ext_row(RamTablePermArg) - curr_ext_row(RamTablePermArg)]
    }

    fn instruction_group_no_io(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        vec![
            Self::running_evaluation_for_standard_input_remains_unchanged(circuit_builder),
            Self::running_evaluation_for_standard_output_remains_unchanged(circuit_builder),
        ]
    }

    fn instruction_group_op_stack_remains_and_top_three_elements_unconstrained(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let curr_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };

        vec![
            next_base_row(ST3) - curr_base_row(ST3),
            next_base_row(ST4) - curr_base_row(ST4),
            next_base_row(ST5) - curr_base_row(ST5),
            next_base_row(ST6) - curr_base_row(ST6),
            next_base_row(ST7) - curr_base_row(ST7),
            next_base_row(ST8) - curr_base_row(ST8),
            next_base_row(ST9) - curr_base_row(ST9),
            next_base_row(ST10) - curr_base_row(ST10),
            next_base_row(ST11) - curr_base_row(ST11),
            next_base_row(ST12) - curr_base_row(ST12),
            next_base_row(ST13) - curr_base_row(ST13),
            next_base_row(ST14) - curr_base_row(ST14),
            next_base_row(ST15) - curr_base_row(ST15),
            next_base_row(OpStackPointer) - curr_base_row(OpStackPointer),
            next_ext_row(OpStackTablePermArg) - curr_ext_row(OpStackTablePermArg),
        ]
    }

    fn instruction_group_unop(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let specific_constraints = vec![
            next_base_row(ST1) - curr_base_row(ST1),
            next_base_row(ST2) - curr_base_row(ST2),
        ];
        let inherited_constraints =
            Self::instruction_group_op_stack_remains_and_top_three_elements_unconstrained(
                circuit_builder,
            );

        [specific_constraints, inherited_constraints].concat()
    }

    fn instruction_group_keep_op_stack(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let specific_constraints = vec![next_base_row(ST0) - curr_base_row(ST0)];
        let inherited_constraints = Self::instruction_group_unop(circuit_builder);

        [specific_constraints, inherited_constraints].concat()
    }

    fn instruction_group_grow_op_stack_and_top_two_elements_unconstrained(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        vec![
            next_base_row(ST2) - curr_base_row(ST1),
            next_base_row(ST3) - curr_base_row(ST2),
            next_base_row(ST4) - curr_base_row(ST3),
            next_base_row(ST5) - curr_base_row(ST4),
            next_base_row(ST6) - curr_base_row(ST5),
            next_base_row(ST7) - curr_base_row(ST6),
            next_base_row(ST8) - curr_base_row(ST7),
            next_base_row(ST9) - curr_base_row(ST8),
            next_base_row(ST10) - curr_base_row(ST9),
            next_base_row(ST11) - curr_base_row(ST10),
            next_base_row(ST12) - curr_base_row(ST11),
            next_base_row(ST13) - curr_base_row(ST12),
            next_base_row(ST14) - curr_base_row(ST13),
            next_base_row(ST15) - curr_base_row(ST14),
            next_base_row(OpStackPointer) - curr_base_row(OpStackPointer) - constant(1),
            Self::running_product_op_stack_accounts_for_growing_stack_by(circuit_builder, 1),
        ]
    }

    fn instruction_group_grow_op_stack(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let specific_constraints = vec![next_base_row(ST1) - curr_base_row(ST0)];
        let inherited_constraints =
            Self::instruction_group_grow_op_stack_and_top_two_elements_unconstrained(
                circuit_builder,
            );

        [specific_constraints, inherited_constraints].concat()
    }

    fn instruction_group_op_stack_shrinks_and_top_three_elements_unconstrained(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        vec![
            next_base_row(ST3) - curr_base_row(ST4),
            next_base_row(ST4) - curr_base_row(ST5),
            next_base_row(ST5) - curr_base_row(ST6),
            next_base_row(ST6) - curr_base_row(ST7),
            next_base_row(ST7) - curr_base_row(ST8),
            next_base_row(ST8) - curr_base_row(ST9),
            next_base_row(ST9) - curr_base_row(ST10),
            next_base_row(ST10) - curr_base_row(ST11),
            next_base_row(ST11) - curr_base_row(ST12),
            next_base_row(ST12) - curr_base_row(ST13),
            next_base_row(ST13) - curr_base_row(ST14),
            next_base_row(ST14) - curr_base_row(ST15),
            next_base_row(OpStackPointer) - curr_base_row(OpStackPointer) + constant(1),
            Self::running_product_op_stack_accounts_for_shrinking_stack_by(circuit_builder, 1),
        ]
    }

    fn instruction_group_binop(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let specific_constraints = vec![
            next_base_row(ST1) - curr_base_row(ST2),
            next_base_row(ST2) - curr_base_row(ST3),
        ];
        let inherited_constraints =
            Self::instruction_group_op_stack_shrinks_and_top_three_elements_unconstrained(
                circuit_builder,
            );

        [specific_constraints, inherited_constraints].concat()
    }

    fn instruction_group_shrink_op_stack(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let specific_constraints = vec![next_base_row(ST0) - curr_base_row(ST1)];
        let inherited_constraints = Self::instruction_group_binop(circuit_builder);

        [specific_constraints, inherited_constraints].concat()
    }

    fn instruction_group_keep_jump_stack(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let jsp_does_not_change = next_base_row(JSP) - curr_base_row(JSP);
        let jso_does_not_change = next_base_row(JSO) - curr_base_row(JSO);
        let jsd_does_not_change = next_base_row(JSD) - curr_base_row(JSD);

        vec![
            jsp_does_not_change,
            jso_does_not_change,
            jsd_does_not_change,
        ]
    }

    fn instruction_group_step_1(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let instruction_pointer_increases_by_one =
            next_base_row(IP) - curr_base_row(IP) - constant(1);
        [
            Self::instruction_group_keep_jump_stack(circuit_builder),
            vec![instruction_pointer_increases_by_one],
        ]
        .concat()
    }

    fn instruction_group_step_2(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let instruction_pointer_increases_by_two =
            next_base_row(IP) - curr_base_row(IP) - constant(2);
        [
            Self::instruction_group_keep_jump_stack(circuit_builder),
            vec![instruction_pointer_increases_by_two],
        ]
        .concat()
    }

    /// Internal helper function to de-duplicate functionality common between the similar (but
    /// different on a type level) functions for construction deselectors.
    fn instruction_deselector_common_functionality<II: InputIndicator>(
        circuit_builder: &ConstraintCircuitBuilder<II>,
        instruction: Instruction,
        instruction_bit_polynomials: [ConstraintCircuitMonad<II>; InstructionBit::COUNT],
    ) -> ConstraintCircuitMonad<II> {
        let one = || circuit_builder.b_constant(1);

        let selector_bits: [_; InstructionBit::COUNT] = [
            instruction.ib(InstructionBit::IB0),
            instruction.ib(InstructionBit::IB1),
            instruction.ib(InstructionBit::IB2),
            instruction.ib(InstructionBit::IB3),
            instruction.ib(InstructionBit::IB4),
            instruction.ib(InstructionBit::IB5),
            instruction.ib(InstructionBit::IB6),
        ];
        let deselector_polynomials = selector_bits.map(|b| one() - circuit_builder.b_constant(b));

        instruction_bit_polynomials
            .into_iter()
            .zip_eq(deselector_polynomials)
            .map(|(instruction_bit_poly, deselector_poly)| instruction_bit_poly - deselector_poly)
            .fold(one(), ConstraintCircuitMonad::mul)
    }

    /// A polynomial that has no solutions when `ci` is `instruction`.
    /// The number of variables in the polynomial corresponds to two rows.
    fn instruction_deselector_current_row(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        instruction: Instruction,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };

        let instruction_bit_polynomials = [
            curr_base_row(IB0),
            curr_base_row(IB1),
            curr_base_row(IB2),
            curr_base_row(IB3),
            curr_base_row(IB4),
            curr_base_row(IB5),
            curr_base_row(IB6),
        ];

        Self::instruction_deselector_common_functionality(
            circuit_builder,
            instruction,
            instruction_bit_polynomials,
        )
    }

    /// A polynomial that has no solutions when `ci_next` is `instruction`.
    /// The number of variables in the polynomial corresponds to two rows.
    fn instruction_deselector_next_row(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        instruction: Instruction,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let instruction_bit_polynomials = [
            next_base_row(IB0),
            next_base_row(IB1),
            next_base_row(IB2),
            next_base_row(IB3),
            next_base_row(IB4),
            next_base_row(IB5),
            next_base_row(IB6),
        ];

        Self::instruction_deselector_common_functionality(
            circuit_builder,
            instruction,
            instruction_bit_polynomials,
        )
    }

    /// A polynomial that has no solutions when `ci` is `instruction`.
    /// The number of variables in the polynomial corresponds to a single row.
    fn instruction_deselector_single_row(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
        instruction: Instruction,
    ) -> ConstraintCircuitMonad<SingleRowIndicator> {
        let base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(BaseRow(col.master_base_table_index()))
        };

        let instruction_bit_polynomials = [
            base_row(IB0),
            base_row(IB1),
            base_row(IB2),
            base_row(IB3),
            base_row(IB4),
            base_row(IB5),
            base_row(IB6),
        ];

        Self::instruction_deselector_common_functionality(
            circuit_builder,
            instruction,
            instruction_bit_polynomials,
        )
    }

    fn instruction_pop(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_2(circuit_builder),
            Self::instruction_group_decompose_arg(circuit_builder),
            Self::stack_shrinks_by_any_of(circuit_builder, &NumberOfWords::legal_values()),
            Self::prohibit_any_illegal_number_of_words(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_push(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let specific_constraints = vec![next_base_row(ST0) - curr_base_row(NIA)];
        [
            specific_constraints,
            Self::instruction_group_grow_op_stack(circuit_builder),
            Self::instruction_group_step_2(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_divine(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_2(circuit_builder),
            Self::instruction_group_decompose_arg(circuit_builder),
            Self::stack_grows_by_any_of(circuit_builder, &NumberOfWords::legal_values()),
            Self::prohibit_any_illegal_number_of_words(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_dup(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let indicator_poly = |idx| Self::indicator_polynomial(circuit_builder, idx);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let specific_constraints = vec![
            indicator_poly(0) * (next_base_row(ST0) - curr_base_row(ST0)),
            indicator_poly(1) * (next_base_row(ST0) - curr_base_row(ST1)),
            indicator_poly(2) * (next_base_row(ST0) - curr_base_row(ST2)),
            indicator_poly(3) * (next_base_row(ST0) - curr_base_row(ST3)),
            indicator_poly(4) * (next_base_row(ST0) - curr_base_row(ST4)),
            indicator_poly(5) * (next_base_row(ST0) - curr_base_row(ST5)),
            indicator_poly(6) * (next_base_row(ST0) - curr_base_row(ST6)),
            indicator_poly(7) * (next_base_row(ST0) - curr_base_row(ST7)),
            indicator_poly(8) * (next_base_row(ST0) - curr_base_row(ST8)),
            indicator_poly(9) * (next_base_row(ST0) - curr_base_row(ST9)),
            indicator_poly(10) * (next_base_row(ST0) - curr_base_row(ST10)),
            indicator_poly(11) * (next_base_row(ST0) - curr_base_row(ST11)),
            indicator_poly(12) * (next_base_row(ST0) - curr_base_row(ST12)),
            indicator_poly(13) * (next_base_row(ST0) - curr_base_row(ST13)),
            indicator_poly(14) * (next_base_row(ST0) - curr_base_row(ST14)),
            indicator_poly(15) * (next_base_row(ST0) - curr_base_row(ST15)),
        ];
        [
            specific_constraints,
            Self::instruction_group_decompose_arg(circuit_builder),
            Self::instruction_group_step_2(circuit_builder),
            Self::instruction_group_grow_op_stack(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_swap(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let one = || circuit_builder.b_constant(1);
        let indicator_poly = |idx| Self::indicator_polynomial(circuit_builder, idx);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let curr_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };
        let next_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };

        let specific_constraints = vec![
            indicator_poly(0),
            indicator_poly(1) * (next_base_row(ST1) - curr_base_row(ST0)),
            indicator_poly(2) * (next_base_row(ST2) - curr_base_row(ST0)),
            indicator_poly(3) * (next_base_row(ST3) - curr_base_row(ST0)),
            indicator_poly(4) * (next_base_row(ST4) - curr_base_row(ST0)),
            indicator_poly(5) * (next_base_row(ST5) - curr_base_row(ST0)),
            indicator_poly(6) * (next_base_row(ST6) - curr_base_row(ST0)),
            indicator_poly(7) * (next_base_row(ST7) - curr_base_row(ST0)),
            indicator_poly(8) * (next_base_row(ST8) - curr_base_row(ST0)),
            indicator_poly(9) * (next_base_row(ST9) - curr_base_row(ST0)),
            indicator_poly(10) * (next_base_row(ST10) - curr_base_row(ST0)),
            indicator_poly(11) * (next_base_row(ST11) - curr_base_row(ST0)),
            indicator_poly(12) * (next_base_row(ST12) - curr_base_row(ST0)),
            indicator_poly(13) * (next_base_row(ST13) - curr_base_row(ST0)),
            indicator_poly(14) * (next_base_row(ST14) - curr_base_row(ST0)),
            indicator_poly(15) * (next_base_row(ST15) - curr_base_row(ST0)),
            indicator_poly(1) * (next_base_row(ST0) - curr_base_row(ST1)),
            indicator_poly(2) * (next_base_row(ST0) - curr_base_row(ST2)),
            indicator_poly(3) * (next_base_row(ST0) - curr_base_row(ST3)),
            indicator_poly(4) * (next_base_row(ST0) - curr_base_row(ST4)),
            indicator_poly(5) * (next_base_row(ST0) - curr_base_row(ST5)),
            indicator_poly(6) * (next_base_row(ST0) - curr_base_row(ST6)),
            indicator_poly(7) * (next_base_row(ST0) - curr_base_row(ST7)),
            indicator_poly(8) * (next_base_row(ST0) - curr_base_row(ST8)),
            indicator_poly(9) * (next_base_row(ST0) - curr_base_row(ST9)),
            indicator_poly(10) * (next_base_row(ST0) - curr_base_row(ST10)),
            indicator_poly(11) * (next_base_row(ST0) - curr_base_row(ST11)),
            indicator_poly(12) * (next_base_row(ST0) - curr_base_row(ST12)),
            indicator_poly(13) * (next_base_row(ST0) - curr_base_row(ST13)),
            indicator_poly(14) * (next_base_row(ST0) - curr_base_row(ST14)),
            indicator_poly(15) * (next_base_row(ST0) - curr_base_row(ST15)),
            (one() - indicator_poly(1)) * (next_base_row(ST1) - curr_base_row(ST1)),
            (one() - indicator_poly(2)) * (next_base_row(ST2) - curr_base_row(ST2)),
            (one() - indicator_poly(3)) * (next_base_row(ST3) - curr_base_row(ST3)),
            (one() - indicator_poly(4)) * (next_base_row(ST4) - curr_base_row(ST4)),
            (one() - indicator_poly(5)) * (next_base_row(ST5) - curr_base_row(ST5)),
            (one() - indicator_poly(6)) * (next_base_row(ST6) - curr_base_row(ST6)),
            (one() - indicator_poly(7)) * (next_base_row(ST7) - curr_base_row(ST7)),
            (one() - indicator_poly(8)) * (next_base_row(ST8) - curr_base_row(ST8)),
            (one() - indicator_poly(9)) * (next_base_row(ST9) - curr_base_row(ST9)),
            (one() - indicator_poly(10)) * (next_base_row(ST10) - curr_base_row(ST10)),
            (one() - indicator_poly(11)) * (next_base_row(ST11) - curr_base_row(ST11)),
            (one() - indicator_poly(12)) * (next_base_row(ST12) - curr_base_row(ST12)),
            (one() - indicator_poly(13)) * (next_base_row(ST13) - curr_base_row(ST13)),
            (one() - indicator_poly(14)) * (next_base_row(ST14) - curr_base_row(ST14)),
            (one() - indicator_poly(15)) * (next_base_row(ST15) - curr_base_row(ST15)),
            next_base_row(OpStackPointer) - curr_base_row(OpStackPointer),
            next_ext_row(OpStackTablePermArg) - curr_ext_row(OpStackTablePermArg),
        ];
        [
            specific_constraints,
            Self::instruction_group_decompose_arg(circuit_builder),
            Self::instruction_group_step_2(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_nop(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_keep_op_stack(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_skiz(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let one = || constant(1);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let hv0_is_inverse_of_st0 = curr_base_row(HV0) * curr_base_row(ST0) - one();
        let hv0_is_inverse_of_st0_or_hv0_is_0 = hv0_is_inverse_of_st0.clone() * curr_base_row(HV0);
        let hv0_is_inverse_of_st0_or_st0_is_0 = hv0_is_inverse_of_st0 * curr_base_row(ST0);

        // The next instruction nia is decomposed into helper variables hv.
        let nia_decomposes_to_hvs = curr_base_row(NIA)
            - curr_base_row(HV1)
            - constant(1 << 1) * curr_base_row(HV2)
            - constant(1 << 3) * curr_base_row(HV3)
            - constant(1 << 5) * curr_base_row(HV4)
            - constant(1 << 7) * curr_base_row(HV5);

        // If `st0` is non-zero, register `ip` is incremented by 1.
        // If `st0` is 0 and `nia` takes no argument, register `ip` is incremented by 2.
        // If `st0` is 0 and `nia` takes an argument, register `ip` is incremented by 3.
        //
        // The opcodes are constructed such that hv1 == 1 means that nia takes an argument.
        //
        // Written as Disjunctive Normal Form, the constraint can be expressed as:
        // (Register `st0` is 0 or `ip` is incremented by 1), and
        // (`st0` has a multiplicative inverse or `hv1` is 1 or `ip` is incremented by 2), and
        // (`st0` has a multiplicative inverse or `hv1` is 0 or `ip` is incremented by 3).
        let ip_case_1 = (next_base_row(IP) - curr_base_row(IP) - constant(1)) * curr_base_row(ST0);
        let ip_case_2 = (next_base_row(IP) - curr_base_row(IP) - constant(2))
            * (curr_base_row(ST0) * curr_base_row(HV0) - one())
            * (curr_base_row(HV1) - one());
        let ip_case_3 = (next_base_row(IP) - curr_base_row(IP) - constant(3))
            * (curr_base_row(ST0) * curr_base_row(HV0) - one())
            * curr_base_row(HV1);
        let ip_incr_by_1_or_2_or_3 = ip_case_1 + ip_case_2 + ip_case_3;

        let specific_constraints = vec![
            hv0_is_inverse_of_st0_or_hv0_is_0,
            hv0_is_inverse_of_st0_or_st0_is_0,
            nia_decomposes_to_hvs,
            ip_incr_by_1_or_2_or_3,
        ];
        [
            specific_constraints,
            Self::next_instruction_range_check_constraints_for_instruction_skiz(circuit_builder),
            Self::instruction_group_keep_jump_stack(circuit_builder),
            Self::instruction_group_shrink_op_stack(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn next_instruction_range_check_constraints_for_instruction_skiz(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };

        let is_0_or_1 =
            |var: ProcessorBaseTableColumn| curr_base_row(var) * (curr_base_row(var) - constant(1));
        let is_0_or_1_or_2_or_3 = |var: ProcessorBaseTableColumn| {
            curr_base_row(var)
                * (curr_base_row(var) - constant(1))
                * (curr_base_row(var) - constant(2))
                * (curr_base_row(var) - constant(3))
        };

        vec![
            is_0_or_1(HV1),
            is_0_or_1_or_2_or_3(HV2),
            is_0_or_1_or_2_or_3(HV3),
            is_0_or_1_or_2_or_3(HV4),
            is_0_or_1_or_2_or_3(HV5),
        ]
    }

    fn instruction_call(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        // The jump stack pointer jsp is incremented by 1.
        let jsp_incr_1 = next_base_row(JSP) - curr_base_row(JSP) - constant(1);

        // The jump's origin jso is set to the current instruction pointer ip plus 2.
        let jso_becomes_ip_plus_2 = next_base_row(JSO) - curr_base_row(IP) - constant(2);

        // The jump's destination jsd is set to the instruction's argument.
        let jsd_becomes_nia = next_base_row(JSD) - curr_base_row(NIA);

        // The instruction pointer ip is set to the instruction's argument.
        let ip_becomes_nia = next_base_row(IP) - curr_base_row(NIA);

        let specific_constraints = vec![
            jsp_incr_1,
            jso_becomes_ip_plus_2,
            jsd_becomes_nia,
            ip_becomes_nia,
        ];
        [
            specific_constraints,
            Self::instruction_group_keep_op_stack(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_return(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        // The jump stack pointer jsp is decremented by 1.
        let jsp_incr_1 = next_base_row(JSP) - (curr_base_row(JSP) - constant(1));

        // The instruction pointer ip is set to the last call's origin jso.
        let ip_becomes_jso = next_base_row(IP) - curr_base_row(JSO);

        let specific_constraints = vec![jsp_incr_1, ip_becomes_jso];
        [
            specific_constraints,
            Self::instruction_group_keep_op_stack(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_recurse(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        // The instruction pointer ip is set to the last jump's destination jsd.
        let ip_becomes_jsd = next_base_row(IP) - curr_base_row(JSD);
        let specific_constraints = vec![ip_becomes_jsd];
        [
            specific_constraints,
            Self::instruction_group_keep_jump_stack(circuit_builder),
            Self::instruction_group_keep_op_stack(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_assert(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };

        // The current top of the stack st0 is 1.
        let st_0_is_1 = curr_base_row(ST0) - constant(1);

        let specific_constraints = vec![st_0_is_1];
        [
            specific_constraints,
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_shrink_op_stack(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_halt(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        // The instruction executed in the following step is instruction halt.
        let halt_is_followed_by_halt = next_base_row(CI) - curr_base_row(CI);

        let specific_constraints = vec![halt_is_followed_by_halt];
        [
            specific_constraints,
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_keep_op_stack(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_read_mem(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_2(circuit_builder),
            Self::instruction_group_decompose_arg(circuit_builder),
            Self::read_from_ram_any_of(circuit_builder, &NumberOfWords::legal_values()),
            Self::prohibit_any_illegal_number_of_words(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_write_mem(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_2(circuit_builder),
            Self::instruction_group_decompose_arg(circuit_builder),
            Self::write_to_ram_any_of(circuit_builder, &NumberOfWords::legal_values()),
            Self::prohibit_any_illegal_number_of_words(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    /// Two Evaluation Arguments with the Hash Table guarantee correct transition.
    fn instruction_hash(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let op_stack_shrinks_by_5_and_top_5_unconstrained = vec![
            next_base_row(ST5) - curr_base_row(ST10),
            next_base_row(ST6) - curr_base_row(ST11),
            next_base_row(ST7) - curr_base_row(ST12),
            next_base_row(ST8) - curr_base_row(ST13),
            next_base_row(ST9) - curr_base_row(ST14),
            next_base_row(ST10) - curr_base_row(ST15),
            next_base_row(OpStackPointer) - curr_base_row(OpStackPointer) + constant(5),
            Self::running_product_op_stack_accounts_for_shrinking_stack_by(circuit_builder, 5),
        ];

        [
            Self::instruction_group_step_1(circuit_builder),
            op_stack_shrinks_by_5_and_top_5_unconstrained,
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    /// Recall that in a Merkle tree, the indices of left (respectively right) leafs have 0
    /// (respectively 1) as their least significant bit. The first two polynomials achieve that
    /// helper variable hv0 holds the result of st5 mod 2. The second polynomial sets the new value
    /// of st10 to st5 div 2.
    fn instruction_divine_sibling(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let one = || constant(1);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let hv0_is_0_or_1 = curr_base_row(HV0) * (curr_base_row(HV0) - one());

        let new_st10_is_previous_st5_div_2 =
            next_base_row(ST10) * constant(2) + curr_base_row(HV0) - curr_base_row(ST5);

        let maybe_move_st0 = (one() - curr_base_row(HV0))
            * (next_base_row(ST0) - curr_base_row(ST0))
            + curr_base_row(HV0) * (next_base_row(ST5) - curr_base_row(ST0));
        let maybe_move_st1 = (one() - curr_base_row(HV0))
            * (next_base_row(ST1) - curr_base_row(ST1))
            + curr_base_row(HV0) * (next_base_row(ST6) - curr_base_row(ST1));
        let maybe_move_st2 = (one() - curr_base_row(HV0))
            * (next_base_row(ST2) - curr_base_row(ST2))
            + curr_base_row(HV0) * (next_base_row(ST7) - curr_base_row(ST2));
        let maybe_move_st3 = (one() - curr_base_row(HV0))
            * (next_base_row(ST3) - curr_base_row(ST3))
            + curr_base_row(HV0) * (next_base_row(ST8) - curr_base_row(ST3));
        let maybe_move_st4 = (one() - curr_base_row(HV0))
            * (next_base_row(ST4) - curr_base_row(ST4))
            + curr_base_row(HV0) * (next_base_row(ST9) - curr_base_row(ST4));

        let maybe_shift_known_digest_down_the_stack = vec![
            hv0_is_0_or_1,
            new_st10_is_previous_st5_div_2,
            maybe_move_st0,
            maybe_move_st1,
            maybe_move_st2,
            maybe_move_st3,
            maybe_move_st4,
        ];

        let op_stack_grows_by_5_and_top_11_elements_unconstrained = vec![
            next_base_row(ST11) - curr_base_row(ST6),
            next_base_row(ST12) - curr_base_row(ST7),
            next_base_row(ST13) - curr_base_row(ST8),
            next_base_row(ST14) - curr_base_row(ST9),
            next_base_row(ST15) - curr_base_row(ST10),
            next_base_row(OpStackPointer) - curr_base_row(OpStackPointer) - constant(5),
            Self::running_product_op_stack_accounts_for_growing_stack_by(circuit_builder, 5),
        ];
        [
            maybe_shift_known_digest_down_the_stack,
            op_stack_grows_by_5_and_top_11_elements_unconstrained,
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_assert_vector(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };

        let specific_constraints = vec![
            curr_base_row(ST5) - curr_base_row(ST0),
            curr_base_row(ST6) - curr_base_row(ST1),
            curr_base_row(ST7) - curr_base_row(ST2),
            curr_base_row(ST8) - curr_base_row(ST3),
            curr_base_row(ST9) - curr_base_row(ST4),
        ];
        [
            specific_constraints,
            Self::instruction_group_step_1(circuit_builder),
            Self::constraints_for_shrinking_stack_by(circuit_builder, 5),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_sponge_init(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_keep_op_stack(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_sponge_absorb(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_1(circuit_builder),
            Self::constraints_for_shrinking_stack_by(circuit_builder, 10),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_sponge_squeeze(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_1(circuit_builder),
            Self::constraints_for_growing_stack_by(circuit_builder, 10),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_add(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let specific_constraints =
            vec![next_base_row(ST0) - curr_base_row(ST0) - curr_base_row(ST1)];
        [
            specific_constraints,
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_binop(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_mul(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let specific_constraints =
            vec![next_base_row(ST0) - curr_base_row(ST0) * curr_base_row(ST1)];
        [
            specific_constraints,
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_binop(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_invert(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let one = || constant(1);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let specific_constraints = vec![next_base_row(ST0) * curr_base_row(ST0) - one()];
        [
            specific_constraints,
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_unop(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_eq(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let one = || constant(1);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let st0_eq_st1 = || one() - curr_base_row(HV0) * (curr_base_row(ST1) - curr_base_row(ST0));

        // Helper variable hv0 is the inverse-or-zero of the difference of the stack's two top-most
        // elements: `hv0·(1 - hv0·(st1 - st0))`
        let hv0_is_inverse_of_diff_or_hv0_is_0 = curr_base_row(HV0) * st0_eq_st1();

        // Helper variable hv0 is the inverse-or-zero of the difference of the stack's two
        // top-most elements: `(st1 - st0)·(1 - hv0·(st1 - st0))`
        let hv0_is_inverse_of_diff_or_diff_is_0 =
            (curr_base_row(ST1) - curr_base_row(ST0)) * st0_eq_st1();

        // The new top of the stack is 1 if the difference between the stack's two top-most
        // elements is not invertible, 0 otherwise: `st0' - (1 - hv0·(st1 - st0))`
        let st0_becomes_1_if_diff_is_not_invertible = next_base_row(ST0) - st0_eq_st1();

        let specific_constraints = vec![
            hv0_is_inverse_of_diff_or_hv0_is_0,
            hv0_is_inverse_of_diff_or_diff_is_0,
            st0_becomes_1_if_diff_is_not_invertible,
        ];
        [
            specific_constraints,
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_binop(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_split(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u64| circuit_builder.b_constant(c);
        let one = || constant(1);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        // The top of the stack is decomposed as 32-bit chunks into the stack's top-most elements:
        // st0 - (2^32·st0' + st1') = 0$
        let st0_decomposes_to_two_32_bit_chunks =
            curr_base_row(ST0) - (constant(1 << 32) * next_base_row(ST1) + next_base_row(ST0));

        // Helper variable `hv0` = 0 if either
        // 1. `hv0` is the difference between (2^32 - 1) and the high 32 bits (`st0'`), or
        // 1. the low 32 bits (`st1'`) are 0.
        //
        // st1'·(hv0·(st0' - (2^32 - 1)) - 1)
        //   lo·(hv0·(hi - 0xffff_ffff)) - 1)
        let hv0_holds_inverse_of_chunk_difference_or_low_bits_are_0 = {
            let hv0 = curr_base_row(HV0);
            let hi = next_base_row(ST1);
            let lo = next_base_row(ST0);
            let ffff_ffff = constant(0xffff_ffff);

            lo * (hv0 * (hi - ffff_ffff) - one())
        };

        let specific_constraints = vec![
            st0_decomposes_to_two_32_bit_chunks,
            hv0_holds_inverse_of_chunk_difference_or_low_bits_are_0,
        ];
        [
            specific_constraints,
            Self::instruction_group_grow_op_stack_and_top_two_elements_unconstrained(
                circuit_builder,
            ),
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_lt(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_binop(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_and(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_binop(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_xor(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_binop(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_log_2_floor(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_unop(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_pow(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_binop(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_div_mod(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        // `n == d·q + r` means `st0 - st1·st1' - st0'`
        let numerator_is_quotient_times_denominator_plus_remainder =
            curr_base_row(ST0) - curr_base_row(ST1) * next_base_row(ST1) - next_base_row(ST0);

        let st2_does_not_change = next_base_row(ST2) - curr_base_row(ST2);

        let specific_constraints = vec![
            numerator_is_quotient_times_denominator_plus_remainder,
            st2_does_not_change,
        ];
        [
            specific_constraints,
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_op_stack_remains_and_top_three_elements_unconstrained(
                circuit_builder,
            ),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_pop_count(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_unop(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_xxadd(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let st0_becomes_st0_plus_st3 = next_base_row(ST0) - curr_base_row(ST0) - curr_base_row(ST3);
        let st1_becomes_st1_plus_st4 = next_base_row(ST1) - curr_base_row(ST1) - curr_base_row(ST4);
        let st2_becomes_st2_plus_st5 = next_base_row(ST2) - curr_base_row(ST2) - curr_base_row(ST5);
        let specific_constraints = vec![
            st0_becomes_st0_plus_st3,
            st1_becomes_st1_plus_st4,
            st2_becomes_st2_plus_st5,
        ];

        [
            specific_constraints,
            Self::constraints_for_shrinking_stack_by_3_and_top_3_unconstrained(circuit_builder),
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_xxmul(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let st0_becomes_coefficient_0 = next_base_row(ST0)
            - (curr_base_row(ST0) * curr_base_row(ST3)
                - curr_base_row(ST2) * curr_base_row(ST4)
                - curr_base_row(ST1) * curr_base_row(ST5));
        let st1_becomes_coefficient_1 = next_base_row(ST1)
            - (curr_base_row(ST1) * curr_base_row(ST3) + curr_base_row(ST0) * curr_base_row(ST4)
                - curr_base_row(ST2) * curr_base_row(ST5)
                + curr_base_row(ST2) * curr_base_row(ST4)
                + curr_base_row(ST1) * curr_base_row(ST5));
        let st2_becomes_coefficient_2 = next_base_row(ST2)
            - (curr_base_row(ST2) * curr_base_row(ST3)
                + curr_base_row(ST1) * curr_base_row(ST4)
                + curr_base_row(ST0) * curr_base_row(ST5)
                + curr_base_row(ST2) * curr_base_row(ST5));

        let specific_constraints = vec![
            st0_becomes_coefficient_0,
            st1_becomes_coefficient_1,
            st2_becomes_coefficient_2,
        ];
        [
            specific_constraints,
            Self::constraints_for_shrinking_stack_by_3_and_top_3_unconstrained(circuit_builder),
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_xinv(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u64| circuit_builder.b_constant(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let first_coefficient_of_product_of_element_and_inverse_is_1 = curr_base_row(ST0)
            * next_base_row(ST0)
            - curr_base_row(ST2) * next_base_row(ST1)
            - curr_base_row(ST1) * next_base_row(ST2)
            - constant(1);

        let second_coefficient_of_product_of_element_and_inverse_is_0 =
            curr_base_row(ST1) * next_base_row(ST0) + curr_base_row(ST0) * next_base_row(ST1)
                - curr_base_row(ST2) * next_base_row(ST2)
                + curr_base_row(ST2) * next_base_row(ST1)
                + curr_base_row(ST1) * next_base_row(ST2);

        let third_coefficient_of_product_of_element_and_inverse_is_0 = curr_base_row(ST2)
            * next_base_row(ST0)
            + curr_base_row(ST1) * next_base_row(ST1)
            + curr_base_row(ST0) * next_base_row(ST2)
            + curr_base_row(ST2) * next_base_row(ST2);

        let specific_constraints = vec![
            first_coefficient_of_product_of_element_and_inverse_is_1,
            second_coefficient_of_product_of_element_and_inverse_is_0,
            third_coefficient_of_product_of_element_and_inverse_is_0,
        ];
        [
            specific_constraints,
            Self::instruction_group_op_stack_remains_and_top_three_elements_unconstrained(
                circuit_builder,
            ),
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_xbmul(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let first_coeff_scalar_multiplication =
            next_base_row(ST0) - curr_base_row(ST0) * curr_base_row(ST1);
        let secnd_coeff_scalar_multiplication =
            next_base_row(ST1) - curr_base_row(ST0) * curr_base_row(ST2);
        let third_coeff_scalar_multiplication =
            next_base_row(ST2) - curr_base_row(ST0) * curr_base_row(ST3);

        let specific_constraints = vec![
            first_coeff_scalar_multiplication,
            secnd_coeff_scalar_multiplication,
            third_coeff_scalar_multiplication,
        ];
        [
            specific_constraints,
            Self::instruction_group_op_stack_shrinks_and_top_three_elements_unconstrained(
                circuit_builder,
            ),
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            Self::instruction_group_no_io(circuit_builder),
        ]
        .concat()
    }

    fn instruction_read_io(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constraint_groups_for_legal_arguments = NumberOfWords::legal_values()
            .map(|n| Self::grow_stack_by_n_and_read_n_symbols_from_input(circuit_builder, n))
            .to_vec();
        let read_any_legal_number_of_words = Self::combine_mutually_exclusive_constraint_groups(
            circuit_builder,
            constraint_groups_for_legal_arguments,
        );

        [
            Self::instruction_group_step_2(circuit_builder),
            Self::instruction_group_decompose_arg(circuit_builder),
            read_any_legal_number_of_words,
            Self::prohibit_any_illegal_number_of_words(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            vec![Self::running_evaluation_for_standard_output_remains_unchanged(circuit_builder)],
        ]
        .concat()
    }

    fn instruction_write_io(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constraint_groups_for_legal_arguments = NumberOfWords::legal_values()
            .map(|n| Self::shrink_stack_by_n_and_write_n_symbols_to_output(circuit_builder, n))
            .to_vec();
        let write_any_of_1_through_5_elements = Self::combine_mutually_exclusive_constraint_groups(
            circuit_builder,
            constraint_groups_for_legal_arguments,
        );

        [
            Self::instruction_group_step_2(circuit_builder),
            Self::instruction_group_decompose_arg(circuit_builder),
            write_any_of_1_through_5_elements,
            Self::prohibit_any_illegal_number_of_words(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
            vec![Self::running_evaluation_for_standard_input_remains_unchanged(circuit_builder)],
        ]
        .concat()
    }

    fn instruction_xxdotstep(
        _circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        todo!("todo: define constraints for xxdotstep")
    }

    fn instruction_xbdotstep(
        _circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        todo!("todo: define constraints for xbdotstep")
    }

    fn get_transition_constraints_for_instruction(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        instruction: Instruction,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        match instruction {
            Pop(_) => ExtProcessorTable::instruction_pop(circuit_builder),
            Push(_) => ExtProcessorTable::instruction_push(circuit_builder),
            Divine(_) => ExtProcessorTable::instruction_divine(circuit_builder),
            Dup(_) => ExtProcessorTable::instruction_dup(circuit_builder),
            Swap(_) => ExtProcessorTable::instruction_swap(circuit_builder),
            Halt => ExtProcessorTable::instruction_halt(circuit_builder),
            Nop => ExtProcessorTable::instruction_nop(circuit_builder),
            Skiz => ExtProcessorTable::instruction_skiz(circuit_builder),
            Call(_) => ExtProcessorTable::instruction_call(circuit_builder),
            Return => ExtProcessorTable::instruction_return(circuit_builder),
            Recurse => ExtProcessorTable::instruction_recurse(circuit_builder),
            Assert => ExtProcessorTable::instruction_assert(circuit_builder),
            ReadMem(_) => ExtProcessorTable::instruction_read_mem(circuit_builder),
            WriteMem(_) => ExtProcessorTable::instruction_write_mem(circuit_builder),
            Hash => ExtProcessorTable::instruction_hash(circuit_builder),
            DivineSibling => ExtProcessorTable::instruction_divine_sibling(circuit_builder),
            AssertVector => ExtProcessorTable::instruction_assert_vector(circuit_builder),
            SpongeInit => ExtProcessorTable::instruction_sponge_init(circuit_builder),
            SpongeAbsorb => ExtProcessorTable::instruction_sponge_absorb(circuit_builder),
            SpongeSqueeze => ExtProcessorTable::instruction_sponge_squeeze(circuit_builder),
            Add => ExtProcessorTable::instruction_add(circuit_builder),
            Mul => ExtProcessorTable::instruction_mul(circuit_builder),
            Invert => ExtProcessorTable::instruction_invert(circuit_builder),
            Eq => ExtProcessorTable::instruction_eq(circuit_builder),
            Split => ExtProcessorTable::instruction_split(circuit_builder),
            Lt => ExtProcessorTable::instruction_lt(circuit_builder),
            And => ExtProcessorTable::instruction_and(circuit_builder),
            Xor => ExtProcessorTable::instruction_xor(circuit_builder),
            Log2Floor => ExtProcessorTable::instruction_log_2_floor(circuit_builder),
            Pow => ExtProcessorTable::instruction_pow(circuit_builder),
            DivMod => ExtProcessorTable::instruction_div_mod(circuit_builder),
            PopCount => ExtProcessorTable::instruction_pop_count(circuit_builder),
            XxAdd => ExtProcessorTable::instruction_xxadd(circuit_builder),
            XxMul => ExtProcessorTable::instruction_xxmul(circuit_builder),
            XInvert => ExtProcessorTable::instruction_xinv(circuit_builder),
            XbMul => ExtProcessorTable::instruction_xbmul(circuit_builder),
            ReadIo(_) => ExtProcessorTable::instruction_read_io(circuit_builder),
            WriteIo(_) => ExtProcessorTable::instruction_write_io(circuit_builder),
            XxDotStep => ExtProcessorTable::instruction_xxdotstep(circuit_builder),
            XbDotStep => ExtProcessorTable::instruction_xbdotstep(circuit_builder),
        }
    }

    fn prohibit_any_illegal_number_of_words(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        NumberOfWords::illegal_values()
            .map(|n| Self::indicator_polynomial(circuit_builder, n))
            .to_vec()
    }

    fn log_derivative_accumulates_clk_next(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let challenge = |c: ChallengeId| circuit_builder.challenge(c);
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };
        let curr_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };

        (next_ext_row(ClockJumpDifferenceLookupServerLogDerivative)
            - curr_ext_row(ClockJumpDifferenceLookupServerLogDerivative))
            * (challenge(ClockJumpDifferenceLookupIndeterminate) - next_base_row(CLK))
            - next_base_row(ClockJumpDifferenceLookupMultiplicity)
    }

    fn running_evaluation_for_standard_input_remains_unchanged(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let curr_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };

        next_ext_row(InputTableEvalArg) - curr_ext_row(InputTableEvalArg)
    }

    fn running_evaluation_for_standard_output_remains_unchanged(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let curr_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };

        next_ext_row(OutputTableEvalArg) - curr_ext_row(OutputTableEvalArg)
    }

    fn grow_stack_by_n_and_read_n_symbols_from_input(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        n: usize,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let running_evaluation_update =
            Self::running_evaluation_standard_input_accumulates_n_symbols(circuit_builder, n);
        let conditional_running_evaluation_update =
            running_evaluation_update * Self::indicator_polynomial(circuit_builder, n);

        let mut constraints =
            Self::conditional_constraints_for_growing_stack_by(circuit_builder, n);
        constraints.push(conditional_running_evaluation_update);
        constraints
    }

    fn running_evaluation_standard_input_accumulates_n_symbols(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        n: usize,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let indeterminate = || circuit_builder.challenge(StandardInputIndeterminate);
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };
        let curr_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };

        let mut running_evaluation = curr_ext_row(InputTableEvalArg);
        for i in (0..n).rev() {
            let stack_element = ProcessorTable::op_stack_column_by_index(i);
            running_evaluation =
                indeterminate() * running_evaluation + next_base_row(stack_element);
        }
        next_ext_row(InputTableEvalArg) - running_evaluation
    }

    fn shrink_stack_by_n_and_write_n_symbols_to_output(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        n: usize,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let running_evaluation_update =
            Self::running_evaluation_standard_output_accumulates_n_symbols(circuit_builder, n);
        let conditional_running_evaluation_update =
            running_evaluation_update * Self::indicator_polynomial(circuit_builder, n);

        let mut constraints =
            Self::conditional_constraints_for_shrinking_stack_by(circuit_builder, n);
        constraints.push(conditional_running_evaluation_update);
        constraints
    }

    fn running_evaluation_standard_output_accumulates_n_symbols(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        n: usize,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let indeterminate = || circuit_builder.challenge(StandardOutputIndeterminate);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let curr_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };

        let mut running_evaluation = curr_ext_row(OutputTableEvalArg);
        for i in 0..n {
            let stack_element = ProcessorTable::op_stack_column_by_index(i);
            running_evaluation =
                indeterminate() * running_evaluation + curr_base_row(stack_element);
        }
        next_ext_row(OutputTableEvalArg) - running_evaluation
    }

    fn log_derivative_for_instruction_lookup_updates_correctly(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let one = || circuit_builder.b_constant(1);
        let challenge = |c: ChallengeId| circuit_builder.challenge(c);
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };
        let curr_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };

        let compressed_row = challenge(ProgramAddressWeight) * next_base_row(IP)
            + challenge(ProgramInstructionWeight) * next_base_row(CI)
            + challenge(ProgramNextInstructionWeight) * next_base_row(NIA);
        let log_derivative_updates = (next_ext_row(InstructionLookupClientLogDerivative)
            - curr_ext_row(InstructionLookupClientLogDerivative))
            * (challenge(InstructionLookupIndeterminate) - compressed_row)
            - one();
        let log_derivative_remains = next_ext_row(InstructionLookupClientLogDerivative)
            - curr_ext_row(InstructionLookupClientLogDerivative);

        (one() - next_base_row(IsPadding)) * log_derivative_updates
            + next_base_row(IsPadding) * log_derivative_remains
    }

    fn constraints_for_shrinking_stack_by_3_and_top_3_unconstrained(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u64| circuit_builder.b_constant(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        vec![
            next_base_row(ST3) - curr_base_row(ST6),
            next_base_row(ST4) - curr_base_row(ST7),
            next_base_row(ST5) - curr_base_row(ST8),
            next_base_row(ST6) - curr_base_row(ST9),
            next_base_row(ST7) - curr_base_row(ST10),
            next_base_row(ST8) - curr_base_row(ST11),
            next_base_row(ST9) - curr_base_row(ST12),
            next_base_row(ST10) - curr_base_row(ST13),
            next_base_row(ST11) - curr_base_row(ST14),
            next_base_row(ST12) - curr_base_row(ST15),
            next_base_row(OpStackPointer) - curr_base_row(OpStackPointer) + constant(3),
            Self::running_product_op_stack_accounts_for_shrinking_stack_by(circuit_builder, 3),
        ]
    }

    fn stack_shrinks_by_any_of(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        shrinkages: &[usize],
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let all_constraints_for_all_shrinkages = shrinkages
            .iter()
            .map(|&n| Self::conditional_constraints_for_shrinking_stack_by(circuit_builder, n))
            .collect_vec();

        Self::combine_mutually_exclusive_constraint_groups(
            circuit_builder,
            all_constraints_for_all_shrinkages,
        )
    }

    fn stack_grows_by_any_of(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        growths: &[usize],
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let all_constraints_for_all_growths = growths
            .iter()
            .map(|&n| Self::conditional_constraints_for_growing_stack_by(circuit_builder, n))
            .collect_vec();

        Self::combine_mutually_exclusive_constraint_groups(
            circuit_builder,
            all_constraints_for_all_growths,
        )
    }

    /// Reduces the number of constraints by summing mutually exclusive constraints. The mutual
    /// exclusion is due to the conditional nature of the constraints, which has to be guaranteed by
    /// the caller.
    ///
    /// For example, the constraints for shrinking the stack by 2, 3, and 4 elements are:
    ///
    /// ```markdown
    /// |            shrink by 2 |            shrink by 3 |            shrink by 4 |
    /// |-----------------------:|-----------------------:|-----------------------:|
    /// |     ind_2·(st0' - st2) |     ind_3·(st0' - st3) |     ind_4·(st0' - st4) |
    /// |     ind_2·(st1' - st3) |     ind_3·(st1' - st4) |     ind_4·(st1' - st5) |
    /// |                      … |                      … |                      … |
    /// |   ind_2·(st11' - st13) |   ind_3·(st11' - st14) |   ind_4·(st11' - st15) |
    /// |   ind_2·(st12' - st14) |   ind_3·(st12' - st15) | ind_4·(osp' - osp + 4) |
    /// |   ind_2·(st13' - st15) | ind_3·(osp' - osp + 3) | ind_4·(rp' - rp·fac_4) |
    /// | ind_2·(osp' - osp + 2) | ind_3·(rp' - rp·fac_3) |                        |
    /// | ind_2·(rp' - rp·fac_2) |                        |                        |
    /// ```
    ///
    /// This method sums these constraints “per row”. That is, the resulting constraints are:
    ///
    /// ```markdown
    /// |                                                  shrink by 2 or 3 or 4 |
    /// |-----------------------------------------------------------------------:|
    /// |           ind_2·(st0' - st2) + ind_3·(st0' - st3) + ind_4·(st0' - st4) |
    /// |           ind_2·(st1' - st3) + ind_3·(st1' - st4) + ind_4·(st1' - st5) |
    /// |                                                                      … |
    /// |     ind_2·(st11' - st13) + ind_3·(st11' - st14) + ind_4·(st11' - st15) |
    /// |   ind_2·(st12' - st14) + ind_3·(st12' - st15) + ind_4·(osp' - osp + 4) |
    /// | ind_2·(st13' - st15) + ind_3·(osp' - osp + 3) + ind_4·(rp' - rp·fac_4) |
    /// |                        ind_2·(osp' - osp + 2) + ind_3·(rp' - rp·fac_3) |
    /// |                                                 ind_2·(rp' - rp·fac_2) |
    /// ```
    ///
    /// Syntax in above example:
    /// - `ind_n` is the indicator polynomial for `n`
    /// - `osp` is the op stack pointer
    /// - `rp` is the running product for the permutation argument
    /// - `fac_n` is the factor for the running product
    fn combine_mutually_exclusive_constraint_groups(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        all_constraint_groups: Vec<Vec<ConstraintCircuitMonad<DualRowIndicator>>>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constraint_group_lengths = all_constraint_groups.iter().map(|x| x.len());
        let num_constraints = constraint_group_lengths.max().unwrap_or(0);

        let zero_constraint = || circuit_builder.b_constant(0);
        let mut combined_constraints = vec![];
        for i in 0..num_constraints {
            let combined_constraint = all_constraint_groups
                .iter()
                .filter_map(|constraint_group| constraint_group.get(i))
                .fold(zero_constraint(), |acc, summand| acc + summand.clone());
            combined_constraints.push(combined_constraint);
        }
        combined_constraints
    }

    fn constraints_for_shrinking_stack_by(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        n: usize,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: usize| circuit_builder.b_constant(u64::try_from(c).unwrap());
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let op_stack_pointer_shrinks_by_n =
            next_base_row(OpStackPointer) - curr_base_row(OpStackPointer) + constant(n);

        let mut constraints = vec![
            op_stack_pointer_shrinks_by_n,
            Self::running_product_op_stack_accounts_for_shrinking_stack_by(circuit_builder, n),
        ];

        for i in n..OpStackElement::COUNT {
            let curr_stack_element = ProcessorTable::op_stack_column_by_index(i);
            let next_stack_element = ProcessorTable::op_stack_column_by_index(i - n);
            let element_i_is_shifted_by_n =
                next_base_row(next_stack_element) - curr_base_row(curr_stack_element);
            constraints.push(element_i_is_shifted_by_n);
        }
        constraints
    }

    fn constraints_for_growing_stack_by(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        n: usize,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: usize| circuit_builder.b_constant(u32::try_from(c).unwrap());
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let op_stack_pointer_grows_by_n =
            next_base_row(OpStackPointer) - curr_base_row(OpStackPointer) - constant(n);

        let mut constraints = vec![
            op_stack_pointer_grows_by_n,
            Self::running_product_op_stack_accounts_for_growing_stack_by(circuit_builder, n),
        ];

        for i in 0..OpStackElement::COUNT - n {
            let curr_stack_element = ProcessorTable::op_stack_column_by_index(i);
            let next_stack_element = ProcessorTable::op_stack_column_by_index(i + n);
            let element_i_is_shifted_by_n =
                next_base_row(next_stack_element) - curr_base_row(curr_stack_element);
            constraints.push(element_i_is_shifted_by_n);
        }
        constraints
    }

    fn conditional_constraints_for_shrinking_stack_by(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        n: usize,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        Self::constraints_for_shrinking_stack_by(circuit_builder, n)
            .into_iter()
            .map(|constraint| Self::indicator_polynomial(circuit_builder, n) * constraint)
            .collect()
    }

    fn conditional_constraints_for_growing_stack_by(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        n: usize,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        Self::constraints_for_growing_stack_by(circuit_builder, n)
            .into_iter()
            .map(|constraint| Self::indicator_polynomial(circuit_builder, n) * constraint)
            .collect()
    }

    fn running_product_op_stack_accounts_for_growing_stack_by(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        n: usize,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let curr_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };
        let single_grow_factor = |op_stack_pointer_offset| {
            Self::single_factor_for_permutation_argument_with_op_stack_table(
                circuit_builder,
                CurrentBaseRow,
                op_stack_pointer_offset,
            )
        };

        let mut factor = constant(1);
        for op_stack_pointer_offset in 0..n {
            factor = factor * single_grow_factor(op_stack_pointer_offset);
        }

        next_ext_row(OpStackTablePermArg) - curr_ext_row(OpStackTablePermArg) * factor
    }

    fn running_product_op_stack_accounts_for_shrinking_stack_by(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        n: usize,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let curr_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };
        let single_shrink_factor = |op_stack_pointer_offset| {
            Self::single_factor_for_permutation_argument_with_op_stack_table(
                circuit_builder,
                NextBaseRow,
                op_stack_pointer_offset,
            )
        };

        let mut factor = constant(1);
        for op_stack_pointer_offset in 0..n {
            factor = factor * single_shrink_factor(op_stack_pointer_offset);
        }

        next_ext_row(OpStackTablePermArg) - curr_ext_row(OpStackTablePermArg) * factor
    }

    fn single_factor_for_permutation_argument_with_op_stack_table(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        row_with_shorter_stack_indicator: fn(usize) -> DualRowIndicator,
        op_stack_pointer_offset: usize,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let challenge = |c: ChallengeId| circuit_builder.challenge(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let row_with_shorter_stack = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(row_with_shorter_stack_indicator(
                col.master_base_table_index(),
            ))
        };

        let max_stack_element_index = OpStackElement::COUNT - 1;
        let stack_element_index = max_stack_element_index - op_stack_pointer_offset;
        let stack_element = ProcessorTable::op_stack_column_by_index(stack_element_index);
        let underflow_element = row_with_shorter_stack(stack_element);

        let op_stack_pointer = row_with_shorter_stack(OpStackPointer);
        let offset = constant(op_stack_pointer_offset as u32);
        let offset_op_stack_pointer = op_stack_pointer + offset;

        let compressed_row = challenge(OpStackClkWeight) * curr_base_row(CLK)
            + challenge(OpStackIb1Weight) * curr_base_row(IB1)
            + challenge(OpStackPointerWeight) * offset_op_stack_pointer
            + challenge(OpStackFirstUnderflowElementWeight) * underflow_element;
        challenge(OpStackIndeterminate) - compressed_row
    }

    fn write_to_ram_any_of(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        number_of_words: &[usize],
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let all_constraint_groups = number_of_words
            .iter()
            .map(|&n| {
                Self::conditional_constraints_for_writing_n_elements_to_ram(circuit_builder, n)
            })
            .collect_vec();
        Self::combine_mutually_exclusive_constraint_groups(circuit_builder, all_constraint_groups)
    }

    fn read_from_ram_any_of(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        number_of_words: &[usize],
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let all_constraint_groups = number_of_words
            .iter()
            .map(|&n| {
                Self::conditional_constraints_for_reading_n_elements_from_ram(circuit_builder, n)
            })
            .collect_vec();
        Self::combine_mutually_exclusive_constraint_groups(circuit_builder, all_constraint_groups)
    }

    fn conditional_constraints_for_writing_n_elements_to_ram(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        n: usize,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        Self::shrink_stack_by_n_and_write_n_elements_to_ram(circuit_builder, n)
            .into_iter()
            .map(|constraint| Self::indicator_polynomial(circuit_builder, n) * constraint)
            .collect()
    }

    fn conditional_constraints_for_reading_n_elements_from_ram(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        n: usize,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        Self::grow_stack_by_n_and_read_n_elements_from_ram(circuit_builder, n)
            .into_iter()
            .map(|constraint| Self::indicator_polynomial(circuit_builder, n) * constraint)
            .collect()
    }

    fn shrink_stack_by_n_and_write_n_elements_to_ram(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        n: usize,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: usize| circuit_builder.b_constant(u32::try_from(c).unwrap());
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let op_stack_pointer_shrinks_by_n =
            next_base_row(OpStackPointer) - curr_base_row(OpStackPointer) + constant(n);
        let ram_pointer_grows_by_n = next_base_row(ST0) - curr_base_row(ST0) - constant(n);

        let mut constraints = vec![
            op_stack_pointer_shrinks_by_n,
            ram_pointer_grows_by_n,
            Self::running_product_op_stack_accounts_for_shrinking_stack_by(circuit_builder, n),
            Self::running_product_ram_accounts_for_writing_n_elements(circuit_builder, n),
        ];

        let num_ram_pointers = 1;
        for i in n + num_ram_pointers..OpStackElement::COUNT {
            let curr_stack_element = ProcessorTable::op_stack_column_by_index(i);
            let next_stack_element = ProcessorTable::op_stack_column_by_index(i - n);
            let element_i_is_shifted_by_n =
                next_base_row(next_stack_element) - curr_base_row(curr_stack_element);
            constraints.push(element_i_is_shifted_by_n);
        }
        constraints
    }

    fn grow_stack_by_n_and_read_n_elements_from_ram(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        n: usize,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: usize| circuit_builder.b_constant(u64::try_from(c).unwrap());
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let op_stack_pointer_grows_by_n =
            next_base_row(OpStackPointer) - curr_base_row(OpStackPointer) - constant(n);
        let ram_pointer_shrinks_by_n = next_base_row(ST0) - curr_base_row(ST0) + constant(n);

        let mut constraints = vec![
            op_stack_pointer_grows_by_n,
            ram_pointer_shrinks_by_n,
            Self::running_product_op_stack_accounts_for_growing_stack_by(circuit_builder, n),
            Self::running_product_ram_accounts_for_reading_n_elements(circuit_builder, n),
        ];

        let num_ram_pointers = 1;
        for i in num_ram_pointers..OpStackElement::COUNT - n {
            let curr_stack_element = ProcessorTable::op_stack_column_by_index(i);
            let next_stack_element = ProcessorTable::op_stack_column_by_index(i + n);
            let element_i_is_shifted_by_n =
                next_base_row(next_stack_element) - curr_base_row(curr_stack_element);
            constraints.push(element_i_is_shifted_by_n);
        }
        constraints
    }

    fn running_product_ram_accounts_for_writing_n_elements(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        n: usize,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let curr_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };
        let single_write_factor = |ram_pointer_offset| {
            Self::single_factor_for_permutation_argument_with_ram_table(
                circuit_builder,
                CurrentBaseRow,
                ram_table::INSTRUCTION_TYPE_WRITE,
                ram_pointer_offset,
            )
        };

        let mut factor = constant(1);
        for ram_pointer_offset in 0..n {
            factor = factor * single_write_factor(ram_pointer_offset);
        }

        next_ext_row(RamTablePermArg) - curr_ext_row(RamTablePermArg) * factor
    }

    fn running_product_ram_accounts_for_reading_n_elements(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        n: usize,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let curr_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };
        let single_read_factor = |ram_pointer_offset| {
            Self::single_factor_for_permutation_argument_with_ram_table(
                circuit_builder,
                NextBaseRow,
                ram_table::INSTRUCTION_TYPE_READ,
                ram_pointer_offset,
            )
        };

        let mut factor = constant(1);
        for ram_pointer_offset in 0..n {
            factor = factor * single_read_factor(ram_pointer_offset);
        }

        next_ext_row(RamTablePermArg) - curr_ext_row(RamTablePermArg) * factor
    }

    fn single_factor_for_permutation_argument_with_ram_table(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        row_with_longer_stack_indicator: fn(usize) -> DualRowIndicator,
        instruction_type: BFieldElement,
        ram_pointer_offset: usize,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let b_constant = |c| circuit_builder.b_constant(c);
        let challenge = |c: ChallengeId| circuit_builder.challenge(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let row_with_longer_stack = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(row_with_longer_stack_indicator(
                col.master_base_table_index(),
            ))
        };

        let num_ram_pointers = 1;
        let ram_value_index = ram_pointer_offset + num_ram_pointers;
        let ram_value_column = ProcessorTable::op_stack_column_by_index(ram_value_index);
        let ram_value = row_with_longer_stack(ram_value_column);

        let additional_offset = match instruction_type {
            ram_table::INSTRUCTION_TYPE_READ => 1,
            ram_table::INSTRUCTION_TYPE_WRITE => 0,
            _ => panic!("Invalid instruction type"),
        };

        let ram_pointer = row_with_longer_stack(ST0);
        let offset = constant(additional_offset + ram_pointer_offset as u32);
        let offset_ram_pointer = ram_pointer + offset;

        let compressed_row = curr_base_row(CLK) * challenge(RamClkWeight)
            + b_constant(instruction_type) * challenge(RamInstructionTypeWeight)
            + offset_ram_pointer * challenge(RamPointerWeight)
            + ram_value * challenge(RamValueWeight);
        challenge(RamIndeterminate) - compressed_row
    }

    fn running_product_for_jump_stack_table_updates_correctly(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let challenge = |c: ChallengeId| circuit_builder.challenge(c);
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };
        let curr_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };

        let compressed_row = challenge(JumpStackClkWeight) * next_base_row(CLK)
            + challenge(JumpStackCiWeight) * next_base_row(CI)
            + challenge(JumpStackJspWeight) * next_base_row(JSP)
            + challenge(JumpStackJsoWeight) * next_base_row(JSO)
            + challenge(JumpStackJsdWeight) * next_base_row(JSD);

        next_ext_row(JumpStackTablePermArg)
            - curr_ext_row(JumpStackTablePermArg)
                * (challenge(JumpStackIndeterminate) - compressed_row)
    }

    fn running_evaluation_hash_input_updates_correctly(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let challenge = |c: ChallengeId| circuit_builder.challenge(c);
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };
        let curr_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };

        let hash_deselector =
            Self::instruction_deselector_next_row(circuit_builder, Instruction::Hash);
        let hash_selector = next_base_row(CI) - constant(Instruction::Hash.opcode());

        let weights = [
            HashStateWeight0,
            HashStateWeight1,
            HashStateWeight2,
            HashStateWeight3,
            HashStateWeight4,
            HashStateWeight5,
            HashStateWeight6,
            HashStateWeight7,
            HashStateWeight8,
            HashStateWeight9,
        ]
        .map(challenge);
        let state = [ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7, ST8, ST9].map(next_base_row);
        let compressed_row = weights
            .into_iter()
            .zip_eq(state)
            .map(|(weight, state)| weight * state)
            .sum();

        let running_evaluation_updates = next_ext_row(HashInputEvalArg)
            - challenge(HashInputIndeterminate) * curr_ext_row(HashInputEvalArg)
            - compressed_row;
        let running_evaluation_remains =
            next_ext_row(HashInputEvalArg) - curr_ext_row(HashInputEvalArg);

        hash_selector * running_evaluation_remains + hash_deselector * running_evaluation_updates
    }

    fn running_evaluation_hash_digest_updates_correctly(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let challenge = |c: ChallengeId| circuit_builder.challenge(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };
        let curr_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };

        let hash_deselector =
            Self::instruction_deselector_current_row(circuit_builder, Instruction::Hash);
        let hash_selector = curr_base_row(CI) - constant(Instruction::Hash.opcode());

        let weights = [
            HashStateWeight0,
            HashStateWeight1,
            HashStateWeight2,
            HashStateWeight3,
            HashStateWeight4,
        ]
        .map(challenge);
        let state = [ST0, ST1, ST2, ST3, ST4].map(next_base_row);
        let compressed_row = weights
            .into_iter()
            .zip_eq(state)
            .map(|(weight, state)| weight * state)
            .sum();

        let running_evaluation_updates = next_ext_row(HashDigestEvalArg)
            - challenge(HashDigestIndeterminate) * curr_ext_row(HashDigestEvalArg)
            - compressed_row;
        let running_evaluation_remains =
            next_ext_row(HashDigestEvalArg) - curr_ext_row(HashDigestEvalArg);

        hash_selector * running_evaluation_remains + hash_deselector * running_evaluation_updates
    }

    fn running_evaluation_sponge_updates_correctly(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let challenge = |c: ChallengeId| circuit_builder.challenge(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };
        let curr_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };

        let sponge_init_deselector =
            Self::instruction_deselector_current_row(circuit_builder, Instruction::SpongeInit);
        let sponge_absorb_deselector =
            Self::instruction_deselector_current_row(circuit_builder, Instruction::SpongeAbsorb);
        let sponge_squeeze_deselector =
            Self::instruction_deselector_current_row(circuit_builder, Instruction::SpongeSqueeze);

        let sponge_instruction_selector = (curr_base_row(CI)
            - constant(Instruction::SpongeInit.opcode()))
            * (curr_base_row(CI) - constant(Instruction::SpongeAbsorb.opcode()))
            * (curr_base_row(CI) - constant(Instruction::SpongeSqueeze.opcode()));

        let weights = [
            HashStateWeight0,
            HashStateWeight1,
            HashStateWeight2,
            HashStateWeight3,
            HashStateWeight4,
            HashStateWeight5,
            HashStateWeight6,
            HashStateWeight7,
            HashStateWeight8,
            HashStateWeight9,
        ]
        .map(challenge);
        let state = [ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7, ST8, ST9];
        let compressed_row_current = weights
            .clone()
            .into_iter()
            .zip_eq(state.map(curr_base_row))
            .map(|(weight, st_curr)| weight * st_curr)
            .sum();
        let compressed_row_next = weights
            .into_iter()
            .zip_eq(state.map(next_base_row))
            .map(|(weight, st_next)| weight * st_next)
            .sum();

        // Use domain-specific knowledge: the compressed row (i.e., random linear sum) of the
        // initial Sponge state is 0.
        let running_evaluation_updates_for_sponge_init = next_ext_row(SpongeEvalArg)
            - challenge(SpongeIndeterminate) * curr_ext_row(SpongeEvalArg)
            - challenge(HashCIWeight) * curr_base_row(CI);
        let running_evaluation_updates_for_absorb =
            running_evaluation_updates_for_sponge_init.clone() - compressed_row_current;
        let running_evaluation_updates_for_squeeze =
            running_evaluation_updates_for_sponge_init.clone() - compressed_row_next;
        let running_evaluation_remains = next_ext_row(SpongeEvalArg) - curr_ext_row(SpongeEvalArg);

        sponge_instruction_selector * running_evaluation_remains
            + sponge_init_deselector * running_evaluation_updates_for_sponge_init
            + sponge_absorb_deselector * running_evaluation_updates_for_absorb
            + sponge_squeeze_deselector * running_evaluation_updates_for_squeeze
    }

    fn log_derivative_with_u32_table_updates_correctly(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let one = || constant(1);
        let two_inverse = circuit_builder.b_constant(bfe!(2).inverse());
        let challenge = |c: ChallengeId| circuit_builder.challenge(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };
        let curr_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
        };
        let next_ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(NextExtRow(col.master_ext_table_index()))
        };

        let split_deselector =
            Self::instruction_deselector_current_row(circuit_builder, Instruction::Split);
        let lt_deselector =
            Self::instruction_deselector_current_row(circuit_builder, Instruction::Lt);
        let and_deselector =
            Self::instruction_deselector_current_row(circuit_builder, Instruction::And);
        let xor_deselector =
            Self::instruction_deselector_current_row(circuit_builder, Instruction::Xor);
        let pow_deselector =
            Self::instruction_deselector_current_row(circuit_builder, Instruction::Pow);
        let log_2_floor_deselector =
            Self::instruction_deselector_current_row(circuit_builder, Instruction::Log2Floor);
        let div_mod_deselector =
            Self::instruction_deselector_current_row(circuit_builder, Instruction::DivMod);
        let pop_count_deselector =
            Self::instruction_deselector_current_row(circuit_builder, Instruction::PopCount);

        let running_sum = curr_ext_row(U32LookupClientLogDerivative);
        let running_sum_next = next_ext_row(U32LookupClientLogDerivative);

        let split_factor = challenge(U32Indeterminate)
            - challenge(U32LhsWeight) * next_base_row(ST0)
            - challenge(U32RhsWeight) * next_base_row(ST1)
            - challenge(U32CiWeight) * curr_base_row(CI);
        let binop_factor = challenge(U32Indeterminate)
            - challenge(U32LhsWeight) * curr_base_row(ST0)
            - challenge(U32RhsWeight) * curr_base_row(ST1)
            - challenge(U32CiWeight) * curr_base_row(CI)
            - challenge(U32ResultWeight) * next_base_row(ST0);
        let xor_factor = challenge(U32Indeterminate)
            - challenge(U32LhsWeight) * curr_base_row(ST0)
            - challenge(U32RhsWeight) * curr_base_row(ST1)
            - challenge(U32CiWeight) * constant(Instruction::And.opcode())
            - challenge(U32ResultWeight)
                * (curr_base_row(ST0) + curr_base_row(ST1) - next_base_row(ST0))
                * two_inverse;
        let unop_factor = challenge(U32Indeterminate)
            - challenge(U32LhsWeight) * curr_base_row(ST0)
            - challenge(U32CiWeight) * curr_base_row(CI)
            - challenge(U32ResultWeight) * next_base_row(ST0);
        let div_mod_factor_for_lt = challenge(U32Indeterminate)
            - challenge(U32LhsWeight) * next_base_row(ST0)
            - challenge(U32RhsWeight) * curr_base_row(ST1)
            - challenge(U32CiWeight) * constant(Instruction::Lt.opcode())
            - challenge(U32ResultWeight);
        let div_mod_factor_for_range_check = challenge(U32Indeterminate)
            - challenge(U32LhsWeight) * curr_base_row(ST0)
            - challenge(U32RhsWeight) * next_base_row(ST1)
            - challenge(U32CiWeight) * constant(Instruction::Split.opcode());

        let running_sum_absorbs_split_factor =
            (running_sum_next.clone() - running_sum.clone()) * split_factor - one();
        let running_sum_absorbs_binop_factor =
            (running_sum_next.clone() - running_sum.clone()) * binop_factor - one();
        let running_sum_absorb_xor_factor =
            (running_sum_next.clone() - running_sum.clone()) * xor_factor - one();
        let running_sum_absorbs_unop_factor =
            (running_sum_next.clone() - running_sum.clone()) * unop_factor - one();

        let split_summand = split_deselector * running_sum_absorbs_split_factor;
        let lt_summand = lt_deselector * running_sum_absorbs_binop_factor.clone();
        let and_summand = and_deselector * running_sum_absorbs_binop_factor.clone();
        let xor_summand = xor_deselector * running_sum_absorb_xor_factor;
        let pow_summand = pow_deselector * running_sum_absorbs_binop_factor;
        let log_2_floor_summand = log_2_floor_deselector * running_sum_absorbs_unop_factor.clone();
        let div_mod_summand = div_mod_deselector
            * ((running_sum_next.clone() - running_sum.clone())
                * div_mod_factor_for_lt.clone()
                * div_mod_factor_for_range_check.clone()
                - div_mod_factor_for_lt
                - div_mod_factor_for_range_check);
        let pop_count_summand = pop_count_deselector * running_sum_absorbs_unop_factor;
        let no_update_summand = (one() - curr_base_row(IB2)) * (running_sum_next - running_sum);

        split_summand
            + lt_summand
            + and_summand
            + xor_summand
            + pow_summand
            + log_2_floor_summand
            + div_mod_summand
            + pop_count_summand
            + no_update_summand
    }

    pub fn transition_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u64| circuit_builder.b_constant(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        // constraints common to all instructions
        let clk_increases_by_1 = next_base_row(CLK) - curr_base_row(CLK) - constant(1);
        let is_padding_is_0_or_does_not_change =
            curr_base_row(IsPadding) * (next_base_row(IsPadding) - curr_base_row(IsPadding));

        let instruction_independent_constraints =
            vec![clk_increases_by_1, is_padding_is_0_or_does_not_change];

        // instruction-specific constraints
        let all_transition_constraints_by_instruction = ALL_INSTRUCTIONS.map(|instruction| {
            Self::get_transition_constraints_for_instruction(circuit_builder, instruction)
        });
        let all_instructions_and_their_transition_constraints = ALL_INSTRUCTIONS
            .into_iter()
            .zip_eq(all_transition_constraints_by_instruction)
            .collect_vec()
            .try_into()
            .unwrap();

        let deselected_transition_constraints =
            Self::combine_instruction_constraints_with_deselectors(
                circuit_builder,
                all_instructions_and_their_transition_constraints,
            );

        // if next row is padding row: disable transition constraints, enable padding constraints
        let doubly_deselected_transition_constraints =
            Self::combine_transition_constraints_with_padding_constraints(
                circuit_builder,
                deselected_transition_constraints,
            );

        let table_linking_constraints = vec![
            Self::log_derivative_accumulates_clk_next(circuit_builder),
            Self::log_derivative_for_instruction_lookup_updates_correctly(circuit_builder),
            Self::running_product_for_jump_stack_table_updates_correctly(circuit_builder),
            Self::running_evaluation_hash_input_updates_correctly(circuit_builder),
            Self::running_evaluation_hash_digest_updates_correctly(circuit_builder),
            Self::running_evaluation_sponge_updates_correctly(circuit_builder),
            Self::log_derivative_with_u32_table_updates_correctly(circuit_builder),
        ];

        [
            instruction_independent_constraints,
            doubly_deselected_transition_constraints,
            table_linking_constraints,
        ]
        .concat()
    }

    pub fn terminal_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(BaseRow(col.master_base_table_index()))
        };
        let constant = |c| circuit_builder.b_constant(c);

        // In the last row, register “current instruction” `ci` corresponds to instruction `halt`.
        let last_ci_is_halt = base_row(CI) - constant(Instruction::Halt.opcode_b());

        vec![last_ci_is_halt]
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

    use assert2::assert;
    use ndarray::Array2;
    use proptest::collection::vec;
    use proptest_arbitrary_interop::arb;
    use rand::thread_rng;
    use rand::Rng;
    use strum::IntoEnumIterator;
    use test_strategy::proptest;

    use crate::error::InstructionError::DivisionByZero;
    use crate::instruction::Instruction;
    use crate::op_stack::NumberOfWords::*;
    use crate::op_stack::OpStackElement;
    use crate::program::Program;
    use crate::shared_tests::ProgramAndInput;
    use crate::stark::tests::master_tables_for_low_security_level;
    use crate::table::master_table::*;
    use crate::triton_asm;
    use crate::triton_program;
    use crate::vm::NUM_HELPER_VARIABLE_REGISTERS;
    use crate::NonDeterminism;

    use super::*;

    /// Does printing recurse infinitely?
    #[test]
    fn print_simple_processor_table_row() {
        let program = triton_program!(push 2 sponge_init assert halt);
        let err = program.run([].into(), [].into()).unwrap_err();
        println!("\n{}", err.vm_state);
    }

    #[derive(Debug, Clone)]
    struct TestRows {
        pub challenges: Challenges,
        pub consecutive_master_base_table_rows: Array2<BFieldElement>,
        pub consecutive_ext_base_table_rows: Array2<XFieldElement>,
    }

    #[derive(Debug, Clone)]
    struct TestRowsDebugInfo {
        pub instruction: Instruction,
        pub debug_cols_curr_row: Vec<ProcessorBaseTableColumn>,
        pub debug_cols_next_row: Vec<ProcessorBaseTableColumn>,
    }

    fn test_row_from_program(program: Program, row_num: usize) -> TestRows {
        test_row_from_program_with_input(ProgramAndInput::new(program), row_num)
    }

    fn test_row_from_program_with_input(
        program_and_input: ProgramAndInput,
        row_num: usize,
    ) -> TestRows {
        let (_, _, master_base_table, master_ext_table, challenges) =
            master_tables_for_low_security_level(program_and_input);
        TestRows {
            challenges,
            consecutive_master_base_table_rows: master_base_table
                .trace_table()
                .slice(s![row_num..=row_num + 1, ..])
                .to_owned(),
            consecutive_ext_base_table_rows: master_ext_table
                .trace_table()
                .slice(s![row_num..=row_num + 1, ..])
                .to_owned(),
        }
    }

    fn assert_constraints_for_rows_with_debug_info(
        test_rows: &[TestRows],
        debug_info: TestRowsDebugInfo,
    ) {
        let instruction = debug_info.instruction;
        let circuit_builder = ConstraintCircuitBuilder::new();
        let transition_constraints = ExtProcessorTable::get_transition_constraints_for_instruction(
            &circuit_builder,
            instruction,
        );

        for (case_idx, rows) in test_rows.iter().enumerate() {
            let curr_row = rows.consecutive_master_base_table_rows.slice(s![0, ..]);
            let next_row = rows.consecutive_master_base_table_rows.slice(s![1, ..]);

            println!("Testing all constraints of {instruction} for test case {case_idx}…");
            for &c in &debug_info.debug_cols_curr_row {
                print!("{c}  = {}, ", curr_row[c.master_base_table_index()]);
            }
            println!();
            for &c in &debug_info.debug_cols_next_row {
                print!("{c}' = {}, ", next_row[c.master_base_table_index()]);
            }
            println!();

            assert!(
                instruction.opcode_b() == curr_row[CI.master_base_table_index()],
                "The test is trying to check the wrong transition constraint polynomials."
            );

            for (constraint_idx, constraint) in transition_constraints.iter().enumerate() {
                let evaluation_result = constraint.clone().consume().evaluate(
                    rows.consecutive_master_base_table_rows.view(),
                    rows.consecutive_ext_base_table_rows.view(),
                    &rows.challenges.challenges,
                );
                assert!(
                    evaluation_result.is_zero(),
                    "For case {case_idx}, transition constraint polynomial with \
                    index {constraint_idx} must evaluate to zero. Got {evaluation_result} instead.",
                );
            }
        }
    }

    #[proptest(cases = 20)]
    fn transition_constraints_for_instruction_pop_n(#[strategy(arb())] n: NumberOfWords) {
        let program = triton_program!(push 1 push 2 push 3 push 4 push 5 pop {n} halt);

        let test_rows = [test_row_from_program(program, 5)];
        let debug_info = TestRowsDebugInfo {
            instruction: Pop(n),
            debug_cols_curr_row: vec![ST0, ST1, ST2],
            debug_cols_next_row: vec![ST0, ST1, ST2],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_push() {
        let test_rows = [test_row_from_program(triton_program!(push 1 halt), 0)];

        let debug_info = TestRowsDebugInfo {
            instruction: Push(bfe!(1)),
            debug_cols_curr_row: vec![ST0, ST1, ST2],
            debug_cols_next_row: vec![ST0, ST1, ST2],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[proptest(cases = 20)]
    fn transition_constraints_for_instruction_divine_n(#[strategy(arb())] n: NumberOfWords) {
        let program = triton_program! { divine {n} halt };

        let non_determinism = (1..=16).map(|b| bfe!(b)).collect_vec();
        let program_and_input = ProgramAndInput::new(program).with_non_determinism(non_determinism);
        let test_rows = [test_row_from_program_with_input(program_and_input, 0)];
        let debug_info = TestRowsDebugInfo {
            instruction: Divine(n),
            debug_cols_curr_row: vec![ST0, ST1, ST2],
            debug_cols_next_row: vec![ST0, ST1, ST2],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_dup() {
        let programs = [
            triton_program!(dup  0 halt),
            triton_program!(dup  1 halt),
            triton_program!(dup  2 halt),
            triton_program!(dup  3 halt),
            triton_program!(dup  4 halt),
            triton_program!(dup  5 halt),
            triton_program!(dup  6 halt),
            triton_program!(dup  7 halt),
            triton_program!(dup  8 halt),
            triton_program!(dup  9 halt),
            triton_program!(dup 10 halt),
            triton_program!(dup 11 halt),
            triton_program!(dup 12 halt),
            triton_program!(dup 13 halt),
            triton_program!(dup 14 halt),
            triton_program!(dup 15 halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 0));

        let debug_info = TestRowsDebugInfo {
            instruction: Dup(OpStackElement::ST0),
            debug_cols_curr_row: vec![ST0, ST1, ST2],
            debug_cols_next_row: vec![ST0, ST1, ST2],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_swap() {
        let programs = [
            triton_program!(swap  1 halt),
            triton_program!(swap  2 halt),
            triton_program!(swap  3 halt),
            triton_program!(swap  4 halt),
            triton_program!(swap  5 halt),
            triton_program!(swap  6 halt),
            triton_program!(swap  7 halt),
            triton_program!(swap  8 halt),
            triton_program!(swap  9 halt),
            triton_program!(swap 10 halt),
            triton_program!(swap 11 halt),
            triton_program!(swap 12 halt),
            triton_program!(swap 13 halt),
            triton_program!(swap 14 halt),
            triton_program!(swap 15 halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 0));
        let debug_info = TestRowsDebugInfo {
            instruction: Swap(OpStackElement::ST0),
            debug_cols_curr_row: vec![ST0, ST1, ST2],
            debug_cols_next_row: vec![ST0, ST1, ST2],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_skiz() {
        let programs = [
            triton_program!(push 1 skiz halt),        // ST0 is non-zero
            triton_program!(push 0 skiz assert halt), // ST0 is zero, next instruction of size 1
            triton_program!(push 0 skiz push 1 halt), // ST0 is zero, next instruction of size 2
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 1));
        let debug_info = TestRowsDebugInfo {
            instruction: Skiz,
            debug_cols_curr_row: vec![IP, NIA, ST0, HV5, HV4, HV3, HV2],
            debug_cols_next_row: vec![IP],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_call() {
        let programs = [triton_program!(call label label: halt)];
        let test_rows = programs.map(|program| test_row_from_program(program, 0));
        let debug_info = TestRowsDebugInfo {
            instruction: Call(BFieldElement::default()),
            debug_cols_curr_row: vec![IP, CI, NIA, JSP, JSO, JSD],
            debug_cols_next_row: vec![IP, CI, NIA, JSP, JSO, JSD],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_return() {
        let programs = [triton_program!(call label halt label: return)];
        let test_rows = programs.map(|program| test_row_from_program(program, 1));
        let debug_info = TestRowsDebugInfo {
            instruction: Return,
            debug_cols_curr_row: vec![IP, JSP, JSO, JSD],
            debug_cols_next_row: vec![IP, JSP, JSO, JSD],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_recurse() {
        let programs =
            [triton_program!(push 2 call label halt label: push -1 add dup 0 skiz recurse return)];
        let test_rows = programs.map(|program| test_row_from_program(program, 6));
        let debug_info = TestRowsDebugInfo {
            instruction: Recurse,
            debug_cols_curr_row: vec![IP, JSP, JSO, JSD],
            debug_cols_next_row: vec![IP, JSP, JSO, JSD],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_read_mem() {
        let programs = [
            triton_program!(push 1 read_mem 1 push 0 eq assert assert halt),
            triton_program!(push 2 read_mem 2 push 0 eq assert swap 1 push 2 eq assert halt),
            triton_program!(push 3 read_mem 3 push 0 eq assert swap 2 push 3 eq assert halt),
            triton_program!(push 4 read_mem 4 push 0 eq assert swap 3 push 4 eq assert halt),
            triton_program!(push 5 read_mem 5 push 0 eq assert swap 4 push 5 eq assert halt),
        ];
        let initial_ram = (1..=5)
            .map(|i| (bfe!(i), bfe!(i)))
            .collect::<HashMap<_, _>>();
        let non_determinism = NonDeterminism::default().with_ram(initial_ram);
        let programs_with_input = programs.map(|program| {
            ProgramAndInput::new(program).with_non_determinism(non_determinism.clone())
        });
        let test_rows = programs_with_input.map(|p_w_i| test_row_from_program_with_input(p_w_i, 1));
        let debug_info = TestRowsDebugInfo {
            instruction: ReadMem(N1),
            debug_cols_curr_row: vec![ST0, ST1],
            debug_cols_next_row: vec![ST0, ST1],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_write_mem() {
        let push_10_elements = triton_asm![push 2; 10];
        let programs = [
            triton_program!({&push_10_elements} write_mem 1 halt),
            triton_program!({&push_10_elements} write_mem 2 halt),
            triton_program!({&push_10_elements} write_mem 3 halt),
            triton_program!({&push_10_elements} write_mem 4 halt),
            triton_program!({&push_10_elements} write_mem 5 halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 10));
        let debug_info = TestRowsDebugInfo {
            instruction: WriteMem(N1),
            debug_cols_curr_row: vec![ST0, ST1],
            debug_cols_next_row: vec![ST0, ST1],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_divine_sibling() {
        let programs = [
            triton_program!(push 2 swap 5 divine_sibling halt),
            triton_program!(push 3 swap 5 divine_sibling halt),
        ];
        let dummy_digest = Digest::new([1, 2, 3, 4, 5].map(BFieldElement::new));
        let non_determinism = NonDeterminism::new(vec![]).with_digests(vec![dummy_digest]);
        let programs_with_input = programs.map(|program| {
            ProgramAndInput::new(program).with_non_determinism(non_determinism.clone())
        });
        let test_rows = programs_with_input.map(|p_w_i| test_row_from_program_with_input(p_w_i, 2));

        let debug_info = TestRowsDebugInfo {
            instruction: DivineSibling,
            debug_cols_curr_row: vec![ST0, ST1, ST2, ST3, ST4, ST5],
            debug_cols_next_row: vec![ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7, ST8, ST9, ST10],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_sponge_init() {
        let programs = [triton_program!(sponge_init halt)];
        let test_rows = programs.map(|program| test_row_from_program(program, 0));
        let debug_info = TestRowsDebugInfo {
            instruction: SpongeInit,
            debug_cols_curr_row: vec![],
            debug_cols_next_row: vec![],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_sponge_absorb() {
        let push_10_zeros = triton_asm![push 0; 10];
        let push_10_ones = triton_asm![push 1; 10];
        let programs = [
            triton_program!(sponge_init {&push_10_zeros} sponge_absorb halt),
            triton_program!(sponge_init {&push_10_ones} sponge_absorb halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 11));
        let debug_info = TestRowsDebugInfo {
            instruction: SpongeAbsorb,
            debug_cols_curr_row: vec![ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7, ST8, ST9, ST10],
            debug_cols_next_row: vec![ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7, ST8, ST9, ST10],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_sponge_squeeze() {
        let programs = [triton_program!(sponge_init sponge_squeeze halt)];
        let test_rows = programs.map(|program| test_row_from_program(program, 1));
        let debug_info = TestRowsDebugInfo {
            instruction: SpongeSqueeze,
            debug_cols_curr_row: vec![ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7, ST8, ST9, ST10],
            debug_cols_next_row: vec![ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7, ST8, ST9, ST10],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_eq() {
        let programs = [
            triton_program!(push 3 push 3 eq assert halt),
            triton_program!(push 3 push 2 eq push 0 eq assert halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 2));
        let debug_info = TestRowsDebugInfo {
            instruction: Eq,
            debug_cols_curr_row: vec![ST0, ST1, HV0],
            debug_cols_next_row: vec![ST0],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_split() {
        let programs = [
            triton_program!(push -1 split halt),
            triton_program!(push  0 split halt),
            triton_program!(push  1 split halt),
            triton_program!(push  2 split halt),
            triton_program!(push  3 split halt),
            // test pushing push 2^32 +- 1
            triton_program!(push 4294967295 split halt),
            triton_program!(push 4294967296 split halt),
            triton_program!(push 4294967297 split halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 1));
        let debug_info = TestRowsDebugInfo {
            instruction: Split,
            debug_cols_curr_row: vec![ST0, ST1, HV0],
            debug_cols_next_row: vec![ST0, ST1, HV0],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_lt() {
        let programs = [
            triton_program!(push   3 push   3 lt push 0 eq assert halt),
            triton_program!(push   3 push   2 lt push 1 eq assert halt),
            triton_program!(push   2 push   3 lt push 0 eq assert halt),
            triton_program!(push 512 push 513 lt push 0 eq assert halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 2));
        let debug_info = TestRowsDebugInfo {
            instruction: Lt,
            debug_cols_curr_row: vec![ST0, ST1],
            debug_cols_next_row: vec![ST0],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_and() {
        let test_rows = [test_row_from_program(
            triton_program!(push 5 push 12 and push 4 eq assert halt),
            2,
        )];
        let debug_info = TestRowsDebugInfo {
            instruction: And,
            debug_cols_curr_row: vec![ST0, ST1],
            debug_cols_next_row: vec![ST0],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_xor() {
        let test_rows = [test_row_from_program(
            triton_program!(push 5 push 12 xor push 9 eq assert halt),
            2,
        )];
        let debug_info = TestRowsDebugInfo {
            instruction: Xor,
            debug_cols_curr_row: vec![ST0, ST1],
            debug_cols_next_row: vec![ST0],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_log2floor() {
        let programs = [
            triton_program!(push  1 log_2_floor push  0 eq assert halt),
            triton_program!(push  2 log_2_floor push  1 eq assert halt),
            triton_program!(push  3 log_2_floor push  1 eq assert halt),
            triton_program!(push  4 log_2_floor push  2 eq assert halt),
            triton_program!(push  5 log_2_floor push  2 eq assert halt),
            triton_program!(push  6 log_2_floor push  2 eq assert halt),
            triton_program!(push  7 log_2_floor push  2 eq assert halt),
            triton_program!(push  8 log_2_floor push  3 eq assert halt),
            triton_program!(push  9 log_2_floor push  3 eq assert halt),
            triton_program!(push 10 log_2_floor push  3 eq assert halt),
            triton_program!(push 11 log_2_floor push  3 eq assert halt),
            triton_program!(push 12 log_2_floor push  3 eq assert halt),
            triton_program!(push 13 log_2_floor push  3 eq assert halt),
            triton_program!(push 14 log_2_floor push  3 eq assert halt),
            triton_program!(push 15 log_2_floor push  3 eq assert halt),
            triton_program!(push 16 log_2_floor push  4 eq assert halt),
            triton_program!(push 17 log_2_floor push  4 eq assert halt),
        ];

        let test_rows = programs.map(|program| test_row_from_program(program, 1));
        let debug_info = TestRowsDebugInfo {
            instruction: Log2Floor,
            debug_cols_curr_row: vec![ST0, ST1],
            debug_cols_next_row: vec![ST0],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_pow() {
        let programs = [
            triton_program!(push 0 push  0 pow push   1 eq assert halt),
            triton_program!(push 1 push  0 pow push   0 eq assert halt),
            triton_program!(push 2 push  0 pow push   0 eq assert halt),
            triton_program!(push 0 push  1 pow push   1 eq assert halt),
            triton_program!(push 1 push  1 pow push   1 eq assert halt),
            triton_program!(push 2 push  1 pow push   1 eq assert halt),
            triton_program!(push 0 push  2 pow push   1 eq assert halt),
            triton_program!(push 1 push  2 pow push   2 eq assert halt),
            triton_program!(push 2 push  2 pow push   4 eq assert halt),
            triton_program!(push 3 push  2 pow push   8 eq assert halt),
            triton_program!(push 4 push  2 pow push  16 eq assert halt),
            triton_program!(push 5 push  2 pow push  32 eq assert halt),
            triton_program!(push 0 push  3 pow push   1 eq assert halt),
            triton_program!(push 1 push  3 pow push   3 eq assert halt),
            triton_program!(push 2 push  3 pow push   9 eq assert halt),
            triton_program!(push 3 push  3 pow push  27 eq assert halt),
            triton_program!(push 4 push  3 pow push  81 eq assert halt),
            triton_program!(push 0 push 17 pow push   1 eq assert halt),
            triton_program!(push 1 push 17 pow push  17 eq assert halt),
            triton_program!(push 2 push 17 pow push 289 eq assert halt),
        ];

        let test_rows = programs.map(|program| test_row_from_program(program, 2));
        let debug_info = TestRowsDebugInfo {
            instruction: Pow,
            debug_cols_curr_row: vec![ST0, ST1],
            debug_cols_next_row: vec![ST0],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_div_mod() {
        let programs = [
            triton_program!(push 2 push 3 div_mod push 1 eq assert push 1 eq assert halt),
            triton_program!(push 3 push 7 div_mod push 1 eq assert push 2 eq assert halt),
            triton_program!(push 4 push 7 div_mod push 3 eq assert push 1 eq assert halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 2));
        let debug_info = TestRowsDebugInfo {
            instruction: DivMod,
            debug_cols_curr_row: vec![ST0, ST1],
            debug_cols_next_row: vec![ST0, ST1],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn division_by_zero_is_impossible() {
        let program = ProgramAndInput::new(triton_program! { div_mod });
        let err = program.run().unwrap_err();
        assert_eq!(DivisionByZero, err.source);
    }

    #[test]
    fn transition_constraints_for_instruction_xxadd() {
        let programs = [
            triton_program!(push 5 push 6 push 7 push 8 push 9 push 10 xxadd halt),
            triton_program!(push 2 push 3 push 4 push -2 push -3 push -4 xxadd halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 6));
        let debug_info = TestRowsDebugInfo {
            instruction: XxAdd,
            debug_cols_curr_row: vec![ST0, ST1, ST2, ST3, ST4, ST5],
            debug_cols_next_row: vec![ST0, ST1, ST2, ST3, ST4, ST5],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_xxmul() {
        let programs = [
            triton_program!(push 5 push 6 push 7 push 8 push 9 push 10 xxmul halt),
            triton_program!(push 2 push 3 push 4 push -2 push -3 push -4 xxmul halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 6));
        let debug_info = TestRowsDebugInfo {
            instruction: XxMul,
            debug_cols_curr_row: vec![ST0, ST1, ST2, ST3, ST4, ST5],
            debug_cols_next_row: vec![ST0, ST1, ST2, ST3, ST4, ST5],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_xinvert() {
        let programs = [
            triton_program!(push 5 push 6 push 7 xinvert halt),
            triton_program!(push -2 push -3 push -4 xinvert halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 3));
        let debug_info = TestRowsDebugInfo {
            instruction: XInvert,
            debug_cols_curr_row: vec![ST0, ST1, ST2],
            debug_cols_next_row: vec![ST0, ST1, ST2],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_xbmul() {
        let programs = [
            triton_program!(push 5 push 6 push 7 push 2 xbmul halt),
            triton_program!(push 2 push 3 push 4 push -2 xbmul halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 4));
        let debug_info = TestRowsDebugInfo {
            instruction: XbMul,
            debug_cols_curr_row: vec![ST0, ST1, ST2, ST3, OpStackPointer, HV0],
            debug_cols_next_row: vec![ST0, ST1, ST2, ST3, OpStackPointer, HV0],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[proptest(cases = 20)]
    fn transition_constraints_for_instruction_read_io_n(#[strategy(arb())] n: NumberOfWords) {
        let program = triton_program! {read_io {n} halt};

        let public_input = (1..=16).map(|i| bfe!(i)).collect_vec();
        let program_and_input = ProgramAndInput::new(program).with_input(public_input);
        let test_rows = [test_row_from_program_with_input(program_and_input, 0)];
        let debug_info = TestRowsDebugInfo {
            instruction: ReadIo(n),
            debug_cols_curr_row: vec![ST0, ST1, ST2],
            debug_cols_next_row: vec![ST0, ST1, ST2],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[proptest(cases = 20)]
    fn transition_constraints_for_instruction_write_io_n(#[strategy(arb())] n: NumberOfWords) {
        let program = triton_program! {divine 5 write_io {n} halt};

        let non_determinism = (1..=16).map(|b| bfe!(b)).collect_vec();
        let program_and_input = ProgramAndInput::new(program).with_non_determinism(non_determinism);
        let test_rows = [test_row_from_program_with_input(program_and_input, 1)];
        let debug_info = TestRowsDebugInfo {
            instruction: WriteIo(n),
            debug_cols_curr_row: vec![ST0, ST1, ST2, ST3, ST4, ST5],
            debug_cols_next_row: vec![ST0, ST1, ST2, ST3, ST4, ST5],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn instruction_deselector_gives_0_for_all_other_instructions() {
        let circuit_builder = ConstraintCircuitBuilder::new();

        let mut master_base_table = Array2::zeros([2, NUM_BASE_COLUMNS]);
        let master_ext_table = Array2::zeros([2, NUM_EXT_COLUMNS]);

        // For this test, dummy challenges suffice to evaluate the constraints.
        let dummy_challenges = Challenges::default().challenges;
        for instruction in ALL_INSTRUCTIONS {
            use ProcessorBaseTableColumn::*;
            let deselector = ExtProcessorTable::instruction_deselector_current_row(
                &circuit_builder,
                instruction,
            );

            println!("\n\nThe Deselector for instruction {instruction} is:\n{deselector}",);

            // Negative tests
            for other_instruction in ALL_INSTRUCTIONS
                .into_iter()
                .filter(|other_instruction| *other_instruction != instruction)
            {
                let mut curr_row = master_base_table.slice_mut(s![0, ..]);
                curr_row[IB0.master_base_table_index()] = other_instruction.ib(InstructionBit::IB0);
                curr_row[IB1.master_base_table_index()] = other_instruction.ib(InstructionBit::IB1);
                curr_row[IB2.master_base_table_index()] = other_instruction.ib(InstructionBit::IB2);
                curr_row[IB3.master_base_table_index()] = other_instruction.ib(InstructionBit::IB3);
                curr_row[IB4.master_base_table_index()] = other_instruction.ib(InstructionBit::IB4);
                curr_row[IB5.master_base_table_index()] = other_instruction.ib(InstructionBit::IB5);
                curr_row[IB6.master_base_table_index()] = other_instruction.ib(InstructionBit::IB6);
                let result = deselector.clone().consume().evaluate(
                    master_base_table.view(),
                    master_ext_table.view(),
                    &dummy_challenges,
                );

                assert!(
                    result.is_zero(),
                    "Deselector for {instruction} should return 0 for all other instructions, \
                    including {other_instruction} whose opcode is {}",
                    other_instruction.opcode()
                )
            }

            // Positive tests
            let mut curr_row = master_base_table.slice_mut(s![0, ..]);
            curr_row[IB0.master_base_table_index()] = instruction.ib(InstructionBit::IB0);
            curr_row[IB1.master_base_table_index()] = instruction.ib(InstructionBit::IB1);
            curr_row[IB2.master_base_table_index()] = instruction.ib(InstructionBit::IB2);
            curr_row[IB3.master_base_table_index()] = instruction.ib(InstructionBit::IB3);
            curr_row[IB4.master_base_table_index()] = instruction.ib(InstructionBit::IB4);
            curr_row[IB5.master_base_table_index()] = instruction.ib(InstructionBit::IB5);
            curr_row[IB6.master_base_table_index()] = instruction.ib(InstructionBit::IB6);
            let result = deselector.consume().evaluate(
                master_base_table.view(),
                master_ext_table.view(),
                &dummy_challenges,
            );
            assert!(
                !result.is_zero(),
                "Deselector for {instruction} should be non-zero when CI is {}",
                instruction.opcode()
            )
        }
    }

    #[test]
    fn print_number_and_degrees_of_transition_constraints_for_all_instructions() {
        println!();
        println!("| Instruction     | #polys | max deg | Degrees");
        println!("|:----------------|-------:|--------:|:------------");
        let circuit_builder = ConstraintCircuitBuilder::new();
        for instruction in ALL_INSTRUCTIONS {
            let constraints = ExtProcessorTable::get_transition_constraints_for_instruction(
                &circuit_builder,
                instruction,
            );
            let degrees = constraints
                .iter()
                .map(|circuit| circuit.clone().consume().degree())
                .collect_vec();
            let max_degree = degrees.iter().max().unwrap_or(&0);
            let degrees_str = degrees.iter().map(|d| format!("{d}")).join(", ");
            println!(
                "| {:<15} | {:>6} | {:>7} | [{}]",
                format!("{instruction}"),
                constraints.len(),
                max_degree,
                degrees_str,
            );
        }
    }

    #[test]
    fn opcode_decomposition_for_skiz_is_unique() {
        let max_value_of_skiz_constraint_for_nia_decomposition =
            (3 << 7) * (3 << 5) * (3 << 3) * (3 << 1) * 2;
        for instruction in Instruction::iter() {
            assert!(
                instruction.opcode() < max_value_of_skiz_constraint_for_nia_decomposition,
                "Opcode for {instruction} is too high."
            );
        }
    }

    #[test]
    fn range_check_for_skiz_is_as_efficient_as_possible() {
        let range_check_constraints =
            ExtProcessorTable::next_instruction_range_check_constraints_for_instruction_skiz(
                &ConstraintCircuitBuilder::new(),
            );
        let range_check_constraints = range_check_constraints.iter();
        let all_degrees = range_check_constraints.map(|c| c.clone().consume().degree());
        let max_constraint_degree = all_degrees.max().unwrap_or(0);
        assert!(
            AIR_TARGET_DEGREE <= max_constraint_degree,
            "Can the range check constraints be of a higher degree, saving columns?"
        );
    }

    #[test]
    fn helper_variables_in_bounds() {
        let circuit_builder = ConstraintCircuitBuilder::new();
        for index in 0..NUM_HELPER_VARIABLE_REGISTERS {
            ExtProcessorTable::helper_variable(&circuit_builder, index);
        }
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn helper_variables_out_of_bounds() {
        let index = thread_rng().gen_range(NUM_HELPER_VARIABLE_REGISTERS..usize::MAX);
        let circuit_builder = ConstraintCircuitBuilder::new();
        ExtProcessorTable::helper_variable(&circuit_builder, index);
    }

    #[test]
    fn indicator_polynomial_in_bounds() {
        let circuit_builder = ConstraintCircuitBuilder::new();
        for index in 0..16 {
            ExtProcessorTable::indicator_polynomial(&circuit_builder, index);
        }
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn indicator_polynomial_out_of_bounds() {
        let index = thread_rng().gen_range(16..usize::MAX);
        let circuit_builder = ConstraintCircuitBuilder::new();
        ExtProcessorTable::indicator_polynomial(&circuit_builder, index);
    }

    #[test]
    fn can_get_op_stack_column_for_in_range_index() {
        for index in 0..OpStackElement::COUNT {
            let _ = ProcessorTable::op_stack_column_by_index(index);
        }
    }

    #[proptest]
    #[should_panic(expected = "[0, 15]")]
    fn cannot_get_op_stack_column_for_out_of_range_index(
        #[strategy(OpStackElement::COUNT..)] index: usize,
    ) {
        let _ = ProcessorTable::op_stack_column_by_index(index);
    }

    #[proptest]
    fn constructing_factor_for_op_stack_table_running_product_never_panics(
        has_previous_row: bool,
        #[strategy(vec(arb(), BASE_WIDTH))] previous_row: Vec<BFieldElement>,
        #[strategy(vec(arb(), BASE_WIDTH))] current_row: Vec<BFieldElement>,
        #[strategy(arb())] challenges: Challenges,
    ) {
        let previous_row = Array1::from(previous_row);
        let current_row = Array1::from(current_row);
        let maybe_previous_row = match has_previous_row {
            true => Some(previous_row.view()),
            false => None,
        };
        let _ = ProcessorTable::factor_for_op_stack_table_running_product(
            maybe_previous_row,
            current_row.view(),
            &challenges,
        );
    }

    #[proptest]
    fn constructing_factor_for_ram_table_running_product_never_panics(
        has_previous_row: bool,
        #[strategy(vec(arb(), BASE_WIDTH))] previous_row: Vec<BFieldElement>,
        #[strategy(vec(arb(), BASE_WIDTH))] current_row: Vec<BFieldElement>,
        #[strategy(arb())] challenges: Challenges,
    ) {
        let previous_row = Array1::from(previous_row);
        let current_row = Array1::from(current_row);
        let maybe_previous_row = match has_previous_row {
            true => Some(previous_row.view()),
            false => None,
        };
        let _ = ProcessorTable::factor_for_ram_table_running_product(
            maybe_previous_row,
            current_row.view(),
            &challenges,
        );
    }
}
