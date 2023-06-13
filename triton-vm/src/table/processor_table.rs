use std::cmp::max;
use std::fmt::Display;
use std::ops::Mul;

use itertools::Itertools;
use ndarray::parallel::prelude::*;
use ndarray::s;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use ndarray::Axis;
use num_traits::One;
use num_traits::Zero;
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::b_field_element::BFIELD_ONE;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use triton_opcodes::instruction::AnInstruction::*;
use triton_opcodes::instruction::Instruction;
use triton_opcodes::instruction::ALL_INSTRUCTIONS;
use triton_opcodes::ord_n::Ord8;

use crate::table::challenges::ChallengeId;
use crate::table::challenges::ChallengeId::*;
use crate::table::challenges::Challenges;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::ConstraintCircuitMonad;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::DualRowIndicator::*;
use crate::table::constraint_circuit::InputIndicator;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::*;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::EvalArg;
use crate::table::cross_table_argument::LookupArg;
use crate::table::cross_table_argument::PermArg;
use crate::table::table_column::MasterBaseTableColumn;
use crate::table::table_column::MasterExtTableColumn;
use crate::table::table_column::ProcessorBaseTableColumn;
use crate::table::table_column::ProcessorBaseTableColumn::*;
use crate::table::table_column::ProcessorExtTableColumn;
use crate::table::table_column::ProcessorExtTableColumn::*;
use crate::vm::AlgebraicExecutionTrace;

pub const BASE_WIDTH: usize = ProcessorBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = ProcessorExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

#[derive(Debug, Clone)]
pub struct ProcessorTable {}

impl ProcessorTable {
    pub fn fill_trace(
        processor_table: &mut ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
        clk_jump_diffs_op_stack: &[BFieldElement],
        clk_jump_diffs_ram: &[BFieldElement],
        clk_jump_diffs_jump_stack: &[BFieldElement],
    ) {
        // compute the lookup multiplicities of the clock jump differences
        let num_rows = aet.processor_trace.nrows();
        let mut clk_jump_diff_multiplicities = Array1::zeros([num_rows]);
        for clk_jump_diff in clk_jump_diffs_op_stack.iter() {
            let clk = clk_jump_diff.value() as usize;
            match clk < num_rows {
                true => clk_jump_diff_multiplicities[clk] += BFIELD_ONE,
                false => panic!(
                    "Op Stack: clock jump diff {clk} must fit in trace with {num_rows} rows."
                ),
            }
        }
        for clk_jump_diff in clk_jump_diffs_ram.iter() {
            let clk = clk_jump_diff.value() as usize;
            match clk < num_rows {
                true => clk_jump_diff_multiplicities[clk] += BFIELD_ONE,
                false => {
                    panic!("RAM: clock jump diff {clk} must fit in trace with {num_rows} rows.")
                }
            }
        }
        for clk_jump_diff in clk_jump_diffs_jump_stack.iter() {
            let clk = clk_jump_diff.value() as usize;
            match clk < num_rows {
                true => clk_jump_diff_multiplicities[clk] += BFIELD_ONE,
                false => panic!(
                    "Jump Stack: clock jump diff {clk} must fit in trace with {num_rows} rows."
                ),
            }
        }

        // fill the processor table from the AET and the lookup multiplicities
        let mut processor_table_to_fill =
            processor_table.slice_mut(s![0..aet.processor_trace.nrows(), ..]);
        aet.processor_trace
            .clone()
            .move_into(&mut processor_table_to_fill);
        processor_table_to_fill
            .column_mut(ClockJumpDifferenceLookupMultiplicity.base_table_index())
            .assign(&clk_jump_diff_multiplicities);
    }

    pub fn pad_trace(
        processor_table: &mut ArrayViewMut2<BFieldElement>,
        processor_table_len: usize,
    ) {
        assert!(
            processor_table_len > 0,
            "Processor Table must have at least one row."
        );
        let mut padding_template = processor_table.row(processor_table_len - 1).to_owned();
        padding_template[IsPadding.base_table_index()] = BFieldElement::one();
        padding_template[ClockJumpDifferenceLookupMultiplicity.base_table_index()] =
            BFieldElement::zero();
        processor_table
            .slice_mut(s![processor_table_len.., ..])
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| row.assign(&padding_template));

        let clk_range = processor_table_len..processor_table.nrows();
        let clk_col = Array1::from_iter(clk_range.map(|a| BFieldElement::new(a as u64)));
        clk_col.move_into(
            processor_table.slice_mut(s![processor_table_len.., CLK.base_table_index()]),
        );

        // The 3 memory-like tables do not have a padding indicator. Hence, clock jump differences
        // are being looked up in their padding sections. The clock jump differences in that
        // section are always 1. The lookup multiplicities of clock value 1 must be increased
        // accordingly: one per padding row, times the number of memory-like tables, which is 3.
        let num_padding_rows = 3 * (processor_table.nrows() - processor_table_len);
        let num_pad_rows = BFieldElement::new(num_padding_rows as u64);
        let mut row_1 = processor_table.row_mut(1);
        row_1[ClockJumpDifferenceLookupMultiplicity.base_table_index()] += num_pad_rows;
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

            // Input table
            if let Some(prev_row) = previous_row {
                if prev_row[CI.base_table_index()] == Instruction::ReadIo.opcode_b() {
                    let input_symbol = current_row[ST0.base_table_index()];
                    input_table_running_evaluation = input_table_running_evaluation
                        * challenges.get_challenge(StandardInputIndeterminate)
                        + input_symbol;
                }
            }

            // Output table
            if current_row[CI.base_table_index()] == Instruction::WriteIo.opcode_b() {
                let output_symbol = current_row[ST0.base_table_index()];
                output_table_running_evaluation = output_table_running_evaluation
                    * challenges.get_challenge(StandardOutputIndeterminate)
                    + output_symbol;
            }

            // Program table
            if current_row[IsPadding.base_table_index()].is_zero() {
                let ip = current_row[IP.base_table_index()];
                let ci = current_row[CI.base_table_index()];
                let nia = current_row[NIA.base_table_index()];
                let compressed_row_for_instruction_lookup = ip
                    * challenges.get_challenge(ProgramAddressWeight)
                    + ci * challenges.get_challenge(ProgramInstructionWeight)
                    + nia * challenges.get_challenge(ProgramNextInstructionWeight);
                instruction_lookup_log_derivative += (challenges
                    .get_challenge(InstructionLookupIndeterminate)
                    - compressed_row_for_instruction_lookup)
                    .inverse();
            }

            // OpStack table
            let clk = current_row[CLK.base_table_index()];
            let ib1 = current_row[IB1.base_table_index()];
            let osp = current_row[OSP.base_table_index()];
            let osv = current_row[OSV.base_table_index()];
            let compressed_row_for_op_stack_table_permutation_argument = clk
                * challenges.get_challenge(OpStackClkWeight)
                + ib1 * challenges.get_challenge(OpStackIb1Weight)
                + osp * challenges.get_challenge(OpStackOspWeight)
                + osv * challenges.get_challenge(OpStackOsvWeight);
            op_stack_table_running_product *= challenges.get_challenge(OpStackIndeterminate)
                - compressed_row_for_op_stack_table_permutation_argument;

            // RAM Table
            let ramv = current_row[RAMV.base_table_index()];
            let ramp = current_row[RAMP.base_table_index()];
            let previous_instruction = current_row[PreviousInstruction.base_table_index()];
            let compressed_row_for_ram_table_permutation_argument = clk
                * challenges.get_challenge(RamClkWeight)
                + ramp * challenges.get_challenge(RamRampWeight)
                + ramv * challenges.get_challenge(RamRamvWeight)
                + previous_instruction * challenges.get_challenge(RamPreviousInstructionWeight);
            ram_table_running_product *= challenges.get_challenge(RamIndeterminate)
                - compressed_row_for_ram_table_permutation_argument;

            // JumpStack Table
            let ci = current_row[CI.base_table_index()];
            let jsp = current_row[JSP.base_table_index()];
            let jso = current_row[JSO.base_table_index()];
            let jsd = current_row[JSD.base_table_index()];
            let compressed_row_for_jump_stack_table = clk
                * challenges.get_challenge(JumpStackClkWeight)
                + ci * challenges.get_challenge(JumpStackCiWeight)
                + jsp * challenges.get_challenge(JumpStackJspWeight)
                + jso * challenges.get_challenge(JumpStackJsoWeight)
                + jsd * challenges.get_challenge(JumpStackJsdWeight);
            jump_stack_running_product *= challenges.get_challenge(JumpStackIndeterminate)
                - compressed_row_for_jump_stack_table;

            // Hash Table – Hash's input from Processor to Hash Coprocessor
            let st_0_through_9 = [ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7, ST8, ST9]
                .map(|st| current_row[st.base_table_index()]);
            let hash_state_weights = [
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
            .map(|id| challenges.get_challenge(id));
            let compressed_row_for_hash_input_and_sponge: XFieldElement = st_0_through_9
                .into_iter()
                .zip_eq(hash_state_weights.into_iter())
                .map(|(st, weight)| weight * st)
                .sum();
            let compressed_row_for_hash_digest: XFieldElement = st_0_through_9[5..=9]
                .iter()
                .zip_eq(hash_state_weights[0..=4].iter())
                .map(|(&st, &weight)| weight * st)
                .sum();

            if current_row[CI.base_table_index()] == Instruction::Hash.opcode_b() {
                hash_input_running_evaluation = hash_input_running_evaluation
                    * challenges.get_challenge(HashInputIndeterminate)
                    + compressed_row_for_hash_input_and_sponge;
            }

            // Hash Table – Hash's output from Hash Coprocessor to Processor
            if let Some(prev_row) = previous_row {
                if prev_row[CI.base_table_index()] == Instruction::Hash.opcode_b() {
                    hash_digest_running_evaluation = hash_digest_running_evaluation
                        * challenges.get_challenge(HashDigestIndeterminate)
                        + compressed_row_for_hash_digest;
                }
            }

            // Hash Table – Sponge
            if let Some(prev_row) = previous_row {
                if prev_row[CI.base_table_index()] == Instruction::AbsorbInit.opcode_b()
                    || prev_row[CI.base_table_index()] == Instruction::Absorb.opcode_b()
                    || prev_row[CI.base_table_index()] == Instruction::Squeeze.opcode_b()
                {
                    sponge_running_evaluation = sponge_running_evaluation
                        * challenges.get_challenge(SpongeIndeterminate)
                        + challenges.get_challenge(HashCIWeight) * prev_row[CI.base_table_index()]
                        + compressed_row_for_hash_input_and_sponge;
                }
            }

            // U32 Table
            if let Some(prev_row) = previous_row {
                let previously_current_instruction = prev_row[CI.base_table_index()];
                if previously_current_instruction == Instruction::Split.opcode_b() {
                    let compressed_row = current_row[ST0.base_table_index()]
                        * challenges.get_challenge(U32LhsWeight)
                        + current_row[ST1.base_table_index()]
                            * challenges.get_challenge(U32RhsWeight)
                        + prev_row[CI.base_table_index()] * challenges.get_challenge(U32CiWeight);
                    u32_table_running_sum_log_derivative +=
                        (challenges.get_challenge(U32Indeterminate) - compressed_row).inverse();
                }
                if previously_current_instruction == Instruction::Lt.opcode_b()
                    || previously_current_instruction == Instruction::And.opcode_b()
                    || previously_current_instruction == Instruction::Pow.opcode_b()
                {
                    let compressed_row = prev_row[ST0.base_table_index()]
                        * challenges.get_challenge(U32LhsWeight)
                        + prev_row[ST1.base_table_index()] * challenges.get_challenge(U32RhsWeight)
                        + prev_row[CI.base_table_index()] * challenges.get_challenge(U32CiWeight)
                        + current_row[ST0.base_table_index()]
                            * challenges.get_challenge(U32ResultWeight);
                    u32_table_running_sum_log_derivative +=
                        (challenges.get_challenge(U32Indeterminate) - compressed_row).inverse();
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
                        (st0_prev + st1_prev - st0) / BFieldElement::new(2);
                    let compressed_row = st0_prev * challenges.get_challenge(U32LhsWeight)
                        + st1_prev * challenges.get_challenge(U32RhsWeight)
                        + Instruction::And.opcode_b() * challenges.get_challenge(U32CiWeight)
                        + from_xor_in_processor_to_and_in_u32_coprocessor
                            * challenges.get_challenge(U32ResultWeight);
                    u32_table_running_sum_log_derivative +=
                        (challenges.get_challenge(U32Indeterminate) - compressed_row).inverse();
                }
                if previously_current_instruction == Instruction::Log2Floor.opcode_b()
                    || previously_current_instruction == Instruction::PopCount.opcode_b()
                {
                    let compressed_row = prev_row[ST0.base_table_index()]
                        * challenges.get_challenge(U32LhsWeight)
                        + prev_row[CI.base_table_index()] * challenges.get_challenge(U32CiWeight)
                        + current_row[ST0.base_table_index()]
                            * challenges.get_challenge(U32ResultWeight);
                    u32_table_running_sum_log_derivative +=
                        (challenges.get_challenge(U32Indeterminate) - compressed_row).inverse();
                }
                if previously_current_instruction == Instruction::Div.opcode_b() {
                    let compressed_row_for_lt_check = current_row[ST0.base_table_index()]
                        * challenges.get_challenge(U32LhsWeight)
                        + prev_row[ST1.base_table_index()] * challenges.get_challenge(U32RhsWeight)
                        + Instruction::Lt.opcode_b() * challenges.get_challenge(U32CiWeight)
                        + BFieldElement::one() * challenges.get_challenge(U32ResultWeight);
                    let compressed_row_for_range_check = prev_row[ST0.base_table_index()]
                        * challenges.get_challenge(U32LhsWeight)
                        + current_row[ST1.base_table_index()]
                            * challenges.get_challenge(U32RhsWeight)
                        + Instruction::Split.opcode_b() * challenges.get_challenge(U32CiWeight);
                    u32_table_running_sum_log_derivative +=
                        (challenges.get_challenge(U32Indeterminate) - compressed_row_for_lt_check)
                            .inverse();
                    u32_table_running_sum_log_derivative += (challenges
                        .get_challenge(U32Indeterminate)
                        - compressed_row_for_range_check)
                        .inverse();
                }
            }

            // Lookup Argument for clock jump differences
            let lookup_multiplicity =
                current_row[ClockJumpDifferenceLookupMultiplicity.base_table_index()];
            clock_jump_diff_lookup_op_stack_log_derivative +=
                (challenges.get_challenge(ClockJumpDifferenceLookupIndeterminate) - clk).inverse()
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
}

#[derive(Debug, Clone)]
pub struct ExtProcessorTable {}

impl ExtProcessorTable {
    pub fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c.into());
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
        let st11_is_0 = base_row(ST11);
        let st12_is_0 = base_row(ST12);
        let st13_is_0 = base_row(ST13);
        let st14_is_0 = base_row(ST14);
        let st15_is_0 = base_row(ST15);
        let osp_is_16 = base_row(OSP) - constant(16);
        let osv_is_0 = base_row(OSV);
        let ramv_is_0 = base_row(RAMV);
        let ramp_is_0 = base_row(RAMP);
        let previous_instruction_is_0 = base_row(PreviousInstruction);

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

        // op-stack table
        let op_stack_indeterminate = challenge(OpStackIndeterminate);
        let op_stack_ib1_weight = challenge(OpStackIb1Weight);
        let op_stack_osp_weight = challenge(OpStackOspWeight);
        // note: `clk` and `osv` are already constrained to be 0, `osp` to be 16
        let compressed_row_for_op_stack_table =
            op_stack_ib1_weight * base_row(IB1) + op_stack_osp_weight * constant(16);
        let running_product_for_op_stack_table_is_initialized_correctly =
            ext_row(OpStackTablePermArg)
                - x_constant(PermArg::default_initial())
                    * (op_stack_indeterminate - compressed_row_for_op_stack_table);

        // ram table
        let ram_indeterminate = challenge(RamIndeterminate);
        // note: `clk`, `ramp`, and `ramv` are already constrained to be 0.
        let compressed_row_for_ram_table = constant(0);
        let running_product_for_ram_table_is_initialized_correctly = ext_row(RamTablePermArg)
            - x_constant(PermArg::default_initial())
                * (ram_indeterminate - compressed_row_for_ram_table);

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
            st11_is_0,
            st12_is_0,
            st13_is_0,
            st14_is_0,
            st15_is_0,
            osp_is_16,
            osv_is_0,
            ramv_is_0,
            ramp_is_0,
            previous_instruction_is_0,
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
        let constant = |c: u32| circuit_builder.b_constant(c.into());
        let base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(BaseRow(col.master_base_table_index()))
        };

        // The composition of instruction buckets ib0-ib7 corresponds the current instruction ci.
        let ib_composition = base_row(IB0)
            + constant(1 << 1) * base_row(IB1)
            + constant(1 << 2) * base_row(IB2)
            + constant(1 << 3) * base_row(IB3)
            + constant(1 << 4) * base_row(IB4)
            + constant(1 << 5) * base_row(IB5)
            + constant(1 << 6) * base_row(IB6)
            + constant(1 << 7) * base_row(IB7);
        let ci_corresponds_to_ib0_thru_ib7 = base_row(CI) - ib_composition;

        let ib0_is_bit = base_row(IB0) * (base_row(IB0) - constant(1));
        let ib1_is_bit = base_row(IB1) * (base_row(IB1) - constant(1));
        let ib2_is_bit = base_row(IB2) * (base_row(IB2) - constant(1));
        let ib3_is_bit = base_row(IB3) * (base_row(IB3) - constant(1));
        let ib4_is_bit = base_row(IB4) * (base_row(IB4) - constant(1));
        let ib5_is_bit = base_row(IB5) * (base_row(IB5) - constant(1));
        let ib6_is_bit = base_row(IB6) * (base_row(IB6) - constant(1));
        let ib7_is_bit = base_row(IB7) * (base_row(IB7) - constant(1));
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
            ib7_is_bit,
            is_padding_is_bit,
            ci_corresponds_to_ib0_thru_ib7,
            clock_jump_diff_lookup_multiplicity_is_0_in_padding_rows,
        ]
    }

    fn indicator_polynomial(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        i: usize,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let one = || circuit_builder.b_constant(1_u32.into());
        let hv = |idx| match idx {
            0 => circuit_builder.input(CurrentBaseRow(HV0.master_base_table_index())),
            1 => circuit_builder.input(CurrentBaseRow(HV1.master_base_table_index())),
            2 => circuit_builder.input(CurrentBaseRow(HV2.master_base_table_index())),
            3 => circuit_builder.input(CurrentBaseRow(HV3.master_base_table_index())),
            j => panic!("Index for helper variable must be in range 0..4, got {j}."),
        };

        match i {
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
            _ => panic!("No indicator polynomial with index {i} exists: there are only 16."),
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

        let zero_poly = circuit_builder.b_constant(0_u32.into());
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
        let constant = |c: u64| circuit_builder.b_constant(c.into());
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
        let constant = |c: u32| circuit_builder.b_constant(c.into());
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
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        vec![
            next_base_row(RAMV) - curr_base_row(RAMV),
            next_base_row(RAMP) - curr_base_row(RAMP),
        ]
    }

    fn instruction_group_op_stack_remains_and_top_eleven_elements_unconstrained(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        vec![
            next_base_row(ST11) - curr_base_row(ST11),
            next_base_row(ST12) - curr_base_row(ST12),
            next_base_row(ST13) - curr_base_row(ST13),
            next_base_row(ST14) - curr_base_row(ST14),
            next_base_row(ST15) - curr_base_row(ST15),
            // The top of the OpStack underflow, i.e., osv, does not change.
            next_base_row(OSV) - curr_base_row(OSV),
            // The OpStack pointer, osp, does not change.
            next_base_row(OSP) - curr_base_row(OSP),
        ]
    }

    fn instruction_group_op_stack_remains_and_top_ten_elements_unconstrained(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        let specific_constraints = vec![next_base_row(ST10) - curr_base_row(ST10)];
        let inherited_constraints =
            Self::instruction_group_op_stack_remains_and_top_eleven_elements_unconstrained(
                circuit_builder,
            );

        [specific_constraints, inherited_constraints].concat()
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

        let specific_constraints = vec![
            next_base_row(ST3) - curr_base_row(ST3),
            next_base_row(ST4) - curr_base_row(ST4),
            next_base_row(ST5) - curr_base_row(ST5),
            next_base_row(ST6) - curr_base_row(ST6),
            next_base_row(ST7) - curr_base_row(ST7),
            next_base_row(ST8) - curr_base_row(ST8),
            next_base_row(ST9) - curr_base_row(ST9),
        ];
        let inherited_constraints =
            Self::instruction_group_op_stack_remains_and_top_ten_elements_unconstrained(
                circuit_builder,
            );

        [specific_constraints, inherited_constraints].concat()
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
        let constant = |c: u32| circuit_builder.b_constant(c.into());
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        vec![
            // The stack element in st1 is moved into st2.
            next_base_row(ST2) - curr_base_row(ST1),
            // And so on...
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
            // The stack element in st15 is moved to the top of OpStack underflow, i.e., osv.
            next_base_row(OSV) - curr_base_row(ST15),
            // The OpStack pointer is incremented by 1.
            next_base_row(OSP) - (curr_base_row(OSP) + constant(1)),
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
        let constant = |c: u32| circuit_builder.b_constant(c.into());
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        vec![
            // The stack element in st4 is moved into st3.
            next_base_row(ST3) - curr_base_row(ST4),
            // The stack element in st5 is moved into st4.
            next_base_row(ST4) - curr_base_row(ST5),
            // And so on...
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
            // The stack element at the top of OpStack underflow, i.e., osv, is moved into st15.
            next_base_row(ST15) - curr_base_row(OSV),
            // The OpStack pointer, osp, is decremented by 1.
            next_base_row(OSP) - (curr_base_row(OSP) - constant(1)),
            // The helper variable register hv3 holds the inverse of (osp - 16).
            (curr_base_row(OSP) - constant(16)) * curr_base_row(HV3) - constant(1),
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
        let constant = |c: u32| circuit_builder.b_constant(c.into());
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
        let constant = |c: u32| circuit_builder.b_constant(c.into());
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
        instruction_bucket_polynomials: [ConstraintCircuitMonad<II>; Ord8::COUNT],
    ) -> ConstraintCircuitMonad<II> {
        let one = circuit_builder.b_constant(1_u32.into());

        let selector_bits: [_; Ord8::COUNT] = [
            instruction.ib(Ord8::IB0),
            instruction.ib(Ord8::IB1),
            instruction.ib(Ord8::IB2),
            instruction.ib(Ord8::IB3),
            instruction.ib(Ord8::IB4),
            instruction.ib(Ord8::IB5),
            instruction.ib(Ord8::IB6),
            instruction.ib(Ord8::IB7),
        ];
        let deselector_polynomials =
            selector_bits.map(|b| one.clone() - circuit_builder.b_constant(b));

        instruction_bucket_polynomials
            .into_iter()
            .zip_eq(deselector_polynomials.into_iter())
            .map(|(bucket_poly, deselector_poly)| bucket_poly - deselector_poly)
            .fold(one, ConstraintCircuitMonad::mul)
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

        let instruction_bucket_polynomials = [
            curr_base_row(IB0),
            curr_base_row(IB1),
            curr_base_row(IB2),
            curr_base_row(IB3),
            curr_base_row(IB4),
            curr_base_row(IB5),
            curr_base_row(IB6),
            curr_base_row(IB7),
        ];

        Self::instruction_deselector_common_functionality(
            circuit_builder,
            instruction,
            instruction_bucket_polynomials,
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

        let instruction_bucket_polynomials = [
            next_base_row(IB0),
            next_base_row(IB1),
            next_base_row(IB2),
            next_base_row(IB3),
            next_base_row(IB4),
            next_base_row(IB5),
            next_base_row(IB6),
            next_base_row(IB7),
        ];

        Self::instruction_deselector_common_functionality(
            circuit_builder,
            instruction,
            instruction_bucket_polynomials,
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

        let instruction_bucket_polynomials = [
            base_row(IB0),
            base_row(IB1),
            base_row(IB2),
            base_row(IB3),
            base_row(IB4),
            base_row(IB5),
            base_row(IB6),
            base_row(IB7),
        ];

        Self::instruction_deselector_common_functionality(
            circuit_builder,
            instruction,
            instruction_bucket_polynomials,
        )
    }

    fn instruction_pop(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_shrink_op_stack(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
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
        ]
        .concat()
    }

    fn instruction_divine(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_grow_op_stack(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
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
        ]
        .concat()
    }

    fn instruction_swap(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let one = || circuit_builder.b_constant(1_u32.into());
        let indicator_poly = |idx| Self::indicator_polynomial(circuit_builder, idx);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
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
            next_base_row(OSV) - curr_base_row(OSV),
            next_base_row(OSP) - curr_base_row(OSP),
        ];
        [
            specific_constraints,
            Self::instruction_group_decompose_arg(circuit_builder),
            Self::instruction_group_step_2(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
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
        ]
        .concat()
    }

    fn instruction_skiz(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c.into());
        let one = || constant(1);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        // The next instruction nia is decomposed into helper variables hv.
        let nia_decomposes_to_hvs =
            curr_base_row(NIA) - (curr_base_row(HV0) + constant(2) * curr_base_row(HV1));

        // The relevant helper variable hv1 is either 0 or 1.
        // Here, hv0 == 1 means that nia takes an argument.
        let hv0_is_0_or_1 = curr_base_row(HV0) * (curr_base_row(HV0) - one());

        // If `st0` is non-zero, register `ip` is incremented by 1.
        // If `st0` is 0 and `nia` takes no argument, register `ip` is incremented by 2.
        // If `st0` is 0 and `nia` takes an argument, register `ip` is incremented by 3.
        //
        // Written as Disjunctive Normal Form, the last constraint can be expressed as:
        // 6. (Register `st0` is 0 or `ip` is incremented by 1), and
        // (`st0` has a multiplicative inverse or `hv` is 1 or `ip` is incremented by 2), and
        // (`st0` has a multiplicative inverse or `hv0` is 0 or `ip` is incremented by 3).
        let ip_case_1 = (next_base_row(IP) - curr_base_row(IP) - constant(1)) * curr_base_row(ST0);
        let ip_case_2 = (next_base_row(IP) - curr_base_row(IP) - constant(2))
            * (curr_base_row(ST0) * curr_base_row(HV2) - one())
            * (curr_base_row(HV0) - one());
        let ip_case_3 = (next_base_row(IP) - curr_base_row(IP) - constant(3))
            * (curr_base_row(ST0) * curr_base_row(HV2) - one())
            * curr_base_row(HV0);
        let ip_incr_by_1_or_2_or_3 = ip_case_1 + ip_case_2 + ip_case_3;

        let specific_constraints =
            vec![nia_decomposes_to_hvs, hv0_is_0_or_1, ip_incr_by_1_or_2_or_3];
        [
            specific_constraints,
            Self::instruction_group_keep_jump_stack(circuit_builder),
            Self::instruction_group_shrink_op_stack(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
        ]
        .concat()
    }

    fn instruction_call(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c.into());
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
        ]
        .concat()
    }

    fn instruction_return(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c.into());
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
        ]
        .concat()
    }

    fn instruction_assert(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c.into());
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
        ]
        .concat()
    }

    fn instruction_read_mem(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        // the RAM pointer is overwritten with st0
        let update_ramp = next_base_row(RAMP) - curr_base_row(ST0);

        // The top of the stack is overwritten with the RAM value.
        let st0_becomes_ramv = next_base_row(ST0) - next_base_row(RAMV);

        let specific_constraints = vec![update_ramp, st0_becomes_ramv];
        [
            specific_constraints,
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_grow_op_stack(circuit_builder),
        ]
        .concat()
    }

    fn instruction_write_mem(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        // the RAM pointer is overwritten with st1
        let update_ramp = next_base_row(RAMP) - curr_base_row(ST1);

        // The RAM value is overwritten with the top of the stack.
        let ramv_becomes_st0 = next_base_row(RAMV) - curr_base_row(ST0);

        let specific_constraints = vec![update_ramp, ramv_becomes_st0];
        [
            specific_constraints,
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_shrink_op_stack(circuit_builder),
        ]
        .concat()
    }

    /// Two Evaluation Arguments with the Hash Table guarantee correct transition.
    fn instruction_hash(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_op_stack_remains_and_top_ten_elements_unconstrained(
                circuit_builder,
            ),
            Self::instruction_group_keep_ram(circuit_builder),
        ]
        .concat()
    }

    /// Recall that in a Merkle tree, the indices of left (respectively right)
    /// leafs have 0 (respectively 1) as their least significant bit. The first
    /// two polynomials achieve that helper variable hv0 holds the result of
    /// st10 mod 2. The second polynomial sets the new value of st10 to st10 div 2.
    fn instruction_divine_sibling(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c.into());
        let one = || constant(1);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        // Helper variable hv0 is either 0 or 1.
        let hv0_is_0_or_1 = curr_base_row(HV0) * (curr_base_row(HV0) - one());

        // The 11th stack register is shifted by 1 bit to the right.
        let st10_is_shifted_1_bit_right =
            next_base_row(ST10) * constant(2) + curr_base_row(HV0) - curr_base_row(ST10);

        // The second pentuplet either stays where it is, or is moved to the top
        let maybe_move_st5 = (one() - curr_base_row(HV0))
            * (curr_base_row(ST5) - next_base_row(ST0))
            + curr_base_row(HV0) * (curr_base_row(ST5) - next_base_row(ST5));
        let maybe_move_st6 = (one() - curr_base_row(HV0))
            * (curr_base_row(ST6) - next_base_row(ST1))
            + curr_base_row(HV0) * (curr_base_row(ST6) - next_base_row(ST6));
        let maybe_move_st7 = (one() - curr_base_row(HV0))
            * (curr_base_row(ST7) - next_base_row(ST2))
            + curr_base_row(HV0) * (curr_base_row(ST7) - next_base_row(ST7));
        let maybe_move_st8 = (one() - curr_base_row(HV0))
            * (curr_base_row(ST8) - next_base_row(ST3))
            + curr_base_row(HV0) * (curr_base_row(ST8) - next_base_row(ST8));
        let maybe_move_st9 = (one() - curr_base_row(HV0))
            * (curr_base_row(ST9) - next_base_row(ST4))
            + curr_base_row(HV0) * (curr_base_row(ST9) - next_base_row(ST9));

        let specific_constraints = vec![
            hv0_is_0_or_1,
            st10_is_shifted_1_bit_right,
            maybe_move_st5,
            maybe_move_st6,
            maybe_move_st7,
            maybe_move_st8,
            maybe_move_st9,
        ];
        [
            specific_constraints,
            Self::instruction_group_op_stack_remains_and_top_eleven_elements_unconstrained(
                circuit_builder,
            ),
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
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
            Self::instruction_group_keep_op_stack(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
        ]
        .concat()
    }

    fn instruction_absorb_init(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_keep_op_stack(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
        ]
        .concat()
    }

    fn instruction_absorb(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_keep_op_stack(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
        ]
        .concat()
    }

    fn instruction_squeeze(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_op_stack_remains_and_top_ten_elements_unconstrained(
                circuit_builder,
            ),
            Self::instruction_group_keep_ram(circuit_builder),
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
        ]
        .concat()
    }

    fn instruction_invert(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c.into());
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
        ]
        .concat()
    }

    fn instruction_eq(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c.into());
        let one = || constant(1);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        // Helper variable hv0 is the inverse-or-zero of the difference of the stack's two top-most
        // elements: `hv0·(hv0·(st1 - st0) - 1)`
        let hv0_is_inverse_of_diff_or_hv0_is_0 = curr_base_row(HV0)
            * (curr_base_row(HV0) * (curr_base_row(ST1) - curr_base_row(ST0)) - one());

        // Helper variable hv0 is the inverse-or-zero of the difference of the stack's two
        // top-most elements: `(st1 - st0)·(hv0·(st1 - st0) - 1)`
        let hv0_is_inverse_of_diff_or_diff_is_0 = (curr_base_row(ST1) - curr_base_row(ST0))
            * (curr_base_row(HV0) * (curr_base_row(ST1) - curr_base_row(ST0)) - one());

        // The new top of the stack is 1 if the difference between the stack's two top-most
        // elements is not invertible, 0 otherwise: `st0' - (1 - hv0·(st1 - st0))`
        let st0_becomes_1_if_diff_is_not_invertible = next_base_row(ST0)
            - (one() - curr_base_row(HV0) * (curr_base_row(ST1) - curr_base_row(ST0)));

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
        ]
        .concat()
    }

    fn instruction_split(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u64| circuit_builder.b_constant(c.into());
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
        ]
        .concat()
    }

    fn instruction_div(
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

        let st0_becomes_st0_plus_st3 =
            next_base_row(ST0) - (curr_base_row(ST0) + curr_base_row(ST3));
        let st1_becomes_st1_plus_st4 =
            next_base_row(ST1) - (curr_base_row(ST1) + curr_base_row(ST4));
        let st2_becomes_st2_plus_st5 =
            next_base_row(ST2) - (curr_base_row(ST2) + curr_base_row(ST5));

        let specific_constraints = vec![
            st0_becomes_st0_plus_st3,
            st1_becomes_st1_plus_st4,
            st2_becomes_st2_plus_st5,
        ];
        [
            specific_constraints,
            Self::instruction_group_op_stack_remains_and_top_three_elements_unconstrained(
                circuit_builder,
            ),
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
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
            Self::instruction_group_op_stack_remains_and_top_three_elements_unconstrained(
                circuit_builder,
            ),
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
        ]
        .concat()
    }

    fn instruction_xinv(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u64| circuit_builder.b_constant(c.into());
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
        ]
        .concat()
    }

    fn instruction_read_io(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_grow_op_stack(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
        ]
        .concat()
    }

    fn instruction_write_io(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        [
            Self::instruction_group_step_1(circuit_builder),
            Self::instruction_group_shrink_op_stack(circuit_builder),
            Self::instruction_group_keep_ram(circuit_builder),
        ]
        .concat()
    }

    fn get_transition_constraints_for_instruction(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
        instruction: Instruction,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        match instruction {
            Pop => ExtProcessorTable::instruction_pop(circuit_builder),
            Push(_) => ExtProcessorTable::instruction_push(circuit_builder),
            Divine(_) => ExtProcessorTable::instruction_divine(circuit_builder),
            Dup(_) => ExtProcessorTable::instruction_dup(circuit_builder),
            Swap(_) => ExtProcessorTable::instruction_swap(circuit_builder),
            Nop => ExtProcessorTable::instruction_nop(circuit_builder),
            Skiz => ExtProcessorTable::instruction_skiz(circuit_builder),
            Call(_) => ExtProcessorTable::instruction_call(circuit_builder),
            Return => ExtProcessorTable::instruction_return(circuit_builder),
            Recurse => ExtProcessorTable::instruction_recurse(circuit_builder),
            Assert => ExtProcessorTable::instruction_assert(circuit_builder),
            Halt => ExtProcessorTable::instruction_halt(circuit_builder),
            ReadMem => ExtProcessorTable::instruction_read_mem(circuit_builder),
            WriteMem => ExtProcessorTable::instruction_write_mem(circuit_builder),
            Hash => ExtProcessorTable::instruction_hash(circuit_builder),
            DivineSibling => ExtProcessorTable::instruction_divine_sibling(circuit_builder),
            AssertVector => ExtProcessorTable::instruction_assert_vector(circuit_builder),
            AbsorbInit => ExtProcessorTable::instruction_absorb_init(circuit_builder),
            Absorb => ExtProcessorTable::instruction_absorb(circuit_builder),
            Squeeze => ExtProcessorTable::instruction_squeeze(circuit_builder),
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
            Div => ExtProcessorTable::instruction_div(circuit_builder),
            PopCount => ExtProcessorTable::instruction_pop_count(circuit_builder),
            XxAdd => ExtProcessorTable::instruction_xxadd(circuit_builder),
            XxMul => ExtProcessorTable::instruction_xxmul(circuit_builder),
            XInvert => ExtProcessorTable::instruction_xinv(circuit_builder),
            XbMul => ExtProcessorTable::instruction_xbmul(circuit_builder),
            ReadIo => ExtProcessorTable::instruction_read_io(circuit_builder),
            WriteIo => ExtProcessorTable::instruction_write_io(circuit_builder),
        }
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

    fn running_evaluation_for_standard_input_updates_correctly(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let constant = |c: u32| circuit_builder.b_constant(c.into());
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

        let read_io_deselector =
            Self::instruction_deselector_current_row(circuit_builder, Instruction::ReadIo);
        let read_io_selector = curr_base_row(CI) - constant(Instruction::ReadIo.opcode());

        let running_evaluation_updates = next_ext_row(InputTableEvalArg)
            - challenge(StandardInputIndeterminate) * curr_ext_row(InputTableEvalArg)
            - next_base_row(ST0);
        let running_evaluation_remains =
            next_ext_row(InputTableEvalArg) - curr_ext_row(InputTableEvalArg);

        read_io_selector * running_evaluation_remains
            + read_io_deselector * running_evaluation_updates
    }

    fn log_derivative_for_instruction_lookup_updates_correctly(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let one = || circuit_builder.b_constant(1_u32.into());
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

    fn running_evaluation_for_standard_output_updates_correctly(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let constant = |c: u32| circuit_builder.b_constant(c.into());
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

        let write_io_deselector =
            Self::instruction_deselector_next_row(circuit_builder, Instruction::WriteIo);
        let write_io_selector = next_base_row(CI) - constant(Instruction::WriteIo.opcode());

        let running_evaluation_updates = next_ext_row(OutputTableEvalArg)
            - challenge(StandardOutputIndeterminate) * curr_ext_row(OutputTableEvalArg)
            - next_base_row(ST0);
        let running_evaluation_remains =
            next_ext_row(OutputTableEvalArg) - curr_ext_row(OutputTableEvalArg);

        write_io_selector * running_evaluation_remains
            + write_io_deselector * running_evaluation_updates
    }

    fn running_product_for_op_stack_table_updates_correctly(
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

        let compressed_row = challenge(OpStackClkWeight) * next_base_row(CLK)
            + challenge(OpStackIb1Weight) * next_base_row(IB1)
            + challenge(OpStackOspWeight) * next_base_row(OSP)
            + challenge(OpStackOsvWeight) * next_base_row(OSV);

        next_ext_row(OpStackTablePermArg)
            - curr_ext_row(OpStackTablePermArg) * (challenge(OpStackIndeterminate) - compressed_row)
    }

    fn running_product_for_ram_table_updates_correctly(
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

        let compressed_row = challenge(RamClkWeight) * next_base_row(CLK)
            + challenge(RamRampWeight) * next_base_row(RAMP)
            + challenge(RamRamvWeight) * next_base_row(RAMV)
            + challenge(RamPreviousInstructionWeight) * next_base_row(PreviousInstruction);

        next_ext_row(RamTablePermArg)
            - curr_ext_row(RamTablePermArg) * (challenge(RamIndeterminate) - compressed_row)
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
        let constant = |c: u32| circuit_builder.b_constant(c.into());
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
        let state = [
            next_base_row(ST0),
            next_base_row(ST1),
            next_base_row(ST2),
            next_base_row(ST3),
            next_base_row(ST4),
            next_base_row(ST5),
            next_base_row(ST6),
            next_base_row(ST7),
            next_base_row(ST8),
            next_base_row(ST9),
        ];
        let compressed_row = weights
            .into_iter()
            .zip_eq(state.into_iter())
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
        let constant = |c: u32| circuit_builder.b_constant(c.into());
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
        let state = [
            next_base_row(ST5),
            next_base_row(ST6),
            next_base_row(ST7),
            next_base_row(ST8),
            next_base_row(ST9),
        ];
        let compressed_row = weights
            .into_iter()
            .zip_eq(state.into_iter())
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
        let constant = |c: u32| circuit_builder.b_constant(c.into());
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

        let absorb_init_deselector =
            Self::instruction_deselector_current_row(circuit_builder, Instruction::AbsorbInit);
        let absorb_deselector =
            Self::instruction_deselector_current_row(circuit_builder, Instruction::Absorb);
        let squeeze_deselector =
            Self::instruction_deselector_current_row(circuit_builder, Instruction::Squeeze);

        let sponge_instruction_selector = (curr_base_row(CI)
            - constant(Instruction::AbsorbInit.opcode()))
            * (curr_base_row(CI) - constant(Instruction::Absorb.opcode()))
            * (curr_base_row(CI) - constant(Instruction::Squeeze.opcode()));

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
        let state_next = [
            next_base_row(ST0),
            next_base_row(ST1),
            next_base_row(ST2),
            next_base_row(ST3),
            next_base_row(ST4),
            next_base_row(ST5),
            next_base_row(ST6),
            next_base_row(ST7),
            next_base_row(ST8),
            next_base_row(ST9),
        ];
        let compressed_row_next = weights
            .into_iter()
            .zip_eq(state_next.into_iter())
            .map(|(weight, st_next)| weight * st_next)
            .sum();

        let running_evaluation_updates = next_ext_row(SpongeEvalArg)
            - challenge(SpongeIndeterminate) * curr_ext_row(SpongeEvalArg)
            - challenge(HashCIWeight) * curr_base_row(CI)
            - compressed_row_next;
        let running_evaluation_remains = next_ext_row(SpongeEvalArg) - curr_ext_row(SpongeEvalArg);

        sponge_instruction_selector * running_evaluation_remains
            + absorb_init_deselector * running_evaluation_updates.clone()
            + absorb_deselector * running_evaluation_updates.clone()
            + squeeze_deselector * running_evaluation_updates
    }

    fn log_derivative_with_u32_table_updates_correctly(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> ConstraintCircuitMonad<DualRowIndicator> {
        let constant = |c: u32| circuit_builder.b_constant(c.into());
        let one = || constant(1);
        let two_inverse = circuit_builder.b_constant(BFieldElement::new(2).inverse());
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
        let div_deselector =
            Self::instruction_deselector_current_row(circuit_builder, Instruction::Div);
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
        let div_factor_for_lt = challenge(U32Indeterminate)
            - challenge(U32LhsWeight) * next_base_row(ST0)
            - challenge(U32RhsWeight) * curr_base_row(ST1)
            - challenge(U32CiWeight) * constant(Instruction::Lt.opcode())
            - challenge(U32ResultWeight);
        let div_factor_for_range_check = challenge(U32Indeterminate)
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
        let div_summand = div_deselector
            * ((running_sum_next.clone() - running_sum.clone())
                * div_factor_for_lt.clone()
                * div_factor_for_range_check.clone()
                - div_factor_for_lt
                - div_factor_for_range_check);
        let pop_count_summand = pop_count_deselector * running_sum_absorbs_unop_factor;
        let no_update_summand = (one() - curr_base_row(IB2)) * (running_sum_next - running_sum);

        split_summand
            + lt_summand
            + and_summand
            + xor_summand
            + pow_summand
            + log_2_floor_summand
            + div_summand
            + pop_count_summand
            + no_update_summand
    }

    pub fn transition_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u64| circuit_builder.b_constant(c.into());
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
        let previous_instruction_is_copied_correctly = (next_base_row(PreviousInstruction)
            - curr_base_row(CI))
            * (constant(1) - next_base_row(IsPadding));

        let instruction_independent_constraints = vec![
            clk_increases_by_1,
            is_padding_is_0_or_does_not_change,
            previous_instruction_is_copied_correctly,
        ];

        // instruction-specific constraints
        let all_transition_constraints_by_instruction = ALL_INSTRUCTIONS.map(|instruction| {
            Self::get_transition_constraints_for_instruction(circuit_builder, instruction)
        });
        let all_instructions_and_their_transition_constraints = ALL_INSTRUCTIONS
            .into_iter()
            .zip_eq(all_transition_constraints_by_instruction.into_iter())
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
            Self::running_evaluation_for_standard_input_updates_correctly(circuit_builder),
            Self::log_derivative_for_instruction_lookup_updates_correctly(circuit_builder),
            Self::running_evaluation_for_standard_output_updates_correctly(circuit_builder),
            Self::running_product_for_op_stack_table_updates_correctly(circuit_builder),
            Self::running_product_for_ram_table_updates_correctly(circuit_builder),
            Self::running_product_for_jump_stack_table_updates_correctly(circuit_builder),
            Self::running_evaluation_hash_input_updates_correctly(circuit_builder),
            Self::running_evaluation_hash_digest_updates_correctly(circuit_builder),
            Self::running_evaluation_sponge_updates_correctly(circuit_builder),
            Self::log_derivative_with_u32_table_updates_correctly(circuit_builder),
        ];

        vec![
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

pub struct ProcessorTraceRow<'a> {
    pub row: ArrayView1<'a, BFieldElement>,
}

impl<'a> Display for ProcessorTraceRow<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn row(f: &mut std::fmt::Formatter<'_>, s: String) -> std::fmt::Result {
            writeln!(f, "│ {s: <103} │")
        }

        fn row_blank(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            row(f, "".into())
        }

        let instruction = self.row[CI.base_table_index()].value().try_into().unwrap();
        let instruction_with_arg = match instruction {
            Push(_) => Push(self.row[NIA.base_table_index()]),
            Call(_) => Call(self.row[NIA.base_table_index()]),
            Dup(_) => Dup((self.row[NIA.base_table_index()].value() as u32)
                .try_into()
                .unwrap()),
            Swap(_) => Swap(
                (self.row[NIA.base_table_index()].value() as u32)
                    .try_into()
                    .unwrap(),
            ),
            _ => instruction,
        };

        writeln!(f, " ╭───────────────────────────╮")?;
        writeln!(f, " │ {: <25} │", format!("{instruction_with_arg}"))?;
        writeln!(
            f,
            "╭┴───────────────────────────┴────────────────────────────────────\
            ────────────────────┬───────────────────╮"
        )?;

        let width = 20;
        row(
            f,
            format!(
                "ip:   {:>width$} ╷ ci:   {:>width$} ╷ nia: {:>width$} │ {:>17}",
                self.row[IP.base_table_index()].value(),
                self.row[CI.base_table_index()].value(),
                self.row[NIA.base_table_index()].value(),
                self.row[CLK.base_table_index()].value(),
            ),
        )?;

        writeln!(
            f,
            "│ jsp:  {:>width$} │ jso:  {:>width$} │ jsd: {:>width$} ╰───────────────────┤",
            self.row[JSP.base_table_index()].value(),
            self.row[JSO.base_table_index()].value(),
            self.row[JSD.base_table_index()].value(),
        )?;
        row(
            f,
            format!(
                "ramp: {:>width$} │ ramv: {:>width$} │",
                self.row[RAMP.base_table_index()].value(),
                self.row[RAMV.base_table_index()].value(),
            ),
        )?;
        row(
            f,
            format!(
                "osp:  {:>width$} │ osv:  {:>width$} ╵",
                self.row[OSP.base_table_index()].value(),
                self.row[OSV.base_table_index()].value(),
            ),
        )?;

        row_blank(f)?;

        row(
            f,
            format!(
                "st0-3:    [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[ST0.base_table_index()].value(),
                self.row[ST1.base_table_index()].value(),
                self.row[ST2.base_table_index()].value(),
                self.row[ST3.base_table_index()].value(),
            ),
        )?;
        row(
            f,
            format!(
                "st4-7:    [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[ST4.base_table_index()].value(),
                self.row[ST5.base_table_index()].value(),
                self.row[ST6.base_table_index()].value(),
                self.row[ST7.base_table_index()].value(),
            ),
        )?;
        row(
            f,
            format!(
                "st8-11:   [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[ST8.base_table_index()].value(),
                self.row[ST9.base_table_index()].value(),
                self.row[ST10.base_table_index()].value(),
                self.row[ST11.base_table_index()].value(),
            ),
        )?;
        row(
            f,
            format!(
                "st12-15:  [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[ST12.base_table_index()].value(),
                self.row[ST13.base_table_index()].value(),
                self.row[ST14.base_table_index()].value(),
                self.row[ST15.base_table_index()].value(),
            ),
        )?;

        row_blank(f)?;

        let w = 2;
        row(
            f,
            format!(
                "hv0-3:    [ {:>w$} | {:>w$} | {:>w$} | {:>w$} ]",
                self.row[HV0.base_table_index()].value(),
                self.row[HV1.base_table_index()].value(),
                self.row[HV2.base_table_index()].value(),
                self.row[HV3.base_table_index()].value(),
            ),
        )?;
        row(
            f,
            format!(
                "ib0-7:    \
                [ {:>w$} | {:>w$} | {:>w$} | {:>w$} | {:>w$} | {:>w$} | {:>w$} | {:>w$} ]",
                self.row[IB0.base_table_index()].value(),
                self.row[IB1.base_table_index()].value(),
                self.row[IB2.base_table_index()].value(),
                self.row[IB3.base_table_index()].value(),
                self.row[IB4.base_table_index()].value(),
                self.row[IB5.base_table_index()].value(),
                self.row[IB6.base_table_index()].value(),
                self.row[IB7.base_table_index()].value(),
            ),
        )?;
        write!(
            f,
            "╰─────────────────────────────────────────────────────────────────\
            ────────────────────────────────────────╯"
        )
    }
}

pub struct ExtProcessorTraceRow<'a> {
    pub row: ArrayView1<'a, XFieldElement>,
}

impl<'a> Display for ExtProcessorTraceRow<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let row = |form: &mut std::fmt::Formatter<'_>,
                   desc: &str,
                   col: ProcessorExtTableColumn|
         -> std::fmt::Result {
            // without the extra `format!()`, alignment in `writeln!()` fails
            let formatted_col_elem = format!("{}", self.row[col.ext_table_index()]);
            writeln!(form, "     │ {desc: <18}  {formatted_col_elem:>73} │")
        };
        writeln!(
            f,
            "     ╭───────────────────────────────────────────────────────\
            ────────────────────────────────────────╮"
        )?;
        row(f, "input_table_ea", InputTableEvalArg)?;
        row(f, "output_table_ea", OutputTableEvalArg)?;
        row(f, "instr_lookup_ld", InstructionLookupClientLogDerivative)?;
        row(f, "opstack_table_pa", OpStackTablePermArg)?;
        row(f, "ram_table_pa", RamTablePermArg)?;
        row(f, "jumpstack_table_pa", JumpStackTablePermArg)?;
        row(f, "hash_input_ea", HashInputEvalArg)?;
        row(f, "hash_digest_ea", HashDigestEvalArg)?;
        row(f, "sponge_absorb_ea", SpongeEvalArg)?;
        row(f, "u32_table_pa", U32LookupClientLogDerivative)?;
        write!(
            f,
            "     ╰───────────────────────────────────────────────────────\
            ────────────────────────────────────────╯"
        )
    }
}

#[cfg(test)]
mod constraint_polynomial_tests {
    use ndarray::Array2;

    use triton_opcodes::ord_n::Ord16;
    use triton_opcodes::program::Program;

    use crate::error::InstructionError;
    use crate::error::InstructionError::DivisionByZero;
    use crate::shared_tests::SourceCodeAndInput;
    use crate::stark::triton_stark_tests::parse_simulate_pad;
    use crate::table::master_table::MasterTable;
    use crate::table::master_table::NUM_BASE_COLUMNS;
    use crate::table::master_table::NUM_EXT_COLUMNS;
    use crate::table::processor_table::ProcessorTraceRow;
    use crate::vm::simulate;

    use super::*;

    #[test]
    /// helps identifying whether the printing causes an infinite loop
    fn print_simple_processor_table_row_test() {
        let code = "push 2 push -1 add assert halt";
        let program = Program::from_code(code).unwrap();
        let (aet, _, _) = simulate(&program, vec![], vec![]);
        println!();
        for row in aet.processor_trace.rows() {
            println!("{}", ProcessorTraceRow { row });
        }
    }

    fn get_test_row_from_source_code(source_code: &str, row_num: usize) -> Array2<BFieldElement> {
        let (_, _, unpadded_master_base_table, _) = parse_simulate_pad(source_code, vec![], vec![]);
        unpadded_master_base_table
            .trace_table()
            .slice(s![row_num..=row_num + 1, ..])
            .to_owned()
    }

    fn test_constraints_for_rows_with_debug_info(
        instruction: Instruction,
        master_base_tables: &[Array2<BFieldElement>],
        debug_cols_curr_row: &[ProcessorBaseTableColumn],
        debug_cols_next_row: &[ProcessorBaseTableColumn],
    ) {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let challenges = Challenges::placeholder(&[], &[]);
        let fake_ext_table = Array2::zeros([2, NUM_EXT_COLUMNS]);
        for (case_idx, test_rows) in master_base_tables.iter().enumerate() {
            let curr_row = test_rows.slice(s![0, ..]);
            let next_row = test_rows.slice(s![1, ..]);

            // Print debug information
            println!(
                "Testing all constraints of {instruction} for test row with index {case_idx}…"
            );
            for &c in debug_cols_curr_row {
                print!("{} = {}, ", c, curr_row[c.master_base_table_index()]);
            }
            for &c in debug_cols_next_row {
                print!("{}' = {}, ", c, next_row[c.master_base_table_index()]);
            }
            println!();

            assert_eq!(
                instruction.opcode_b(),
                curr_row[CI.master_base_table_index()],
                "The test is trying to check the wrong transition constraint polynomials."
            );
            for (constraint_idx, constraint_circuit) in
                ExtProcessorTable::get_transition_constraints_for_instruction(
                    &circuit_builder,
                    instruction,
                )
                .into_iter()
                .enumerate()
            {
                let evaluation_result = constraint_circuit.consume().evaluate(
                    test_rows.view(),
                    fake_ext_table.view(),
                    &challenges,
                );
                assert_eq!(
                    XFieldElement::zero(),
                    evaluation_result,
                    "For case {case_idx}, transition constraint polynomial with \
                    index {constraint_idx} must evaluate to zero. Got {evaluation_result} instead.",
                );
            }
        }
    }

    #[test]
    fn transition_constraints_for_instruction_pop_test() {
        let test_rows = [get_test_row_from_source_code("push 1 pop halt", 1)];
        test_constraints_for_rows_with_debug_info(
            Pop,
            &test_rows,
            &[ST0, ST1, ST2],
            &[ST0, ST1, ST2],
        );
    }

    #[test]
    fn transition_constraints_for_instruction_push_test() {
        let test_rows = [get_test_row_from_source_code("push 1 halt", 0)];
        test_constraints_for_rows_with_debug_info(
            Push(BFieldElement::one()),
            &test_rows,
            &[ST0, ST1, ST2],
            &[ST0, ST1, ST2],
        );
    }

    #[test]
    fn transition_constraints_for_instruction_dup_test() {
        let test_rows = [get_test_row_from_source_code("push 1 dup 0 halt", 1)];
        test_constraints_for_rows_with_debug_info(
            Dup(Ord16::ST0),
            &test_rows,
            &[ST0, ST1, ST2],
            &[ST0, ST1, ST2],
        );
    }

    #[test]
    fn transition_constraints_for_instruction_swap_test() {
        let test_rows = [get_test_row_from_source_code(
            "push 1 push 2 swap 1 halt",
            2,
        )];
        test_constraints_for_rows_with_debug_info(
            Swap(Ord16::ST0),
            &test_rows,
            &[ST0, ST1, ST2],
            &[ST0, ST1, ST2],
        );
    }

    #[test]
    fn transition_constraints_for_instruction_skiz_test() {
        // Case 0: ST0 is non-zero
        // Case 1: ST0 is zero, nia is instruction of size 1
        // Case 2: ST0 is zero, nia is instruction of size 2
        let test_rows = [
            get_test_row_from_source_code("push 1 skiz halt", 1),
            get_test_row_from_source_code("push 0 skiz assert halt", 1),
            get_test_row_from_source_code("push 0 skiz push 1 halt", 1),
        ];
        test_constraints_for_rows_with_debug_info(Skiz, &test_rows, &[IP, ST0, HV0, HV1], &[IP]);
    }

    #[test]
    fn transition_constraints_for_instruction_call_test() {
        let test_rows = [get_test_row_from_source_code("call label label: halt", 0)];
        test_constraints_for_rows_with_debug_info(
            Call(Default::default()),
            &test_rows,
            &[IP, CI, NIA, JSP, JSO, JSD],
            &[IP, CI, NIA, JSP, JSO, JSD],
        );
    }

    #[test]
    fn transition_constraints_for_instruction_return_test() {
        let test_rows = [get_test_row_from_source_code(
            "call label halt label: return",
            1,
        )];
        test_constraints_for_rows_with_debug_info(
            Return,
            &test_rows,
            &[IP, JSP, JSO, JSD],
            &[IP, JSP, JSO, JSD],
        );
    }

    #[test]
    fn transition_constraints_for_instruction_recurse_test() {
        let test_rows = [get_test_row_from_source_code(
            "push 2 call label halt label: push -1 add dup 0 skiz recurse return ",
            6,
        )];
        test_constraints_for_rows_with_debug_info(
            Recurse,
            &test_rows,
            &[IP, JSP, JSO, JSD],
            &[IP, JSP, JSO, JSD],
        );
    }

    #[test]
    fn transition_constraints_for_instruction_read_mem_test() {
        let test_rows = [get_test_row_from_source_code(
            "push 5 push 3 write_mem read_mem halt",
            3,
        )];
        test_constraints_for_rows_with_debug_info(
            ReadMem,
            &test_rows,
            &[ST0, ST1, RAMP, RAMV],
            &[ST0, ST1, RAMP, RAMV],
        );
    }

    #[test]
    fn transition_constraints_for_instruction_write_mem_test() {
        let test_rows = [get_test_row_from_source_code(
            "push 5 push 3 write_mem read_mem halt",
            2,
        )];
        test_constraints_for_rows_with_debug_info(
            WriteMem,
            &test_rows,
            &[ST0, ST1, RAMP, RAMV],
            &[ST0, ST1, RAMP, RAMV],
        );
    }

    #[test]
    fn transition_constraints_for_instruction_eq_test() {
        let test_rows = [
            get_test_row_from_source_code("push 3 push 3 eq assert halt", 2),
            get_test_row_from_source_code("push 3 push 2 eq push 0 eq assert halt", 2),
        ];
        test_constraints_for_rows_with_debug_info(Eq, &test_rows, &[ST0, ST1, HV0], &[ST0]);
    }

    #[test]
    fn transition_constraints_for_instruction_split_test() {
        let test_rows = [
            get_test_row_from_source_code("push -1 split halt", 1),
            get_test_row_from_source_code("push  0 split halt", 1),
            get_test_row_from_source_code("push  1 split halt", 1),
            get_test_row_from_source_code("push  2 split halt", 1),
            get_test_row_from_source_code("push  3 split halt", 1),
            // test pushing push 2^32 +- 1
            get_test_row_from_source_code("push 4294967295 split halt", 1),
            get_test_row_from_source_code("push 4294967296 split halt", 1),
            get_test_row_from_source_code("push 4294967297 split halt", 1),
        ];
        test_constraints_for_rows_with_debug_info(
            Split,
            &test_rows,
            &[ST0, ST1, HV0],
            &[ST0, ST1, HV0],
        );
    }

    #[test]
    fn transition_constraints_for_instruction_lt_test() {
        let test_rows = [
            get_test_row_from_source_code("push 3 push 3 lt push 0 eq assert halt", 2),
            get_test_row_from_source_code("push 3 push 2 lt push 1 eq assert halt", 2),
            get_test_row_from_source_code("push 2 push 3 lt push 0 eq assert halt", 2),
            get_test_row_from_source_code("push 512 push 513 lt push 0 eq assert halt", 2),
        ];
        test_constraints_for_rows_with_debug_info(Lt, &test_rows, &[ST0, ST1], &[ST0]);
    }

    #[test]
    fn transition_constraints_for_instruction_and_test() {
        let test_rows = [get_test_row_from_source_code(
            "push 5 push 12 and push 4 eq assert halt",
            2,
        )];
        test_constraints_for_rows_with_debug_info(And, &test_rows, &[ST0, ST1], &[ST0]);
    }

    #[test]
    fn transition_constraints_for_instruction_xor_test() {
        let test_rows = [get_test_row_from_source_code(
            "push 5 push 12 xor push 9 eq assert halt",
            2,
        )];
        test_constraints_for_rows_with_debug_info(Xor, &test_rows, &[ST0, ST1], &[ST0]);
    }

    #[test]
    fn transition_constraints_for_instruction_log2floor_test() {
        let test_rows = [
            get_test_row_from_source_code("push  1 log_2_floor push  0 eq assert halt", 1),
            get_test_row_from_source_code("push  2 log_2_floor push  1 eq assert halt", 1),
            get_test_row_from_source_code("push  3 log_2_floor push  1 eq assert halt", 1),
            get_test_row_from_source_code("push  4 log_2_floor push  2 eq assert halt", 1),
            get_test_row_from_source_code("push  5 log_2_floor push  2 eq assert halt", 1),
            get_test_row_from_source_code("push  6 log_2_floor push  2 eq assert halt", 1),
            get_test_row_from_source_code("push  7 log_2_floor push  2 eq assert halt", 1),
            get_test_row_from_source_code("push  8 log_2_floor push  3 eq assert halt", 1),
            get_test_row_from_source_code("push  9 log_2_floor push  3 eq assert halt", 1),
            get_test_row_from_source_code("push 10 log_2_floor push  3 eq assert halt", 1),
            get_test_row_from_source_code("push 11 log_2_floor push  3 eq assert halt", 1),
            get_test_row_from_source_code("push 12 log_2_floor push  3 eq assert halt", 1),
            get_test_row_from_source_code("push 13 log_2_floor push  3 eq assert halt", 1),
            get_test_row_from_source_code("push 14 log_2_floor push  3 eq assert halt", 1),
            get_test_row_from_source_code("push 15 log_2_floor push  3 eq assert halt", 1),
            get_test_row_from_source_code("push 16 log_2_floor push  4 eq assert halt", 1),
            get_test_row_from_source_code("push 17 log_2_floor push  4 eq assert halt", 1),
        ];
        test_constraints_for_rows_with_debug_info(Log2Floor, &test_rows, &[ST0, ST1], &[ST0]);
    }

    #[test]
    fn transition_constraints_for_instruction_pow_test() {
        let test_rows = [
            get_test_row_from_source_code("push 0 push  0 pow push   1 eq assert halt", 2),
            get_test_row_from_source_code("push 1 push  0 pow push   0 eq assert halt", 2),
            get_test_row_from_source_code("push 2 push  0 pow push   0 eq assert halt", 2),
            get_test_row_from_source_code("push 0 push  1 pow push   1 eq assert halt", 2),
            get_test_row_from_source_code("push 1 push  1 pow push   1 eq assert halt", 2),
            get_test_row_from_source_code("push 2 push  1 pow push   1 eq assert halt", 2),
            get_test_row_from_source_code("push 0 push  2 pow push   1 eq assert halt", 2),
            get_test_row_from_source_code("push 1 push  2 pow push   2 eq assert halt", 2),
            get_test_row_from_source_code("push 2 push  2 pow push   4 eq assert halt", 2),
            get_test_row_from_source_code("push 3 push  2 pow push   8 eq assert halt", 2),
            get_test_row_from_source_code("push 4 push  2 pow push  16 eq assert halt", 2),
            get_test_row_from_source_code("push 5 push  2 pow push  32 eq assert halt", 2),
            get_test_row_from_source_code("push 0 push  3 pow push   1 eq assert halt", 2),
            get_test_row_from_source_code("push 1 push  3 pow push   3 eq assert halt", 2),
            get_test_row_from_source_code("push 2 push  3 pow push   9 eq assert halt", 2),
            get_test_row_from_source_code("push 3 push  3 pow push  27 eq assert halt", 2),
            get_test_row_from_source_code("push 4 push  3 pow push  81 eq assert halt", 2),
            get_test_row_from_source_code("push 0 push 17 pow push   1 eq assert halt", 2),
            get_test_row_from_source_code("push 1 push 17 pow push  17 eq assert halt", 2),
            get_test_row_from_source_code("push 2 push 17 pow push 289 eq assert halt", 2),
        ];
        test_constraints_for_rows_with_debug_info(Pow, &test_rows, &[ST0, ST1], &[ST0]);
    }

    #[test]
    fn transition_constraints_for_instruction_div_test() {
        let test_rows = [
            get_test_row_from_source_code(
                "push 2 push 3 div push 1 eq assert push 1 eq assert halt",
                2,
            ),
            get_test_row_from_source_code(
                "push 3 push 7 div push 1 eq assert push 2 eq assert halt",
                2,
            ),
            get_test_row_from_source_code(
                "push 4 push 7 div push 3 eq assert push 1 eq assert halt",
                2,
            ),
        ];
        test_constraints_for_rows_with_debug_info(Div, &test_rows, &[ST0, ST1], &[ST0, ST1]);
    }

    #[test]
    fn division_by_zero_is_impossible_test() {
        let (_aet, _out, err) = SourceCodeAndInput::without_input("div").simulate();
        let Some(err) = err else {
            panic!("Dividing by 0 must fail.");
        };
        let Ok(err) = err.downcast::<InstructionError>() else {
            panic!("Dividing by 0 must fail with InstructionError.");
        };
        let DivisionByZero = err else {
            panic!("Dividing by 0 must fail with DivisionByZero.");
        };
    }

    #[test]
    fn transition_constraints_for_instruction_xxadd_test() {
        let test_rows = [
            get_test_row_from_source_code(
                "push 5 push 6 push 7 push 8 push 9 push 10 xxadd halt",
                6,
            ),
            get_test_row_from_source_code(
                "push 2 push 3 push 4 push -2 push -3 push -4 xxadd halt",
                6,
            ),
        ];
        test_constraints_for_rows_with_debug_info(
            XxAdd,
            &test_rows,
            &[ST0, ST1, ST2, ST3, ST4, ST5],
            &[ST0, ST1, ST2, ST3, ST4, ST5],
        );
    }

    #[test]
    fn transition_constraints_for_instruction_xxmul_test() {
        let test_rows = [
            get_test_row_from_source_code(
                "push 5 push 6 push 7 push 8 push 9 push 10 xxmul halt",
                6,
            ),
            get_test_row_from_source_code(
                "push 2 push 3 push 4 push -2 push -3 push -4 xxmul halt",
                6,
            ),
        ];
        test_constraints_for_rows_with_debug_info(
            XxMul,
            &test_rows,
            &[ST0, ST1, ST2, ST3, ST4, ST5],
            &[ST0, ST1, ST2, ST3, ST4, ST5],
        );
    }

    #[test]
    fn transition_constraints_for_instruction_xinvert_test() {
        let test_rows = [
            get_test_row_from_source_code("push 5 push 6 push 7 xinvert halt", 3),
            get_test_row_from_source_code("push -2 push -3 push -4 xinvert halt", 3),
        ];
        test_constraints_for_rows_with_debug_info(
            XInvert,
            &test_rows,
            &[ST0, ST1, ST2],
            &[ST0, ST1, ST2],
        );
    }

    #[test]
    fn transition_constraints_for_instruction_xbmul_test() {
        let test_rows = [
            get_test_row_from_source_code("push 5 push 6 push 7 push 2 xbmul halt", 4),
            get_test_row_from_source_code("push 2 push 3 push 4 push -2 xbmul halt", 4),
        ];
        test_constraints_for_rows_with_debug_info(
            XbMul,
            &test_rows,
            &[ST0, ST1, ST2, ST3, OSP, HV3],
            &[ST0, ST1, ST2, ST3, OSP, HV3],
        );
    }

    #[test]
    fn instruction_deselector_gives_0_for_all_other_instructions_test() {
        let circuit_builder = ConstraintCircuitBuilder::new();

        let mut master_base_table = Array2::zeros([2, NUM_BASE_COLUMNS]);
        let master_ext_table = Array2::zeros([2, NUM_EXT_COLUMNS]);

        // We need dummy challenges to evaluate.
        let dummy_challenges = Challenges::placeholder(&[], &[]);
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
                curr_row[IB0.master_base_table_index()] = other_instruction.ib(Ord8::IB0);
                curr_row[IB1.master_base_table_index()] = other_instruction.ib(Ord8::IB1);
                curr_row[IB2.master_base_table_index()] = other_instruction.ib(Ord8::IB2);
                curr_row[IB3.master_base_table_index()] = other_instruction.ib(Ord8::IB3);
                curr_row[IB4.master_base_table_index()] = other_instruction.ib(Ord8::IB4);
                curr_row[IB5.master_base_table_index()] = other_instruction.ib(Ord8::IB5);
                curr_row[IB6.master_base_table_index()] = other_instruction.ib(Ord8::IB6);
                curr_row[IB7.master_base_table_index()] = other_instruction.ib(Ord8::IB7);
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
            curr_row[IB0.master_base_table_index()] = instruction.ib(Ord8::IB0);
            curr_row[IB1.master_base_table_index()] = instruction.ib(Ord8::IB1);
            curr_row[IB2.master_base_table_index()] = instruction.ib(Ord8::IB2);
            curr_row[IB3.master_base_table_index()] = instruction.ib(Ord8::IB3);
            curr_row[IB4.master_base_table_index()] = instruction.ib(Ord8::IB4);
            curr_row[IB5.master_base_table_index()] = instruction.ib(Ord8::IB5);
            curr_row[IB6.master_base_table_index()] = instruction.ib(Ord8::IB6);
            curr_row[IB7.master_base_table_index()] = instruction.ib(Ord8::IB7);
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
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub fn constraints_evaluate_to_zero(
        master_base_trace_table: ArrayView2<BFieldElement>,
        master_ext_trace_table: ArrayView2<XFieldElement>,
        challenges: &Challenges,
    ) -> bool {
        let zero = XFieldElement::zero();
        assert_eq!(
            master_base_trace_table.nrows(),
            master_ext_trace_table.nrows()
        );

        let circuit_builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in ExtProcessorTable::initial_constraints(&circuit_builder)
            .into_iter()
            .map(|constraint_monad| constraint_monad.consume())
            .enumerate()
        {
            let evaluated_constraint = constraint.evaluate(
                master_base_trace_table.slice(s![..1, ..]),
                master_ext_trace_table.slice(s![..1, ..]),
                challenges,
            );
            assert_eq!(
                zero, evaluated_constraint,
                "Initial constraint {constraint_idx} failed."
            );
        }

        let circuit_builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in
            ExtProcessorTable::consistency_constraints(&circuit_builder)
                .into_iter()
                .map(|constraint_monad| constraint_monad.consume())
                .enumerate()
        {
            for row_idx in 0..master_base_trace_table.nrows() {
                let evaluated_constraint = constraint.evaluate(
                    master_base_trace_table.slice(s![row_idx..row_idx + 1, ..]),
                    master_ext_trace_table.slice(s![row_idx..row_idx + 1, ..]),
                    challenges,
                );
                assert_eq!(
                    zero, evaluated_constraint,
                    "Consistency constraint {constraint_idx} failed on row {row_idx}."
                );
            }
        }

        let circuit_builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in
            ExtProcessorTable::transition_constraints(&circuit_builder)
                .into_iter()
                .map(|constraint_monad| constraint_monad.consume())
                .enumerate()
        {
            for row_idx in 0..master_base_trace_table.nrows() - 1 {
                let evaluated_constraint = constraint.evaluate(
                    master_base_trace_table.slice(s![row_idx..row_idx + 2, ..]),
                    master_ext_trace_table.slice(s![row_idx..row_idx + 2, ..]),
                    challenges,
                );
                assert_eq!(
                    zero, evaluated_constraint,
                    "Transition constraint {constraint_idx} failed on row {row_idx}."
                );
            }
        }

        let circuit_builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in
            ExtProcessorTable::terminal_constraints(&circuit_builder)
                .into_iter()
                .map(|constraint_monad| constraint_monad.consume())
                .enumerate()
        {
            let evaluated_constraint = constraint.evaluate(
                master_base_trace_table.slice(s![-1.., ..]),
                master_ext_trace_table.slice(s![-1.., ..]),
                challenges,
            );
            assert_eq!(
                zero, evaluated_constraint,
                "Terminal constraint {constraint_idx} failed."
            );
        }

        true
    }
}
