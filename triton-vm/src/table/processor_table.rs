use std::cmp::max;
use std::cmp::Eq;
use std::collections::HashMap;
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
use strum_macros::Display;
use strum_macros::EnumCount as EnumCountMacro;
use strum_macros::EnumIter;
use triton_opcodes::instruction::all_instructions_without_args;
use triton_opcodes::instruction::{AnInstruction::*, Instruction};
use triton_opcodes::ord_n::Ord7;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use ProcessorTableChallengeId::*;

use crate::table::challenges::TableChallenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::ConstraintCircuitMonad;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::InputIndicator;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::EvalArg;
use crate::table::cross_table_argument::PermArg;
use crate::table::master_table::NUM_BASE_COLUMNS;
use crate::table::master_table::NUM_EXT_COLUMNS;
use crate::table::table_column::BaseTableColumn;
use crate::table::table_column::ExtTableColumn;
use crate::table::table_column::MasterBaseTableColumn;
use crate::table::table_column::MasterExtTableColumn;
use crate::table::table_column::ProcessorBaseTableColumn;
use crate::table::table_column::ProcessorBaseTableColumn::*;
use crate::table::table_column::ProcessorExtTableColumn;
use crate::table::table_column::ProcessorExtTableColumn::*;
use crate::vm::AlgebraicExecutionTrace;

pub const PROCESSOR_TABLE_NUM_PERMUTATION_ARGUMENTS: usize = 5;
pub const PROCESSOR_TABLE_NUM_EVALUATION_ARGUMENTS: usize = 5;
pub const PROCESSOR_TABLE_NUM_EXTENSION_CHALLENGES: usize = ProcessorTableChallengeId::COUNT;

pub const BASE_WIDTH: usize = ProcessorBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = ProcessorExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

#[derive(Debug, Clone)]
pub struct ProcessorTable {}

impl ProcessorTable {
    pub fn fill_trace(
        processor_table: &mut ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
        mut all_clk_jump_diffs: Vec<BFieldElement>,
    ) {
        // fill the processor table from the AET
        let mut processor_table_to_fill =
            processor_table.slice_mut(s![0..aet.processor_matrix.nrows(), ..]);
        aet.processor_matrix
            .clone()
            .move_into(&mut processor_table_to_fill);

        let zero = BFieldElement::zero();
        all_clk_jump_diffs.sort_by_key(|bfe| std::cmp::Reverse(bfe.value()));

        let mut previous_row: Option<Array1<_>> = None;
        for mut row in processor_table_to_fill.rows_mut() {
            // add all clock jump differences and their inverses
            let clk_jump_difference = all_clk_jump_diffs.pop().unwrap_or(zero);
            let clk_jump_difference_inv = clk_jump_difference.inverse_or_zero();
            row[ClockJumpDifference.base_table_index()] = clk_jump_difference;
            row[ClockJumpDifferenceInverse.base_table_index()] = clk_jump_difference_inv;

            // add inverses of unique clock jump difference differences
            if let Some(prow) = previous_row {
                let previous_clk_jump_difference = prow[ClockJumpDifference.base_table_index()];
                if previous_clk_jump_difference != clk_jump_difference {
                    let clk_diff_diff: BFieldElement =
                        clk_jump_difference - previous_clk_jump_difference;
                    let clk_diff_diff_inv = clk_diff_diff.inverse();
                    row[UniqueClockJumpDiffDiffInverse.base_table_index()] = clk_diff_diff_inv;
                }
            }

            previous_row = Some(row.to_owned());
        }

        assert!(
            all_clk_jump_diffs.is_empty(),
            "Processor Table must record all clock jump differences, but didn't have enough space \
            for the remaining {}.",
            all_clk_jump_diffs.len()
        );
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
    }

    pub fn extend(
        base_table: ArrayView2<BFieldElement>,
        mut ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &ProcessorTableChallenges,
    ) {
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());
        let mut unique_clock_jump_differences = vec![];

        let mut input_table_running_evaluation = EvalArg::default_initial();
        let mut output_table_running_evaluation = EvalArg::default_initial();
        let mut instruction_table_running_product = PermArg::default_initial();
        let mut op_stack_table_running_product = PermArg::default_initial();
        let mut ram_table_running_product = PermArg::default_initial();
        let mut jump_stack_running_product = PermArg::default_initial();
        let mut to_hash_table_running_evaluation = EvalArg::default_initial();
        let mut from_hash_table_running_evaluation = EvalArg::default_initial();
        let mut unique_clock_jump_differences_running_evaluation = EvalArg::default_initial();
        let mut all_clock_jump_differences_running_product =
            PermArg::default_initial() * PermArg::default_initial() * PermArg::default_initial();

        let mut previous_row: Option<ArrayView1<BFieldElement>> = None;
        for row_idx in 0..base_table.nrows() {
            let current_row = base_table.row(row_idx);

            // Input table
            if let Some(prev_row) = previous_row {
                if prev_row[CI.base_table_index()] == Instruction::ReadIo.opcode_b() {
                    let input_symbol = current_row[ST0.base_table_index()];
                    input_table_running_evaluation = input_table_running_evaluation
                        * challenges.standard_input_eval_indeterminate
                        + input_symbol;
                }
            }

            // Output table
            if current_row[CI.base_table_index()] == Instruction::WriteIo.opcode_b() {
                let output_symbol = current_row[ST0.base_table_index()];
                output_table_running_evaluation = output_table_running_evaluation
                    * challenges.standard_output_eval_indeterminate
                    + output_symbol;
            }

            // Instruction table
            if current_row[IsPadding.base_table_index()].is_zero() {
                let ip = current_row[IP.base_table_index()];
                let ci = current_row[CI.base_table_index()];
                let nia = current_row[NIA.base_table_index()];
                let compressed_row_for_instruction_table_permutation_argument = ip
                    * challenges.instruction_table_ip_weight
                    + ci * challenges.instruction_table_ci_processor_weight
                    + nia * challenges.instruction_table_nia_weight;
                instruction_table_running_product *= challenges.instruction_perm_indeterminate
                    - compressed_row_for_instruction_table_permutation_argument;
            }

            // OpStack table
            let clk = current_row[CLK.base_table_index()];
            let ib1 = current_row[IB1.base_table_index()];
            let osp = current_row[OSP.base_table_index()];
            let osv = current_row[OSV.base_table_index()];
            let compressed_row_for_op_stack_table_permutation_argument = clk
                * challenges.op_stack_table_clk_weight
                + ib1 * challenges.op_stack_table_ib1_weight
                + osp * challenges.op_stack_table_osp_weight
                + osv * challenges.op_stack_table_osv_weight;
            op_stack_table_running_product *= challenges.op_stack_perm_indeterminate
                - compressed_row_for_op_stack_table_permutation_argument;

            // RAM Table
            let ramv = current_row[RAMV.base_table_index()];
            let ramp = current_row[RAMP.base_table_index()];
            let previous_instruction = current_row[PreviousInstruction.base_table_index()];
            let compressed_row_for_ram_table_permutation_argument = clk
                * challenges.ram_table_clk_weight
                + ramv * challenges.ram_table_ramv_weight
                + ramp * challenges.ram_table_ramp_weight
                + previous_instruction * challenges.ram_table_previous_instruction_weight;
            ram_table_running_product *= challenges.ram_perm_indeterminate
                - compressed_row_for_ram_table_permutation_argument;

            // JumpStack Table
            let ci = current_row[CI.base_table_index()];
            let jsp = current_row[JSP.base_table_index()];
            let jso = current_row[JSO.base_table_index()];
            let jsd = current_row[JSD.base_table_index()];
            let compressed_row_for_jump_stack_table = clk * challenges.jump_stack_table_clk_weight
                + ci * challenges.jump_stack_table_ci_weight
                + jsp * challenges.jump_stack_table_jsp_weight
                + jso * challenges.jump_stack_table_jso_weight
                + jsd * challenges.jump_stack_table_jsd_weight;
            jump_stack_running_product *=
                challenges.jump_stack_perm_indeterminate - compressed_row_for_jump_stack_table;

            if current_row[CI.base_table_index()] == Instruction::Hash.opcode_b() {
                let st_0_through_9 = [
                    current_row[ST0.base_table_index()],
                    current_row[ST1.base_table_index()],
                    current_row[ST2.base_table_index()],
                    current_row[ST3.base_table_index()],
                    current_row[ST4.base_table_index()],
                    current_row[ST5.base_table_index()],
                    current_row[ST6.base_table_index()],
                    current_row[ST7.base_table_index()],
                    current_row[ST8.base_table_index()],
                    current_row[ST9.base_table_index()],
                ];
                let hash_table_stack_input_challenges = [
                    challenges.hash_table_stack_input_weight0,
                    challenges.hash_table_stack_input_weight1,
                    challenges.hash_table_stack_input_weight2,
                    challenges.hash_table_stack_input_weight3,
                    challenges.hash_table_stack_input_weight4,
                    challenges.hash_table_stack_input_weight5,
                    challenges.hash_table_stack_input_weight6,
                    challenges.hash_table_stack_input_weight7,
                    challenges.hash_table_stack_input_weight8,
                    challenges.hash_table_stack_input_weight9,
                ];
                let compressed_row_for_hash_input: XFieldElement = st_0_through_9
                    .into_iter()
                    .zip_eq(hash_table_stack_input_challenges.into_iter())
                    .map(|(st, weight)| weight * st)
                    .sum();
                to_hash_table_running_evaluation = to_hash_table_running_evaluation
                    * challenges.to_hash_table_eval_indeterminate
                    + compressed_row_for_hash_input;
            }

            // Hash Table – Hash's output from Hash Coprocessor to Processor
            if let Some(prev_row) = previous_row {
                if prev_row[CI.base_table_index()] == Instruction::Hash.opcode_b() {
                    let st_5_through_9 = [
                        current_row[ST5.base_table_index()],
                        current_row[ST6.base_table_index()],
                        current_row[ST7.base_table_index()],
                        current_row[ST8.base_table_index()],
                        current_row[ST9.base_table_index()],
                    ];
                    let hash_table_digest_output_challenges = [
                        challenges.hash_table_digest_output_weight0,
                        challenges.hash_table_digest_output_weight1,
                        challenges.hash_table_digest_output_weight2,
                        challenges.hash_table_digest_output_weight3,
                        challenges.hash_table_digest_output_weight4,
                    ];
                    let compressed_row_for_hash_digest: XFieldElement = st_5_through_9
                        .into_iter()
                        .zip_eq(hash_table_digest_output_challenges.into_iter())
                        .map(|(st, weight)| weight * st)
                        .sum();
                    from_hash_table_running_evaluation = from_hash_table_running_evaluation
                        * challenges.from_hash_table_eval_indeterminate
                        + compressed_row_for_hash_digest;
                }
            }

            // Clock Jump Difference
            let current_clock_jump_difference = current_row[ClockJumpDifference.base_table_index()];
            if !current_clock_jump_difference.is_zero() {
                all_clock_jump_differences_running_product *= challenges
                    .all_clock_jump_differences_multi_perm_indeterminate
                    - current_clock_jump_difference;
            }

            // Update the running evaluation of unique clock jump differences if and only if
            // 1. the current clock jump difference is not 0, and either
            // 2a. there is no previous row, or
            // 2b. the previous clock jump difference is different from the current one.
            // We can merge (2a) and (2b) by setting the “previous” clock jump difference to
            // anything unequal to the current clock jump difference if no previous row exists.
            let previous_clock_jump_difference = if let Some(prev_row) = previous_row {
                prev_row[ClockJumpDifference.base_table_index()]
            } else {
                current_clock_jump_difference - BFieldElement::one()
            };
            if !current_clock_jump_difference.is_zero()
                && previous_clock_jump_difference != current_clock_jump_difference
            {
                unique_clock_jump_differences.push(current_clock_jump_difference);
                unique_clock_jump_differences_running_evaluation =
                    unique_clock_jump_differences_running_evaluation
                        * challenges.unique_clock_jump_differences_eval_indeterminate
                        + current_clock_jump_difference;
            }

            let mut extension_row = ext_table.row_mut(row_idx);
            extension_row[InputTableEvalArg.ext_table_index()] = input_table_running_evaluation;
            extension_row[OutputTableEvalArg.ext_table_index()] = output_table_running_evaluation;
            extension_row[InstructionTablePermArg.ext_table_index()] =
                instruction_table_running_product;
            extension_row[OpStackTablePermArg.ext_table_index()] = op_stack_table_running_product;
            extension_row[RamTablePermArg.ext_table_index()] = ram_table_running_product;
            extension_row[JumpStackTablePermArg.ext_table_index()] = jump_stack_running_product;
            extension_row[ToHashTableEvalArg.ext_table_index()] = to_hash_table_running_evaluation;
            extension_row[FromHashTableEvalArg.ext_table_index()] =
                from_hash_table_running_evaluation;
            extension_row[AllClockJumpDifferencesPermArg.ext_table_index()] =
                all_clock_jump_differences_running_product;
            extension_row[UniqueClockJumpDifferencesEvalArg.ext_table_index()] =
                unique_clock_jump_differences_running_evaluation;
            previous_row = Some(current_row);
        }

        // pre-process the unique clock jump differences for faster accesses later
        unique_clock_jump_differences.sort_by_key(|&bfe| bfe.value());
        unique_clock_jump_differences.reverse();
        if std::env::var("DEBUG").is_ok() {
            let mut unique_clock_jump_differences_copy = unique_clock_jump_differences.clone();
            unique_clock_jump_differences_copy.dedup();
            assert_eq!(
                unique_clock_jump_differences_copy,
                unique_clock_jump_differences
            );
        }

        // second pass over Processor Table to compute evaluation over all relevant clock cycles
        let mut selected_clock_cycles_running_evaluation = EvalArg::default_initial();
        for row_idx in 0..base_table.nrows() {
            let current_clk = base_table[[row_idx, CLK.base_table_index()]];
            let mut extension_row = ext_table.row_mut(row_idx);
            if unique_clock_jump_differences.last() == Some(&current_clk) {
                unique_clock_jump_differences.pop();
                selected_clock_cycles_running_evaluation = selected_clock_cycles_running_evaluation
                    * challenges.unique_clock_jump_differences_eval_indeterminate
                    + current_clk;
            }
            extension_row[SelectedClockCyclesEvalArg.ext_table_index()] =
                selected_clock_cycles_running_evaluation;
        }

        assert!(
            unique_clock_jump_differences.is_empty(),
            "Unhandled unique clock jump differences: {unique_clock_jump_differences:?}"
        );
        assert_eq!(
            unique_clock_jump_differences_running_evaluation,
            selected_clock_cycles_running_evaluation,
            "Even though all unique clock jump differences were handled, the running evaluation of \
             unique clock jump differences is not equal to the running evaluation of selected \
             clock cycles."
        );
    }
}

impl ExtProcessorTable {
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
        factory: &mut DualRowConstraints,
        instr_tc_polys_tuples: [(
            Instruction,
            Vec<
                ConstraintCircuitMonad<
                    ProcessorTableChallenges,
                    DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
                >,
            >,
        ); Instruction::COUNT],
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let (all_instructions, all_tc_polys_for_all_instructions): (Vec<_>, Vec<Vec<_>>) =
            instr_tc_polys_tuples.into_iter().unzip();

        let instruction_deselectors = InstructionDeselectors::new(factory);

        let all_instruction_deselectors = all_instructions
            .into_iter()
            .map(|instr| instruction_deselectors.get(instr))
            .collect_vec();

        let max_number_of_constraints = all_tc_polys_for_all_instructions
            .iter()
            .map(|tc_polys_for_instr| tc_polys_for_instr.len())
            .max()
            .unwrap();
        let zero_poly = DualRowConstraints::default().zero();

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
        factory: &DualRowConstraints,
        instruction_transition_constraints: Vec<
            ConstraintCircuitMonad<
                ProcessorTableChallenges,
                DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
            >,
        >,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let ip_remains = factory.ip_next() - factory.ip();
        let ci_remains = factory.ci_next() - factory.ci();
        let nia_remains = factory.nia_next() - factory.nia();
        let padding_row_transition_constraints = [
            vec![ip_remains, ci_remains, nia_remains],
            factory.keep_jump_stack(),
            factory.keep_stack(),
            factory.keep_ram(),
        ]
        .concat();

        let padding_row_deselector = factory.one() - factory.is_padding_next();
        let padding_row_selector = factory.is_padding_next();

        let max_number_of_constraints = max(
            instruction_transition_constraints.len(),
            padding_row_transition_constraints.len(),
        );

        (0..max_number_of_constraints)
            .map(|idx| {
                let instruction_constraint = instruction_transition_constraints
                    .get(idx)
                    .unwrap_or(&factory.zero())
                    .clone();
                let padding_constraint = padding_row_transition_constraints
                    .get(idx)
                    .unwrap_or(&factory.zero())
                    .clone();

                instruction_constraint * padding_row_deselector.clone()
                    + padding_constraint * padding_row_selector.clone()
            })
            .collect_vec()
    }
}

#[derive(Debug, Copy, Clone, Display, EnumCountMacro, EnumIter, PartialEq, Eq, Hash)]
pub enum ProcessorTableChallengeId {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the processor table.
    StandardInputEvalIndeterminate,
    StandardOutputEvalIndeterminate,
    ToHashTableEvalIndeterminate,
    FromHashTableEvalIndeterminate,

    InstructionPermIndeterminate,
    OpStackPermIndeterminate,
    RamPermIndeterminate,
    JumpStackPermIndeterminate,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    InstructionTableIpWeight,
    InstructionTableCiProcessorWeight,
    InstructionTableNiaWeight,

    OpStackTableClkWeight,
    OpStackTableIb1Weight,
    OpStackTableOspWeight,
    OpStackTableOsvWeight,

    RamTableClkWeight,
    RamTableRamvWeight,
    RamTableRampWeight,
    RamTablePreviousInstructionWeight,

    JumpStackTableClkWeight,
    JumpStackTableCiWeight,
    JumpStackTableJspWeight,
    JumpStackTableJsoWeight,
    JumpStackTableJsdWeight,

    UniqueClockJumpDifferencesEvalIndeterminate,
    AllClockJumpDifferencesMultiPermIndeterminate,

    // 2 * DIGEST_LENGTH elements of these
    HashTableStackInputWeight0,
    HashTableStackInputWeight1,
    HashTableStackInputWeight2,
    HashTableStackInputWeight3,
    HashTableStackInputWeight4,
    HashTableStackInputWeight5,
    HashTableStackInputWeight6,
    HashTableStackInputWeight7,
    HashTableStackInputWeight8,
    HashTableStackInputWeight9,

    // DIGEST_LENGTH elements of these
    HashTableDigestOutputWeight0,
    HashTableDigestOutputWeight1,
    HashTableDigestOutputWeight2,
    HashTableDigestOutputWeight3,
    HashTableDigestOutputWeight4,
}

impl From<ProcessorTableChallengeId> for usize {
    fn from(val: ProcessorTableChallengeId) -> Self {
        val as usize
    }
}

#[derive(Debug, Clone)]
pub struct ProcessorTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the processor table.
    pub standard_input_eval_indeterminate: XFieldElement,
    pub standard_output_eval_indeterminate: XFieldElement,
    pub to_hash_table_eval_indeterminate: XFieldElement,
    pub from_hash_table_eval_indeterminate: XFieldElement,

    pub instruction_perm_indeterminate: XFieldElement,
    pub op_stack_perm_indeterminate: XFieldElement,
    pub ram_perm_indeterminate: XFieldElement,
    pub jump_stack_perm_indeterminate: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub instruction_table_ip_weight: XFieldElement,
    pub instruction_table_ci_processor_weight: XFieldElement,
    pub instruction_table_nia_weight: XFieldElement,

    pub op_stack_table_clk_weight: XFieldElement,
    pub op_stack_table_ib1_weight: XFieldElement,
    pub op_stack_table_osp_weight: XFieldElement,
    pub op_stack_table_osv_weight: XFieldElement,

    pub ram_table_clk_weight: XFieldElement,
    pub ram_table_ramp_weight: XFieldElement,
    pub ram_table_ramv_weight: XFieldElement,
    pub ram_table_previous_instruction_weight: XFieldElement,

    pub jump_stack_table_clk_weight: XFieldElement,
    pub jump_stack_table_ci_weight: XFieldElement,
    pub jump_stack_table_jsp_weight: XFieldElement,
    pub jump_stack_table_jso_weight: XFieldElement,
    pub jump_stack_table_jsd_weight: XFieldElement,

    pub unique_clock_jump_differences_eval_indeterminate: XFieldElement,
    pub all_clock_jump_differences_multi_perm_indeterminate: XFieldElement,

    // 2 * DIGEST_LENGTH elements of these
    pub hash_table_stack_input_weight0: XFieldElement,
    pub hash_table_stack_input_weight1: XFieldElement,
    pub hash_table_stack_input_weight2: XFieldElement,
    pub hash_table_stack_input_weight3: XFieldElement,
    pub hash_table_stack_input_weight4: XFieldElement,
    pub hash_table_stack_input_weight5: XFieldElement,
    pub hash_table_stack_input_weight6: XFieldElement,
    pub hash_table_stack_input_weight7: XFieldElement,
    pub hash_table_stack_input_weight8: XFieldElement,
    pub hash_table_stack_input_weight9: XFieldElement,

    // DIGEST_LENGTH elements of these
    pub hash_table_digest_output_weight0: XFieldElement,
    pub hash_table_digest_output_weight1: XFieldElement,
    pub hash_table_digest_output_weight2: XFieldElement,
    pub hash_table_digest_output_weight3: XFieldElement,
    pub hash_table_digest_output_weight4: XFieldElement,
}

impl TableChallenges for ProcessorTableChallenges {
    type Id = ProcessorTableChallengeId;

    #[inline]
    fn get_challenge(&self, id: Self::Id) -> XFieldElement {
        match id {
            StandardInputEvalIndeterminate => self.standard_input_eval_indeterminate,
            StandardOutputEvalIndeterminate => self.standard_output_eval_indeterminate,
            ToHashTableEvalIndeterminate => self.to_hash_table_eval_indeterminate,
            FromHashTableEvalIndeterminate => self.from_hash_table_eval_indeterminate,
            InstructionPermIndeterminate => self.instruction_perm_indeterminate,
            OpStackPermIndeterminate => self.op_stack_perm_indeterminate,
            RamPermIndeterminate => self.ram_perm_indeterminate,
            JumpStackPermIndeterminate => self.jump_stack_perm_indeterminate,
            InstructionTableIpWeight => self.instruction_table_ip_weight,
            InstructionTableCiProcessorWeight => self.instruction_table_ci_processor_weight,
            InstructionTableNiaWeight => self.instruction_table_nia_weight,
            OpStackTableClkWeight => self.op_stack_table_clk_weight,
            OpStackTableIb1Weight => self.op_stack_table_ib1_weight,
            OpStackTableOspWeight => self.op_stack_table_osp_weight,
            OpStackTableOsvWeight => self.op_stack_table_osv_weight,
            RamTableClkWeight => self.ram_table_clk_weight,
            RamTableRamvWeight => self.ram_table_ramv_weight,
            RamTableRampWeight => self.ram_table_ramp_weight,
            RamTablePreviousInstructionWeight => self.ram_table_previous_instruction_weight,
            JumpStackTableClkWeight => self.jump_stack_table_clk_weight,
            JumpStackTableCiWeight => self.jump_stack_table_ci_weight,
            JumpStackTableJspWeight => self.jump_stack_table_jsp_weight,
            JumpStackTableJsoWeight => self.jump_stack_table_jso_weight,
            JumpStackTableJsdWeight => self.jump_stack_table_jsd_weight,
            UniqueClockJumpDifferencesEvalIndeterminate => {
                self.unique_clock_jump_differences_eval_indeterminate
            }
            AllClockJumpDifferencesMultiPermIndeterminate => {
                self.all_clock_jump_differences_multi_perm_indeterminate
            }
            HashTableStackInputWeight0 => self.hash_table_stack_input_weight0,
            HashTableStackInputWeight1 => self.hash_table_stack_input_weight1,
            HashTableStackInputWeight2 => self.hash_table_stack_input_weight2,
            HashTableStackInputWeight3 => self.hash_table_stack_input_weight3,
            HashTableStackInputWeight4 => self.hash_table_stack_input_weight4,
            HashTableStackInputWeight5 => self.hash_table_stack_input_weight5,
            HashTableStackInputWeight6 => self.hash_table_stack_input_weight6,
            HashTableStackInputWeight7 => self.hash_table_stack_input_weight7,
            HashTableStackInputWeight8 => self.hash_table_stack_input_weight8,
            HashTableStackInputWeight9 => self.hash_table_stack_input_weight9,
            HashTableDigestOutputWeight0 => self.hash_table_digest_output_weight0,
            HashTableDigestOutputWeight1 => self.hash_table_digest_output_weight1,
            HashTableDigestOutputWeight2 => self.hash_table_digest_output_weight2,
            HashTableDigestOutputWeight3 => self.hash_table_digest_output_weight3,
            HashTableDigestOutputWeight4 => self.hash_table_digest_output_weight4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IOChallenges {
    /// weight for updating the running evaluation with the next i/o symbol in the i/o list
    pub processor_eval_indeterminate: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct ExtProcessorTable {}

impl ExtProcessorTable {
    pub fn ext_initial_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            ProcessorTableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        use ProcessorTableChallengeId::*;

        let factory = SingleRowConstraints::default();
        let constant = |c| factory.constant_from_i32(c);
        let constant_x = |x| factory.constant_from_xfe(x);
        let challenge = |c| factory.circuit_builder.challenge(c);

        let clk_is_0 = factory.clk();
        let ip_is_0 = factory.ip();
        let jsp_is_0 = factory.jsp();
        let jso_is_0 = factory.jso();
        let jsd_is_0 = factory.jsd();
        let st0_is_0 = factory.st0();
        let st1_is_0 = factory.st1();
        let st2_is_0 = factory.st2();
        let st3_is_0 = factory.st3();
        let st4_is_0 = factory.st4();
        let st5_is_0 = factory.st5();
        let st6_is_0 = factory.st6();
        let st7_is_0 = factory.st7();
        let st8_is_0 = factory.st8();
        let st9_is_0 = factory.st9();
        let st10_is_0 = factory.st10();
        let st11_is_0 = factory.st11();
        let st12_is_0 = factory.st12();
        let st13_is_0 = factory.st13();
        let st14_is_0 = factory.st14();
        let st15_is_0 = factory.st15();
        let osp_is_16 = factory.osp() - constant(16);
        let osv_is_0 = factory.osv();
        let ramv_is_0 = factory.ramv();
        let ramp_is_0 = factory.ramp();
        let previous_instruction = factory.previous_instruction();

        // The running evaluation of relevant clock cycles `rer` starts with the initial.
        let rer_starts_correctly = factory.rer() - constant_x(EvalArg::default_initial());

        // The running evaluation of unique clock jump differences
        // starts off having applied one evaluation step with the clock
        // jump difference, unless the clock jump difference column
        // is all zeros.
        let reu_indeterminate = challenge(UniqueClockJumpDifferencesEvalIndeterminate);
        let reu_starts_correctly = factory.cjd()
            * (factory.reu() - reu_indeterminate - factory.cjd())
            + (factory.one() - factory.cjd() * factory.invm())
                * (factory.reu() - constant_x(PermArg::default_initial()));

        // The running product for all clock jump differences
        // starts off having accumulated the first factor, but
        // only if the `cjd` is nonzero
        let rpm_indeterminate = challenge(AllClockJumpDifferencesMultiPermIndeterminate);
        let rpm_starts_correctly = factory.cjd()
            * (factory.rpm() - rpm_indeterminate + factory.cjd())
            + (factory.one() - factory.invm() * factory.cjd())
                * (factory.rpm()
                    - constant_x(PermArg::default_initial())
                        * constant_x(PermArg::default_initial())
                        * constant_x(PermArg::default_initial()));

        // Permutation and Evaluation Arguments with all tables the Processor Table relates to

        // standard input
        let running_evaluation_for_standard_input_is_initialized_correctly =
            factory.running_evaluation_standard_input() - constant_x(EvalArg::default_initial());

        // instruction table
        let instruction_indeterminate = challenge(InstructionPermIndeterminate);
        let instruction_ci_weight = challenge(InstructionTableCiProcessorWeight);
        let instruction_nia_weight = challenge(InstructionTableNiaWeight);
        let compressed_row_for_instruction_table =
            instruction_ci_weight * factory.ci() + instruction_nia_weight * factory.nia();
        let running_product_for_instruction_table_is_initialized_correctly = factory
            .running_product_instruction_table()
            - constant_x(PermArg::default_initial())
                * (instruction_indeterminate - compressed_row_for_instruction_table);

        // standard output
        let running_evaluation_for_standard_output_is_initialized_correctly =
            factory.running_evaluation_standard_output() - constant_x(EvalArg::default_initial());

        // op-stack table
        let op_stack_indeterminate = challenge(OpStackPermIndeterminate);
        let op_stack_ib1_weight = challenge(OpStackTableIb1Weight);
        let op_stack_osp_weight = challenge(OpStackTableOspWeight);
        // note: `clk` and `osv` are already constrained to be 0, `osp` to be 16
        let compressed_row_for_op_stack_table =
            op_stack_ib1_weight * factory.ib1() + op_stack_osp_weight * constant(16);
        let running_product_for_op_stack_table_is_initialized_correctly = factory
            .running_product_op_stack_table()
            - constant_x(PermArg::default_initial())
                * (op_stack_indeterminate - compressed_row_for_op_stack_table);

        // ram table
        let ram_indeterminate = challenge(RamPermIndeterminate);
        // note: `clk`, `ramp`, and `ramv` are already constrained to be 0.
        let compressed_row_for_ram_table = constant(0);
        let running_product_for_ram_table_is_initialized_correctly = factory
            .running_product_ram_table()
            - constant_x(PermArg::default_initial())
                * (ram_indeterminate - compressed_row_for_ram_table);

        // jump-stack table
        let jump_stack_indeterminate = challenge(JumpStackPermIndeterminate);
        let jump_stack_ci_weight = challenge(JumpStackTableCiWeight);
        // note: `clk`, `jsp`, `jso`, and `jsd` are already constrained to be 0.
        let compressed_row_for_jump_stack_table = jump_stack_ci_weight * factory.ci();
        let running_product_for_jump_stack_table_is_initialized_correctly = factory
            .running_product_jump_stack_table()
            - constant_x(PermArg::default_initial())
                * (jump_stack_indeterminate - compressed_row_for_jump_stack_table);

        // from processor to hash table
        let hash_selector = factory.ci() - constant(Instruction::Hash.opcode() as i32);
        let hash_deselector =
            InstructionDeselectors::instruction_deselector_single_row(&factory, Instruction::Hash);
        let to_hash_table_indeterminate = challenge(ToHashTableEvalIndeterminate);
        // the opStack is guaranteed to be initialized to 0 by virtue of other initial constraints
        let compressed_row_to_hash_table = constant(0);
        let running_evaluation_to_hash_table_has_absorbed_first_row = factory
            .running_evaluation_to_hash_table()
            - to_hash_table_indeterminate * constant_x(EvalArg::default_initial())
            - compressed_row_to_hash_table;
        let running_evaluation_to_hash_table_is_default_initial =
            factory.running_evaluation_to_hash_table() - constant_x(EvalArg::default_initial());
        let running_evaluation_to_hash_table_is_initialized_correctly = hash_selector
            * running_evaluation_to_hash_table_is_default_initial
            + hash_deselector * running_evaluation_to_hash_table_has_absorbed_first_row;

        // from hash table to processor
        let running_evaluation_from_hash_table_is_initialized_correctly =
            factory.running_evaluation_from_hash_table() - constant_x(EvalArg::default_initial());

        [
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
            previous_instruction,
            rer_starts_correctly,
            reu_starts_correctly,
            rpm_starts_correctly,
            running_evaluation_for_standard_input_is_initialized_correctly,
            running_product_for_instruction_table_is_initialized_correctly,
            running_evaluation_for_standard_output_is_initialized_correctly,
            running_product_for_op_stack_table_is_initialized_correctly,
            running_product_for_ram_table_is_initialized_correctly,
            running_product_for_jump_stack_table_is_initialized_correctly,
            running_evaluation_to_hash_table_is_initialized_correctly,
            running_evaluation_from_hash_table_is_initialized_correctly,
        ]
        .map(|circuit| circuit.consume())
        .to_vec()
    }

    pub fn ext_consistency_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            ProcessorTableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let factory = SingleRowConstraints::default();
        let one = factory.one();
        let constant = |c| factory.constant_from_i32(c);

        // The composition of instruction buckets ib0-ib9 corresponds the current instruction ci.v
        let ib_composition = one.clone() * factory.ib0()
            + constant(1 << 1) * factory.ib1()
            + constant(1 << 2) * factory.ib2()
            + constant(1 << 3) * factory.ib3()
            + constant(1 << 4) * factory.ib4()
            + constant(1 << 5) * factory.ib5()
            + constant(1 << 6) * factory.ib6();
        let ci_corresponds_to_ib0_thru_ib12 = factory.ci() - ib_composition;

        let ib0_is_bit = factory.ib0() * (factory.ib0() - one.clone());
        let ib1_is_bit = factory.ib1() * (factory.ib1() - one.clone());
        let ib2_is_bit = factory.ib2() * (factory.ib2() - one.clone());
        let ib3_is_bit = factory.ib3() * (factory.ib3() - one.clone());
        let ib4_is_bit = factory.ib4() * (factory.ib4() - one.clone());
        let ib5_is_bit = factory.ib5() * (factory.ib5() - one.clone());
        let ib6_is_bit = factory.ib6() * (factory.ib6() - one.clone());
        let is_padding_is_bit = factory.is_padding() * (factory.is_padding() - one);

        // The inverse of clock jump difference with multiplicity `invm` is the inverse-or-zero of
        // the clock jump difference `cjd`.
        let invm_is_cjd_inverse = factory.invm() * factory.cjd() - factory.one();
        let invm_is_zero_or_cjd_inverse = factory.invm() * invm_is_cjd_inverse.clone();
        let cjd_is_zero_or_invm_inverse = factory.cjd() * invm_is_cjd_inverse;

        [
            ib0_is_bit,
            ib1_is_bit,
            ib2_is_bit,
            ib3_is_bit,
            ib4_is_bit,
            ib5_is_bit,
            ib6_is_bit,
            is_padding_is_bit,
            ci_corresponds_to_ib0_thru_ib12,
            invm_is_zero_or_cjd_inverse,
            cjd_is_zero_or_invm_inverse,
        ]
        .map(|circuit| circuit.consume())
        .to_vec()
    }

    pub fn ext_transition_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let mut factory = DualRowConstraints::default();

        // instruction-specific constraints
        let all_instruction_transition_constraints: [_; Instruction::COUNT] = [
            (Pop, factory.instruction_pop()),
            (Push(Default::default()), factory.instruction_push()),
            (Divine(Default::default()), factory.instruction_divine()),
            (Dup(Default::default()), factory.instruction_dup()),
            (Swap(Default::default()), factory.instruction_swap()),
            (Nop, factory.instruction_nop()),
            (Skiz, factory.instruction_skiz()),
            (Call(Default::default()), factory.instruction_call()),
            (Return, factory.instruction_return()),
            (Recurse, factory.instruction_recurse()),
            (Assert, factory.instruction_assert()),
            (Halt, factory.instruction_halt()),
            (ReadMem, factory.instruction_read_mem()),
            (WriteMem, factory.instruction_write_mem()),
            (Hash, factory.instruction_hash()),
            (DivineSibling, factory.instruction_divine_sibling()),
            (AssertVector, factory.instruction_assert_vector()),
            (Add, factory.instruction_add()),
            (Mul, factory.instruction_mul()),
            (Invert, factory.instruction_invert()),
            (Split, factory.instruction_split()),
            (Eq, factory.instruction_eq()),
            (Lsb, factory.instruction_lsb()),
            (XxAdd, factory.instruction_xxadd()),
            (XxMul, factory.instruction_xxmul()),
            (XInvert, factory.instruction_xinv()),
            (XbMul, factory.instruction_xbmul()),
            (ReadIo, factory.instruction_read_io()),
            (WriteIo, factory.instruction_write_io()),
        ];

        let mut transition_constraints = Self::combine_instruction_constraints_with_deselectors(
            &mut factory,
            all_instruction_transition_constraints,
        );

        // if next row is padding row: disable transition constraints, enable padding constraints
        transition_constraints = Self::combine_transition_constraints_with_padding_constraints(
            &factory,
            transition_constraints,
        );

        // constraints common to all instructions
        transition_constraints.insert(0, factory.clk_always_increases_by_one());
        transition_constraints.insert(1, factory.is_padding_is_zero_or_does_not_change());
        transition_constraints.insert(2, factory.previous_instruction_is_copied_correctly());

        // constraints related to clock jump difference argument

        // The unique inverse column `invu` holds the inverse-or-zero
        // of the difference of consecutive `cjd`'s.
        // invu' = (cjd' - cjd)^(p-2)
        let cjdd = factory.cjd_next() - factory.cjd();
        let invu_next_is_cjdd_inverse = factory.invu_next() * cjdd.clone() - factory.one();
        let invu_next_is_zero_or_cjdd_inverse =
            factory.invu_next() * invu_next_is_cjdd_inverse.clone();
        let cjdd_is_zero_if_invu_inverse = cjdd.clone() * invu_next_is_cjdd_inverse.clone();

        // The running product `rpm` of cjd's with multiplicities
        // accumulates a factor α - cjd' in every row, provided that
        // `cjd'` is nonzero.
        // cjd' · (rpm' - rpm · (α - cjd')) + (cjd' · invm' - 1) · (rpm' - rpm)
        let indeterminate_alpha = factory
            .circuit_builder
            .challenge(AllClockJumpDifferencesMultiPermIndeterminate);
        let rpm_updates_correctly = factory.cjd_next()
            * (factory.rpm_next() - factory.rpm() * (indeterminate_alpha - factory.cjd_next()))
            + (factory.cjd_next() * factory.invm_next() - factory.one())
                * (factory.rpm_next() - factory.rpm());

        // The running evaluation `reu` of unique `cjd`'s is updated
        // relative to evaluation point β whenever the difference of
        // `cjd`'s is nonzero *and* `cjd'` is nonzero.
        // `(1 - (cjd' - cjd) · invu) · (reu' - reu)
        //  + · (1 - cjd' · invm) · (reu' - reu)
        //  + cjd' · (cjd' - cjd) · (reu' - β · reu - cjd')`
        let indeterminate_beta = factory
            .circuit_builder
            .challenge(UniqueClockJumpDifferencesEvalIndeterminate);
        let reu_updates_correctly = invu_next_is_cjdd_inverse
            * (factory.reu_next() - factory.reu())
            + (factory.one() - factory.invm_next() * factory.cjd_next())
                * (factory.reu_next() - factory.reu())
            + factory.cjd_next()
                * cjdd
                * (factory.reu_next()
                    - indeterminate_beta.clone() * factory.reu()
                    - factory.cjd_next());

        // The running evaluation `rer` of relevant clock cycles is
        // updated relative to evaluation point β or not at all.
        // (rer' - rer · β - clk') · (rer' - rer)
        let rer_updates_correctly =
            (factory.rer_next() - factory.rer() * indeterminate_beta - factory.clk_next())
                * (factory.rer_next() - factory.rer());

        transition_constraints.append(&mut vec![
            invu_next_is_zero_or_cjdd_inverse * factory.cjd_next(),
            cjdd_is_zero_if_invu_inverse * factory.cjd_next(),
            rpm_updates_correctly,
            reu_updates_correctly,
            rer_updates_correctly,
        ]);

        // constraints related to evaluation and permutation arguments

        transition_constraints
            .push(factory.running_evaluation_for_standard_input_updates_correctly());
        transition_constraints
            .push(factory.running_product_for_instruction_table_updates_correctly());
        transition_constraints
            .push(factory.running_evaluation_for_standard_output_updates_correctly());
        transition_constraints.push(factory.running_product_for_op_stack_table_updates_correctly());
        transition_constraints.push(factory.running_product_for_ram_table_updates_correctly());
        transition_constraints
            .push(factory.running_product_for_jump_stack_table_updates_correctly());
        transition_constraints.push(factory.running_evaluation_to_hash_table_updates_correctly());
        transition_constraints.push(factory.running_evaluation_from_hash_table_updates_correctly());

        let mut built_transition_constraints = transition_constraints
            .into_iter()
            .map(|tc_ref| tc_ref.consume())
            .collect_vec();
        ConstraintCircuit::constant_folding(
            &mut built_transition_constraints.iter_mut().collect_vec(),
        );
        built_transition_constraints
    }

    pub fn ext_terminal_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            ProcessorTableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let factory = SingleRowConstraints::default();

        // In the last row, current instruction register ci is 0, corresponding to instruction halt.
        let last_ci_is_halt = factory.ci();

        // In the last row, the completed evaluations of
        // a) relevant clock cycles, and
        // b) unique clock jump differences are equal.
        let rer_equals_reu = factory.rer() - factory.reu();

        vec![last_ci_is_halt.consume(), rer_equals_reu.consume()]
    }
}

#[derive(Debug, Clone)]
pub struct SingleRowConstraints {
    base_row_variables: [ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    >; NUM_BASE_COLUMNS],
    ext_row_variables: [ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    >; NUM_EXT_COLUMNS],
    circuit_builder: ConstraintCircuitBuilder<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    >,
    one: ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    >,
    two: ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    >,
}

impl Default for SingleRowConstraints {
    fn default() -> Self {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let base_row_variables = (0..NUM_BASE_COLUMNS)
            .map(|i| circuit_builder.input(SingleRowIndicator::BaseRow(i)))
            .collect_vec()
            .try_into()
            .expect("Create variables for single base row constraints");
        let ext_row_variables = (0..NUM_EXT_COLUMNS)
            .map(|i| circuit_builder.input(SingleRowIndicator::ExtRow(i)))
            .collect_vec()
            .try_into()
            .expect("Create variables for single ext row constraints");

        let one = circuit_builder.b_constant(1u32.into());
        let two = circuit_builder.b_constant(2u32.into());

        Self {
            base_row_variables,
            ext_row_variables,
            circuit_builder,
            one,
            two,
        }
    }
}

impl SingleRowConstraints {
    pub fn constant_from_xfe(
        &self,
        constant: XFieldElement,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.circuit_builder.x_constant(constant)
    }
    pub fn constant_from_i32(
        &self,
        constant: i32,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        let bfe = if constant < 0 {
            BFieldElement::new(BFieldElement::QUOTIENT - ((-constant) as u64))
        } else {
            BFieldElement::new(constant as u64)
        };
        self.circuit_builder.b_constant(bfe)
    }

    pub fn one(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.one.clone()
    }
    pub fn two(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.two.clone()
    }
    pub fn clk(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[CLK.master_base_table_index()].clone()
    }
    pub fn is_padding(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[IsPadding.master_base_table_index()].clone()
    }
    pub fn ip(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[IP.master_base_table_index()].clone()
    }
    pub fn ci(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[CI.master_base_table_index()].clone()
    }
    pub fn nia(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[NIA.master_base_table_index()].clone()
    }
    pub fn ib0(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[IB0.master_base_table_index()].clone()
    }
    pub fn ib1(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[IB1.master_base_table_index()].clone()
    }
    pub fn ib2(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[IB2.master_base_table_index()].clone()
    }
    pub fn ib3(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[IB3.master_base_table_index()].clone()
    }
    pub fn ib4(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[IB4.master_base_table_index()].clone()
    }
    pub fn ib5(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[IB5.master_base_table_index()].clone()
    }
    pub fn ib6(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[IB6.master_base_table_index()].clone()
    }
    pub fn jsp(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[JSP.master_base_table_index()].clone()
    }
    pub fn jsd(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[JSD.master_base_table_index()].clone()
    }
    pub fn jso(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[JSO.master_base_table_index()].clone()
    }
    pub fn st0(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[ST0.master_base_table_index()].clone()
    }
    pub fn st1(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[ST1.master_base_table_index()].clone()
    }
    pub fn st2(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[ST2.master_base_table_index()].clone()
    }
    pub fn st3(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[ST3.master_base_table_index()].clone()
    }
    pub fn st4(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[ST4.master_base_table_index()].clone()
    }
    pub fn st5(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[ST5.master_base_table_index()].clone()
    }
    pub fn st6(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[ST6.master_base_table_index()].clone()
    }
    pub fn st7(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[ST7.master_base_table_index()].clone()
    }
    pub fn st8(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[ST8.master_base_table_index()].clone()
    }
    pub fn st9(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[ST9.master_base_table_index()].clone()
    }
    pub fn st10(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[ST10.master_base_table_index()].clone()
    }
    pub fn st11(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[ST11.master_base_table_index()].clone()
    }
    pub fn st12(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[ST12.master_base_table_index()].clone()
    }
    pub fn st13(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[ST13.master_base_table_index()].clone()
    }
    pub fn st14(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[ST14.master_base_table_index()].clone()
    }
    pub fn st15(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[ST15.master_base_table_index()].clone()
    }
    pub fn osp(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[OSP.master_base_table_index()].clone()
    }
    pub fn osv(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[OSV.master_base_table_index()].clone()
    }
    pub fn hv0(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[HV0.master_base_table_index()].clone()
    }
    pub fn hv1(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[HV1.master_base_table_index()].clone()
    }
    pub fn hv2(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[HV2.master_base_table_index()].clone()
    }
    pub fn hv3(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[HV3.master_base_table_index()].clone()
    }
    pub fn ramv(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[RAMV.master_base_table_index()].clone()
    }
    pub fn ramp(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[RAMP.master_base_table_index()].clone()
    }
    pub fn previous_instruction(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[PreviousInstruction.master_base_table_index()].clone()
    }

    pub fn cjd(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[ClockJumpDifference.master_base_table_index()].clone()
    }

    pub fn invm(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[ClockJumpDifferenceInverse.master_base_table_index()].clone()
    }

    pub fn invu(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.base_row_variables[UniqueClockJumpDiffDiffInverse.master_base_table_index()].clone()
    }

    pub fn rer(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.ext_row_variables[SelectedClockCyclesEvalArg.master_ext_table_index()].clone()
    }

    pub fn reu(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.ext_row_variables[UniqueClockJumpDifferencesEvalArg.master_ext_table_index()].clone()
    }

    pub fn rpm(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.ext_row_variables[AllClockJumpDifferencesPermArg.master_ext_table_index()].clone()
    }

    pub fn running_evaluation_standard_input(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.ext_row_variables[InputTableEvalArg.master_ext_table_index()].clone()
    }
    pub fn running_evaluation_standard_output(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.ext_row_variables[OutputTableEvalArg.master_ext_table_index()].clone()
    }
    pub fn running_product_instruction_table(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.ext_row_variables[InstructionTablePermArg.master_ext_table_index()].clone()
    }
    pub fn running_product_op_stack_table(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.ext_row_variables[OpStackTablePermArg.master_ext_table_index()].clone()
    }
    pub fn running_product_ram_table(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.ext_row_variables[RamTablePermArg.master_ext_table_index()].clone()
    }
    pub fn running_product_jump_stack_table(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.ext_row_variables[JumpStackTablePermArg.master_ext_table_index()].clone()
    }
    pub fn running_evaluation_to_hash_table(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.ext_row_variables[ToHashTableEvalArg.master_ext_table_index()].clone()
    }
    pub fn running_evaluation_from_hash_table(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.ext_row_variables[FromHashTableEvalArg.master_ext_table_index()].clone()
    }
}

#[derive(Debug, Clone)]
pub struct DualRowConstraints {
    current_base_row_variables: [ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    >; NUM_BASE_COLUMNS],
    current_ext_row_variables: [ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    >; NUM_EXT_COLUMNS],
    next_base_row_variables: [ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    >; NUM_BASE_COLUMNS],
    next_ext_row_variables: [ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    >; NUM_EXT_COLUMNS],
    circuit_builder: ConstraintCircuitBuilder<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    >,
    zero: ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    >,
    one: ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    >,
    two: ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    >,
}

impl Default for DualRowConstraints {
    fn default() -> Self {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let current_base_row_variables = (0..NUM_BASE_COLUMNS)
            .map(|i| circuit_builder.input(DualRowIndicator::CurrentBaseRow(i)))
            .collect_vec()
            .try_into()
            .expect("Create variables for dual rows – current base row");
        let current_ext_row_variables = (0..NUM_EXT_COLUMNS)
            .map(|i| circuit_builder.input(DualRowIndicator::CurrentExtRow(i)))
            .collect_vec()
            .try_into()
            .expect("Create variables for dual rows – current ext row");
        let next_base_row_variables = (0..NUM_BASE_COLUMNS)
            .map(|i| circuit_builder.input(DualRowIndicator::NextBaseRow(i)))
            .collect_vec()
            .try_into()
            .expect("Create variables for dual rows – next base row");
        let next_ext_row_variables = (0..NUM_EXT_COLUMNS)
            .map(|i| circuit_builder.input(DualRowIndicator::NextExtRow(i)))
            .collect_vec()
            .try_into()
            .expect("Create variables for dual rows – next ext row");

        let zero = circuit_builder.b_constant(0u32.into());
        let one = circuit_builder.b_constant(1u32.into());
        let two = circuit_builder.b_constant(2u32.into());

        Self {
            current_base_row_variables,
            current_ext_row_variables,
            next_base_row_variables,
            next_ext_row_variables,
            circuit_builder,
            zero,
            one,
            two,
        }
    }
}

impl DualRowConstraints {
    /// ## The cycle counter (`clk`) always increases by one
    ///
    /// $$
    /// p(..., clk, clk_next, ...) = clk_next - clk - 1
    /// $$
    ///
    /// In general, for all $clk = a$, and $clk_next = a + 1$,
    ///
    /// $$
    /// p(..., a, a+1, ...) = (a+1) - a - 1 = a + 1 - a - 1 = a - a + 1 - 1 = 0
    /// $$
    ///
    /// So the `clk_increase_by_one` base transition constraint polynomial holds exactly
    /// when every `clk` register $a$ is one less than `clk` register $a + 1$.
    pub fn clk_always_increases_by_one(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        let one = self.one();
        let clk = self.clk();
        let clk_next = self.clk_next();

        clk_next - clk - one
    }

    pub fn is_padding_is_zero_or_does_not_change(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.is_padding() * (self.is_padding_next() - self.is_padding())
    }

    pub fn previous_instruction_is_copied_correctly(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        (self.previous_instruction_next() - self.ci()) * (self.one() - self.is_padding_next())
    }

    pub fn indicator_polynomial(
        &self,
        i: usize,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        let hv0 = self.hv0();
        let hv1 = self.hv1();
        let hv2 = self.hv2();
        let hv3 = self.hv3();

        match i {
            0 => (self.one() - hv3) * (self.one() - hv2) * (self.one() - hv1) * (self.one() - hv0),
            1 => (self.one() - hv3) * (self.one() - hv2) * (self.one() - hv1) * hv0,
            2 => (self.one() - hv3) * (self.one() - hv2) * hv1 * (self.one() - hv0),
            3 => (self.one() - hv3) * (self.one() - hv2) * hv1 * hv0,
            4 => (self.one() - hv3) * hv2 * (self.one() - hv1) * (self.one() - hv0),
            5 => (self.one() - hv3) * hv2 * (self.one() - hv1) * hv0,
            6 => (self.one() - hv3) * hv2 * hv1 * (self.one() - hv0),
            7 => (self.one() - hv3) * hv2 * hv1 * hv0,
            8 => hv3 * (self.one() - hv2) * (self.one() - hv1) * (self.one() - hv0),
            9 => hv3 * (self.one() - hv2) * (self.one() - hv1) * hv0,
            10 => hv3 * (self.one() - hv2) * hv1 * (self.one() - hv0),
            11 => hv3 * (self.one() - hv2) * hv1 * hv0,
            12 => hv3 * hv2 * (self.one() - hv1) * (self.one() - hv0),
            13 => hv3 * hv2 * (self.one() - hv1) * hv0,
            14 => hv3 * hv2 * hv1 * (self.one() - hv0),
            15 => hv3 * hv2 * hv1 * hv0,
            _ => panic!("No indicator polynomial with index {i} exists: there are only 16."),
        }
    }

    pub fn instruction_pop(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        [self.step_1(), self.shrink_stack(), self.keep_ram()].concat()
    }

    /// push'es argument should be on the stack after execution
    /// $st0_next == nia  =>  st0_next - nia == 0$
    pub fn instruction_push(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let specific_constraints = vec![self.st0_next() - self.nia()];
        [
            specific_constraints,
            self.grow_stack(),
            self.step_2(),
            self.keep_ram(),
        ]
        .concat()
    }

    pub fn instruction_divine(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        [self.step_1(), self.grow_stack(), self.keep_ram()].concat()
    }

    pub fn instruction_dup(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let specific_constraints = vec![
            self.indicator_polynomial(0) * (self.st0_next() - self.st0()),
            self.indicator_polynomial(1) * (self.st0_next() - self.st1()),
            self.indicator_polynomial(2) * (self.st0_next() - self.st2()),
            self.indicator_polynomial(3) * (self.st0_next() - self.st3()),
            self.indicator_polynomial(4) * (self.st0_next() - self.st4()),
            self.indicator_polynomial(5) * (self.st0_next() - self.st5()),
            self.indicator_polynomial(6) * (self.st0_next() - self.st6()),
            self.indicator_polynomial(7) * (self.st0_next() - self.st7()),
            self.indicator_polynomial(8) * (self.st0_next() - self.st8()),
            self.indicator_polynomial(9) * (self.st0_next() - self.st9()),
            self.indicator_polynomial(10) * (self.st0_next() - self.st10()),
            self.indicator_polynomial(11) * (self.st0_next() - self.st11()),
            self.indicator_polynomial(12) * (self.st0_next() - self.st12()),
            self.indicator_polynomial(13) * (self.st0_next() - self.st13()),
            self.indicator_polynomial(14) * (self.st0_next() - self.st14()),
            self.indicator_polynomial(15) * (self.st0_next() - self.st15()),
        ];
        [
            specific_constraints,
            self.decompose_arg(),
            self.step_2(),
            self.grow_stack(),
            self.keep_ram(),
        ]
        .concat()
    }

    pub fn instruction_swap(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let specific_constraints = vec![
            self.indicator_polynomial(0),
            self.indicator_polynomial(1) * (self.st1_next() - self.st0()),
            self.indicator_polynomial(2) * (self.st2_next() - self.st0()),
            self.indicator_polynomial(3) * (self.st3_next() - self.st0()),
            self.indicator_polynomial(4) * (self.st4_next() - self.st0()),
            self.indicator_polynomial(5) * (self.st5_next() - self.st0()),
            self.indicator_polynomial(6) * (self.st6_next() - self.st0()),
            self.indicator_polynomial(7) * (self.st7_next() - self.st0()),
            self.indicator_polynomial(8) * (self.st8_next() - self.st0()),
            self.indicator_polynomial(9) * (self.st9_next() - self.st0()),
            self.indicator_polynomial(10) * (self.st10_next() - self.st0()),
            self.indicator_polynomial(11) * (self.st11_next() - self.st0()),
            self.indicator_polynomial(12) * (self.st12_next() - self.st0()),
            self.indicator_polynomial(13) * (self.st13_next() - self.st0()),
            self.indicator_polynomial(14) * (self.st14_next() - self.st0()),
            self.indicator_polynomial(15) * (self.st15_next() - self.st0()),
            self.indicator_polynomial(1) * (self.st0_next() - self.st1()),
            self.indicator_polynomial(2) * (self.st0_next() - self.st2()),
            self.indicator_polynomial(3) * (self.st0_next() - self.st3()),
            self.indicator_polynomial(4) * (self.st0_next() - self.st4()),
            self.indicator_polynomial(5) * (self.st0_next() - self.st5()),
            self.indicator_polynomial(6) * (self.st0_next() - self.st6()),
            self.indicator_polynomial(7) * (self.st0_next() - self.st7()),
            self.indicator_polynomial(8) * (self.st0_next() - self.st8()),
            self.indicator_polynomial(9) * (self.st0_next() - self.st9()),
            self.indicator_polynomial(10) * (self.st0_next() - self.st10()),
            self.indicator_polynomial(11) * (self.st0_next() - self.st11()),
            self.indicator_polynomial(12) * (self.st0_next() - self.st12()),
            self.indicator_polynomial(13) * (self.st0_next() - self.st13()),
            self.indicator_polynomial(14) * (self.st0_next() - self.st14()),
            self.indicator_polynomial(15) * (self.st0_next() - self.st15()),
            (self.one() - self.indicator_polynomial(1)) * (self.st1_next() - self.st1()),
            (self.one() - self.indicator_polynomial(2)) * (self.st2_next() - self.st2()),
            (self.one() - self.indicator_polynomial(3)) * (self.st3_next() - self.st3()),
            (self.one() - self.indicator_polynomial(4)) * (self.st4_next() - self.st4()),
            (self.one() - self.indicator_polynomial(5)) * (self.st5_next() - self.st5()),
            (self.one() - self.indicator_polynomial(6)) * (self.st6_next() - self.st6()),
            (self.one() - self.indicator_polynomial(7)) * (self.st7_next() - self.st7()),
            (self.one() - self.indicator_polynomial(8)) * (self.st8_next() - self.st8()),
            (self.one() - self.indicator_polynomial(9)) * (self.st9_next() - self.st9()),
            (self.one() - self.indicator_polynomial(10)) * (self.st10_next() - self.st10()),
            (self.one() - self.indicator_polynomial(11)) * (self.st11_next() - self.st11()),
            (self.one() - self.indicator_polynomial(12)) * (self.st12_next() - self.st12()),
            (self.one() - self.indicator_polynomial(13)) * (self.st13_next() - self.st13()),
            (self.one() - self.indicator_polynomial(14)) * (self.st14_next() - self.st14()),
            (self.one() - self.indicator_polynomial(15)) * (self.st15_next() - self.st15()),
            self.osv_next() - self.osv(),
            self.osp_next() - self.osp(),
        ];
        [
            specific_constraints,
            self.decompose_arg(),
            self.step_2(),
            self.keep_ram(),
        ]
        .concat()
    }

    pub fn instruction_nop(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        [self.step_1(), self.keep_stack(), self.keep_ram()].concat()
    }

    pub fn instruction_skiz(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        // The next instruction nia is decomposed into helper variables hv.
        let nia_decomposes_to_hvs = self.nia() - (self.hv0() + self.two() * self.hv1());

        // The relevant helper variable hv1 is either 0 or 1.
        // Here, hv0 == 1 means that nia takes an argument.
        let hv0_is_0_or_1 = self.hv0() * (self.hv0() - self.one());

        // If `st0` is non-zero, register `ip` is incremented by 1.
        // If `st0` is 0 and `nia` takes no argument, register `ip` is incremented by 2.
        // If `st0` is 0 and `nia` takes an argument, register `ip` is incremented by 3.
        //
        // Written as Disjunctive Normal Form, the last constraint can be expressed as:
        // 6. (Register `st0` is 0 or `ip` is incremented by 1), and
        // (`st0` has a multiplicative inverse or `hv` is 1 or `ip` is incremented by 2), and
        // (`st0` has a multiplicative inverse or `hv0` is 0 or `ip` is incremented by 3).
        let ip_case_1 = (self.ip_next() - (self.ip() + self.one())) * self.st0();
        let ip_case_2 = (self.ip_next() - (self.ip() + self.two()))
            * (self.st0() * self.hv2() - self.one())
            * (self.hv0() - self.one());
        let ip_case_3 = (self.ip_next() - (self.ip() + self.constant(3)))
            * (self.st0() * self.hv2() - self.one())
            * self.hv0();
        let ip_incr_by_1_or_2_or_3 = ip_case_1 + ip_case_2 + ip_case_3;

        let specific_constraints =
            vec![nia_decomposes_to_hvs, hv0_is_0_or_1, ip_incr_by_1_or_2_or_3];
        [
            specific_constraints,
            self.keep_jump_stack(),
            self.shrink_stack(),
            self.keep_ram(),
        ]
        .concat()
    }

    pub fn instruction_call(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        // The jump stack pointer jsp is incremented by 1.
        let jsp_incr_1 = self.jsp_next() - (self.jsp() + self.one());

        // The jump's origin jso is set to the current instruction pointer ip plus 2.
        let jso_becomes_ip_plus_2 = self.jso_next() - (self.ip() + self.two());

        // The jump's destination jsd is set to the instruction's argument.
        let jsd_becomes_nia = self.jsd_next() - self.nia();

        // The instruction pointer ip is set to the instruction's argument.
        let ip_becomes_nia = self.ip_next() - self.nia();

        let specific_constraints = vec![
            jsp_incr_1,
            jso_becomes_ip_plus_2,
            jsd_becomes_nia,
            ip_becomes_nia,
        ];
        [specific_constraints, self.keep_stack(), self.keep_ram()].concat()
    }

    pub fn instruction_return(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        // The jump stack pointer jsp is decremented by 1.
        let jsp_incr_1 = self.jsp_next() - (self.jsp() - self.one());

        // The instruction pointer ip is set to the last call's origin jso.
        let ip_becomes_jso = self.ip_next() - self.jso();

        let specific_constraints = vec![jsp_incr_1, ip_becomes_jso];
        [specific_constraints, self.keep_stack(), self.keep_ram()].concat()
    }

    pub fn instruction_recurse(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        // The instruction pointer ip is set to the last jump's destination jsd.
        let ip_becomes_jsd = self.ip_next() - self.jsd();
        let specific_constraints = vec![ip_becomes_jsd];
        [
            specific_constraints,
            self.keep_jump_stack(),
            self.keep_stack(),
            self.keep_ram(),
        ]
        .concat()
    }

    pub fn instruction_assert(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        // The current top of the stack st0 is 1.
        let st_0_is_1 = self.st0() - self.one();

        let specific_constraints = vec![st_0_is_1];
        [
            specific_constraints,
            self.step_1(),
            self.shrink_stack(),
            self.keep_ram(),
        ]
        .concat()
    }

    pub fn instruction_halt(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        // The instruction executed in the following step is instruction halt.
        let halt_is_followed_by_halt = self.ci_next() - self.ci();

        let specific_constraints = vec![halt_is_followed_by_halt];
        [
            specific_constraints,
            self.step_1(),
            self.keep_stack(),
            self.keep_ram(),
        ]
        .concat()
    }

    pub fn instruction_read_mem(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        // the RAM pointer is overwritten with st1
        let update_ramp = self.ramp_next() - self.st1();

        // The top of the stack is overwritten with the RAM value.
        let st0_becomes_ramv = self.st0_next() - self.ramv_next();

        let specific_constraints = vec![update_ramp, st0_becomes_ramv];
        [specific_constraints, self.step_1(), self.unop()].concat()
    }

    pub fn instruction_write_mem(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        // the RAM pointer is overwritten with st1
        let update_ramp = self.ramp_next() - self.st1();

        // The RAM value is overwritten with the top of the stack.
        let ramv_becomes_st0 = self.ramv_next() - self.st0();

        let specific_constraints = vec![update_ramp, ramv_becomes_st0];
        [specific_constraints, self.step_1(), self.keep_stack()].concat()
    }

    /// Two Evaluation Arguments with the Hash Table guarantee correct transition.
    pub fn instruction_hash(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        [
            self.step_1(),
            self.stack_remains_and_top_ten_elements_unconstrained(),
            self.keep_ram(),
        ]
        .concat()
    }

    /// Recall that in a Merkle tree, the indices of left (respectively right)
    /// leafs have 0 (respectively 1) as their least significant bit. The first
    /// two polynomials achieve that helper variable hv0 holds the result of
    /// st10 mod 2. The second polynomial sets the new value of st10 to st10 div 2.
    pub fn instruction_divine_sibling(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        // Helper variable hv0 is either 0 or 1.
        let hv0_is_0_or_1 = self.hv0() * (self.hv0() - self.one());

        // The 11th stack register is shifted by 1 bit to the right.
        let st10_is_shifted_1_bit_right = self.st10_next() * self.two() + self.hv0() - self.st10();

        // The second pentuplet either stays where it is, or is moved to the top
        let maybe_move_st5 = (self.one() - self.hv0()) * (self.st5() - self.st0_next())
            + self.hv0() * (self.st5() - self.st5_next());
        let maybe_move_st6 = (self.one() - self.hv0()) * (self.st6() - self.st1_next())
            + self.hv0() * (self.st6() - self.st6_next());
        let maybe_move_st7 = (self.one() - self.hv0()) * (self.st7() - self.st2_next())
            + self.hv0() * (self.st7() - self.st7_next());
        let maybe_move_st8 = (self.one() - self.hv0()) * (self.st8() - self.st3_next())
            + self.hv0() * (self.st8() - self.st8_next());
        let maybe_move_st9 = (self.one() - self.hv0()) * (self.st9() - self.st4_next())
            + self.hv0() * (self.st9() - self.st9_next());

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
            self.stack_remains_and_top_eleven_elements_unconstrained(),
            self.step_1(),
            self.keep_ram(),
        ]
        .concat()
    }

    pub fn instruction_assert_vector(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let specific_constraints = vec![
            // Register st0 is equal to st5.
            self.st5() - self.st0(),
            // Register st1 is equal to st6.
            self.st6() - self.st1(),
            // and so on
            self.st7() - self.st2(),
            self.st8() - self.st3(),
            self.st9() - self.st4(),
        ];
        [
            specific_constraints,
            self.step_1(),
            self.keep_stack(),
            self.keep_ram(),
        ]
        .concat()
    }

    /// The sum of the top two stack elements is moved into the top of the stack.
    ///
    /// $st0' - (st0 + st1) = 0$
    pub fn instruction_add(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let specific_constraints = vec![self.st0_next() - (self.st0() + self.st1())];
        [
            specific_constraints,
            self.step_1(),
            self.binop(),
            self.keep_ram(),
        ]
        .concat()
    }

    /// The product of the top two stack elements is moved into the top of the stack.
    ///
    /// $st0' - (st0 * st1) = 0$
    pub fn instruction_mul(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let specific_constraints = vec![self.st0_next() - (self.st0() * self.st1())];
        [
            specific_constraints,
            self.step_1(),
            self.binop(),
            self.keep_ram(),
        ]
        .concat()
    }

    /// The top of the stack's inverse is moved into the top of the stack.
    ///
    /// $st0'·st0 - 1 = 0$
    pub fn instruction_invert(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let specific_constraints = vec![self.st0_next() * self.st0() - self.one()];
        [
            specific_constraints,
            self.step_1(),
            self.unop(),
            self.keep_ram(),
        ]
        .concat()
    }

    pub fn instruction_split(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let two_pow_32 = self.constant_b(BFieldElement::new(1_u64 << 32));

        // The top of the stack is decomposed as 32-bit chunks into the stack's top-most elements.
        //
        // $st0 - (2^32·st0' + st1') = 0$
        let st0_decomposes_to_two_32_bit_chunks =
            self.st0() - (two_pow_32.clone() * self.st0_next() + self.st1_next());

        // Helper variable `hv0` = 0 if either
        // 1. `hv0` is the difference between (2^32 - 1) and the high 32 bits (`st0'`), or
        // 1. the low 32 bits (`st1'`) are 0.
        //
        // st1'·(hv0·(st0' - (2^32 - 1)) - 1)
        //   lo·(hv0·(hi - 0xffff_ffff)) - 1)
        let hv0_holds_inverse_of_chunk_difference_or_low_bits_are_0 = {
            let hv0 = self.hv0();
            let hi = self.st0_next();
            let lo = self.st1_next();
            let ffff_ffff = two_pow_32 - self.one();

            lo * (hv0 * (hi - ffff_ffff) - self.one())
        };

        let specific_constraints = vec![
            st0_decomposes_to_two_32_bit_chunks,
            hv0_holds_inverse_of_chunk_difference_or_low_bits_are_0,
        ];
        [
            specific_constraints,
            self.grow_stack_and_top_two_elements_unconstrained(),
            self.step_1(),
            self.keep_ram(),
        ]
        .concat()
    }

    pub fn instruction_eq(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        // Helper variable hv0 is the inverse of the difference of the stack's two top-most elements or 0.
        //
        // $ hv0·(hv0·(st1 - st0) - 1) = 0 $
        let hv0_is_inverse_of_diff_or_hv0_is_0 =
            self.hv0() * (self.hv0() * (self.st1() - self.st0()) - self.one());

        // Helper variable hv0 is the inverse of the difference of the stack's two top-most elements or the difference is 0.
        //
        // $ (st1 - st0)·(hv0·(st1 - st0) - 1) = 0 $
        let hv0_is_inverse_of_diff_or_diff_is_0 =
            (self.st1() - self.st0()) * (self.hv0() * (self.st1() - self.st0()) - self.one());

        // The new top of the stack is 1 if the difference between the stack's two top-most elements is not invertible, 0 otherwise.
        //
        // $ st0' - (1 - hv0·(st1 - st0)) = 0 $
        let st0_becomes_1_if_diff_is_not_invertible =
            self.st0_next() - (self.one() - self.hv0() * (self.st1() - self.st0()));

        let specific_constraints = vec![
            hv0_is_inverse_of_diff_or_hv0_is_0,
            hv0_is_inverse_of_diff_or_diff_is_0,
            st0_becomes_1_if_diff_is_not_invertible,
        ];
        [
            specific_constraints,
            self.step_1(),
            self.binop(),
            self.keep_ram(),
        ]
        .concat()
    }

    /// 1. The lsb is a bit
    /// 2. The operand decomposes into right-shifted operand and the lsb
    pub fn instruction_lsb(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let operand = self.current_base_row_variables[ST0.master_base_table_index()].clone();
        let shifted_operand = self.next_base_row_variables[ST1.master_base_table_index()].clone();
        let lsb = self.next_base_row_variables[ST0.master_base_table_index()].clone();

        let lsb_is_a_bit = lsb.clone() * (lsb.clone() - self.one());
        let correct_decomposition = self.two() * shifted_operand + lsb - operand;

        let specific_constraints = vec![lsb_is_a_bit, correct_decomposition];
        [
            specific_constraints,
            self.step_1(),
            self.grow_stack_and_top_two_elements_unconstrained(),
            self.keep_ram(),
        ]
        .concat()
    }

    pub fn instruction_xxadd(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        // The result of adding st0 to st3 is moved into st0.
        let st0_becomes_st0_plus_st3 = self.st0_next() - (self.st0() + self.st3());

        // The result of adding st1 to st4 is moved into st1.
        let st1_becomes_st1_plus_st4 = self.st1_next() - (self.st1() + self.st4());

        // The result of adding st2 to st5 is moved into st2.
        let st2_becomes_st2_plus_st5 = self.st2_next() - (self.st2() + self.st5());

        let specific_constraints = vec![
            st0_becomes_st0_plus_st3,
            st1_becomes_st1_plus_st4,
            st2_becomes_st2_plus_st5,
        ];
        [
            specific_constraints,
            self.stack_remains_and_top_three_elements_unconstrained(),
            self.step_1(),
            self.keep_ram(),
        ]
        .concat()
    }

    pub fn instruction_xxmul(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        // The coefficient of x^0 of multiplying the two X-Field elements on the stack is moved into st0.
        //
        // $st0' - (st0·st3 - st2·st4 - st1·st5)$
        let st0_becomes_coefficient_0 = self.st0_next()
            - (self.st0() * self.st3() - self.st2() * self.st4() - self.st1() * self.st5());

        // The coefficient of x^1 of multiplying the two X-Field elements on the stack is moved into st1.
        //
        // st1' - (st1·st3 + st0·st4 - st2·st5 + st2·st4 + st1·st5)
        let st1_becomes_coefficient_1 = self.st1_next()
            - (self.st1() * self.st3() + self.st0() * self.st4() - self.st2() * self.st5()
                + self.st2() * self.st4()
                + self.st1() * self.st5());

        // The coefficient of x^2 of multiplying the two X-Field elements on the stack is moved into st2.
        //
        // st2' - (st2·st3 + st1·st4 + st0·st5 + st2·st5)
        let st2_becomes_coefficient_2 = self.st2_next()
            - (self.st2() * self.st3()
                + self.st1() * self.st4()
                + self.st0() * self.st5()
                + self.st2() * self.st5());

        let specific_constraints = vec![
            st0_becomes_coefficient_0,
            st1_becomes_coefficient_1,
            st2_becomes_coefficient_2,
        ];
        [
            specific_constraints,
            self.stack_remains_and_top_three_elements_unconstrained(),
            self.step_1(),
            self.keep_ram(),
        ]
        .concat()
    }

    pub fn instruction_xinv(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        // The coefficient of x^0 of multiplying X-Field element on top of the current stack and on top of the next stack is 1.
        //
        // $st0·st0' - st2·st1' - st1·st2' - 1 = 0$
        let first_coefficient_of_product_of_element_and_inverse_is_1 = self.st0() * self.st0_next()
            - self.st2() * self.st1_next()
            - self.st1() * self.st2_next()
            - self.one();

        // The coefficient of x^1 of multiplying X-Field element on top of the current stack and on top of the next stack is 0.
        //
        // $st1·st0' + st0·st1' - st2·st2' + st2·st1' + st1·st2' = 0$
        let second_coefficient_of_product_of_element_and_inverse_is_0 =
            self.st1() * self.st0_next() + self.st0() * self.st1_next()
                - self.st2() * self.st2_next()
                + self.st2() * self.st1_next()
                + self.st1() * self.st2_next();

        // The coefficient of x^2 of multiplying X-Field element on top of the current stack and on top of the next stack is 0.
        //
        // $st2·st0' + st1·st1' + st0·st2' + st2·st2' = 0$
        let third_coefficient_of_product_of_element_and_inverse_is_0 = self.st2() * self.st0_next()
            + self.st1() * self.st1_next()
            + self.st0() * self.st2_next()
            + self.st2() * self.st2_next();

        let specific_constraints = vec![
            first_coefficient_of_product_of_element_and_inverse_is_1,
            second_coefficient_of_product_of_element_and_inverse_is_0,
            third_coefficient_of_product_of_element_and_inverse_is_0,
        ];
        [
            specific_constraints,
            self.stack_remains_and_top_three_elements_unconstrained(),
            self.step_1(),
            self.keep_ram(),
        ]
        .concat()
    }

    pub fn instruction_xbmul(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        // The result of multiplying the top of the stack with the X-Field element's coefficient for x^0 is moved into st0.
        //
        // st0' - st0·st1
        let first_coeff_scalar_multiplication = self.st0_next() - self.st0() * self.st1();

        // The result of multiplying the top of the stack with the X-Field element's coefficient for x^1 is moved into st1.
        //
        // st1' - st0·st2
        let secnd_coeff_scalar_multiplication = self.st1_next() - self.st0() * self.st2();

        // The result of multiplying the top of the stack with the X-Field element's coefficient for x^2 is moved into st2.
        //
        // st2' - st0·st3
        let third_coeff_scalar_multiplication = self.st2_next() - self.st0() * self.st3();

        let specific_constraints = vec![
            first_coeff_scalar_multiplication,
            secnd_coeff_scalar_multiplication,
            third_coeff_scalar_multiplication,
        ];
        [
            specific_constraints,
            self.stack_shrinks_and_top_three_elements_unconstrained(),
            self.step_1(),
            self.keep_ram(),
        ]
        .concat()
    }

    /// This instruction has no additional transition constraints.
    ///
    /// An Evaluation Argument with the list of input symbols guarantees correct transition.
    pub fn instruction_read_io(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        [self.step_1(), self.grow_stack(), self.keep_ram()].concat()
    }

    /// This instruction has no additional transition constraints.
    ///
    /// An Evaluation Argument with the list of output symbols guarantees correct transition.
    pub fn instruction_write_io(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        [self.step_1(), self.shrink_stack(), self.keep_ram()].concat()
    }

    pub fn zero(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.zero.to_owned()
    }

    pub fn one(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.one.to_owned()
    }

    pub fn two(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.two.to_owned()
    }

    pub fn constant(
        &self,
        constant: u32,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.circuit_builder.b_constant(constant.into())
    }

    pub fn constant_b(
        &self,
        constant: BFieldElement,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.circuit_builder.b_constant(constant)
    }

    pub fn clk(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[CLK.master_base_table_index()].clone()
    }

    pub fn ip(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[IP.master_base_table_index()].clone()
    }

    pub fn ci(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[CI.master_base_table_index()].clone()
    }

    pub fn nia(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[NIA.master_base_table_index()].clone()
    }

    pub fn ib0(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[IB0.master_base_table_index()].clone()
    }

    pub fn ib1(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[IB1.master_base_table_index()].clone()
    }

    pub fn ib2(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[IB2.master_base_table_index()].clone()
    }

    pub fn ib3(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[IB3.master_base_table_index()].clone()
    }

    pub fn ib4(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[IB4.master_base_table_index()].clone()
    }

    pub fn ib5(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[IB5.master_base_table_index()].clone()
    }

    pub fn ib6(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[IB6.master_base_table_index()].clone()
    }

    pub fn jsp(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[JSP.master_base_table_index()].clone()
    }

    pub fn jsd(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[JSD.master_base_table_index()].clone()
    }

    pub fn jso(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[JSO.master_base_table_index()].clone()
    }

    pub fn st0(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[ST0.master_base_table_index()].clone()
    }

    pub fn st1(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[ST1.master_base_table_index()].clone()
    }

    pub fn st2(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[ST2.master_base_table_index()].clone()
    }

    pub fn st3(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[ST3.master_base_table_index()].clone()
    }

    pub fn st4(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[ST4.master_base_table_index()].clone()
    }

    pub fn st5(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[ST5.master_base_table_index()].clone()
    }

    pub fn st6(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[ST6.master_base_table_index()].clone()
    }

    pub fn st7(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[ST7.master_base_table_index()].clone()
    }

    pub fn st8(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[ST8.master_base_table_index()].clone()
    }

    pub fn st9(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[ST9.master_base_table_index()].clone()
    }

    pub fn st10(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[ST10.master_base_table_index()].clone()
    }

    pub fn st11(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[ST11.master_base_table_index()].clone()
    }

    pub fn st12(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[ST12.master_base_table_index()].clone()
    }

    pub fn st13(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[ST13.master_base_table_index()].clone()
    }

    pub fn st14(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[ST14.master_base_table_index()].clone()
    }

    pub fn st15(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[ST15.master_base_table_index()].clone()
    }

    pub fn osp(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[OSP.master_base_table_index()].clone()
    }

    pub fn osv(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[OSV.master_base_table_index()].clone()
    }

    pub fn hv0(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[HV0.master_base_table_index()].clone()
    }

    pub fn hv1(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[HV1.master_base_table_index()].clone()
    }

    pub fn hv2(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[HV2.master_base_table_index()].clone()
    }

    pub fn hv3(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[HV3.master_base_table_index()].clone()
    }

    pub fn ramp(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[RAMP.master_base_table_index()].clone()
    }

    pub fn previous_instruction(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[PreviousInstruction.master_base_table_index()].clone()
    }

    pub fn ramv(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[RAMV.master_base_table_index()].clone()
    }

    pub fn is_padding(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[IsPadding.master_base_table_index()].clone()
    }

    pub fn cjd(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[ClockJumpDifference.master_base_table_index()].clone()
    }

    pub fn invm(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[ClockJumpDifferenceInverse.master_base_table_index()]
            .clone()
    }

    pub fn invu(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_base_row_variables[UniqueClockJumpDiffDiffInverse.master_base_table_index()]
            .clone()
    }

    pub fn rer(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_ext_row_variables[SelectedClockCyclesEvalArg.master_ext_table_index()].clone()
    }

    pub fn reu(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_ext_row_variables[UniqueClockJumpDifferencesEvalArg.master_ext_table_index()]
            .clone()
    }

    pub fn rpm(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_ext_row_variables[AllClockJumpDifferencesPermArg.master_ext_table_index()]
            .clone()
    }

    pub fn running_evaluation_standard_input(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_ext_row_variables[InputTableEvalArg.master_ext_table_index()].clone()
    }
    pub fn running_evaluation_standard_output(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_ext_row_variables[OutputTableEvalArg.master_ext_table_index()].clone()
    }
    pub fn running_product_instruction_table(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_ext_row_variables[InstructionTablePermArg.master_ext_table_index()].clone()
    }
    pub fn running_product_op_stack_table(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_ext_row_variables[OpStackTablePermArg.master_ext_table_index()].clone()
    }
    pub fn running_product_ram_table(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_ext_row_variables[RamTablePermArg.master_ext_table_index()].clone()
    }
    pub fn running_product_jump_stack_table(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_ext_row_variables[JumpStackTablePermArg.master_ext_table_index()].clone()
    }
    pub fn running_evaluation_to_hash_table(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_ext_row_variables[ToHashTableEvalArg.master_ext_table_index()].clone()
    }
    pub fn running_evaluation_from_hash_table(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.current_ext_row_variables[FromHashTableEvalArg.master_ext_table_index()].clone()
    }

    // Property: All polynomial variables that contain '_next' have the same
    // variable position / value as the one without '_next', +/- NUM_COLUMNS.
    pub fn clk_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[CLK.master_base_table_index()].clone()
    }

    pub fn ip_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[IP.master_base_table_index()].clone()
    }

    pub fn ci_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[CI.master_base_table_index()].clone()
    }

    pub fn nia_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[NIA.master_base_table_index()].clone()
    }

    pub fn ib0_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[IB0.master_base_table_index()].clone()
    }
    pub fn ib1_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[IB1.master_base_table_index()].clone()
    }
    pub fn ib2_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[IB2.master_base_table_index()].clone()
    }
    pub fn ib3_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[IB3.master_base_table_index()].clone()
    }
    pub fn ib4_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[IB4.master_base_table_index()].clone()
    }
    pub fn ib5_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[IB5.master_base_table_index()].clone()
    }
    pub fn ib6_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[IB6.master_base_table_index()].clone()
    }

    pub fn jsp_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[JSP.master_base_table_index()].clone()
    }

    pub fn jsd_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[JSD.master_base_table_index()].clone()
    }

    pub fn jso_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[JSO.master_base_table_index()].clone()
    }

    pub fn st0_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[ST0.master_base_table_index()].clone()
    }

    pub fn st1_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[ST1.master_base_table_index()].clone()
    }

    pub fn st2_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[ST2.master_base_table_index()].clone()
    }

    pub fn st3_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[ST3.master_base_table_index()].clone()
    }

    pub fn st4_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[ST4.master_base_table_index()].clone()
    }

    pub fn st5_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[ST5.master_base_table_index()].clone()
    }

    pub fn st6_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[ST6.master_base_table_index()].clone()
    }

    pub fn st7_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[ST7.master_base_table_index()].clone()
    }

    pub fn st8_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[ST8.master_base_table_index()].clone()
    }

    pub fn st9_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[ST9.master_base_table_index()].clone()
    }

    pub fn st10_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[ST10.master_base_table_index()].clone()
    }

    pub fn st11_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[ST11.master_base_table_index()].clone()
    }

    pub fn st12_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[ST12.master_base_table_index()].clone()
    }

    pub fn st13_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[ST13.master_base_table_index()].clone()
    }

    pub fn st14_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[ST14.master_base_table_index()].clone()
    }

    pub fn st15_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[ST15.master_base_table_index()].clone()
    }

    pub fn osp_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[OSP.master_base_table_index()].clone()
    }

    pub fn osv_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[OSV.master_base_table_index()].clone()
    }

    pub fn ramp_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[RAMP.master_base_table_index()].clone()
    }

    pub fn previous_instruction_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[PreviousInstruction.master_base_table_index()].clone()
    }

    pub fn ramv_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[RAMV.master_base_table_index()].clone()
    }

    pub fn is_padding_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[IsPadding.master_base_table_index()].clone()
    }

    pub fn cjd_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[ClockJumpDifference.master_base_table_index()].clone()
    }

    pub fn invm_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[ClockJumpDifferenceInverse.master_base_table_index()].clone()
    }

    pub fn invu_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_base_row_variables[UniqueClockJumpDiffDiffInverse.master_base_table_index()]
            .clone()
    }

    pub fn rer_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_ext_row_variables[SelectedClockCyclesEvalArg.master_ext_table_index()].clone()
    }

    pub fn reu_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_ext_row_variables[UniqueClockJumpDifferencesEvalArg.master_ext_table_index()]
            .clone()
    }

    pub fn rpm_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_ext_row_variables[AllClockJumpDifferencesPermArg.master_ext_table_index()].clone()
    }

    pub fn running_evaluation_standard_input_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_ext_row_variables[InputTableEvalArg.master_ext_table_index()].clone()
    }
    pub fn running_evaluation_standard_output_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_ext_row_variables[OutputTableEvalArg.master_ext_table_index()].clone()
    }
    pub fn running_product_instruction_table_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_ext_row_variables[InstructionTablePermArg.master_ext_table_index()].clone()
    }
    pub fn running_product_op_stack_table_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_ext_row_variables[OpStackTablePermArg.master_ext_table_index()].clone()
    }
    pub fn running_product_ram_table_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_ext_row_variables[RamTablePermArg.master_ext_table_index()].clone()
    }
    pub fn running_product_jump_stack_table_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_ext_row_variables[JumpStackTablePermArg.master_ext_table_index()].clone()
    }
    pub fn running_evaluation_to_hash_table_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_ext_row_variables[ToHashTableEvalArg.master_ext_table_index()].clone()
    }
    pub fn running_evaluation_from_hash_table_next(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.next_ext_row_variables[FromHashTableEvalArg.master_ext_table_index()].clone()
    }

    pub fn decompose_arg(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let hv0_is_a_bit = self.hv0() * (self.hv0() - self.one());
        let hv1_is_a_bit = self.hv1() * (self.hv1() - self.one());
        let hv2_is_a_bit = self.hv2() * (self.hv2() - self.one());
        let hv3_is_a_bit = self.hv3() * (self.hv3() - self.one());
        let helper_variables_are_binary_decomposition_of_nia = self.nia()
            - self.constant(8) * self.hv3()
            - self.constant(4) * self.hv2()
            - self.constant(2) * self.hv1()
            - self.hv0();
        vec![
            hv0_is_a_bit,
            hv1_is_a_bit,
            hv2_is_a_bit,
            hv3_is_a_bit,
            helper_variables_are_binary_decomposition_of_nia,
        ]
    }

    pub fn keep_jump_stack(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let jsp_does_not_change = self.jsp_next() - self.jsp();
        let jso_does_not_change = self.jso_next() - self.jso();
        let jsd_does_not_change = self.jsd_next() - self.jsd();
        vec![
            jsp_does_not_change,
            jso_does_not_change,
            jsd_does_not_change,
        ]
    }

    pub fn step_1(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let instruction_pointer_increases_by_one = self.ip_next() - self.ip() - self.one();
        let specific_constraints = vec![instruction_pointer_increases_by_one];
        [specific_constraints, self.keep_jump_stack()].concat()
    }

    pub fn step_2(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let instruction_pointer_increases_by_two = self.ip_next() - self.ip() - self.two();
        let specific_constraints = vec![instruction_pointer_increases_by_two];
        [specific_constraints, self.keep_jump_stack()].concat()
    }

    pub fn grow_stack_and_top_two_elements_unconstrained(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        vec![
            // The stack element in st1 is moved into st2.
            self.st2_next() - self.st1(),
            // And so on...
            self.st3_next() - self.st2(),
            self.st4_next() - self.st3(),
            self.st5_next() - self.st4(),
            self.st6_next() - self.st5(),
            self.st7_next() - self.st6(),
            self.st8_next() - self.st7(),
            self.st9_next() - self.st8(),
            self.st10_next() - self.st9(),
            self.st11_next() - self.st10(),
            self.st12_next() - self.st11(),
            self.st13_next() - self.st12(),
            self.st14_next() - self.st13(),
            self.st15_next() - self.st14(),
            // The stack element in st15 is moved to the top of OpStack underflow, i.e., osv.
            self.osv_next() - self.st15(),
            // The OpStack pointer is incremented by 1.
            self.osp_next() - (self.osp() + self.one()),
        ]
    }

    pub fn grow_stack(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let specific_constraints = vec![
            // The stack element in st0 is moved into st1.
            self.st1_next() - self.st0(),
        ];
        [
            specific_constraints,
            self.grow_stack_and_top_two_elements_unconstrained(),
        ]
        .concat()
    }

    pub fn stack_shrinks_and_top_three_elements_unconstrained(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        vec![
            // The stack element in st4 is moved into st3.
            self.st3_next() - self.st4(),
            // The stack element in st5 is moved into st4.
            self.st4_next() - self.st5(),
            // And so on...
            self.st5_next() - self.st6(),
            self.st6_next() - self.st7(),
            self.st7_next() - self.st8(),
            self.st8_next() - self.st9(),
            self.st9_next() - self.st10(),
            self.st10_next() - self.st11(),
            self.st11_next() - self.st12(),
            self.st12_next() - self.st13(),
            self.st13_next() - self.st14(),
            self.st14_next() - self.st15(),
            // The stack element at the top of OpStack underflow, i.e., osv, is moved into st15.
            self.st15_next() - self.osv(),
            // The OpStack pointer, osp, is decremented by 1.
            self.osp_next() - (self.osp() - self.one()),
            // The helper variable register hv3 holds the inverse of (osp - 16).
            (self.osp() - self.constant(16)) * self.hv3() - self.one(),
        ]
    }

    pub fn binop(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let specific_constraints = vec![
            // The stack element in st2 is moved into st1.
            self.st1_next() - self.st2(),
            // The stack element in st3 is moved into st2.
            self.st2_next() - self.st3(),
        ];
        [
            specific_constraints,
            self.stack_shrinks_and_top_three_elements_unconstrained(),
        ]
        .concat()
    }

    pub fn shrink_stack(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let specific_constrants = vec![self.st0_next() - self.st1()];
        [specific_constrants, self.binop()].concat()
    }

    pub fn stack_remains_and_top_eleven_elements_unconstrained(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        vec![
            self.st11_next() - self.st11(),
            self.st12_next() - self.st12(),
            self.st13_next() - self.st13(),
            self.st14_next() - self.st14(),
            self.st15_next() - self.st15(),
            // The top of the OpStack underflow, i.e., osv, does not change.
            self.osv_next() - self.osv(),
            // The OpStack pointer, osp, does not change.
            self.osp_next() - self.osp(),
        ]
    }

    pub fn stack_remains_and_top_ten_elements_unconstrained(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let specific_constraints = vec![self.st10_next() - self.st10()];
        [
            specific_constraints,
            self.stack_remains_and_top_eleven_elements_unconstrained(),
        ]
        .concat()
    }

    pub fn stack_remains_and_top_three_elements_unconstrained(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let specific_constraints = vec![
            self.st3_next() - self.st3(),
            self.st4_next() - self.st4(),
            self.st5_next() - self.st5(),
            self.st6_next() - self.st6(),
            self.st7_next() - self.st7(),
            self.st8_next() - self.st8(),
            self.st9_next() - self.st9(),
        ];
        [
            specific_constraints,
            self.stack_remains_and_top_ten_elements_unconstrained(),
        ]
        .concat()
    }

    pub fn unop(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let specific_constraints = vec![
            // The stack element in st1 does not change.
            self.st1_next() - self.st1(),
            // The stack element in st2 does not change.
            self.st2_next() - self.st2(),
        ];
        [
            specific_constraints,
            self.stack_remains_and_top_three_elements_unconstrained(),
        ]
        .concat()
    }

    pub fn keep_stack(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let specific_constraints = vec![self.st0_next() - self.st0()];
        [specific_constraints, self.unop()].concat()
    }

    pub fn keep_ram(
        &self,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        vec![
            self.ramv_next() - self.ramv(),
            self.ramp_next() - self.ramp(),
        ]
    }

    pub fn running_evaluation_for_standard_input_updates_correctly(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        let indeterminate = self
            .circuit_builder
            .challenge(StandardInputEvalIndeterminate);
        let read_io_deselector =
            InstructionDeselectors::instruction_deselector(self, Instruction::ReadIo);
        let read_io_selector = self.ci() - self.constant_b(Instruction::ReadIo.opcode_b());
        let input_symbol = self.st0_next();
        let running_evaluation_updates = self.running_evaluation_standard_input_next()
            - indeterminate * self.running_evaluation_standard_input()
            - input_symbol;
        let running_evaluation_remains = self.running_evaluation_standard_input_next()
            - self.running_evaluation_standard_input();

        read_io_selector * running_evaluation_remains
            + read_io_deselector * running_evaluation_updates
    }

    pub fn running_product_for_instruction_table_updates_correctly(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        let indeterminate = self.circuit_builder.challenge(InstructionPermIndeterminate);
        let ip_weight = self.circuit_builder.challenge(InstructionTableIpWeight);
        let ci_weight = self
            .circuit_builder
            .challenge(InstructionTableCiProcessorWeight);
        let nia_weight = self.circuit_builder.challenge(InstructionTableNiaWeight);
        let compressed_row =
            ip_weight * self.ip_next() + ci_weight * self.ci_next() + nia_weight * self.nia_next();
        let running_product_updates = self.running_product_instruction_table_next()
            - self.running_product_instruction_table() * (indeterminate - compressed_row);
        let running_product_remains = self.running_product_instruction_table_next()
            - self.running_product_instruction_table();

        (self.one() - self.is_padding_next()) * running_product_updates
            + self.is_padding_next() * running_product_remains
    }

    pub fn running_evaluation_for_standard_output_updates_correctly(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        let indeterminate = self
            .circuit_builder
            .challenge(StandardOutputEvalIndeterminate);
        let write_io_deselector =
            InstructionDeselectors::instruction_deselector_next(self, Instruction::WriteIo);
        let write_io_selector = self.ci_next() - self.constant_b(Instruction::WriteIo.opcode_b());
        let output_symbol = self.st0_next();
        let running_evaluation_updates = self.running_evaluation_standard_output_next()
            - indeterminate * self.running_evaluation_standard_output()
            - output_symbol;
        let running_evaluation_remains = self.running_evaluation_standard_output_next()
            - self.running_evaluation_standard_output();

        write_io_selector * running_evaluation_remains
            + write_io_deselector * running_evaluation_updates
    }

    pub fn running_product_for_op_stack_table_updates_correctly(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        let indeterminate = self.circuit_builder.challenge(OpStackPermIndeterminate);
        let clk_weight = self.circuit_builder.challenge(OpStackTableClkWeight);
        let ib1_weight = self.circuit_builder.challenge(OpStackTableIb1Weight);
        let osp_weight = self.circuit_builder.challenge(OpStackTableOspWeight);
        let osv_weight = self.circuit_builder.challenge(OpStackTableOsvWeight);
        let compressed_row = clk_weight * self.clk_next()
            + ib1_weight * self.ib1_next()
            + osp_weight * self.osp_next()
            + osv_weight * self.osv_next();

        self.running_product_op_stack_table_next()
            - self.running_product_op_stack_table() * (indeterminate - compressed_row)
    }

    pub fn running_product_for_ram_table_updates_correctly(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        let indeterminate = self.circuit_builder.challenge(RamPermIndeterminate);
        let clk_weight = self.circuit_builder.challenge(RamTableClkWeight);
        let ramp_weight = self.circuit_builder.challenge(RamTableRampWeight);
        let ramv_weight = self.circuit_builder.challenge(RamTableRamvWeight);
        let previous_instruction_weight = self
            .circuit_builder
            .challenge(RamTablePreviousInstructionWeight);
        let compressed_row = clk_weight * self.clk_next()
            + ramp_weight * self.ramp_next()
            + ramv_weight * self.ramv_next()
            + previous_instruction_weight * self.previous_instruction_next();

        self.running_product_ram_table_next()
            - self.running_product_ram_table() * (indeterminate - compressed_row)
    }

    pub fn running_product_for_jump_stack_table_updates_correctly(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        let indeterminate = self.circuit_builder.challenge(JumpStackPermIndeterminate);
        let clk_weight = self.circuit_builder.challenge(JumpStackTableClkWeight);
        let ci_weight = self.circuit_builder.challenge(JumpStackTableCiWeight);
        let jsp_weight = self.circuit_builder.challenge(JumpStackTableJspWeight);
        let jso_weight = self.circuit_builder.challenge(JumpStackTableJsoWeight);
        let jsd_weight = self.circuit_builder.challenge(JumpStackTableJsdWeight);
        let compressed_row = clk_weight * self.clk_next()
            + ci_weight * self.ci_next()
            + jsp_weight * self.jsp_next()
            + jso_weight * self.jso_next()
            + jsd_weight * self.jsd_next();

        self.running_product_jump_stack_table_next()
            - self.running_product_jump_stack_table() * (indeterminate - compressed_row)
    }

    pub fn running_evaluation_to_hash_table_updates_correctly(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        let hash_deselector =
            InstructionDeselectors::instruction_deselector_next(self, Instruction::Hash);
        let hash_selector = self.ci_next() - self.constant_b(Instruction::Hash.opcode_b());

        let indeterminate = self.circuit_builder.challenge(ToHashTableEvalIndeterminate);

        let weights = [
            self.circuit_builder.challenge(HashTableStackInputWeight0),
            self.circuit_builder.challenge(HashTableStackInputWeight1),
            self.circuit_builder.challenge(HashTableStackInputWeight2),
            self.circuit_builder.challenge(HashTableStackInputWeight3),
            self.circuit_builder.challenge(HashTableStackInputWeight4),
            self.circuit_builder.challenge(HashTableStackInputWeight5),
            self.circuit_builder.challenge(HashTableStackInputWeight6),
            self.circuit_builder.challenge(HashTableStackInputWeight7),
            self.circuit_builder.challenge(HashTableStackInputWeight8),
            self.circuit_builder.challenge(HashTableStackInputWeight9),
        ];
        let state = [
            self.st0_next(),
            self.st1_next(),
            self.st2_next(),
            self.st3_next(),
            self.st4_next(),
            self.st5_next(),
            self.st6_next(),
            self.st7_next(),
            self.st8_next(),
            self.st9_next(),
        ];
        let compressed_row = weights
            .into_iter()
            .zip_eq(state.into_iter())
            .map(|(weight, state)| weight * state)
            .sum();
        let running_evaluation_updates = self.running_evaluation_to_hash_table_next()
            - indeterminate * self.running_evaluation_to_hash_table()
            - compressed_row;
        let running_evaluation_remains =
            self.running_evaluation_to_hash_table_next() - self.running_evaluation_to_hash_table();

        hash_selector * running_evaluation_remains + hash_deselector * running_evaluation_updates
    }

    pub fn running_evaluation_from_hash_table_updates_correctly(
        &self,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        let hash_deselector =
            InstructionDeselectors::instruction_deselector(self, Instruction::Hash);
        let hash_selector = self.ci() - self.constant_b(Instruction::Hash.opcode_b());

        let indeterminate = self
            .circuit_builder
            .challenge(FromHashTableEvalIndeterminate);

        let weights = [
            self.circuit_builder.challenge(HashTableDigestOutputWeight0),
            self.circuit_builder.challenge(HashTableDigestOutputWeight1),
            self.circuit_builder.challenge(HashTableDigestOutputWeight2),
            self.circuit_builder.challenge(HashTableDigestOutputWeight3),
            self.circuit_builder.challenge(HashTableDigestOutputWeight4),
        ];
        let state = [
            self.st5_next(),
            self.st6_next(),
            self.st7_next(),
            self.st8_next(),
            self.st9_next(),
        ];
        let compressed_row = weights
            .into_iter()
            .zip_eq(state.into_iter())
            .map(|(weight, state)| weight * state)
            .sum();
        let running_evaluation_updates = self.running_evaluation_from_hash_table_next()
            - indeterminate * self.running_evaluation_from_hash_table()
            - compressed_row;
        let running_evaluation_remains = self.running_evaluation_from_hash_table_next()
            - self.running_evaluation_from_hash_table();

        hash_selector * running_evaluation_remains + hash_deselector * running_evaluation_updates
    }
}

#[derive(Debug, Clone)]
pub struct InstructionDeselectors {
    deselectors: HashMap<
        Instruction,
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    >,
}

impl InstructionDeselectors {
    fn new(factory: &mut DualRowConstraints) -> Self {
        let deselectors = Self::create(factory);

        Self { deselectors }
    }

    /// A polynomial that has solutions when `ci` is not `instruction`.
    ///
    /// This is naively achieved by constructing a polynomial that has
    /// a solution when `ci` is any other instruction. This deselector
    /// can be replaced with an efficient one based on `ib` registers.
    pub fn get(
        &self,
        instruction: Instruction,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        self.deselectors
            .get(&instruction)
            .unwrap_or_else(|| panic!("The instruction {} does not exist!", instruction))
            .clone()
    }

    /// internal helper function to de-duplicate functionality common between the similar (but
    /// different on a type level) functions for construction deselectors
    fn instruction_deselector_common_functionality<II: InputIndicator>(
        circuit_builder: &ConstraintCircuitBuilder<ProcessorTableChallenges, II>,
        instruction: Instruction,
        instruction_bucket_polynomials: [ConstraintCircuitMonad<ProcessorTableChallenges, II>;
            Ord7::COUNT],
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, II> {
        let one = circuit_builder.b_constant(1u32.into());

        let selector_bits: [_; Ord7::COUNT] = [
            instruction.ib(Ord7::IB0),
            instruction.ib(Ord7::IB1),
            instruction.ib(Ord7::IB2),
            instruction.ib(Ord7::IB3),
            instruction.ib(Ord7::IB4),
            instruction.ib(Ord7::IB5),
            instruction.ib(Ord7::IB6),
        ];
        let deselector_polynomials =
            selector_bits.map(|b| one.clone() - circuit_builder.b_constant(b));

        instruction_bucket_polynomials
            .into_iter()
            .zip_eq(deselector_polynomials.into_iter())
            .map(|(bucket_poly, deselector_poly)| bucket_poly - deselector_poly)
            .fold(one, ConstraintCircuitMonad::mul)
    }

    /// A polynomial that has no solutions when ci is 'instruction'
    pub fn instruction_deselector(
        factory: &DualRowConstraints,
        instruction: Instruction,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        let instruction_bucket_polynomials = [
            factory.ib0(),
            factory.ib1(),
            factory.ib2(),
            factory.ib3(),
            factory.ib4(),
            factory.ib5(),
            factory.ib6(),
        ];

        Self::instruction_deselector_common_functionality(
            &factory.circuit_builder,
            instruction,
            instruction_bucket_polynomials,
        )
    }

    /// A polynomial that has no solutions when ci is 'instruction'
    pub fn instruction_deselector_single_row(
        factory: &SingleRowConstraints,
        instruction: Instruction,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        let instruction_bucket_polynomials = [
            factory.ib0(),
            factory.ib1(),
            factory.ib2(),
            factory.ib3(),
            factory.ib4(),
            factory.ib5(),
            factory.ib6(),
        ];

        Self::instruction_deselector_common_functionality(
            &factory.circuit_builder,
            instruction,
            instruction_bucket_polynomials,
        )
    }

    /// A polynomial that has no solutions when ci_next is 'instruction'
    pub fn instruction_deselector_next(
        factory: &DualRowConstraints,
        instruction: Instruction,
    ) -> ConstraintCircuitMonad<
        ProcessorTableChallenges,
        DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
    > {
        let instruction_bucket_polynomials = [
            factory.ib0_next(),
            factory.ib1_next(),
            factory.ib2_next(),
            factory.ib3_next(),
            factory.ib4_next(),
            factory.ib5_next(),
            factory.ib6_next(),
        ];

        Self::instruction_deselector_common_functionality(
            &factory.circuit_builder,
            instruction,
            instruction_bucket_polynomials,
        )
    }

    pub fn create(
        factory: &mut DualRowConstraints,
    ) -> HashMap<
        Instruction,
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        all_instructions_without_args()
            .into_iter()
            .map(|instrctn| (instrctn, Self::instruction_deselector(factory, instrctn)))
            .collect()
    }
}

pub struct ProcessorMatrixRow<'a> {
    pub row: ArrayView1<'a, BFieldElement>,
}

impl<'a> Display for ProcessorMatrixRow<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn row(f: &mut std::fmt::Formatter<'_>, s: String) -> std::fmt::Result {
            writeln!(f, "│ {: <103} │", s)
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
        writeln!(f, " │ {: <25} │", format!("{}", instruction_with_arg))?;
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

        row(
            f,
            format!(
                "hv0-3:    [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[HV0.base_table_index()].value(),
                self.row[HV1.base_table_index()].value(),
                self.row[HV2.base_table_index()].value(),
                self.row[HV3.base_table_index()].value(),
            ),
        )?;
        let w = 2;
        row(
            f,
            format!(
                "ib0-6:    \
                [ {:>w$} | {:>w$} | {:>w$} | {:>w$} | {:>w$} | {:>w$} | {:>w$} ]",
                self.row[IB0.base_table_index()].value(),
                self.row[IB1.base_table_index()].value(),
                self.row[IB2.base_table_index()].value(),
                self.row[IB3.base_table_index()].value(),
                self.row[IB4.base_table_index()].value(),
                self.row[IB5.base_table_index()].value(),
                self.row[IB6.base_table_index()].value(),
            ),
        )?;
        write!(
            f,
            "╰─────────────────────────────────────────────────────────────────\
            ────────────────────────────────────────╯"
        )
    }
}

pub struct ExtProcessorMatrixRow<'a> {
    pub row: ArrayView1<'a, XFieldElement>,
}

impl<'a> Display for ExtProcessorMatrixRow<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let row = |form: &mut std::fmt::Formatter<'_>,
                   desc: &str,
                   col: ProcessorExtTableColumn|
         -> std::fmt::Result {
            // without the extra `format!()`, alignment in `writeln!()` fails
            let formatted_col_elem = format!("{}", self.row[col.ext_table_index()]);
            writeln!(form, "     │ {: <18}  {:>73} │", desc, formatted_col_elem,)
        };
        writeln!(
            f,
            "     ╭───────────────────────────────────────────────────────\
            ────────────────────────────────────────╮"
        )?;
        row(f, "input_table_ea", InputTableEvalArg)?;
        row(f, "output_table_ea", OutputTableEvalArg)?;
        row(f, "instr_table_pa", InstructionTablePermArg)?;
        row(f, "opstack_table_pa", OpStackTablePermArg)?;
        row(f, "ram_table_pa", RamTablePermArg)?;
        row(f, "jumpstack_table_pa", JumpStackTablePermArg)?;
        row(f, "to_hash_table_ea", ToHashTableEvalArg)?;
        row(f, "from_hash_table_ea", FromHashTableEvalArg)?;
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

    use crate::stark::triton_stark_tests::parse_simulate_pad;
    use crate::table::challenges::AllChallenges;
    use crate::table::master_table::MasterTable;
    use crate::table::processor_table::ProcessorMatrixRow;
    use crate::vm::simulate_no_input;
    use triton_opcodes::ord_n::Ord16;
    use triton_opcodes::program::Program;

    use super::*;

    #[test]
    /// helps identifying whether the printing causes an infinite loop
    fn print_simple_processor_table_row_test() {
        let code = "push 2 push -1 add assert halt";
        let program = Program::from_code(code).unwrap();
        let (aet, _, _) = simulate_no_input(&program);
        for row in aet.processor_matrix.rows() {
            println!("{}", ProcessorMatrixRow { row });
        }
    }

    fn get_test_row_from_source_code(source_code: &str, row_num: usize) -> Array2<BFieldElement> {
        let (_, unpadded_master_base_table, _) = parse_simulate_pad(source_code, vec![], vec![]);
        unpadded_master_base_table
            .trace_table()
            .slice(s![row_num..=row_num + 1, ..])
            .to_owned()
    }

    fn get_transition_constraints_for_instruction(
        instruction: Instruction,
    ) -> Vec<
        ConstraintCircuitMonad<
            ProcessorTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let tc = DualRowConstraints::default();
        match instruction {
            Pop => tc.instruction_pop(),
            Push(_) => tc.instruction_push(),
            Divine(_) => tc.instruction_divine(),
            Dup(_) => tc.instruction_dup(),
            Swap(_) => tc.instruction_swap(),
            Nop => tc.instruction_nop(),
            Skiz => tc.instruction_skiz(),
            Call(_) => tc.instruction_call(),
            Return => tc.instruction_return(),
            Recurse => tc.instruction_recurse(),
            Assert => tc.instruction_assert(),
            Halt => tc.instruction_halt(),
            ReadMem => tc.instruction_read_mem(),
            WriteMem => tc.instruction_write_mem(),
            Hash => tc.instruction_hash(),
            DivineSibling => tc.instruction_divine_sibling(),
            AssertVector => tc.instruction_assert_vector(),
            Add => tc.instruction_add(),
            Mul => tc.instruction_mul(),
            Invert => tc.instruction_invert(),
            Split => tc.instruction_split(),
            Eq => tc.instruction_eq(),
            Lsb => tc.instruction_lsb(),
            XxAdd => tc.instruction_xxadd(),
            XxMul => tc.instruction_xxmul(),
            XInvert => tc.instruction_xinv(),
            XbMul => tc.instruction_xbmul(),
            ReadIo => tc.instruction_read_io(),
            WriteIo => tc.instruction_write_io(),
        }
    }

    fn test_constraints_for_rows_with_debug_info(
        instruction: Instruction,
        master_base_tables: &[Array2<BFieldElement>],
        debug_cols_curr_row: &[ProcessorBaseTableColumn],
        debug_cols_next_row: &[ProcessorBaseTableColumn],
    ) {
        let challenges = AllChallenges::placeholder(&[], &[]);
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
                get_transition_constraints_for_instruction(instruction)
                    .into_iter()
                    .enumerate()
            {
                let evaluation_result = constraint_circuit.consume().evaluate(
                    test_rows.view(),
                    fake_ext_table.view(),
                    &challenges.processor_table_challenges,
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
        let test_rows = [get_test_row_from_source_code("push 1 dup0 halt", 1)];
        test_constraints_for_rows_with_debug_info(
            Dup(Ord16::ST0),
            &test_rows,
            &[ST0, ST1, ST2],
            &[ST0, ST1, ST2],
        );
    }

    #[test]
    fn transition_constraints_for_instruction_swap_test() {
        let test_rows = [get_test_row_from_source_code("push 1 push 2 swap1 halt", 2)];
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
            "push 2 call label halt label: push -1 add dup0 skiz recurse return ",
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
    fn transition_constraints_for_instruction_lsb_test() {
        let test_rows = [get_test_row_from_source_code(
            "push 3 lsb assert assert halt",
            1,
        )];
        test_constraints_for_rows_with_debug_info(Lsb, &test_rows, &[ST0], &[ST0, ST1]);
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
        let mut factory = DualRowConstraints::default();
        let deselectors = InstructionDeselectors::new(&mut factory);

        let mut master_base_table = Array2::zeros([2, NUM_BASE_COLUMNS]);
        let master_ext_table = Array2::zeros([2, NUM_EXT_COLUMNS]);

        // We need dummy challenges to evaluate.
        let dummy_challenges = AllChallenges::placeholder(&[], &[]);
        for instruction in all_instructions_without_args() {
            use ProcessorBaseTableColumn::*;
            let deselector = deselectors.get(instruction);

            println!("\n\nThe Deselector for instruction {instruction} is:\n{deselector}",);

            // Negative tests
            for other_instruction in all_instructions_without_args()
                .into_iter()
                .filter(|other_instruction| *other_instruction != instruction)
            {
                let mut curr_row = master_base_table.slice_mut(s![0, ..]);
                curr_row[IB0.master_base_table_index()] = other_instruction.ib(Ord7::IB0);
                curr_row[IB1.master_base_table_index()] = other_instruction.ib(Ord7::IB1);
                curr_row[IB2.master_base_table_index()] = other_instruction.ib(Ord7::IB2);
                curr_row[IB3.master_base_table_index()] = other_instruction.ib(Ord7::IB3);
                curr_row[IB4.master_base_table_index()] = other_instruction.ib(Ord7::IB4);
                curr_row[IB5.master_base_table_index()] = other_instruction.ib(Ord7::IB5);
                curr_row[IB6.master_base_table_index()] = other_instruction.ib(Ord7::IB6);
                let result = deselector.clone().consume().evaluate(
                    master_base_table.view(),
                    master_ext_table.view(),
                    &dummy_challenges.processor_table_challenges,
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
            curr_row[IB0.master_base_table_index()] = instruction.ib(Ord7::IB0);
            curr_row[IB1.master_base_table_index()] = instruction.ib(Ord7::IB1);
            curr_row[IB2.master_base_table_index()] = instruction.ib(Ord7::IB2);
            curr_row[IB3.master_base_table_index()] = instruction.ib(Ord7::IB3);
            curr_row[IB4.master_base_table_index()] = instruction.ib(Ord7::IB4);
            curr_row[IB5.master_base_table_index()] = instruction.ib(Ord7::IB5);
            curr_row[IB6.master_base_table_index()] = instruction.ib(Ord7::IB6);
            let result = deselector.consume().evaluate(
                master_base_table.view(),
                master_ext_table.view(),
                &dummy_challenges.processor_table_challenges,
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
        let factory = DualRowConstraints::default();
        let all_instructions_and_their_transition_constraints: [(Instruction, _);
            Instruction::COUNT] = [
            (Pop, factory.instruction_pop()),
            (Push(Default::default()), factory.instruction_push()),
            (Divine(Default::default()), factory.instruction_divine()),
            (Dup(Default::default()), factory.instruction_dup()),
            (Swap(Default::default()), factory.instruction_swap()),
            (Nop, factory.instruction_nop()),
            (Skiz, factory.instruction_skiz()),
            (Call(Default::default()), factory.instruction_call()),
            (Return, factory.instruction_return()),
            (Recurse, factory.instruction_recurse()),
            (Assert, factory.instruction_assert()),
            (Halt, factory.instruction_halt()),
            (ReadMem, factory.instruction_read_mem()),
            (WriteMem, factory.instruction_write_mem()),
            (Hash, factory.instruction_hash()),
            (DivineSibling, factory.instruction_divine_sibling()),
            (AssertVector, factory.instruction_assert_vector()),
            (Add, factory.instruction_add()),
            (Mul, factory.instruction_mul()),
            (Invert, factory.instruction_invert()),
            (Split, factory.instruction_split()),
            (Eq, factory.instruction_eq()),
            (Lsb, factory.instruction_lsb()),
            (XxAdd, factory.instruction_xxadd()),
            (XxMul, factory.instruction_xxmul()),
            (XInvert, factory.instruction_xinv()),
            (XbMul, factory.instruction_xbmul()),
            (ReadIo, factory.instruction_read_io()),
            (WriteIo, factory.instruction_write_io()),
        ];

        println!("| Instruction     | #polys | max deg | Degrees");
        println!("|:----------------|-------:|--------:|:------------");
        for (instruction, constraints) in all_instructions_and_their_transition_constraints {
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
