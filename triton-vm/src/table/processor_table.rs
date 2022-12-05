use std::cmp::max;
use std::cmp::Eq;
use std::collections::HashMap;
use std::ops::Mul;

use itertools::Itertools;
use ndarray::s;
use ndarray::Array1;
use ndarray::ArrayBase;
use ndarray::ArrayViewMut2;
use ndarray::Ix1;
use num_traits::One;
use num_traits::Zero;
use strum::EnumCount;
use strum_macros::Display;
use strum_macros::EnumCount as EnumCountMacro;
use strum_macros::EnumIter;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::MPolynomial;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use ProcessorTableChallengeId::*;

use crate::cross_table_arguments::CrossTableArg;
use crate::cross_table_arguments::EvalArg;
use crate::cross_table_arguments::PermArg;
use crate::instruction::all_instructions_without_args;
use crate::instruction::AnInstruction::*;
use crate::instruction::Instruction;
use crate::ord_n::Ord7;
use crate::table::base_matrix::AlgebraicExecutionTrace;
use crate::table::base_table::Extendable;
use crate::table::base_table::InheritsFromTable;
use crate::table::base_table::Table;
use crate::table::base_table::TableLike;
use crate::table::challenges::TableChallenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::ConstraintCircuitMonad;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::InputIndicator;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::extension_table::ExtensionTable;
use crate::table::extension_table::QuotientableExtensionTable;
use crate::table::table_column::ProcessorBaseTableColumn;
use crate::table::table_column::ProcessorBaseTableColumn::*;
use crate::table::table_column::ProcessorExtTableColumn;
use crate::table::table_column::ProcessorExtTableColumn::*;

pub const PROCESSOR_TABLE_NUM_PERMUTATION_ARGUMENTS: usize = 5;
pub const PROCESSOR_TABLE_NUM_EVALUATION_ARGUMENTS: usize = 5;
pub const PROCESSOR_TABLE_NUM_EXTENSION_CHALLENGES: usize = ProcessorTableChallengeId::COUNT;

pub const BASE_WIDTH: usize = ProcessorBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = ProcessorExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

#[derive(Debug, Clone)]
pub struct ProcessorTable {
    inherited_table: Table<BFieldElement>,
}

impl InheritsFromTable<BFieldElement> for ProcessorTable {
    fn inherited_table(&self) -> &Table<BFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<BFieldElement> {
        &mut self.inherited_table
    }
}

impl ProcessorTable {
    pub fn new(inherited_table: Table<BFieldElement>) -> Self {
        Self { inherited_table }
    }

    pub fn new_prover(matrix: Vec<Vec<BFieldElement>>) -> Self {
        let inherited_table =
            Table::new(BASE_WIDTH, FULL_WIDTH, matrix, "ProcessorTable".to_string());
        Self { inherited_table }
    }

    pub fn fill_trace(
        processor_table: &mut ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
        mut all_clk_jump_diffs: Vec<BFieldElement>,
    ) {
        all_clk_jump_diffs.sort_by_key(|bfe| std::cmp::Reverse(bfe.value()));

        // - fill the processor table from the AET
        // - add all clock jump differences and their inverses
        // - add inverses of unique clock jump difference differences
        let zero = BFieldElement::zero();
        let mut previous_row: Option<ArrayBase<_, Ix1>> = None;
        for (row_idx, &processor_row) in aet.processor_matrix.iter().enumerate() {
            let mut row = Array1::from(processor_row.to_vec());

            let clk_jump_difference = all_clk_jump_diffs.pop().unwrap_or(zero);
            let clk_jump_difference_inv = clk_jump_difference.inverse_or_zero();
            row[usize::from(ClockJumpDifference)] = clk_jump_difference;
            row[usize::from(ClockJumpDifferenceInverse)] = clk_jump_difference_inv;

            if let Some(prow) = previous_row {
                let previous_clk_jump_difference = prow[usize::from(ClockJumpDifference)];
                if previous_clk_jump_difference != clk_jump_difference {
                    let clk_diff_diff: BFieldElement =
                        clk_jump_difference - previous_clk_jump_difference;
                    let clk_diff_diff_inv = clk_diff_diff.inverse();
                    row[usize::from(UniqueClockJumpDiffDiffInverse)] = clk_diff_diff_inv;
                }
            }

            previous_row = Some(row.clone());
            row.move_into(processor_table.slice_mut(s![row_idx, ..]));
        }

        assert!(
            all_clk_jump_diffs.is_empty(),
            "Processor Table must record all clock jump differences, but didn't have enough space \
            for the remaining {}.",
            all_clk_jump_diffs.len()
        );
    }

    pub fn extend(&self, challenges: &ProcessorTableChallenges) -> ExtProcessorTable {
        let mut unique_clock_jump_differences = vec![];
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());

        let mut input_table_running_evaluation = EvalArg::default_initial();
        let mut output_table_running_evaluation = EvalArg::default_initial();
        let mut instruction_table_running_product = PermArg::default_initial();
        let mut opstack_table_running_product = PermArg::default_initial();
        let mut ram_table_running_product = PermArg::default_initial();
        let mut jump_stack_running_product = PermArg::default_initial();
        let mut to_hash_table_running_evaluation = EvalArg::default_initial();
        let mut from_hash_table_running_evaluation = EvalArg::default_initial();
        let mut selected_clock_cycles_running_evaluation = EvalArg::default_initial();
        let mut unique_clock_jump_differences_running_evaluation = EvalArg::default_initial();
        let mut all_clock_jump_differences_running_product =
            PermArg::default_initial() * PermArg::default_initial() * PermArg::default_initial();

        let mut previous_row: Option<Vec<BFieldElement>> = None;
        for row in self.data().iter() {
            let mut extension_row = [0.into(); FULL_WIDTH];
            extension_row[..BASE_WIDTH]
                .copy_from_slice(&row.iter().map(|elem| elem.lift()).collect_vec());

            // Input table
            if let Some(prow) = previous_row.clone() {
                if prow[usize::from(CI)] == Instruction::ReadIo.opcode_b() {
                    let input_symbol = extension_row[usize::from(ST0)];
                    input_table_running_evaluation = input_table_running_evaluation
                        * challenges.standard_input_eval_indeterminate
                        + input_symbol;
                }
            }
            extension_row[usize::from(InputTableEvalArg)] = input_table_running_evaluation;

            // Output table
            if row[usize::from(CI)] == Instruction::WriteIo.opcode_b() {
                let output_symbol = extension_row[usize::from(ST0)];
                output_table_running_evaluation = output_table_running_evaluation
                    * challenges.standard_output_eval_indeterminate
                    + output_symbol;
            }
            extension_row[usize::from(OutputTableEvalArg)] = output_table_running_evaluation;

            // Instruction table
            let ip = extension_row[usize::from(IP)];
            let ci = extension_row[usize::from(CI)];
            let nia = extension_row[usize::from(NIA)];

            let ip_w = challenges.instruction_table_ip_weight;
            let ci_w = challenges.instruction_table_ci_processor_weight;
            let nia_w = challenges.instruction_table_nia_weight;

            if row[usize::from(IsPadding)].is_zero() {
                let compressed_row_for_instruction_table_permutation_argument =
                    ip * ip_w + ci * ci_w + nia * nia_w;
                instruction_table_running_product *= challenges.instruction_perm_indeterminate
                    - compressed_row_for_instruction_table_permutation_argument;
            }
            extension_row[usize::from(InstructionTablePermArg)] = instruction_table_running_product;

            // OpStack table
            let clk = extension_row[usize::from(CLK)];
            let ib1 = extension_row[usize::from(IB1)];
            let osp = extension_row[usize::from(OSP)];
            let osv = extension_row[usize::from(OSV)];

            let compressed_row_for_op_stack_table_permutation_argument = clk
                * challenges.op_stack_table_clk_weight
                + ib1 * challenges.op_stack_table_ib1_weight
                + osp * challenges.op_stack_table_osp_weight
                + osv * challenges.op_stack_table_osv_weight;
            opstack_table_running_product *= challenges.op_stack_perm_indeterminate
                - compressed_row_for_op_stack_table_permutation_argument;
            extension_row[usize::from(OpStackTablePermArg)] = opstack_table_running_product;

            // RAM Table
            let ramv = extension_row[usize::from(RAMV)];
            let ramp = extension_row[usize::from(RAMP)];

            let compressed_row_for_ram_table_permutation_argument = clk
                * challenges.ram_table_clk_weight
                + ramv * challenges.ram_table_ramv_weight
                + ramp * challenges.ram_table_ramp_weight;
            ram_table_running_product *= challenges.ram_perm_indeterminate
                - compressed_row_for_ram_table_permutation_argument;
            extension_row[usize::from(RamTablePermArg)] = ram_table_running_product;

            // JumpStack Table
            let jsp = extension_row[usize::from(JSP)];
            let jso = extension_row[usize::from(JSO)];
            let jsd = extension_row[usize::from(JSD)];
            let compressed_row_for_jump_stack_table = clk * challenges.jump_stack_table_clk_weight
                + ci * challenges.jump_stack_table_ci_weight
                + jsp * challenges.jump_stack_table_jsp_weight
                + jso * challenges.jump_stack_table_jso_weight
                + jsd * challenges.jump_stack_table_jsd_weight;
            jump_stack_running_product *=
                challenges.jump_stack_perm_indeterminate - compressed_row_for_jump_stack_table;
            extension_row[usize::from(JumpStackTablePermArg)] = jump_stack_running_product;

            // Hash Table – Hash's input from Processor to Hash Coprocessor
            if row[usize::from(CI)] == Instruction::Hash.opcode_b() {
                let st_0_through_9 = [
                    extension_row[usize::from(ST0)],
                    extension_row[usize::from(ST1)],
                    extension_row[usize::from(ST2)],
                    extension_row[usize::from(ST3)],
                    extension_row[usize::from(ST4)],
                    extension_row[usize::from(ST5)],
                    extension_row[usize::from(ST6)],
                    extension_row[usize::from(ST7)],
                    extension_row[usize::from(ST8)],
                    extension_row[usize::from(ST9)],
                ];
                let compressed_row_for_hash_input: XFieldElement = st_0_through_9
                    .into_iter()
                    .zip_eq(
                        [
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
                        ]
                        .into_iter(),
                    )
                    .map(|(st, weight)| weight * st)
                    .sum();
                to_hash_table_running_evaluation = to_hash_table_running_evaluation
                    * challenges.to_hash_table_eval_indeterminate
                    + compressed_row_for_hash_input;
            }
            extension_row[usize::from(ToHashTableEvalArg)] = to_hash_table_running_evaluation;

            // Hash Table – Hash's output from Hash Coprocessor to Processor
            if let Some(prow) = previous_row.clone() {
                if prow[usize::from(CI)] == Instruction::Hash.opcode_b() {
                    let st_5_through_9 = [
                        extension_row[usize::from(ST5)],
                        extension_row[usize::from(ST6)],
                        extension_row[usize::from(ST7)],
                        extension_row[usize::from(ST8)],
                        extension_row[usize::from(ST9)],
                    ];
                    let compressed_row_for_hash_digest: XFieldElement = st_5_through_9
                        .into_iter()
                        .zip_eq(
                            [
                                challenges.hash_table_digest_output_weight0,
                                challenges.hash_table_digest_output_weight1,
                                challenges.hash_table_digest_output_weight2,
                                challenges.hash_table_digest_output_weight3,
                                challenges.hash_table_digest_output_weight4,
                            ]
                            .into_iter(),
                        )
                        .map(|(st, weight)| weight * st)
                        .sum();
                    from_hash_table_running_evaluation = from_hash_table_running_evaluation
                        * challenges.from_hash_table_eval_indeterminate
                        + compressed_row_for_hash_digest;
                }
            }
            extension_row[usize::from(FromHashTableEvalArg)] = from_hash_table_running_evaluation;

            // Clock Jump Difference
            let current_clock_jump_difference = row[usize::from(ClockJumpDifference)].lift();
            if !current_clock_jump_difference.is_zero() {
                all_clock_jump_differences_running_product *= challenges
                    .all_clock_jump_differences_multi_perm_indeterminate
                    - current_clock_jump_difference;
            }
            extension_row[usize::from(AllClockJumpDifferencesPermArg)] =
                all_clock_jump_differences_running_product;

            if let Some(prow) = previous_row {
                let previous_clock_jump_difference = prow[usize::from(ClockJumpDifference)].lift();
                if previous_clock_jump_difference != current_clock_jump_difference
                    && !current_clock_jump_difference.is_zero()
                {
                    unique_clock_jump_differences.push(current_clock_jump_difference);
                    unique_clock_jump_differences_running_evaluation =
                        unique_clock_jump_differences_running_evaluation
                            * challenges.unique_clock_jump_differences_eval_indeterminate
                            + current_clock_jump_difference;
                }
            } else if !current_clock_jump_difference.is_zero() {
                unique_clock_jump_differences.push(current_clock_jump_difference);
                unique_clock_jump_differences_running_evaluation = challenges
                    .unique_clock_jump_differences_eval_indeterminate
                    + current_clock_jump_difference;
            }
            extension_row[usize::from(UniqueClockJumpDifferencesEvalArg)] =
                unique_clock_jump_differences_running_evaluation;

            previous_row = Some(row.clone());
            extension_matrix.push(extension_row.to_vec());
        }

        if std::env::var("DEBUG").is_ok() {
            let mut unique_clock_jump_differences_copy = unique_clock_jump_differences.clone();
            unique_clock_jump_differences_copy.sort_by_key(|xfe| xfe.unlift().unwrap().value());
            unique_clock_jump_differences_copy.dedup();
            assert_eq!(
                unique_clock_jump_differences_copy,
                unique_clock_jump_differences
            );
        }

        // second pass over Processor Table to compute evaluation over
        // all relevant clock cycles
        for extension_row in extension_matrix.iter_mut() {
            let current_clk = extension_row[usize::from(CLK)];
            if unique_clock_jump_differences.contains(&current_clk) {
                selected_clock_cycles_running_evaluation = selected_clock_cycles_running_evaluation
                    * challenges.unique_clock_jump_differences_eval_indeterminate
                    + current_clk;
            }
            extension_row[usize::from(SelectedClockCyclesEvalArg)] =
                selected_clock_cycles_running_evaluation;
        }

        assert_eq!(self.data().len(), extension_matrix.len());
        let inherited_table = self.new_from_lifted_matrix(extension_matrix);
        ExtProcessorTable { inherited_table }
    }
}

impl ExtProcessorTable {
    pub fn new(inherited_table: Table<XFieldElement>) -> Self {
        Self { inherited_table }
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
        factory: &mut DualRowConstraints,
        instr_tc_polys_tuples: [(
            Instruction,
            Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>>,
        ); Instruction::COUNT],
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
            ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>,
        >,
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
pub struct ExtProcessorTable {
    pub(crate) inherited_table: Table<XFieldElement>,
}

impl QuotientableExtensionTable for ExtProcessorTable {}

impl InheritsFromTable<XFieldElement> for ExtProcessorTable {
    fn inherited_table(&self) -> &Table<XFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<XFieldElement> {
        &mut self.inherited_table
    }
}

impl TableLike<BFieldElement> for ProcessorTable {}

impl Extendable for ProcessorTable {
    fn get_padding_rows(&self) -> (Option<usize>, Vec<Vec<BFieldElement>>) {
        let zero = BFieldElement::zero();
        let one = BFieldElement::one();
        if let Some(row) = self.data().last() {
            let mut padding_row = row.clone();
            padding_row[usize::from(CLK)] += one;
            padding_row[usize::from(IsPadding)] = one;
            (None, vec![padding_row])
        } else {
            let mut padding_row = vec![zero; BASE_WIDTH];
            padding_row[usize::from(IsPadding)] = one;
            (None, vec![padding_row])
        }
    }
}

impl TableLike<XFieldElement> for ExtProcessorTable {}

impl ExtProcessorTable {
    pub fn ext_initial_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>>> {
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

    pub fn ext_consistency_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>>> {
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

    pub fn ext_transition_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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

    pub fn ext_terminal_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>>> {
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
    variables: [ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>>;
        FULL_WIDTH],
    circuit_builder:
        ConstraintCircuitBuilder<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>>,
    one: ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>>,
    two: ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>>,
}

impl Default for SingleRowConstraints {
    fn default() -> Self {
        let circuit_builder = ConstraintCircuitBuilder::new(FULL_WIDTH);
        let variables_as_circuits = (0..FULL_WIDTH)
            .map(|i| {
                let input: SingleRowIndicator<FULL_WIDTH> = i.into();
                circuit_builder.input(input)
            })
            .collect_vec();

        let variables = variables_as_circuits
            .try_into()
            .expect("Create variables for single row constraints");

        let one = circuit_builder.b_constant(1u32.into());
        let two = circuit_builder.b_constant(2u32.into());

        Self {
            variables,
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
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.circuit_builder.x_constant(constant)
    }
    pub fn constant_from_i32(
        &self,
        constant: i32,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        let bfe = if constant < 0 {
            BFieldElement::new(BFieldElement::QUOTIENT - ((-constant) as u64))
        } else {
            BFieldElement::new(constant as u64)
        };
        self.circuit_builder.b_constant(bfe)
    }

    pub fn one(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.one.clone()
    }
    pub fn two(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.two.clone()
    }
    pub fn clk(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(CLK)].clone()
    }
    pub fn is_padding(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(IsPadding)].clone()
    }
    pub fn ip(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(IP)].clone()
    }
    pub fn ci(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(CI)].clone()
    }
    pub fn nia(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(NIA)].clone()
    }
    pub fn ib0(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(IB0)].clone()
    }
    pub fn ib1(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(IB1)].clone()
    }
    pub fn ib2(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(IB2)].clone()
    }
    pub fn ib3(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(IB3)].clone()
    }
    pub fn ib4(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(IB4)].clone()
    }
    pub fn ib5(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(IB5)].clone()
    }
    pub fn ib6(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(IB6)].clone()
    }
    pub fn jsp(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(JSP)].clone()
    }
    pub fn jsd(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(JSD)].clone()
    }
    pub fn jso(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(JSO)].clone()
    }
    pub fn st0(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST0)].clone()
    }
    pub fn st1(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST1)].clone()
    }
    pub fn st2(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST2)].clone()
    }
    pub fn st3(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST3)].clone()
    }
    pub fn st4(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST4)].clone()
    }
    pub fn st5(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST5)].clone()
    }
    pub fn st6(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST6)].clone()
    }
    pub fn st7(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST7)].clone()
    }
    pub fn st8(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST8)].clone()
    }
    pub fn st9(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST9)].clone()
    }
    pub fn st10(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST10)].clone()
    }
    pub fn st11(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST11)].clone()
    }
    pub fn st12(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST12)].clone()
    }
    pub fn st13(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST13)].clone()
    }
    pub fn st14(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST14)].clone()
    }
    pub fn st15(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST15)].clone()
    }
    pub fn osp(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(OSP)].clone()
    }
    pub fn osv(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(OSV)].clone()
    }
    pub fn hv0(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(HV0)].clone()
    }
    pub fn hv1(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(HV1)].clone()
    }
    pub fn hv2(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(HV2)].clone()
    }
    pub fn hv3(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(HV3)].clone()
    }
    pub fn ramv(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(RAMV)].clone()
    }
    pub fn ramp(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(RAMP)].clone()
    }

    pub fn cjd(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ClockJumpDifference)].clone()
    }

    pub fn invm(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ClockJumpDifferenceInverse)].clone()
    }

    pub fn invu(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(UniqueClockJumpDiffDiffInverse)].clone()
    }

    pub fn rer(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(SelectedClockCyclesEvalArg)].clone()
    }

    pub fn reu(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(UniqueClockJumpDifferencesEvalArg)].clone()
    }

    pub fn rpm(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(AllClockJumpDifferencesPermArg)].clone()
    }

    pub fn running_evaluation_standard_input(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(InputTableEvalArg)].clone()
    }
    pub fn running_evaluation_standard_output(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(OutputTableEvalArg)].clone()
    }
    pub fn running_product_instruction_table(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(InstructionTablePermArg)].clone()
    }
    pub fn running_product_op_stack_table(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(OpStackTablePermArg)].clone()
    }
    pub fn running_product_ram_table(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(RamTablePermArg)].clone()
    }
    pub fn running_product_jump_stack_table(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(JumpStackTablePermArg)].clone()
    }
    pub fn running_evaluation_to_hash_table(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ToHashTableEvalArg)].clone()
    }
    pub fn running_evaluation_from_hash_table(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(FromHashTableEvalArg)].clone()
    }
}

#[derive(Debug, Clone)]
pub struct DualRowConstraints {
    variables: [ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>;
        2 * FULL_WIDTH],
    circuit_builder:
        ConstraintCircuitBuilder<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>,
    zero: ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>,
    one: ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>,
    two: ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>,
}

impl Default for DualRowConstraints {
    fn default() -> Self {
        let circuit_builder = ConstraintCircuitBuilder::new(2 * FULL_WIDTH);
        let variables_as_circuits = (0..2 * FULL_WIDTH)
            .map(|i| {
                let input: DualRowIndicator<FULL_WIDTH> = i.into();
                circuit_builder.input(input)
            })
            .collect_vec();

        let variables = variables_as_circuits
            .try_into()
            .expect("Create variables for transition constraints");

        let zero = circuit_builder.b_constant(0u32.into());
        let one = circuit_builder.b_constant(1u32.into());
        let two = circuit_builder.b_constant(2u32.into());

        Self {
            variables,
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
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        let one = self.one();
        let clk = self.clk();
        let clk_next = self.clk_next();

        clk_next - clk - one
    }

    pub fn is_padding_is_zero_or_does_not_change(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.is_padding() * (self.is_padding_next() - self.is_padding())
    }

    pub fn indicator_polynomial(
        &self,
        i: usize,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
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
            _ => panic!(
                "No indicator polynomial with index {} exists: there are only 16.",
                i
            ),
        }
    }

    pub fn instruction_pop(
        &self,
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
        [self.step_1(), self.shrink_stack(), self.keep_ram()].concat()
    }

    /// push'es argument should be on the stack after execution
    /// $st0_next == nia  =>  st0_next - nia == 0$
    pub fn instruction_push(
        &self,
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
        [self.step_1(), self.grow_stack(), self.keep_ram()].concat()
    }

    pub fn instruction_dup(
        &self,
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
        [self.step_1(), self.keep_stack(), self.keep_ram()].concat()
    }

    pub fn instruction_skiz(
        &self,
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
        // The jump stack pointer jsp is decremented by 1.
        let jsp_incr_1 = self.jsp_next() - (self.jsp() - self.one());

        // The instruction pointer ip is set to the last call's origin jso.
        let ip_becomes_jso = self.ip_next() - self.jso();

        let specific_constraints = vec![jsp_incr_1, ip_becomes_jso];
        [specific_constraints, self.keep_stack(), self.keep_ram()].concat()
    }

    pub fn instruction_recurse(
        &self,
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
        // the RAM pointer is overwritten with st1
        let update_ramp = self.ramp_next() - self.st1();

        // The top of the stack is overwritten with the RAM value.
        let st0_becomes_ramv = self.st0_next() - self.ramv();

        let specific_constraints = vec![update_ramp, st0_becomes_ramv];
        [specific_constraints, self.step_1(), self.unop()].concat()
    }

    pub fn instruction_write_mem(
        &self,
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
        let operand = self.variables[usize::from(ST0)].clone();
        let shifted_operand = self.variables[FULL_WIDTH + usize::from(ST1)].clone();
        let lsb = self.variables[FULL_WIDTH + usize::from(ST0)].clone();

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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
        [self.step_1(), self.grow_stack(), self.keep_ram()].concat()
    }

    /// This instruction has no additional transition constraints.
    ///
    /// An Evaluation Argument with the list of output symbols guarantees correct transition.
    pub fn instruction_write_io(
        &self,
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
        [self.step_1(), self.shrink_stack(), self.keep_ram()].concat()
    }

    pub fn zero(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.zero.to_owned()
    }

    pub fn one(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.one.to_owned()
    }

    pub fn two(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.two.to_owned()
    }

    pub fn constant(
        &self,
        constant: u32,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.circuit_builder.b_constant(constant.into())
    }

    pub fn constant_b(
        &self,
        constant: BFieldElement,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.circuit_builder.b_constant(constant)
    }

    pub fn constant_x(&self, constant: XFieldElement) -> MPolynomial<XFieldElement> {
        MPolynomial::from_constant(constant, 2 * FULL_WIDTH)
    }

    pub fn clk(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(CLK)].clone()
    }

    pub fn ip(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(IP)].clone()
    }

    pub fn ci(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(CI)].clone()
    }

    pub fn nia(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(NIA)].clone()
    }

    pub fn ib0(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(IB0)].clone()
    }

    pub fn ib1(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(IB1)].clone()
    }

    pub fn ib2(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(IB2)].clone()
    }

    pub fn ib3(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(IB3)].clone()
    }

    pub fn ib4(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(IB4)].clone()
    }

    pub fn ib5(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(IB5)].clone()
    }

    pub fn ib6(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(IB6)].clone()
    }

    pub fn jsp(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(JSP)].clone()
    }

    pub fn jsd(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(JSD)].clone()
    }

    pub fn jso(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(JSO)].clone()
    }

    pub fn st0(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST0)].clone()
    }

    pub fn st1(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST1)].clone()
    }

    pub fn st2(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST2)].clone()
    }

    pub fn st3(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST3)].clone()
    }

    pub fn st4(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST4)].clone()
    }

    pub fn st5(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST5)].clone()
    }

    pub fn st6(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST6)].clone()
    }

    pub fn st7(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST7)].clone()
    }

    pub fn st8(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST8)].clone()
    }

    pub fn st9(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST9)].clone()
    }

    pub fn st10(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST10)].clone()
    }

    pub fn st11(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST11)].clone()
    }

    pub fn st12(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST12)].clone()
    }

    pub fn st13(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST13)].clone()
    }

    pub fn st14(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST14)].clone()
    }

    pub fn st15(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ST15)].clone()
    }

    pub fn osp(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(OSP)].clone()
    }

    pub fn osv(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(OSV)].clone()
    }

    pub fn hv0(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(HV0)].clone()
    }

    pub fn hv1(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(HV1)].clone()
    }

    pub fn hv2(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(HV2)].clone()
    }

    pub fn hv3(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(HV3)].clone()
    }

    pub fn ramp(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(RAMP)].clone()
    }

    pub fn ramv(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(RAMV)].clone()
    }

    pub fn is_padding(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(IsPadding)].clone()
    }

    pub fn cjd(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ClockJumpDifference)].clone()
    }

    pub fn invm(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ClockJumpDifferenceInverse)].clone()
    }

    pub fn invu(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(UniqueClockJumpDiffDiffInverse)].clone()
    }

    pub fn rer(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(SelectedClockCyclesEvalArg)].clone()
    }

    pub fn reu(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(UniqueClockJumpDifferencesEvalArg)].clone()
    }

    pub fn rpm(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(AllClockJumpDifferencesPermArg)].clone()
    }

    pub fn running_evaluation_standard_input(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(InputTableEvalArg)].clone()
    }
    pub fn running_evaluation_standard_output(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(OutputTableEvalArg)].clone()
    }
    pub fn running_product_instruction_table(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(InstructionTablePermArg)].clone()
    }
    pub fn running_product_op_stack_table(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(OpStackTablePermArg)].clone()
    }
    pub fn running_product_ram_table(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(RamTablePermArg)].clone()
    }
    pub fn running_product_jump_stack_table(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(JumpStackTablePermArg)].clone()
    }
    pub fn running_evaluation_to_hash_table(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(ToHashTableEvalArg)].clone()
    }
    pub fn running_evaluation_from_hash_table(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[usize::from(FromHashTableEvalArg)].clone()
    }

    // Property: All polynomial variables that contain '_next' have the same
    // variable position / value as the one without '_next', +/- FULL_WIDTH.
    pub fn clk_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(CLK)].clone()
    }

    pub fn ip_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(IP)].clone()
    }

    pub fn ci_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(CI)].clone()
    }

    pub fn nia_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(NIA)].clone()
    }

    pub fn ib0_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(IB0)].clone()
    }
    pub fn ib1_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(IB1)].clone()
    }
    pub fn ib2_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(IB2)].clone()
    }
    pub fn ib3_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(IB3)].clone()
    }
    pub fn ib4_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(IB4)].clone()
    }
    pub fn ib5_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(IB5)].clone()
    }
    pub fn ib6_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(IB6)].clone()
    }

    pub fn jsp_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(JSP)].clone()
    }

    pub fn jsd_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(JSD)].clone()
    }

    pub fn jso_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(JSO)].clone()
    }

    pub fn st0_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ST0)].clone()
    }

    pub fn st1_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ST1)].clone()
    }

    pub fn st2_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ST2)].clone()
    }

    pub fn st3_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ST3)].clone()
    }

    pub fn st4_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ST4)].clone()
    }

    pub fn st5_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ST5)].clone()
    }

    pub fn st6_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ST6)].clone()
    }

    pub fn st7_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ST7)].clone()
    }

    pub fn st8_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ST8)].clone()
    }

    pub fn st9_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ST9)].clone()
    }

    pub fn st10_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ST10)].clone()
    }

    pub fn st11_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ST11)].clone()
    }

    pub fn st12_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ST12)].clone()
    }

    pub fn st13_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ST13)].clone()
    }

    pub fn st14_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ST14)].clone()
    }

    pub fn st15_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ST15)].clone()
    }

    pub fn osp_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(OSP)].clone()
    }

    pub fn osv_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(OSV)].clone()
    }

    pub fn ramp_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(RAMP)].clone()
    }

    pub fn ramv_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(RAMV)].clone()
    }

    pub fn is_padding_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(IsPadding)].clone()
    }

    pub fn cjd_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ClockJumpDifference)].clone()
    }

    pub fn invm_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ClockJumpDifferenceInverse)].clone()
    }

    pub fn invu_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(UniqueClockJumpDiffDiffInverse)].clone()
    }

    pub fn rer_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(SelectedClockCyclesEvalArg)].clone()
    }

    pub fn reu_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(UniqueClockJumpDifferencesEvalArg)].clone()
    }

    pub fn rpm_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(AllClockJumpDifferencesPermArg)].clone()
    }

    pub fn running_evaluation_standard_input_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(InputTableEvalArg)].clone()
    }
    pub fn running_evaluation_standard_output_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(OutputTableEvalArg)].clone()
    }
    pub fn running_product_instruction_table_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(InstructionTablePermArg)].clone()
    }
    pub fn running_product_op_stack_table_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(OpStackTablePermArg)].clone()
    }
    pub fn running_product_ram_table_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(RamTablePermArg)].clone()
    }
    pub fn running_product_jump_stack_table_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(JumpStackTablePermArg)].clone()
    }
    pub fn running_evaluation_to_hash_table_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(ToHashTableEvalArg)].clone()
    }
    pub fn running_evaluation_from_hash_table_next(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        self.variables[FULL_WIDTH + usize::from(FromHashTableEvalArg)].clone()
    }

    pub fn decompose_arg(
        &self,
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
        let instruction_pointer_increases_by_one = self.ip_next() - self.ip() - self.one();
        let specific_constraints = vec![instruction_pointer_increases_by_one];
        [specific_constraints, self.keep_jump_stack()].concat()
    }

    pub fn step_2(
        &self,
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
        let instruction_pointer_increases_by_two = self.ip_next() - self.ip() - self.two();
        let specific_constraints = vec![instruction_pointer_increases_by_two];
        [specific_constraints, self.keep_jump_stack()].concat()
    }

    pub fn grow_stack_and_top_two_elements_unconstrained(
        &self,
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
        let specific_constrants = vec![self.st0_next() - self.st1()];
        [specific_constrants, self.binop()].concat()
    }

    pub fn stack_remains_and_top_eleven_elements_unconstrained(
        &self,
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
        let specific_constraints = vec![self.st10_next() - self.st10()];
        [
            specific_constraints,
            self.stack_remains_and_top_eleven_elements_unconstrained(),
        ]
        .concat()
    }

    pub fn stack_remains_and_top_three_elements_unconstrained(
        &self,
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
        let specific_constraints = vec![self.st0_next() - self.st0()];
        [specific_constraints, self.unop()].concat()
    }

    pub fn keep_ram(
        &self,
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
        vec![
            self.ramv_next() - self.ramv(),
            self.ramp_next() - self.ramp(),
        ]
    }

    pub fn running_evaluation_for_standard_input_updates_correctly(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
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
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
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
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
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
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
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
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
        let indeterminate = self.circuit_builder.challenge(RamPermIndeterminate);
        let clk_weight = self.circuit_builder.challenge(RamTableClkWeight);
        let ramp_weight = self.circuit_builder.challenge(RamTableRampWeight);
        let ramv_weight = self.circuit_builder.challenge(RamTableRamvWeight);
        let compressed_row = clk_weight * self.clk_next()
            + ramp_weight * self.ramp_next()
            + ramv_weight * self.ramv_next();

        self.running_product_ram_table_next()
            - self.running_product_ram_table() * (indeterminate - compressed_row)
    }

    pub fn running_product_for_jump_stack_table_updates_correctly(
        &self,
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
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
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
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
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
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
        ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>,
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
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
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
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
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
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, SingleRowIndicator<FULL_WIDTH>> {
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
    ) -> ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>> {
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
        ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>,
    > {
        all_instructions_without_args()
            .into_iter()
            .map(|instrctn| (instrctn, Self::instruction_deselector(factory, instrctn)))
            .collect()
    }
}

impl ExtensionTable for ExtProcessorTable {}

#[cfg(test)]
mod constraint_polynomial_tests {
    use crate::ord_n::Ord16;
    use crate::table::base_matrix::ProcessorMatrixRow;
    use crate::table::challenges::AllChallenges;
    use crate::vm::Program;

    use super::*;

    #[test]
    /// helps identifying whether the printing causes an infinite loop
    fn print_simple_processor_table_row_test() {
        let program = Program::from_code("push 2 push -1 add assert halt").unwrap();
        let (base_matrices, _, _) = program.simulate_no_input();
        for row in base_matrices.processor_matrix {
            println!("{}", ProcessorMatrixRow { row });
        }
    }

    fn get_test_row_from_source_code(source_code: &str, row_num: usize) -> Vec<XFieldElement> {
        let fake_extension_columns = [BFieldElement::zero(); FULL_WIDTH - BASE_WIDTH].to_vec();

        let program = Program::from_code(source_code).unwrap();
        let (base_matrices, _, err) = program.simulate_no_input();
        if let Some(e) = err {
            panic!("The VM crashed because: {}", e);
        }

        let test_row = [
            base_matrices.processor_matrix[row_num].to_vec(),
            fake_extension_columns.clone(),
            base_matrices.processor_matrix[row_num + 1].to_vec(),
            fake_extension_columns,
        ]
        .concat();
        test_row.into_iter().map(|belem| belem.lift()).collect()
    }

    fn get_transition_constraints_for_instruction(
        instruction: Instruction,
    ) -> Vec<ConstraintCircuitMonad<ProcessorTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
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
        test_rows: &[Vec<XFieldElement>],
        debug_cols_curr_row: &[ProcessorBaseTableColumn],
        debug_cols_next_row: &[ProcessorBaseTableColumn],
    ) {
        for (case_idx, test_row) in test_rows.iter().enumerate() {
            // Print debug information
            println!(
                "Testing all constraint polynomials of {} for test row with index {}…",
                instruction, case_idx
            );
            for c in debug_cols_curr_row {
                print!("{} = {}, ", c, test_row[usize::from(*c)]);
            }
            for c in debug_cols_next_row {
                print!("{}' = {}, ", c, test_row[*c as usize + FULL_WIDTH]);
            }
            println!();

            // We need dummy challenges to do partial evaluate. Even though we are
            // not looking at extension constraints, only base constraints
            let dummy_challenges = AllChallenges::placeholder();
            for (poly_idx, poly) in get_transition_constraints_for_instruction(instruction)
                .iter()
                .enumerate()
            {
                assert_eq!(
                    instruction.opcode_b().lift(),
                    test_row[usize::from(CI)],
                    "The test is trying to check the wrong transition constraint polynomials."
                );
                assert_eq!(
                    XFieldElement::zero(),
                    poly.partial_evaluate(&dummy_challenges.processor_table_challenges).evaluate(test_row),
                    "For case {}, transition constraint polynomial with index {} must evaluate to zero.",
                    case_idx,
                    poly_idx,
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

        let mut row = vec![0.into(); 2 * FULL_WIDTH];

        // We need dummy challenges to do partial evaluate. Even though we are
        // not looking at extension constraints, only base constraints
        let dummy_challenges = AllChallenges::placeholder();
        for instruction in all_instructions_without_args() {
            use ProcessorBaseTableColumn::*;
            let deselector = deselectors.get(instruction);

            println!(
                "\n\nThe Deselector for instruction {} is:\n{}",
                instruction, deselector
            );

            // Negative tests
            for other_instruction in all_instructions_without_args()
                .into_iter()
                .filter(|other_instruction| *other_instruction != instruction)
            {
                row[usize::from(IB0)] = other_instruction.ib(Ord7::IB0).lift();
                row[usize::from(IB1)] = other_instruction.ib(Ord7::IB1).lift();
                row[usize::from(IB2)] = other_instruction.ib(Ord7::IB2).lift();
                row[usize::from(IB3)] = other_instruction.ib(Ord7::IB3).lift();
                row[usize::from(IB4)] = other_instruction.ib(Ord7::IB4).lift();
                row[usize::from(IB5)] = other_instruction.ib(Ord7::IB5).lift();
                row[usize::from(IB6)] = other_instruction.ib(Ord7::IB6).lift();
                let result = deselector
                    .partial_evaluate(&dummy_challenges.processor_table_challenges)
                    .evaluate(&row);

                assert!(
                    result.is_zero(),
                    "Deselector for {} should return 0 for all other instructions, including {} whose opcode is {}",
                    instruction,
                    other_instruction,
                    other_instruction.opcode()
                )
            }

            // Positive tests
            row[usize::from(IB0)] = instruction.ib(Ord7::IB0).lift();
            row[usize::from(IB1)] = instruction.ib(Ord7::IB1).lift();
            row[usize::from(IB2)] = instruction.ib(Ord7::IB2).lift();
            row[usize::from(IB3)] = instruction.ib(Ord7::IB3).lift();
            row[usize::from(IB4)] = instruction.ib(Ord7::IB4).lift();
            row[usize::from(IB5)] = instruction.ib(Ord7::IB5).lift();
            row[usize::from(IB6)] = instruction.ib(Ord7::IB6).lift();
            let result = deselector
                .partial_evaluate(&dummy_challenges.processor_table_challenges)
                .evaluate(&row);
            assert!(
                !result.is_zero(),
                "Deselector for {} should be non-zero when CI is {}",
                instruction,
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

        // We need dummy challenges to do partial evaluate. Even though we are
        // not looking at extension constraints, only base constraints
        let dummy_challenges = AllChallenges::placeholder();

        println!("| Instruction     | #polys | max deg | Degrees");
        println!("|:----------------|-------:|--------:|:------------");
        for (instruction, constraints) in all_instructions_and_their_transition_constraints {
            let degrees = constraints
                .iter()
                .map(|circuit| {
                    circuit
                        .partial_evaluate(&dummy_challenges.processor_table_challenges)
                        .degree()
                })
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
