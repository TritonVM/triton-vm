use super::base_table::{self, BaseTable, HasBaseTable, Table};
use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use super::extension_table::ExtensionTable;
use super::table_column::ProcessorTableColumn::{self, *};
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::stark::triton::fri_domain::FriDomain;
use crate::shared_math::stark::triton::instruction::{
    all_instructions_without_args, AnInstruction::*, Instruction,
};
use crate::shared_math::stark::triton::ord_n::Ord6;
use crate::shared_math::stark::triton::state::DIGEST_LEN;
use crate::shared_math::x_field_element::XFieldElement;
use itertools::Itertools;
use std::collections::HashMap;

pub const PROCESSOR_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 9;
pub const PROCESSOR_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 4;
pub const PROCESSOR_TABLE_INITIALS_COUNT: usize =
    PROCESSOR_TABLE_PERMUTATION_ARGUMENTS_COUNT + PROCESSOR_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 47 because it combines all other tables (except program).
pub const PROCESSOR_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 47;

pub const BASE_WIDTH: usize = 36;
/// BASE_WIDTH + 2 * INITIALS_COUNT - 2 (because IOSymbols don't need compression)
pub const FULL_WIDTH: usize = 60;

type BWord = BFieldElement;
type XWord = XFieldElement;

#[derive(Debug, Clone)]
pub struct ProcessorTable {
    base: BaseTable<BWord>,
}

impl HasBaseTable<BWord> for ProcessorTable {
    fn to_base(&self) -> &BaseTable<BWord> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<BWord> {
        &mut self.base
    }
}

impl ProcessorTable {
    pub fn new_prover(num_trace_randomizers: usize, matrix: Vec<Vec<BWord>>) -> Self {
        let unpadded_height = matrix.len();
        let padded_height = base_table::pad_height(unpadded_height, num_trace_randomizers);

        let omicron = base_table::derive_omicron(padded_height as u64);
        let base = BaseTable::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            matrix,
            "ProcessorTable".to_string(),
        );

        Self { base }
    }

    pub fn codeword_table(&self, fri_domain: &FriDomain<BWord>) -> Self {
        let base_columns = 0..self.base_width();
        let codewords =
            self.low_degree_extension(fri_domain, self.num_trace_randomizers(), base_columns);

        let base = self.base.with_data(codewords);
        Self { base }
    }

    pub fn extend(
        &self,
        challenges: &ProcessorTableChallenges,
        initials: &ProcessorTableEndpoints,
    ) -> (ExtProcessorTable, ProcessorTableEndpoints) {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());

        assert_eq!(
            XFieldElement::ring_zero(),
            initials.input_table_eval_sum,
            "The Evaluation Argument's initial for the Input Table must be 0."
        );
        assert_eq!(
            XFieldElement::ring_zero(),
            initials.output_table_eval_sum,
            "The Evaluation Argument's initial for the Output Table must be 0."
        );

        let mut input_table_running_sum = initials.input_table_eval_sum;
        let mut output_table_running_sum = initials.output_table_eval_sum;
        let mut instruction_table_running_product = initials.instruction_table_perm_product;
        let mut opstack_table_running_product = initials.opstack_table_perm_product;
        let mut ram_table_running_product = initials.ram_table_perm_product;
        let mut jump_stack_running_product = initials.jump_stack_perm_product;
        let mut to_hash_table_running_sum = initials.to_hash_table_eval_sum;
        let mut from_hash_table_running_sum = initials.from_hash_table_eval_sum;
        let mut u32_table_lt_running_product = initials.u32_table_lt_perm_product;
        let mut u32_table_and_running_product = initials.u32_table_and_perm_product;
        let mut u32_table_xor_running_product = initials.u32_table_xor_perm_product;
        let mut u32_table_reverse_running_product = initials.u32_table_reverse_perm_product;
        let mut u32_table_div_running_product = initials.u32_table_div_perm_product;

        let mut previous_row: Option<Vec<BFieldElement>> = None;
        for row in self.data().iter() {
            let mut extension_row = Vec::with_capacity(FULL_WIDTH);
            extension_row.extend(row.iter().map(|elem| elem.lift()));

            // Input table
            extension_row.push(input_table_running_sum);
            if let Some(prow) = previous_row.clone() {
                if prow[CI as usize] == Instruction::ReadIo.opcode_b() {
                    let input_symbol = extension_row[ST0 as usize];
                    input_table_running_sum = input_table_running_sum
                        * challenges.input_table_eval_row_weight
                        + input_symbol;
                }
            }

            // Output table
            extension_row.push(output_table_running_sum);
            if row[CI as usize] == Instruction::WriteIo.opcode_b() {
                let output_symbol = extension_row[ST0 as usize];
                output_table_running_sum = output_table_running_sum
                    * challenges.output_table_eval_row_weight
                    + output_symbol;
            }

            // Instruction table
            let ip = extension_row[IP as usize];
            let ci = extension_row[CI as usize];
            let nia = extension_row[NIA as usize];

            let ip_w = challenges.instruction_table_ip_weight;
            let ci_w = challenges.instruction_table_ci_processor_weight;
            let nia_w = challenges.instruction_table_nia_weight;

            let compressed_row_for_instruction_table_permutation_argument =
                ip * ip_w + ci * ci_w + nia * nia_w;
            extension_row.push(compressed_row_for_instruction_table_permutation_argument);

            extension_row.push(instruction_table_running_product);
            instruction_table_running_product *= challenges.instruction_perm_row_weight
                - compressed_row_for_instruction_table_permutation_argument;

            // OpStack table
            let clk = extension_row[CLK as usize];
            let osv = extension_row[OSV as usize];
            let osp = extension_row[OSP as usize];

            let compressed_row_for_op_stack_table_permutation_argument = clk
                * challenges.op_stack_table_clk_weight
                + ci * challenges.op_stack_table_ci_weight
                + osv * challenges.op_stack_table_osv_weight
                + osp * challenges.op_stack_table_osp_weight;
            extension_row.push(compressed_row_for_op_stack_table_permutation_argument);

            extension_row.push(opstack_table_running_product);
            opstack_table_running_product *= challenges.op_stack_perm_row_weight
                - compressed_row_for_op_stack_table_permutation_argument;

            // RAM Table
            let ramv = extension_row[RAMV as usize];
            let ramp = extension_row[ST1 as usize];

            let compressed_row_for_ram_table_permutation_argument = clk
                * challenges.ram_table_clk_weight
                + ramv * challenges.ram_table_ramv_weight
                + ramp * challenges.ram_table_ramp_weight;
            extension_row.push(compressed_row_for_ram_table_permutation_argument);

            extension_row.push(ram_table_running_product);
            ram_table_running_product *=
                challenges.ram_perm_row_weight - compressed_row_for_ram_table_permutation_argument;

            // JumpStack Table
            let jsp = extension_row[JSP as usize];
            let jso = extension_row[JSO as usize];
            let jsd = extension_row[JSD as usize];
            let compressed_row_for_jump_stack_table = clk * challenges.jump_stack_table_clk_weight
                + ci * challenges.jump_stack_table_ci_weight
                + jsp * challenges.jump_stack_table_jsp_weight
                + jso * challenges.jump_stack_table_jso_weight
                + jsd * challenges.jump_stack_table_jsd_weight;
            extension_row.push(compressed_row_for_jump_stack_table);

            extension_row.push(jump_stack_running_product);
            jump_stack_running_product *=
                challenges.jump_stack_perm_row_weight - compressed_row_for_jump_stack_table;

            // Hash Table – Hash's input from Processor to Hash Coprocessor
            let st_0_through_11 = [
                extension_row[ST0 as usize],
                extension_row[ST1 as usize],
                extension_row[ST2 as usize],
                extension_row[ST3 as usize],
                extension_row[ST4 as usize],
                extension_row[ST5 as usize],
                extension_row[ST6 as usize],
                extension_row[ST7 as usize],
                extension_row[ST8 as usize],
                extension_row[ST9 as usize],
                extension_row[ST10 as usize],
                extension_row[ST11 as usize],
            ];
            let compressed_row_for_hash_input = st_0_through_11
                .iter()
                .zip(challenges.hash_table_stack_input_weights.iter())
                .map(|(st, weight)| *weight * *st)
                .fold(XWord::ring_zero(), |sum, summand| sum + summand);
            extension_row.push(compressed_row_for_hash_input);

            extension_row.push(to_hash_table_running_sum);
            if row[CI as usize] == Instruction::Hash.opcode_b() {
                to_hash_table_running_sum = to_hash_table_running_sum
                    * challenges.to_hash_table_eval_row_weight
                    + compressed_row_for_hash_input;
            }

            // Hash Table – Hash's output from Hash Coprocessor to Processor
            let st_0_through_5 = [
                extension_row[ST0 as usize],
                extension_row[ST1 as usize],
                extension_row[ST2 as usize],
                extension_row[ST3 as usize],
                extension_row[ST4 as usize],
                extension_row[ST5 as usize],
            ];
            let compressed_row_for_hash_digest = st_0_through_5
                .iter()
                .zip(challenges.hash_table_digest_output_weights.iter())
                .map(|(st, weight)| *weight * *st)
                .fold(XWord::ring_zero(), |sum, summand| sum + summand);
            extension_row.push(compressed_row_for_hash_digest);

            extension_row.push(from_hash_table_running_sum);
            if let Some(prow) = previous_row.clone() {
                if prow[CI as usize] == Instruction::Hash.opcode_b() {
                    from_hash_table_running_sum = from_hash_table_running_sum
                        * challenges.to_hash_table_eval_row_weight
                        + compressed_row_for_hash_digest;
                }
            }

            // U32 Table
            if let Some(prow) = previous_row {
                let lhs = prow[ST0 as usize].lift();
                let rhs = prow[ST1 as usize].lift();
                let u32_op_result = extension_row[ST0 as usize];

                // less than
                let compressed_row_for_u32_lt = lhs * challenges.u32_op_table_lt_lhs_weight
                    + rhs * challenges.u32_op_table_lt_rhs_weight
                    + u32_op_result * challenges.u32_op_table_lt_result_weight;
                extension_row.push(compressed_row_for_u32_lt);

                extension_row.push(u32_table_lt_running_product);
                if prow[CI as usize] == Instruction::Lt.opcode_b() {
                    u32_table_lt_running_product *=
                        challenges.u32_lt_perm_row_weight - compressed_row_for_u32_lt;
                }

                // and
                let compressed_row_for_u32_and = lhs * challenges.u32_op_table_and_lhs_weight
                    + rhs * challenges.u32_op_table_and_rhs_weight
                    + u32_op_result * challenges.u32_op_table_and_result_weight;
                extension_row.push(compressed_row_for_u32_and);

                extension_row.push(u32_table_and_running_product);
                if prow[CI as usize] == Instruction::And.opcode_b() {
                    u32_table_and_running_product *=
                        challenges.u32_and_perm_row_weight - compressed_row_for_u32_and;
                }

                // xor
                let compressed_row_for_u32_xor = lhs * challenges.u32_op_table_xor_lhs_weight
                    + rhs * challenges.u32_op_table_xor_rhs_weight
                    + u32_op_result * challenges.u32_op_table_xor_result_weight;
                extension_row.push(compressed_row_for_u32_xor);

                extension_row.push(u32_table_xor_running_product);
                if prow[CI as usize] == Instruction::Xor.opcode_b() {
                    u32_table_xor_running_product *=
                        challenges.u32_xor_perm_row_weight - compressed_row_for_u32_xor;
                }

                // reverse
                let compressed_row_for_u32_reverse = lhs
                    * challenges.u32_op_table_reverse_lhs_weight
                    + u32_op_result * challenges.u32_op_table_reverse_result_weight;
                extension_row.push(compressed_row_for_u32_reverse);

                extension_row.push(u32_table_reverse_running_product);
                if prow[CI as usize] == Instruction::Reverse.opcode_b() {
                    u32_table_reverse_running_product *=
                        challenges.u32_reverse_perm_row_weight - compressed_row_for_u32_reverse;
                }

                // div
                let divisor = prow[ST0 as usize].lift();
                let remainder = extension_row[ST0 as usize];
                let lt_for_div_result = extension_row[HV0 as usize];
                let compressed_row_for_u32_div = divisor
                    * challenges.u32_op_table_div_divisor_weight
                    + remainder * challenges.u32_op_table_div_remainder_weight
                    + lt_for_div_result * challenges.u32_op_table_div_result_weight;
                extension_row.push(compressed_row_for_u32_div);

                extension_row.push(u32_table_div_running_product);
                if prow[CI as usize] == Instruction::Div.opcode_b() {
                    u32_table_div_running_product *=
                        challenges.u32_lt_perm_row_weight - compressed_row_for_u32_div;
                }
            } else {
                // If there is no previous row, none of the u32 operations make sense. The extension
                // columns must still be filled in. All stack registers are initialized to 0, and
                // the stack in the non-existing previous row can be safely assumed to be all 0.
                // Thus, all the compressed_row-values are 0 for the very first extension_row.
                // The running products can be used as-are, amounting to pushing the initials.
                let zero = XFieldElement::ring_zero();
                extension_row.push(zero);
                extension_row.push(u32_table_lt_running_product);
                extension_row.push(zero);
                extension_row.push(u32_table_and_running_product);
                extension_row.push(zero);
                extension_row.push(u32_table_xor_running_product);
                extension_row.push(zero);
                extension_row.push(u32_table_reverse_running_product);
                extension_row.push(zero);
                extension_row.push(u32_table_div_running_product);
            }

            debug_assert_eq!(
                FULL_WIDTH,
                extension_row.len(),
                "After extending, the row must match the table's full width."
            );
            previous_row = Some(row.clone());
            extension_matrix.push(extension_row);
        }

        let base = self.base.with_lifted_data(extension_matrix);
        let table = ExtProcessorTable::new(base);
        let terminals = ProcessorTableEndpoints {
            input_table_eval_sum: input_table_running_sum,
            output_table_eval_sum: output_table_running_sum,
            instruction_table_perm_product: instruction_table_running_product,
            opstack_table_perm_product: opstack_table_running_product,
            ram_table_perm_product: ram_table_running_product,
            jump_stack_perm_product: jump_stack_running_product,
            to_hash_table_eval_sum: to_hash_table_running_sum,
            from_hash_table_eval_sum: from_hash_table_running_sum,
            u32_table_lt_perm_product: u32_table_lt_running_product,
            u32_table_and_perm_product: u32_table_and_running_product,
            u32_table_xor_perm_product: u32_table_xor_running_product,
            u32_table_reverse_perm_product: u32_table_reverse_running_product,
            u32_table_div_perm_product: u32_table_div_running_product,
        };

        (table, terminals)
    }
}

impl ExtProcessorTable {
    pub fn with_padded_height(num_trace_randomizers: usize, padded_height: usize) -> Self {
        let matrix: Vec<Vec<XWord>> = vec![];

        let omicron = base_table::derive_omicron(padded_height as u64);
        let base = BaseTable::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            matrix,
            "ExtProcessorTable".to_string(),
        );

        Self::new(base)
    }

    pub fn ext_codeword_table(
        &self,
        fri_domain: &FriDomain<XWord>,
        base_codewords: &[Vec<BWord>],
    ) -> Self {
        // Extension Tables do not have a randomized trace
        let num_trace_randomizers = 0;
        let ext_columns = self.base_width()..self.full_width();
        let ext_codewords =
            self.low_degree_extension(fri_domain, num_trace_randomizers, ext_columns);

        let lifted_base_codewords = base_codewords
            .iter()
            .map(|base_codeword| base_codeword.iter().map(|bfe| bfe.lift()).collect_vec())
            .collect_vec();
        let all_codewords = vec![lifted_base_codewords, ext_codewords].concat();
        assert_eq!(self.full_width(), all_codewords.len());

        let base = self.base.with_data(all_codewords);
        Self::new(base)
    }

    pub fn new(base: BaseTable<XFieldElement>) -> ExtProcessorTable {
        Self {
            base,
            transition_constraints: TransitionConstraints::default(),
            consistency_boundary_constraints: ConsistencyBoundaryConstraints::default(),
            instruction_deselectors: InstructionDeselectors::default(),
        }
    }

    /// Transition constraints are combined with deselectors in such a way
    /// that arbitrary sets of mutually exclusive combinations are summed, i.e.,
    ///
    /// ```norun
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
    fn combine_transition_constraints_with_deselectors(
        &self,
        instr_tc_polys_tuples: Vec<(Instruction, Vec<MPolynomial<XWord>>)>,
    ) -> Vec<MPolynomial<XWord>> {
        let (all_instructions, all_tc_polys_for_all_instructions): (Vec<_>, Vec<Vec<_>>) =
            instr_tc_polys_tuples.into_iter().unzip();

        let all_instruction_deselectors = all_instructions
            .into_iter()
            .map(|instr| self.instruction_deselectors.get(instr))
            .collect_vec();

        let max_number_of_constraints = all_tc_polys_for_all_instructions
            .iter()
            .map(|tc_polys_for_instr| tc_polys_for_instr.len())
            .max()
            .unwrap();
        let zero_poly = self.transition_constraints.zero();

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
}

#[derive(Debug, Clone)]
pub struct ProcessorTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the processor table.
    pub input_table_eval_row_weight: XFieldElement,
    pub output_table_eval_row_weight: XFieldElement,
    pub to_hash_table_eval_row_weight: XFieldElement,
    pub from_hash_table_eval_row_weight: XFieldElement,

    pub instruction_perm_row_weight: XFieldElement,
    pub op_stack_perm_row_weight: XFieldElement,
    pub ram_perm_row_weight: XFieldElement,
    pub jump_stack_perm_row_weight: XFieldElement,

    pub u32_lt_perm_row_weight: XFieldElement,
    pub u32_and_perm_row_weight: XFieldElement,
    pub u32_xor_perm_row_weight: XFieldElement,
    pub u32_reverse_perm_row_weight: XFieldElement,
    pub u32_div_perm_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub instruction_table_ip_weight: XFieldElement,
    pub instruction_table_ci_processor_weight: XFieldElement,
    pub instruction_table_nia_weight: XFieldElement,

    pub op_stack_table_clk_weight: XFieldElement,
    pub op_stack_table_ci_weight: XFieldElement,
    pub op_stack_table_osv_weight: XFieldElement,
    pub op_stack_table_osp_weight: XFieldElement,

    pub ram_table_clk_weight: XFieldElement,
    pub ram_table_ramv_weight: XFieldElement,
    pub ram_table_ramp_weight: XFieldElement,

    pub jump_stack_table_clk_weight: XFieldElement,
    pub jump_stack_table_ci_weight: XFieldElement,
    pub jump_stack_table_jsp_weight: XFieldElement,
    pub jump_stack_table_jso_weight: XFieldElement,
    pub jump_stack_table_jsd_weight: XFieldElement,

    pub hash_table_stack_input_weights: [XFieldElement; 2 * DIGEST_LEN],
    pub hash_table_digest_output_weights: [XFieldElement; DIGEST_LEN],

    pub u32_op_table_lt_lhs_weight: XFieldElement,
    pub u32_op_table_lt_rhs_weight: XFieldElement,
    pub u32_op_table_lt_result_weight: XFieldElement,

    pub u32_op_table_and_lhs_weight: XFieldElement,
    pub u32_op_table_and_rhs_weight: XFieldElement,
    pub u32_op_table_and_result_weight: XFieldElement,

    pub u32_op_table_xor_lhs_weight: XFieldElement,
    pub u32_op_table_xor_rhs_weight: XFieldElement,
    pub u32_op_table_xor_result_weight: XFieldElement,

    pub u32_op_table_reverse_lhs_weight: XFieldElement,
    pub u32_op_table_reverse_result_weight: XFieldElement,

    pub u32_op_table_div_divisor_weight: XFieldElement,
    pub u32_op_table_div_remainder_weight: XFieldElement,
    pub u32_op_table_div_result_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct ProcessorTableEndpoints {
    pub input_table_eval_sum: XFieldElement,
    pub output_table_eval_sum: XFieldElement,

    pub instruction_table_perm_product: XFieldElement,
    pub opstack_table_perm_product: XFieldElement,
    pub ram_table_perm_product: XFieldElement,
    pub jump_stack_perm_product: XFieldElement,

    pub to_hash_table_eval_sum: XFieldElement,
    pub from_hash_table_eval_sum: XFieldElement,

    pub u32_table_lt_perm_product: XFieldElement,
    pub u32_table_and_perm_product: XFieldElement,
    pub u32_table_xor_perm_product: XFieldElement,
    pub u32_table_reverse_perm_product: XFieldElement,
    pub u32_table_div_perm_product: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct IOChallenges {
    /// The weight that combines the eval arg's running sum with the next i/o symbol in the i/o list
    pub processor_eval_row_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct ExtProcessorTable {
    base: BaseTable<XFieldElement>,
    transition_constraints: TransitionConstraints,
    consistency_boundary_constraints: ConsistencyBoundaryConstraints,
    instruction_deselectors: InstructionDeselectors,
}

impl HasBaseTable<XFieldElement> for ExtProcessorTable {
    fn to_base(&self) -> &BaseTable<XFieldElement> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<XFieldElement> {
        &mut self.base
    }
}

impl Table<BWord> for ProcessorTable {
    fn get_padding_row(&self) -> Vec<BWord> {
        let mut padding_row = self.data().last().unwrap().clone();
        padding_row[ProcessorTableColumn::CLK as usize] += 1.into();
        padding_row
    }
}

impl Table<XFieldElement> for ExtProcessorTable {
    fn get_padding_row(&self) -> Vec<XFieldElement> {
        panic!("Extension tables don't get padded");
    }
}

impl ExtensionTable for ExtProcessorTable {
    fn ext_boundary_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        let factory = &self.consistency_boundary_constraints;

        // The cycle counter `clk` is 0.
        //
        // $clk = 0  ⇒  clk - 0 = 0  ⇒  clk - 0  ⇒  clk$
        let clk_is_0 = factory.clk();

        // The instruction pointer `ip` is 0.
        //
        // $ip = 0  ⇒  ip - 0 = 0  ⇒  ip - 0  ⇒  ip$
        let ip_is_0 = factory.ip();

        // The jump address stack pointer `jsp` is 0.
        //
        // $jsp = 0  ⇒  jsp - 0 == 0  ⇒  jsp - 0  ⇒  jsp$
        let jsp_is_0 = factory.jsp();

        // The jump address origin `jso` is 0.
        //
        // $jso = 0  ⇒  jso - 0 = 0  ⇒  jso - 0  ⇒  jso$
        let jso_is_0 = factory.jso();

        // The jump address destination `jsd` is 0.
        //
        // $jsd = 0  ⇒  jsd - 0 = 0  ⇒  jsd - 0  ⇒  jsd$
        let jsd_is_0 = factory.jsd();

        // The operational stack element `st0` is 0.
        //
        // $st0 = 0  ⇒  st0 - 0 = 0  ⇒  st0 - 0  ⇒  st0$
        let st0_is_0 = factory.st0();

        // The operational stack element `st1` is 0.
        //
        // $st1 = 0  ⇒  st1 - 0 = 0  ⇒  st1 - 0  ⇒  st1$
        let st1_is_0 = factory.st1();

        // The operational stack element `st2` is 0.
        //
        // $st2 = 0  ⇒  st2 - 0 = 0  ⇒  st2 - 0  ⇒  st2$
        let st2_is_0 = factory.st2();

        // The operational stack element `st3` is 0.
        //
        // $st3 = 0  ⇒  st3 - 0 = 0  ⇒  st3 - 0  ⇒  st3$
        let st3_is_0 = factory.st3();

        // The operational stack element `st4` is 0.
        //
        // $st4 = 0  ⇒  st4 - 0 = 0  ⇒  st4 - 0  ⇒  st4$
        let st4_is_0 = factory.st4();

        // The operational stack element `st5` is 0.
        //
        // $st5 = 0  ⇒  st5 - 0 = 0  ⇒  st5 - 0  ⇒  st5$
        let st5_is_0 = factory.st5();

        // The operational stack element `st6` is 0.
        //
        // $st6 = 0  ⇒  st6 - 0 = 0  ⇒  st6 - 0  ⇒  st6$
        let st6_is_0 = factory.st6();

        // The operational stack element `st7` is 0.
        //
        // $st7 = 0  ⇒  st7 - 0 = 0  ⇒  st7 - 0  ⇒  st7$
        let st7_is_0 = factory.st7();

        // The operational stack element `st8` is 0.
        //
        // $st8 = 0  ⇒  st8 - 0 = 0  ⇒  st8 - 0  ⇒  st8$
        let st8_is_0 = factory.st8();

        // The operational stack element `st9` is 0.
        //
        // $st9 = 0  ⇒  st9 - 0 = 0  ⇒  st9 - 0  ⇒  st9$
        let st9_is_0 = factory.st9();

        // The operational stack element `st10` is 0.
        //
        // $st10 = 0  ⇒  st10 - 0 = 0  ⇒  st10 - 0  ⇒  st10$
        let st10_is_0 = factory.st10();

        // The operational stack element `st11` is 0.
        //
        // $st11 = 0  ⇒  st11 - 0 = 0  ⇒  st11 - 0  ⇒  st11$
        let st11_is_0 = factory.st11();

        // The operational stack element `st12` is 0.
        //
        // $st12 = 0  ⇒  st12 - 0 = 0  ⇒  st12 - 0  ⇒  st12$
        let st12_is_0 = factory.st12();

        // The operational stack element `st13` is 0.
        //
        // $st13 = 0  ⇒  st13 - 0 = 0  ⇒  st13 - 0  ⇒  st13$
        let st13_is_0 = factory.st13();

        // The operational stack element `st14` is 0.
        //
        // $st14 = 0  ⇒  st14 - 0 = 0  ⇒  st14 - 0  ⇒  st14$
        let st14_is_0 = factory.st14();

        // The operational stack element `st15` is 0.
        //
        // $st15 = 0  ⇒  st15 - 0 = 0  ⇒  st15 - 0  ⇒  st15$
        let st15_is_0 = factory.st15();

        // The operational stack pointer `osp` is 16.
        //
        // $osp = 16  ⇒  osp - 16 == 0  ⇒  osp - 16$
        let osp_is_16 = factory.osp() - factory.constant(16);

        // The operational stack value `osv` is 0.
        //
        // $osv = 0  ⇒  osv - 0 = 0  ⇒  osv - 0  ⇒  osv$
        let osv_is_0 = factory.osv();

        // The RAM value ramv is 0.
        //
        // $ramv = 0  ⇒  ramv - 0 = 0  ⇒  ramv$
        let ramv_is_0 = factory.ramv();

        vec![
            clk_is_0, ip_is_0, jsp_is_0, jso_is_0, jsd_is_0, st0_is_0, st1_is_0, st2_is_0,
            st3_is_0, st4_is_0, st5_is_0, st6_is_0, st7_is_0, st8_is_0, st9_is_0, st10_is_0,
            st11_is_0, st12_is_0, st13_is_0, st14_is_0, st15_is_0, osp_is_16, osv_is_0, ramv_is_0,
        ]
    }

    fn ext_consistency_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        let factory = &self.consistency_boundary_constraints;

        // The composition of instruction buckets ib0-ib5 corresponds the current instruction ci.
        //
        // $ci - (2^5·ib5 + 2^4·ib4 + 2^3·ib3 + 2^2·ib2 + 2^1·ib1 + 2^0·ib0) = 0$
        let ci_corresponds_to_ib0_thru_ib5 = {
            let mut ib_composition = factory.one() * factory.ib0();
            ib_composition += factory.constant(2) * factory.ib1();
            ib_composition += factory.constant(4) * factory.ib2();
            ib_composition += factory.constant(8) * factory.ib3();
            ib_composition += factory.constant(16) * factory.ib4();
            ib_composition += factory.constant(32) * factory.ib5();

            factory.ci() - ib_composition
        };

        vec![ci_corresponds_to_ib0_thru_ib5]
    }

    fn ext_transition_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        let factory = &self.transition_constraints;

        let all_instruction_transition_constraints = vec![
            (Pop, factory.instruction_pop()),
            (Push(Default::default()), factory.instruction_push()),
            (Divine, factory.instruction_divine()),
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
            (Lt, factory.instruction_lt()),
            (And, factory.instruction_and()),
            (Xor, factory.instruction_xor()),
            (Reverse, factory.instruction_reverse()),
            (Div, factory.instruction_div()),
            (XxAdd, factory.instruction_xxadd()),
            (XxMul, factory.instruction_xxmul()),
            (XInvert, factory.instruction_xinv()),
            (XbMul, factory.instruction_xbmul()),
            (ReadIo, factory.instruction_read_io()),
            (WriteIo, factory.instruction_write_io()),
        ];

        let mut transition_constraints = self.combine_transition_constraints_with_deselectors(
            all_instruction_transition_constraints,
        );
        transition_constraints.insert(0, factory.clk_always_increases_by_one());
        transition_constraints
    }

    fn ext_terminal_constraints(
        &self,
        _challenges: &AllChallenges,
        _terminals: &AllEndpoints,
    ) -> Vec<MPolynomial<XWord>> {
        let factory = ConsistencyBoundaryConstraints::default();

        // In the last row, current instruction register ci is 0, corresponding to instruction halt.
        //
        // $ci - halt = 0  =>  ci - 0 = 0  =>  ci$
        let last_ci_is_halt = factory.ci();

        vec![last_ci_is_halt]
    }
}

#[derive(Debug, Clone)]
pub struct ConsistencyBoundaryConstraints {
    variables: [MPolynomial<XWord>; FULL_WIDTH],
}

impl Default for ConsistencyBoundaryConstraints {
    fn default() -> Self {
        let variables = MPolynomial::variables(FULL_WIDTH, 1.into())
            .try_into()
            .expect("Create variables for boundary/consistency constraints");

        Self { variables }
    }
}

impl ConsistencyBoundaryConstraints {
    // FIXME: This does not need a self reference.
    pub fn constant(&self, constant: u32) -> MPolynomial<XWord> {
        MPolynomial::from_constant(constant.into(), FULL_WIDTH)
    }

    pub fn one(&self) -> MPolynomial<XWord> {
        self.constant(1)
    }

    pub fn two(&self) -> MPolynomial<XWord> {
        self.constant(2)
    }

    pub fn clk(&self) -> MPolynomial<XWord> {
        self.variables[CLK as usize].clone()
    }

    pub fn ip(&self) -> MPolynomial<XWord> {
        self.variables[IP as usize].clone()
    }

    pub fn ci(&self) -> MPolynomial<XWord> {
        self.variables[CI as usize].clone()
    }

    pub fn nia(&self) -> MPolynomial<XWord> {
        self.variables[NIA as usize].clone()
    }

    pub fn ib0(&self) -> MPolynomial<XWord> {
        self.variables[IB0 as usize].clone()
    }

    pub fn ib1(&self) -> MPolynomial<XWord> {
        self.variables[IB1 as usize].clone()
    }

    pub fn ib2(&self) -> MPolynomial<XWord> {
        self.variables[IB2 as usize].clone()
    }

    pub fn ib3(&self) -> MPolynomial<XWord> {
        self.variables[IB3 as usize].clone()
    }

    pub fn ib4(&self) -> MPolynomial<XWord> {
        self.variables[IB4 as usize].clone()
    }

    pub fn ib5(&self) -> MPolynomial<XWord> {
        self.variables[IB5 as usize].clone()
    }

    pub fn jsp(&self) -> MPolynomial<XWord> {
        self.variables[JSP as usize].clone()
    }

    pub fn jsd(&self) -> MPolynomial<XWord> {
        self.variables[JSD as usize].clone()
    }

    pub fn jso(&self) -> MPolynomial<XWord> {
        self.variables[JSO as usize].clone()
    }

    pub fn st0(&self) -> MPolynomial<XWord> {
        self.variables[ST0 as usize].clone()
    }

    pub fn st1(&self) -> MPolynomial<XWord> {
        self.variables[ST1 as usize].clone()
    }

    pub fn st2(&self) -> MPolynomial<XWord> {
        self.variables[ST2 as usize].clone()
    }

    pub fn st3(&self) -> MPolynomial<XWord> {
        self.variables[ST3 as usize].clone()
    }

    pub fn st4(&self) -> MPolynomial<XWord> {
        self.variables[ST4 as usize].clone()
    }

    pub fn st5(&self) -> MPolynomial<XWord> {
        self.variables[ST5 as usize].clone()
    }

    pub fn st6(&self) -> MPolynomial<XWord> {
        self.variables[ST6 as usize].clone()
    }

    pub fn st7(&self) -> MPolynomial<XWord> {
        self.variables[ST7 as usize].clone()
    }

    pub fn st8(&self) -> MPolynomial<XWord> {
        self.variables[ST8 as usize].clone()
    }

    pub fn st9(&self) -> MPolynomial<XWord> {
        self.variables[ST9 as usize].clone()
    }

    pub fn st10(&self) -> MPolynomial<XWord> {
        self.variables[ST10 as usize].clone()
    }

    pub fn st11(&self) -> MPolynomial<XWord> {
        self.variables[ST11 as usize].clone()
    }

    pub fn st12(&self) -> MPolynomial<XWord> {
        self.variables[ST12 as usize].clone()
    }

    pub fn st13(&self) -> MPolynomial<XWord> {
        self.variables[ST13 as usize].clone()
    }

    pub fn st14(&self) -> MPolynomial<XWord> {
        self.variables[ST14 as usize].clone()
    }

    pub fn st15(&self) -> MPolynomial<XWord> {
        self.variables[ST15 as usize].clone()
    }

    pub fn osp(&self) -> MPolynomial<XWord> {
        self.variables[OSP as usize].clone()
    }

    pub fn osv(&self) -> MPolynomial<XWord> {
        self.variables[OSV as usize].clone()
    }

    pub fn hv0(&self) -> MPolynomial<XWord> {
        self.variables[HV0 as usize].clone()
    }

    pub fn hv1(&self) -> MPolynomial<XWord> {
        self.variables[HV1 as usize].clone()
    }

    pub fn hv2(&self) -> MPolynomial<XWord> {
        self.variables[HV2 as usize].clone()
    }

    pub fn hv3(&self) -> MPolynomial<XWord> {
        self.variables[HV3 as usize].clone()
    }

    pub fn ramv(&self) -> MPolynomial<XWord> {
        self.variables[RAMV as usize].clone()
    }
}

#[derive(Debug, Clone)]
pub struct TransitionConstraints {
    variables: [MPolynomial<XWord>; 2 * FULL_WIDTH],
}

impl Default for TransitionConstraints {
    fn default() -> Self {
        let variables = MPolynomial::variables(2 * FULL_WIDTH, 1.into())
            .try_into()
            .expect("Create variables for transition constraints");

        Self { variables }
    }
}

impl TransitionConstraints {
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
    pub fn clk_always_increases_by_one(&self) -> MPolynomial<XWord> {
        let one = self.one();
        let clk = self.clk();
        let clk_next = self.clk_next();

        clk_next - clk - one
    }

    pub fn indicator_polynomial(&self, i: usize) -> MPolynomial<XWord> {
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

    pub fn instruction_pop(&self) -> Vec<MPolynomial<XWord>> {
        // no further constraints
        vec![]
    }

    /// push'es argument should be on the stack after execution
    /// $st0_next == nia  =>  st0_next - nia == 0$
    pub fn instruction_push(&self) -> Vec<MPolynomial<XWord>> {
        let st0_next = self.st0_next();
        let nia = self.nia();

        vec![st0_next - nia]
    }

    pub fn instruction_divine(&self) -> Vec<MPolynomial<XWord>> {
        // no further constraints
        vec![]
    }

    pub fn instruction_dup(&self) -> Vec<MPolynomial<XWord>> {
        vec![
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
        ]
    }

    pub fn instruction_swap(&self) -> Vec<MPolynomial<XWord>> {
        vec![
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
            (self.one() - self.indicator_polynomial(1)) * (self.ramv_next() - self.ramv()),
        ]
    }

    pub fn instruction_nop(&self) -> Vec<MPolynomial<XWord>> {
        // no further constraints
        vec![]
    }

    pub fn instruction_skiz(&self) -> Vec<MPolynomial<XWord>> {
        // The jump stack pointer jsp does not change.
        let jsp_does_not_change = self.jsp_next() - self.jsp();

        // The last jump's origin jso does not change.
        let jso_does_not_change = self.jso_next() - self.jso();

        // The last jump's destination jsd does not change.
        let jsd_does_not_change = self.jsd_next() - self.jsd();

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

        vec![
            jsp_does_not_change,
            jso_does_not_change,
            jsd_does_not_change,
            nia_decomposes_to_hvs,
            hv0_is_0_or_1,
            ip_incr_by_1_or_2_or_3,
        ]
    }

    pub fn instruction_call(&self) -> Vec<MPolynomial<XWord>> {
        // The jump stack pointer jsp is incremented by 1.
        let jsp_incr_1 = self.jsp_next() - (self.jsp() + self.one());

        // The jump's origin jso is set to the current instruction pointer ip plus 2.
        let jso_becomes_ip_plus_2 = self.jso_next() - (self.ip() + self.two());

        // The jump's destination jsd is set to the instruction's argument.
        let jsd_becomes_nia = self.jsd_next() - self.nia();

        // The instruction pointer ip is set to the instruction's argument.
        let ip_becomes_nia = self.ip_next() - self.nia();

        vec![
            jsp_incr_1,
            jso_becomes_ip_plus_2,
            jsd_becomes_nia,
            ip_becomes_nia,
        ]
    }

    pub fn instruction_return(&self) -> Vec<MPolynomial<XWord>> {
        // The jump stack pointer jsp is decremented by 1.
        let jsp_incr_1 = self.jsp_next() - (self.jsp() - self.one());

        // The instruction pointer ip is set to the last call's origin jso.
        let ip_becomes_jso = self.ip_next() - self.jso();

        vec![jsp_incr_1, ip_becomes_jso]
    }

    pub fn instruction_recurse(&self) -> Vec<MPolynomial<XWord>> {
        // The jump stack pointer jsp does not change.
        let jsp_does_not_change = self.jsp_next() - self.jsp();

        // The last jump's origin jso does not change.
        let jso_does_not_change = self.jso_next() - self.jso();

        // The last jump's destination jsd does not change.
        let jsd_does_not_change = self.jsd_next() - self.jsd();

        // The instruction pointer ip is set to the last jump's destination jsd.
        let ip_becomes_jsd = self.ip_next() - self.jsd();

        vec![
            jsp_does_not_change,
            jso_does_not_change,
            jsd_does_not_change,
            ip_becomes_jsd,
        ]
    }

    pub fn instruction_assert(&self) -> Vec<MPolynomial<XWord>> {
        // The current top of the stack st0 is 1.
        let st_0_is_1 = self.st0() - self.one();

        vec![st_0_is_1]
    }

    pub fn instruction_halt(&self) -> Vec<MPolynomial<XWord>> {
        // The instruction executed in the following step is instruction halt.
        let halt_is_followed_by_halt = self.ci_next() - self.ci();

        vec![halt_is_followed_by_halt]
    }

    pub fn instruction_read_mem(&self) -> Vec<MPolynomial<XWord>> {
        // The top of the stack is overwritten with the RAM value.
        let st0_becomes_ramv = self.st0_next() - self.ramv();

        vec![st0_becomes_ramv]
    }

    pub fn instruction_write_mem(&self) -> Vec<MPolynomial<XWord>> {
        // The RAM value is overwritten with the top of the stack.
        let ramv_becomes_st0 = self.ramv_next() - self.st0();

        vec![ramv_becomes_st0]
    }

    /// Two Evaluation Arguments with the Hash Table guarantee correct transition.
    pub fn instruction_hash(&self) -> Vec<MPolynomial<XWord>> {
        // no further constraints
        vec![]
    }

    /// Recall that in a Merkle tree, the indices of left (respectively right)
    /// leafs have 0 (respectively 1) as their least significant bit. The first
    /// two polynomials achieve that helper variable hv0 holds the result of
    /// st12 mod 2. The third polynomial sets the new value of st12 to st12 div 2.
    pub fn instruction_divine_sibling(&self) -> Vec<MPolynomial<XWord>> {
        // Helper variable hv0 is either 0 or 1.
        let hv0_is_0_or_1 = self.hv0() * (self.hv0() - self.one());

        // The 13th stack element decomposes into helper variables hv1 and hv0.
        let st12_decomposes_to_hvs = self.st12() - (self.two() * self.hv1() + self.hv0());

        // The 13th stack register is shifted by 1 bit to the right.
        let st12_becomes_shifted_1_bit_right = self.st12_next() - self.hv1();

        // If hv0 is 0, then st0-st5 contains a left sibling in a Merkle tree and so does not change.
        let left_siblings_remain_left = vec![
            // If hv0 is 0, then st0 does not change.
            (self.one() - self.hv0()) * (self.st0_next() - self.st0()),
            // If hv0 is 0, then st1 does not change.
            (self.one() - self.hv0()) * (self.st1_next() - self.st1()),
            // If hv0 is 0, then st2 does not change.
            (self.one() - self.hv0()) * (self.st2_next() - self.st2()),
            // If hv0 is 0, then st3 does not change.
            (self.one() - self.hv0()) * (self.st3_next() - self.st3()),
            // If hv0 is 0, then st4 does not change.
            (self.one() - self.hv0()) * (self.st4_next() - self.st4()),
            // If hv0 is 0, then st5 does not change.
            (self.one() - self.hv0()) * (self.st5_next() - self.st5()),
        ];

        // If hv0 is 1, then st0-st5 contains a right sibling in a Merkle tree and so are copied to st6-st11.
        let right_siblings_are_copied_right = vec![
            // If hv0 is 1, then st0 is copied to st6.
            self.hv0() * (self.st6_next() - self.st0()),
            // If hv0 is 1, then st1 is copied to st7.
            self.hv0() * (self.st7_next() - self.st1()),
            // If hv0 is 1, then st2 is copied to st8.
            self.hv0() * (self.st8_next() - self.st2()),
            // If hv0 is 1, then st3 is copied to st9.
            self.hv0() * (self.st9_next() - self.st3()),
            // If hv0 is 1, then st4 is copied to st10.
            self.hv0() * (self.st10_next() - self.st4()),
            // If hv0 is 1, then st5 is copied to st11.
            self.hv0() * (self.st11_next() - self.st5()),
        ];

        // The stack element in st13 does not change.
        let st13_does_not_change = self.st13_next() - self.st13();

        // The stack element in st14 does not change.
        let st14_does_not_change = self.st14_next() - self.st14();

        // The stack element in st15 does not change.
        let st15_does_not_change = self.st15_next() - self.st15();

        // The top of the OpStack underflow, i.e., osv, does not change.
        let osv_does_not_change = self.osv_next() - self.osv();

        // The OpStack pointer does not change.
        let osp_does_not_change = self.osp_next() - self.osp();

        // If hv0 is 0, then the RAM value ramv does not change.
        let ramv_does_not_change_when_hv0_is_0 =
            (self.one() - self.hv0()) * (self.ramv_next() - self.ramv());

        vec![
            vec![
                hv0_is_0_or_1,
                st12_decomposes_to_hvs,
                st12_becomes_shifted_1_bit_right,
            ],
            left_siblings_remain_left,
            right_siblings_are_copied_right,
            vec![
                st13_does_not_change,
                st14_does_not_change,
                st15_does_not_change,
            ],
            vec![
                osv_does_not_change,
                osp_does_not_change,
                ramv_does_not_change_when_hv0_is_0,
            ],
        ]
        .concat()
    }

    pub fn instruction_assert_vector(&self) -> Vec<MPolynomial<XWord>> {
        vec![
            // Register st0 is equal to st6.
            // $st6 - st0 = 0$
            self.st6() - self.st0(),
            // Register st1 is equal to st7.
            // $st7 - st1 = 0$
            self.st6() - self.st0(),
            // Register st2 is equal to st8.
            // $st8 - st2 = 0$
            self.st6() - self.st0(),
            // Register st3 is equal to st9.
            // $st9 - st3 = 0$
            self.st6() - self.st0(),
            // Register st4 is equal to st10.
            // $st10 - st4 = 0$
            self.st6() - self.st0(),
            // Register st5 is equal to st11.
            // $st11 - st5 = 0$
            self.st6() - self.st0(),
        ]
    }

    /// The sum of the top two stack elements is moved into the top of the stack.
    ///
    /// $st0' - (st0 + st1) = 0$
    pub fn instruction_add(&self) -> Vec<MPolynomial<XWord>> {
        vec![self.st0_next() - (self.st0() + self.st1())]
    }

    /// The product of the top two stack elements is moved into the top of the stack.
    ///
    /// $st0' - (st0 * st1) = 0$
    pub fn instruction_mul(&self) -> Vec<MPolynomial<XWord>> {
        vec![self.st0_next() - (self.st0() * self.st1())]
    }

    /// The top of the stack's inverse is moved into the top of the stack.
    ///
    /// $st0'·st0 - 1 = 0$
    pub fn instruction_invert(&self) -> Vec<MPolynomial<XWord>> {
        vec![self.st0_next() * self.st0() - self.one()]
    }

    pub fn instruction_split(&self) -> Vec<MPolynomial<XWord>> {
        let two_pow_32 = self.constant_b(BWord::new(2u64.pow(32)));

        // The top of the stack is decomposed as 32-bit chunks into the stack's top-most two elements.
        //
        // $st0 - (2^32·st0' + st1') = 0$
        let st0_decomposes_to_two_32_bit_chunks =
            self.st0() - (two_pow_32.clone() * self.st0_next() + self.st1_next());

        // Helper variable `hv0` = 0 either if `hv0` is the difference between
        // (2^32 - 1) and the high 32 bits in `st0'`, or if the low 32 bits in
        // `st1'` are 0.
        //
        // $st1'·(hv0·(st0' - (2^32 - 1)) - 1) = 0$
        let hv0_holds_inverse_of_chunk_difference_or_low_bits_are_0 = {
            let diff = self.st0_next() - (two_pow_32 - self.one());

            self.st1_next() * (self.hv0() * diff - self.one())
        };

        // Stack register st1 is moved into st2
        //
        // $st2' - st1 = 0$
        let st2_becomes_st1 = self.st2_next() - self.st1();

        // Stack register st2 is moved into st3
        //
        // $st3' - st2 = 0$
        let st3_becomes_st2 = self.st3_next() - self.st2();

        // Stack register st3 is moved into st4
        //
        // $st4' - st3 = 0$
        let st4_becomes_st3 = self.st4_next() - self.st3();

        // Stack register st4 is moved into st5
        //
        // $st5' - st4 = 0$
        let st5_becomes_st4 = self.st5_next() - self.st4();

        // Stack register st5 is moved into st6
        //
        // $st6' - st5 = 0$
        let st6_becomes_st5 = self.st6_next() - self.st5();

        // Stack register st6 is moved into st7
        //
        // $st7' - st6 = 0$
        let st7_becomes_st6 = self.st7_next() - self.st6();

        // Stack register st7 is moved into osv
        //
        // $osv' - st7 = 0$
        let osv_becomes_st7 = self.osv_next() - self.st7();

        // The stack pointer increases by 1.
        //
        // $osp' - (osp + 1) = 0$
        let osp_is_incremented = self.osp_next() - (self.osp() + self.one());

        vec![
            st0_decomposes_to_two_32_bit_chunks,
            hv0_holds_inverse_of_chunk_difference_or_low_bits_are_0,
            st2_becomes_st1,
            st3_becomes_st2,
            st4_becomes_st3,
            st5_becomes_st4,
            st6_becomes_st5,
            st7_becomes_st6,
            osv_becomes_st7,
            osp_is_incremented,
        ]
    }

    pub fn instruction_eq(&self) -> Vec<MPolynomial<XWord>> {
        // Helper variable hv0 is the inverse of the difference of the stack's two top-most elements or 0.
        //
        // $ hv0·(hv0·(st1 - st0) - 1) = 0 $
        let hv0_is_inverse_of_diff_or_0 =
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

        vec![
            hv0_is_inverse_of_diff_or_0,
            hv0_is_inverse_of_diff_or_diff_is_0,
            st0_becomes_1_if_diff_is_not_invertible,
        ]
    }

    /// This instruction has no additional transition constraints.
    ///
    /// A Permutation Argument with the Uint32 Operations Table guarantees correct transition.
    pub fn instruction_lt(&self) -> Vec<MPolynomial<XWord>> {
        // no further constraints
        vec![]
    }

    /// This instruction has no additional transition constraints.
    ///
    /// A Permutation Argument with the Uint32 Operations Table guarantees correct transition.
    pub fn instruction_and(&self) -> Vec<MPolynomial<XWord>> {
        // no further constraints
        vec![]
    }

    /// This instruction has no additional transition constraints.
    ///
    /// A Permutation Argument with the Uint32 Operations Table guarantees correct transition.
    pub fn instruction_xor(&self) -> Vec<MPolynomial<XWord>> {
        // no further constraints
        vec![]
    }

    /// This instruction has no additional transition constraints.
    ///
    /// A Permutation Argument with the Uint32 Operations Table guarantees correct transition.
    pub fn instruction_reverse(&self) -> Vec<MPolynomial<XWord>> {
        // no further constraints
        vec![]
    }

    /// For correct division, it is required that the remainder r is smaller than the divisor d.
    ///
    /// The result of comparing r to d is stored in helper variable hv0.
    ///
    /// A Permutation Argument with the Uint32 Operations Table guarantees that hv0 = (r < d).
    pub fn instruction_div(&self) -> Vec<MPolynomial<XWord>> {
        // Denominator d is not zero.
        //
        // $st0·hv2 - 1 = 0$
        let denominator_is_not_zero = self.st0() * self.hv2() - self.one();

        // Result of division, i.e., quotient q and remainder r, are moved into
        // st1 and st0 respectively, and match with numerator n and denominator d.
        //
        // $st1 - st0·st1' - st0' = 0$
        let st1_becomes_quotient_and_st0_becomes_remainder =
            self.st1() - self.st0() * self.st1_next() - self.st0_next();

        // The stack element in st2 does not change.
        //
        // $st2' - st2 = 0$
        let st2_does_not_change = self.st2_next() - self.st2();

        // The stack element in st3 does not change.
        //
        // $st3' - st3 = 0$
        let st3_does_not_change = self.st3_next() - self.st3();

        // The stack element in st4 does not change.
        //
        // $st4' - st4 = 0$
        let st4_does_not_change = self.st4_next() - self.st4();

        // The stack element in st5 does not change.
        //
        // $st5' - st5 = 0$
        let st5_does_not_change = self.st5_next() - self.st5();

        // The stack element in st6 does not change.
        //
        // $st6' - st6 = 0$
        let st6_does_not_change = self.st6_next() - self.st6();

        // The stack element in st7 does not change.
        //
        // $st7' - st7 = 0$
        let st7_does_not_change = self.st7_next() - self.st7();

        // The stack element in st8 does not change.
        //
        // $st8' - st8 = 0$
        let st8_does_not_change = self.st8_next() - self.st8();

        // The stack element in st9 does not change.
        //
        // $st9' - st9 = 0$
        let st9_does_not_change = self.st9_next() - self.st9();

        // The stack element in st10 does not change.
        //
        // $st10' - st10 = 0$
        let st10_does_not_change = self.st10_next() - self.st10();

        // The stack element in st11 does not change.
        //
        // $st11' - st11 = 0$
        let st11_does_not_change = self.st11_next() - self.st11();

        // The stack element in st12 does not change.
        //
        // $st12' - st12 = 0$
        let st12_does_not_change = self.st12_next() - self.st12();

        // The stack element in st13 does not change.
        //
        // $st13' - st13 = 0$
        let st13_does_not_change = self.st13_next() - self.st13();

        // The stack element in st14 does not change.
        //
        // $st14' - st14 = 0$
        let st14_does_not_change = self.st14_next() - self.st14();

        // The stack element in st15 does not change.
        //
        // $st15' - st15 = 0$
        let st15_does_not_change = self.st15_next() - self.st15();

        // The top of the OpStack underflow, i.e., osv, does not change.
        //
        // $osv' - osv = 0$
        let osv_does_not_change = self.osv_next() - self.osv();

        // The OpStack pointer does not change.
        //
        // $osp' - osp = 0$
        let osp_does_not_change = self.osp_next() - self.osp();

        // Helper variable hv0 is 1, indicating that r < d.
        //
        // $hv0 - 1 = 0$
        let hv0_is_1 = self.hv0() - self.one();

        vec![
            denominator_is_not_zero,
            st1_becomes_quotient_and_st0_becomes_remainder,
            st2_does_not_change,
            st3_does_not_change,
            st4_does_not_change,
            st5_does_not_change,
            st6_does_not_change,
            st7_does_not_change,
            st8_does_not_change,
            st9_does_not_change,
            st10_does_not_change,
            st11_does_not_change,
            st12_does_not_change,
            st13_does_not_change,
            st14_does_not_change,
            st15_does_not_change,
            osv_does_not_change,
            osp_does_not_change,
            hv0_is_1,
        ]
    }

    pub fn instruction_xxadd(&self) -> Vec<MPolynomial<XWord>> {
        // The result of adding st0 to st3 is moved into st0.
        //
        // $st0' - (st0 + st3)$
        let st0_becomes_st0_plus_st3 = self.st0_next() - (self.st0() + self.st3());

        // The result of adding st1 to st4 is moved into st1.
        //
        // $st1' - (st1 + st4)$
        let st1_becomes_st1_plus_st4 = self.st1_next() - (self.st1() + self.st4());

        // The result of adding st2 to st5 is moved into st2.
        //
        // $st2' - (st2 + st5)$
        let st2_becomes_st2_plus_st5 = self.st2_next() - (self.st2() + self.st5());

        // The stack element in st3 does not change.
        //
        // $st3' - st3 = 0$
        let st3_does_not_change = self.st3_next() - self.st3();

        // The stack element in st4 does not change.
        //
        // $st4' - st4 = 0$
        let st4_does_not_change = self.st4_next() - self.st4();

        // The stack element in st5 does not change.
        //
        // $st5' - st5 = 0$
        let st5_does_not_change = self.st5_next() - self.st5();

        // The stack element in st6 does not change.
        //
        // $st6' - st6 = 0$
        let st6_does_not_change = self.st6_next() - self.st6();

        // The stack element in st7 does not change.
        //
        // $st7' - st7 = 0$
        let st7_does_not_change = self.st7_next() - self.st7();

        // The stack element in st8 does not change.
        //
        // $st8' - st8 = 0$
        let st8_does_not_change = self.st8_next() - self.st8();

        // The stack element in st9 does not change.
        //
        // $st9' - st9 = 0$
        let st9_does_not_change = self.st9_next() - self.st9();

        // The stack element in st10 does not change.
        //
        // $st10' - st10 = 0$
        let st10_does_not_change = self.st10_next() - self.st10();

        // The stack element in st11 does not change.
        //
        // $st11' - st11 = 0$
        let st11_does_not_change = self.st11_next() - self.st11();

        // The stack element in st12 does not change.
        //
        // $st12' - st12 = 0$
        let st12_does_not_change = self.st12_next() - self.st12();

        // The stack element in st13 does not change.
        //
        // $st13' - st13 = 0$
        let st13_does_not_change = self.st13_next() - self.st13();

        // The stack element in st14 does not change.
        //
        // $st14' - st14 = 0$
        let st14_does_not_change = self.st14_next() - self.st14();

        // The stack element in st15 does not change.
        //
        // $st15' - st15 = 0$
        let st15_does_not_change = self.st15_next() - self.st15();

        // The top of the OpStack underflow, i.e., osv, does not change.
        let osv_does_not_change = self.osv_next() - self.osv();

        // The OpStack pointer does not change.
        let osp_does_not_change = self.osp_next() - self.osp();

        vec![
            st0_becomes_st0_plus_st3,
            st1_becomes_st1_plus_st4,
            st2_becomes_st2_plus_st5,
            st3_does_not_change,
            st4_does_not_change,
            st5_does_not_change,
            st6_does_not_change,
            st7_does_not_change,
            st8_does_not_change,
            st9_does_not_change,
            st10_does_not_change,
            st11_does_not_change,
            st12_does_not_change,
            st13_does_not_change,
            st14_does_not_change,
            st15_does_not_change,
            osv_does_not_change,
            osp_does_not_change,
        ]
    }

    pub fn instruction_xxmul(&self) -> Vec<MPolynomial<XWord>> {
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

        // The stack element in st3 does not change.
        //
        // $st3' - st3 = 0$
        let st3_does_not_change = self.st3_next() - self.st3();

        // The stack element in st4 does not change.
        //
        // $st4' - st4 = 0$
        let st4_does_not_change = self.st4_next() - self.st4();

        // The stack element in st5 does not change.
        //
        // $st5' - st5 = 0$
        let st5_does_not_change = self.st5_next() - self.st5();

        // The stack element in st6 does not change.
        //
        // $st6' - st6 = 0$
        let st6_does_not_change = self.st6_next() - self.st6();

        // The stack element in st7 does not change.
        //
        // $st7' - st7 = 0$
        let st7_does_not_change = self.st7_next() - self.st7();

        vec![
            st0_becomes_coefficient_0,
            st1_becomes_coefficient_1,
            st2_becomes_coefficient_2,
            st3_does_not_change,
            st4_does_not_change,
            st5_does_not_change,
            st6_does_not_change,
            st7_does_not_change,
        ]
    }

    pub fn instruction_xinv(&self) -> Vec<MPolynomial<XWord>> {
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
        vec![
            first_coefficient_of_product_of_element_and_inverse_is_1,
            second_coefficient_of_product_of_element_and_inverse_is_0,
            third_coefficient_of_product_of_element_and_inverse_is_0,
            // st3 -- st15 do not change
            self.st3_next() - self.st3(),
            self.st4_next() - self.st4(),
            self.st5_next() - self.st5(),
            self.st6_next() - self.st6(),
            self.st7_next() - self.st7(),
            self.st8_next() - self.st8(),
            self.st9_next() - self.st9(),
            self.st10_next() - self.st10(),
            self.st11_next() - self.st11(),
            self.st12_next() - self.st12(),
            self.st13_next() - self.st13(),
            self.st14_next() - self.st14(),
            self.st15_next() - self.st15(),
            // osv and osp do not change
            self.osv_next() - self.osv(),
            self.osp_next() - self.osp(),
        ]
    }

    pub fn instruction_xbmul(&self) -> Vec<MPolynomial<XWord>> {
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

        vec![
            first_coeff_scalar_multiplication,
            secnd_coeff_scalar_multiplication,
            third_coeff_scalar_multiplication,
            self.st3_next() - self.st4_next(),
            self.st4_next() - self.st5_next(),
            self.st5_next() - self.st6_next(),
            self.st6_next() - self.st7_next(),
            self.st7_next() - self.st8_next(),
            self.st8_next() - self.st9_next(),
            self.st9_next() - self.st10_next(),
            self.st10_next() - self.st11_next(),
            self.st11_next() - self.st12_next(),
            self.st12_next() - self.st13_next(),
            self.st13_next() - self.st14_next(),
            self.st14_next() - self.st15_next(),
            self.st15_next() - self.osv(),
            self.osp_next() - (self.osp() - self.one()),
            (self.osp() - self.constant(16)) * self.hv3() - self.one(),
        ]
    }

    /// This instruction has no additional transition constraints.
    ///
    /// An Evaluation Argument with the list of input symbols guarantees correct transition.
    pub fn instruction_read_io(&self) -> Vec<MPolynomial<XWord>> {
        // no further constraints
        vec![]
    }

    /// This instruction has no additional transition constraints.
    ///
    /// An Evaluation Argument with the list of output symbols guarantees correct transition.
    pub fn instruction_write_io(&self) -> Vec<MPolynomial<XWord>> {
        // no further constraints
        vec![]
    }

    pub fn zero(&self) -> MPolynomial<XWord> {
        self.constant(0)
    }

    pub fn one(&self) -> MPolynomial<XWord> {
        self.constant(1)
    }

    pub fn two(&self) -> MPolynomial<XWord> {
        self.constant(2)
    }

    pub fn constant(&self, constant: u32) -> MPolynomial<XWord> {
        MPolynomial::from_constant(constant.into(), 2 * FULL_WIDTH)
    }

    pub fn constant_b(&self, constant: BFieldElement) -> MPolynomial<XWord> {
        MPolynomial::from_constant(constant.lift(), 2 * FULL_WIDTH)
    }

    pub fn clk(&self) -> MPolynomial<XWord> {
        self.variables[CLK as usize].clone()
    }

    pub fn ip(&self) -> MPolynomial<XWord> {
        self.variables[IP as usize].clone()
    }

    pub fn ci(&self) -> MPolynomial<XWord> {
        self.variables[CI as usize].clone()
    }

    pub fn nia(&self) -> MPolynomial<XWord> {
        self.variables[NIA as usize].clone()
    }

    pub fn ib0(&self) -> MPolynomial<XWord> {
        self.variables[IB0 as usize].clone()
    }

    pub fn ib1(&self) -> MPolynomial<XWord> {
        self.variables[IB1 as usize].clone()
    }

    pub fn ib2(&self) -> MPolynomial<XWord> {
        self.variables[IB2 as usize].clone()
    }

    pub fn ib3(&self) -> MPolynomial<XWord> {
        self.variables[IB3 as usize].clone()
    }

    pub fn ib4(&self) -> MPolynomial<XWord> {
        self.variables[IB4 as usize].clone()
    }

    pub fn ib5(&self) -> MPolynomial<XWord> {
        self.variables[IB5 as usize].clone()
    }

    pub fn jsp(&self) -> MPolynomial<XWord> {
        self.variables[JSP as usize].clone()
    }

    pub fn jsd(&self) -> MPolynomial<XWord> {
        self.variables[JSD as usize].clone()
    }

    pub fn jso(&self) -> MPolynomial<XWord> {
        self.variables[JSO as usize].clone()
    }

    pub fn st0(&self) -> MPolynomial<XWord> {
        self.variables[ST0 as usize].clone()
    }

    pub fn st1(&self) -> MPolynomial<XWord> {
        self.variables[ST1 as usize].clone()
    }

    pub fn st2(&self) -> MPolynomial<XWord> {
        self.variables[ST2 as usize].clone()
    }

    pub fn st3(&self) -> MPolynomial<XWord> {
        self.variables[ST3 as usize].clone()
    }

    pub fn st4(&self) -> MPolynomial<XWord> {
        self.variables[ST4 as usize].clone()
    }

    pub fn st5(&self) -> MPolynomial<XWord> {
        self.variables[ST5 as usize].clone()
    }

    pub fn st6(&self) -> MPolynomial<XWord> {
        self.variables[ST6 as usize].clone()
    }

    pub fn st7(&self) -> MPolynomial<XWord> {
        self.variables[ST7 as usize].clone()
    }

    pub fn st8(&self) -> MPolynomial<XWord> {
        self.variables[ST8 as usize].clone()
    }

    pub fn st9(&self) -> MPolynomial<XWord> {
        self.variables[ST9 as usize].clone()
    }

    pub fn st10(&self) -> MPolynomial<XWord> {
        self.variables[ST10 as usize].clone()
    }

    pub fn st11(&self) -> MPolynomial<XWord> {
        self.variables[ST11 as usize].clone()
    }

    pub fn st12(&self) -> MPolynomial<XWord> {
        self.variables[ST12 as usize].clone()
    }

    pub fn st13(&self) -> MPolynomial<XWord> {
        self.variables[ST13 as usize].clone()
    }

    pub fn st14(&self) -> MPolynomial<XWord> {
        self.variables[ST14 as usize].clone()
    }

    pub fn st15(&self) -> MPolynomial<XWord> {
        self.variables[ST15 as usize].clone()
    }

    pub fn osp(&self) -> MPolynomial<XWord> {
        self.variables[OSP as usize].clone()
    }

    pub fn osv(&self) -> MPolynomial<XWord> {
        self.variables[OSV as usize].clone()
    }

    pub fn hv0(&self) -> MPolynomial<XWord> {
        self.variables[HV0 as usize].clone()
    }

    pub fn hv1(&self) -> MPolynomial<XWord> {
        self.variables[HV1 as usize].clone()
    }

    pub fn hv2(&self) -> MPolynomial<XWord> {
        self.variables[HV2 as usize].clone()
    }

    pub fn hv3(&self) -> MPolynomial<XWord> {
        self.variables[HV3 as usize].clone()
    }

    pub fn ramv(&self) -> MPolynomial<XWord> {
        self.variables[RAMV as usize].clone()
    }

    // Property: All polynomial variables that contain '_next' have the same
    // variable position / value as the one without '_next', +/- FULL_WIDTH.
    pub fn clk_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + CLK as usize].clone()
    }

    pub fn ip_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + IP as usize].clone()
    }

    pub fn ci_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + CI as usize].clone()
    }

    pub fn jsp_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + JSP as usize].clone()
    }

    pub fn jsd_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + JSD as usize].clone()
    }

    pub fn jso_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + JSO as usize].clone()
    }

    pub fn st0_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + ST0 as usize].clone()
    }

    pub fn st1_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + ST1 as usize].clone()
    }

    pub fn st2_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + ST2 as usize].clone()
    }

    pub fn st3_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + ST3 as usize].clone()
    }

    pub fn st4_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + ST4 as usize].clone()
    }

    pub fn st5_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + ST5 as usize].clone()
    }

    pub fn st6_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + ST6 as usize].clone()
    }

    pub fn st7_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + ST7 as usize].clone()
    }

    pub fn st8_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + ST8 as usize].clone()
    }

    pub fn st9_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + ST9 as usize].clone()
    }

    pub fn st10_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + ST10 as usize].clone()
    }

    pub fn st11_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + ST11 as usize].clone()
    }

    pub fn st12_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + ST12 as usize].clone()
    }

    pub fn st13_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + ST13 as usize].clone()
    }

    pub fn st14_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + ST14 as usize].clone()
    }

    pub fn st15_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + ST15 as usize].clone()
    }

    pub fn osp_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + OSP as usize].clone()
    }

    pub fn osv_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + OSV as usize].clone()
    }

    pub fn ramv_next(&self) -> MPolynomial<XWord> {
        self.variables[FULL_WIDTH + RAMV as usize].clone()
    }

    pub fn decompose_arg(&self) -> Vec<MPolynomial<XWord>> {
        vec![
            self.nia()
                - (self.constant(8) * self.hv3()
                    + self.constant(4) * self.hv2()
                    + self.constant(2) * self.hv1()
                    + self.hv0()),
            self.hv1() * (self.hv1() - self.one()),
            self.hv0() * (self.hv0() - self.one()),
            self.hv2() * (self.hv2() - self.one()),
            self.hv3() * (self.hv3() - self.one()),
        ]
    }

    pub fn step_1(&self) -> Vec<MPolynomial<XWord>> {
        let one = self.one();
        let ip = self.ip();
        let ip_next = self.ip_next();

        vec![ip_next - ip - one]
    }

    pub fn step_2(&self) -> Vec<MPolynomial<XWord>> {
        let one = self.one();
        let ip = self.ip();
        let ip_next = self.ip_next();

        vec![ip_next - ip - (one.clone() + one)]
    }

    /// This group has no constraints. It is used for the Permutation Argument with the uint32 table.
    pub fn u32_op(&self) -> Vec<MPolynomial<XWord>> {
        // no further constraints
        vec![]
    }

    pub fn grow_stack(&self) -> Vec<MPolynomial<XWord>> {
        vec![
            // The stack element in st0 is moved into st1.
            self.st1_next() - self.st0(),
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
            self.osp_next() - (self.osp_next() + self.one()),
        ]
    }

    pub fn keep_stack(&self) -> Vec<MPolynomial<XWord>> {
        vec![
            // The stack element st0 does not change
            self.st0_next() - self.st0(),
            // The stack element st1 does not change
            self.st1_next() - self.st1(),
            // And so on...
            self.st2_next() - self.st2(),
            self.st3_next() - self.st3(),
            self.st4_next() - self.st4(),
            self.st5_next() - self.st5(),
            self.st6_next() - self.st6(),
            self.st7_next() - self.st7(),
            self.st8_next() - self.st8(),
            self.st9_next() - self.st9(),
            self.st10_next() - self.st10(),
            self.st11_next() - self.st11(),
            self.st12_next() - self.st12(),
            self.st13_next() - self.st13(),
            self.st14_next() - self.st14(),
            self.st15_next() - self.st15(),
            // The value of the OpStack underflow, osv, does not change.
            self.osv_next() - self.osv(),
            // The operational stack pointer, osp, does not change.
            self.osp_next() - self.osp(),
            // The RAM value, ramv, does not change.
            self.ramv_next() - self.ramv(),
        ]
    }

    pub fn shrink_stack(&self) -> Vec<MPolynomial<XWord>> {
        vec![
            // The stack element in st1 is moved into st0.
            self.st0_next() - self.st1(),
            // The stack element in st2 is moved into st1.
            self.st1_next() - self.st2(),
            // And so on...
            self.st2_next() - self.st3(),
            self.st3_next() - self.st4(),
            self.st4_next() - self.st5(),
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
            // The helper variable register hv4 holds the inverse of (osp - 16).
            (self.osp() - self.constant(16)) * self.hv3() - self.one(),
        ]
    }

    pub fn unop(&self) -> Vec<MPolynomial<XWord>> {
        vec![
            // The stack element in st1 does not change.
            self.st1_next() - self.st1(),
            // The stack element in st2 does not change.
            self.st2_next() - self.st2(),
            // And so on...
            self.st3_next() - self.st3(),
            self.st4_next() - self.st4(),
            self.st5_next() - self.st5(),
            self.st6_next() - self.st6(),
            self.st7_next() - self.st7(),
            self.st8_next() - self.st8(),
            self.st9_next() - self.st9(),
            self.st10_next() - self.st10(),
            self.st11_next() - self.st11(),
            self.st12_next() - self.st12(),
            self.st13_next() - self.st13(),
            self.st14_next() - self.st14(),
            self.st15_next() - self.st15(),
            // The top of the OpStack underflow, i.e., osv, does not change.
            self.osv_next() - self.osv(),
            // The OpStack pointer, osp, does not change.
            self.osp_next() - self.osp(),
            // The RAM value, ramv, does not change.
            self.ramv_next() - self.ramv(),
        ]
    }

    pub fn binop(&self) -> Vec<MPolynomial<XWord>> {
        vec![
            // The stack element in st2 is moved into st1.
            self.st1_next() - self.st2(),
            // The stack element in st3 is moved into st2.
            self.st2_next() - self.st3(),
            // And so on...
            self.st3_next() - self.st4(),
            self.st4_next() - self.st5(),
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
            // The OpStack pointer is decremented by 1.
            self.osp_next() - (self.osp() - self.one()),
            // The helper variable register hv4 holds the inverse of (osp - 16).
            (self.osp() - self.constant(16)) * self.hv3() - self.one(),
        ]
    }
}

#[derive(Debug, Clone)]
pub struct InstructionDeselectors {
    deselectors: HashMap<Instruction, MPolynomial<XWord>>,
}

impl Default for InstructionDeselectors {
    fn default() -> Self {
        let factory = TransitionConstraints::default();
        let deselectors = Self::create(&factory);

        Self { deselectors }
    }
}

impl InstructionDeselectors {
    /// A polynomial that has solutions when `ci` is not `instruction`.
    ///
    /// This is naively achieved by constructing a polynomial that has
    /// a solution when `ci` is any other instruction. This deselector
    /// can be replaced with an efficient one based on `ib` registers.
    pub fn get(&self, instruction: Instruction) -> MPolynomial<XWord> {
        self.deselectors
            .get(&instruction)
            .unwrap_or_else(|| panic!("The instruction {} does not exist!", instruction))
            .clone()
        // self.deselectors[&instruction].clone()
    }

    /// A polynomial that has no solutions when ci is 'instruction'
    pub fn instruction_deselector(
        factory: &TransitionConstraints,
        instruction: Instruction,
    ) -> MPolynomial<XWord> {
        let one = XWord::ring_one();
        let num_vars = factory.variables.len();

        let ib0 = instruction.ib(Ord6::IB0).lift();
        let ib1 = instruction.ib(Ord6::IB1).lift();
        let ib2 = instruction.ib(Ord6::IB2).lift();
        let ib3 = instruction.ib(Ord6::IB3).lift();
        let ib4 = instruction.ib(Ord6::IB4).lift();
        let ib5 = instruction.ib(Ord6::IB5).lift();
        let deselect_ib0 = MPolynomial::from_constant(one - ib0, num_vars);
        let deselect_ib1 = MPolynomial::from_constant(one - ib1, num_vars);
        let deselect_ib2 = MPolynomial::from_constant(one - ib2, num_vars);
        let deselect_ib3 = MPolynomial::from_constant(one - ib3, num_vars);
        let deselect_ib4 = MPolynomial::from_constant(one - ib4, num_vars);
        let deselect_ib5 = MPolynomial::from_constant(one - ib5, num_vars);
        (factory.ib0() - deselect_ib0)
            * (factory.ib1() - deselect_ib1)
            * (factory.ib2() - deselect_ib2)
            * (factory.ib3() - deselect_ib3)
            * (factory.ib4() - deselect_ib4)
            * (factory.ib5() - deselect_ib5)
    }

    pub fn create(factory: &TransitionConstraints) -> HashMap<Instruction, MPolynomial<XWord>> {
        all_instructions_without_args()
            .into_iter()
            .map(|instrctn| (instrctn, Self::instruction_deselector(factory, instrctn)))
            .collect()
    }
}

#[cfg(test)]
mod constraint_polynomial_tests {
    use super::*;
    use crate::shared_math::stark::triton::ord_n::Ord16;
    use crate::shared_math::stark::triton::table::base_matrix::ProcessorMatrixRow;
    use crate::shared_math::stark::triton::table::processor_table;
    use crate::shared_math::stark::triton::vm::Program;
    use crate::shared_math::traits::IdentityValues;

    #[test]
    /// helps identifying whether the printing causes an infinite loop
    fn print_simple_processor_table_row_test() {
        let program = Program::from_code("push 2 push -1 add assert halt").unwrap();
        let (base_matrices, _) = program.simulate_with_input(&[], &[]);
        for row in base_matrices.processor_matrix {
            println!("{}", ProcessorMatrixRow { row });
        }
    }

    fn get_test_row_from_source_code(source_code: &str, row_num: usize) -> Vec<XWord> {
        let fake_extension_columns = [BFieldElement::ring_zero();
            processor_table::FULL_WIDTH - processor_table::BASE_WIDTH]
            .to_vec();

        let program = Program::from_code(source_code).unwrap();
        let (base_matrices, err) = program.simulate_with_input(&[], &[]);
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

    fn get_constraints_for_instruction(instruction: Instruction) -> Vec<MPolynomial<XWord>> {
        let tc = TransitionConstraints::default();
        match instruction {
            Pop => tc.instruction_pop(),
            Push(_) => tc.instruction_push(),
            Divine => tc.instruction_divine(),
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
            Lt => tc.instruction_lt(),
            And => tc.instruction_and(),
            Xor => tc.instruction_xor(),
            Reverse => tc.instruction_reverse(),
            Div => tc.instruction_div(),
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
        test_rows: &[Vec<XWord>],
        debug_cols_curr_row: &[ProcessorTableColumn],
        debug_cols_next_row: &[ProcessorTableColumn],
    ) {
        for (case_idx, test_row) in test_rows.iter().enumerate() {
            // Print debug information
            println!(
                "Testing all constraint polynomials of {} for test row with index {}…",
                instruction, case_idx
            );
            for c in debug_cols_curr_row {
                print!("{} = {}, ", c, test_row[*c as usize]);
            }
            for c in debug_cols_next_row {
                print!("{}' = {}, ", c, test_row[*c as usize + FULL_WIDTH]);
            }
            println!();

            for (poly_idx, poly) in get_constraints_for_instruction(instruction)
                .iter()
                .enumerate()
            {
                assert_eq!(
                    instruction.opcode_b().lift(),
                    test_row[CI as usize],
                    "The test is trying to check the wrong constraint polynomials."
                );
                assert_eq!(
                    XFieldElement::ring_zero(),
                    poly.evaluate(&test_row),
                    "For case {}, polynomial with index {} must evaluate to zero.",
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
            Push(1.into()),
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
        let deselectors = InstructionDeselectors::default();

        let mut row = vec![0.into(); 2 * FULL_WIDTH];

        for instruction in all_instructions_without_args() {
            use ProcessorTableColumn::*;
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
                row[usize::from(IB0)] = other_instruction.ib(Ord6::IB0).lift();
                row[usize::from(IB1)] = other_instruction.ib(Ord6::IB1).lift();
                row[usize::from(IB2)] = other_instruction.ib(Ord6::IB2).lift();
                row[usize::from(IB3)] = other_instruction.ib(Ord6::IB3).lift();
                row[usize::from(IB4)] = other_instruction.ib(Ord6::IB4).lift();
                row[usize::from(IB5)] = other_instruction.ib(Ord6::IB5).lift();
                let result = deselector.evaluate(&row);

                assert!(
                    result.is_zero(),
                    "Deselector for {} should return 0 for all other instructions, including {}",
                    instruction,
                    other_instruction
                )
            }

            // Positive tests
            row[usize::from(IB0)] = instruction.ib(Ord6::IB0).lift();
            row[usize::from(IB1)] = instruction.ib(Ord6::IB1).lift();
            row[usize::from(IB2)] = instruction.ib(Ord6::IB2).lift();
            row[usize::from(IB3)] = instruction.ib(Ord6::IB3).lift();
            row[usize::from(IB4)] = instruction.ib(Ord6::IB4).lift();
            row[usize::from(IB5)] = instruction.ib(Ord6::IB5).lift();
            let result = deselector.evaluate(&row);
            assert!(
                !result.is_zero(),
                "Deselector for {} should be non-zero when CI is {}",
                instruction,
                instruction.opcode()
            )
        }
    }
}
