use std::collections::HashMap;
use std::ops::Add;

use itertools::Itertools;
use num_traits::{One, Zero};
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::{Degree, MPolynomial};
use twenty_first::shared_math::rescue_prime_regular::DIGEST_LENGTH;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::cross_table_arguments::{CrossTableArg, EvalArg, PermArg};
use crate::fri_domain::FriDomain;
use crate::instruction::{all_instructions_without_args, AnInstruction::*, Instruction};
use crate::ord_n::Ord7;
use crate::table::base_table::{Extendable, InheritsFromTable, Table, TableLike};
use crate::table::challenges::AllChallenges;
use crate::table::extension_table::{Evaluable, ExtensionTable};
use crate::table::table_column::ProcessorBaseTableColumn::{self, *};
use crate::table::table_column::ProcessorExtTableColumn::{self, *};

use super::extension_table::{Quotientable, QuotientableExtensionTable};

pub const PROCESSOR_TABLE_NUM_PERMUTATION_ARGUMENTS: usize = 4;
pub const PROCESSOR_TABLE_NUM_EVALUATION_ARGUMENTS: usize = 4;

/// This is 43 because it combines all other tables (except program).
pub const PROCESSOR_TABLE_NUM_EXTENSION_CHALLENGES: usize = 43;

pub const BASE_WIDTH: usize = ProcessorBaseTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + ProcessorExtTableColumn::COUNT;

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
    pub fn new_prover(matrix: Vec<Vec<BFieldElement>>) -> Self {
        let inherited_table =
            Table::new(BASE_WIDTH, FULL_WIDTH, matrix, "ProcessorTable".to_string());
        Self { inherited_table }
    }

    pub fn codeword_table(
        &self,
        fri_domain: &FriDomain<BFieldElement>,
        omicron: BFieldElement,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Self {
        let base_columns = 0..self.base_width();
        let codewords = self.low_degree_extension(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
            base_columns,
        );
        let inherited_table = self.inherited_table.with_data(codewords);
        Self { inherited_table }
    }

    pub fn extend(
        &self,
        challenges: &ProcessorTableChallenges,
        interpolant_degree: Degree,
    ) -> ExtProcessorTable {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());

        let mut input_table_running_evaluation = EvalArg::default_initial();
        let mut output_table_running_evaluation = EvalArg::default_initial();
        let mut instruction_table_running_product = PermArg::default_initial();
        let mut opstack_table_running_product = PermArg::default_initial();
        let mut ram_table_running_product = PermArg::default_initial();
        let mut jump_stack_running_product = PermArg::default_initial();
        let mut to_hash_table_running_evaluation = EvalArg::default_initial();
        let mut from_hash_table_running_evaluation = EvalArg::default_initial();

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
                        * challenges.input_table_eval_row_weight
                        + input_symbol;
                }
            }
            extension_row[usize::from(InputTableEvalArg)] = input_table_running_evaluation;

            // Output table
            if row[usize::from(CI)] == Instruction::WriteIo.opcode_b() {
                let output_symbol = extension_row[usize::from(ST0)];
                output_table_running_evaluation = output_table_running_evaluation
                    * challenges.output_table_eval_row_weight
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
                instruction_table_running_product *= challenges.instruction_perm_row_weight
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
            opstack_table_running_product *= challenges.op_stack_perm_row_weight
                - compressed_row_for_op_stack_table_permutation_argument;
            extension_row[usize::from(OpStackTablePermArg)] = opstack_table_running_product;

            // RAM Table
            let ramv = extension_row[usize::from(RAMV)];
            let ramp = extension_row[usize::from(ST1)];

            let compressed_row_for_ram_table_permutation_argument = clk
                * challenges.ram_table_clk_weight
                + ramv * challenges.ram_table_ramv_weight
                + ramp * challenges.ram_table_ramp_weight;
            ram_table_running_product *=
                challenges.ram_perm_row_weight - compressed_row_for_ram_table_permutation_argument;
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
                challenges.jump_stack_perm_row_weight - compressed_row_for_jump_stack_table;
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
                let compressed_row_for_hash_input = st_0_through_9
                    .iter()
                    .zip_eq(challenges.hash_table_stack_input_weights.iter())
                    .map(|(&st, &weight)| weight * st)
                    .fold(XFieldElement::zero(), XFieldElement::add);
                to_hash_table_running_evaluation = to_hash_table_running_evaluation
                    * challenges.to_hash_table_eval_row_weight
                    + compressed_row_for_hash_input;
            }
            extension_row[usize::from(ToHashTableEvalArg)] = to_hash_table_running_evaluation;

            // Hash Table – Hash's output from Hash Coprocessor to Processor
            if let Some(prow) = previous_row.clone() {
                let st_5_through_9 = [
                    extension_row[usize::from(ST5)],
                    extension_row[usize::from(ST6)],
                    extension_row[usize::from(ST7)],
                    extension_row[usize::from(ST8)],
                    extension_row[usize::from(ST9)],
                ];
                let compressed_row_for_hash_digest = st_5_through_9
                    .iter()
                    .zip_eq(challenges.hash_table_digest_output_weights.iter())
                    .map(|(&st, &weight)| weight * st)
                    .fold(XFieldElement::zero(), XFieldElement::add);
                if prow[usize::from(CI)] == Instruction::Hash.opcode_b() {
                    from_hash_table_running_evaluation = from_hash_table_running_evaluation
                        * challenges.from_hash_table_eval_row_weight
                        + compressed_row_for_hash_digest;
                }
            }
            extension_row[usize::from(FromHashTableEvalArg)] = from_hash_table_running_evaluation;

            previous_row = Some(row.clone());
            extension_matrix.push(extension_row.to_vec());
        }

        assert_eq!(self.data().len(), extension_matrix.len());
        let padded_height = extension_matrix.len();
        let inherited_table = self.extension(
            extension_matrix,
            interpolant_degree,
            padded_height,
            ExtProcessorTable::ext_initial_constraints(challenges),
            ExtProcessorTable::ext_consistency_constraints(challenges),
            ExtProcessorTable::ext_transition_constraints(challenges),
            ExtProcessorTable::ext_terminal_constraints(challenges),
        );
        ExtProcessorTable { inherited_table }
    }

    pub fn for_verifier(
        interpolant_degree: Degree,
        padded_height: usize,
        all_challenges: &AllChallenges,
    ) -> ExtProcessorTable {
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            vec![],
            "ExtProcessorTable".to_string(),
        );
        let base_table = Self { inherited_table };
        let empty_matrix: Vec<Vec<XFieldElement>> = vec![];
        let extension_table = base_table.extension(
            empty_matrix,
            interpolant_degree,
            padded_height,
            ExtProcessorTable::ext_initial_constraints(&all_challenges.processor_table_challenges),
            ExtProcessorTable::ext_consistency_constraints(
                &all_challenges.processor_table_challenges,
            ),
            ExtProcessorTable::ext_transition_constraints(
                &all_challenges.processor_table_challenges,
            ),
            ExtProcessorTable::ext_terminal_constraints(&all_challenges.processor_table_challenges),
        );

        ExtProcessorTable {
            inherited_table: extension_table,
        }
    }
}

impl ExtProcessorTable {
    pub fn ext_codeword_table(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        omicron: XFieldElement,
        padded_height: usize,
        num_trace_randomizers: usize,
        base_codewords: &[Vec<BFieldElement>],
    ) -> Self {
        let ext_columns = self.base_width()..self.full_width();
        let ext_codewords = self.low_degree_extension(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
            ext_columns,
        );

        let lifted_base_codewords = base_codewords
            .iter()
            .map(|base_codeword| base_codeword.iter().map(|bfe| bfe.lift()).collect_vec())
            .collect_vec();
        let all_codewords = vec![lifted_base_codewords, ext_codewords].concat();
        assert_eq!(self.full_width(), all_codewords.len());

        let inherited_table = self.inherited_table.with_data(all_codewords);
        Self::new(inherited_table)
    }

    pub fn new(base: Table<XFieldElement>) -> ExtProcessorTable {
        Self {
            inherited_table: base,
        }
    }

    /// Transition constraints are combined with deselectors in such a way
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
    fn combine_transition_constraints_with_deselectors(
        instr_tc_polys_tuples: Vec<(Instruction, Vec<MPolynomial<XFieldElement>>)>,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let (all_instructions, all_tc_polys_for_all_instructions): (Vec<_>, Vec<Vec<_>>) =
            instr_tc_polys_tuples.into_iter().unzip();

        let instruction_deselectors = InstructionDeselectors::default();

        let all_instruction_deselectors = all_instructions
            .into_iter()
            .map(|instr| instruction_deselectors.get(instr))
            .collect_vec();

        let max_number_of_constraints = all_tc_polys_for_all_instructions
            .iter()
            .map(|tc_polys_for_instr| tc_polys_for_instr.len())
            .max()
            .unwrap();
        let zero_poly = RowPairConstraints::default().zero();

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

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub instruction_table_ip_weight: XFieldElement,
    pub instruction_table_ci_processor_weight: XFieldElement,
    pub instruction_table_nia_weight: XFieldElement,

    pub op_stack_table_clk_weight: XFieldElement,
    pub op_stack_table_ib1_weight: XFieldElement,
    pub op_stack_table_osp_weight: XFieldElement,
    pub op_stack_table_osv_weight: XFieldElement,

    pub ram_table_clk_weight: XFieldElement,
    pub ram_table_ramv_weight: XFieldElement,
    pub ram_table_ramp_weight: XFieldElement,

    pub jump_stack_table_clk_weight: XFieldElement,
    pub jump_stack_table_ci_weight: XFieldElement,
    pub jump_stack_table_jsp_weight: XFieldElement,
    pub jump_stack_table_jso_weight: XFieldElement,
    pub jump_stack_table_jsd_weight: XFieldElement,

    pub hash_table_stack_input_weights: [XFieldElement; 2 * DIGEST_LENGTH],
    pub hash_table_digest_output_weights: [XFieldElement; DIGEST_LENGTH],
}

#[derive(Debug, Clone)]
pub struct IOChallenges {
    /// weight for updating the running evaluation with the next i/o symbol in the i/o list
    pub processor_eval_row_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct ExtProcessorTable {
    inherited_table: Table<XFieldElement>,
}

impl Evaluable for ExtProcessorTable {}
impl Quotientable for ExtProcessorTable {}
impl QuotientableExtensionTable for ExtProcessorTable {}

impl InheritsFromTable<XFieldElement> for ExtProcessorTable {
    fn inherited_table(&self) -> &Table<XFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<XFieldElement> {
        &mut self.inherited_table
    }
}

impl Default for ExtProcessorTable {
    fn default() -> Self {
        Self {
            inherited_table: Table::new(
                BASE_WIDTH,
                FULL_WIDTH,
                vec![],
                "EmptyExtProcessorTable".to_string(),
            ),
        }
    }
}

impl TableLike<BFieldElement> for ProcessorTable {}

impl Extendable for ProcessorTable {
    fn get_padding_rows(&self) -> (Option<usize>, Vec<Vec<BFieldElement>>) {
        let zero = BFieldElement::zero();
        let one = BFieldElement::one();
        if let Some(row) = self.data().last() {
            let mut padding_row = row.clone();
            padding_row[usize::from(ProcessorBaseTableColumn::CLK)] += one;
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
    fn ext_initial_constraints(
        _challenges: &ProcessorTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let factory = SingleRowConstraints::default();

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

    fn ext_consistency_constraints(
        _challenges: &ProcessorTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let factory = SingleRowConstraints::default();
        let one = factory.one();

        // The composition of instruction buckets ib0-ib5 corresponds the current instruction ci.
        //
        // $ci - (2^6·ib6 + 2^5·ib5 + 2^4·ib4 + 2^3·ib3 + 2^2·ib2 + 2^1·ib1 + 2^0·ib0) = 0$
        let ci_corresponds_to_ib0_thru_ib5 = {
            let ib_composition = one.clone() * factory.ib0()
                + factory.constant(2) * factory.ib1()
                + factory.constant(4) * factory.ib2()
                + factory.constant(8) * factory.ib3()
                + factory.constant(16) * factory.ib4()
                + factory.constant(32) * factory.ib5()
                + factory.constant(64) * factory.ib6();

            factory.ci() - ib_composition
        };

        let ib0 = factory.ib0();
        let ib0_is_bit = ib0.clone() * (ib0 - one.clone());
        let ib1 = factory.ib1();
        let ib1_is_bit = ib1.clone() * (ib1 - one.clone());
        let ib2 = factory.ib2();
        let ib2_is_bit = ib2.clone() * (ib2 - one.clone());
        let ib3 = factory.ib3();
        let ib3_is_bit = ib3.clone() * (ib3 - one.clone());
        let ib4 = factory.ib4();
        let ib4_is_bit = ib4.clone() * (ib4 - one.clone());
        let ib5 = factory.ib5();
        let ib5_is_bit = ib5.clone() * (ib5 - one.clone());
        let ib6 = factory.ib6();
        let ib6_is_bit = ib6.clone() * (ib6 - one);

        vec![
            ib0_is_bit,
            ib1_is_bit,
            ib2_is_bit,
            ib3_is_bit,
            ib4_is_bit,
            ib5_is_bit,
            ib6_is_bit,
            ci_corresponds_to_ib0_thru_ib5,
        ]
    }

    fn ext_transition_constraints(
        _challenges: &ProcessorTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let factory = RowPairConstraints::default();

        let all_instruction_transition_constraints = vec![
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
        assert_eq!(
            Instruction::COUNT,
            all_instruction_transition_constraints.len()
        );

        let mut transition_constraints = Self::combine_transition_constraints_with_deselectors(
            all_instruction_transition_constraints,
        );
        transition_constraints.insert(0, factory.clk_always_increases_by_one());
        transition_constraints
    }

    fn ext_terminal_constraints(
        _challenges: &ProcessorTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let factory = SingleRowConstraints::default();

        // In the last row, current instruction register ci is 0, corresponding to instruction halt.
        //
        // $ci - halt = 0  =>  ci - 0 = 0  =>  ci$
        let last_ci_is_halt = factory.ci();

        vec![last_ci_is_halt]
    }
}

#[derive(Debug, Clone)]
pub struct SingleRowConstraints {
    variables: [MPolynomial<XFieldElement>; FULL_WIDTH],
}

impl Default for SingleRowConstraints {
    fn default() -> Self {
        let variables = MPolynomial::variables(FULL_WIDTH)
            .try_into()
            .expect("Create variables for initial/consistency/terminal constraints");

        Self { variables }
    }
}

impl SingleRowConstraints {
    // FIXME: This does not need a self reference.
    pub fn constant(&self, constant: u32) -> MPolynomial<XFieldElement> {
        MPolynomial::from_constant(constant.into(), FULL_WIDTH)
    }

    pub fn one(&self) -> MPolynomial<XFieldElement> {
        self.constant(1)
    }

    pub fn two(&self) -> MPolynomial<XFieldElement> {
        self.constant(2)
    }

    pub fn clk(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(CLK)].clone()
    }

    pub fn ip(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(IP)].clone()
    }

    pub fn ci(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(CI)].clone()
    }

    pub fn nia(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(NIA)].clone()
    }

    pub fn ib0(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(IB0)].clone()
    }

    pub fn ib1(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(IB1)].clone()
    }

    pub fn ib2(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(IB2)].clone()
    }

    pub fn ib3(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(IB3)].clone()
    }

    pub fn ib4(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(IB4)].clone()
    }

    pub fn ib5(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(IB5)].clone()
    }

    pub fn ib6(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(IB6)].clone()
    }

    pub fn jsp(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(JSP)].clone()
    }

    pub fn jsd(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(JSD)].clone()
    }

    pub fn jso(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(JSO)].clone()
    }

    pub fn st0(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST0)].clone()
    }

    pub fn st1(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST1)].clone()
    }

    pub fn st2(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST2)].clone()
    }

    pub fn st3(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST3)].clone()
    }

    pub fn st4(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST4)].clone()
    }

    pub fn st5(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST5)].clone()
    }

    pub fn st6(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST6)].clone()
    }

    pub fn st7(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST7)].clone()
    }

    pub fn st8(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST8)].clone()
    }

    pub fn st9(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST9)].clone()
    }

    pub fn st10(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST10)].clone()
    }

    pub fn st11(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST11)].clone()
    }

    pub fn st12(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST12)].clone()
    }

    pub fn st13(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST13)].clone()
    }

    pub fn st14(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST14)].clone()
    }

    pub fn st15(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST15)].clone()
    }

    pub fn osp(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(OSP)].clone()
    }

    pub fn osv(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(OSV)].clone()
    }

    pub fn hv0(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(HV0)].clone()
    }

    pub fn hv1(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(HV1)].clone()
    }

    pub fn hv2(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(HV2)].clone()
    }

    pub fn hv3(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(HV3)].clone()
    }

    pub fn ramv(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(RAMV)].clone()
    }
}

#[derive(Debug, Clone)]
pub struct RowPairConstraints {
    variables: [MPolynomial<XFieldElement>; 2 * FULL_WIDTH],
}

impl Default for RowPairConstraints {
    fn default() -> Self {
        let variables = MPolynomial::variables(2 * FULL_WIDTH)
            .try_into()
            .expect("Create variables for transition constraints");

        Self { variables }
    }
}

impl RowPairConstraints {
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
    pub fn clk_always_increases_by_one(&self) -> MPolynomial<XFieldElement> {
        let one = self.one();
        let clk = self.clk();
        let clk_next = self.clk_next();

        clk_next - clk - one
    }

    pub fn indicator_polynomial(&self, i: usize) -> MPolynomial<XFieldElement> {
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

    pub fn instruction_pop(&self) -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }

    /// push'es argument should be on the stack after execution
    /// $st0_next == nia  =>  st0_next - nia == 0$
    pub fn instruction_push(&self) -> Vec<MPolynomial<XFieldElement>> {
        let st0_next = self.st0_next();
        let nia = self.nia();

        vec![st0_next - nia]
    }

    pub fn instruction_divine(&self) -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }

    pub fn instruction_dup(&self) -> Vec<MPolynomial<XFieldElement>> {
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

    pub fn instruction_swap(&self) -> Vec<MPolynomial<XFieldElement>> {
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

    pub fn instruction_nop(&self) -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }

    pub fn instruction_skiz(&self) -> Vec<MPolynomial<XFieldElement>> {
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

    pub fn instruction_call(&self) -> Vec<MPolynomial<XFieldElement>> {
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

    pub fn instruction_return(&self) -> Vec<MPolynomial<XFieldElement>> {
        // The jump stack pointer jsp is decremented by 1.
        let jsp_incr_1 = self.jsp_next() - (self.jsp() - self.one());

        // The instruction pointer ip is set to the last call's origin jso.
        let ip_becomes_jso = self.ip_next() - self.jso();

        vec![jsp_incr_1, ip_becomes_jso]
    }

    pub fn instruction_recurse(&self) -> Vec<MPolynomial<XFieldElement>> {
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

    pub fn instruction_assert(&self) -> Vec<MPolynomial<XFieldElement>> {
        // The current top of the stack st0 is 1.
        let st_0_is_1 = self.st0() - self.one();

        vec![st_0_is_1]
    }

    pub fn instruction_halt(&self) -> Vec<MPolynomial<XFieldElement>> {
        // The instruction executed in the following step is instruction halt.
        let halt_is_followed_by_halt = self.ci_next() - self.ci();

        vec![halt_is_followed_by_halt]
    }

    pub fn instruction_read_mem(&self) -> Vec<MPolynomial<XFieldElement>> {
        // The top of the stack is overwritten with the RAM value.
        let st0_becomes_ramv = self.st0_next() - self.ramv();

        vec![st0_becomes_ramv]
    }

    pub fn instruction_write_mem(&self) -> Vec<MPolynomial<XFieldElement>> {
        // The RAM value is overwritten with the top of the stack.
        let ramv_becomes_st0 = self.ramv_next() - self.st0();

        vec![ramv_becomes_st0]
    }

    /// Two Evaluation Arguments with the Hash Table guarantee correct transition.
    pub fn instruction_hash(&self) -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }

    /// Recall that in a Merkle tree, the indices of left (respectively right)
    /// leafs have 0 (respectively 1) as their least significant bit. The first
    /// two polynomials achieve that helper variable hv0 holds the result of
    /// st10 mod 2. The second polynomial sets the new value of st10 to st10 div 2.
    pub fn instruction_divine_sibling(&self) -> Vec<MPolynomial<XFieldElement>> {
        // Helper variable hv0 is either 0 or 1.
        let hv0_is_0_or_1 = self.hv0() * (self.hv0() - self.one());

        // The 11th stack register is shifted by 1 bit to the right.
        let st10_is_shifted_1_bit_right = self.st10_next() * self.two() + self.hv0() - self.st10();

        // The the second pentuplet either stays where it is, or is moved to the top
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

        // The stack element in st11 does not change.
        let st11_does_not_change = self.st11_next() - self.st11();

        // The stack element in st12 does not change.
        let st12_does_not_change = self.st12_next() - self.st12();

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
            hv0_is_0_or_1,
            st10_is_shifted_1_bit_right,
            maybe_move_st5,
            maybe_move_st6,
            maybe_move_st7,
            maybe_move_st8,
            maybe_move_st9,
            st11_does_not_change,
            st12_does_not_change,
            st13_does_not_change,
            st14_does_not_change,
            st15_does_not_change,
            osv_does_not_change,
            osp_does_not_change,
            ramv_does_not_change_when_hv0_is_0,
        ]
    }

    pub fn instruction_assert_vector(&self) -> Vec<MPolynomial<XFieldElement>> {
        vec![
            // Register st0 is equal to st5.
            // $st6 - st0 = 0$
            self.st5() - self.st0(),
            // Register st1 is equal to st6.
            // $st7 - st1 = 0$
            self.st6() - self.st1(),
            // Register st2 is equal to st7.
            // $st8 - st2 = 0$
            self.st7() - self.st2(),
            // Register st3 is equal to st8.
            // $st9 - st3 = 0$
            self.st8() - self.st3(),
            // Register st4 is equal to st9.
            // $st10 - st4 = 0$
            self.st9() - self.st4(),
        ]
    }

    /// The sum of the top two stack elements is moved into the top of the stack.
    ///
    /// $st0' - (st0 + st1) = 0$
    pub fn instruction_add(&self) -> Vec<MPolynomial<XFieldElement>> {
        vec![self.st0_next() - (self.st0() + self.st1())]
    }

    /// The product of the top two stack elements is moved into the top of the stack.
    ///
    /// $st0' - (st0 * st1) = 0$
    pub fn instruction_mul(&self) -> Vec<MPolynomial<XFieldElement>> {
        vec![self.st0_next() - (self.st0() * self.st1())]
    }

    /// The top of the stack's inverse is moved into the top of the stack.
    ///
    /// $st0'·st0 - 1 = 0$
    pub fn instruction_invert(&self) -> Vec<MPolynomial<XFieldElement>> {
        vec![self.st0_next() * self.st0() - self.one()]
    }

    pub fn instruction_split(&self) -> Vec<MPolynomial<XFieldElement>> {
        let two_pow_32 = self.constant_b(BFieldElement::new(2u64.pow(32)));

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

        let st2_becomes_st1 = self.st2_next() - self.st1();
        let st3_becomes_st2 = self.st3_next() - self.st2();
        let st4_becomes_st3 = self.st4_next() - self.st3();
        let st5_becomes_st4 = self.st5_next() - self.st4();
        let st6_becomes_st5 = self.st6_next() - self.st5();
        let st7_becomes_st6 = self.st7_next() - self.st6();
        let st8_becomes_st7 = self.st8_next() - self.st7();
        let st9_becomes_st8 = self.st9_next() - self.st8();
        let st10_becomes_st9 = self.st10_next() - self.st9();
        let st11_becomes_st10 = self.st11_next() - self.st10();
        let st12_becomes_st11 = self.st12_next() - self.st11();
        let st13_becomes_st12 = self.st13_next() - self.st12();
        let st14_becomes_st13 = self.st14_next() - self.st13();
        let st15_becomes_st14 = self.st15_next() - self.st14();
        let osv_becomes_st15 = self.osv_next() - self.st15();
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
            st8_becomes_st7,
            st9_becomes_st8,
            st10_becomes_st9,
            st11_becomes_st10,
            st12_becomes_st11,
            st13_becomes_st12,
            st14_becomes_st13,
            st15_becomes_st14,
            osv_becomes_st15,
            osp_is_incremented,
        ]
    }

    pub fn instruction_eq(&self) -> Vec<MPolynomial<XFieldElement>> {
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

    /// 1. The lsb is a bit
    /// 2. The operand decomposes into right-shifted operand and the lsb
    pub fn instruction_lsb(&self) -> Vec<MPolynomial<XFieldElement>> {
        let operand = self.variables[usize::from(ST0)].clone();
        let shifted_operand = self.variables[FULL_WIDTH + usize::from(ST1)].clone();
        let lsb = self.variables[FULL_WIDTH + usize::from(ST0)].clone();

        let lsb_is_a_bit = lsb.clone() * (lsb.clone() - self.one());

        let correct_decomposition = self.two() * shifted_operand + lsb - operand;

        vec![lsb_is_a_bit, correct_decomposition]
    }

    pub fn instruction_xxadd(&self) -> Vec<MPolynomial<XFieldElement>> {
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

    pub fn instruction_xxmul(&self) -> Vec<MPolynomial<XFieldElement>> {
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

    pub fn instruction_xinv(&self) -> Vec<MPolynomial<XFieldElement>> {
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

    pub fn instruction_xbmul(&self) -> Vec<MPolynomial<XFieldElement>> {
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
    pub fn instruction_read_io(&self) -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }

    /// This instruction has no additional transition constraints.
    ///
    /// An Evaluation Argument with the list of output symbols guarantees correct transition.
    pub fn instruction_write_io(&self) -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }

    pub fn zero(&self) -> MPolynomial<XFieldElement> {
        self.constant(0)
    }

    pub fn one(&self) -> MPolynomial<XFieldElement> {
        self.constant(1)
    }

    pub fn two(&self) -> MPolynomial<XFieldElement> {
        self.constant(2)
    }

    pub fn constant(&self, constant: u32) -> MPolynomial<XFieldElement> {
        MPolynomial::from_constant(constant.into(), 2 * FULL_WIDTH)
    }

    pub fn constant_b(&self, constant: BFieldElement) -> MPolynomial<XFieldElement> {
        MPolynomial::from_constant(constant.lift(), 2 * FULL_WIDTH)
    }

    pub fn clk(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(CLK)].clone()
    }

    pub fn ip(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(IP)].clone()
    }

    pub fn ci(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(CI)].clone()
    }

    pub fn nia(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(NIA)].clone()
    }

    pub fn ib0(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(IB0)].clone()
    }

    pub fn ib1(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(IB1)].clone()
    }

    pub fn ib2(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(IB2)].clone()
    }

    pub fn ib3(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(IB3)].clone()
    }

    pub fn ib4(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(IB4)].clone()
    }

    pub fn ib5(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(IB5)].clone()
    }

    pub fn ib6(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(IB6)].clone()
    }

    pub fn jsp(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(JSP)].clone()
    }

    pub fn jsd(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(JSD)].clone()
    }

    pub fn jso(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(JSO)].clone()
    }

    pub fn st0(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST0)].clone()
    }

    pub fn st1(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST1)].clone()
    }

    pub fn st2(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST2)].clone()
    }

    pub fn st3(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST3)].clone()
    }

    pub fn st4(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST4)].clone()
    }

    pub fn st5(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST5)].clone()
    }

    pub fn st6(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST6)].clone()
    }

    pub fn st7(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST7)].clone()
    }

    pub fn st8(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST8)].clone()
    }

    pub fn st9(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST9)].clone()
    }

    pub fn st10(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST10)].clone()
    }

    pub fn st11(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST11)].clone()
    }

    pub fn st12(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST12)].clone()
    }

    pub fn st13(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST13)].clone()
    }

    pub fn st14(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST14)].clone()
    }

    pub fn st15(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(ST15)].clone()
    }

    pub fn osp(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(OSP)].clone()
    }

    pub fn osv(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(OSV)].clone()
    }

    pub fn hv0(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(HV0)].clone()
    }

    pub fn hv1(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(HV1)].clone()
    }

    pub fn hv2(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(HV2)].clone()
    }

    pub fn hv3(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(HV3)].clone()
    }

    pub fn ramv(&self) -> MPolynomial<XFieldElement> {
        self.variables[usize::from(RAMV)].clone()
    }

    // Property: All polynomial variables that contain '_next' have the same
    // variable position / value as the one without '_next', +/- FULL_WIDTH.
    pub fn clk_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(CLK)].clone()
    }

    pub fn ip_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(IP)].clone()
    }

    pub fn ci_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(CI)].clone()
    }

    pub fn jsp_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(JSP)].clone()
    }

    pub fn jsd_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(JSD)].clone()
    }

    pub fn jso_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(JSO)].clone()
    }

    pub fn st0_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(ST0)].clone()
    }

    pub fn st1_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(ST1)].clone()
    }

    pub fn st2_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(ST2)].clone()
    }

    pub fn st3_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(ST3)].clone()
    }

    pub fn st4_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(ST4)].clone()
    }

    pub fn st5_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(ST5)].clone()
    }

    pub fn st6_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(ST6)].clone()
    }

    pub fn st7_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(ST7)].clone()
    }

    pub fn st8_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(ST8)].clone()
    }

    pub fn st9_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(ST9)].clone()
    }

    pub fn st10_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(ST10)].clone()
    }

    pub fn st11_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(ST11)].clone()
    }

    pub fn st12_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(ST12)].clone()
    }

    pub fn st13_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(ST13)].clone()
    }

    pub fn st14_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(ST14)].clone()
    }

    pub fn st15_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(ST15)].clone()
    }

    pub fn osp_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(OSP)].clone()
    }

    pub fn osv_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(OSV)].clone()
    }

    pub fn ramv_next(&self) -> MPolynomial<XFieldElement> {
        self.variables[FULL_WIDTH + usize::from(RAMV)].clone()
    }

    pub fn decompose_arg(&self) -> Vec<MPolynomial<XFieldElement>> {
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

    pub fn step_1(&self) -> Vec<MPolynomial<XFieldElement>> {
        let one = self.one();
        let ip = self.ip();
        let ip_next = self.ip_next();

        vec![ip_next - ip - one]
    }

    pub fn step_2(&self) -> Vec<MPolynomial<XFieldElement>> {
        let one = self.one();
        let ip = self.ip();
        let ip_next = self.ip_next();

        vec![ip_next - ip - (one.clone() + one)]
    }

    pub fn grow_stack(&self) -> Vec<MPolynomial<XFieldElement>> {
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

    pub fn keep_stack(&self) -> Vec<MPolynomial<XFieldElement>> {
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

    pub fn shrink_stack(&self) -> Vec<MPolynomial<XFieldElement>> {
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

    pub fn unop(&self) -> Vec<MPolynomial<XFieldElement>> {
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

    pub fn binop(&self) -> Vec<MPolynomial<XFieldElement>> {
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
    deselectors: HashMap<Instruction, MPolynomial<XFieldElement>>,
}

impl Default for InstructionDeselectors {
    fn default() -> Self {
        let factory = RowPairConstraints::default();
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
    pub fn get(&self, instruction: Instruction) -> MPolynomial<XFieldElement> {
        self.deselectors
            .get(&instruction)
            .unwrap_or_else(|| panic!("The instruction {} does not exist!", instruction))
            .clone()
        // self.deselectors[&instruction].clone()
    }

    /// A polynomial that has no solutions when ci is 'instruction'
    pub fn instruction_deselector(
        factory: &RowPairConstraints,
        instruction: Instruction,
    ) -> MPolynomial<XFieldElement> {
        let one = XFieldElement::one();
        let num_vars = factory.variables.len();

        let ib0 = instruction.ib(Ord7::IB0).lift();
        let ib1 = instruction.ib(Ord7::IB1).lift();
        let ib2 = instruction.ib(Ord7::IB2).lift();
        let ib3 = instruction.ib(Ord7::IB3).lift();
        let ib4 = instruction.ib(Ord7::IB4).lift();
        let ib5 = instruction.ib(Ord7::IB5).lift();
        let ib6 = instruction.ib(Ord7::IB6).lift();
        let deselect_ib0 = MPolynomial::from_constant(one - ib0, num_vars);
        let deselect_ib1 = MPolynomial::from_constant(one - ib1, num_vars);
        let deselect_ib2 = MPolynomial::from_constant(one - ib2, num_vars);
        let deselect_ib3 = MPolynomial::from_constant(one - ib3, num_vars);
        let deselect_ib4 = MPolynomial::from_constant(one - ib4, num_vars);
        let deselect_ib5 = MPolynomial::from_constant(one - ib5, num_vars);
        let deselect_ib6 = MPolynomial::from_constant(one - ib6, num_vars);
        (factory.ib0() - deselect_ib0)
            * (factory.ib1() - deselect_ib1)
            * (factory.ib2() - deselect_ib2)
            * (factory.ib3() - deselect_ib3)
            * (factory.ib4() - deselect_ib4)
            * (factory.ib5() - deselect_ib5)
            * (factory.ib6() - deselect_ib6)
    }

    pub fn create(
        factory: &RowPairConstraints,
    ) -> HashMap<Instruction, MPolynomial<XFieldElement>> {
        all_instructions_without_args()
            .into_iter()
            .map(|instrctn| (instrctn, Self::instruction_deselector(factory, instrctn)))
            .collect()
    }
}

impl ExtensionTable for ExtProcessorTable {
    fn dynamic_initial_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtProcessorTable::ext_initial_constraints(&challenges.processor_table_challenges)
    }

    fn dynamic_consistency_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtProcessorTable::ext_consistency_constraints(&challenges.processor_table_challenges)
    }

    fn dynamic_transition_constraints(
        &self,
        challenges: &super::challenges::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtProcessorTable::ext_transition_constraints(&challenges.processor_table_challenges)
    }

    fn dynamic_terminal_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtProcessorTable::ext_terminal_constraints(&challenges.processor_table_challenges)
    }
}

#[cfg(test)]
mod constraint_polynomial_tests {
    use crate::ord_n::Ord16;
    use crate::table::base_matrix::ProcessorMatrixRow;
    use crate::table::processor_table;
    use crate::vm::Program;

    use super::*;

    #[test]
    /// helps identifying whether the printing causes an infinite loop
    fn print_simple_processor_table_row_test() {
        let program = Program::from_code("push 2 push -1 add assert halt").unwrap();
        let (base_matrices, _, _) = program.simulate_with_input(&[], &[]);
        for row in base_matrices.processor_matrix {
            println!("{}", ProcessorMatrixRow { row });
        }
    }

    fn get_test_row_from_source_code(source_code: &str, row_num: usize) -> Vec<XFieldElement> {
        let fake_extension_columns = [BFieldElement::zero();
            processor_table::FULL_WIDTH - processor_table::BASE_WIDTH]
            .to_vec();

        let program = Program::from_code(source_code).unwrap();
        let (base_matrices, err, _) = program.simulate_with_input(&[], &[]);
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

    fn get_constraints_for_instruction(
        instruction: Instruction,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let tc = RowPairConstraints::default();
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

            for (poly_idx, poly) in get_constraints_for_instruction(instruction)
                .iter()
                .enumerate()
            {
                assert_eq!(
                    instruction.opcode_b().lift(),
                    test_row[usize::from(CI)],
                    "The test is trying to check the wrong constraint polynomials."
                );
                assert_eq!(
                    XFieldElement::zero(),
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
        let deselectors = InstructionDeselectors::default();

        let mut row = vec![0.into(); 2 * FULL_WIDTH];

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
                let result = deselector.evaluate(&row);

                assert!(
                    result.is_zero(),
                    "Deselector for {} should return 0 for all other instructions, including {}",
                    instruction,
                    other_instruction
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
