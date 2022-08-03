use super::base_table::{self, BaseTable, HasBaseTable, Table};
use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use super::extension_table::ExtensionTable;
use super::table_column::JumpStackTableColumn::*;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::stark::triton::fri_domain::FriDomain;
use crate::shared_math::stark::triton::instruction::Instruction;
use crate::shared_math::x_field_element::XFieldElement;
use itertools::Itertools;

pub const JUMP_STACK_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 1;
pub const JUMP_STACK_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 0;
pub const JUMP_STACK_TABLE_INITIALS_COUNT: usize =
    JUMP_STACK_TABLE_PERMUTATION_ARGUMENTS_COUNT + JUMP_STACK_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 5 because it combines: clk, ci, jsp, jso, jsd,
pub const JUMP_STACK_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 5;

pub const BASE_WIDTH: usize = 5;
pub const FULL_WIDTH: usize = 7; // BASE_WIDTH + 2 * INITIALS_COUNT

type BWord = BFieldElement;
type XWord = XFieldElement;

#[derive(Debug, Clone)]
pub struct JumpStackTable {
    base: BaseTable<BWord>,
}

impl HasBaseTable<BWord> for JumpStackTable {
    fn to_base(&self) -> &BaseTable<BWord> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<BWord> {
        &mut self.base
    }
}

#[derive(Debug, Clone)]
pub struct ExtJumpStackTable {
    base: BaseTable<XFieldElement>,
}

impl HasBaseTable<XFieldElement> for ExtJumpStackTable {
    fn to_base(&self) -> &BaseTable<XFieldElement> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<XFieldElement> {
        &mut self.base
    }
}

impl Table<BWord> for JumpStackTable {
    fn get_padding_row(&self) -> Vec<BWord> {
        let mut padding_row = self.data().last().unwrap().clone();
        // add same clk padding as in processor table
        padding_row[CLK as usize] = (self.data().len() as u32).into();
        padding_row
    }
}

impl Table<XFieldElement> for ExtJumpStackTable {
    fn get_padding_row(&self) -> Vec<XFieldElement> {
        panic!("Extension tables don't get padded");
    }
}

impl ExtensionTable for ExtJumpStackTable {
    fn ext_boundary_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        let variables: Vec<MPolynomial<XWord>> = MPolynomial::variables(FULL_WIDTH, 1.into());

        // 1. Cycle count clk is 0.
        let clk = variables[usize::from(CLK)].clone();

        // 2. Jump Stack Pointer jsp is 0.
        let jsp = variables[usize::from(JSP)].clone();

        // 3. Jump Stack Origin jso is 0.
        let jso = variables[usize::from(JSO)].clone();

        // 4. Jump Stack Destination jsd is 0.
        let jsd = variables[usize::from(JSD)].clone();

        vec![clk, jsp, jso, jsd]
    }

    fn ext_consistency_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        // no further constraints
        vec![]
    }

    fn ext_transition_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        let variables: Vec<MPolynomial<XWord>> = MPolynomial::variables(2 * FULL_WIDTH, 1.into());
        let one = MPolynomial::<XFieldElement>::from_constant(1.into(), 2 * FULL_WIDTH);

        let clk = variables[usize::from(CLK)].clone();
        let ci = variables[usize::from(CI)].clone();
        let jsp = variables[usize::from(JSP)].clone();
        let jso = variables[usize::from(JSO)].clone();
        let jsd = variables[usize::from(JSD)].clone();
        let clk_next = variables[FULL_WIDTH + usize::from(CLK)].clone();
        let _ci_next = variables[FULL_WIDTH + usize::from(CI)].clone();
        let jsp_next = variables[FULL_WIDTH + usize::from(JSP)].clone();
        let jso_next = variables[FULL_WIDTH + usize::from(JSO)].clone();
        let jsd_next = variables[FULL_WIDTH + usize::from(JSD)].clone();

        let call_opcode = MPolynomial::<XFieldElement>::from_constant(
            Instruction::Call(Default::default()).opcode_b().lift(),
            2 * FULL_WIDTH,
        );

        let return_opcode = MPolynomial::<XFieldElement>::from_constant(
            Instruction::Return.opcode_b().lift(),
            2 * FULL_WIDTH,
        );

        // 1. The jump stack pointer jsp increases by 1
        //      or the jump stack pointer jsp does not change
        let jsp_inc_or_stays =
            (jsp_next.clone() - (jsp.clone() + one.clone())) * (jsp_next.clone() - jsp.clone());

        // 2. The jump stack pointer jsp increases by 1
        //      or the jump stack origin jso does not change
        //      or current instruction ci is return
        let jsp_inc_or_jso_stays_or_ci_is_ret = (jsp_next.clone() - (jsp.clone() + one.clone()))
            * (jso_next - jso)
            * (ci.clone() - return_opcode.clone());

        // 3. The jump stack pointer jsp increases by 1
        //      or the jump stack destination jsd does not change
        //      or current instruction ci is return
        let jsp_inc_or_jsd_stays_or_ci_ret = (jsp_next.clone() - (jsp.clone() + one.clone()))
            * (jsd_next - jsd)
            * (ci.clone() - return_opcode.clone());

        // 4. The jump stack pointer jsp increases by 1
        //      or the cycle count clk increases by 1
        //      or current instruction ci is call
        //      or current instruction ci is return
        let jsp_inc_or_clk_inc_or_ci_call_or_ci_ret = (jsp_next - (jsp + one.clone()))
            * (clk_next - (clk + one))
            * (ci.clone() - call_opcode)
            * (ci - return_opcode);

        vec![
            jsp_inc_or_stays,
            jsp_inc_or_jso_stays_or_ci_is_ret,
            jsp_inc_or_jsd_stays_or_ci_ret,
            jsp_inc_or_clk_inc_or_ci_call_or_ci_ret,
        ]
    }

    fn ext_terminal_constraints(
        &self,
        _challenges: &AllChallenges,
        _terminals: &AllEndpoints,
    ) -> Vec<MPolynomial<XWord>> {
        vec![]
    }
}

impl JumpStackTable {
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
            "JumpStackTable".to_string(),
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
        challenges: &JumpStackTableChallenges,
        initials: &JumpStackTableEndpoints,
    ) -> (ExtJumpStackTable, JumpStackTableEndpoints) {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        let mut running_product = initials.processor_perm_product;

        for row in self.data().iter() {
            let mut extension_row = Vec::with_capacity(FULL_WIDTH);
            extension_row.extend(row.iter().map(|elem| elem.lift()));

            let (clk, ci, jsp, jso, jsd) = (
                extension_row[CLK as usize],
                extension_row[CI as usize],
                extension_row[JSP as usize],
                extension_row[JSO as usize],
                extension_row[JSD as usize],
            );

            let (clk_w, ci_w, jsp_w, jso_w, jsd_w) = (
                challenges.clk_weight,
                challenges.ci_weight,
                challenges.jsp_weight,
                challenges.jso_weight,
                challenges.jsd_weight,
            );

            // 1. Compress multiple values within one row so they become one value.
            let compressed_row_for_permutation_argument =
                clk * clk_w + ci * ci_w + jsp * jsp_w + jso * jso_w + jsd * jsd_w;

            extension_row.push(compressed_row_for_permutation_argument);

            // 2. Compute the running *product* of the compressed column (permutation value)
            extension_row.push(running_product);
            running_product *=
                challenges.processor_perm_row_weight - compressed_row_for_permutation_argument;

            extension_matrix.push(extension_row);
        }

        let base = self.base.with_lifted_data(extension_matrix);
        let table = ExtJumpStackTable { base };
        let terminals = JumpStackTableEndpoints {
            processor_perm_product: running_product,
        };

        (table, terminals)
    }
}

impl ExtJumpStackTable {
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
            "ExtJumpStackTable".to_string(),
        );

        Self { base }
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
        ExtJumpStackTable { base }
    }
}

#[derive(Debug, Clone)]
pub struct JumpStackTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the op-stack table.
    pub processor_perm_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub clk_weight: XFieldElement,
    pub ci_weight: XFieldElement,
    pub jsp_weight: XFieldElement,
    pub jso_weight: XFieldElement,
    pub jsd_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct JumpStackTableEndpoints {
    /// Values randomly generated by the prover for zero-knowledge.
    pub processor_perm_product: XFieldElement,
}
