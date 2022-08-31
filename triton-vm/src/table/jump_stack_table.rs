use super::base_table::{self, InheritsFromTable, Table, TableLike};
use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use super::extension_table::{ExtensionTable, Quotientable, QuotientableExtensionTable};
use super::table_column::JumpStackTableColumn::*;
use crate::fri_domain::FriDomain;
use crate::instruction::Instruction;
use crate::table::base_table::Extendable;
use crate::table::extension_table::Evaluable;
use itertools::Itertools;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::MPolynomial;
use twenty_first::shared_math::x_field_element::XFieldElement;

pub const JUMP_STACK_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 1;
pub const JUMP_STACK_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 0;
pub const JUMP_STACK_TABLE_INITIALS_COUNT: usize =
    JUMP_STACK_TABLE_PERMUTATION_ARGUMENTS_COUNT + JUMP_STACK_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 5 because it combines: clk, ci, jsp, jso, jsd,
pub const JUMP_STACK_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 5;

pub const BASE_WIDTH: usize = 5;
pub const FULL_WIDTH: usize = 7; // BASE_WIDTH + 2 * INITIALS_COUNT

#[derive(Debug, Clone)]
pub struct JumpStackTable {
    inherited_table: Table<BFieldElement>,
}

impl InheritsFromTable<BFieldElement> for JumpStackTable {
    fn inherited_table(&self) -> &Table<BFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<BFieldElement> {
        &mut self.inherited_table
    }
}

#[derive(Debug, Clone)]
pub struct ExtJumpStackTable {
    inherited_table: Table<XFieldElement>,
}

impl Evaluable for ExtJumpStackTable {}
impl Quotientable for ExtJumpStackTable {}
impl QuotientableExtensionTable for ExtJumpStackTable {}

impl InheritsFromTable<XFieldElement> for ExtJumpStackTable {
    fn inherited_table(&self) -> &Table<XFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<XFieldElement> {
        &mut self.inherited_table
    }
}

impl TableLike<BFieldElement> for JumpStackTable {}

impl Extendable for JumpStackTable {
    fn get_padding_row(&self) -> Vec<BFieldElement> {
        if let Some(row) = self.data().last() {
            let mut padding_row = row.clone();
            // add same clk padding as in processor table
            padding_row[CLK as usize] = (self.data().len() as u32).into();
            padding_row
        } else {
            vec![0.into(); BASE_WIDTH]
        }
        // todo: use code below once the table is derived from the processor after that got padded
        // panic!("This table gets derived from the padded processor table – no more padding here.")
    }
}

impl TableLike<XFieldElement> for ExtJumpStackTable {}

impl ExtJumpStackTable {
    fn ext_boundary_constraints() -> Vec<MPolynomial<XFieldElement>> {
        let variables: Vec<MPolynomial<XFieldElement>> =
            MPolynomial::variables(FULL_WIDTH, 1.into());

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

    // TODO actually use consistency constraints
    fn ext_consistency_constraints() -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }

    fn ext_transition_constraints(
        _challenges: &JumpStackTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let variables: Vec<MPolynomial<XFieldElement>> =
            MPolynomial::variables(2 * FULL_WIDTH, 1.into());
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
        _challenges: &JumpStackTableChallenges,
        _terminals: &JumpStackTableEndpoints,
    ) -> Vec<MPolynomial<XFieldElement>> {
        vec![]
    }
}

impl JumpStackTable {
    pub fn new_prover(num_trace_randomizers: usize, matrix: Vec<Vec<BFieldElement>>) -> Self {
        let unpadded_height = matrix.len();
        let padded_height = base_table::pad_height(unpadded_height);

        let omicron = base_table::derive_omicron(padded_height as u64);
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            matrix,
            "JumpStackTable".to_string(),
        );

        Self { inherited_table }
    }

    pub fn codeword_table(&self, fri_domain: &FriDomain<BFieldElement>) -> Self {
        let base_columns = 0..self.base_width();
        let codewords = self.low_degree_extension(fri_domain, base_columns);

        let inherited_table = self.inherited_table.with_data(codewords);
        Self { inherited_table }
    }

    pub fn sort(&mut self) {
        self.mut_data()
            .sort_by_key(|row| (row[JSP as usize].value(), row[CLK as usize].value()))
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

        let terminals = JumpStackTableEndpoints {
            processor_perm_product: running_product,
        };

        let inherited_table = self.extension(
            extension_matrix,
            ExtJumpStackTable::ext_boundary_constraints(),
            ExtJumpStackTable::ext_transition_constraints(challenges),
            ExtJumpStackTable::ext_consistency_constraints(),
            ExtJumpStackTable::ext_terminal_constraints(challenges, &terminals),
        );
        (ExtJumpStackTable { inherited_table }, terminals)
    }

    pub fn for_verifier(
        num_trace_randomizers: usize,
        padded_height: usize,
        all_challenges: &AllChallenges,
        all_terminals: &AllEndpoints,
    ) -> ExtJumpStackTable {
        let omicron = base_table::derive_omicron(padded_height as u64);
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            vec![],
            "ExtJumpStackTable".to_string(),
        );
        let base_table = Self { inherited_table };
        let empty_matrix: Vec<Vec<XFieldElement>> = vec![];
        let extension_table = base_table.extension(
            empty_matrix,
            ExtJumpStackTable::ext_boundary_constraints(),
            ExtJumpStackTable::ext_transition_constraints(
                &all_challenges.jump_stack_table_challenges,
            ),
            ExtJumpStackTable::ext_consistency_constraints(),
            ExtJumpStackTable::ext_terminal_constraints(
                &all_challenges.jump_stack_table_challenges,
                &all_terminals.jump_stack_table_endpoints,
            ),
        );

        ExtJumpStackTable {
            inherited_table: extension_table,
        }
    }
}

impl ExtJumpStackTable {
    pub fn with_padded_height(num_trace_randomizers: usize, padded_height: usize) -> Self {
        let matrix: Vec<Vec<XFieldElement>> = vec![];

        let omicron = base_table::derive_omicron(padded_height as u64);
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            matrix,
            "ExtJumpStackTable".to_string(),
        );

        Self { inherited_table }
    }

    pub fn ext_codeword_table(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        base_codewords: &[Vec<BFieldElement>],
    ) -> Self {
        let ext_columns = self.base_width()..self.full_width();
        let ext_codewords = self.low_degree_extension(fri_domain, ext_columns);

        let lifted_base_codewords = base_codewords
            .iter()
            .map(|base_codeword| base_codeword.iter().map(|bfe| bfe.lift()).collect_vec())
            .collect_vec();
        let all_codewords = vec![lifted_base_codewords, ext_codewords].concat();
        assert_eq!(self.full_width(), all_codewords.len());

        let inherited_table = self.inherited_table.with_data(all_codewords);
        ExtJumpStackTable { inherited_table }
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

impl ExtensionTable for ExtJumpStackTable {
    fn dynamic_boundary_constraints(&self) -> Vec<MPolynomial<XFieldElement>> {
        ExtJumpStackTable::ext_boundary_constraints()
    }

    fn dynamic_transition_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtJumpStackTable::ext_transition_constraints(&challenges.jump_stack_table_challenges)
    }

    fn dynamic_consistency_constraints(&self) -> Vec<MPolynomial<XFieldElement>> {
        ExtJumpStackTable::ext_consistency_constraints()
    }

    fn dynamic_terminal_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
        terminals: &super::challenges_endpoints::AllEndpoints,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtJumpStackTable::ext_terminal_constraints(
            &challenges.jump_stack_table_challenges,
            &terminals.jump_stack_table_endpoints,
        )
    }
}
