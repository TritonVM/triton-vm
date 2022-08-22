use super::base_table::{self, BaseTable, BaseTableTrait, HasBaseTable};

use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use super::extension_table::{ExtensionTable, Quotientable, QuotientableExtensionTable};
use super::table_collection::TableId;
use super::table_column::InstructionTableColumn::{self, *};
use crate::fri_domain::FriDomain;
use crate::table::extension_table::Evaluable;
use itertools::Itertools;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::MPolynomial;
use twenty_first::shared_math::x_field_element::XFieldElement;

pub const INSTRUCTION_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 1;
pub const INSTRUCTION_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 1;
pub const INSTRUCTION_TABLE_INITIALS_COUNT: usize =
    INSTRUCTION_TABLE_PERMUTATION_ARGUMENTS_COUNT + INSTRUCTION_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 6 because it combines: (ip, ci, nia) and (addr, instruction, next_instruction).
pub const INSTRUCTION_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 6;

pub const BASE_WIDTH: usize = 3;
pub const FULL_WIDTH: usize = 7; // BASE_WIDTH + 2 * INITIALS_COUNT

type BWord = BFieldElement;
type XWord = XFieldElement;

#[derive(Debug, Clone)]
pub struct InstructionTable {
    base: BaseTable<BWord>,
}

impl HasBaseTable<BWord> for InstructionTable {
    fn to_base(&self) -> &BaseTable<BWord> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<BWord> {
        &mut self.base
    }
}

#[derive(Debug, Clone)]
pub struct ExtInstructionTable {
    base: BaseTable<XFieldElement>,
}

impl Evaluable for ExtInstructionTable {}
impl Quotientable for ExtInstructionTable {}
impl QuotientableExtensionTable for ExtInstructionTable {}

impl HasBaseTable<XFieldElement> for ExtInstructionTable {
    fn to_base(&self) -> &BaseTable<XFieldElement> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<XFieldElement> {
        &mut self.base
    }
}

impl BaseTableTrait<BWord> for InstructionTable {
    fn get_padding_row(&self) -> Vec<BWord> {
        if let Some(row) = self.data().last() {
            let mut padding_row = row.clone();
            // address keeps increasing
            padding_row[InstructionTableColumn::Address as usize] += 1.into();
            padding_row
        } else {
            vec![0.into(); BASE_WIDTH]
        }
    }
}

impl BaseTableTrait<XFieldElement> for ExtInstructionTable {
    fn get_padding_row(&self) -> Vec<XFieldElement> {
        panic!("Extension tables don't get padded");
    }
}

impl ExtInstructionTable {
    fn ext_boundary_constraints() -> Vec<MPolynomial<XWord>> {
        let variables: Vec<MPolynomial<XWord>> = MPolynomial::variables(FULL_WIDTH, 1.into());
        let addr = variables[usize::from(Address)].clone();

        // The first address is 0.
        let fst_addr_is_zero = addr;

        vec![fst_addr_is_zero]
    }

    // TODO actually use consistency constraints
    fn ext_consistency_constraints() -> Vec<MPolynomial<XWord>> {
        // no further constraints
        vec![]
    }

    fn ext_transition_constraints(
        _challenges: &InstructionTableChallenges,
    ) -> Vec<MPolynomial<XWord>> {
        let variables: Vec<MPolynomial<XWord>> = MPolynomial::variables(2 * FULL_WIDTH, 1.into());
        let addr = variables[usize::from(Address)].clone();
        let addr_next = variables[FULL_WIDTH + usize::from(Address)].clone();
        let current_instruction = variables[CI as usize].clone();
        let current_instruction_next = variables[FULL_WIDTH + CI as usize].clone();
        let next_instruction = variables[NIA as usize].clone();
        let next_instruction_next = variables[FULL_WIDTH + NIA as usize].clone();
        let one = MPolynomial::<XFieldElement>::from_constant(1.into(), 2 * FULL_WIDTH);

        let addr_incr_by_one = addr_next - (addr + one);

        // The address increases by 1 or `current_instruction` does not change.
        let addr_incr_by_one_or_ci_stays =
            addr_incr_by_one.clone() * (current_instruction_next - current_instruction);

        // The address increases by 1 or `next_instruction_or_arg` does not change.
        let addr_incr_by_one_or_nia_stays =
            addr_incr_by_one * (next_instruction_next - next_instruction);

        vec![addr_incr_by_one_or_ci_stays, addr_incr_by_one_or_nia_stays]
    }

    fn ext_terminal_constraints(
        _challenges: &InstructionTableChallenges,
        _terminals: &InstructionTableEndpoints,
    ) -> Vec<MPolynomial<XWord>> {
        // no further constraints
        vec![]
    }

    pub fn for_verifier(
        num_trace_randomizers: usize,
        padded_height: usize,
        all_challenges: &AllChallenges,
        all_terminals: &AllEndpoints,
    ) -> Self {
        let omicron = base_table::derive_omicron(padded_height as u64);
        let base = BaseTable::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            vec![],
            "ExtInstructionTable".to_string(),
            TableId::InstructionTable,
        );
        let table = BaseTable::extension(
            base,
            ExtInstructionTable::ext_boundary_constraints(),
            ExtInstructionTable::ext_transition_constraints(
                &all_challenges.instruction_table_challenges,
            ),
            ExtInstructionTable::ext_consistency_constraints(),
            ExtInstructionTable::ext_terminal_constraints(
                &all_challenges.instruction_table_challenges,
                &all_terminals.instruction_table_endpoints,
            ),
        );

        ExtInstructionTable { base: table }
    }
}

impl InstructionTable {
    pub fn new_prover(num_trace_randomizers: usize, matrix: Vec<Vec<BWord>>) -> Self {
        let unpadded_height = matrix.len();
        let padded_height = base_table::pad_height(unpadded_height);

        let omicron = base_table::derive_omicron(padded_height as u64);
        let base = BaseTable::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            matrix,
            "InstructionTable".to_string(),
            TableId::InstructionTable,
        );

        Self { base }
    }

    pub fn codeword_table(&self, fri_domain: &FriDomain<BWord>) -> Self {
        let base_columns = 0..self.base_width();
        let codewords = self.low_degree_extension(fri_domain, base_columns);

        let base = self.base.with_data(codewords);
        Self { base }
    }

    pub fn extend(
        &self,
        challenges: &InstructionTableChallenges,
        initials: &InstructionTableEndpoints,
    ) -> (ExtInstructionTable, InstructionTableEndpoints) {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        let mut processor_table_running_product = initials.processor_perm_product;
        let mut program_table_running_sum = initials.program_eval_sum;
        let mut previous_row: Option<Vec<_>> = None;

        for row in self.data().iter() {
            let mut extension_row = Vec::with_capacity(FULL_WIDTH);
            extension_row.extend(row.iter().map(|elem| elem.lift()));

            // Is the current row's address different from the previous row's address?
            // Different: update running sum of Evaluation Argument with Program Table.
            // Not different: update running product of Permutation Argument with Processor Table.
            let mut is_duplicate_row = false;
            if let Some(prow) = previous_row {
                if prow[Address as usize] == row[Address as usize] {
                    is_duplicate_row = true;
                    debug_assert_eq!(prow[CI as usize], row[CI as usize]);
                    debug_assert_eq!(prow[NIA as usize], row[NIA as usize]);
                }
            }

            // Compress values of current row for Permutation Argument with Processor Table
            let ip = row[Address as usize].lift();
            let ci = row[CI as usize].lift();
            let nia = row[NIA as usize].lift();
            let compressed_row_for_permutation_argument = ip * challenges.ip_processor_weight
                + ci * challenges.ci_processor_weight
                + nia * challenges.nia_processor_weight;
            extension_row.push(compressed_row_for_permutation_argument);

            // Update running product for Permutation Argument if same row has been seen before
            extension_row.push(processor_table_running_product);
            if is_duplicate_row {
                processor_table_running_product *=
                    challenges.processor_perm_row_weight - compressed_row_for_permutation_argument;
            }

            // Compress values of current row for Evaluation Argument with Program Table
            let address = row[Address as usize].lift();
            let instruction = row[CI as usize].lift();
            let next_instruction = row[NIA as usize].lift();
            let compressed_row_for_evaluation_argument = address * challenges.address_weight
                + instruction * challenges.instruction_weight
                + next_instruction * challenges.next_instruction_weight;
            extension_row.push(compressed_row_for_evaluation_argument);

            // Update running sum for Evaluation Argument if same row has _not_ been seen before
            extension_row.push(program_table_running_sum);
            if !is_duplicate_row {
                program_table_running_sum = program_table_running_sum
                    * challenges.program_eval_row_weight
                    + compressed_row_for_evaluation_argument;
            }

            debug_assert_eq!(
                FULL_WIDTH,
                extension_row.len(),
                "After extending, the row must match the table's full width."
            );

            previous_row = Some(row.clone());
            extension_matrix.push(extension_row);
        }

        let terminals = InstructionTableEndpoints {
            processor_perm_product: processor_table_running_product,
            program_eval_sum: program_table_running_sum,
        };

        let base = self.base.with_lifted_data(extension_matrix);
        let table = BaseTable::extension(
            base,
            ExtInstructionTable::ext_boundary_constraints(),
            ExtInstructionTable::ext_transition_constraints(challenges),
            ExtInstructionTable::ext_consistency_constraints(),
            ExtInstructionTable::ext_terminal_constraints(challenges, &terminals),
        );

        (ExtInstructionTable { base: table }, terminals)
    }
}

impl ExtInstructionTable {
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
            "ExtInstructionTable".to_string(),
            TableId::InstructionTable,
        );

        Self { base }
    }

    pub fn ext_codeword_table(
        &self,
        fri_domain: &FriDomain<XWord>,
        base_codewords: &[Vec<BWord>],
    ) -> Self {
        let ext_columns = self.base_width()..self.full_width();
        let ext_codewords = self.low_degree_extension(fri_domain, ext_columns);

        let lifted_base_codewords = base_codewords
            .iter()
            .map(|base_codeword| base_codeword.iter().map(|bfe| bfe.lift()).collect_vec())
            .collect_vec();
        let all_codewords = vec![lifted_base_codewords, ext_codewords].concat();
        assert_eq!(self.full_width(), all_codewords.len());

        let base = self.base.with_data(all_codewords);
        ExtInstructionTable { base }
    }
}

#[derive(Debug, Clone)]
pub struct InstructionTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the instruction table.
    pub processor_perm_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub ip_processor_weight: XFieldElement,
    pub ci_processor_weight: XFieldElement,
    pub nia_processor_weight: XFieldElement,

    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the instruction table.
    pub program_eval_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to program table.)
    pub address_weight: XFieldElement,
    pub instruction_weight: XFieldElement,
    pub next_instruction_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct InstructionTableEndpoints {
    /// Values randomly generated by the prover for zero-knowledge.
    pub processor_perm_product: XFieldElement,
    pub program_eval_sum: XFieldElement,
}

impl ExtensionTable for ExtInstructionTable {
    fn dynamic_boundary_constraints(&self) -> Vec<MPolynomial<XFieldElement>> {
        ExtInstructionTable::ext_boundary_constraints()
    }

    fn dynamic_transition_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtInstructionTable::ext_transition_constraints(&challenges.instruction_table_challenges)
    }

    fn dynamic_consistency_constraints(&self) -> Vec<MPolynomial<XFieldElement>> {
        ExtInstructionTable::ext_consistency_constraints()
    }

    fn dynamic_terminal_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
        terminals: &super::challenges_endpoints::AllEndpoints,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtInstructionTable::ext_terminal_constraints(
            &challenges.instruction_table_challenges,
            &terminals.instruction_table_endpoints,
        )
    }
}
