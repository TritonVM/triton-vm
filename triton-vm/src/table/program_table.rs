use super::base_table::{self, BaseTable, BaseTableTrait, HasBaseTable};
use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use super::extension_table::{ExtensionTable, Quotientable, QuotientableExtensionTable};
use super::table_collection::TableId;
use super::table_column::ProgramTableColumn;
use crate::fri_domain::FriDomain;
use itertools::Itertools;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::MPolynomial;
use twenty_first::shared_math::x_field_element::XFieldElement;

pub const PROGRAM_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 0;
pub const PROGRAM_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 1;
pub const PROGRAM_TABLE_INITIALS_COUNT: usize =
    PROGRAM_TABLE_PERMUTATION_ARGUMENTS_COUNT + PROGRAM_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 3 because it combines: addr, instruction, instruction in next row
pub const PROGRAM_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 3;

pub const BASE_WIDTH: usize = 2;
pub const FULL_WIDTH: usize = 4; // BASE_WIDTH + 2 * INITIALS_COUNT

type BWord = BFieldElement;
type XWord = XFieldElement;

#[derive(Debug, Clone)]
pub struct ProgramTable {
    base: BaseTable<BWord>,
}

impl HasBaseTable<BWord> for ProgramTable {
    fn to_base(&self) -> &BaseTable<BWord> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<BWord> {
        &mut self.base
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtProgramTable {
    base: BaseTable<XFieldElement>,
}

impl Quotientable for ExtProgramTable {}
impl QuotientableExtensionTable for ExtProgramTable {}

impl HasBaseTable<XFieldElement> for ExtProgramTable {
    fn to_base(&self) -> &BaseTable<XFieldElement> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<XFieldElement> {
        &mut self.base
    }
}

impl BaseTableTrait<BWord> for ProgramTable {
    fn get_padding_row(&self) -> Vec<BWord> {
        let mut padding_row = self.data().last().unwrap().clone();
        // address keeps increasing
        padding_row[ProgramTableColumn::Address as usize] += 1.into();
        padding_row
    }
}

impl BaseTableTrait<XFieldElement> for ExtProgramTable {
    fn get_padding_row(&self) -> Vec<XFieldElement> {
        panic!("Extension tables don't get padded");
    }
}

impl ExtProgramTable {
    fn ext_boundary_constraints(_challenges: &ProgramTableChallenges) -> Vec<MPolynomial<XWord>> {
        use ProgramTableColumn::*;

        let variables: Vec<MPolynomial<XWord>> = MPolynomial::variables(FULL_WIDTH, 1.into());

        let addr = variables[Address as usize].clone();

        // The first address is 0.
        //
        // $addr - 0 = 0  =>  addr$
        vec![addr]
    }

    // TODO actually use consistency constraints
    fn ext_consistency_constraints(
        _challenges: &ProgramTableChallenges,
    ) -> Vec<MPolynomial<XWord>> {
        // no further constraints
        vec![]
    }

    fn ext_transition_constraints(_challenges: &ProgramTableChallenges) -> Vec<MPolynomial<XWord>> {
        use ProgramTableColumn::*;

        let variables: Vec<MPolynomial<XWord>> = MPolynomial::variables(2 * FULL_WIDTH, 1.into());

        let addr = variables[Address as usize].clone();
        let addr_next = variables[FULL_WIDTH + Address as usize].clone();
        let one = MPolynomial::<XWord>::from_constant(1.into(), 2 * FULL_WIDTH);

        // The address increases by 1.
        //
        // $addr' - (addr + 1) = 0$
        vec![addr_next - (addr + one)]
    }

    fn ext_terminal_constraints(
        _challenges: &ProgramTableChallenges,
        _terminals: &ProgramTableEndpoints,
    ) -> Vec<MPolynomial<XWord>> {
        // no further constraints
        vec![]
    }
}

impl ProgramTable {
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
            "ProgramTable".to_string(),
            TableId::ProgramTable,
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
        challenges: &ProgramTableChallenges,
        initials: &ProgramTableEndpoints,
    ) -> (ExtProgramTable, ProgramTableEndpoints) {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        let mut instruction_table_running_sum = initials.instruction_eval_sum;

        let mut data_with_0 = self.data().clone();
        data_with_0.push(vec![0.into(); BASE_WIDTH]);

        for (row, next_row) in data_with_0.into_iter().tuple_windows() {
            let mut extension_row = Vec::with_capacity(FULL_WIDTH);
            extension_row.extend(row.iter().map(|elem| elem.lift()));

            let address = row[ProgramTableColumn::Address as usize].lift();
            let instruction = row[ProgramTableColumn::Instruction as usize].lift();
            let next_instruction = next_row[ProgramTableColumn::Instruction as usize].lift();

            // Compress address, instruction, and next instruction (or argument) into single value
            let compressed_row_for_evaluation_argument = address * challenges.address_weight
                + instruction * challenges.instruction_weight
                + next_instruction * challenges.next_instruction_weight;
            extension_row.push(compressed_row_for_evaluation_argument);

            // Update the Evaluation Argument's running sum with the compressed column
            extension_row.push(instruction_table_running_sum);
            instruction_table_running_sum = instruction_table_running_sum
                * challenges.instruction_eval_row_weight
                + compressed_row_for_evaluation_argument;

            debug_assert_eq!(
                FULL_WIDTH,
                extension_row.len(),
                "After extending, the row must match the table's full width."
            );

            extension_matrix.push(extension_row);
        }

        let terminals = ProgramTableEndpoints {
            instruction_eval_sum: instruction_table_running_sum,
        };

        let base = self.base.with_lifted_data(extension_matrix);
        let table = BaseTable::extension(
            base,
            ExtProgramTable::ext_boundary_constraints(challenges),
            ExtProgramTable::ext_transition_constraints(&challenges),
            ExtProgramTable::ext_consistency_constraints(&challenges),
            ExtProgramTable::ext_terminal_constraints(&challenges, &terminals),
        );

        (ExtProgramTable { base: table }, terminals)
    }
}

impl ExtProgramTable {
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
            "ExtProgramTable".to_string(),
            TableId::ProgramTable,
        );

        Self { base }
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
            "ExtProgramTable".to_string(),
            TableId::ProgramTable,
        );
        let table = BaseTable::extension(
            base,
            ExtProgramTable::ext_boundary_constraints(&all_challenges.program_table_challenges),
            ExtProgramTable::ext_transition_constraints(&all_challenges.program_table_challenges),
            ExtProgramTable::ext_consistency_constraints(&all_challenges.program_table_challenges),
            ExtProgramTable::ext_terminal_constraints(
                &all_challenges.program_table_challenges,
                &all_terminals.program_table_endpoints,
            ),
        );

        ExtProgramTable { base: table }
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
        ExtProgramTable { base }
    }
}

#[derive(Debug, Clone)]
pub struct ProgramTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the program table.
    pub instruction_eval_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to program table.)
    pub address_weight: XFieldElement,
    pub instruction_weight: XFieldElement,
    pub next_instruction_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct ProgramTableEndpoints {
    /// Values randomly generated by the prover for zero-knowledge.
    pub instruction_eval_sum: XFieldElement,
}

impl ExtensionTable for ExtProgramTable {
    fn dynamic_boundary_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtProgramTable::ext_boundary_constraints(&challenges.program_table_challenges)
    }

    fn dynamic_transition_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtProgramTable::ext_transition_constraints(&challenges.program_table_challenges)
    }

    fn dynamic_consistency_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtProgramTable::ext_consistency_constraints(&challenges.program_table_challenges)
    }

    fn dynamic_terminal_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
        terminals: &super::challenges_endpoints::AllEndpoints,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtProgramTable::ext_terminal_constraints(
            &challenges.program_table_challenges,
            &terminals.program_table_endpoints,
        )
    }
}
