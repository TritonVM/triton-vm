use itertools::Itertools;
use num_traits::{One, Zero};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::{Degree, MPolynomial};
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::fri_domain::FriDomain;
use crate::table::base_table::Extendable;
use crate::table::extension_table::Evaluable;
use crate::table::table_column::ExtProgramTableColumn::*;
use crate::table::table_column::ProgramTableColumn::*;

use super::base_table::{InheritsFromTable, Table, TableLike};
use super::challenges::AllChallenges;
use super::extension_table::{ExtensionTable, Quotientable, QuotientableExtensionTable};

pub const PROGRAM_TABLE_NUM_PERMUTATION_ARGUMENTS: usize = 0;
pub const PROGRAM_TABLE_NUM_EVALUATION_ARGUMENTS: usize = 1;

/// This is 3 because it combines: addr, instruction, instruction in next row
pub const PROGRAM_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 3;

pub const BASE_WIDTH: usize = 3;
pub const FULL_WIDTH: usize =
    BASE_WIDTH + PROGRAM_TABLE_NUM_PERMUTATION_ARGUMENTS + PROGRAM_TABLE_NUM_EVALUATION_ARGUMENTS;

#[derive(Debug, Clone)]
pub struct ProgramTable {
    inherited_table: Table<BFieldElement>,
}

impl InheritsFromTable<BFieldElement> for ProgramTable {
    fn inherited_table(&self) -> &Table<BFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<BFieldElement> {
        &mut self.inherited_table
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtProgramTable {
    inherited_table: Table<XFieldElement>,
}

impl Default for ExtProgramTable {
    fn default() -> Self {
        Self {
            inherited_table: Table::new(
                BASE_WIDTH,
                FULL_WIDTH,
                vec![],
                "EmptyExtProgramTable".to_string(),
            ),
        }
    }
}

impl Evaluable for ExtProgramTable {}
impl Quotientable for ExtProgramTable {}
impl QuotientableExtensionTable for ExtProgramTable {}

impl InheritsFromTable<XFieldElement> for ExtProgramTable {
    fn inherited_table(&self) -> &Table<XFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<XFieldElement> {
        &mut self.inherited_table
    }
}

impl TableLike<BFieldElement> for ProgramTable {}

impl Extendable for ProgramTable {
    fn get_padding_rows(&self) -> (Option<usize>, Vec<Vec<BFieldElement>>) {
        let zero = BFieldElement::zero();
        let one = BFieldElement::one();
        if let Some(row) = self.data().last() {
            let mut padding_row = row.clone();
            // address keeps increasing
            padding_row[Address as usize] += one;
            padding_row[Instruction as usize] = zero;
            padding_row[IsPadding as usize] = one;
            (None, vec![padding_row])
        } else {
            // Not that it makes much sense to run a program with no instructions.
            let mut padding_row = [zero; BASE_WIDTH];
            padding_row[IsPadding as usize] = one;
            (None, vec![padding_row.to_vec()])
        }
    }
}

impl TableLike<XFieldElement> for ExtProgramTable {}

impl ExtProgramTable {
    fn ext_initial_constraints(
        _challenges: &ProgramTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let variables: Vec<MPolynomial<XFieldElement>> =
            MPolynomial::variables(FULL_WIDTH, 1.into());

        let address = variables[Address as usize].clone();

        let first_address_is_zero = address;

        vec![first_address_is_zero]
    }

    fn ext_consistency_constraints(
        _challenges: &ProgramTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }

    fn ext_transition_constraints(
        _challenges: &ProgramTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let variables: Vec<MPolynomial<XFieldElement>> =
            MPolynomial::variables(2 * FULL_WIDTH, 1.into());

        let addr = variables[Address as usize].clone();
        let addr_next = variables[FULL_WIDTH + Address as usize].clone();
        let one = MPolynomial::<XFieldElement>::from_constant(1.into(), 2 * FULL_WIDTH);

        let address_increases_by_one = addr_next - (addr + one);
        vec![address_increases_by_one]
    }

    fn ext_terminal_constraints(
        _challenges: &ProgramTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }
}

impl ProgramTable {
    pub fn new_prover(matrix: Vec<Vec<BFieldElement>>) -> Self {
        let inherited_table =
            Table::new(BASE_WIDTH, FULL_WIDTH, matrix, "ProgramTable".to_string());
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
        challenges: &ProgramTableChallenges,
        interpolant_degree: Degree,
    ) -> ExtProgramTable {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        let mut instruction_table_running_evaluation = XFieldElement::zero();

        let mut data_with_0 = self.data().clone();
        data_with_0.push(vec![BFieldElement::zero(); BASE_WIDTH]);

        for (row, next_row) in data_with_0.into_iter().tuple_windows() {
            let mut extension_row = [0.into(); FULL_WIDTH];
            extension_row[..BASE_WIDTH]
                .copy_from_slice(&row.iter().map(|elem| elem.lift()).collect_vec());

            let address = row[Address as usize].lift();
            let instruction = row[Instruction as usize].lift();
            let next_instruction = next_row[Instruction as usize].lift();

            // Update the running evaluation if not a padding row
            if row[IsPadding as usize].is_zero() {
                // Compress address, instruction, and next instruction (or argument) into single value
                let compressed_row_for_evaluation_argument = address * challenges.address_weight
                    + instruction * challenges.instruction_weight
                    + next_instruction * challenges.next_instruction_weight;

                instruction_table_running_evaluation = instruction_table_running_evaluation
                    * challenges.instruction_eval_row_weight
                    + compressed_row_for_evaluation_argument;
            }
            extension_row[usize::from(RunningEvaluation)] = instruction_table_running_evaluation;

            extension_matrix.push(extension_row.to_vec());
        }

        let inherited_table = self.extension(
            extension_matrix,
            interpolant_degree,
            ExtProgramTable::ext_initial_constraints(challenges),
            ExtProgramTable::ext_consistency_constraints(challenges),
            ExtProgramTable::ext_transition_constraints(challenges),
            ExtProgramTable::ext_terminal_constraints(challenges),
        );
        ExtProgramTable { inherited_table }
    }

    pub fn for_verifier(
        interpolant_degree: Degree,
        all_challenges: &AllChallenges,
    ) -> ExtProgramTable {
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            vec![],
            "ExtProgramTable".to_string(),
        );
        let base_table = Self { inherited_table };
        let empty_matrix: Vec<Vec<XFieldElement>> = vec![];
        let extension_table = base_table.extension(
            empty_matrix,
            interpolant_degree,
            ExtProgramTable::ext_initial_constraints(&all_challenges.program_table_challenges),
            ExtProgramTable::ext_consistency_constraints(&all_challenges.program_table_challenges),
            ExtProgramTable::ext_transition_constraints(&all_challenges.program_table_challenges),
            ExtProgramTable::ext_terminal_constraints(&all_challenges.program_table_challenges),
        );

        ExtProgramTable {
            inherited_table: extension_table,
        }
    }
}

impl ExtProgramTable {
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
        ExtProgramTable { inherited_table }
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

impl ExtensionTable for ExtProgramTable {
    fn dynamic_initial_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtProgramTable::ext_initial_constraints(&challenges.program_table_challenges)
    }

    fn dynamic_consistency_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtProgramTable::ext_consistency_constraints(&challenges.program_table_challenges)
    }

    fn dynamic_transition_constraints(
        &self,
        challenges: &super::challenges::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtProgramTable::ext_transition_constraints(&challenges.program_table_challenges)
    }

    fn dynamic_terminal_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtProgramTable::ext_terminal_constraints(&challenges.program_table_challenges)
    }
}
