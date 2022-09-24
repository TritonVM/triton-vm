use itertools::Itertools;
use num_traits::{One, Zero};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::{Degree, MPolynomial};
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::fri_domain::FriDomain;
use crate::table::base_table::Extendable;
use crate::table::extension_table::Evaluable;

use super::base_table::{InheritsFromTable, Table, TableLike};
use super::challenges::AllChallenges;
use super::extension_table::{ExtensionTable, Quotientable, QuotientableExtensionTable};
use super::table_column::ExtInstructionTableColumn::*;
use super::table_column::InstructionTableColumn::*;

pub const INSTRUCTION_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 1;
pub const INSTRUCTION_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 1;
pub const INSTRUCTION_TABLE_INITIALS_COUNT: usize =
    INSTRUCTION_TABLE_PERMUTATION_ARGUMENTS_COUNT + INSTRUCTION_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 6 because it combines: (ip, ci, nia) and (addr, instruction, next_instruction).
pub const INSTRUCTION_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 6;

pub const BASE_WIDTH: usize = 4;
pub const FULL_WIDTH: usize = 8; // BASE_WIDTH + 2 * INITIALS_COUNT

#[derive(Debug, Clone)]
pub struct InstructionTable {
    inherited_table: Table<BFieldElement>,
}

impl InheritsFromTable<BFieldElement> for InstructionTable {
    fn inherited_table(&self) -> &Table<BFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<BFieldElement> {
        &mut self.inherited_table
    }
}

#[derive(Debug, Clone)]
pub struct ExtInstructionTable {
    inherited_table: Table<XFieldElement>,
}

impl Default for ExtInstructionTable {
    fn default() -> Self {
        Self {
            inherited_table: Table::new(
                BASE_WIDTH,
                FULL_WIDTH,
                vec![],
                "EmptyExtInstructionTable".to_string(),
            ),
        }
    }
}

impl Evaluable for ExtInstructionTable {}
impl Quotientable for ExtInstructionTable {}
impl QuotientableExtensionTable for ExtInstructionTable {}

impl InheritsFromTable<XFieldElement> for ExtInstructionTable {
    fn inherited_table(&self) -> &Table<XFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<XFieldElement> {
        &mut self.inherited_table
    }
}

impl TableLike<BFieldElement> for InstructionTable {}

impl Extendable for InstructionTable {
    fn get_padding_rows(&self) -> (Option<usize>, Vec<Vec<BFieldElement>>) {
        let zero = BFieldElement::zero();
        let one = BFieldElement::one();
        if let Some(row) = self.data().last() {
            let mut padding_row = row.clone();
            // address keeps increasing
            padding_row[Address as usize] += one;
            padding_row[IsPadding as usize] = one;
            (None, vec![padding_row])
        } else {
            let mut padding_row = [zero; BASE_WIDTH];
            padding_row[IsPadding as usize] = one;
            (None, vec![padding_row.to_vec()])
        }
    }
}

impl TableLike<XFieldElement> for ExtInstructionTable {}

impl ExtInstructionTable {
    fn ext_initial_constraints() -> Vec<MPolynomial<XFieldElement>> {
        let variables: Vec<MPolynomial<XFieldElement>> =
            MPolynomial::variables(FULL_WIDTH, 1.into());
        let addr = variables[usize::from(Address)].clone();
        let compressed_row_for_eval_arg = variables[usize::from(CompressedRowEvalArg)].clone();
        let running_evaluation = variables[usize::from(RunningEvaluation)].clone();

        let first_address_is_zero = addr;

        let running_evaluation_is_initialized_correctly =
            compressed_row_for_eval_arg - running_evaluation;

        vec![
            first_address_is_zero,
            running_evaluation_is_initialized_correctly,
        ]
    }

    fn ext_consistency_constraints(
        challenges: &InstructionTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let variables = MPolynomial::variables(FULL_WIDTH, 1.into());

        let addr = variables[Address as usize].clone();
        let current_instruction = variables[CI as usize].clone();
        let next_instruction = variables[NIA as usize].clone();
        let compressed_row_for_eval_arg = variables[usize::from(CompressedRowEvalArg)].clone();
        let compressed_row_for_perm_arg = variables[usize::from(CompressedRowPermArg)].clone();

        let computed_compressed_row_for_eval_arg = addr.scalar_mul(challenges.address_weight)
            + current_instruction.scalar_mul(challenges.instruction_weight)
            + next_instruction.scalar_mul(challenges.next_instruction_weight);
        let row_compression_for_evaluation_argument_is_correct =
            computed_compressed_row_for_eval_arg - compressed_row_for_eval_arg;

        let computed_compressed_row_for_perm_arg = addr.scalar_mul(challenges.ip_processor_weight)
            + current_instruction.scalar_mul(challenges.ci_processor_weight)
            + next_instruction.scalar_mul(challenges.nia_processor_weight);
        let row_compression_for_permutation_argument_is_correct =
            computed_compressed_row_for_perm_arg - compressed_row_for_perm_arg;

        vec![
            row_compression_for_evaluation_argument_is_correct,
            row_compression_for_permutation_argument_is_correct,
        ]
    }

    fn ext_transition_constraints(
        challenges: &InstructionTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let one = MPolynomial::from_constant(1.into(), 2 * FULL_WIDTH);
        let variables = MPolynomial::variables(2 * FULL_WIDTH, 1.into());

        let addr = variables[Address as usize].clone();
        let addr_next = variables[FULL_WIDTH + Address as usize].clone();
        let current_instruction = variables[CI as usize].clone();
        let current_instruction_next = variables[FULL_WIDTH + CI as usize].clone();
        let next_instruction = variables[NIA as usize].clone();
        let next_instruction_next = variables[FULL_WIDTH + NIA as usize].clone();
        // beware: for polynomials, “0” is true
        let is_padding_row = one.clone() - variables[FULL_WIDTH + IsPadding as usize].clone();

        // Base Table Constraints
        let address_increases_by_one = addr_next.clone() - (addr.clone() + one.clone());
        let address_increases_by_one_or_ci_stays =
            address_increases_by_one.clone() * (current_instruction_next - current_instruction);
        let address_increases_by_one_or_nia_stays =
            address_increases_by_one.clone() * (next_instruction_next - next_instruction);

        // Extension Table Constraints
        let processor_perm_row_weight =
            MPolynomial::from_constant(challenges.processor_perm_row_weight, 2 * FULL_WIDTH);
        let compressed_row_for_eval_arg =
            variables[FULL_WIDTH + usize::from(CompressedRowEvalArg)].clone();
        let running_evaluation = variables[usize::from(RunningEvaluation)].clone();
        let running_evaluation_next =
            variables[FULL_WIDTH + usize::from(RunningEvaluation)].clone();
        let compressed_row_for_perm_arg =
            variables[FULL_WIDTH + usize::from(CompressedRowPermArg)].clone();
        let running_product = variables[usize::from(RunningProductPermArg)].clone();
        let running_product_next =
            variables[FULL_WIDTH + usize::from(RunningProductPermArg)].clone();

        // The running evaluation is updated if and only if
        // 1. the address changes, and
        // 2. the current row is not a padding row.
        // Stated differently:
        // 1. the address doesn't change
        //      or the current row is a padding row
        //      or the running evaluation is updated,
        // 2. the address does change
        //      or the running evaluation is not updated, and
        // 3. the current row is not a padding row
        //      or the running evaluation is not updated.
        let address_stays = addr_next - addr;
        let running_evaluations_stays =
            running_evaluation_next.clone() - running_evaluation.clone();
        let running_evaluation_update = running_evaluation_next
            - running_evaluation.scalar_mul(challenges.program_eval_row_weight)
            - compressed_row_for_eval_arg;

        let running_evaluation_is_well_formed =
            address_stays.clone() * is_padding_row.clone() * running_evaluation_update
                + address_increases_by_one.clone() * running_evaluations_stays.clone()
                + (one.clone() - is_padding_row.clone()) * running_evaluations_stays;

        // The running product is updated if and only if
        // 1. the address doesn't change, and
        // 2. the current row is not a padding row.
        // Stated differently:
        // 1. the address does change
        //      or the current row is a padding row
        //      or the running product is updated,
        // 2. the address doesn't change
        //      or the running product is not updated, and
        // 3. the current row is not a padding row
        //      or the running product is not updated.
        let running_product_stays = running_product_next.clone() - running_product.clone();
        let running_product_update = running_product_next
            - running_product * (processor_perm_row_weight - compressed_row_for_perm_arg);

        let running_product_is_well_formed =
            address_increases_by_one * is_padding_row.clone() * running_product_update
                + address_stays * running_product_stays.clone()
                + (one - is_padding_row) * running_product_stays;

        vec![
            address_increases_by_one_or_ci_stays,
            address_increases_by_one_or_nia_stays,
            running_evaluation_is_well_formed,
            running_product_is_well_formed,
        ]
    }

    fn ext_terminal_constraints(
        _challenges: &InstructionTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }
}

impl InstructionTable {
    pub fn new_prover(matrix: Vec<Vec<BFieldElement>>) -> Self {
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            matrix,
            "InstructionTable".to_string(),
        );
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
        challenges: &InstructionTableChallenges,
        interpolant_degree: Degree,
    ) -> ExtInstructionTable {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        let mut processor_table_running_product = XFieldElement::one();
        let mut program_table_running_evaluation = XFieldElement::zero();
        let mut previous_row: Option<Vec<_>> = None;

        for row in self.data().iter() {
            let mut extension_row = [0.into(); FULL_WIDTH];
            extension_row[..BASE_WIDTH]
                .copy_from_slice(&row.iter().map(|elem| elem.lift()).collect_vec());

            // Is the current row's address different from the previous row's address?
            // Different: update running evaluation of Evaluation Argument with Program Table.
            // Not different: update running product of Permutation Argument with Processor Table.
            let mut is_duplicate_row = false;
            if let Some(prow) = previous_row {
                if prow[Address as usize] == row[Address as usize] {
                    is_duplicate_row = true;
                    debug_assert_eq!(prow[CI as usize], row[CI as usize]);
                    debug_assert_eq!(prow[NIA as usize], row[NIA as usize]);
                } else {
                    debug_assert_eq!(prow[Address as usize] + 1_u64.into(), row[Address as usize]);
                }
            }

            // Compress values of current row for Permutation Argument with Processor Table
            let ip = row[Address as usize].lift();
            let ci = row[CI as usize].lift();
            let nia = row[NIA as usize].lift();
            let compressed_row_for_permutation_argument = ip * challenges.ip_processor_weight
                + ci * challenges.ci_processor_weight
                + nia * challenges.nia_processor_weight;
            extension_row[usize::from(CompressedRowPermArg)] =
                compressed_row_for_permutation_argument;

            // Update running product if same row has been seen before and not padding row
            if is_duplicate_row && row[IsPadding as usize].is_zero() {
                processor_table_running_product *=
                    challenges.processor_perm_row_weight - compressed_row_for_permutation_argument;
            }
            extension_row[usize::from(RunningProductPermArg)] = processor_table_running_product;

            // Compress values of current row for Evaluation Argument with Program Table
            let compressed_row_for_evaluation_argument = ip * challenges.address_weight
                + ci * challenges.instruction_weight
                + nia * challenges.next_instruction_weight;
            extension_row[usize::from(CompressedRowEvalArg)] =
                compressed_row_for_evaluation_argument;

            // Update running evaluation if same row has _not_ been seen before and not padding row
            if !is_duplicate_row && row[IsPadding as usize].is_zero() {
                program_table_running_evaluation = program_table_running_evaluation
                    * challenges.program_eval_row_weight
                    + compressed_row_for_evaluation_argument;
            }
            extension_row[usize::from(RunningEvaluation)] = program_table_running_evaluation;

            previous_row = Some(row.clone());
            extension_matrix.push(extension_row.to_vec());
        }

        let inherited_table = self.extension(
            extension_matrix,
            interpolant_degree,
            ExtInstructionTable::ext_initial_constraints(),
            ExtInstructionTable::ext_consistency_constraints(challenges),
            ExtInstructionTable::ext_transition_constraints(challenges),
            ExtInstructionTable::ext_terminal_constraints(challenges),
        );
        ExtInstructionTable { inherited_table }
    }

    pub fn for_verifier(
        interpolant_degree: Degree,
        all_challenges: &AllChallenges,
    ) -> ExtInstructionTable {
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            vec![],
            "ExtInstructionTable".to_string(),
        );
        let base_table = Self { inherited_table };
        let empty_matrix: Vec<Vec<XFieldElement>> = vec![];
        let extension_table = base_table.extension(
            empty_matrix,
            interpolant_degree,
            ExtInstructionTable::ext_initial_constraints(),
            ExtInstructionTable::ext_consistency_constraints(
                &all_challenges.instruction_table_challenges,
            ),
            ExtInstructionTable::ext_transition_constraints(
                &all_challenges.instruction_table_challenges,
            ),
            ExtInstructionTable::ext_terminal_constraints(
                &all_challenges.instruction_table_challenges,
            ),
        );

        ExtInstructionTable {
            inherited_table: extension_table,
        }
    }
}

impl ExtInstructionTable {
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
        ExtInstructionTable { inherited_table }
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

impl ExtensionTable for ExtInstructionTable {
    fn dynamic_initial_constraints(&self) -> Vec<MPolynomial<XFieldElement>> {
        ExtInstructionTable::ext_initial_constraints()
    }

    fn dynamic_consistency_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtInstructionTable::ext_consistency_constraints(&challenges.instruction_table_challenges)
    }

    fn dynamic_transition_constraints(
        &self,
        challenges: &super::challenges::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtInstructionTable::ext_transition_constraints(&challenges.instruction_table_challenges)
    }

    fn dynamic_terminal_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtInstructionTable::ext_terminal_constraints(&challenges.instruction_table_challenges)
    }
}
