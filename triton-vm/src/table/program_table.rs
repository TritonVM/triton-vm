use itertools::Itertools;
use num_traits::{One, Zero};
use strum::EnumCount;
use strum_macros::{Display, EnumCount as EnumCountMacro, EnumIter};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::{Degree, MPolynomial};
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::cross_table_arguments::{CrossTableArg, EvalArg};
use crate::fri_domain::FriDomain;
use crate::table::base_table::Extendable;
use crate::table::extension_table::Evaluable;
use crate::table::table_column::ProgramBaseTableColumn::{self, *};
use crate::table::table_column::ProgramExtTableColumn::{self, *};

use super::base_table::{InheritsFromTable, Table, TableLike};
use super::challenges::{AllChallenges, TableChallenges};
use super::constraint_circuit::{ConstraintCircuit, ConstraintCircuitBuilder};
use super::extension_table::{ExtensionTable, Quotientable, QuotientableExtensionTable};

pub const PROGRAM_TABLE_NUM_PERMUTATION_ARGUMENTS: usize = 0;
pub const PROGRAM_TABLE_NUM_EVALUATION_ARGUMENTS: usize = 1;

/// This is 3 because it combines: addr, instruction, instruction in next row
pub const PROGRAM_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 3;

pub const BASE_WIDTH: usize = ProgramBaseTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + ProgramExtTableColumn::COUNT;

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
    pub(crate) inherited_table: Table<XFieldElement>,
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

        let mut padding_row = [zero; BASE_WIDTH];
        if let Some(row) = self.data().last() {
            padding_row[usize::from(Address)] = row[usize::from(Address)] + one;
        }
        padding_row[usize::from(IsPadding)] = one;

        (None, vec![padding_row.to_vec()])
    }
}

impl TableLike<XFieldElement> for ExtProgramTable {}

impl ExtProgramTable {
    fn ext_initial_constraints(
        _challenges: &ProgramTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let variables = MPolynomial::variables(FULL_WIDTH);
        let constant = |xfe| MPolynomial::from_constant(xfe, FULL_WIDTH);
        let one = constant(XFieldElement::one());

        let address = variables[usize::from(Address)].clone();
        let running_evaluation = variables[usize::from(RunningEvaluation)].clone();

        let first_address_is_zero = address;

        let running_evaluation_is_initialized_correctly = running_evaluation - one;

        vec![
            first_address_is_zero,
            running_evaluation_is_initialized_correctly,
        ]
    }

    fn ext_consistency_constraints(
        _challenges: &ProgramTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let one = MPolynomial::from_constant(XFieldElement::one(), FULL_WIDTH);

        let variables = MPolynomial::variables(FULL_WIDTH);
        let is_padding = variables[usize::from(IsPadding)].clone();

        let is_padding_is_bit = is_padding.clone() * (is_padding - one);

        vec![is_padding_is_bit]
    }

    pub fn ext_transition_constraints_as_circuits() -> Vec<ConstraintCircuit<ProgramTableChallenges>>
    {
        let circuit_builder = ConstraintCircuitBuilder::new(2 * FULL_WIDTH);
        let address = circuit_builder.input(usize::from(Address));
        let address_next = circuit_builder.input(FULL_WIDTH + usize::from(Address));
        let one = circuit_builder.b_constant(1u32.into());
        let instruction = circuit_builder.input(usize::from(Instruction));
        let is_padding = circuit_builder.input(usize::from(IsPadding));
        let running_evaluation = circuit_builder.input(usize::from(RunningEvaluation));
        let instruction_next = circuit_builder.input(FULL_WIDTH + usize::from(Instruction));
        let is_padding_next = circuit_builder.input(FULL_WIDTH + usize::from(IsPadding));
        let running_evaluation_next =
            circuit_builder.input(FULL_WIDTH + usize::from(RunningEvaluation));

        let address_increases_by_one = address_next - (address.clone() + one.clone());
        let is_padding_is_0_or_remains_unchanged =
            is_padding.clone() * (is_padding_next - is_padding.clone());

        let running_evaluation_remains =
            running_evaluation_next.clone() - running_evaluation.clone();
        let compressed_row = circuit_builder.challenge(ProgramTableChallengeId::AddressWeight)
            * address
            + circuit_builder.challenge(ProgramTableChallengeId::InstructionWeight) * instruction
            + circuit_builder.challenge(ProgramTableChallengeId::NextInstructionWeight)
                * instruction_next;

        let indeterminate =
            circuit_builder.challenge(ProgramTableChallengeId::InstructionEvalIndeterminate);
        let running_evaluation_updates =
            running_evaluation_next - (indeterminate * running_evaluation + compressed_row);
        let running_evaluation_updates_if_and_only_if_not_a_padding_row =
            (one - is_padding.clone()) * running_evaluation_updates
                + is_padding * running_evaluation_remains;

        vec![
            address_increases_by_one.consume(),
            is_padding_is_0_or_remains_unchanged.consume(),
            running_evaluation_updates_if_and_only_if_not_a_padding_row.consume(),
        ]
    }

    fn ext_transition_constraints(
        challenges: &ProgramTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let circuits = Self::ext_transition_constraints_as_circuits();
        circuits
            .into_iter()
            .map(|circ| circ.partial_evaluate(challenges))
            .collect_vec()
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

    pub fn to_fri_domain_table(
        &self,
        fri_domain: &FriDomain<BFieldElement>,
        omicron: BFieldElement,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Self {
        let base_columns = 0..self.base_width();
        let fri_domain_codewords = self.low_degree_extension(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
            base_columns,
        );
        let inherited_table = self.inherited_table.with_data(fri_domain_codewords);
        Self { inherited_table }
    }

    pub fn extend(
        &self,
        challenges: &ProgramTableChallenges,
        interpolant_degree: Degree,
    ) -> ExtProgramTable {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        let mut instruction_table_running_evaluation = EvalArg::default_initial();

        let data_with_0 = {
            let mut tmp = self.data().clone();
            tmp.push(vec![BFieldElement::zero(); BASE_WIDTH]);
            tmp
        };

        for (row, next_row) in data_with_0.into_iter().tuple_windows() {
            let mut extension_row = [0.into(); FULL_WIDTH];
            extension_row[..BASE_WIDTH]
                .copy_from_slice(&row.iter().map(|elem| elem.lift()).collect_vec());

            let address = row[usize::from(Address)].lift();
            let instruction = row[usize::from(Instruction)].lift();
            let next_instruction = next_row[usize::from(Instruction)].lift();

            // The running evaluation linking Program Table and Instruction Table does record the
            // initial in the first row, contrary to most other running evaluations and products.
            // The running product's final value, allowing for a meaningful cross-table argument,
            // is recorded in the first padding row. This row is guaranteed to exist.
            extension_row[usize::from(RunningEvaluation)] = instruction_table_running_evaluation;
            // update the running evaluation if not a padding row
            if row[usize::from(IsPadding)].is_zero() {
                // compress address, instruction, and next instruction (or argument) into one value
                let compressed_row_for_evaluation_argument = address * challenges.address_weight
                    + instruction * challenges.instruction_weight
                    + next_instruction * challenges.next_instruction_weight;

                instruction_table_running_evaluation = instruction_table_running_evaluation
                    * challenges.instruction_eval_indeterminate
                    + compressed_row_for_evaluation_argument;
            }

            extension_matrix.push(extension_row.to_vec());
        }

        assert_eq!(self.data().len(), extension_matrix.len());
        let padded_height = extension_matrix.len();
        let inherited_table = self.extension(
            extension_matrix,
            interpolant_degree,
            padded_height,
            ExtProgramTable::ext_initial_constraints(challenges),
            ExtProgramTable::ext_consistency_constraints(challenges),
            ExtProgramTable::ext_transition_constraints(challenges),
            ExtProgramTable::ext_terminal_constraints(challenges),
        );
        ExtProgramTable { inherited_table }
    }

    pub fn for_verifier(
        interpolant_degree: Degree,
        padded_height: usize,
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
            padded_height,
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
    pub fn to_fri_domain_table(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        omicron: XFieldElement,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Self {
        let ext_columns = self.base_width()..self.full_width();
        let fri_domain_codewords_ext = self.low_degree_extension(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
            ext_columns,
        );

        let inherited_table = self.inherited_table.with_data(fri_domain_codewords_ext);
        ExtProgramTable { inherited_table }
    }
}

#[derive(Debug, Copy, Clone, Display, EnumCountMacro, EnumIter, PartialEq, Eq, Hash)]
pub enum ProgramTableChallengeId {
    InstructionEvalIndeterminate,
    AddressWeight,
    InstructionWeight,
    NextInstructionWeight,
}

impl From<ProgramTableChallengeId> for usize {
    fn from(val: ProgramTableChallengeId) -> Self {
        val as usize
    }
}

impl TableChallenges for ProgramTableChallenges {
    type Id = ProgramTableChallengeId;

    #[inline]
    fn get_challenge(&self, id: Self::Id) -> XFieldElement {
        match id {
            ProgramTableChallengeId::InstructionEvalIndeterminate => {
                self.instruction_eval_indeterminate
            }
            ProgramTableChallengeId::AddressWeight => self.address_weight,
            ProgramTableChallengeId::InstructionWeight => self.instruction_weight,
            ProgramTableChallengeId::NextInstructionWeight => self.next_instruction_weight,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProgramTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the program table.
    pub instruction_eval_indeterminate: XFieldElement,

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
