use itertools::Itertools;
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis};
use num_traits::{One, Zero};
use strum::EnumCount;
use strum_macros::{Display, EnumCount as EnumCountMacro, EnumIter};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;

use ProgramTableChallengeId::*;

use crate::cross_table_arguments::{CrossTableArg, EvalArg};
use crate::table::base_table::Extendable;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::Row;
use crate::table::table_column::ProgramBaseTableColumn::{self, *};
use crate::table::table_column::ProgramExtTableColumn::{self, *};

use super::base_table::{InheritsFromTable, Table, TableLike};
use super::challenges::TableChallenges;
use super::constraint_circuit::DualRowIndicator::*;
use super::constraint_circuit::{ConstraintCircuit, ConstraintCircuitBuilder, DualRowIndicator};
use super::extension_table::{ExtensionTable, QuotientableExtensionTable};

pub const PROGRAM_TABLE_NUM_PERMUTATION_ARGUMENTS: usize = 0;
pub const PROGRAM_TABLE_NUM_EVALUATION_ARGUMENTS: usize = 1;

/// This is 3 because it combines: addr, instruction, instruction in next row
pub const PROGRAM_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 3;

pub const BASE_WIDTH: usize = ProgramBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = ProgramExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

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
    pub fn ext_initial_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<ProgramTableChallenges, SingleRowIndicator<FULL_WIDTH>>> {
        let circuit_builder = ConstraintCircuitBuilder::new(FULL_WIDTH);
        let one = circuit_builder.b_constant(1_u32.into());

        let address = circuit_builder.input(Row(Address.into()));
        let running_evaluation = circuit_builder.input(Row(RunningEvaluation.into()));

        let first_address_is_zero = address;

        let running_evaluation_is_initialized_correctly = running_evaluation - one;

        vec![
            first_address_is_zero.consume(),
            running_evaluation_is_initialized_correctly.consume(),
        ]
    }

    pub fn ext_consistency_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<ProgramTableChallenges, SingleRowIndicator<FULL_WIDTH>>> {
        let circuit_builder = ConstraintCircuitBuilder::new(FULL_WIDTH);
        let one = circuit_builder.b_constant(1_u32.into());

        let is_padding = circuit_builder.input(Row(IsPadding.into()));
        let is_padding_is_bit = is_padding.clone() * (is_padding - one);

        vec![is_padding_is_bit.consume()]
    }

    pub fn ext_transition_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<ProgramTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
        let circuit_builder = ConstraintCircuitBuilder::new(2 * FULL_WIDTH);
        let address = circuit_builder.input(CurrentRow(Address.into()));
        let address_next = circuit_builder.input(NextRow(Address.into()));
        let one = circuit_builder.b_constant(1u32.into());
        let instruction = circuit_builder.input(CurrentRow(Instruction.into()));
        let is_padding = circuit_builder.input(CurrentRow(IsPadding.into()));
        let running_evaluation = circuit_builder.input(CurrentRow(RunningEvaluation.into()));
        let instruction_next = circuit_builder.input(NextRow(Instruction.into()));
        let is_padding_next = circuit_builder.input(NextRow(IsPadding.into()));
        let running_evaluation_next = circuit_builder.input(NextRow(RunningEvaluation.into()));

        let address_increases_by_one = address_next - (address.clone() + one.clone());
        let is_padding_is_0_or_remains_unchanged =
            is_padding.clone() * (is_padding_next - is_padding.clone());

        let running_evaluation_remains =
            running_evaluation_next.clone() - running_evaluation.clone();
        let compressed_row = circuit_builder.challenge(AddressWeight) * address
            + circuit_builder.challenge(InstructionWeight) * instruction
            + circuit_builder.challenge(NextInstructionWeight) * instruction_next;

        let indeterminate = circuit_builder.challenge(InstructionEvalIndeterminate);
        let running_evaluation_updates =
            running_evaluation_next - (indeterminate * running_evaluation + compressed_row);
        let running_evaluation_updates_if_and_only_if_not_a_padding_row =
            (one - is_padding.clone()) * running_evaluation_updates
                + is_padding * running_evaluation_remains;

        [
            address_increases_by_one,
            is_padding_is_0_or_remains_unchanged,
            running_evaluation_updates_if_and_only_if_not_a_padding_row,
        ]
        .map(|circuit| circuit.consume())
        .to_vec()
    }

    pub fn ext_terminal_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<ProgramTableChallenges, SingleRowIndicator<FULL_WIDTH>>> {
        // no further constraints
        vec![]
    }
}

impl ProgramTable {
    pub fn new(inherited_table: Table<BFieldElement>) -> Self {
        Self { inherited_table }
    }

    pub fn new_prover(matrix: Vec<Vec<BFieldElement>>) -> Self {
        let inherited_table =
            Table::new(BASE_WIDTH, FULL_WIDTH, matrix, "ProgramTable".to_string());
        Self { inherited_table }
    }

    /// todo rename to “extend()” once the old “extend()” is removed
    pub fn the_new_extend_method_is_in_place(
        _base_program_table: &ArrayView2<BFieldElement>,
        _ext_program_table: &mut ArrayViewMut2<XFieldElement>,
        _challenges: &ProgramTableChallenges,
    ) {
        let base = Array2::<BFieldElement>::zeros([64, 3]);
        let /*mut*/ ext = Array2::<XFieldElement>::zeros([64, 2]);

        let _b_w = base.axis_windows(Axis(0), 2);
        let /*mut*/ _e_w = ext.axis_windows(Axis(0), 2);

        // todo traverse base windows & ext windows in lockstep
    }

    pub fn extend(&self, challenges: &ProgramTableChallenges) -> ExtProgramTable {
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
        let inherited_table = self.new_from_lifted_matrix(extension_matrix);
        ExtProgramTable { inherited_table }
    }

    pub fn for_verifier() -> ExtProgramTable {
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            vec![],
            "ExtProgramTable".to_string(),
        );
        let base_table = Self { inherited_table };
        let empty_matrix: Vec<Vec<XFieldElement>> = vec![];
        let extension_table = base_table.new_from_lifted_matrix(empty_matrix);

        ExtProgramTable {
            inherited_table: extension_table,
        }
    }
}

impl ExtProgramTable {
    pub fn new(inherited_table: Table<XFieldElement>) -> Self {
        Self { inherited_table }
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
            InstructionEvalIndeterminate => self.instruction_eval_indeterminate,
            AddressWeight => self.address_weight,
            InstructionWeight => self.instruction_weight,
            NextInstructionWeight => self.next_instruction_weight,
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

impl ExtensionTable for ExtProgramTable {}
