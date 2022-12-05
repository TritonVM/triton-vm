use ndarray::par_azip;
use ndarray::s;
use ndarray::Array1;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use num_traits::One;
use num_traits::Zero;
use strum::EnumCount;
use strum_macros::Display;
use strum_macros::EnumCount as EnumCountMacro;
use strum_macros::EnumIter;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;

use ProgramTableChallengeId::*;

use crate::cross_table_arguments::CrossTableArg;
use crate::cross_table_arguments::EvalArg;
use crate::table::base_table::Extendable;
use crate::table::base_table::InheritsFromTable;
use crate::table::base_table::Table;
use crate::table::base_table::TableLike;
use crate::table::challenges::TableChallenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::DualRowIndicator::*;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::Row;
use crate::table::extension_table::ExtensionTable;
use crate::table::extension_table::QuotientableExtensionTable;
use crate::table::table_column::ProgramBaseTableColumn;
use crate::table::table_column::ProgramBaseTableColumn::*;
use crate::table::table_column::ProgramExtTableColumn;
use crate::table::table_column::ProgramExtTableColumn::*;

pub const PROGRAM_TABLE_NUM_PERMUTATION_ARGUMENTS: usize = 0;
pub const PROGRAM_TABLE_NUM_EVALUATION_ARGUMENTS: usize = 1;
pub const PROGRAM_TABLE_NUM_EXTENSION_CHALLENGES: usize = ProgramTableChallengeId::COUNT;

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

    pub fn fill_trace(program_table: &mut ArrayViewMut2<BFieldElement>, program: &[BFieldElement]) {
        let program_len = program.len();
        let address_column = program_table.slice_mut(s![..program_len, usize::from(Address)]);
        let addresses = Array1::from_iter((0..program_len).map(|a| BFieldElement::new(a as u64)));
        addresses.move_into(address_column);

        let mut instruction_column =
            program_table.slice_mut(s![..program_len, usize::from(Instruction)]);
        par_azip!((ic in &mut instruction_column, &instr in program)  *ic = instr);
    }

    pub fn pad_trace(program_table: &mut ArrayViewMut2<BFieldElement>, program_len: usize) {
        let padded_height = program_table.nrows();

        let address_column = program_table.slice_mut(s![program_len.., usize::from(Address)]);
        let addresses =
            Array1::from_iter((program_len..padded_height).map(|a| BFieldElement::new(a as u64)));
        addresses.move_into(address_column);

        let mut is_padding_column =
            program_table.slice_mut(s![program_len.., usize::from(IsPadding)]);
        is_padding_column.fill(BFieldElement::one());
    }

    pub fn extend(
        base_table: &ArrayView2<BFieldElement>,
        ext_table: &mut ArrayViewMut2<XFieldElement>,
        challenges: &ProgramTableChallenges,
    ) {
        let mut instruction_table_running_evaluation = EvalArg::default_initial();

        for (idx, window) in base_table.windows([2, BASE_WIDTH]).into_iter().enumerate() {
            let row = window.slice(s![0, ..]);
            let next_row = window.slice(s![1, ..]);
            let mut extension_row = ext_table.slice_mut(s![idx, ..]);

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
        }

        let mut last_row = ext_table
            .rows_mut()
            .into_iter()
            .last()
            .expect("Program Table must not be empty.");
        last_row[usize::from(RunningEvaluation)] = instruction_table_running_evaluation;
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
