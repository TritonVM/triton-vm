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

use crate::table::challenges::TableChallenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::DualRowIndicator::*;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::*;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::EvalArg;
use crate::table::master_table::NUM_BASE_COLUMNS;
use crate::table::master_table::NUM_EXT_COLUMNS;
use crate::table::table_column::BaseTableColumn;
use crate::table::table_column::ExtTableColumn;
use crate::table::table_column::MasterBaseTableColumn;
use crate::table::table_column::MasterExtTableColumn;
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
pub struct ProgramTable {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtProgramTable {}

impl ExtProgramTable {
    pub fn ext_initial_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            ProgramTableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let one = circuit_builder.b_constant(1_u32.into());

        let address = circuit_builder.input(BaseRow(Address.master_base_table_index()));
        let running_evaluation =
            circuit_builder.input(ExtRow(RunningEvaluation.master_ext_table_index()));

        let first_address_is_zero = address;

        let running_evaluation_is_initialized_correctly = running_evaluation - one;

        vec![
            first_address_is_zero.consume(),
            running_evaluation_is_initialized_correctly.consume(),
        ]
    }

    pub fn ext_consistency_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            ProgramTableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let one = circuit_builder.b_constant(1_u32.into());

        let is_padding = circuit_builder.input(BaseRow(IsPadding.master_base_table_index()));
        let is_padding_is_bit = is_padding.clone() * (is_padding - one);

        vec![is_padding_is_bit.consume()]
    }

    pub fn ext_transition_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            ProgramTableChallenges,
            DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let one = circuit_builder.b_constant(1u32.into());
        let address = circuit_builder.input(CurrentBaseRow(Address.master_base_table_index()));
        let instruction =
            circuit_builder.input(CurrentBaseRow(Instruction.master_base_table_index()));
        let is_padding = circuit_builder.input(CurrentBaseRow(IsPadding.master_base_table_index()));
        let running_evaluation =
            circuit_builder.input(CurrentExtRow(RunningEvaluation.master_ext_table_index()));
        let address_next = circuit_builder.input(NextBaseRow(Address.master_base_table_index()));
        let instruction_next =
            circuit_builder.input(NextBaseRow(Instruction.master_base_table_index()));
        let is_padding_next =
            circuit_builder.input(NextBaseRow(IsPadding.master_base_table_index()));
        let running_evaluation_next =
            circuit_builder.input(NextExtRow(RunningEvaluation.master_ext_table_index()));

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

    pub fn ext_terminal_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            ProgramTableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        // no further constraints
        vec![]
    }
}

impl ProgramTable {
    pub fn fill_trace(program_table: &mut ArrayViewMut2<BFieldElement>, program: &[BFieldElement]) {
        let program_len = program.len();
        let address_column = program_table.slice_mut(s![..program_len, Address.base_table_index()]);
        let addresses = Array1::from_iter((0..program_len).map(|a| BFieldElement::new(a as u64)));
        addresses.move_into(address_column);

        let instructions = Array1::from(program.to_owned());
        let instruction_column =
            program_table.slice_mut(s![..program_len, Instruction.base_table_index()]);
        instructions.move_into(instruction_column);
    }

    pub fn pad_trace(program_table: &mut ArrayViewMut2<BFieldElement>, program_len: usize) {
        let addresses = Array1::from_iter(
            (program_len..program_table.nrows()).map(|a| BFieldElement::new(a as u64)),
        );
        addresses.move_into(program_table.slice_mut(s![program_len.., Address.base_table_index()]));

        program_table
            .slice_mut(s![program_len.., IsPadding.base_table_index()])
            .fill(BFieldElement::one());
    }

    pub fn extend(
        base_table: ArrayView2<BFieldElement>,
        mut ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &ProgramTableChallenges,
    ) {
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());
        let mut instruction_table_running_evaluation = EvalArg::default_initial();

        for (idx, window) in base_table.windows([2, BASE_WIDTH]).into_iter().enumerate() {
            let row = window.slice(s![0, ..]);
            let next_row = window.slice(s![1, ..]);
            let mut extension_row = ext_table.slice_mut(s![idx, ..]);

            // The running evaluation linking Program Table and Instruction Table does record the
            // initial in the first row, contrary to most other running evaluations and products.
            // The running product's final value, allowing for a meaningful cross-table argument,
            // is recorded in the first padding row. This row is guaranteed to exist.
            extension_row[RunningEvaluation.ext_table_index()] =
                instruction_table_running_evaluation;
            // update the running evaluation if not a padding row
            if row[IsPadding.base_table_index()].is_zero() {
                let address = row[Address.base_table_index()];
                let instruction = row[Instruction.base_table_index()];
                let next_instruction = next_row[Instruction.base_table_index()];
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
        last_row[RunningEvaluation.ext_table_index()] = instruction_table_running_evaluation;
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
