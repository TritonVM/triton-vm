use ndarray::s;
use ndarray::Array1;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use num_traits::One;
use num_traits::Zero;
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::challenges::ChallengeId::*;
use crate::table::challenges::Challenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::DualRowIndicator::*;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::*;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::LookupArg;
use crate::table::table_column::MasterBaseTableColumn;
use crate::table::table_column::MasterExtTableColumn;
use crate::table::table_column::ProgramBaseTableColumn;
use crate::table::table_column::ProgramBaseTableColumn::*;
use crate::table::table_column::ProgramExtTableColumn;
use crate::table::table_column::ProgramExtTableColumn::*;
use crate::vm::AlgebraicExecutionTrace;

pub const BASE_WIDTH: usize = ProgramBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = ProgramExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

#[derive(Debug, Clone)]
pub struct ProgramTable {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtProgramTable {}

impl ExtProgramTable {
    pub fn ext_initial_constraints_as_circuits() -> Vec<ConstraintCircuit<SingleRowIndicator>> {
        let circuit_builder = ConstraintCircuitBuilder::new();

        let address = circuit_builder.input(BaseRow(Address.master_base_table_index()));
        let instruction_lookup_log_derivative = circuit_builder.input(ExtRow(
            InstructionLookupServerLogDerivative.master_ext_table_index(),
        ));

        let first_address_is_zero = address;

        let instruction_lookup_log_derivative_is_initialized_correctly =
            instruction_lookup_log_derivative
                - circuit_builder.x_constant(LookupArg::default_initial());

        vec![
            first_address_is_zero.consume(),
            instruction_lookup_log_derivative_is_initialized_correctly.consume(),
        ]
    }

    pub fn ext_consistency_constraints_as_circuits() -> Vec<ConstraintCircuit<SingleRowIndicator>> {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let one = circuit_builder.b_constant(1_u32.into());

        let is_padding = circuit_builder.input(BaseRow(IsPadding.master_base_table_index()));
        let is_padding_is_bit = is_padding.clone() * (is_padding - one);

        vec![is_padding_is_bit.consume()]
    }

    pub fn ext_transition_constraints_as_circuits() -> Vec<ConstraintCircuit<DualRowIndicator>> {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let one = circuit_builder.b_constant(1u32.into());
        let address = circuit_builder.input(CurrentBaseRow(Address.master_base_table_index()));
        let instruction =
            circuit_builder.input(CurrentBaseRow(Instruction.master_base_table_index()));
        let lookup_multiplicity =
            circuit_builder.input(CurrentBaseRow(LookupMultiplicity.master_base_table_index()));
        let is_padding = circuit_builder.input(CurrentBaseRow(IsPadding.master_base_table_index()));
        let log_derivative = circuit_builder.input(CurrentExtRow(
            InstructionLookupServerLogDerivative.master_ext_table_index(),
        ));
        let address_next = circuit_builder.input(NextBaseRow(Address.master_base_table_index()));
        let instruction_next =
            circuit_builder.input(NextBaseRow(Instruction.master_base_table_index()));
        let is_padding_next =
            circuit_builder.input(NextBaseRow(IsPadding.master_base_table_index()));
        let log_derivative_next = circuit_builder.input(NextExtRow(
            InstructionLookupServerLogDerivative.master_ext_table_index(),
        ));

        let address_increases_by_one = address_next - (address.clone() + one.clone());
        let is_padding_is_0_or_remains_unchanged =
            is_padding.clone() * (is_padding_next - is_padding.clone());

        let log_derivative_remains = log_derivative_next.clone() - log_derivative.clone();
        let compressed_row = circuit_builder.challenge(ProgramAddressWeight) * address
            + circuit_builder.challenge(ProgramInstructionWeight) * instruction
            + circuit_builder.challenge(ProgramNextInstructionWeight) * instruction_next;

        let indeterminate = circuit_builder.challenge(InstructionLookupIndeterminate);

        let log_derivative_updates = (log_derivative_next - log_derivative)
            * (indeterminate - compressed_row)
            - lookup_multiplicity;
        let log_derivative_updates_if_and_only_if_not_a_padding_row = (one - is_padding.clone())
            * log_derivative_updates
            + is_padding * log_derivative_remains;

        [
            address_increases_by_one,
            is_padding_is_0_or_remains_unchanged,
            log_derivative_updates_if_and_only_if_not_a_padding_row,
        ]
        .map(|circuit| circuit.consume())
        .to_vec()
    }

    pub fn ext_terminal_constraints_as_circuits() -> Vec<ConstraintCircuit<SingleRowIndicator>> {
        // no further constraints
        vec![]
    }
}

impl ProgramTable {
    pub fn fill_trace(
        program_table: &mut ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
    ) {
        let program_len = aet.program.len_bwords();
        let address_column = program_table.slice_mut(s![..program_len, Address.base_table_index()]);
        let addresses = Array1::from_iter((0..program_len).map(|a| BFieldElement::new(a as u64)));
        addresses.move_into(address_column);

        let instructions = Array1::from(aet.program.to_bwords());
        let instruction_column =
            program_table.slice_mut(s![..program_len, Instruction.base_table_index()]);
        instructions.move_into(instruction_column);

        let multiplicities = Array1::from_iter(
            aet.instruction_multiplicities
                .iter()
                .map(|&m| BFieldElement::new(m as u64)),
        );
        let multiplicities_column =
            program_table.slice_mut(s![..program_len, LookupMultiplicity.base_table_index()]);
        multiplicities.move_into(multiplicities_column);
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
        challenges: &Challenges,
    ) {
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());

        let address_weight = challenges.get_challenge(ProgramAddressWeight);
        let instruction_weight = challenges.get_challenge(ProgramInstructionWeight);
        let next_instruction_weight = challenges.get_challenge(ProgramNextInstructionWeight);
        let instruction_lookup_indeterminate =
            challenges.get_challenge(InstructionLookupIndeterminate);

        let mut instruction_lookup_log_derivative = LookupArg::default_initial();

        for (idx, window) in base_table.windows([2, BASE_WIDTH]).into_iter().enumerate() {
            let row = window.slice(s![0, ..]);
            let next_row = window.slice(s![1, ..]);
            let mut extension_row = ext_table.slice_mut(s![idx, ..]);

            // In the Program Table, the logarithmic derivative for the instruction lookup
            // argument does record the initial in the first row, as an exception to all other
            // table-linking arguments.
            // The logarithmic derivative's final value, allowing for a meaningful cross-table
            // argument, is recorded in the first padding row. This row is enforced to exist.
            extension_row[InstructionLookupServerLogDerivative.ext_table_index()] =
                instruction_lookup_log_derivative;
            // update the logarithmic derivative if not a padding row
            if row[IsPadding.base_table_index()].is_zero() {
                let lookup_multiplicity = row[LookupMultiplicity.base_table_index()];
                let address = row[Address.base_table_index()];
                let instruction = row[Instruction.base_table_index()];
                let next_instruction = next_row[Instruction.base_table_index()];

                let compressed_row_for_instruction_lookup = address * address_weight
                    + instruction * instruction_weight
                    + next_instruction * next_instruction_weight;
                instruction_lookup_log_derivative += (instruction_lookup_indeterminate
                    - compressed_row_for_instruction_lookup)
                    .inverse()
                    * lookup_multiplicity;
            }
        }

        let mut last_row = ext_table
            .rows_mut()
            .into_iter()
            .last()
            .expect("Program Table must not be empty.");
        last_row[InstructionLookupServerLogDerivative.ext_table_index()] =
            instruction_lookup_log_derivative;
    }
}
