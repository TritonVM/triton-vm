use itertools::Itertools;
use num_traits::{One, Zero};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::traits::{FiniteField, Inverse};
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::fri_domain::FriDomain;
use crate::table::processor_table::PROCESSOR_TABLE_PERMUTATION_ARGUMENTS_COUNT;
use crate::table::table_collection::TableId::{
    HashTable, InstructionTable, ProcessorTable, ProgramTable,
};
use crate::table::table_collection::{interpolant_degree, ExtTableCollection, TableId};
use crate::table::table_column::{
    ExtHashTableColumn, ExtInstructionTableColumn, ExtJumpStackTableColumn, ExtOpStackTableColumn,
    ExtProcessorTableColumn, ExtProgramTableColumn, ExtRamTableColumn, ExtU32OpTableColumn,
};

pub const NUM_PRIVATE_PERM_ARGS: usize = PROCESSOR_TABLE_PERMUTATION_ARGUMENTS_COUNT;
pub const NUM_PRIVATE_EVAL_ARGS: usize = 3;
pub const NUM_CROSS_TABLE_ARGS: usize = NUM_PRIVATE_PERM_ARGS + NUM_PRIVATE_EVAL_ARGS;

pub trait CrossTableArg {
    fn from(&self) -> (TableId, usize);
    fn to(&self) -> (TableId, usize);

    fn default_initial() -> XFieldElement
    where
        Self: Sized;

    fn compute_terminal(
        symbols: &[BFieldElement],
        initial: XFieldElement,
        challenge: XFieldElement,
    ) -> XFieldElement
    where
        Self: Sized;

    fn terminal_quotient(
        &self,
        ext_codeword_tables: &ExtTableCollection,
        fri_domain: &FriDomain<XFieldElement>,
        omicron: XFieldElement,
    ) -> Vec<XFieldElement> {
        let (from_table, from_column) = self.from();
        let (to_table, to_column) = self.to();
        let lhs_codeword = &ext_codeword_tables.data(from_table)[from_column];
        let rhs_codeword = &ext_codeword_tables.data(to_table)[to_column];
        let zerofier = fri_domain
            .domain_values()
            .into_iter()
            .map(|x| x - omicron.inverse())
            .collect();
        let zerofier_inverse = XFieldElement::batch_inversion(zerofier);

        zerofier_inverse
            .into_iter()
            .zip_eq(lhs_codeword.iter().zip_eq(rhs_codeword.iter()))
            .map(|(z, (&from, &to))| (from - to) * z)
            .collect_vec()
    }

    fn quotient_degree_bound(
        &self,
        ext_codeword_tables: &ExtTableCollection,
        num_trace_randomizers: usize,
    ) -> Degree {
        let interpolant_degree =
            interpolant_degree(ext_codeword_tables.padded_height, num_trace_randomizers);
        interpolant_degree - 1
    }

    fn evaluate_difference(&self, cross_table_slice: &[Vec<XFieldElement>]) -> XFieldElement {
        let (from_table, from_column) = self.from();
        let (to_table, to_column) = self.to();
        let lhs = cross_table_slice[from_table as usize][from_column];
        let rhs = cross_table_slice[to_table as usize][to_column];

        lhs - rhs
    }

    fn verify_with_public_data(
        symbols: &[BFieldElement],
        challenge: XFieldElement,
        expected_terminal: XFieldElement,
    ) -> bool
    where
        Self: Sized,
    {
        let initial = Self::default_initial();
        expected_terminal == Self::compute_terminal(symbols, initial, challenge)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct PermArg {
    from_table: TableId,
    from_column: usize,
    to_table: TableId,
    to_column: usize,
}

impl CrossTableArg for PermArg {
    fn from(&self) -> (TableId, usize) {
        (self.from_table, self.from_column)
    }

    fn to(&self) -> (TableId, usize) {
        (self.to_table, self.to_column)
    }

    fn default_initial() -> XFieldElement {
        XFieldElement::one()
    }

    /// Compute the product for a permutation argument using `initial` and `symbols`.
    fn compute_terminal(
        symbols: &[BFieldElement],
        initial: XFieldElement,
        challenge: XFieldElement,
    ) -> XFieldElement {
        let mut running_product = initial;
        for s in symbols.iter() {
            running_product *= challenge - s.lift();
        }
        running_product
    }
}

impl PermArg {
    pub fn new(
        from_table: TableId,
        from_column: usize,
        to_table: TableId,
        to_column: usize,
    ) -> Self {
        PermArg {
            from_table,
            from_column,
            to_table,
            to_column,
        }
    }
    /// A Permutation Argument between Processor Table and Instruction Table.
    pub fn processor_instruction_perm_arg() -> Self {
        Self::new(
            TableId::ProcessorTable,
            ExtProcessorTableColumn::InstructionTablePermArg.into(),
            TableId::InstructionTable,
            ExtInstructionTableColumn::RunningProductPermArg.into(),
        )
    }

    /// A Permutation Argument between Processor Table and Jump-Stack Table.
    pub fn processor_jump_stack_perm_arg() -> Self {
        Self::new(
            TableId::ProcessorTable,
            ExtProcessorTableColumn::JumpStackTablePermArg.into(),
            TableId::JumpStackTable,
            ExtJumpStackTableColumn::RunningProductPermArg.into(),
        )
    }

    /// A Permutation Argument between Processor Table and Op-Stack Table.
    pub fn processor_op_stack_perm_arg() -> Self {
        Self::new(
            TableId::ProcessorTable,
            ExtProcessorTableColumn::OpStackTablePermArg.into(),
            TableId::OpStackTable,
            ExtOpStackTableColumn::RunningProductPermArg.into(),
        )
    }

    /// A Permutation Argument between Processor Table and RAM Table.
    pub fn processor_ram_perm_arg() -> Self {
        Self::new(
            TableId::ProcessorTable,
            ExtProcessorTableColumn::RamTablePermArg.into(),
            TableId::RamTable,
            ExtRamTableColumn::RunningProductPermArg.into(),
        )
    }

    /// A Permutation Argument with the u32 Op-Table.
    pub fn processor_u32_perm_arg() -> Self {
        Self::new(
            TableId::ProcessorTable,
            ExtProcessorTableColumn::U32OpTablePermArg.into(),
            TableId::U32OpTable,
            ExtU32OpTableColumn::RunningProductPermArg.into(),
        )
    }

    pub fn all_permutation_arguments() -> [Self; NUM_PRIVATE_PERM_ARGS] {
        [
            Self::processor_instruction_perm_arg(),
            Self::processor_jump_stack_perm_arg(),
            Self::processor_op_stack_perm_arg(),
            Self::processor_ram_perm_arg(),
            Self::processor_u32_perm_arg(),
        ]
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct EvalArg {
    from_table: TableId,
    from_column: usize,
    to_table: TableId,
    to_column: usize,
}

impl CrossTableArg for EvalArg {
    fn from(&self) -> (TableId, usize) {
        (self.from_table, self.from_column)
    }

    fn to(&self) -> (TableId, usize) {
        (self.to_table, self.to_column)
    }

    fn default_initial() -> XFieldElement {
        XFieldElement::zero()
    }

    /// Compute the running sum for an evaluation argument as specified by `initial`,
    /// This amounts to evaluating polynomial `f(x) = initial·x^n + Σ_i symbols[n-i]·x^i` at position
    /// challenge, i.e., returns `f(challenge)`.
    fn compute_terminal(
        symbols: &[BFieldElement],
        initial: XFieldElement,
        challenge: XFieldElement,
    ) -> XFieldElement {
        let mut running_sum = initial;
        for s in symbols.iter() {
            running_sum = challenge * running_sum + s.lift();
        }
        running_sum
    }
}

impl EvalArg {
    /// The Evaluation Argument between the Program Table and the Instruction Table
    pub fn program_instruction_eval_arg() -> Self {
        Self {
            from_table: ProgramTable,
            from_column: ExtProgramTableColumn::EvalArgRunningSum.into(),
            to_table: InstructionTable,
            to_column: ExtInstructionTableColumn::RunningSumEvalArg.into(),
        }
    }

    pub fn processor_to_hash_eval_arg() -> Self {
        Self {
            from_table: ProcessorTable,
            from_column: ExtProcessorTableColumn::ToHashTableEvalArg.into(),
            to_table: HashTable,
            to_column: ExtHashTableColumn::FromProcessorRunningSum.into(),
        }
    }

    pub fn hash_to_processor_eval_arg() -> Self {
        Self {
            from_table: HashTable,
            from_column: ExtHashTableColumn::ToProcessorRunningSum.into(),
            to_table: ProcessorTable,
            to_column: ExtProcessorTableColumn::FromHashTableEvalArg.into(),
        }
    }

    pub fn all_private_evaluation_arguments() -> [Self; NUM_PRIVATE_EVAL_ARGS] {
        [
            Self::program_instruction_eval_arg(),
            Self::processor_to_hash_eval_arg(),
            Self::hash_to_processor_eval_arg(),
        ]
    }
}

pub struct AllCrossTableArgs {
    processor_instruction_perm_arg: PermArg,
    processor_jump_stack_perm_arg: PermArg,
    processor_op_stack_perm_arg: PermArg,
    processor_ram_perm_arg: PermArg,
    processor_u32_perm_arg: PermArg,
    program_instruction_eval_arg: EvalArg,
    processor_to_hash_eval_arg: EvalArg,
    hash_to_processor_eval_arg: EvalArg,
}

impl Default for AllCrossTableArgs {
    fn default() -> Self {
        Self {
            processor_instruction_perm_arg: PermArg::processor_instruction_perm_arg(),
            processor_jump_stack_perm_arg: PermArg::processor_jump_stack_perm_arg(),
            processor_op_stack_perm_arg: PermArg::processor_op_stack_perm_arg(),
            processor_ram_perm_arg: PermArg::processor_ram_perm_arg(),
            processor_u32_perm_arg: PermArg::processor_u32_perm_arg(),
            program_instruction_eval_arg: EvalArg::program_instruction_eval_arg(),
            processor_to_hash_eval_arg: EvalArg::processor_to_hash_eval_arg(),
            hash_to_processor_eval_arg: EvalArg::hash_to_processor_eval_arg(),
        }
    }
}

impl<'a> IntoIterator for &'a AllCrossTableArgs {
    type Item = &'a dyn CrossTableArg;

    type IntoIter = std::array::IntoIter<&'a dyn CrossTableArg, NUM_CROSS_TABLE_ARGS>;

    fn into_iter(self) -> Self::IntoIter {
        [
            &self.processor_instruction_perm_arg as &'a dyn CrossTableArg,
            &self.processor_jump_stack_perm_arg as &'a dyn CrossTableArg,
            &self.processor_op_stack_perm_arg as &'a dyn CrossTableArg,
            &self.processor_ram_perm_arg as &'a dyn CrossTableArg,
            &self.processor_u32_perm_arg as &'a dyn CrossTableArg,
            &self.program_instruction_eval_arg as &'a dyn CrossTableArg,
            &self.processor_to_hash_eval_arg as &'a dyn CrossTableArg,
            &self.hash_to_processor_eval_arg as &'a dyn CrossTableArg,
        ]
        .into_iter()
    }
}

#[cfg(test)]
mod permutation_argument_tests {
    use super::*;

    #[test]
    fn all_permutation_arguments_link_from_processor_table_test() {
        for perm_arg in PermArg::all_permutation_arguments() {
            assert_eq!(TableId::ProcessorTable, perm_arg.from_table);
        }
    }
}
