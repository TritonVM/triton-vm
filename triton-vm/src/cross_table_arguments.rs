use itertools::Itertools;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::traits::PrimeField;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::fri_domain::FriDomain;
use crate::table::processor_table::PROCESSOR_TABLE_PERMUTATION_ARGUMENTS_COUNT;
use crate::table::table_collection::TableId::{
    HashTable, InstructionTable, ProcessorTable, ProgramTable,
};
use crate::table::table_collection::{ExtTableCollection, TableId};
use crate::table::table_column::{
    ExtHashTableColumn, ExtInstructionTableColumn, ExtJumpStackTableColumn, ExtOpStackTableColumn,
    ExtProcessorTableColumn, ExtProgramTableColumn, ExtRamTableColumn, ExtU32OpTableColumn,
};

pub const NUM_PRIVATE_EVAL_ARGS: usize = 3;

pub trait CrossTableArg {
    fn from(&self) -> (TableId, usize);
    fn to(&self) -> (TableId, usize);

    fn default_initial() -> XFieldElement;

    fn compute_terminal(
        symbols: &[BFieldElement],
        initial: XFieldElement,
        challenge: XFieldElement,
    ) -> XFieldElement;

    fn boundary_quotient(
        &self,
        ext_codeword_tables: &ExtTableCollection,
        fri_domain: &FriDomain<XFieldElement>,
    ) -> Vec<XFieldElement> {
        let (from_table, from_column) = self.from();
        let (to_table, to_column) = self.to();
        let lhs_codeword = &ext_codeword_tables.data(from_table)[from_column];
        let rhs_codeword = &ext_codeword_tables.data(to_table)[to_column];
        let zerofier = fri_domain
            .domain_values()
            .into_iter()
            .map(|x| x - 1.into())
            .collect();
        let inverse_zerofier = XFieldElement::batch_inversion(zerofier);

        inverse_zerofier
            .into_iter()
            .zip_eq(lhs_codeword.into_iter().zip_eq(rhs_codeword.into_iter()))
            .map(|(z, (from, to))| (*from - *to) * z)
            .collect_vec()
    }

    fn quotient_degree_bound(&self, ext_codeword_tables: &ExtTableCollection) -> Degree {
        let (from_table, _) = self.from();
        let (to_table, _) = self.to();
        let lhs_interpolant_degree = ext_codeword_tables.interpolant_degree(from_table);
        let rhs_interpolant_degree = ext_codeword_tables.interpolant_degree(to_table);
        let degree = std::cmp::max(lhs_interpolant_degree, rhs_interpolant_degree);

        degree - 1
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
    ) -> bool {
        let initial = Self::default_initial();
        Self::compute_terminal(symbols, initial, challenge) == expected_terminal
    }
}

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
        XFieldElement::ring_one()
    }

    /// Compute the product for a permutation argument using `initial` and `symbols`.
    fn compute_terminal(
        symbols: &[BFieldElement],
        initial: XFieldElement,
        challenge: XFieldElement,
    ) -> XFieldElement {
        let mut running_product = initial;
        for s in symbols.iter() {
            running_product = running_product * (challenge - s.lift());
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

    // FIXME: PROCESSOR_TABLE_PERMUTATION_ARGUMENTS_COUNT is incidentally ALL permutation arguments; create new constant?
    pub fn all_permutation_arguments() -> [Self; PROCESSOR_TABLE_PERMUTATION_ARGUMENTS_COUNT] {
        [
            Self::processor_instruction_perm_arg(),
            Self::processor_jump_stack_perm_arg(),
            Self::processor_op_stack_perm_arg(),
            Self::processor_ram_perm_arg(),
            Self::processor_u32_perm_arg(),
        ]
    }
}

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
        XFieldElement::ring_zero()
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
