use std::cmp::max;
use std::ops::{Add, Mul};

use itertools::Itertools;
use num_traits::{One, Zero};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::traits::{FiniteField, Inverse};
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::fri_domain::FriDomain;
use crate::table::processor_table::PROCESSOR_TABLE_NUM_PERMUTATION_ARGUMENTS;
use crate::table::table_collection::TableId::{
    HashTable, InstructionTable, JumpStackTable, OpStackTable, ProcessorTable, ProgramTable,
    RamTable,
};
use crate::table::table_collection::{interpolant_degree, ExtTableCollection, TableId};
use crate::table::table_column::{
    HashExtTableColumn, InstructionExtTableColumn, JumpStackExtTableColumn, OpStackExtTableColumn,
    ProcessorExtTableColumn, ProgramExtTableColumn, RamExtTableColumn,
};

pub const NUM_PRIVATE_PERM_ARGS: usize = PROCESSOR_TABLE_NUM_PERMUTATION_ARGUMENTS;
pub const NUM_PRIVATE_EVAL_ARGS: usize = 4;
pub const NUM_CROSS_TABLE_ARGS: usize = NUM_PRIVATE_PERM_ARGS + NUM_PRIVATE_EVAL_ARGS;
pub const NUM_PUBLIC_EVAL_ARGS: usize = 2;

pub trait CrossTableArg {
    fn from(&self) -> Vec<(TableId, usize)>;
    fn to(&self) -> Vec<(TableId, usize)>;

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
        let from_codeword = self.combined_from_codeword(ext_codeword_tables);
        let to_codeword = self.combined_to_codeword(ext_codeword_tables);

        let zerofier = fri_domain
            .domain_values()
            .into_iter()
            .map(|x| x - omicron.inverse())
            .collect();
        let zerofier_inverse = XFieldElement::batch_inversion(zerofier);

        zerofier_inverse
            .into_iter()
            .zip_eq(from_codeword.iter().zip_eq(to_codeword.iter()))
            .map(|(z, (&from, &to))| (from - to) * z)
            .collect_vec()
    }

    fn combined_from_codeword(
        &self,
        ext_codeword_tables: &ExtTableCollection,
    ) -> Vec<XFieldElement> {
        let from = self.from();
        let &(first_from_table, first_from_col) = from.first().unwrap();
        let codeword_length = ext_codeword_tables.data(first_from_table)[first_from_col].len();
        from.iter()
            .map(|&(from_table, from_col)| ext_codeword_tables.data(from_table)[from_col].clone())
            .fold(
                vec![XFieldElement::one(); codeword_length],
                |accumulator, factor| pointwise_operation(accumulator, factor, XFieldElement::mul),
            )
    }

    fn combined_to_codeword(&self, ext_codeword_tables: &ExtTableCollection) -> Vec<XFieldElement> {
        let to = self.to();
        let &(first_to_table, first_to_col) = to.first().unwrap();
        let codeword_length = ext_codeword_tables.data(first_to_table)[first_to_col].len();
        to.iter()
            .map(|&(to_table, to_col)| ext_codeword_tables.data(to_table)[to_col].clone())
            .fold(
                vec![XFieldElement::one(); codeword_length],
                |accumulator, factor| pointwise_operation(accumulator, factor, XFieldElement::mul),
            )
    }

    fn quotient_degree_bound(
        &self,
        ext_codeword_tables: &ExtTableCollection,
        num_trace_randomizers: usize,
    ) -> Degree {
        let column_interpolant_degree =
            interpolant_degree(ext_codeword_tables.padded_height, num_trace_randomizers);
        let lhs_interpolant_degree = column_interpolant_degree * self.from().len() as Degree;
        let rhs_interpolant_degree = column_interpolant_degree * self.to().len() as Degree;
        let terminal_zerofier_degree = 1;
        max(lhs_interpolant_degree, rhs_interpolant_degree) - terminal_zerofier_degree
    }

    fn evaluate_difference(&self, cross_table_slice: &[Vec<XFieldElement>]) -> XFieldElement {
        let lhs = self
            .from()
            .iter()
            .map(|&(from_table, from_col)| cross_table_slice[from_table as usize][from_col])
            .fold(XFieldElement::one(), XFieldElement::mul);
        let rhs: XFieldElement = self
            .to()
            .iter()
            .map(|&(to_table, to_col)| cross_table_slice[to_table as usize][to_col])
            .fold(XFieldElement::one(), XFieldElement::mul);

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

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PermArg {
    from: Vec<(TableId, usize)>,
    to: Vec<(TableId, usize)>,
}

impl CrossTableArg for PermArg {
    fn from(&self) -> Vec<(TableId, usize)> {
        self.from.clone()
    }

    fn to(&self) -> Vec<(TableId, usize)> {
        self.to.clone()
    }

    fn default_initial() -> XFieldElement {
        XFieldElement::one()
    }

    /// Compute the product for a permutation argument using `initial`, `challenge`, and `symbols`.
    fn compute_terminal(
        symbols: &[BFieldElement],
        initial: XFieldElement,
        challenge: XFieldElement,
    ) -> XFieldElement {
        symbols
            .iter()
            .map(|&symbol| challenge - symbol.lift())
            .fold(initial, XFieldElement::mul)
    }
}

impl PermArg {
    /// A Permutation Argument between Processor Table and Instruction Table.
    pub fn processor_instruction_perm_arg() -> Self {
        let from = vec![(
            ProcessorTable,
            usize::from(ProcessorExtTableColumn::InstructionTablePermArg),
        )];
        let to = vec![(
            InstructionTable,
            usize::from(InstructionExtTableColumn::RunningProductPermArg),
        )];
        Self { from, to }
    }

    /// A Permutation Argument between Processor Table and Jump-Stack Table.
    pub fn processor_jump_stack_perm_arg() -> Self {
        let from = vec![(
            ProcessorTable,
            usize::from(ProcessorExtTableColumn::JumpStackTablePermArg),
        )];
        let to = vec![(
            JumpStackTable,
            usize::from(JumpStackExtTableColumn::RunningProductPermArg),
        )];
        Self { from, to }
    }

    /// A Permutation Argument between Processor Table and Op-Stack Table.
    pub fn processor_op_stack_perm_arg() -> Self {
        let from = vec![(
            ProcessorTable,
            usize::from(ProcessorExtTableColumn::OpStackTablePermArg),
        )];
        let to = vec![(
            OpStackTable,
            usize::from(OpStackExtTableColumn::RunningProductPermArg),
        )];
        Self { from, to }
    }

    /// A Permutation Argument between Processor Table and RAM Table.
    pub fn processor_ram_perm_arg() -> Self {
        let from = vec![(
            ProcessorTable,
            usize::from(ProcessorExtTableColumn::RamTablePermArg),
        )];
        let to = vec![(
            RamTable,
            usize::from(RamExtTableColumn::RunningProductPermArg),
        )];
        Self { from, to }
    }

    pub fn all_permutation_arguments() -> [Self; NUM_PRIVATE_PERM_ARGS] {
        [
            Self::processor_instruction_perm_arg(),
            Self::processor_jump_stack_perm_arg(),
            Self::processor_op_stack_perm_arg(),
            Self::processor_ram_perm_arg(),
        ]
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct EvalArg {
    from: Vec<(TableId, usize)>,
    to: Vec<(TableId, usize)>,
}

impl CrossTableArg for EvalArg {
    fn from(&self) -> Vec<(TableId, usize)> {
        self.from.clone()
    }

    fn to(&self) -> Vec<(TableId, usize)> {
        self.to.clone()
    }

    fn default_initial() -> XFieldElement {
        XFieldElement::one()
    }

    /// Compute the running evaluation for an evaluation argument as specified by `initial`,
    /// `challenge`, and `symbols`. This amounts to evaluating polynomial
    /// `f(x) = initial·x^n + Σ_i symbols[n-i]·x^i`
    /// at position challenge, i.e., returns `f(challenge)`.
    fn compute_terminal(
        symbols: &[BFieldElement],
        initial: XFieldElement,
        challenge: XFieldElement,
    ) -> XFieldElement {
        symbols.iter().fold(initial, |running_evaluation, symbol| {
            challenge * running_evaluation + symbol.lift()
        })
    }
}

impl EvalArg {
    /// The Evaluation Argument between the Program Table and the Instruction Table
    pub fn program_instruction_eval_arg() -> Self {
        let from = vec![(
            ProgramTable,
            usize::from(ProgramExtTableColumn::RunningEvaluation),
        )];
        let to = vec![(
            InstructionTable,
            usize::from(InstructionExtTableColumn::RunningEvaluation),
        )];
        Self { from, to }
    }

    pub fn processor_to_hash_eval_arg() -> Self {
        let from = vec![(
            ProcessorTable,
            usize::from(ProcessorExtTableColumn::ToHashTableEvalArg),
        )];
        let to = vec![(
            HashTable,
            usize::from(HashExtTableColumn::FromProcessorRunningEvaluation),
        )];
        Self { from, to }
    }

    pub fn hash_to_processor_eval_arg() -> Self {
        let from = vec![(
            HashTable,
            usize::from(HashExtTableColumn::ToProcessorRunningEvaluation),
        )];
        let to = vec![(
            ProcessorTable,
            usize::from(ProcessorExtTableColumn::FromHashTableEvalArg),
        )];
        Self { from, to }
    }

    pub fn processor_to_processor_clock_jump_diff_arg() -> Self {
        let from = vec![(
            ProcessorTable,
            usize::from(ProcessorExtTableColumn::SelectedClockCyclesEvalArg),
        )];
        let to = vec![(
            ProcessorTable,
            usize::from(ProcessorExtTableColumn::UniqueClockJumpDifferencesEvalArg),
        )];
        Self { from, to }
    }

    pub fn all_private_evaluation_arguments() -> [Self; NUM_PRIVATE_EVAL_ARGS] {
        [
            Self::program_instruction_eval_arg(),
            Self::processor_to_hash_eval_arg(),
            Self::hash_to_processor_eval_arg(),
            Self::processor_to_processor_clock_jump_diff_arg(),
        ]
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct GrandCrossTableArg {
    program_to_instruction: EvalArg,
    processor_to_instruction: PermArg,
    processor_to_op_stack: PermArg,
    processor_to_ram: PermArg,
    processor_to_jump_stack: PermArg,
    processor_to_hash: EvalArg,
    hash_to_processor: EvalArg,

    program_to_instruction_weight: XFieldElement,
    processor_to_instruction_weight: XFieldElement,
    processor_to_op_stack_weight: XFieldElement,
    processor_to_ram_weight: XFieldElement,
    processor_to_jump_stack_weight: XFieldElement,
    processor_to_hash_weight: XFieldElement,
    hash_to_processor_weight: XFieldElement,

    unique_clock_jumps: EvalArg,
    unique_clock_jumps_weight: XFieldElement,

    input_terminal: XFieldElement,
    input_to_processor: (TableId, usize),
    input_to_processor_weight: XFieldElement,

    output_terminal: XFieldElement,
    processor_to_output: (TableId, usize),
    processor_to_output_weight: XFieldElement,
}

impl<'a> IntoIterator for &'a GrandCrossTableArg {
    type Item = (&'a dyn CrossTableArg, XFieldElement);

    type IntoIter =
        std::array::IntoIter<(&'a dyn CrossTableArg, XFieldElement), NUM_CROSS_TABLE_ARGS>;

    fn into_iter(self) -> Self::IntoIter {
        [
            (
                &self.program_to_instruction as &'a dyn CrossTableArg,
                self.program_to_instruction_weight,
            ),
            (
                &self.processor_to_instruction as &'a dyn CrossTableArg,
                self.processor_to_instruction_weight,
            ),
            (
                &self.processor_to_op_stack as &'a dyn CrossTableArg,
                self.processor_to_op_stack_weight,
            ),
            (
                &self.processor_to_ram as &'a dyn CrossTableArg,
                self.processor_to_ram_weight,
            ),
            (
                &self.processor_to_jump_stack as &'a dyn CrossTableArg,
                self.processor_to_jump_stack_weight,
            ),
            (
                &self.processor_to_hash as &'a dyn CrossTableArg,
                self.processor_to_hash_weight,
            ),
            (
                &self.hash_to_processor as &'a dyn CrossTableArg,
                self.hash_to_processor_weight,
            ),
            (
                &self.unique_clock_jumps as &'a dyn CrossTableArg,
                self.unique_clock_jumps_weight,
            ),
        ]
        .into_iter()
    }
}

impl GrandCrossTableArg {
    pub fn new(
        weights: &[XFieldElement; NUM_CROSS_TABLE_ARGS + NUM_PUBLIC_EVAL_ARGS],
        input_terminal: XFieldElement,
        output_terminal: XFieldElement,
    ) -> Self {
        let mut weights_stack = weights.to_vec();
        Self {
            program_to_instruction: EvalArg::program_instruction_eval_arg(),
            processor_to_instruction: PermArg::processor_instruction_perm_arg(),
            processor_to_op_stack: PermArg::processor_op_stack_perm_arg(),
            processor_to_ram: PermArg::processor_ram_perm_arg(),
            processor_to_jump_stack: PermArg::processor_jump_stack_perm_arg(),
            processor_to_hash: EvalArg::processor_to_hash_eval_arg(),
            hash_to_processor: EvalArg::hash_to_processor_eval_arg(),

            program_to_instruction_weight: weights_stack.pop().unwrap(),
            processor_to_instruction_weight: weights_stack.pop().unwrap(),
            processor_to_op_stack_weight: weights_stack.pop().unwrap(),
            processor_to_ram_weight: weights_stack.pop().unwrap(),
            processor_to_jump_stack_weight: weights_stack.pop().unwrap(),
            processor_to_hash_weight: weights_stack.pop().unwrap(),
            hash_to_processor_weight: weights_stack.pop().unwrap(),

            unique_clock_jumps: EvalArg::processor_to_processor_clock_jump_diff_arg(),
            unique_clock_jumps_weight: weights_stack.pop().unwrap(),

            input_terminal,
            input_to_processor: (
                ProcessorTable,
                usize::from(ProcessorExtTableColumn::InputTableEvalArg),
            ),
            input_to_processor_weight: weights_stack.pop().unwrap(),

            output_terminal,
            processor_to_output: (
                ProcessorTable,
                usize::from(ProcessorExtTableColumn::OutputTableEvalArg),
            ),
            processor_to_output_weight: weights_stack.pop().unwrap(),
        }
    }

    pub fn terminal_quotient_codeword(
        &self,
        ext_codeword_tables: &ExtTableCollection,
        fri_domain: &FriDomain<XFieldElement>,
        omicron: XFieldElement,
    ) -> Vec<XFieldElement> {
        let mut non_linear_sum_codeword = vec![XFieldElement::zero(); fri_domain.length];

        // cross-table arguments
        for (arg, weight) in self.into_iter() {
            let from_codeword = arg.combined_from_codeword(ext_codeword_tables);
            let to_codeword = arg.combined_to_codeword(ext_codeword_tables);
            let non_linear_summand =
                weighted_difference_codeword(&from_codeword, &to_codeword, weight);
            non_linear_sum_codeword = pointwise_operation(
                non_linear_sum_codeword,
                non_linear_summand,
                XFieldElement::add,
            );
        }

        // input
        let input_terminal_codeword = vec![self.input_terminal; fri_domain.length];
        let (to_table, to_column) = self.input_to_processor;
        let to_codeword = &ext_codeword_tables.data(to_table)[to_column];
        let weight = self.input_to_processor_weight;
        let non_linear_summand =
            weighted_difference_codeword(&input_terminal_codeword, to_codeword, weight);
        non_linear_sum_codeword = pointwise_operation(
            non_linear_sum_codeword,
            non_linear_summand,
            XFieldElement::add,
        );

        // output
        let (from_table, from_column) = self.processor_to_output;
        let from_codeword = &ext_codeword_tables.data(from_table)[from_column];
        let output_terminal_codeword = vec![self.output_terminal; fri_domain.length];
        let weight = self.processor_to_output_weight;
        let non_linear_summand =
            weighted_difference_codeword(from_codeword, &output_terminal_codeword, weight);
        non_linear_sum_codeword = pointwise_operation(
            non_linear_sum_codeword,
            non_linear_summand,
            XFieldElement::add,
        );

        let zerofier = fri_domain
            .domain_values()
            .into_iter()
            .map(|x| x - omicron.inverse())
            .collect();
        let zerofier_inverse = XFieldElement::batch_inversion(zerofier);

        zerofier_inverse
            .into_iter()
            .zip_eq(non_linear_sum_codeword.into_iter())
            .map(|(z, nls)| nls * z)
            .collect_vec()
    }

    pub fn quotient_degree_bound(
        &self,
        ext_codeword_tables: &ExtTableCollection,
        num_trace_randomizers: usize,
    ) -> Degree {
        self.into_iter()
            .map(|(arg, _)| arg.quotient_degree_bound(ext_codeword_tables, num_trace_randomizers))
            .max()
            .unwrap_or(0)
    }

    pub fn evaluate_non_linear_sum_of_differences(
        &self,
        cross_table_slice: &[Vec<XFieldElement>],
    ) -> XFieldElement {
        // cross-table arguments
        let mut non_linear_sum = self
            .into_iter()
            .map(|(arg, weight)| weight * arg.evaluate_difference(cross_table_slice))
            .sum();

        // input
        let (to_table, to_column) = self.input_to_processor;
        let processor_in = cross_table_slice[to_table as usize][to_column];
        non_linear_sum += self.input_to_processor_weight * (self.input_terminal - processor_in);

        // output
        let (from_table, from_colum) = self.processor_to_output;
        let processor_out = cross_table_slice[from_table as usize][from_colum];
        non_linear_sum += self.processor_to_output_weight * (processor_out - self.output_terminal);

        non_linear_sum
    }
}

fn pointwise_operation<F>(
    left: Vec<XFieldElement>,
    right: Vec<XFieldElement>,
    operation: F,
) -> Vec<XFieldElement>
where
    F: Fn(XFieldElement, XFieldElement) -> XFieldElement,
{
    left.into_iter()
        .zip_eq(right.into_iter())
        .map(|(l, r)| operation(l, r))
        .collect_vec()
}

fn weighted_difference_codeword(
    from_codeword: &[XFieldElement],
    to_codeword: &[XFieldElement],
    weight: XFieldElement,
) -> Vec<XFieldElement> {
    from_codeword
        .iter()
        .zip_eq(to_codeword.iter())
        .map(|(&from, &to)| weight * (from - to))
        .collect_vec()
}

#[cfg(test)]
mod permutation_argument_tests {
    use crate::stark::triton_stark_tests::parse_simulate_pad_extend;
    use crate::vm::triton_vm_tests::test_hash_nop_nop_lt;

    use super::*;

    #[test]
    fn all_permutation_arguments_link_from_processor_table_test() {
        for perm_arg in PermArg::all_permutation_arguments() {
            assert!(perm_arg
                .from()
                .iter()
                .map(|(table, _)| table)
                .contains(&TableId::ProcessorTable));
        }
    }

    #[test]
    fn almost_all_quotient_degree_bounds_of_grand_cross_table_argument_are_equal_test() {
        let num_trace_randomizers = 10;
        let code_with_input = test_hash_nop_nop_lt();
        let code = code_with_input.source_code;
        let input = code_with_input.input;
        let secret_input = code_with_input.secret_input;
        let (output, _, _, ext_codeword_tables, all_challenges, _) =
            parse_simulate_pad_extend(&code, &input, &secret_input);

        let input_terminal = EvalArg::compute_terminal(
            &input,
            EvalArg::default_initial(),
            all_challenges
                .processor_table_challenges
                .input_table_eval_row_weight,
        );

        let output_terminal = EvalArg::compute_terminal(
            &output.to_bword_vec(),
            EvalArg::default_initial(),
            all_challenges
                .processor_table_challenges
                .output_table_eval_row_weight,
        );

        let gxta = GrandCrossTableArg::new(
            &[XFieldElement::one(); NUM_CROSS_TABLE_ARGS + 2],
            input_terminal,
            output_terminal,
        );
        let quotient_degree_bound = gxta
            .program_to_instruction
            .quotient_degree_bound(&ext_codeword_tables, num_trace_randomizers);
        for (arg, _) in gxta.into_iter().take(8) {
            assert_eq!(
                quotient_degree_bound,
                arg.quotient_degree_bound(&ext_codeword_tables, num_trace_randomizers)
            );
        }
    }
}
