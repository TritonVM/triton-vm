use std::cmp::max;
use std::ops::Mul;

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::Zip;
use num_traits::One;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::arithmetic_domain::ArithmeticDomain;
use crate::table::processor_table::PROCESSOR_TABLE_NUM_PERMUTATION_ARGUMENTS;
use crate::table::table_collection::interpolant_degree;
use crate::table::table_collection::terminal_quotient_zerofier_inverse;
use crate::table::table_column::HashExtTableColumn;
use crate::table::table_column::InstructionExtTableColumn;
use crate::table::table_column::JumpStackExtTableColumn;
use crate::table::table_column::MasterExtTableColumn;
use crate::table::table_column::OpStackExtTableColumn;
use crate::table::table_column::ProcessorExtTableColumn;
use crate::table::table_column::ProgramExtTableColumn;
use crate::table::table_column::RamExtTableColumn;

pub const NUM_PRIVATE_PERM_ARGS: usize = PROCESSOR_TABLE_NUM_PERMUTATION_ARGUMENTS;
pub const NUM_PRIVATE_EVAL_ARGS: usize = 3;
pub const NUM_CROSS_TABLE_ARGS: usize = NUM_PRIVATE_PERM_ARGS + NUM_PRIVATE_EVAL_ARGS;
pub const NUM_PUBLIC_EVAL_ARGS: usize = 2;

pub trait CrossTableArg {
    fn from(&self) -> Vec<usize>;
    fn to(&self) -> Vec<usize>;

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
        master_ext_table: ArrayView2<XFieldElement>,
        quotient_domain: ArithmeticDomain,
        trace_domain_generator: BFieldElement,
    ) -> Array1<XFieldElement> {
        let from_codeword = self.combined_from_codeword(master_ext_table);
        let to_codeword = self.combined_to_codeword(master_ext_table);
        let zerofier_inverse =
            terminal_quotient_zerofier_inverse(quotient_domain, trace_domain_generator);

        (from_codeword - to_codeword) * zerofier_inverse
    }

    /// Hadamard (element-wise) product of the `from` codewords.
    fn combined_from_codeword(
        &self,
        master_ext_table: ArrayView2<XFieldElement>,
    ) -> Array1<XFieldElement> {
        self.from()
            .iter()
            .map(|&from_col| master_ext_table.column(from_col))
            .fold(
                Array1::ones(master_ext_table.nrows()),
                |accumulator, factor| accumulator * factor,
            )
    }

    /// Hadamard (element-wise) product of the `to` codewords.
    fn combined_to_codeword(
        &self,
        master_ext_table: ArrayView2<XFieldElement>,
    ) -> Array1<XFieldElement> {
        self.to()
            .iter()
            .map(|&to_col| master_ext_table.column(to_col))
            .fold(
                Array1::ones(master_ext_table.nrows()),
                |accumulator, factor| accumulator * factor,
            )
    }

    /// The highest possible degree of the quotient for this cross-table argument.
    fn quotient_degree_bound(&self, padded_height: usize, num_trace_randomizers: usize) -> Degree {
        let column_interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);
        let lhs_interpolant_degree = column_interpolant_degree * self.from().len() as Degree;
        let rhs_interpolant_degree = column_interpolant_degree * self.to().len() as Degree;
        let terminal_zerofier_degree = 1;
        max(lhs_interpolant_degree, rhs_interpolant_degree) - terminal_zerofier_degree
    }

    fn evaluate_difference(
        &self,
        master_ext_table_row: ArrayView1<XFieldElement>,
    ) -> XFieldElement {
        let lhs = self
            .from()
            .iter()
            .map(|&from_col| master_ext_table_row[from_col])
            .fold(XFieldElement::one(), XFieldElement::mul);
        let rhs = self
            .to()
            .iter()
            .map(|&to_col| master_ext_table_row[to_col])
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
    from: Vec<usize>,
    to: Vec<usize>,
}

impl CrossTableArg for PermArg {
    fn from(&self) -> Vec<usize> {
        self.from.clone()
    }

    fn to(&self) -> Vec<usize> {
        self.to.clone()
    }

    fn default_initial() -> XFieldElement {
        XFieldElement::one()
    }

    /// Compute the product for a permutation argument as specified by `initial`, `challenge`,
    /// and `symbols`. This amounts to evaluating polynomial
    ///  `f(x) = initial · Π_i (x - symbols[i])`
    /// at point `challenge`, i.e., returns `f(challenge)`.
    fn compute_terminal(
        symbols: &[BFieldElement],
        initial: XFieldElement,
        challenge: XFieldElement,
    ) -> XFieldElement {
        symbols
            .iter()
            .map(|&symbol| challenge - symbol)
            .fold(initial, XFieldElement::mul)
    }
}

impl PermArg {
    pub fn processor_instruction_perm_arg() -> Self {
        Self {
            from: vec![ProcessorExtTableColumn::InstructionTablePermArg.master_ext_table_index()],
            to: vec![InstructionExtTableColumn::RunningProductPermArg.master_ext_table_index()],
        }
    }

    pub fn processor_jump_stack_perm_arg() -> Self {
        Self {
            from: vec![ProcessorExtTableColumn::JumpStackTablePermArg.master_ext_table_index()],
            to: vec![JumpStackExtTableColumn::RunningProductPermArg.master_ext_table_index()],
        }
    }

    pub fn processor_op_stack_perm_arg() -> Self {
        Self {
            from: vec![ProcessorExtTableColumn::OpStackTablePermArg.master_ext_table_index()],
            to: vec![OpStackExtTableColumn::RunningProductPermArg.master_ext_table_index()],
        }
    }

    pub fn processor_ram_perm_arg() -> Self {
        Self {
            from: vec![ProcessorExtTableColumn::RamTablePermArg.master_ext_table_index()],
            to: vec![RamExtTableColumn::RunningProductPermArg.master_ext_table_index()],
        }
    }

    pub fn clock_jump_difference_multi_table_perm_arg() -> Self {
        Self {
            from: vec![
                ProcessorExtTableColumn::AllClockJumpDifferencesPermArg.master_ext_table_index()
            ],
            to: vec![
                OpStackExtTableColumn::AllClockJumpDifferencesPermArg.master_ext_table_index(),
                RamExtTableColumn::AllClockJumpDifferencesPermArg.master_ext_table_index(),
                JumpStackExtTableColumn::AllClockJumpDifferencesPermArg.master_ext_table_index(),
            ],
        }
    }

    pub fn all_permutation_arguments() -> [Self; NUM_PRIVATE_PERM_ARGS] {
        [
            Self::processor_instruction_perm_arg(),
            Self::processor_jump_stack_perm_arg(),
            Self::processor_op_stack_perm_arg(),
            Self::processor_ram_perm_arg(),
            Self::clock_jump_difference_multi_table_perm_arg(),
        ]
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct EvalArg {
    from: Vec<usize>,
    to: Vec<usize>,
}

impl CrossTableArg for EvalArg {
    fn from(&self) -> Vec<usize> {
        self.from.clone()
    }

    fn to(&self) -> Vec<usize> {
        self.to.clone()
    }

    fn default_initial() -> XFieldElement {
        XFieldElement::one()
    }

    /// Compute the evaluation for an evaluation argument as specified by `initial`, `challenge`,
    /// and `symbols`. This amounts to evaluating polynomial
    /// `f(x) = initial·x^n + Σ_i symbols[n-i]·x^i`
    /// at point `challenge`, i.e., returns `f(challenge)`.
    fn compute_terminal(
        symbols: &[BFieldElement],
        initial: XFieldElement,
        challenge: XFieldElement,
    ) -> XFieldElement {
        symbols.iter().fold(initial, |running_evaluation, &symbol| {
            challenge * running_evaluation + symbol
        })
    }
}

impl EvalArg {
    pub fn program_instruction_eval_arg() -> Self {
        Self {
            from: vec![ProgramExtTableColumn::RunningEvaluation.master_ext_table_index()],
            to: vec![InstructionExtTableColumn::RunningEvaluation.master_ext_table_index()],
        }
    }

    pub fn processor_to_hash_eval_arg() -> Self {
        Self {
            from: vec![ProcessorExtTableColumn::ToHashTableEvalArg.master_ext_table_index()],
            to: vec![HashExtTableColumn::FromProcessorRunningEvaluation.master_ext_table_index()],
        }
    }

    pub fn hash_to_processor_eval_arg() -> Self {
        Self {
            from: vec![HashExtTableColumn::ToProcessorRunningEvaluation.master_ext_table_index()],
            to: vec![ProcessorExtTableColumn::FromHashTableEvalArg.master_ext_table_index()],
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

    all_clock_jump_differences: PermArg,
    all_clock_jump_differences_weight: XFieldElement,

    input_terminal: XFieldElement,
    input_to_processor: usize,
    input_to_processor_weight: XFieldElement,

    output_terminal: XFieldElement,
    processor_to_output: usize,
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
                &self.all_clock_jump_differences as &'a dyn CrossTableArg,
                self.all_clock_jump_differences_weight,
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

            all_clock_jump_differences: PermArg::clock_jump_difference_multi_table_perm_arg(),
            all_clock_jump_differences_weight: weights_stack.pop().unwrap(),

            input_terminal,
            input_to_processor: ProcessorExtTableColumn::InputTableEvalArg.master_ext_table_index(),
            input_to_processor_weight: weights_stack.pop().unwrap(),

            output_terminal,
            processor_to_output: ProcessorExtTableColumn::OutputTableEvalArg
                .master_ext_table_index(),
            processor_to_output_weight: weights_stack.pop().unwrap(),
        }
    }

    pub fn terminal_quotient_codeword(
        &self,
        master_ext_table: ArrayView2<XFieldElement>,
        quotient_domain: ArithmeticDomain,
        trace_domain_generator: BFieldElement,
    ) -> Array1<XFieldElement> {
        let mut non_linear_sum_codeword = Array1::zeros(quotient_domain.length);

        // cross-table arguments
        for (arg, weight) in self.into_iter() {
            let from_codeword = arg.combined_from_codeword(master_ext_table);
            let to_codeword = arg.combined_to_codeword(master_ext_table);
            Zip::from(&mut non_linear_sum_codeword)
                .and(&from_codeword)
                .and(&to_codeword)
                .par_for_each(|accumulator, &from, &to| *accumulator += weight * (from - to));
        }

        // standard input
        Zip::from(&mut non_linear_sum_codeword)
            .and(master_ext_table.column(self.input_to_processor))
            .par_for_each(|accumulator, &to| {
                *accumulator += self.input_to_processor_weight * (self.input_terminal - to)
            });

        // standard output
        Zip::from(&mut non_linear_sum_codeword)
            .and(master_ext_table.column(self.processor_to_output))
            .par_for_each(|accumulator, &from| {
                *accumulator += self.processor_to_output_weight * (from - self.output_terminal)
            });

        let zerofier_inverse =
            terminal_quotient_zerofier_inverse(quotient_domain, trace_domain_generator);
        non_linear_sum_codeword * zerofier_inverse
    }

    pub fn quotient_degree_bound(
        &self,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Degree {
        self.into_iter()
            .map(|(arg, _)| arg.quotient_degree_bound(padded_height, num_trace_randomizers))
            .max()
            .unwrap_or(0)
    }

    pub fn evaluate_non_linear_sum_of_differences(
        &self,
        ext_row: ArrayView1<XFieldElement>,
    ) -> XFieldElement {
        // cross-table arguments
        let mut non_linear_sum = self
            .into_iter()
            .map(|(arg, weight)| weight * arg.evaluate_difference(ext_row))
            .sum();

        // input
        let processor_in = ext_row[self.input_to_processor];
        non_linear_sum += self.input_to_processor_weight * (self.input_terminal - processor_in);

        // output
        let processor_out = ext_row[self.processor_to_output];
        non_linear_sum += self.processor_to_output_weight * (processor_out - self.output_terminal);

        non_linear_sum
    }
}

#[cfg(test)]
mod permutation_argument_tests {
    use crate::stark::triton_stark_tests::parse_simulate_pad_extend;
    use crate::table::table_collection::EXT_PROCESSOR_TABLE_END;
    use crate::table::table_collection::EXT_PROCESSOR_TABLE_START;
    use crate::vm::triton_vm_tests::test_hash_nop_nop_lt;

    use super::*;

    #[test]
    fn all_permutation_arguments_link_from_processor_table_test() {
        for perm_arg in PermArg::all_permutation_arguments() {
            let goes_from_processor_table = perm_arg.from().iter().any(|&column| {
                (EXT_PROCESSOR_TABLE_START..EXT_PROCESSOR_TABLE_END).contains(&column)
            });
            let goes_to_processor_table = perm_arg.to().iter().any(|&column| {
                (EXT_PROCESSOR_TABLE_START..EXT_PROCESSOR_TABLE_END).contains(&column)
            });
            assert!(goes_from_processor_table || goes_to_processor_table);
        }
    }

    #[test]
    fn almost_all_quotient_degree_bounds_of_grand_cross_table_argument_are_equal_test() {
        let num_trace_randomizers = 10;
        let code_with_input = test_hash_nop_nop_lt();
        let code = code_with_input.source_code;
        let input = code_with_input.input;
        let secret_input = code_with_input.secret_input;
        let (stark, _, master_base_table, _, all_challenges) =
            parse_simulate_pad_extend(&code, input, secret_input);

        let input_terminal = EvalArg::compute_terminal(
            &stark.claim.input,
            EvalArg::default_initial(),
            all_challenges
                .processor_table_challenges
                .standard_input_eval_indeterminate,
        );

        let output_terminal = EvalArg::compute_terminal(
            &stark.claim.output,
            EvalArg::default_initial(),
            all_challenges
                .processor_table_challenges
                .standard_output_eval_indeterminate,
        );

        let gxta = GrandCrossTableArg::new(
            &[XFieldElement::one(); NUM_CROSS_TABLE_ARGS + 2],
            input_terminal,
            output_terminal,
        );

        let padded_height = master_base_table.padded_height;
        let quotient_degree_bound = gxta
            .program_to_instruction
            .quotient_degree_bound(padded_height, num_trace_randomizers);
        for (arg, _) in gxta.into_iter().take(7) {
            assert_eq!(
                quotient_degree_bound,
                arg.quotient_degree_bound(padded_height, num_trace_randomizers)
            );
        }
    }
}
