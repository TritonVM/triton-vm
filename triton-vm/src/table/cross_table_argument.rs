use std::ops::Mul;

use ndarray::ArrayView1;
use num_traits::One;
use strum_macros::Display;
use strum_macros::EnumCount as EnumCountMacro;
use strum_macros::EnumIter;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::x_field_element::XFieldElement;

use CrossTableChallengeId::*;

use crate::table::challenges::AllChallenges;
use crate::table::challenges::TableChallenges;
use crate::table::extension_table::Evaluable;
use crate::table::extension_table::Quotientable;
use crate::table::processor_table::PROCESSOR_TABLE_NUM_PERMUTATION_ARGUMENTS;
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
pub const NUM_CROSS_TABLE_WEIGHTS: usize = NUM_CROSS_TABLE_ARGS + NUM_PUBLIC_EVAL_ARGS;

pub trait CrossTableArg {
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
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PermArg {}

impl CrossTableArg for PermArg {
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

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct EvalArg {}

impl CrossTableArg for EvalArg {
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

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct GrandCrossTableArg {}

#[derive(Clone, Debug)]
pub struct CrossTableChallenges {
    pub input_terminal: XFieldElement,
    pub output_terminal: XFieldElement,

    pub program_to_instruction_weight: XFieldElement,
    pub processor_to_instruction_weight: XFieldElement,
    pub processor_to_op_stack_weight: XFieldElement,
    pub processor_to_ram_weight: XFieldElement,
    pub processor_to_jump_stack_weight: XFieldElement,
    pub processor_to_hash_weight: XFieldElement,
    pub hash_to_processor_weight: XFieldElement,
    pub all_clock_jump_differences_weight: XFieldElement,
    pub input_to_processor_weight: XFieldElement,
    pub processor_to_output_weight: XFieldElement,
}

#[derive(Debug, Copy, Clone, Display, EnumCountMacro, EnumIter, PartialEq, Eq, Hash)]
pub enum CrossTableChallengeId {
    InputTerminal,
    OutputTerminal,

    ProgramToInstructionWeight,
    ProcessorToInstructionWeight,
    ProcessorToOpStackWeight,
    ProcessorToRamWeight,
    ProcessorToJumpStackWeight,
    ProcessorToHashWeight,
    HashToProcessorWeight,
    AllClockJumpDifferencesWeight,
    InputToProcessorWeight,
    ProcessorToOutputWeight,
}

impl From<CrossTableChallengeId> for usize {
    fn from(val: CrossTableChallengeId) -> Self {
        val as usize
    }
}

impl TableChallenges for CrossTableChallenges {
    type Id = CrossTableChallengeId;

    #[inline]
    fn get_challenge(&self, id: Self::Id) -> XFieldElement {
        match id {
            InputTerminal => self.input_terminal,
            OutputTerminal => self.output_terminal,
            ProgramToInstructionWeight => self.program_to_instruction_weight,
            ProcessorToInstructionWeight => self.processor_to_instruction_weight,
            ProcessorToOpStackWeight => self.processor_to_op_stack_weight,
            ProcessorToRamWeight => self.processor_to_ram_weight,
            ProcessorToJumpStackWeight => self.processor_to_jump_stack_weight,
            ProcessorToHashWeight => self.processor_to_hash_weight,
            HashToProcessorWeight => self.hash_to_processor_weight,
            AllClockJumpDifferencesWeight => self.all_clock_jump_differences_weight,
            InputToProcessorWeight => self.input_to_processor_weight,
            ProcessorToOutputWeight => self.processor_to_output_weight,
        }
    }
}

impl Evaluable for GrandCrossTableArg {
    fn evaluate_initial_constraints(
        _base_row: ArrayView1<BFieldElement>,
        _ext_row: ArrayView1<XFieldElement>,
        _challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {
        vec![]
    }

    fn evaluate_consistency_constraints(
        _base_row: ArrayView1<BFieldElement>,
        _ext_row: ArrayView1<XFieldElement>,
        _challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {
        vec![]
    }

    fn evaluate_transition_constraints(
        _current_base_row: ArrayView1<BFieldElement>,
        _current_ext_row: ArrayView1<XFieldElement>,
        _next_base_row: ArrayView1<BFieldElement>,
        _next_ext_row: ArrayView1<XFieldElement>,
        _challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {
        vec![]
    }

    fn evaluate_terminal_constraints(
        _base_row: ArrayView1<BFieldElement>,
        ext_row: ArrayView1<XFieldElement>,
        challenges: &AllChallenges,
    ) -> Vec<XFieldElement> {
        let challenges = &challenges.cross_table_challenges;

        let input_to_processor = challenges.get_challenge(InputTerminal)
            - ext_row[ProcessorExtTableColumn::InputTableEvalArg.master_ext_table_index()];
        let processor_to_output = ext_row
            [ProcessorExtTableColumn::OutputTableEvalArg.master_ext_table_index()]
            - challenges.get_challenge(OutputTerminal);

        let program_to_instruction = ext_row
            [ProgramExtTableColumn::RunningEvaluation.master_ext_table_index()]
            - ext_row[InstructionExtTableColumn::RunningEvaluation.master_ext_table_index()];
        let processor_to_instruction = ext_row
            [ProcessorExtTableColumn::InstructionTablePermArg.master_ext_table_index()]
            - ext_row[InstructionExtTableColumn::RunningProductPermArg.master_ext_table_index()];
        let processor_to_op_stack = ext_row
            [ProcessorExtTableColumn::OpStackTablePermArg.master_ext_table_index()]
            - ext_row[OpStackExtTableColumn::RunningProductPermArg.master_ext_table_index()];
        let processor_to_ram = ext_row
            [ProcessorExtTableColumn::RamTablePermArg.master_ext_table_index()]
            - ext_row[RamExtTableColumn::RunningProductPermArg.master_ext_table_index()];
        let processor_to_jump_stack = ext_row
            [ProcessorExtTableColumn::JumpStackTablePermArg.master_ext_table_index()]
            - ext_row[JumpStackExtTableColumn::RunningProductPermArg.master_ext_table_index()];
        let processor_to_hash = ext_row
            [ProcessorExtTableColumn::ToHashTableEvalArg.master_ext_table_index()]
            - ext_row[HashExtTableColumn::FromProcessorRunningEvaluation.master_ext_table_index()];
        let hash_to_processor = ext_row
            [HashExtTableColumn::ToProcessorRunningEvaluation.master_ext_table_index()]
            - ext_row[ProcessorExtTableColumn::FromHashTableEvalArg.master_ext_table_index()];
        let all_clock_jump_differences = ext_row
            [ProcessorExtTableColumn::AllClockJumpDifferencesPermArg.master_ext_table_index()]
            - ext_row
                [OpStackExtTableColumn::AllClockJumpDifferencesPermArg.master_ext_table_index()]
                * ext_row
                    [RamExtTableColumn::AllClockJumpDifferencesPermArg.master_ext_table_index()]
                * ext_row[JumpStackExtTableColumn::AllClockJumpDifferencesPermArg
                    .master_ext_table_index()];

        let non_linear_sum = challenges.get_challenge(InputToProcessorWeight) * input_to_processor
            + challenges.get_challenge(ProcessorToOutputWeight) * processor_to_output
            + challenges.get_challenge(ProgramToInstructionWeight) * program_to_instruction
            + challenges.get_challenge(ProcessorToInstructionWeight) * processor_to_instruction
            + challenges.get_challenge(ProcessorToOpStackWeight) * processor_to_op_stack
            + challenges.get_challenge(ProcessorToRamWeight) * processor_to_ram
            + challenges.get_challenge(ProcessorToJumpStackWeight) * processor_to_jump_stack
            + challenges.get_challenge(ProcessorToHashWeight) * processor_to_hash
            + challenges.get_challenge(HashToProcessorWeight) * hash_to_processor
            + challenges.get_challenge(AllClockJumpDifferencesWeight) * all_clock_jump_differences;
        vec![non_linear_sum]
    }
}

impl Quotientable for GrandCrossTableArg {
    fn num_initial_quotients() -> usize {
        0
    }

    fn num_consistency_quotients() -> usize {
        0
    }

    fn num_transition_quotients() -> usize {
        0
    }

    fn num_terminal_quotients() -> usize {
        1
    }

    fn initial_quotient_degree_bounds(_interpolant_degree: Degree) -> Vec<Degree> {
        vec![]
    }

    fn consistency_quotient_degree_bounds(
        _interpolant_degree: Degree,
        _padded_height: usize,
    ) -> Vec<Degree> {
        vec![]
    }

    fn transition_quotient_degree_bounds(
        _interpolant_degree: Degree,
        _padded_height: usize,
    ) -> Vec<Degree> {
        vec![]
    }

    fn terminal_quotient_degree_bounds(interpolant_degree: Degree) -> Vec<Degree> {
        let zerofier_degree = 1 as Degree;
        let max_columns_involved_in_one_cross_table_argument = 3;
        vec![
            interpolant_degree * max_columns_involved_in_one_cross_table_argument - zerofier_degree,
        ]
    }
}
