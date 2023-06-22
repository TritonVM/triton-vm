use std::ops::Add;
use std::ops::Mul;

use itertools::Itertools;
use num_traits::One;
use num_traits::Zero;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::challenges::ChallengeId::*;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::ConstraintCircuitMonad;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::ExtRow;
use crate::table::table_column::CascadeExtTableColumn;
use crate::table::table_column::HashExtTableColumn;
use crate::table::table_column::HashExtTableColumn::*;
use crate::table::table_column::JumpStackExtTableColumn;
use crate::table::table_column::LookupExtTableColumn;
use crate::table::table_column::LookupExtTableColumn::*;
use crate::table::table_column::MasterExtTableColumn;
use crate::table::table_column::OpStackExtTableColumn;
use crate::table::table_column::ProcessorExtTableColumn;
use crate::table::table_column::ProcessorExtTableColumn::*;
use crate::table::table_column::ProgramExtTableColumn;
use crate::table::table_column::ProgramExtTableColumn::*;
use crate::table::table_column::RamExtTableColumn;
use crate::table::table_column::U32ExtTableColumn;
use crate::table::table_column::U32ExtTableColumn::*;

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
    /// at point `challenge`, _i.e._, returns `f(challenge)`.
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
    /// at point `challenge`, _i.e._, returns `f(challenge)`.
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
pub struct LookupArg {}

impl CrossTableArg for LookupArg {
    fn default_initial() -> XFieldElement {
        XFieldElement::zero()
    }

    fn compute_terminal(
        symbols: &[BFieldElement],
        initial: XFieldElement,
        challenge: XFieldElement,
    ) -> XFieldElement {
        symbols
            .iter()
            .map(|symbol| (challenge - symbol.lift()).inverse())
            .fold(initial, XFieldElement::add)
    }
}

impl LookupArg {
    pub fn compute_terminal_with_multiplicities(
        symbols: &[BFieldElement],
        multiplicities: &[u32],
        initial: XFieldElement,
        challenge: XFieldElement,
    ) -> XFieldElement {
        symbols
            .iter()
            .zip_eq(multiplicities.iter())
            .map(|(symbol, &multiplicity)| {
                (challenge - symbol.lift()).inverse() * XFieldElement::from(multiplicity)
            })
            .fold(initial, XFieldElement::add)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct GrandCrossTableArg {}

impl GrandCrossTableArg {
    pub fn initial_constraints(
        _circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        // no further constraints
        vec![]
    }

    pub fn consistency_constraints(
        _circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        // no further constraints
        vec![]
    }

    pub fn transition_constraints(
        _circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        // no further constraints
        vec![]
    }

    pub fn terminal_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let challenge = |c| circuit_builder.challenge(c);
        let ext_row = |col_index| circuit_builder.input(ExtRow(col_index));

        // Closures cannot take arguments of type `impl Trait`. Hence: some more helpers. \o/
        let program_ext_row =
            |column: ProgramExtTableColumn| ext_row(column.master_ext_table_index());
        let processor_ext_row =
            |column: ProcessorExtTableColumn| ext_row(column.master_ext_table_index());
        let op_stack_ext_row =
            |column: OpStackExtTableColumn| ext_row(column.master_ext_table_index());
        let ram_ext_row = |column: RamExtTableColumn| ext_row(column.master_ext_table_index());
        let jump_stack_ext_row =
            |column: JumpStackExtTableColumn| ext_row(column.master_ext_table_index());
        let hash_ext_row = |column: HashExtTableColumn| ext_row(column.master_ext_table_index());
        let cascade_ext_row =
            |column: CascadeExtTableColumn| ext_row(column.master_ext_table_index());
        let lookup_ext_row =
            |column: LookupExtTableColumn| ext_row(column.master_ext_table_index());
        let u32_ext_row = |column: U32ExtTableColumn| ext_row(column.master_ext_table_index());

        let program_attestation = program_ext_row(SendChunkRunningEvaluation)
            - hash_ext_row(ReceiveChunkRunningEvaluation);
        let input_to_processor =
            challenge(StandardInputTerminal) - processor_ext_row(InputTableEvalArg);
        let processor_to_output =
            processor_ext_row(OutputTableEvalArg) - challenge(StandardOutputTerminal);
        let instruction_lookup = processor_ext_row(InstructionLookupClientLogDerivative)
            - program_ext_row(InstructionLookupServerLogDerivative);
        let processor_to_op_stack = processor_ext_row(OpStackTablePermArg)
            - op_stack_ext_row(OpStackExtTableColumn::RunningProductPermArg);
        let processor_to_ram = processor_ext_row(RamTablePermArg)
            - ram_ext_row(RamExtTableColumn::RunningProductPermArg);
        let processor_to_jump_stack = processor_ext_row(JumpStackTablePermArg)
            - jump_stack_ext_row(JumpStackExtTableColumn::RunningProductPermArg);
        let hash_input =
            processor_ext_row(HashInputEvalArg) - hash_ext_row(HashInputRunningEvaluation);
        let hash_digest =
            hash_ext_row(HashDigestRunningEvaluation) - processor_ext_row(HashDigestEvalArg);
        let sponge = processor_ext_row(SpongeEvalArg) - hash_ext_row(SpongeRunningEvaluation);
        let hash_to_cascade = cascade_ext_row(CascadeExtTableColumn::HashTableServerLogDerivative)
            - hash_ext_row(CascadeState0HighestClientLogDerivative)
            - hash_ext_row(CascadeState0MidHighClientLogDerivative)
            - hash_ext_row(CascadeState0MidLowClientLogDerivative)
            - hash_ext_row(CascadeState0LowestClientLogDerivative)
            - hash_ext_row(CascadeState1HighestClientLogDerivative)
            - hash_ext_row(CascadeState1MidHighClientLogDerivative)
            - hash_ext_row(CascadeState1MidLowClientLogDerivative)
            - hash_ext_row(CascadeState1LowestClientLogDerivative)
            - hash_ext_row(CascadeState2HighestClientLogDerivative)
            - hash_ext_row(CascadeState2MidHighClientLogDerivative)
            - hash_ext_row(CascadeState2MidLowClientLogDerivative)
            - hash_ext_row(CascadeState2LowestClientLogDerivative)
            - hash_ext_row(CascadeState3HighestClientLogDerivative)
            - hash_ext_row(CascadeState3MidHighClientLogDerivative)
            - hash_ext_row(CascadeState3MidLowClientLogDerivative)
            - hash_ext_row(CascadeState3LowestClientLogDerivative);
        let cascade_to_lookup =
            cascade_ext_row(CascadeExtTableColumn::LookupTableClientLogDerivative)
                - lookup_ext_row(CascadeTableServerLogDerivative);
        let processor_to_u32 = processor_ext_row(U32LookupClientLogDerivative)
            - u32_ext_row(LookupServerLogDerivative);

        // Introduce new variable names to increase readability. Potentially opinionated.
        let processor_cjdld = ClockJumpDifferenceLookupServerLogDerivative;
        let op_stack_cjdld = OpStackExtTableColumn::ClockJumpDifferenceLookupClientLogDerivative;
        let ram_cjdld = RamExtTableColumn::ClockJumpDifferenceLookupClientLogDerivative;
        let j_stack_cjdld = JumpStackExtTableColumn::ClockJumpDifferenceLookupClientLogDerivative;
        let clock_jump_difference_lookup = processor_ext_row(processor_cjdld)
            - op_stack_ext_row(op_stack_cjdld)
            - ram_ext_row(ram_cjdld)
            - jump_stack_ext_row(j_stack_cjdld);

        vec![
            program_attestation,
            input_to_processor,
            processor_to_output,
            instruction_lookup,
            processor_to_op_stack,
            processor_to_ram,
            processor_to_jump_stack,
            hash_input,
            hash_digest,
            sponge,
            hash_to_cascade,
            cascade_to_lookup,
            processor_to_u32,
            clock_jump_difference_lookup,
        ]
    }
}
