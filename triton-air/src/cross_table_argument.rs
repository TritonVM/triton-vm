use std::ops::Add;
use std::ops::Mul;

use constraint_circuit::ConstraintCircuitBuilder;
use constraint_circuit::ConstraintCircuitMonad;
use constraint_circuit::DualRowIndicator;
use constraint_circuit::SingleRowIndicator;
use constraint_circuit::SingleRowIndicator::Aux;
use twenty_first::prelude::*;

use crate::challenge_id::ChallengeId;

use crate::table_column::CascadeAuxColumn;
use crate::table_column::HashAuxColumn;
use crate::table_column::JumpStackAuxColumn;
use crate::table_column::LookupAuxColumn;
use crate::table_column::MasterAuxColumn;
use crate::table_column::OpStackAuxColumn;
use crate::table_column::ProcessorAuxColumn;
use crate::table_column::ProgramAuxColumn;
use crate::table_column::RamAuxColumn;
use crate::table_column::U32AuxColumn;

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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct PermArg;

impl CrossTableArg for PermArg {
    fn default_initial() -> XFieldElement {
        1.into()
    }

    /// Compute the product for a permutation argument as specified by
    /// `initial`, `challenge`, and `symbols`. This amounts to evaluating
    /// polynomial  `f(x) = initial · Π_i (x - symbols[i])`
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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct EvalArg;

impl CrossTableArg for EvalArg {
    fn default_initial() -> XFieldElement {
        1.into()
    }

    /// Compute the evaluation for an evaluation argument as specified by
    /// `initial`, `challenge`, and `symbols`. This amounts to evaluating
    /// polynomial `f(x) = initial·x^n + Σ_i symbols[n-i]·x^i`
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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct LookupArg;

impl CrossTableArg for LookupArg {
    fn default_initial() -> XFieldElement {
        0.into()
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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct GrandCrossTableArg;

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
        let aux_row = |col_index| circuit_builder.input(Aux(col_index));

        // Closures cannot take arguments of type `impl Trait`. Hence: some more
        // helpers. \o/
        let program_aux_row = |column: ProgramAuxColumn| aux_row(column.master_aux_index());
        let processor_aux_row = |column: ProcessorAuxColumn| aux_row(column.master_aux_index());
        let op_stack_aux_row = |column: OpStackAuxColumn| aux_row(column.master_aux_index());
        let ram_aux_row = |column: RamAuxColumn| aux_row(column.master_aux_index());
        let j_stack_aux_row = |column: JumpStackAuxColumn| aux_row(column.master_aux_index());
        let hash_aux_row = |column: HashAuxColumn| aux_row(column.master_aux_index());
        let cascade_aux_row = |column: CascadeAuxColumn| aux_row(column.master_aux_index());
        let lookup_aux_row = |column: LookupAuxColumn| aux_row(column.master_aux_index());
        let u32_aux_row = |column: U32AuxColumn| aux_row(column.master_aux_index());

        let program_attestation = program_aux_row(ProgramAuxColumn::SendChunkRunningEvaluation)
            - hash_aux_row(HashAuxColumn::ReceiveChunkRunningEvaluation);
        let input_to_processor = challenge(ChallengeId::StandardInputTerminal)
            - processor_aux_row(ProcessorAuxColumn::InputTableEvalArg);
        let processor_to_output = processor_aux_row(ProcessorAuxColumn::OutputTableEvalArg)
            - challenge(ChallengeId::StandardOutputTerminal);
        let instruction_lookup =
            processor_aux_row(ProcessorAuxColumn::InstructionLookupClientLogDerivative)
                - program_aux_row(ProgramAuxColumn::InstructionLookupServerLogDerivative);
        let processor_to_op_stack = processor_aux_row(ProcessorAuxColumn::OpStackTablePermArg)
            - op_stack_aux_row(OpStackAuxColumn::RunningProductPermArg);
        let processor_to_ram = processor_aux_row(ProcessorAuxColumn::RamTablePermArg)
            - ram_aux_row(RamAuxColumn::RunningProductPermArg);
        let processor_to_jump_stack = processor_aux_row(ProcessorAuxColumn::JumpStackTablePermArg)
            - j_stack_aux_row(JumpStackAuxColumn::RunningProductPermArg);
        let hash_input = processor_aux_row(ProcessorAuxColumn::HashInputEvalArg)
            - hash_aux_row(HashAuxColumn::HashInputRunningEvaluation);
        let hash_digest = hash_aux_row(HashAuxColumn::HashDigestRunningEvaluation)
            - processor_aux_row(ProcessorAuxColumn::HashDigestEvalArg);
        let sponge = processor_aux_row(ProcessorAuxColumn::SpongeEvalArg)
            - hash_aux_row(HashAuxColumn::SpongeRunningEvaluation);
        let hash_to_cascade = cascade_aux_row(CascadeAuxColumn::HashTableServerLogDerivative)
            - hash_aux_row(HashAuxColumn::CascadeState0HighestClientLogDerivative)
            - hash_aux_row(HashAuxColumn::CascadeState0MidHighClientLogDerivative)
            - hash_aux_row(HashAuxColumn::CascadeState0MidLowClientLogDerivative)
            - hash_aux_row(HashAuxColumn::CascadeState0LowestClientLogDerivative)
            - hash_aux_row(HashAuxColumn::CascadeState1HighestClientLogDerivative)
            - hash_aux_row(HashAuxColumn::CascadeState1MidHighClientLogDerivative)
            - hash_aux_row(HashAuxColumn::CascadeState1MidLowClientLogDerivative)
            - hash_aux_row(HashAuxColumn::CascadeState1LowestClientLogDerivative)
            - hash_aux_row(HashAuxColumn::CascadeState2HighestClientLogDerivative)
            - hash_aux_row(HashAuxColumn::CascadeState2MidHighClientLogDerivative)
            - hash_aux_row(HashAuxColumn::CascadeState2MidLowClientLogDerivative)
            - hash_aux_row(HashAuxColumn::CascadeState2LowestClientLogDerivative)
            - hash_aux_row(HashAuxColumn::CascadeState3HighestClientLogDerivative)
            - hash_aux_row(HashAuxColumn::CascadeState3MidHighClientLogDerivative)
            - hash_aux_row(HashAuxColumn::CascadeState3MidLowClientLogDerivative)
            - hash_aux_row(HashAuxColumn::CascadeState3LowestClientLogDerivative);
        let cascade_to_lookup = cascade_aux_row(CascadeAuxColumn::LookupTableClientLogDerivative)
            - lookup_aux_row(LookupAuxColumn::CascadeTableServerLogDerivative);
        let processor_to_u32 = processor_aux_row(ProcessorAuxColumn::U32LookupClientLogDerivative)
            - u32_aux_row(U32AuxColumn::LookupServerLogDerivative);

        let clock_jump_difference_lookup =
            processor_aux_row(ProcessorAuxColumn::ClockJumpDifferenceLookupServerLogDerivative)
                - op_stack_aux_row(OpStackAuxColumn::ClockJumpDifferenceLookupClientLogDerivative)
                - ram_aux_row(RamAuxColumn::ClockJumpDifferenceLookupClientLogDerivative)
                - j_stack_aux_row(JumpStackAuxColumn::ClockJumpDifferenceLookupClientLogDerivative);

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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use num_traits::Zero;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use super::*;

    #[proptest]
    fn permutation_argument_is_identical_to_evaluating_zerofier_polynomial(
        #[strategy(arb())] roots: Vec<BFieldElement>,
        #[strategy(arb())] initial: XFieldElement,
        #[strategy(arb())] challenge: XFieldElement,
    ) {
        let poly_evaluation =
            initial * Polynomial::zerofier(&roots).evaluate::<_, XFieldElement>(challenge);
        let perm_arg_terminal = PermArg::compute_terminal(&roots, initial, challenge);
        prop_assert_eq!(poly_evaluation, perm_arg_terminal);
    }

    #[proptest]
    fn evaluation_argument_is_identical_to_evaluating_polynomial(
        #[strategy(arb())]
        #[filter(!#polynomial.is_zero())]
        polynomial: Polynomial<'static, BFieldElement>,
        #[strategy(arb())] challenge: XFieldElement,
    ) {
        let poly_evaluation: XFieldElement = polynomial.evaluate(challenge);

        let mut coefficients = polynomial.into_coefficients();
        let initial = coefficients.pop().unwrap();
        coefficients.reverse();
        let eval_arg_terminal = EvalArg::compute_terminal(&coefficients, initial.lift(), challenge);

        prop_assert_eq!(poly_evaluation, eval_arg_terminal);
    }

    #[proptest]
    fn lookup_argument_is_identical_to_inverse_of_evaluation_of_zerofier_polynomial(
        #[strategy(arb())]
        #[filter(#roots.iter().all(|r| r.lift() != #challenge))]
        roots: Vec<BFieldElement>,
        #[strategy(arb())] initial: XFieldElement,
        #[strategy(arb())] challenge: XFieldElement,
    ) {
        let polynomial = Polynomial::zerofier(&roots);
        let derivative = polynomial.formal_derivative();
        let poly_evaluation = derivative.evaluate::<_, XFieldElement>(challenge)
            / polynomial.evaluate::<_, XFieldElement>(challenge);
        let lookup_arg_terminal = LookupArg::compute_terminal(&roots, initial, challenge);
        prop_assert_eq!(initial + poly_evaluation, lookup_arg_terminal);
    }
}
