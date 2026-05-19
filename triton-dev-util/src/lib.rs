//! Utility functions to help test and benchmark Triton VM.
//!
//! This crate is not intended for publication on crates.io.

#![recursion_limit = "4096"]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use bon::Builder;
use triton_vm::aet::AlgebraicExecutionTrace;
use triton_vm::prelude::BFieldElement;
use triton_vm::prelude::Claim;
use triton_vm::prelude::NonDeterminism;
use triton_vm::prelude::Program;
use triton_vm::prelude::Proof;
use triton_vm::prelude::PublicInput;
use triton_vm::prelude::Stark;
use triton_vm::prelude::TableId;
use triton_vm::prelude::VM;
use triton_vm::prelude::bfe_array;
use triton_vm::prelude::triton_program;
use triton_vm::profiler::VMPerformanceProfile;

pub mod example_programs;

/// Ties together a program with its inputs, name, and the proof system to use.
#[derive(Debug, Clone, Eq, PartialEq, Builder)]
#[builder(start_fn = new)]
#[builder(finish_fn = assemble)]
pub struct ProgramToBench {
    #[builder(start_fn)]
    #[builder(into)]
    pub name: String,

    #[builder(start_fn)]
    pub program: Program,

    #[builder(default)]
    #[builder(into)]
    pub public_input: PublicInput,

    #[builder(default)]
    pub non_determinism: NonDeterminism,

    #[builder(default)]
    pub stark: Stark,
}

impl ProgramToBench {
    /// A program that counts down for so long that its
    /// [execution trace](AlgebraicExecutionTrace) reaches the given padded
    /// height.
    pub fn spin(log2_padded_height: u64) -> Self {
        let program = triton_program!(
            read_io 1   // _ log₂(padded height)
            addi -3
            push 2
            pow         // _ num_iterations
            place 5
            call spin
            halt

            spin:
              pick 5 addi -1 place 5
              recurse_or_return
        );

        let name = format!("spin_{log2_padded_height}");
        ProgramToBench::new(name, program)
            .public_input(bfe_array![log2_padded_height])
            .assemble()
    }

    /// Switch to a different [STARK](Stark) for this program.
    #[must_use]
    pub fn with_stark(mut self, stark: Stark) -> Self {
        self.stark = stark;
        self
    }

    /// The base 2, integer logarithm of the length of the low-degree test domain.
    pub fn log2_ldt_domain_length(&self) -> u32 {
        let (aet, _) = self.trace_execution();
        let ldt = self.stark.ldt(aet.padded_height()).unwrap();

        ldt.initial_domain().len().ilog2()
    }

    pub fn prove(&self) -> Proof {
        let (aet, public_output) = self.trace_execution();
        let claim = Claim::about_program(&self.program)
            .with_input(self.public_input.clone())
            .with_output(public_output);

        self.stark.prove(&claim, &aet).unwrap()
    }

    pub fn trace_execution(&self) -> (AlgebraicExecutionTrace, Vec<BFieldElement>) {
        VM::trace_execution(
            self.program.clone(),
            self.public_input.clone(),
            self.non_determinism.clone(),
        )
        .unwrap()
    }

    /// Generate a performance profile of one run of the [prover](Self::prove).
    pub fn performance_profile(&self) -> VMPerformanceProfile {
        triton_vm::profiler::start(&self.name);
        self.prove();
        let profile = triton_vm::profiler::finish();

        self.fill_performance_profile(profile)
    }

    /// Add as much optional data as possible to the given performance profile.
    #[must_use]
    pub fn fill_performance_profile(&self, profile: VMPerformanceProfile) -> VMPerformanceProfile {
        let (aet, _) = self.trace_execution();
        let cycle_count = aet.height_of_table(TableId::Processor);
        let padded_height = aet.padded_height();
        let ldt = self.stark.ldt(padded_height).unwrap();

        profile
            .with_cycle_count(cycle_count)
            .with_padded_height(padded_height)
            .with_ldt_domain_len(ldt.initial_domain().len())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use triton_vm::stark::Stark;

    use super::*;

    #[test]
    fn sanity() {
        let program = ProgramToBench::spin(5).with_stark(Stark::new(40, 2));
        let log2_ldt_domain_len = program.log2_ldt_domain_length();
        let profile = program.performance_profile();

        println!("log2_ldt_domain_len: {log2_ldt_domain_len}\n\n{profile}");
    }
}
