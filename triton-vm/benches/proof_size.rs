use std::collections::HashMap;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::measurement::Measurement;
use criterion::measurement::ValueFormatter;
use itertools::Itertools;
use strum::Display;
use strum::EnumCount;
use strum::EnumIter;
use triton_vm::aet::AlgebraicExecutionTrace;
use triton_vm::low_degree_test::ProximityRegime;
use triton_vm::prelude::*;
use triton_vm::proof_stream::ProofStream;
use triton_vm::stark::LdtChoice;

/// Ties together a program with its inputs, name, and the proof system to use.
struct ProgramToBench {
    name: String,
    program: Program,
    public_input: PublicInput,
    non_determinism: NonDeterminism,
    stark: Stark,
}

/// The measurement unit for Criterion.
#[derive(Debug, Copy, Clone)]
struct ProofSize(f64);

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().with_measurement(ProofSize(0.0)).sample_size(10);
    targets = bench_proof_sizes_for_various_starks
);

fn bench_proof_sizes_for_various_starks(c: &mut Criterion<ProofSize>) {
    bench_proof_sizes(c, LdtChoice::Fri, ProximityRegime::Proven);
    bench_proof_sizes(c, LdtChoice::Fri, ProximityRegime::Conjectured);
    bench_proof_sizes(c, LdtChoice::Stir, ProximityRegime::Proven);
    bench_proof_sizes(c, LdtChoice::Stir, ProximityRegime::Conjectured);
}

fn bench_proof_sizes(
    criterion: &mut Criterion<ProofSize>,
    ldt_choice: LdtChoice,
    soundness_type: ProximityRegime,
) {
    let stark = Stark::default()
        .with_ldt_choice(ldt_choice)
        .with_soundness(soundness_type);
    let mut group = criterion.benchmark_group(format!("Proof_Size_{ldt_choice}_{soundness_type}"));

    // A more meaningful range like 8..=17 is too much for the poor CI machines…
    for log2_padded_height in 8..=11 {
        let program = ProgramToBench::spin(log2_padded_height).with_stark(stark);
        let id = BenchmarkId::new(&program.name, program.log2_ldt_domain_length());
        group.bench_function(id, |b| b.iter_custom(|n| program.sum_of_proof_lengths(n)));
        program.print_proof_size_breakdown();
    }
}

impl ProgramToBench {
    fn spin(log2_padded_height: u64) -> Self {
        let name = format!("spin_{log2_padded_height}");
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
        let public_input = PublicInput::new(bfe_vec![log2_padded_height]);

        ProgramToBench {
            name,
            program,
            public_input,
            non_determinism: NonDeterminism::default(),
            stark: Stark::default(),
        }
    }

    #[must_use]
    fn with_stark(mut self, stark: Stark) -> Self {
        self.stark = stark;
        self
    }

    /// The base 2, integer logarithm of the length of the low-degree test domain.
    fn log2_ldt_domain_length(&self) -> u32 {
        let (aet, _) = self.trace_execution();
        let ldt = self.stark.ldt(aet.padded_height()).unwrap();

        ldt.initial_domain().len().ilog2()
    }

    fn prove(&self) -> Proof {
        let (aet, public_output) = self.trace_execution();
        let claim = Claim::about_program(&self.program)
            .with_input(self.public_input.clone())
            .with_output(public_output);

        self.stark.prove(&claim, &aet).unwrap()
    }

    fn trace_execution(&self) -> (AlgebraicExecutionTrace, Vec<BFieldElement>) {
        VM::trace_execution(
            self.program.clone(),
            self.public_input.clone(),
            self.non_determinism.clone(),
        )
        .unwrap()
    }

    fn sum_of_proof_lengths(&self, num_iterations: u64) -> ProofSize {
        let sum_of_proof_lengths = (0..num_iterations)
            .map(|_| self.prove().encode().len())
            .sum::<usize>();

        ProofSize(sum_of_proof_lengths as f64)
    }

    /// Print a tabular breakdown of the proof size for this
    /// program-to-benchmark.
    //
    // Prints to stderr in order to not interfere with `cargo`.
    fn print_proof_size_breakdown(&self) {
        let proof = self.prove();
        let proof_stream = ProofStream::try_from(&proof).unwrap();
        let mut proof_size_breakdown = HashMap::new();
        for item in &proof_stream.items {
            *proof_size_breakdown.entry(item.to_string()).or_insert(0) += item.encode().len();
        }

        // sort by item's size, in descending order
        let mut proof_size_breakdown = proof_size_breakdown.into_iter().collect_vec();
        proof_size_breakdown.sort_by_key(|&(_, size)| size);
        proof_size_breakdown.reverse();
        let total_proof_size = proof.encode().len();

        eprintln!();
        eprintln!("Proof size breakdown for {}:", &self.name);
        eprintln!(
            "| {:<30} | {:>10} | {:>6} |",
            "Category", "Size [bfe]", "[%]"
        );
        eprintln!("|:{:-<30}-|-{:->10}:|-{:->6}:|", "", "", "");
        for (category, size) in proof_size_breakdown {
            let relative_size = (size as f64) / (total_proof_size as f64) * 100.0;
            eprintln!("| {category:<30} | {size:>10} | {relative_size:>6.2} |");
        }
        eprintln!();
    }
}

impl Measurement for ProofSize {
    type Intermediate = ();
    type Value = Self;

    fn start(&self) -> Self::Intermediate {}

    fn end(&self, _i: Self::Intermediate) -> Self::Value {
        self.to_owned()
    }

    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        ProofSize(v1.0 + v2.0)
    }

    fn zero(&self) -> Self::Value {
        ProofSize(0.0)
    }

    fn to_f64(&self, value: &Self::Value) -> f64 {
        value.0
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &ProofSizeFormatter
    }
}

/// Several orders of magnitude data can come in.
#[derive(Debug, Display, Copy, Clone, EnumCount, EnumIter)]
enum DataSizeOrderOfMagnitude {
    Bytes,
    KiloBytes,
    MegaBytes,
    GigaBytes,
}

impl DataSizeOrderOfMagnitude {
    /// The order of magnitude the given number of bytes falls in.
    fn order_of_magnitude(num_bytes: f64) -> Self {
        match num_bytes {
            b if b < Self::KiloBytes.min_bytes_in_order_of_magnitude() => Self::Bytes,
            b if b < Self::MegaBytes.min_bytes_in_order_of_magnitude() => Self::KiloBytes,
            b if b < Self::GigaBytes.min_bytes_in_order_of_magnitude() => Self::MegaBytes,
            _ => Self::GigaBytes,
        }
    }

    /// The minimal number of bytes to be considered some order of magnitude.
    fn min_bytes_in_order_of_magnitude(self) -> f64 {
        match self {
            Self::Bytes => 1.0,
            Self::KiloBytes => 1024.0,
            Self::MegaBytes => 1024.0 * 1024.0,
            Self::GigaBytes => 1024.0 * 1024.0 * 1024.0,
        }
    }

    /// The typical abbreviation for this order of magnitude.
    fn abbreviation(self) -> &'static str {
        match self {
            Self::Bytes => "bytes",
            Self::KiloBytes => "KiB",
            Self::MegaBytes => "MiB",
            Self::GigaBytes => "GiB",
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct ProofSizeFormatter;

impl ProofSizeFormatter {
    const BYTES_PER_BFE: f64 = BFieldElement::BYTES as f64;
}

impl ValueFormatter for ProofSizeFormatter {
    fn scale_values(&self, typical_value: f64, values: &mut [f64]) -> &'static str {
        let size_in_bytes = typical_value * Self::BYTES_PER_BFE;
        let order_of_magnitude = DataSizeOrderOfMagnitude::order_of_magnitude(size_in_bytes);
        let normalization_divisor = order_of_magnitude.min_bytes_in_order_of_magnitude();
        let scaling_factor = Self::BYTES_PER_BFE / normalization_divisor;
        for value in values {
            *value *= scaling_factor;
        }

        order_of_magnitude.abbreviation()
    }

    fn scale_throughputs(
        &self,
        _typical_value: f64,
        _throughput: &Throughput,
        _values: &mut [f64],
    ) -> &'static str {
        "bfe/s"
    }

    fn scale_for_machines(&self, values: &mut [f64]) -> &'static str {
        for value in values {
            *value *= Self::BYTES_PER_BFE;
        }

        DataSizeOrderOfMagnitude::Bytes.abbreviation()
    }
}
