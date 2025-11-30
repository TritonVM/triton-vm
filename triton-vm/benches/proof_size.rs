use std::collections::HashMap;
use std::ops::AddAssign;

use criterion::BenchmarkGroup;
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
use twenty_first::prelude::*;

use triton_vm::example_programs::FIBONACCI_SEQUENCE;
use triton_vm::example_programs::VERIFY_SUDOKU;
use triton_vm::prelude::*;
use triton_vm::proof_stream::ProofStream;
use triton_vm::prove_program;

const BYTES_PER_BFE: f64 = BFieldElement::BYTES as f64;

/// Ties together a program and its inputs.
struct ProgramAndInput {
    program: Program,
    public_input: PublicInput,
    non_determinism: NonDeterminism,
}

impl ProgramAndInput {
    fn new(program: Program) -> Self {
        Self {
            program,
            public_input: PublicInput::default(),
            non_determinism: NonDeterminism::default(),
        }
    }

    #[must_use]
    fn with_input<I: Into<PublicInput>>(mut self, public_input: I) -> Self {
        self.public_input = public_input.into();
        self
    }
}

/// The measurement unit for Criterion.
#[derive(Debug, Copy, Clone)]
struct ProofSize(f64);

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

impl ValueFormatter for ProofSizeFormatter {
    fn scale_values(&self, typical_value: f64, values: &mut [f64]) -> &'static str {
        let size_in_bytes = typical_value * BYTES_PER_BFE;
        let order_of_magnitude = DataSizeOrderOfMagnitude::order_of_magnitude(size_in_bytes);
        let normalization_divisor = order_of_magnitude.min_bytes_in_order_of_magnitude();
        let scaling_factor = BYTES_PER_BFE / normalization_divisor;
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
            *value *= BYTES_PER_BFE;
        }

        DataSizeOrderOfMagnitude::Bytes.abbreviation()
    }
}

/// The source code for verifying a Sudoku with an example Sudoku provided as
/// input.
fn program_verify_sudoku() -> ProgramAndInput {
    let sudoku = [
        1, 2, 3, /*  */ 4, 5, 6, /*  */ 7, 8, 9, //
        4, 5, 6, /*  */ 7, 8, 9, /*  */ 1, 2, 3, //
        7, 8, 9, /*  */ 1, 2, 3, /*  */ 4, 5, 6, //
        /************************************ */
        2, 3, 4, /*  */ 5, 6, 7, /*  */ 8, 9, 1, //
        5, 6, 7, /*  */ 8, 9, 1, /*  */ 2, 3, 4, //
        8, 9, 1, /*  */ 2, 3, 4, /*  */ 5, 6, 7, //
        /************************************ */
        3, 4, 5, /*  */ 6, 7, 8, /*  */ 9, 1, 2, //
        6, 7, 8, /*  */ 9, 1, 2, /*  */ 3, 4, 5, //
        9, 1, 2, /*  */ 3, 4, 5, /*  */ 6, 7, 8, //
    ];
    ProgramAndInput::new(VERIFY_SUDOKU.clone()).with_input(sudoku.map(|b| bfe!(b)))
}

/// The program for computing some Fibonacci number, accepting as input which
/// number of the sequence to compute.
fn program_fib(nth_element: u64) -> ProgramAndInput {
    ProgramAndInput::new(FIBONACCI_SEQUENCE.clone()).with_input(bfe_array![nth_element])
}

fn program_halt() -> ProgramAndInput {
    ProgramAndInput::new(triton_program!(halt))
}

/// The base 2, integer logarithm of the FRI domain length.
fn log_2_fri_domain_length(stark: Stark, proof: &Proof) -> u32 {
    let padded_height = proof.padded_height().unwrap();
    let fri = stark.fri(padded_height).unwrap();
    fri.domain.len().ilog2()
}

/// List the sizes of the proof's parts. If the same item type is contained
/// multiple times, the sizes for that type are accumulated.
fn break_down_proof_size(proof: &Proof) -> HashMap<String, usize> {
    let mut proof_size_breakdown = HashMap::new();
    let proof_stream = ProofStream::try_from(proof).unwrap();
    for proof_item in &proof_stream.items {
        let item_name = proof_item.to_string();
        let item_len = proof_item.encode().len();
        proof_size_breakdown
            .entry(item_name)
            .or_insert(0)
            .add_assign(item_len);
    }
    proof_size_breakdown
}

/// Sort given HashMap by its values in descending order.
fn sort_hash_map_by_value_descending<K, V>(hash_map: HashMap<K, V>) -> Vec<(K, V)>
where
    V: Ord + Copy,
{
    let mut vec = hash_map.into_iter().collect_vec();
    vec.sort_by_key(|&(_, value)| value);
    vec.reverse();
    vec
}

/// Print a tabular breakdown of the proof size.
fn print_proof_size_breakdown(program_name: &str, proof: &Proof) {
    let total_proof_size = proof.encode().len();
    let proof_size_breakdown = break_down_proof_size(proof);
    let proof_size_breakdown = sort_hash_map_by_value_descending(proof_size_breakdown);

    eprintln!();
    eprintln!("Proof size breakdown for {program_name}:");
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

/// Create `num_iterations` many proofs for the program with the supplied source
/// code and public & private input, summing up the lengths of all proofs.
fn sum_of_proof_lengths_for_source_code(
    program_and_input: &ProgramAndInput,
    num_iterations: u64,
) -> ProofSize {
    let mut sum_of_proof_lengths = 0;
    for _ in 0..num_iterations {
        let (_, _, proof) = prove_program(
            program_and_input.program.clone(),
            program_and_input.public_input.clone(),
            program_and_input.non_determinism.clone(),
        )
        .unwrap();
        sum_of_proof_lengths += proof.encode().len();
    }
    ProofSize(sum_of_proof_lengths as f64)
}

/// Given the name and source for some program, generate a proof for its correct
/// execution and a benchmark ID for that proof. The benchmark ID contains the
/// length of the FRI domain.
fn generate_proof_and_benchmark_id(
    program_name: &str,
    program_and_input: &ProgramAndInput,
) -> (Proof, BenchmarkId) {
    let (stark, _, proof) = prove_program(
        program_and_input.program.clone(),
        program_and_input.public_input.clone(),
        program_and_input.non_determinism.clone(),
    )
    .unwrap();
    let log_2_fri_domain_length = log_2_fri_domain_length(stark, &proof);
    let benchmark_id = BenchmarkId::new(program_name, log_2_fri_domain_length);
    (proof, benchmark_id)
}

/// Benchmark the proof size and print a breakdown of the proof's size.
fn generate_statistics_for_program(
    benchmark_group: &mut BenchmarkGroup<ProofSize>,
    program_name: &str,
    program: &ProgramAndInput,
) {
    let (proof, benchmark_id) = generate_proof_and_benchmark_id(program_name, program);
    print_proof_size_breakdown(program_name, &proof);
    benchmark_proof_size(benchmark_group, benchmark_id, program);
}

/// Benchmark the proof size for the given program.
fn benchmark_proof_size(
    benchmark_group: &mut BenchmarkGroup<ProofSize>,
    benchmark_id: BenchmarkId,
    source: &ProgramAndInput,
) {
    benchmark_group.bench_function(benchmark_id, |bencher| {
        bencher.iter_custom(|num_iterations| {
            sum_of_proof_lengths_for_source_code(source, num_iterations)
        })
    });
}

fn generate_statistics_for_various_programs(criterion: &mut Criterion<ProofSize>) {
    let mut benchmark_group = criterion.benchmark_group("proof_size");

    generate_statistics_for_program(&mut benchmark_group, "halt", &program_halt());
    generate_statistics_for_program(&mut benchmark_group, "fib_100", &program_fib(100));
    generate_statistics_for_program(&mut benchmark_group, "fib_500", &program_fib(500));
    generate_statistics_for_program(&mut benchmark_group, "sudoku", &program_verify_sudoku());
}

fn proof_size_measurements() -> Criterion<ProofSize> {
    Criterion::default()
        .with_measurement(ProofSize(0.0))
        .sample_size(10)
}

criterion_group!(
    name = benches;
    config =  proof_size_measurements();
    targets = generate_statistics_for_various_programs
);
criterion_main!(benches);
