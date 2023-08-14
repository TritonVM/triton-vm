use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use triton_vm::aet::AlgebraicExecutionTrace;
use triton_vm::example_programs::FIBONACCI_SEQUENCE;
use triton_vm::profiler::Report;
use triton_vm::profiler::TritonProfiler;
use triton_vm::proof::Claim;
use triton_vm::stark::Stark;
use triton_vm::stark::StarkHasher;
use triton_vm::PublicInput;
use triton_vm::StarkParameters;

/// cargo criterion --bench prove_fib
fn prove_fib_100(criterion: &mut Criterion) {
    bench_prove_fib(criterion, 100);
}

fn prove_fib_12800(criterion: &mut Criterion) {
    bench_prove_fib(criterion, 12800);
}

fn bench_prove_fib(criterion: &mut Criterion, fibonacci_index: u64) {
    let (claim, aet) = trace_execution(fibonacci_index);
    fib_benchmark_group(criterion, &claim, &aet, fibonacci_index);

    let report = prover_timing_report(&claim, &aet, fibonacci_index);
    println!("Writing report ...");
    println!("{report}");
}

fn fib_benchmark_group(
    criterion: &mut Criterion,
    claim: &Claim,
    aet: &AlgebraicExecutionTrace,
    fibonacci_index: u64,
) {
    let bench_id_name = format!("ProveFib{fibonacci_index}");
    let bench_group_name = format!("prove_fib_{fibonacci_index}");
    let bench_id = BenchmarkId::new(bench_id_name, 0);
    let mut group = criterion.benchmark_group(bench_group_name);

    let parameters = StarkParameters::default();
    group.bench_function(bench_id, |bencher| {
        bencher.iter(|| {
            let _proof = Stark::prove(&parameters, claim, aet, &mut None);
        });
    });
    group.finish();
}

fn prover_timing_report(
    claim: &Claim,
    aet: &AlgebraicExecutionTrace,
    fibonacci_index: u64,
) -> Report {
    let profile_name = format!("Prove Fibonacci {fibonacci_index}");
    let parameters = StarkParameters::default();
    let mut profiler = Some(TritonProfiler::new(&profile_name));
    let proof = Stark::prove(&parameters, claim, aet, &mut profiler);
    let mut profiler = profiler.unwrap();
    profiler.finish();

    let padded_height = proof.padded_height().unwrap();
    let fri = Stark::derive_fri(&parameters, padded_height);
    profiler
        .report()
        .with_cycle_count(aet.processor_trace.nrows())
        .with_padded_height(padded_height)
        .with_fri_domain_len(fri.domain.length)
}

fn trace_execution(fibonacci_index: u64) -> (Claim, AlgebraicExecutionTrace) {
    let program = FIBONACCI_SEQUENCE.clone();
    let public_input: PublicInput = vec![fibonacci_index].into();
    let (aet, output) = program
        .trace_execution(public_input.clone(), [].into())
        .unwrap();

    let claim = Claim {
        input: public_input.individual_tokens,
        program_digest: program.hash::<StarkHasher>(),
        output,
    };
    (claim, aet)
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = prove_fib_100, prove_fib_12800
}

criterion_main!(benches);
