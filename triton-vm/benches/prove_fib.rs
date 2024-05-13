use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use triton_vm::aet::AlgebraicExecutionTrace;
use triton_vm::example_programs::FIBONACCI_SEQUENCE;
use triton_vm::prelude::*;
use triton_vm::profiler::VMPerformanceProfile;

const FIBONACCI_INDEX: BFieldElement = BFieldElement::new(100);

/// cargo criterion --bench prove_fib
fn prove_fib(criterion: &mut Criterion) {
    let (claim, aet) = trace_execution();
    fib_benchmark_group(criterion, &claim, &aet);

    let profile = prover_performance_profile(&claim, &aet);
    eprintln!("{profile}");
}

fn fib_benchmark_group(criterion: &mut Criterion, claim: &Claim, aet: &AlgebraicExecutionTrace) {
    let bench_id_name = format!("ProveFib{FIBONACCI_INDEX}");
    let bench_group_name = format!("prove_fib_{FIBONACCI_INDEX}");
    let bench_id = BenchmarkId::new(bench_id_name, 0);
    let mut group = criterion.benchmark_group(bench_group_name);

    let stark = Stark::default();
    group.bench_function(bench_id, |bencher| bencher.iter(|| stark.prove(claim, aet)));
    group.finish();
}

fn prover_performance_profile(
    claim: &Claim,
    aet: &AlgebraicExecutionTrace,
) -> VMPerformanceProfile {
    let profile_name = format!("Prove Fibonacci {FIBONACCI_INDEX}");
    triton_vm::profiler::start(profile_name);
    let stark = Stark::default();
    let proof = stark.prove(claim, aet).unwrap();
    let profile = triton_vm::profiler::finish();

    let padded_height = proof.padded_height().unwrap();
    let fri = stark.derive_fri(padded_height).unwrap();
    profile
        .with_cycle_count(aet.processor_trace.nrows())
        .with_padded_height(padded_height)
        .with_fri_domain_len(fri.domain.length)
}

fn trace_execution() -> (Claim, AlgebraicExecutionTrace) {
    let program = FIBONACCI_SEQUENCE.clone();
    let public_input = PublicInput::from([FIBONACCI_INDEX]);
    let (aet, output) = program
        .trace_execution(public_input.clone(), [].into())
        .unwrap();

    let claim = Claim::about_program(&program)
        .with_input(public_input.individual_tokens)
        .with_output(output);
    (claim, aet)
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = prove_fib
}

criterion_main!(benches);
