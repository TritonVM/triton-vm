use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use twenty_first::prelude::*;

use triton_vm::config::CacheDecision;
use triton_vm::prelude::*;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = prove_fib<1_000>,
);

fn prove_fib<const N: u64>(c: &mut Criterion) {
    let stark = Stark::default();
    let program = triton_vm::example_programs::FIBONACCI_SEQUENCE.clone();
    let claim = Claim::about_program(&program).with_input(bfe_vec![N]);

    let public_input = PublicInput::from(bfe_array![N]);
    let non_determinism = NonDeterminism::default();
    let (aet, output) = VM::trace_execution(program, public_input, non_determinism).unwrap();
    let claim = claim.with_output(output);

    let mut group = c.benchmark_group(format!("prove_fib_{N}"));
    triton_vm::config::overwrite_lde_trace_caching_to(CacheDecision::Cache);
    group.bench_function("cache", |b| b.iter(|| stark.prove(&claim, &aet)));
    triton_vm::config::overwrite_lde_trace_caching_to(CacheDecision::NoCache);
    group.bench_function("jit", |b| b.iter(|| stark.prove(&claim, &aet)));
    group.finish();
}
