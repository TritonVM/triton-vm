use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use dev_util::ProgramToBench;
use triton_vm::config::CacheDecision;
use triton_vm::prelude::BFieldElement;
use triton_vm::prelude::bfe_array;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = prove_fib<1_000>,
);

fn prove_fib<const N: u64>(c: &mut Criterion) {
    let program = triton_vm::example_programs::FIBONACCI_SEQUENCE.clone();
    let program = ProgramToBench::new(format!("prove_fib_{N}"), program)
        .public_input(bfe_array![N])
        .assemble();

    let mut group = c.benchmark_group(&program.name);
    triton_vm::config::overwrite_lde_trace_caching_to(CacheDecision::Cache);
    group.bench_function("cache", |b| b.iter(|| program.prove()));
    triton_vm::config::overwrite_lde_trace_caching_to(CacheDecision::NoCache);
    group.bench_function("jit", |b| b.iter(|| program.prove()));
    group.finish();
}
