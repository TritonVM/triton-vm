use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use dev_util::ProgramToBench;
use triton_vm::example_programs::FIBONACCI_SEQUENCE;
use triton_vm::prelude::*;

const FIBONACCI_INDEX: u32 = 100;

criterion_main!(benches);
criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = prove_fib
}

/// cargo criterion --bench prove_fib
fn prove_fib(c: &mut Criterion) {
    let program = FIBONACCI_SEQUENCE.clone();
    let program = ProgramToBench::new(format!("Prove Fibonacci {FIBONACCI_INDEX}"), program)
        .public_input(bfe_vec![FIBONACCI_INDEX])
        .assemble();

    triton_vm::profiler::start(&program.name);
    c.bench_function(&program.name, |b| b.iter(|| program.prove()));
    let profile = program.fill_performance_profile(triton_vm::profiler::finish());
    eprintln!("{profile}");
}
