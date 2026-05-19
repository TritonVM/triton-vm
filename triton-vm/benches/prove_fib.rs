use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use dev_util::ProgramToBench;
use dev_util::example_programs::fibonacci_sequence;
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
    let name = format!("Prove Fibonacci {FIBONACCI_INDEX}");
    let program = ProgramToBench::new(name, fibonacci_sequence())
        .public_input(bfe_vec![FIBONACCI_INDEX])
        .assemble();

    triton_vm::profiler::start(&program.name);
    c.bench_function(&program.name, |b| b.iter(|| program.prove()));
    let profile = program.fill_performance_profile(triton_vm::profiler::finish());
    eprintln!("{profile}");
}
