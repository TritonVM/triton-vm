use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use dev_util::ProgramToBench;
use triton_vm::prelude::*;

criterion_main!(benches);

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = prove_halt
}

/// cargo criterion --bench prove_halt
fn prove_halt(c: &mut Criterion) {
    let program = ProgramToBench::new("Prove Halt", triton_program!(halt)).assemble();
    triton_vm::profiler::start(&program.name);
    c.bench_function(&program.name, |b| b.iter(|| program.prove()));
    let profile = program.fill_performance_profile(triton_vm::profiler::finish());
    eprintln!("{profile}");
}
