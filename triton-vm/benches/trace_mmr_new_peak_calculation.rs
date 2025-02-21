use criterion::BatchSize;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use triton_vm::example_programs;
use triton_vm::prelude::VM;

criterion_main!(benches);

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = run_mmr_new_peak_calculation, trace_mmr_new_peak_calculation
}

fn run_mmr_new_peak_calculation(criterion: &mut Criterion) {
    let program = example_programs::CALCULATE_NEW_MMR_PEAKS_FROM_APPEND_WITH_SAFE_LISTS.clone();

    criterion.bench_function("Run finding new peaks for MMR", |bencher| {
        bencher.iter_batched(
            || program.clone(),
            |program| VM::run(program, [].into(), [].into()).unwrap(),
            BatchSize::SmallInput,
        );
    });
}

fn trace_mmr_new_peak_calculation(criterion: &mut Criterion) {
    let program = example_programs::CALCULATE_NEW_MMR_PEAKS_FROM_APPEND_WITH_SAFE_LISTS.clone();

    criterion.bench_function("Trace execution of finding new peaks for MMR", |bencher| {
        bencher.iter_batched(
            || program.clone(),
            |program| VM::trace_execution(program, [].into(), [].into()).unwrap(),
            BatchSize::SmallInput,
        );
    });
}
