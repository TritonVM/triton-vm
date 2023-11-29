use criterion::*;
use triton_vm::example_programs;

criterion_main!(benches);

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = run_mmr_new_peak_calculation, trace_mmr_new_peak_calculation
}

fn run_mmr_new_peak_calculation(criterion: &mut Criterion) {
    let program = example_programs::CALCULATE_NEW_MMR_PEAKS_FROM_APPEND_WITH_SAFE_LISTS.clone();

    criterion.bench_function("Run finding new peaks for MMR", |bencher| {
        bencher.iter(|| {
            program.run([].into(), [].into()).unwrap();
        });
    });
}

fn trace_mmr_new_peak_calculation(criterion: &mut Criterion) {
    let program = example_programs::CALCULATE_NEW_MMR_PEAKS_FROM_APPEND_WITH_SAFE_LISTS.clone();

    criterion.bench_function("Trace execution of finding new peaks for MMR", |bencher| {
        bencher.iter(|| {
            program.trace_execution([].into(), [].into()).unwrap();
        });
    });
}
