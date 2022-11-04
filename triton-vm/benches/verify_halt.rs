use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use triton_profiler::{
    prof_start, prof_stop,
    triton_profiler::{Report, TritonProfiler},
};
use triton_vm::shared_tests::{parse_simulate_prove, test_halt};
use twenty_first::shared_math::b_field_element::BFieldElement;

/// cargo criterion --bench verify_halt
fn verify_halt(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("verify_halt");
    group.sample_size(10); // runs
    let co_set_fri_offset = BFieldElement::generator();
    let code_with_input = test_halt();
    let halt = BenchmarkId::new("VerifyHalt", 0);

    let (stark, proof) = parse_simulate_prove(
        &code_with_input.source_code,
        co_set_fri_offset,
        &code_with_input.input,
        &code_with_input.secret_input,
        &[],
        &mut None,
    );

    let mut maybe_profiler = Some(TritonProfiler::new("Verify Halt"));
    let mut report: Report = Report::placeholder();

    let result = stark.verify(proof.clone());
    if let Err(e) = result {
        panic!("The Verifier is unhappy! {}", e);
    }

    group.bench_function(halt, |bencher| {
        bencher.iter(|| {
            prof_start!(maybe_profiler, "verify");
            let _result = stark.verify(proof.clone());
            prof_stop!(maybe_profiler, "verify");

            if let Some(profiler) = maybe_profiler.as_mut() {
                profiler.finish();
                report = profiler.report();
            }
            maybe_profiler = None;
        });
    });

    group.finish();

    println!("Writing report ...");
    println!("{}", report);
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = verify_halt
}

criterion_main!(benches);
