use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use triton_profiler::triton_profiler::{Report, TritonProfiler};
use triton_vm::shared_tests::{parse_simulate_prove, test_halt};
use twenty_first::shared_math::b_field_element::BFieldElement;

fn prove_verify_halt(criterion: &mut Criterion) {
    let mut maybe_profiler = Some(TritonProfiler::new("Halt"));
    let mut report: Report = Report::placeholder();

    let mut group = criterion.benchmark_group("prove_verify_halt");
    group.sample_size(10); // runs
    let co_set_fri_offset = BFieldElement::generator();
    let code_with_input = test_halt();
    let halt = BenchmarkId::new("Halt", 0);
    group.bench_function(halt, |bencher| {
        bencher.iter(|| {
            let (stark, mut proof_stream) = parse_simulate_prove(
                &code_with_input.source_code,
                co_set_fri_offset,
                &code_with_input.input,
                &code_with_input.secret_input,
                &[],
                &mut maybe_profiler,
            );

            if let Some(profiler) = maybe_profiler.as_mut() {
                profiler.start("verify");
            }

            let result = stark.verify(&mut proof_stream);
            if let Err(e) = result {
                panic!("The Verifier is unhappy! {}", e);
            }
            assert!(result.unwrap());
            if let Some(profiler) = maybe_profiler.as_mut() {
                profiler.stop("verify");
            }

            if let Some(profiler) = maybe_profiler.as_mut() {
                profiler.finish();
                report = profiler.report();
            }
            maybe_profiler = None;
        });
    });

    group.finish();

    println!("{}", report);
}

criterion_group!(benches, prove_verify_halt);

criterion_main!(benches);
