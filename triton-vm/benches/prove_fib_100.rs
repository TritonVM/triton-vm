use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use triton_profiler::prof_start;
use triton_profiler::prof_stop;
use triton_profiler::triton_profiler::{Report, TritonProfiler};
use triton_vm::instruction::sample_programs;
use triton_vm::proof::Claim;
use triton_vm::stark::Stark;
use triton_vm::vm::Program;

/// cargo criterion --bench prove_fib_100
fn prove_fib_100(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("prove_fib_100");
    group.sample_size(10); // runs

    let fib_100 = BenchmarkId::new("ProveFib100", 0);

    let mut maybe_profiler = Some(TritonProfiler::new("Prove Fibonacci 100"));
    let mut report: Report = Report::placeholder();

    // stark object
    let program = match Program::from_code(sample_programs::FIBONACCI_VIT) {
        Err(e) => panic!("Cannot compile source code into program: {}", e),
        Ok(p) => p,
    };
    let input = vec![100_u64.into()];
    let (aet, output, err) = program.simulate(input.clone(), vec![]);
    if let Some(error) = err {
        panic!("The VM encountered the following problem: {}", error);
    }

    let claim = Claim {
        input,
        program: program.to_bwords(),
        output,
        padded_height: 0,
    };
    let stark = Stark::new(claim, Default::default());
    //start the profiler
    prof_start!(maybe_profiler, "prove");
    let proof = stark.prove(aet.clone(), &mut maybe_profiler);
    let padded_height = Some(proof.padded_height());

    prof_stop!(maybe_profiler, "prove");

    let cycle_count = Some(aet.processor_matrix.len());

    if let Some(profiler) = maybe_profiler.as_mut() {
        report = profiler.finish_and_report(cycle_count, padded_height);
    }
    //start the benchmarking
    group.bench_function(fib_100, |bencher| {
        bencher.iter(|| {
            // TODO 2: Remove profiler from benchmark:
            let _proof = stark.prove(aet.clone(), &mut None);
        });
    });

    group.finish();

    println!("Writing report ...");
    println!("{}", report);
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = prove_fib_100
}

criterion_main!(benches);
