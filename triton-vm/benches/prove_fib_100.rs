use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use triton_opcodes::program::Program;
use triton_profiler::prof_start;
use triton_profiler::prof_stop;
use triton_profiler::triton_profiler::Report;
use triton_profiler::triton_profiler::TritonProfiler;
use triton_vm::proof::Claim;
use triton_vm::shared_tests::FIBONACCI_VIT;
use triton_vm::stark::Stark;
use triton_vm::table::master_table::MasterBaseTable;
use triton_vm::vm::simulate;

/// cargo criterion --bench prove_fib_100
fn prove_fib_100(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("prove_fib_100");
    group.sample_size(10); // runs

    let fib_100 = BenchmarkId::new("ProveFib100", 0);

    let mut maybe_profiler = Some(TritonProfiler::new("Prove Fibonacci 100"));
    let mut report: Report = Report::placeholder();

    // stark object
    let program = match Program::from_code(FIBONACCI_VIT) {
        Err(e) => panic!("Cannot compile source code into program: {}", e),
        Ok(p) => p,
    };
    let input = vec![100_u64.into()];
    let (aet, output, err) = simulate(&program, input.clone(), vec![]);
    if let Some(error) = err {
        panic!("The VM encountered the following problem: {}", error);
    }

    let instructions = program.to_bwords();
    let padded_height = MasterBaseTable::padded_height(&aet, &instructions);
    let claim = Claim {
        input,
        program: instructions,
        output,
        padded_height,
    };
    let stark = Stark::new(claim, Default::default());
    //start the profiler
    prof_start!(maybe_profiler, "prove");
    let _proof = stark.prove(aet.clone(), &mut maybe_profiler);
    prof_stop!(maybe_profiler, "prove");

    if let Some(profiler) = maybe_profiler.as_mut() {
        profiler.finish();
        report = profiler.report(
            Some(aet.processor_matrix.nrows()),
            Some(stark.claim.padded_height),
            Some(stark.fri.domain.length),
        );
    }
    //start the benchmarking
    group.bench_function(fib_100, |bencher| {
        bencher.iter(|| {
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
