use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use triton_opcodes::program::Program;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::tip5::Tip5;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;

use triton_profiler::prof_start;
use triton_profiler::prof_stop;
use triton_profiler::triton_profiler::Report;
use triton_profiler::triton_profiler::TritonProfiler;
use triton_vm::proof::Claim;
use triton_vm::shared_tests::FIBONACCI_SEQUENCE;
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
    prof_start!(maybe_profiler, "parse program");
    let program = match Program::from_code(FIBONACCI_SEQUENCE) {
        Err(e) => panic!("Cannot compile source code into program: {e}"),
        Ok(p) => p,
    };
    prof_stop!(maybe_profiler, "parse program");
    let input = vec![100];
    let public_input = input.iter().map(|&e| BFieldElement::new(e)).collect();
    prof_start!(maybe_profiler, "generate AET");
    let (aet, output, err) = simulate(&program, public_input, vec![]);
    prof_stop!(maybe_profiler, "generate AET");
    if let Some(error) = err {
        panic!("The VM encountered the following problem: {error}");
    }

    let output = output.iter().map(|x| x.value()).collect();
    let padded_height = MasterBaseTable::padded_height(&aet);
    let claim = Claim {
        input,
        program_digest: Tip5::hash(&program),
        output,
        padded_height,
    };
    let stark = Stark::new(claim, Default::default());
    let _proof = stark.prove(aet.clone(), &mut maybe_profiler);

    if let Some(profiler) = maybe_profiler.as_mut() {
        profiler.finish();
        report = profiler.report(
            Some(aet.processor_trace.nrows()),
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
    println!("{report}");
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = prove_fib_100
}

criterion_main!(benches);
