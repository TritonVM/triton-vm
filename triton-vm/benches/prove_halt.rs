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
use triton_vm::shared_tests::save_proof;
use triton_vm::stark::Stark;
use triton_vm::stark::StarkHasher;
use triton_vm::stark::StarkParameters;
use triton_vm::vm::simulate;

/// cargo criterion --bench prove_halt
fn prove_halt(criterion: &mut Criterion) {
    let mut maybe_profiler = Some(TritonProfiler::new("Prove Halt"));
    let mut report: Report = Report::placeholder();

    // stark object
    prof_start!(maybe_profiler, "parse program");
    let program = match Program::from_code("halt") {
        Err(e) => panic!("Cannot compile source code into program: {e}"),
        Ok(p) => p,
    };
    prof_stop!(maybe_profiler, "parse program");

    // witness
    prof_start!(maybe_profiler, "generate AET");
    let (aet, output) = simulate(&program, vec![], vec![]).unwrap();
    prof_stop!(maybe_profiler, "generate AET");

    let cycle_count = aet.processor_trace.nrows();
    let parameters = StarkParameters::default();
    let claim = Claim {
        input: vec![],
        program_digest: program.hash::<StarkHasher>(),
        output,
    };
    let proof = Stark::prove(&parameters, &claim, &aet, &mut maybe_profiler);

    let padded_height = proof.padded_height();
    let max_degree = Stark::derive_max_degree(padded_height, parameters.num_trace_randomizers);
    let fri = Stark::derive_fri(&parameters, max_degree);

    if let Some(profiler) = &mut maybe_profiler {
        profiler.finish();
        report = profiler.report(
            Some(cycle_count),
            Some(padded_height),
            Some(fri.domain.length),
        );
    };

    let bench_id = BenchmarkId::new("ProveHalt", 0);
    let mut group = criterion.benchmark_group("prove_halt");
    group.sample_size(10);
    group.bench_function(bench_id, |bencher| {
        bencher.iter(|| {
            let _ = Stark::prove(&parameters, &claim, &aet, &mut None);
        });
    });
    group.finish();

    // save proof
    let filename = "halt.tsp";
    if let Err(e) = save_proof(filename, proof) {
        println!("Error saving proof: {e:?}");
    }

    println!("{report}");
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = prove_halt
}

criterion_main!(benches);
