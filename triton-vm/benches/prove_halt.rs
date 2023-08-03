use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use triton_vm::prof_start;
use triton_vm::prof_stop;
use triton_vm::profiler::Report;
use triton_vm::profiler::TritonProfiler;
use triton_vm::proof::Claim;
use triton_vm::stark::Stark;
use triton_vm::stark::StarkHasher;
use triton_vm::stark::StarkParameters;
use triton_vm::triton_program;

/// cargo criterion --bench prove_halt
fn prove_halt(criterion: &mut Criterion) {
    let mut maybe_profiler = Some(TritonProfiler::new("Prove Halt"));
    let mut report: Report = Report::placeholder();

    prof_start!(maybe_profiler, "parse program");
    let program = triton_program!(halt);
    prof_stop!(maybe_profiler, "parse program");

    prof_start!(maybe_profiler, "generate AET");
    let (aet, output) = program.trace_execution([].into(), [].into()).unwrap();
    prof_stop!(maybe_profiler, "generate AET");

    let cycle_count = aet.processor_trace.nrows();
    let parameters = StarkParameters::default();
    let claim = Claim {
        input: vec![],
        program_digest: program.hash::<StarkHasher>(),
        output,
    };
    let proof = Stark::prove(&parameters, &claim, &aet, &mut maybe_profiler);

    let padded_height = proof.padded_height().unwrap();
    let fri = Stark::derive_fri(&parameters, padded_height);

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

    println!("{report}");
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = prove_halt
}

criterion_main!(benches);
