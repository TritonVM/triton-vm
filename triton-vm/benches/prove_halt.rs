use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use triton_vm::profiler::TritonProfiler;
use triton_vm::proof::Claim;
use triton_vm::stark::Stark;
use triton_vm::stark::StarkHasher;
use triton_vm::stark::StarkParameters;
use triton_vm::triton_program;

/// cargo criterion --bench prove_halt
fn prove_halt(criterion: &mut Criterion) {
    let program = triton_program!(halt);
    let (aet, output) = program.trace_execution([].into(), [].into()).unwrap();

    let parameters = StarkParameters::default();
    let claim = Claim {
        input: vec![],
        program_digest: program.hash::<StarkHasher>(),
        output,
    };
    let mut profiler = Some(TritonProfiler::new("Prove Halt"));
    let proof = Stark::prove(parameters, &claim, &aet, &mut profiler);
    let mut profiler = profiler.unwrap();
    profiler.finish();

    let bench_id = BenchmarkId::new("ProveHalt", 0);
    let mut group = criterion.benchmark_group("prove_halt");
    group.sample_size(10);
    group.bench_function(bench_id, |bencher| {
        bencher.iter(|| {
            let _proof = Stark::prove(parameters, &claim, &aet, &mut None);
        });
    });
    group.finish();

    println!("Writing report ...");
    let padded_height = proof.padded_height().unwrap();
    let fri = Stark::derive_fri(parameters, padded_height).unwrap();
    let report = profiler
        .report()
        .with_cycle_count(aet.processor_trace.nrows())
        .with_padded_height(padded_height)
        .with_fri_domain_len(fri.domain.length);
    println!("{report}");
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = prove_halt
}

criterion_main!(benches);
