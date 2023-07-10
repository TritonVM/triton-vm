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
use triton_vm::vm::simulate;

/// cargo criterion --bench verify_halt
fn verify_halt(criterion: &mut Criterion) {
    let program = triton_program!(halt);

    let parameters = StarkParameters::default();
    let claim = Claim {
        input: vec![],
        program_digest: program.hash::<StarkHasher>(),
        output: vec![],
    };

    let (aet, _) = simulate(&program, vec![], vec![]).unwrap();
    let proof = Stark::prove(&parameters, &claim, &aet, &mut None);

    let mut profiler = Some(TritonProfiler::new("Verify Halt"));
    let verdict = Stark::verify(&parameters, &claim, &proof, &mut profiler).unwrap();
    assert!(verdict);

    let mut profiler = profiler.unwrap();
    profiler.finish();
    let padded_height = proof.padded_height().unwrap();
    let max_degree = Stark::derive_max_degree(padded_height, parameters.num_trace_randomizers);
    let fri = Stark::derive_fri(&parameters, max_degree);
    let report = profiler.report(
        Some(aet.processor_trace.nrows()),
        Some(padded_height),
        Some(fri.domain.length),
    );

    let bench_id = BenchmarkId::new("VerifyHalt", 0);
    let mut group = criterion.benchmark_group("verify_halt");
    group.sample_size(10);
    group.bench_function(bench_id, |bencher| {
        bencher.iter(|| {
            let _ = Stark::verify(&parameters, &claim, &proof, &mut None);
        });
    });
    group.finish();

    println!("{report}");
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = verify_halt
}

criterion_main!(benches);
