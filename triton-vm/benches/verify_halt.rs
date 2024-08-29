use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use triton_vm::prelude::*;

/// cargo criterion --bench verify_halt
fn verify_halt(criterion: &mut Criterion) {
    let program = triton_program!(halt);

    let stark = Stark::default();
    let claim = Claim::about_program(&program);

    let (aet, _) = VM::trace_execution(&program, [].into(), [].into()).unwrap();
    let proof = stark.prove(&claim, &aet).unwrap();

    triton_vm::profiler::start("Verify Halt");
    stark.verify(&claim, &proof).unwrap();
    let profile = triton_vm::profiler::finish();

    let padded_height = proof.padded_height().unwrap();
    let fri = stark.derive_fri(padded_height).unwrap();
    let profile = profile
        .with_cycle_count(aet.processor_trace.nrows())
        .with_padded_height(padded_height)
        .with_fri_domain_len(fri.domain.length);

    let bench_id = BenchmarkId::new("VerifyHalt", 0);
    let mut group = criterion.benchmark_group("verify_halt");
    group.sample_size(10);
    group.bench_function(bench_id, |bencher| {
        bencher.iter(|| {
            let _ = stark.verify(&claim, &proof);
        });
    });
    group.finish();

    eprintln!("{profile}");
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = verify_halt
}

criterion_main!(benches);
