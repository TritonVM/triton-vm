use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;

use triton_vm::prelude::*;

/// cargo criterion --bench verify_halt
fn verify_halt(criterion: &mut Criterion) {
    let program = triton_program!(halt);

    let stark = Stark::default();
    let claim = Claim::about_program(&program);
    let (aet, _) = VM::trace_execution(program, [].into(), [].into()).unwrap();
    let proof = stark.prove(&claim, &aet).unwrap();

    triton_vm::profiler::start("Verify Halt");
    criterion.bench_function("Verify Halt", |b| {
        b.iter(|| stark.verify(&claim, &proof).unwrap())
    });
    let profile = triton_vm::profiler::finish();

    let padded_height = proof.padded_height().unwrap();
    let fri = stark.fri(padded_height).unwrap();
    let profile = profile
        .with_cycle_count(aet.processor_trace.nrows())
        .with_padded_height(padded_height)
        .with_fri_domain_len(fri.domain.len());
    eprintln!("{profile}");
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = verify_halt
}

criterion_main!(benches);
