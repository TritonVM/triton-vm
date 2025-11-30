use air::table::TableId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;

use triton_vm::prelude::*;

criterion_main!(benches);

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = prove_halt
}

/// cargo criterion --bench prove_halt
fn prove_halt(c: &mut Criterion) {
    let program = triton_program!(halt);
    let claim = Claim::about_program(&program);
    let (aet, output) =
        VM::trace_execution(program, PublicInput::default(), NonDeterminism::default()).unwrap();

    let stark = Stark::default();
    let claim = claim.with_output(output);
    let proof = stark.prove(&claim, &aet).unwrap();

    triton_vm::profiler::start("Prove Halt");
    c.bench_function("Prove Halt", |b| {
        b.iter(|| stark.prove(&claim, &aet).unwrap())
    });
    let profile = triton_vm::profiler::finish();

    let padded_height = proof.padded_height().unwrap();
    let fri = stark.fri(padded_height).unwrap();
    let profile = profile
        .with_cycle_count(aet.height_of_table(TableId::Processor))
        .with_padded_height(padded_height)
        .with_fri_domain_len(fri.domain.len());
    eprintln!("{profile}");
}
