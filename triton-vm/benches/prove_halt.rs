use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;

use triton_vm::prelude::NonDeterminism;
use triton_vm::prelude::PublicInput;
use triton_vm::proof::Claim;
use triton_vm::stark::Stark;
use triton_vm::table::master_table::TableId;
use triton_vm::triton_program;

criterion_main!(benches);

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = prove_halt
}

/// cargo criterion --bench prove_halt
fn prove_halt(c: &mut Criterion) {
    let program = triton_program!(halt);
    let (aet, output) = program
        .trace_execution(PublicInput::default(), NonDeterminism::default())
        .unwrap();

    let stark = Stark::default();
    let claim = Claim::about_program(&program).with_output(output);
    let proof = stark.prove(&claim, &aet).unwrap();

    triton_vm::profiler::start("Prove Halt");
    c.bench_function("Prove Halt", |b| {
        b.iter(|| stark.prove(&claim, &aet).unwrap())
    });
    let profile = triton_vm::profiler::finish();

    let padded_height = proof.padded_height().unwrap();
    let fri = stark.derive_fri(padded_height).unwrap();
    let profile = profile
        .with_cycle_count(aet.height_of_table(TableId::Processor))
        .with_padded_height(padded_height)
        .with_fri_domain_len(fri.domain.length);
    eprintln!("{profile}");
}
