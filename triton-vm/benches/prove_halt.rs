use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use triton_vm::proof::Claim;
use triton_vm::stark::Stark;
use triton_vm::table::master_table::TableId;
use triton_vm::triton_program;

/// cargo criterion --bench prove_halt
fn prove_halt(criterion: &mut Criterion) {
    let program = triton_program!(halt);
    let (aet, output) = program.trace_execution([].into(), [].into()).unwrap();

    let stark = Stark::default();
    let claim = Claim::about_program(&program).with_output(output);
    triton_vm::profiler::start("Prove Halt");
    let proof = stark.prove(&claim, &aet).unwrap();
    let report = triton_vm::profiler::finish();

    let bench_id = BenchmarkId::new("ProveHalt", 0);
    let mut group = criterion.benchmark_group("prove_halt");
    group.sample_size(10);
    group.bench_function(bench_id, |bencher| {
        bencher.iter(|| {
            let _proof = stark.prove(&claim, &aet);
        });
    });
    group.finish();

    let padded_height = proof.padded_height().unwrap();
    let fri = stark.derive_fri(padded_height).unwrap();
    let report = report
        .with_cycle_count(aet.height_of_table(TableId::Processor))
        .with_padded_height(padded_height)
        .with_fri_domain_len(fri.domain.length);
    eprintln!("{report}");
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = prove_halt
}

criterion_main!(benches);
