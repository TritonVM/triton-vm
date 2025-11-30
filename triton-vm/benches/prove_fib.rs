use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;

use triton_vm::example_programs::FIBONACCI_SEQUENCE;
use triton_vm::prelude::*;

const FIBONACCI_INDEX: u32 = 100;

criterion_main!(benches);
criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = prove_fib
}

/// cargo criterion --bench prove_fib
fn prove_fib(c: &mut Criterion) {
    let program = FIBONACCI_SEQUENCE.clone();
    let public_input = PublicInput::new(bfe_vec![FIBONACCI_INDEX]);
    let claim = Claim::about_program(&program).with_input(public_input.clone());

    let (aet, output) =
        VM::trace_execution(program, public_input, NonDeterminism::default()).unwrap();
    let claim = claim.with_output(output);

    let stark = Stark::default();
    let bench_id = format!("Prove Fibonacci {FIBONACCI_INDEX}");
    triton_vm::profiler::start(&bench_id);
    c.bench_function(&bench_id, |b| b.iter(|| stark.prove(&claim, &aet).unwrap()));
    let profile = triton_vm::profiler::finish();

    let padded_height = aet.padded_height();
    let fri = stark.fri(padded_height).unwrap();
    let profile = profile
        .with_cycle_count(aet.processor_trace.nrows())
        .with_padded_height(padded_height)
        .with_fri_domain_len(fri.domain.len());
    eprintln!("{profile}");
}
