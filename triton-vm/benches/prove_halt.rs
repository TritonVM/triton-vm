use criterion::{criterion_group, criterion_main, Criterion};
use triton_profiler::triton_profiler::{Report, TritonProfiler};
use triton_vm::{
    proof::Claim,
    shared_tests::save_proof,
    stark::{Stark, StarkParameters},
    vm::Program,
};

/// cargo criterion --bench prove_halt
fn prove_halt(_criterion: &mut Criterion) {
    let mut maybe_profiler = Some(TritonProfiler::new("Prove Halt"));
    let mut report: Report = Report::placeholder();

    // stark object
    let program = match Program::from_code("halt") {
        Err(e) => panic!("Cannot compile source code into program: {}", e),
        Ok(p) => p,
    };
    let claim = Claim {
        input: vec![],
        program: program.to_bwords(),
        output: vec![],
        padded_height: 0,
    };
    let parameters = StarkParameters::default();
    let stark = Stark::new(claim, parameters);

    // witness
    let (aet, err, _) = program.simulate_no_input();
    if let Some(error) = err {
        panic!("The VM encountered the following problem: {}", error);
    }

    let proof = stark.prove(aet, &mut maybe_profiler);

    if let Some(profiler) = &mut maybe_profiler {
        profiler.finish();
        report = profiler.report();
    };

    // save proof
    let filename = "halt.tsp";
    if let Err(e) = save_proof(filename, proof) {
        println!("Error saving proof: {:?}", e);
    }

    println!("{}", report);
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = prove_halt
}

criterion_main!(benches);
