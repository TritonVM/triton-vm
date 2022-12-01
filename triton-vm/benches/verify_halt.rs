use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use triton_profiler::{
    prof_start, prof_stop,
    triton_profiler::{Report, TritonProfiler},
};
use triton_vm::{
    proof::Claim,
    shared_tests::{load_proof, proof_file_exists, save_proof},
    stark::{Stark, StarkParameters},
    vm::Program,
};

/// cargo criterion --bench verify_halt
fn verify_halt(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("verify_halt");
    group.sample_size(10); // runs

    let halt = BenchmarkId::new("VerifyHalt", 0);

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

    let filename = "halt.tsp";
    let proof = if proof_file_exists(filename) {
        match load_proof(filename) {
            Ok(p) => p,
            Err(e) => panic!("Could not load proof from disk: {:?}", e),
        }
    } else {
        let (aet, _, err) = program.simulate_no_input();
        if let Some(error) = err {
            panic!("The VM encountered the following problem: {}", error);
        }
        let proof = stark.prove(aet, &mut None);

        if let Err(e) = save_proof(filename, proof.clone()) {
            panic!("Problem! could not save proof to disk: {:?}", e);
        }
        proof
    };

    let result = stark.verify(proof.clone(), &mut None);
    if let Err(e) = result {
        panic!("The Verifier is unhappy! {}", e);
    }

    let mut maybe_profiler = Some(TritonProfiler::new("Verify Halt"));
    let mut report: Report = Report::placeholder();

    group.bench_function(halt, |bencher| {
        bencher.iter(|| {
            prof_start!(maybe_profiler, "verify");
            let _result = stark.verify(proof.clone(), &mut maybe_profiler);
            prof_stop!(maybe_profiler, "verify");

            if let Some(profiler) = maybe_profiler.as_mut() {
                profiler.finish(Some(proof.padded_height()));
                report = profiler.report();
            }
            maybe_profiler = None;
        });
    });

    group.finish();

    println!("Writing report ...");
    println!("{}", report);
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = verify_halt
}

criterion_main!(benches);
