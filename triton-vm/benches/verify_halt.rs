use std::{fs::File, io::Read};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use parity_scale_codec::Output;
use triton_profiler::{
    prof_start, prof_stop,
    triton_profiler::{Report, TritonProfiler},
};
use triton_vm::{
    proof::{Claim, Proof},
    stark::{Stark, StarkParameters},
    vm::Program,
};

/// cargo criterion --bench verify_halt
fn verify_halt(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("verify_halt");
    group.sample_size(10); // runs
    let halt = BenchmarkId::new("VerifyHalt", 0);

    // stark object
    let parameters = StarkParameters::default();
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
    let stark = Stark::new(claim, parameters);

    // if the proof was stored last time, reuse it
    let filename = "proofs/halt.tsp";
    let proof = if let Ok(mut file_handle) = File::open(filename) {
        let mut contents: Vec<u8> = vec![];
        if let Err(e) = file_handle.read(&mut contents) {
            panic!("Cannot read from file {}: {:?}", filename, e);
        }
        let proof: Proof = bincode::deserialize(&contents).expect("Cannot deserialize proof.");

        proof
    }
    // otherwise, create it and store it
    else {
        let (aet, err, _) = program.simulate_no_input();
        if let Some(error) = err {
            panic!("The VM encountered the following problem: {}", error);
        }
        let proof = stark.prove(aet, &mut None);

        let mut file_handle = match File::create(filename) {
            Ok(fh) => fh,
            Err(e) => panic!("Cannot write proof to disk at {}: {:?}", filename, e),
        };
        let binary = match bincode::serialize(&proof) {
            Ok(b) => b,
            Err(e) => panic!("Cannot serialize proof: {:?}", e),
        };
        file_handle.write(&binary);
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
                profiler.finish();
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
