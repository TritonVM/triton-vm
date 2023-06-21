use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use triton_opcodes::program::Program;
use triton_profiler::triton_profiler::TritonProfiler;
use twenty_first::shared_math::tip5::Tip5;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;

use triton_vm::proof::Claim;
use triton_vm::shared_tests::load_proof;
use triton_vm::shared_tests::proof_file_exists;
use triton_vm::shared_tests::save_proof;
use triton_vm::stark::Stark;
use triton_vm::stark::StarkParameters;
use triton_vm::vm::simulate;

/// cargo criterion --bench verify_halt
fn verify_halt(criterion: &mut Criterion) {
    let program = Program::from_code("halt")
        .map_err(|e| panic!("Cannot compile source code into program: {e}"))
        .unwrap();

    let parameters = StarkParameters::default();
    let claim = Claim {
        input: vec![],
        program_digest: Tip5::hash_varlen(&program.to_bwords()),
        output: vec![],
    };

    let filename = "halt.tsp";
    let mut maybe_cycle_count = None;

    let proof = if proof_file_exists(filename) {
        load_proof(filename)
            .map_err(|e| panic!("Could not load proof from disk: {e:?}"))
            .unwrap()
    } else {
        let (aet, _) = simulate(&program, vec![], vec![]).unwrap();
        maybe_cycle_count = Some(aet.processor_trace.nrows());
        let proof = Stark::prove(&parameters, &claim, &aet, &mut None);
        save_proof(filename, proof.clone())
            .map_err(|e| panic!("Problem! could not save proof to disk: {e:?}"))
            .unwrap();
        proof
    };

    let mut profiler = Some(TritonProfiler::new("Verify Halt"));
    let verdict = Stark::verify(&parameters, &proof, &mut profiler)
        .map_err(|e| panic!("The Verifier is unhappy! {e}"))
        .unwrap();
    assert!(verdict);

    let mut profiler = profiler.unwrap();
    profiler.finish();
    let padded_height = proof.padded_height(&parameters);
    let max_degree = Stark::derive_max_degree(padded_height, parameters.num_trace_randomizers);
    let fri = Stark::derive_fri(&parameters, max_degree);
    let report = profiler.report(
        maybe_cycle_count,
        Some(padded_height),
        Some(fri.domain.length),
    );

    let bench_id = BenchmarkId::new("VerifyHalt", 0);
    let mut group = criterion.benchmark_group("verify_halt");
    group.sample_size(10);
    group.bench_function(bench_id, |bencher| {
        bencher.iter(|| {
            let _ = Stark::verify(&parameters, &proof, &mut None);
        });
    });
    group.finish();

    println!("Writing report ...");
    println!("{report}");
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = verify_halt
}

criterion_main!(benches);
