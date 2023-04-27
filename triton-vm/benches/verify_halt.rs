use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use triton_opcodes::program::Program;
use twenty_first::shared_math::tip5::Tip5;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;

use triton_profiler::triton_profiler::Report;
use triton_profiler::triton_profiler::TritonProfiler;
use triton_vm::proof::Claim;
use triton_vm::shared_tests::load_proof;
use triton_vm::shared_tests::proof_file_exists;
use triton_vm::shared_tests::save_proof;
use triton_vm::stark::Stark;
use triton_vm::stark::StarkParameters;
use triton_vm::table::master_table::MasterBaseTable;
use triton_vm::vm::simulate;

/// cargo criterion --bench verify_halt
fn verify_halt(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("verify_halt");
    group.sample_size(10); // runs

    let halt = BenchmarkId::new("VerifyHalt", 0);

    // stark object
    let program = match Program::from_code("halt") {
        Err(e) => panic!("Cannot compile source code into program: {e}"),
        Ok(p) => p,
    };

    let program_digest = Tip5::hash(&program);
    let parameters = StarkParameters::default();
    let filename = "halt.tsp";
    let mut maybe_cycle_count = None;

    let (claim, proof) = if proof_file_exists(filename) {
        let proof = match load_proof(filename) {
            Ok(p) => p,
            Err(e) => panic!("Could not load proof from disk: {e:?}"),
        };
        let padded_height = proof.padded_height();
        let claim = Claim {
            input: vec![],
            program_digest,
            output: vec![],
            padded_height,
        };
        (claim, proof)
    } else {
        let (aet, output, err) = simulate(&program, vec![], vec![]);
        if let Some(error) = err {
            panic!("The VM encountered the following problem: {error}");
        }
        let output = output.iter().map(|x| x.value()).collect();
        maybe_cycle_count = Some(aet.processor_trace.nrows());
        let padded_height = MasterBaseTable::padded_height(&aet);
        let claim = Claim {
            input: vec![],
            program_digest,
            output,
            padded_height,
        };
        let proof = Stark::prove(&parameters, &claim, &aet, &mut None);
        if let Err(e) = save_proof(filename, proof.clone()) {
            panic!("Problem! could not save proof to disk: {e:?}");
        }
        (claim, proof)
    };

    let result = Stark::verify(&parameters, &claim, &proof, &mut None);
    if let Err(e) = result {
        panic!("The Verifier is unhappy! {e}");
    }

    let max_degree =
        Stark::derive_max_degree(claim.padded_height, parameters.num_trace_randomizers);
    let fri = Stark::derive_fri(&parameters, max_degree);

    let mut maybe_profiler = Some(TritonProfiler::new("Verify Halt"));
    let mut report: Report = Report::placeholder();

    group.bench_function(halt, |bencher| {
        bencher.iter(|| {
            let _result = Stark::verify(&parameters, &claim, &proof, &mut maybe_profiler);
            if let Some(profiler) = maybe_profiler.as_mut() {
                profiler.finish();
                report = profiler.report(
                    maybe_cycle_count,
                    Some(claim.padded_height),
                    Some(fri.domain.length),
                );
            }
            maybe_profiler = None;
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
