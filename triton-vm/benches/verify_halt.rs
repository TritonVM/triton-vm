use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use triton_profiler::prof_start;
use triton_profiler::prof_stop;
use triton_profiler::triton_profiler::Report;
use triton_profiler::triton_profiler::TritonProfiler;

use triton_opcodes::program::Program;
use triton_vm::proof::Claim;
use triton_vm::shared_tests::load_proof;
use triton_vm::shared_tests::proof_file_exists;
use triton_vm::shared_tests::save_proof;
use triton_vm::stark::Stark;
use triton_vm::stark::StarkParameters;
use triton_vm::table::master_table::MasterBaseTable;
use triton_vm::vm::simulate_no_input;

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

    let instructions = program.to_bwords();
    let stark_parameters = StarkParameters::default();
    let filename = "halt.tsp";
    let mut maybe_cycle_count = None;
    let (proof, stark) = if proof_file_exists(filename) {
        let proof = match load_proof(filename) {
            Ok(p) => p,
            Err(e) => panic!("Could not load proof from disk: {:?}", e),
        };
        let padded_height = proof.padded_height();
        let claim = Claim {
            input: vec![],
            program: instructions,
            output: vec![],
            padded_height,
        };
        let stark = Stark::new(claim, stark_parameters);
        (proof, stark)
    } else {
        let (aet, output, err) = simulate_no_input(&program);
        if let Some(error) = err {
            panic!("The VM encountered the following problem: {}", error);
        }
        maybe_cycle_count = Some(aet.processor_matrix.nrows());
        let padded_height = MasterBaseTable::padded_height(&aet, &instructions);
        let claim = Claim {
            input: vec![],
            program: instructions,
            output,
            padded_height,
        };
        let stark = Stark::new(claim, stark_parameters);
        let proof = stark.prove(aet, &mut None);
        if let Err(e) = save_proof(filename, proof.clone()) {
            panic!("Problem! could not save proof to disk: {:?}", e);
        }
        (proof, stark)
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
                report = profiler.report(
                    maybe_cycle_count,
                    Some(stark.claim.padded_height),
                    Some(stark.fri.domain.length),
                );
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
