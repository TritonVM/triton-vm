use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;

use triton_opcodes::program::Program;
use triton_profiler::triton_profiler::Report;
use triton_profiler::triton_profiler::TritonProfiler;
use triton_vm::proof::Claim;
use triton_vm::shared_tests::save_proof;
use triton_vm::stark::Stark;
use triton_vm::stark::StarkParameters;
use triton_vm::table::master_table::MasterBaseTable;
use triton_vm::vm::simulate_no_input;

/// cargo criterion --bench prove_halt
fn prove_halt(_criterion: &mut Criterion) {
    let mut maybe_profiler = Some(TritonProfiler::new("Prove Halt"));
    let mut report: Report = Report::placeholder();

    // stark object
    let program = match Program::from_code("halt") {
        Err(e) => panic!("Cannot compile source code into program: {}", e),
        Ok(p) => p,
    };

    // witness
    let (aet, output, err) = simulate_no_input(&program);
    if let Some(error) = err {
        panic!("The VM encountered the following problem: {}", error);
    }

    let code = program.to_bwords();
    let cycle_count = aet.processor_matrix.nrows();
    let padded_height = MasterBaseTable::padded_height(&aet, &code);
    let claim = Claim {
        input: vec![],
        program: code,
        output,
        padded_height,
    };
    let parameters = StarkParameters::default();
    let stark = Stark::new(claim, parameters);
    let proof = stark.prove(aet, &mut maybe_profiler);

    if let Some(profiler) = &mut maybe_profiler {
        profiler.finish();
        report = profiler.report(
            Some(cycle_count),
            Some(stark.claim.padded_height),
            Some(stark.fri.domain.length),
        );
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
