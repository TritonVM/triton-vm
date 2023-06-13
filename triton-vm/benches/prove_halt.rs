use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;
use triton_opcodes::program::Program;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::tip5::Tip5;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;

use triton_profiler::prof_start;
use triton_profiler::prof_stop;
use triton_profiler::triton_profiler::Report;
use triton_profiler::triton_profiler::TritonProfiler;
use triton_vm::proof::Claim;
use triton_vm::shared_tests::save_proof;
use triton_vm::stark::Stark;
use triton_vm::stark::StarkParameters;
use triton_vm::table::master_table::MasterBaseTable;
use triton_vm::vm::simulate;

/// cargo criterion --bench prove_halt
fn prove_halt(_criterion: &mut Criterion) {
    let mut maybe_profiler = Some(TritonProfiler::new("Prove Halt"));
    let mut report: Report = Report::placeholder();

    // stark object
    prof_start!(maybe_profiler, "parse program");
    let program = match Program::from_code("halt") {
        Err(e) => panic!("Cannot compile source code into program: {e}"),
        Ok(p) => p,
    };
    prof_stop!(maybe_profiler, "parse program");

    // witness
    prof_start!(maybe_profiler, "generate AET");
    let (aet, output, err) = simulate(&program, vec![], vec![]);
    prof_stop!(maybe_profiler, "generate AET");
    if let Some(error) = err {
        panic!("The VM encountered the following problem: {error}");
    }

    let cycle_count = aet.processor_trace.nrows();
    let parameters = StarkParameters::default();
    let num_trace_randomizers = parameters.num_trace_randomizers;
    let padded_height = MasterBaseTable::padded_height(&aet, num_trace_randomizers);
    let padded_height = BFieldElement::new(padded_height as u64);
    let claim = Claim {
        input: vec![],
        program_digest: Tip5::hash(&program),
        output,
        padded_height,
    };
    let proof = Stark::prove(&parameters, &claim, &aet, &mut maybe_profiler);

    let max_degree = Stark::derive_max_degree(claim.padded_height(), num_trace_randomizers);
    let fri = Stark::derive_fri(&parameters, max_degree);

    if let Some(profiler) = &mut maybe_profiler {
        profiler.finish();
        report = profiler.report(
            Some(cycle_count),
            Some(claim.padded_height()),
            Some(fri.domain.length),
        );
    };

    // save proof
    let filename = "halt.tsp";
    if let Err(e) = save_proof(filename, proof) {
        println!("Error saving proof: {e:?}");
    }

    println!("{report}");
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = prove_halt
}

criterion_main!(benches);
