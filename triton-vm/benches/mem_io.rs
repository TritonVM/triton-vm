//! The benchmarked program performs a lot of useless RAM access.

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;

use triton_vm::prelude::*;
use triton_vm::profiler::Report;
use triton_vm::profiler::TritonProfiler;
use triton_vm::table::master_table::TableId;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = mem_io
);

fn mem_io(criterion: &mut Criterion) {
    let mem_io = MemIOBench::new();
    let report = mem_io.timing_report();
    eprintln!("{report}");

    criterion.bench_function("Memory I/O", |b| b.iter(|| mem_io.prove()));
}

fn program() -> Program {
    const NUM_ITERATIONS_FOR_TRACE_LEN_2_POW_10: u64 = 63;

    // requires a powerful machine – not suited for CI
    const _NUM_ITERATIONS_FOR_TRACE_LEN_2_POW_20: u64 = 65_535;

    triton_program! {
        dup 15 dup 15 dup 15 dup 15 dup 15
        push {NUM_ITERATIONS_FOR_TRACE_LEN_2_POW_10}
        call write_then_read_digest_loop
        halt

        // INVARIANT: _ [Digest; 5] i
        write_then_read_digest_loop:
            dup 0 push 0 eq skiz return
            write_mem 5 read_mem 5
            push -1 add
            recurse
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct MemIOBench {
    program: Program,
    public_input: PublicInput,
    secret_input: NonDeterminism,
}

impl MemIOBench {
    fn new() -> Self {
        Self {
            program: program(),
            public_input: PublicInput::default(),
            secret_input: NonDeterminism::default(),
        }
    }

    fn prove(&self) {
        let public_input = self.public_input.clone();
        let secret_input = self.secret_input.clone();
        triton_vm::prove_program(&self.program, public_input, secret_input).unwrap();
    }

    fn timing_report(&self) -> Report {
        let mut profiler = TritonProfiler::new("Memory I/O");
        profiler.start("generate AET", Some("gen".into()));
        let (aet, output) = self
            .program
            .trace_execution(self.public_input.clone(), self.secret_input.clone())
            .unwrap();
        profiler.stop("generate AET");

        let claim = Claim::about_program(&self.program).with_output(output);

        let stark = Stark::default();
        let mut profiler = Some(profiler);
        let proof = stark.prove(&claim, &aet, &mut profiler).unwrap();
        let mut profiler = profiler.unwrap();

        let trace_len = aet.height_of_table(TableId::OpStack);
        let padded_height = proof.padded_height().unwrap();
        let fri = stark.derive_fri(padded_height).unwrap();

        profiler
            .report()
            .with_cycle_count(trace_len)
            .with_padded_height(padded_height)
            .with_fri_domain_len(fri.domain.length)
    }
}
