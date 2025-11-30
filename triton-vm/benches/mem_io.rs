//! The benchmarked program performs a lot of useless RAM access.

use criterion::BatchSize;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;

use triton_vm::prelude::*;
use triton_vm::profiler::VMPerformanceProfile;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = mem_io
);

fn mem_io(criterion: &mut Criterion) {
    let mem_io = MemIOBench::new();
    let profile = mem_io.clone().performance_profile();
    eprintln!("{profile}");

    criterion.bench_function("Memory I/O", |b| {
        b.iter_batched(
            || mem_io.clone(),
            |mem_io| mem_io.prove(),
            BatchSize::SmallInput,
        )
    });
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
            addi -1
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

    fn prove(self) {
        triton_vm::prove_program(self.program, self.public_input, self.secret_input).unwrap();
    }

    fn performance_profile(self) -> VMPerformanceProfile {
        let claim = Claim::about_program(&self.program);
        let (aet, output) =
            VM::trace_execution(self.program, self.public_input, self.secret_input).unwrap();
        let claim = claim.with_output(output);

        let stark = Stark::default();
        triton_vm::profiler::start("Memory I/O");
        let proof = stark.prove(&claim, &aet).unwrap();
        let profile = triton_vm::profiler::finish();

        let trace_len = aet.height().height;
        let padded_height = proof.padded_height().unwrap();
        let fri = stark.fri(padded_height).unwrap();

        profile
            .with_cycle_count(trace_len)
            .with_padded_height(padded_height)
            .with_fri_domain_len(fri.domain.len())
    }
}
