//! The benchmarked program performs a lot of (useless) RAM access.

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use dev_util::ProgramToBench;
use triton_vm::prelude::triton_program;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = mem_io
);

fn mem_io(c: &mut Criterion) {
    // Results in a trace length of 2^10. For 2^20, use 65_535.
    const NUM_ITERATIONS: u64 = 63;

    let program = triton_program! {
        dup 15 dup 15 dup 15 dup 15 dup 15
        push {NUM_ITERATIONS}
        call write_then_read_digest_loop
        halt

        // INVARIANT: _ [Digest; 5] i
        write_then_read_digest_loop:
            dup 0 push 0 eq skiz return
            write_mem 5 read_mem 5
            addi -1
            recurse
    };
    let mem_io = ProgramToBench::new(format!("mem_io_{NUM_ITERATIONS}"), program).assemble();

    triton_vm::profiler::start(&mem_io.name);
    c.bench_function("Memory I/O", |b| b.iter(|| mem_io.prove()));
    let profile = mem_io.fill_performance_profile(triton_vm::profiler::finish());
    eprintln!("{profile}");
}
