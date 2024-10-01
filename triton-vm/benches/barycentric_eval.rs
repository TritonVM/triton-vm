use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BatchSize;
use criterion::Criterion;
use itertools::Itertools;
use rand::prelude::StdRng;
use rand::Rng;
use rand_core::SeedableRng;
use triton_vm::fri::barycentric_evaluate;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = barycentric_eval<11>,
              barycentric_eval<13>,
              barycentric_eval<15>,
              barycentric_eval<17>,
              barycentric_eval<19>,
              barycentric_eval<21>,
);

fn barycentric_eval<const LOG2_N: usize>(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(13902833756029401496);
    c.bench_function(&format!("barycentric_evaluation_(1<<{LOG2_N})"), |b| {
        b.iter_batched(
            || ((0..1 << LOG2_N).map(|_| rng.gen()).collect_vec(), rng.gen()),
            |(codeword, indeterminate)| barycentric_evaluate(&codeword, indeterminate),
            BatchSize::SmallInput,
        )
    });
}
