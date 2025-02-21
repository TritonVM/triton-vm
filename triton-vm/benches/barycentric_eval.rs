use criterion::BatchSize;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use rand::prelude::*;
use twenty_first::math::polynomial::barycentric_evaluate;
use twenty_first::prelude::XFieldElement;

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
            || {
                let codeword = (0..1 << LOG2_N).map(|_| rng.random()).collect();
                let indeterminate = rng.random();
                (codeword, indeterminate)
            },
            |(cw, ind): (Vec<XFieldElement>, XFieldElement)| barycentric_evaluate(&cw, ind),
            BatchSize::SmallInput,
        )
    });
}
