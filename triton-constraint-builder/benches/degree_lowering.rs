use criterion::BatchSize::SmallInput;
use criterion::Criterion;
use criterion::black_box;
use criterion::criterion_group;
use criterion::criterion_main;
use triton_constraint_builder::Constraints;

criterion_main!(benches);
criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = degree_lower_all,
}

fn degree_lower_all(criterion: &mut Criterion) {
    let info = Constraints::default_degree_lowering_info();
    criterion.bench_function("Degree-lower all constraints", |bencher| {
        bencher.iter_batched(
            Constraints::all,
            |mut c| black_box(c.lower_to_target_degree_through_substitutions(info)),
            SmallInput,
        )
    });
}
