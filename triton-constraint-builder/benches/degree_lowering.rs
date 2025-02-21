use constraint_circuit::ConstraintCircuitMonad;
use constraint_circuit::InputIndicator;
use criterion::BatchSize::SmallInput;
use criterion::BenchmarkGroup;
use criterion::Criterion;
use criterion::black_box;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::measurement::WallTime;
use triton_constraint_builder::Constraints;

criterion_main!(benches);
criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = assemble_constraints,
              degree_lower_constraint_types,
              degree_lower_all,
}

fn assemble_constraints(criterion: &mut Criterion) {
    criterion.bench_function("Assemble all constraints", |bencher| {
        bencher.iter(|| black_box(Constraints::all()))
    });
}

fn degree_lower_constraint_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("lower to target degree");
    degree_lower_constraints(&mut group, "init", Constraints::initial_constraints());
    degree_lower_constraints(&mut group, "cons", Constraints::consistency_constraints());
    degree_lower_constraints(&mut group, "tran", Constraints::transition_constraints());
    degree_lower_constraints(&mut group, "term", Constraints::terminal_constraints());
    group.finish();
}

fn degree_lower_constraints<II: InputIndicator>(
    group: &mut BenchmarkGroup<WallTime>,
    constraint_group_name: &str,
    constraints: Vec<ConstraintCircuitMonad<II>>,
) {
    let info = Constraints::default_degree_lowering_info();
    let bench_name = format!("Degree-lower {constraint_group_name} constraints");
    group.bench_function(bench_name, |bencher| {
        bencher.iter_batched(
            || constraints.clone(),
            |mut constr| black_box(ConstraintCircuitMonad::lower_to_degree(&mut constr, info)),
            SmallInput,
        )
    });
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
