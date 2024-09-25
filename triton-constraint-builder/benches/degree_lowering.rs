use constraint_circuit::ConstraintCircuitMonad;
use divan::black_box;
use triton_constraint_builder::Constraints;

fn main() {
    divan::main();
}

#[divan::bench(sample_count = 10)]
fn assemble_constraints() {
    black_box(Constraints::all());
}

#[divan::bench(sample_count = 10)]
fn initial_constraints(bencher: divan::Bencher) {
    bencher
        .with_inputs(|| {
            let degree_lowering_info = Constraints::default_degree_lowering_info();
            let constraints = Constraints::initial_constraints();
            (degree_lowering_info, constraints)
        })
        .bench_values(|(degree_lowering_info, mut constraints)| {
            black_box(ConstraintCircuitMonad::lower_to_degree(
                &mut constraints,
                degree_lowering_info,
            ));
        });
}

#[divan::bench(sample_count = 10)]
fn consistency_constraints(bencher: divan::Bencher) {
    bencher
        .with_inputs(|| {
            let degree_lowering_info = Constraints::default_degree_lowering_info();
            let constraints = Constraints::consistency_constraints();
            (degree_lowering_info, constraints)
        })
        .bench_values(|(degree_lowering_info, mut constraints)| {
            black_box(ConstraintCircuitMonad::lower_to_degree(
                &mut constraints,
                degree_lowering_info,
            ));
        });
}

#[divan::bench(sample_count = 1)]
fn transition_constraints(bencher: divan::Bencher) {
    bencher
        .with_inputs(|| {
            let degree_lowering_info = Constraints::default_degree_lowering_info();
            let constraints = Constraints::transition_constraints();
            (degree_lowering_info, constraints)
        })
        .bench_values(|(degree_lowering_info, mut constraints)| {
            black_box(ConstraintCircuitMonad::lower_to_degree(
                &mut constraints,
                degree_lowering_info,
            ));
        });
}

#[divan::bench(sample_count = 10)]
fn terminal_constraints(bencher: divan::Bencher) {
    bencher
        .with_inputs(|| {
            let degree_lowering_info = Constraints::default_degree_lowering_info();
            let constraints = Constraints::terminal_constraints();
            (degree_lowering_info, constraints)
        })
        .bench_values(|(degree_lowering_info, mut constraints)| {
            black_box(ConstraintCircuitMonad::lower_to_degree(
                &mut constraints,
                degree_lowering_info,
            ));
        });
}

#[divan::bench(sample_count = 1)]
fn degree_lower_all(bencher: divan::Bencher) {
    bencher
        .with_inputs(|| {
            let constraints = Constraints::all();
            let degree_lowering_info = Constraints::default_degree_lowering_info();
            (degree_lowering_info, constraints)
        })
        .bench_values(|(degree_lowering_info, mut constraints)| {
            black_box(
                constraints.lower_to_target_degree_through_substitutions(degree_lowering_info),
            );
        });
}
