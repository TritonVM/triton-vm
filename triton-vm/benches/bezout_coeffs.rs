use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;
use num_traits::ConstOne;
use num_traits::Zero;
use twenty_first::prelude::*;

use triton_vm::prelude::*;
use triton_vm::table::ram_table::RamTable;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = current_design<10>,
              current_design<100>,
              current_design<1_000>,
              current_design<10_000>,
              with_xgcd<10>,
              with_xgcd<100>,
              with_xgcd<1_000>,
              with_xgcd<10_000>,
);

fn with_xgcd<const N: u64>(c: &mut Criterion) {
    let roots = unique_roots::<N>();
    let bench_id = format!("Bézout coefficients (XGCD) (degree {N})");
    c.bench_function(&bench_id, |b| b.iter(|| bezout_coeffs_xgcd(&roots)));
}

fn current_design<const N: u64>(c: &mut Criterion) {
    let roots = unique_roots::<N>();
    let bench_id = format!("Bézout coefficients (current design) (degree {N})");
    c.bench_function(&bench_id, |b| {
        b.iter(|| RamTable::bezout_coefficient_polynomials_coefficients(&roots))
    });
}

fn unique_roots<const N: u64>() -> Vec<BFieldElement> {
    (0..N).map(BFieldElement::new).collect()
}

fn bezout_coeffs_xgcd(
    unique_ram_pointers: &[BFieldElement],
) -> (Vec<BFieldElement>, Vec<BFieldElement>) {
    let linear_poly_with_root = |&r: &BFieldElement| Polynomial::new(vec![-r, BFieldElement::ONE]);

    let polynomial_with_ram_pointers_as_roots = unique_ram_pointers
        .iter()
        .map(linear_poly_with_root)
        .reduce(|accumulator, linear_poly| accumulator * linear_poly)
        .unwrap_or_else(Polynomial::zero);
    let formal_derivative = polynomial_with_ram_pointers_as_roots.formal_derivative();

    let (_, bezout_poly_0, bezout_poly_1) =
        Polynomial::xgcd(polynomial_with_ram_pointers_as_roots, formal_derivative);

    let mut coefficients_0 = bezout_poly_0.coefficients;
    let mut coefficients_1 = bezout_poly_1.coefficients;
    coefficients_0.resize(unique_ram_pointers.len(), bfe!(0));
    coefficients_1.resize(unique_ram_pointers.len(), bfe!(0));
    (coefficients_0, coefficients_1)
}
