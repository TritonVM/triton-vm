use std::mem::MaybeUninit;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;
use ndarray::prelude::*;
use num_traits::Zero;
use twenty_first::prelude::*;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = init_array<{1 << 10}>,
              init_array<{1 << 12}>,
              init_array<{1 << 14}>,
              init_array<{1 << 16}>,
              init_array<{1 << 18}>,
              init_array<{1 << 20}>,
              init_array<{1 << 22}>,
              init_array<{1 << 24}>,
);

fn init_array<const N: usize>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("initialize_array_{N}"));
    group.bench_function("default", |b| b.iter(|| Array1::<XFieldElement>::zeros(N)));
    group.bench_function("set_len", |b| b.iter(set_len::<N>));
    group.bench_function("maybe_uninit", |b| b.iter(maybe_uninit::<N>));
    group.finish();
}

#[allow(clippy::uninit_vec)]
fn set_len<const N: usize>() {
    let mut the_vec = Vec::with_capacity(N);
    unsafe {
        the_vec.set_len(N);
    }
    let mut array = Array1::from_vec(the_vec);
    array.par_mapv_inplace(|_| XFieldElement::zero());
}

fn maybe_uninit<const N: usize>() {
    let mut array = Array1::<XFieldElement>::uninit(N);
    array.par_mapv_inplace(|_| MaybeUninit::new(XFieldElement::zero()));
    unsafe { array.assume_init() };
}
