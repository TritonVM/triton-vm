use std::mem::MaybeUninit;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;
use ndarray::prelude::*;
use num_traits::One;
use num_traits::Zero;
use rand::rngs::StdRng;
use rand::thread_rng;
use rand::Rng;
use rand::RngCore;
use rand::SeedableRng;
use twenty_first::prelude::*;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = init_array<{1 << 10}>,
              init_array<{1 << 12}>,
              init_array<{1 << 14}>,
              init_array<{1 << 16}>,
              init_array<{1 << 18}>,
              init_array<{1 << 20}>,
              init_array<{1 << 22}>,
);

fn init_array<const N: usize>(c: &mut Criterion) {
    const W: usize = 50;
    let mut group = c.benchmark_group(format!("initialize_array_{N}x{W}"));
    let mut rng: StdRng = SeedableRng::from_seed(thread_rng().gen());
    let mut matrix = Array2::<XFieldElement>::zeros((1, 1).f());
    group.bench_function("zeros", |b| {
        b.iter(|| {
            matrix = Array2::<XFieldElement>::zeros((N, W).f());
            set_ones::<N, W>(&mut rng, &mut matrix);
        })
    });
    group.bench_function("into_shape", |b| {
        b.iter(|| {
            matrix = into_shape::<N, W>();
            set_ones::<N, W>(&mut rng, &mut matrix);
        })
    });
    group.bench_function("set_len", |b| {
        b.iter(|| {
            matrix = set_len::<N, W>();
            set_ones::<N, W>(&mut rng, &mut matrix);
        })
    });
    group.bench_function("maybe_uninit", |b| {
        b.iter(|| {
            matrix = maybe_uninit::<N, W>();
            set_ones::<N, W>(&mut rng, &mut matrix);
        })
    });
    group.finish();
}

fn set_ones<const H: usize, const W: usize>(rng: &mut StdRng, matrix: &mut Array2<XFieldElement>) {
    assert!(!matrix.is_standard_layout());
    for _ in 0..1000 {
        let r = (rng.next_u32() as usize) % H;
        let c = (rng.next_u32() as usize) % W;
        matrix[[r, c]] = xfe!(1);
    }
}

fn into_shape<const N: usize, const W: usize>() -> Array2<XFieldElement> {
    let array = Array1::<XFieldElement>::zeros(N * W);
    array.into_shape((W, N)).unwrap().reversed_axes()
}

#[allow(clippy::uninit_vec)]
fn set_len<const N: usize, const W: usize>() -> Array2<XFieldElement> {
    let mut the_vec = Vec::with_capacity(N * W);
    unsafe {
        the_vec.set_len(N * W);
    }
    let mut array = Array2::from_shape_vec((N, W).f(), the_vec).unwrap();
    array.par_mapv_inplace(|_| XFieldElement::zero());
    array
}

fn maybe_uninit<const N: usize, const W: usize>() -> Array2<XFieldElement> {
    let mut array = Array2::<XFieldElement>::uninit((N, W).f());
    array.par_mapv_inplace(|_| MaybeUninit::new(XFieldElement::zero()));
    unsafe { array.assume_init() }
}
