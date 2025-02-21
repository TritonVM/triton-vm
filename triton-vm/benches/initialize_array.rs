use std::mem::MaybeUninit;

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use itertools::Itertools;
use ndarray::prelude::*;
use num_traits::Zero;
use rand::prelude::*;
use rayon::prelude::*;
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

fn init_array<const NUM_ROWS: usize>(c: &mut Criterion) {
    const NUM_COLS: usize = 50;
    const NUM_WRITE_INDICES: usize = 1000;

    let mut group = c.benchmark_group(format!("initialize_array_{NUM_ROWS}x{NUM_COLS}"));
    let mut rng = StdRng::seed_from_u64(0);
    let write_to_indices = (0..NUM_WRITE_INDICES)
        .map(|_| (rng.random_range(0..NUM_ROWS), rng.random_range(0..NUM_COLS)))
        .collect_vec();

    let mut matrix = Array2::zeros((0, 0).f());
    group.bench_function("zeros", |b| {
        b.iter(|| {
            matrix = Array2::zeros((NUM_ROWS, NUM_COLS).f());
            set_ones(&mut matrix, &write_to_indices);
        })
    });
    group.bench_function("into_shape", |b| {
        b.iter(|| {
            matrix = into_shape::<NUM_ROWS, NUM_COLS>();
            set_ones(&mut matrix, &write_to_indices);
        })
    });
    group.bench_function("set_len", |b| {
        b.iter(|| {
            matrix = set_len::<NUM_ROWS, NUM_COLS>();
            set_ones(&mut matrix, &write_to_indices);
        })
    });
    group.bench_function("maybe_uninit", |b| {
        b.iter(|| {
            matrix = maybe_uninit::<NUM_ROWS, NUM_COLS>();
            set_ones(&mut matrix, &write_to_indices);
        })
    });
    group.finish();
}

/// Memory allocation is a highly optimized operation, in which many system
/// components play [intricate and often non-obvious roles][alloc]. This method
/// makes sure the allocated memory is actually used, requiring all the
/// involved background machinery to go through _all_ the motions.
///
/// [alloc]: https://stackoverflow.com/questions/2688466
fn set_ones(matrix: &mut Array2<XFieldElement>, indices: &[(usize, usize)]) {
    assert!(!matrix.is_standard_layout());
    for &(r, c) in indices {
        matrix[[r, c]] = xfe!(1);
    }
}

fn into_shape<const NUM_ROWS: usize, const NUM_COLS: usize>() -> Array2<XFieldElement> {
    Array1::zeros(NUM_ROWS * NUM_COLS)
        .into_shape_with_order(((NUM_ROWS, NUM_COLS), ndarray::Order::F))
        .unwrap()
}

fn set_len<const NUM_ROWS: usize, const NUM_COLS: usize>() -> Array2<XFieldElement> {
    let mut vec = Vec::with_capacity(NUM_ROWS * NUM_COLS);
    vec.spare_capacity_mut()
        .par_iter_mut()
        .for_each(|x| *x = MaybeUninit::new(XFieldElement::zero()));
    unsafe {
        vec.set_len(NUM_ROWS * NUM_COLS);
    }
    Array2::from_shape_vec((NUM_ROWS, NUM_COLS).f(), vec).unwrap()
}

fn maybe_uninit<const NUM_ROWS: usize, const NUM_COLS: usize>() -> Array2<XFieldElement> {
    let mut matrix = Array2::uninit((NUM_ROWS, NUM_COLS).f());
    matrix.par_mapv_inplace(|_| MaybeUninit::new(XFieldElement::zero()));
    unsafe { matrix.assume_init() }
}
