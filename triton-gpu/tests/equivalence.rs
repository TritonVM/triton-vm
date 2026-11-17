//! Bit-for-bit equivalence with twenty-first / triton-vm's CPU code.
//!
//! The same generic checks run against every backend that is compiled in:
//! the CPU reference always (it executes the shared kernel code), and Metal /
//! CUDA when their features are enabled and a device is present.

use num_traits::{ConstOne, ConstZero};
use rand::{rngs::StdRng, Rng, SeedableRng};
use triton_gpu::{Backend, Gpu};
use twenty_first::math::ntt::{intt, ntt};
use twenty_first::math::traits::{ModPowU32, PrimitiveRootOfUnity};
use twenty_first::prelude::*;
use twenty_first::util_types::merkle_tree::MerkleTree;

fn rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn bfes(r: &mut StdRng, n: usize) -> Vec<BFieldElement> {
    (0..n)
        .map(|_| BFieldElement::new(r.random_range(0..BFieldElement::P)))
        .collect()
}

fn xfes(r: &mut StdRng, n: usize) -> Vec<XFieldElement> {
    (0..n)
        .map(|_| {
            XFieldElement::new(
                [0; 3].map(|_: u64| BFieldElement::new(r.random_range(0..BFieldElement::P))),
            )
        })
        .collect()
}

/// Edge values: 0, 1, P-1, and their neighbours.
fn edge_bfes(n: usize) -> Vec<BFieldElement> {
    let e = [
        0,
        1,
        2,
        BFieldElement::P - 1,
        BFieldElement::P - 2,
        1 << 32,
        (1 << 32) - 1,
        u32::MAX as u64 + 2,
    ];
    (0..n).map(|i| BFieldElement::new(e[i % e.len()])).collect()
}

fn check_ntt<B: Backend>(gpu: &Gpu<B>, log_ns: &[u32]) {
    let mut r = rng(1);
    for &log_n in log_ns {
        let n = 1 << log_n;
        for num_cols in [1, 3] {
            let input = if log_n % 2 == 0 {
                bfes(&mut r, n * num_cols)
            } else {
                edge_bfes(n * num_cols)
            };
            let mut expected = input.clone();
            expected.chunks_mut(n).for_each(ntt);
            let mut got = input.clone();
            gpu.ntt(&mut got, n).unwrap();
            assert_eq!(expected, got, "ntt, log_n = {log_n}, cols = {num_cols}");

            let mut expected_i = input.clone();
            expected_i.chunks_mut(n).for_each(intt);
            let mut got_i = input.clone();
            gpu.intt(&mut got_i, n).unwrap();
            assert_eq!(expected_i, got_i, "intt, log_n = {log_n}");

            gpu.ntt(&mut got_i, n).unwrap();
            assert_eq!(input, got_i, "roundtrip, log_n = {log_n}");
        }

        let xin = xfes(&mut r, 2 * n);
        let mut expected = xin.clone();
        expected.chunks_mut(n).for_each(ntt);
        let mut got = xin.clone();
        gpu.ntt(&mut got, n).unwrap();
        assert_eq!(expected, got, "xfe ntt, log_n = {log_n}");
        let mut expected = xin.clone();
        expected.chunks_mut(n).for_each(intt);
        let mut got = xin.clone();
        gpu.intt(&mut got, n).unwrap();
        assert_eq!(expected, got, "xfe intt, log_n = {log_n}");
    }
}

/// Copy of triton-vm's `evaluate_randomized_interpolant_into` (master_table.rs).
fn reference_randomized_lde<FF>(
    interpolant: &[FF],
    randomizer: &[FF],
    zerofier: BFieldElement,
    offset: BFieldElement,
    codeword: &mut [FF],
) where
    FF: twenty_first::math::traits::FiniteField
        + std::ops::MulAssign<BFieldElement>
        + std::ops::Mul<BFieldElement, Output = FF>,
{
    let (scaled, pad) = codeword.split_at_mut(interpolant.len());
    let mut power = BFieldElement::ONE;
    for (cell, &c) in scaled.iter_mut().zip(interpolant) {
        *cell = c * power;
        power *= offset;
    }
    pad.fill(FF::ZERO);
    let mut power = zerofier;
    for (cell, &c) in codeword.iter_mut().zip(randomizer) {
        *cell += c * power;
        power *= offset;
    }
    ntt(codeword);
}

fn check_lde<B: Backend>(gpu: &Gpu<B>) {
    let mut r = rng(2);
    for (log_m, log_n) in [(3, 3), (4, 6), (10, 12), (12, 13)] {
        let (m, n) = (1usize << log_m, 1usize << log_n);
        let num_cols = 3;
        let offset = BFieldElement::new(7);
        let zerofier = BFieldElement::new(r.random_range(0..BFieldElement::P));
        let interp = bfes(&mut r, m * num_cols);
        let rands: Vec<Vec<BFieldElement>> = (0..num_cols)
            .map(|j| bfes(&mut r, [0, 5, m][j % 3]))
            .collect();

        let mut expected = vec![BFieldElement::ZERO; n * num_cols];
        for j in 0..num_cols {
            reference_randomized_lde(
                &interp[j * m..(j + 1) * m],
                &rands[j],
                zerofier,
                offset,
                &mut expected[j * n..(j + 1) * n],
            );
        }
        let got = gpu
            .lde(&interp, m, n, offset, Some((&rands, zerofier)))
            .unwrap();
        assert_eq!(expected, got, "bfe lde m = {m}, n = {n}");

        // X-field, no randomizers
        let xi = xfes(&mut r, m * 2);
        let mut expected = vec![XFieldElement::ZERO; n * 2];
        for j in 0..2 {
            reference_randomized_lde(
                &xi[j * m..(j + 1) * m],
                &[],
                zerofier,
                offset,
                &mut expected[j * n..(j + 1) * n],
            );
        }
        let got = gpu.lde(&xi, m, n, offset, None).unwrap();
        assert_eq!(expected, got, "xfe lde m = {m}, n = {n}");
    }
}

fn check_hash_rows<B: Backend>(gpu: &Gpu<B>) {
    let mut r = rng(3);
    for (rows, cols) in [(1, 1), (7, 9), (64, 10), (33, 11), (128, 23)] {
        let t = bfes(&mut r, rows * cols);
        let columns: Vec<&[BFieldElement]> = t.chunks(rows).collect();
        let expected = Tip5::hash_rows_of_columns(&columns);
        assert_eq!(
            expected,
            gpu.hash_rows(&t, rows).unwrap(),
            "bfe rows {rows} x {cols}"
        );
        // also against plain hash_varlen
        let row0: Vec<_> = columns.iter().map(|c| c[0]).collect();
        assert_eq!(Tip5::hash_varlen(&row0), expected[0]);

        let x = xfes(&mut r, rows * cols);
        let xcols: Vec<&[XFieldElement]> = x.chunks(rows).collect();
        assert_eq!(
            Tip5::hash_rows_of_columns(&xcols),
            gpu.hash_rows(&x, rows).unwrap(),
            "xfe rows {rows} x {cols}"
        );
    }
    // edge values (degenerate-looking Montgomery words) in every slot
    let t = edge_bfes(16 * 20);
    let columns: Vec<&[BFieldElement]> = t.chunks(16).collect();
    assert_eq!(
        Tip5::hash_rows_of_columns(&columns),
        gpu.hash_rows(&t, 16).unwrap()
    );
}

fn check_merkle<B: Backend>(gpu: &Gpu<B>) {
    let mut r = rng(4);
    for log in [0, 1, 2, 5, 11, 12] {
        let n = 1usize << log;
        let leafs: Vec<Digest> = (0..n)
            .map(|_| Digest::new(bfes(&mut r, 5).try_into().unwrap()))
            .collect();
        let expected = MerkleTree::par_new(&leafs).unwrap();
        let got = gpu.merkle_tree(&leafs).unwrap();
        assert_eq!(expected.root(), got.root(), "merkle root, {n} leafs");
        for i in 1..2 * n {
            assert_eq!(expected.node(i).unwrap(), got.nodes[i], "node {i} of {n}");
        }
    }
}

fn check_lde_hash_cosets<B: Backend>(gpu: &Gpu<B>) {
    let mut r = rng(5);
    let (rows, cols) = (64usize, 13usize);
    let interp = bfes(&mut r, rows * cols);
    let rands: Vec<Vec<BFieldElement>> = (0..cols)
        .map(|j| bfes(&mut r, if j == 0 { rows } else { 3 }))
        .collect();
    check_lde_hash_cosets_for(gpu, &interp, &rands, rows, 4);

    let interp = xfes(&mut r, rows * 5);
    let rands: Vec<Vec<XFieldElement>> = (0..5).map(|j| xfes(&mut r, j * 7)).collect();
    check_lde_hash_cosets_for(gpu, &interp, &rands, rows, 2);
}

fn check_lde_hash_cosets_for<B: Backend, FF>(
    gpu: &Gpu<B>,
    interp: &[FF],
    rands: &[Vec<FF>],
    rows: usize,
    num_cosets: usize,
) where
    FF: triton_gpu::GpuField
        + twenty_first::math::traits::FiniteField
        + std::ops::MulAssign<BFieldElement>
        + std::ops::Mul<BFieldElement, Output = FF>
        + twenty_first::tip5::ColumnElement,
{
    let cols = interp.len() / rows;
    let fri_len = rows * num_cosets;
    let fri_generator = BFieldElement::primitive_root_of_unity(fri_len as u64).unwrap();
    let fri_offset = BFieldElement::generator();

    let mut expected = vec![Digest::default(); fri_len];
    for coset in 0..num_cosets {
        let off = fri_offset * fri_generator.mod_pow_u32(coset as u32);
        let z = off.mod_pow_u32(rows as u32) - BFieldElement::ONE;
        let mut table = vec![FF::ZERO; rows * cols];
        for j in 0..cols {
            reference_randomized_lde(
                &interp[j * rows..(j + 1) * rows],
                &rands[j],
                z,
                off,
                &mut table[j * rows..(j + 1) * rows],
            );
        }
        let columns: Vec<&[FF]> = table.chunks(rows).collect();
        for (i, d) in Tip5::hash_rows_of_columns(&columns).into_iter().enumerate() {
            expected[coset + i * num_cosets] = d;
        }
    }
    let got = gpu
        .lde_hash_cosets(interp, rows, rands, fri_offset, fri_generator, num_cosets)
        .unwrap();
    assert_eq!(expected, got);
}

/// With `TRITON_GPU_REQUIRE_HARDWARE=1`, a missing GPU fails the test
/// (with the backend's error, e.g. a shader compilation error) instead of
/// skipping it.
#[allow(dead_code)]
fn require_hardware() -> bool {
    std::env::var("TRITON_GPU_REQUIRE_HARDWARE").is_ok_and(|v| v == "1")
}

fn check_all<B: Backend>(gpu: &Gpu<B>, log_ns: &[u32]) {
    check_ntt(gpu, log_ns);
    check_lde(gpu);
    check_hash_rows(gpu);
    check_merkle(gpu);
    check_lde_hash_cosets(gpu);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_reference_matches_twenty_first() {
    use triton_gpu::cpu::CpuBackend;
    check_all(
        &Gpu::new(CpuBackend::default()).unwrap(),
        &(0..=14).chain([16]).collect::<Vec<_>>(),
    );
}

/// Small threadgroups force multi-pass plans (and the coalesced 2^c layout)
/// already at small sizes, which exercises every index mapping.
#[cfg(feature = "cpu")]
#[test]
fn cpu_reference_multipass_plans() {
    use triton_gpu::cpu::CpuBackend;
    for max_threads in [16, 32, 128] {
        let gpu = Gpu::new(CpuBackend { max_threads }).unwrap();
        check_ntt(&gpu, &(0..=13).collect::<Vec<_>>());
        check_lde(&gpu);
        check_lde_hash_cosets(&gpu);
    }
}

/// Tiny chunk budget: every operation streams column by column.
#[cfg(feature = "cpu")]
#[test]
fn cpu_reference_chunked_streaming() {
    use triton_gpu::{cpu::CpuBackend, Config};
    let gpu = Gpu::with_config(
        CpuBackend::default(),
        Config {
            chunk_bytes: 1,
            ..Default::default()
        },
    )
    .unwrap();
    check_ntt(&gpu, &[0, 3, 9]);
    check_lde(&gpu);
    check_hash_rows(&gpu);
    check_lde_hash_cosets(&gpu);
}

/// The hybrid CPU + GPU pipeline gives the same digests, with large and
/// with tiny column chunks (many pipeline steps, both buffers in use).
#[cfg(feature = "cpu")]
#[test]
fn cpu_reference_hybrid_pipeline() {
    use triton_gpu::{cpu::CpuBackend, Config, Hybrid};
    for chunk_bytes in [512 << 20, 1, 3 * 64 * 8 * 2] {
        let config = Config {
            chunk_bytes,
            hybrid: Hybrid::On,
        };
        check_lde_hash_cosets(&Gpu::with_config(CpuBackend::default(), config).unwrap());
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matches_twenty_first() {
    let b = match triton_gpu::cuda::CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) if require_hardware() => panic!("CUDA backend unavailable: {e}"),
        Err(e) => {
            eprintln!("no CUDA device ({e}), skipping");
            return;
        }
    };
    let gpu = Gpu::new(b).unwrap();
    check_all(&gpu, &(0..=20).collect::<Vec<_>>());
    let small = Gpu::with_config(
        triton_gpu::cuda::CudaBackend::new(0).unwrap(),
        triton_gpu::Config {
            chunk_bytes: 1,
            ..Default::default()
        },
    )
    .unwrap();
    check_lde_hash_cosets(&small);
    for hybrid in [triton_gpu::Hybrid::Off, triton_gpu::Hybrid::On] {
        let config = triton_gpu::Config {
            hybrid,
            ..Default::default()
        };
        let gpu = Gpu::with_config(triton_gpu::cuda::CudaBackend::new(0).unwrap(), config);
        check_lde_hash_cosets(&gpu.unwrap());
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_matches_twenty_first() {
    let b = match triton_gpu::metal::MetalBackend::new() {
        Ok(b) => b,
        Err(e) if require_hardware() => panic!("Metal backend unavailable: {e}"),
        Err(e) => {
            eprintln!("no Metal device ({e}), skipping");
            return;
        }
    };
    let gpu = Gpu::new(b).unwrap();
    check_all(&gpu, &(0..=20).collect::<Vec<_>>());
    let small = Gpu::with_config(
        triton_gpu::metal::MetalBackend::new().unwrap(),
        triton_gpu::Config {
            chunk_bytes: 1,
            ..Default::default()
        },
    )
    .unwrap();
    check_lde_hash_cosets(&small);
    for hybrid in [triton_gpu::Hybrid::Off, triton_gpu::Hybrid::On] {
        let config = triton_gpu::Config {
            hybrid,
            ..Default::default()
        };
        let gpu = Gpu::with_config(triton_gpu::metal::MetalBackend::new().unwrap(), config);
        check_lde_hash_cosets(&gpu.unwrap());
    }
}
