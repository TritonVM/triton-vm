//! GPU vs CPU (twenty-first, all cores) on prover-sized workloads.
//!
//!     cargo run --release --features metal --example bench   # Mac
//!     cargo run --release --features cuda  --example bench   # NVIDIA
//!
//! Every GPU result is also compared against the CPU result.

use std::time::Instant;

use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use twenty_first::math::ntt::ntt;
use twenty_first::prelude::*;
use twenty_first::util_types::merkle_tree::MerkleTree;

fn time<T>(label: &str, f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let r = f();
    let s = t.elapsed().as_secs_f64();
    println!("  {label:<34} {:>9.1} ms", s * 1e3);
    (r, s)
}

fn main() -> triton_gpu::Result<()> {
    let mut gpu = triton_gpu::best_available()?;
    println!("backend: {}\n", gpu.name());
    let mut rng = StdRng::seed_from_u64(0);
    let log_n: u32 = std::env::var("LOG_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let cols: usize = std::env::var("COLS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let n = 1usize << log_n;

    println!("NTT: {cols} columns x 2^{log_n}");
    let table: Vec<BFieldElement> = (0..n * cols)
        .map(|_| BFieldElement::new(rng.random_range(0..BFieldElement::P)))
        .collect();
    let mut cpu = table.clone();
    let (_, tc) = time("CPU twenty-first (rayon)", || {
        cpu.par_chunks_mut(n).for_each(ntt)
    });
    let mut g = table.clone();
    gpu.ntt(&mut g, n)?; // warm-up: shader/twiddle setup
    let mut g = table.clone();
    let (r, tg) = time("GPU", || gpu.ntt(&mut g, n));
    r?;
    assert_eq!(cpu, g, "NTT mismatch");
    println!("  speed-up {:.1}x, identical output\n", tc / tg);

    println!("Hash rows: 2^{log_n} rows x {cols} columns");
    let columns: Vec<&[BFieldElement]> = table.chunks(n).collect();
    let (cpu_d, tc) = time("CPU Tip5::hash_rows_of_columns", || {
        columns_par_hash(&columns, n)
    });
    let (gd, tg) = time("GPU", || gpu.hash_rows(&table, n));
    assert_eq!(cpu_d, gd?, "row hash mismatch");
    println!("  speed-up {:.1}x, identical output\n", tc / tg);

    println!("Merkle tree: 2^{log_n} leafs");
    let (mt, tc) = time("CPU MerkleTree::par_new", || {
        MerkleTree::par_new(&cpu_d).unwrap()
    });
    let (gt, tg) = time("GPU", || gpu.merkle_tree(&cpu_d));
    assert_eq!(mt.root(), gt?.root(), "Merkle mismatch");
    println!("  speed-up {:.1}x, identical root", tc / tg);

    commitment(&mut gpu, &mut rng)
}

/// The prover's table commitment: low-degree extend every coset and hash
/// all rows. CPU only, GPU only, and CPU + GPU at the same time.
fn commitment(gpu: &mut triton_gpu::AnyGpu, rng: &mut StdRng) -> triton_gpu::Result<()> {
    use triton_gpu::{Config, GpuField, Hybrid};
    use twenty_first::math::traits::{ModPowU32, PrimitiveRootOfUnity};

    let env = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let rows = 1usize << env("COMMIT_LOG_ROWS", 18);
    let cols = env("COMMIT_COLS", 64);
    let cosets = env("COMMIT_COSETS", 4);
    println!(
        "\nCommitment (LDE + row hashing): 2^{} rows x {cols} columns x {cosets} cosets",
        rows.ilog2()
    );
    let interp: Vec<BFieldElement> = (0..rows * cols)
        .map(|_| BFieldElement::new(rng.random_range(0..BFieldElement::P)))
        .collect();
    let rands: Vec<Vec<BFieldElement>> = (0..cols)
        .map(|_| {
            (0..4)
                .map(|_| BFieldElement::new(rng.random_range(0..BFieldElement::P)))
                .collect()
        })
        .collect();
    let generator = BFieldElement::primitive_root_of_unity((rows * cosets) as u64).unwrap();
    let offset = BFieldElement::generator();

    let (cpu, tc) = time("CPU only (twenty-first, all cores)", || {
        let mut all = vec![Digest::default(); rows * cosets];
        let mut table = vec![BFieldElement::new(0); rows * cols];
        for coset in 0..cosets {
            let off = offset * generator.mod_pow_u32(coset as u32);
            let z = off.mod_pow_u32(rows as u32) - BFieldElement::new(1);
            table.par_chunks_mut(rows).enumerate().for_each(|(j, c)| {
                BFieldElement::cpu_coset_lde(
                    &interp[j * rows..(j + 1) * rows],
                    &rands[j],
                    z,
                    off,
                    c,
                )
            });
            let columns: Vec<&[BFieldElement]> = table.chunks(rows).collect();
            for (i, d) in columns_par_hash(&columns, rows).into_iter().enumerate() {
                all[coset + i * cosets] = d;
            }
        }
        all
    });
    for (label, hybrid) in [
        ("GPU only", Hybrid::Off),
        ("CPU + GPU at the same time", Hybrid::On),
    ] {
        gpu.set_config(Config {
            hybrid,
            ..Default::default()
        });
        let (got, t) = time(label, || {
            gpu.lde_hash_cosets(&interp, rows, &rands, offset, generator, cosets)
        });
        assert_eq!(cpu, got?, "commitment mismatch ({label})");
        println!("  speed-up {:.1}x, identical digests", tc / t);
    }
    gpu.set_config(Config::default());
    Ok(())
}

fn columns_par_hash(columns: &[&[BFieldElement]], n: usize) -> Vec<Digest> {
    const CHUNK: usize = 1024;
    (0..n.div_ceil(CHUNK))
        .into_par_iter()
        .flat_map_iter(|c| {
            let r = c * CHUNK..((c + 1) * CHUNK).min(n);
            let sub: Vec<&[BFieldElement]> = columns.iter().map(|col| &col[r.clone()]).collect();
            Tip5::hash_rows_of_columns(&sub)
        })
        .collect()
}
