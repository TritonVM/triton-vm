//! Peak-memory demo of the memory-light commitment path (any backend).
//!     ROWS_LOG=16 COLS=64 COSETS=8 CHUNK_MB=8 cargo run --release --example memory
use rand::{rngs::StdRng, Rng, SeedableRng};
use triton_gpu::{Config, Gpu};
use twenty_first::math::traits::PrimitiveRootOfUnity;
use twenty_first::prelude::*;

fn env(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn main() -> triton_gpu::Result<()> {
    let (rows, cols, cosets) = (
        1usize << env("ROWS_LOG", 16),
        env("COLS", 64),
        env("COSETS", 8),
    );
    let chunk = env("CHUNK_MB", 8) << 20;
    let mut r = StdRng::seed_from_u64(0);
    let interp: Vec<BFieldElement> = (0..rows * cols)
        .map(|_| BFieldElement::new(r.random_range(0..BFieldElement::P)))
        .collect();
    let rands: Vec<Vec<BFieldElement>> = (0..cols)
        .map(|_| (0..4).map(|_| BFieldElement::new(r.random())).collect())
        .collect();
    let gen = BFieldElement::primitive_root_of_unity((rows * cosets) as u64).unwrap();
    let gpu = Gpu::with_config(
        triton_gpu::cpu::CpuBackend::default(),
        Config { chunk_bytes: chunk },
    )?;
    let t = std::time::Instant::now();
    let d = gpu.lde_hash_cosets(
        &interp,
        rows,
        &rands,
        BFieldElement::generator(),
        gen,
        cosets,
    )?;
    println!(
        "rows {rows}, cols {cols}, cosets {cosets}: {} digests in {:.1}s",
        d.len(),
        t.elapsed().as_secs_f64()
    );
    println!(
        "full LDE table would be {} MB; trace table {} MB; chunk budget {} MB",
        (rows * cosets * cols * 8) >> 20,
        (rows * cols * 8) >> 20,
        chunk >> 20
    );
    Ok(())
}
