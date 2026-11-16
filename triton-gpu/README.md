# triton-gpu

GPU kernels for the [Triton VM](https://github.com/TritonVM/triton-vm) prover (used by Neptune Cash):
**Metal** for Apple silicon and **CUDA** for NVIDIA. They produce outputs **bit-identical** to
twenty-first / triton-vm, so proofs don't change.

| Operation | API | Replaces (CPU) |
|---|---|---|
| Batched NTT / iNTT of table columns (B- and X-field) | `Gpu::ntt`, `Gpu::intt` | `twenty_first::math::ntt::{ntt, intt}` per column |
| Coset low-degree extension with trace randomizers + zero padding fused in | `Gpu::lde` | `evaluate_randomized_interpolant_into` |
| Tip5 hash of every table row | `Gpu::hash_rows` | `Tip5::hash_rows_of_columns` |
| Merkle tree | `Gpu::merkle_tree` | `MerkleTree::par_new` |
| **Memory-light commitment**: interpolants in, FRI-domain row digests out | `Gpu::lde_hash_cosets` | `MasterTable::hash_all_fri_domain_rows` (uncached path) |

## Quick start

```bash
# Mac (Apple silicon)
cargo test --release --features metal          # equivalence tests on the GPU
cargo run  --release --features metal --example bench

# NVIDIA (needs the CUDA toolkit, 11.2+)
cargo test --release --features cuda
cargo run  --release --features cuda  --example bench
TRITON_GPU_CUDA_ARCH=89 cargo build --release --features cuda   # only build for your GPU (faster build)
```

```rust
let gpu = triton_gpu::best_available()?;   // Metal → CUDA → CPU reference
gpu.intt(&mut main_table, num_rows)?;      // column-major, all columns at once
let digests = gpu.lde_hash_cosets(&interpolants, num_rows, &randomizers,
                                  fri_offset, fri_generator, num_cosets)?;
let tree = gpu.merkle_tree(&digests)?;
```

## How it's verified

```bash
cargo test --release                              # shared kernel logic, native 64-bit arithmetic
TRITON_GPU_WIDE_MUL_32=1 cargo test --release     # the same, with Metal's 32-bit-piece arithmetic
```

All kernel logic lives in shared headers (`kernels/common/`) that compile as Metal Shading
Language, CUDA C++ **and** plain C++. The `cpu` backend (on by default) runs exactly the same
per-thread code in loops that emulate the GPU launch geometry (tiles, threads, barriers). The tests in
`tests/equivalence.rs` compare every operation with twenty-first bit for bit, including:

- NTT/iNTT for lengths 2^0…2^16, B- and X-field, random and edge values
- forced multi-pass NTT plans (small threadgroups) to exercise every index mapping
- LDE with and without randomizers and padding, against a copy of triton-vm's function
- row hashing (B/X-field, uneven column counts, degenerate-looking Montgomery words)
- Merkle trees with 1…4096 leafs, every node
- the full coset pipeline, including chunked streaming with a 1-byte memory budget

The same checks run against Metal / CUDA when those features are enabled (sizes up to 2^20).
Mutation tests (breaking one shift in the MDS or one index bit) make them fail, as they should.

Tip5's constants and its MDS function are generated from twenty-first's source by
`tools/gen_headers.py`, so they can't drift by hand-copying.

## GPU techniques used

**Both backends**
- Elements stay in twenty-first's Montgomery form: tables go in and out with zero conversion.
- NTT: decimation-in-frequency, **radix-2^k passes in shared/threadgroup memory** (up to 11 butterfly
  layers per global-memory round trip). Loads and stores come in runs of 8 contiguous words (coalescing).
  Per-stage twiddle tables are laid out contiguously.
- **Kernel fusion:** coset scaling, trace randomizer and zero padding happen inside the first NTT
  pass's loads. The iNTT's 1/n scaling happens inside the bit-reversal gather.
- Tip5: round constants in constant memory (uniform broadcast), lookup table in shared memory,
  sponge state in registers (no dynamic indexing). On CUDA (sm_80): **0 bytes of spills / stack** in every
  kernel (checked with `ptxas -v`).
- **Streaming sponge:** row hashing keeps a structure-of-arrays sponge state on the device, so tables are
  absorbed column chunk by column chunk. Device memory is bounded by `Config::chunk_bytes`, not by the
  table size.
- Merkle trees: one launch per large level, and the top of the tree (≤1024 parents) in a single launch.

**CUDA:** one non-blocking stream, stream-ordered pool allocator (`cudaMallocAsync`/`cudaFreeAsync`),
`__launch_bounds__`, `__restrict__` read-only inputs, optional page-locked host memory
(`CudaBackend::pin`) for full-bandwidth async DMA. Fat binary for sm_75/80/86/89/90 plus PTX for
newer GPUs.

**Metal:** 64-bit multiplication built from 32-bit pieces (`kernels/common/tg_arith.h`). Apple GPUs have
no native 64-bit integer multiplier, so the field multiplication forms its 128-bit product from four native
32×32 products, and Tip5's MDS constants are applied with three native 32-bit instructions each. This path is
verified here by building the CPU reference with it: `TRITON_GPU_WIDE_MUL_32=1 cargo test --release`.
Unified memory (`StorageModeShared`): no bus transfers and no second copy of the table.
Whole pipelines are encoded into **one command buffer** and committed once. The shader library is
compiled from source at startup (no Xcode CLI tools needed).

## Memory: why `lde_hash_cosets` is light

The prover's commitment needs the digest of every FRI-domain row. The FRI domain is `num_cosets`×
longer than the trace. This path extends each coset on the GPU, absorbs it straight into the row
sponges, and only returns digests (40 bytes/row). The low-degree-extended table never exists
in full, on the device or on the host. Device memory = chunk budget + 128 bytes/row.

## Integrating into triton-vm

The APIs take the same column-major layout triton-vm's master tables use (`ndarray` with `.f()`
order → `as_slice_memory_order()`). The natural hook points are in `triton-vm/src/table/master_table.rs`:

1. `intt` / `ntt` over `axis_iter_mut(COL_AXIS)` → `gpu.intt(table_slice, num_rows)` /
   `gpu.ntt(...)`.
2. The coset loop in `hash_all_fri_domain_rows` (LDE + `hash_rows_of_column_major_table`) →
   `gpu.lde_hash_cosets(...)`. The randomizers come from `trace_randomizer_for_column(i)`.
3. `MerkleTree::par_new(&digests)` → `gpu.merkle_tree(&digests)`. twenty-first has no public
   constructor from precomputed nodes yet. A small upstream patch (`MerkleTree::from_nodes_unchecked`)
   would let the GPU tree be used directly. Until then, the GPU tree can serve as a cross-check, or
   the root can be used where only the root is needed.

## Status / limits

- Verified here: shared kernel logic (bit-exact vs twenty-first, CPU-emulated), CUDA kernels
  compile to SASS without spills, Metal source passes a host-side syntax check. **Not run yet on real
  GPUs.** Run the `metal` / `cuda` tests first on your machine.
- Not yet offloaded: quotient computation, FRI folding, polynomial evaluation at out-of-domain
  points. Those are the next candidates for keeping the whole prover on the GPU.

## Layout

```
kernels/common/   shared kernel logic (prelude, field, Tip5, NTT, hashing, Merkle) + C ABI
kernels/cuda/     CUDA kernels and host launch code
kernels/metal/    Metal kernels
kernels/cpu/      CPU reference backend (same code, emulated launches)
src/              Rust: backend trait, CPU/CUDA/Metal backends, high-level Gpu API
tests/            bit-for-bit equivalence tests against twenty-first
examples/bench.rs GPU vs CPU benchmark with output checks
tools/            header generator (Tip5 constants + MDS from twenty-first)
```

License: MIT OR Apache-2.0 (like twenty-first and triton-vm).
