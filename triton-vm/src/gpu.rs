//! Optional GPU acceleration of the prover (feature `gpu`).
//!
//! Every function here returns `None` when no GPU is available, when GPU use
//! is switched off (environment variable [`ENV_VAR_GPU`] set to `off`), or
//! when a GPU operation fails. Callers then continue with the regular CPU
//! code, so enabling the feature never changes behavior, only speed. The GPU
//! results are bit-identical to the CPU results; proofs do not change.
//!
//! - `TVM_GPU=off`: never use the GPU
//! - `TVM_GPU=cpu-reference`: run the GPU code paths on the CPU reference
//!   backend (slow; for testing on machines without a GPU)
//! - unset / anything else: use Metal or CUDA if compiled in and present
//!
//! The mode can also be set per thread with [`set_mode_for_current_thread`].

use std::any::TypeId;
use std::cell::Cell;
use std::cell::RefCell;
use std::mem::MaybeUninit;

use triton_gpu::AnyGpu;
use triton_gpu::GpuField;
use twenty_first::prelude::*;

pub const ENV_VAR_GPU: &str = "TVM_GPU";

/// Which GPU backend the prover uses on the current thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuMode {
    /// Metal or CUDA if compiled in and present, otherwise the CPU code.
    Auto,
    /// Never use the GPU code paths.
    Off,
    /// Run the GPU code paths on the CPU reference backend. Slow; meant for
    /// testing the GPU integration on machines without a GPU.
    CpuReference,
}

thread_local! {
    static OVERRIDE: RefCell<Option<GpuMode>> = const { RefCell::new(None) };
    static INSTANCE: RefCell<Option<(GpuMode, Option<AnyGpu>)>> = const { RefCell::new(None) };
    static SUCCESSFUL_GPU_OPERATIONS: Cell<usize> = const { Cell::new(0) };
}

/// Set the GPU mode for the current thread, overriding [`ENV_VAR_GPU`].
/// `None` restores the default (from the environment).
pub fn set_mode_for_current_thread(mode: Option<GpuMode>) {
    OVERRIDE.with(|o| *o.borrow_mut() = mode);
}

/// The number of prover operations that ran on a GPU backend so far on the
/// current thread. Lets tests check that the GPU code paths were taken.
pub fn num_successful_gpu_operations() -> usize {
    SUCCESSFUL_GPU_OPERATIONS.with(Cell::get)
}

fn current_mode() -> GpuMode {
    if let Some(mode) = OVERRIDE.with(|o| *o.borrow()) {
        return mode;
    }
    let setting = std::env::var(ENV_VAR_GPU)
        .unwrap_or_default()
        .to_ascii_lowercase();
    match setting.as_str() {
        "off" | "0" | "false" | "no" => GpuMode::Off,
        "cpu-reference" => GpuMode::CpuReference,
        _ => GpuMode::Auto,
    }
}

fn instantiate(mode: GpuMode) -> Option<AnyGpu> {
    match mode {
        GpuMode::Off => None,
        GpuMode::CpuReference => triton_gpu::Gpu::new(triton_gpu::cpu::CpuBackend::default())
            .ok()
            .map(AnyGpu::Cpu),
        GpuMode::Auto => triton_gpu::best_hardware().ok(),
    }
}

/// Run `f` on this thread's GPU, if there is one. Errors fall back to `None`.
fn with_gpu<R>(f: impl FnOnce(&AnyGpu) -> triton_gpu::Result<R>) -> Option<R> {
    let mode = current_mode();
    INSTANCE.with(|instance| {
        let mut instance = instance.borrow_mut();
        if instance.as_ref().is_none_or(|(m, _)| *m != mode) {
            *instance = Some((mode, instantiate(mode)));
        }
        let gpu = instance.as_ref()?.1.as_ref()?;
        match f(gpu) {
            Ok(r) => {
                SUCCESSFUL_GPU_OPERATIONS.with(|n| n.set(n.get() + 1));
                Some(r)
            }
            Err(e) => {
                eprintln!("triton-vm: GPU operation failed, falling back to CPU: {e}");
                None
            }
        }
    })
}

/// Reinterpret a slice of `FF` as a slice of `T` if they are the same type.
fn same<FF: 'static, T: 'static>(s: &[FF]) -> Option<&[T]> {
    (TypeId::of::<FF>() == TypeId::of::<T>())
        // SAFETY: FF and T are the same type.
        .then(|| unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<T>(), s.len()) })
}

fn same_mut<FF: 'static, T: 'static>(s: &mut [FF]) -> Option<&mut [T]> {
    (TypeId::of::<FF>() == TypeId::of::<T>())
        // SAFETY: FF and T are the same type.
        .then(|| unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<T>(), s.len()) })
}

fn same_vecs<FF: 'static, T: 'static>(v: &[Vec<FF>]) -> Option<&[Vec<T>]> {
    (TypeId::of::<FF>() == TypeId::of::<T>())
        // SAFETY: FF and T are the same type, hence so are Vec<FF> and Vec<T>.
        .then(|| unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<Vec<T>>(), v.len()) })
}

/// Call `$body` with `$T` bound to the concrete field of `$FF`.
macro_rules! dispatch_field {
    ($FF:ty, |$T:ident| $body:expr) => {{
        if TypeId::of::<$FF>() == TypeId::of::<BFieldElement>() {
            type $T = BFieldElement;
            $body
        } else if TypeId::of::<$FF>() == TypeId::of::<XFieldElement>() {
            type $T = XFieldElement;
            $body
        } else {
            None
        }
    }};
}

/// The row digests of the FRI domain without materializing it: interpolate
/// the trace columns, low-degree extend coset by coset (with trace
/// randomizers), and hash every row on the GPU. Equivalent to the uncached
/// path of `MasterTable::hash_all_fri_domain_rows`.
///
/// `trace` is the column-major trace table (`num_rows` per column). It is
/// interpolated in place and restored before returning.
pub(crate) fn hash_all_fri_domain_rows<FF: 'static>(
    trace: &mut [FF],
    num_rows: usize,
    randomizers: &[Vec<FF>],
    fri_offset: BFieldElement,
    fri_generator: BFieldElement,
    num_cosets: usize,
) -> Option<Vec<Digest>> {
    dispatch_field!(FF, |T| {
        let trace = same_mut::<FF, T>(trace)?;
        let randomizers = same_vecs::<FF, T>(randomizers)?;
        with_gpu(|gpu| {
            run_cosets::<T>(
                gpu,
                trace,
                num_rows,
                randomizers,
                fri_offset,
                fri_generator,
                num_cosets,
            )
        })
    })
}

fn run_cosets<T: GpuField>(
    gpu: &AnyGpu,
    trace: &mut [T],
    num_rows: usize,
    randomizers: &[Vec<T>],
    fri_offset: BFieldElement,
    fri_generator: BFieldElement,
    num_cosets: usize,
) -> triton_gpu::Result<Vec<Digest>> {
    gpu.intt(trace, num_rows)?;
    let digests = gpu.lde_hash_cosets(
        trace,
        num_rows,
        randomizers,
        fri_offset,
        fri_generator,
        num_cosets,
    );
    // Restore the trace even if hashing failed.
    gpu.ntt(trace, num_rows)?;
    digests
}

/// Evaluate column polynomials (column-major coefficients, `m` per column)
/// on the coset `offset · ⟨ω_n⟩` into `out` (column-major, `n` per column).
pub(crate) fn evaluate_columns_into<FF: 'static>(
    coefficients: &[FF],
    m: usize,
    n: usize,
    offset: BFieldElement,
    out: &mut [MaybeUninit<FF>],
) -> Option<()> {
    dispatch_field!(FF, |T| {
        let coefficients = same::<FF, T>(coefficients)?;
        let out = same_mut::<MaybeUninit<FF>, MaybeUninit<T>>(out)?;
        with_gpu(|gpu| gpu.lde_into(coefficients, m, n, offset, None, out))
    })
}

/// `Tip5::hash_varlen` of every row of a contiguous column-major table.
pub(crate) fn hash_rows<FF: 'static>(table: &[FF], num_rows: usize) -> Option<Vec<Digest>> {
    dispatch_field!(FF, |T| {
        let table = same::<FF, T>(table)?;
        with_gpu(|gpu| gpu.hash_rows(table, num_rows))
    })
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use isa::triton_program;

    use super::*;
    use crate::config::CacheDecision;
    use crate::config::overwrite_lde_trace_caching_to;
    use crate::prelude::*;
    use crate::shared_tests::TestableProgram;
    use crate::stark::Prover;
    use crate::tests::test;

    fn prove_with(mode: GpuMode, cache: CacheDecision, program: &TestableProgram) -> Proof {
        set_mode_for_current_thread(Some(mode));
        overwrite_lde_trace_caching_to(cache);
        let (aet, output) = VM::trace_execution(
            program.program.clone(),
            program.public_input.clone(),
            program.non_determinism.clone(),
        )
        .unwrap();
        let claim = Claim::about_program(&program.program)
            .with_input(program.public_input.individual_tokens.clone())
            .with_output(output);
        let proof = Prover::new(program.stark)
            .set_randomness_seed_which_may_break_zero_knowledge([42; 32])
            .prove(&claim, &aet)
            .unwrap();
        set_mode_for_current_thread(None);
        assert!(program.stark.verify(&claim, &proof).is_ok());
        proof
    }

    /// The GPU code paths produce byte-identical proofs, with and without
    /// caching of the low-degree extended trace.
    #[macro_rules_attr::apply(test)]
    fn gpu_code_paths_produce_identical_proofs() {
        let small = TestableProgram::new(triton_program! {
            read_io 1 push 2 mul dup 0 push 7 add swap 1 pow
            push 17 mul write_io 1
            push 3 push 5 add push 1 split pop 2
            halt
        })
        .with_input(bfe_vec![5]);
        assert_identical_proofs(&small);
    }

    /// A longer trace, such that the NTTs need several passes on the GPU.
    #[macro_rules_attr::apply(test)]
    fn gpu_code_paths_produce_identical_proofs_for_longer_traces() {
        let looping = TestableProgram::new(triton_program! {
            read_io 1
            call loop
            write_io 1
            halt
            loop:
                dup 0 push 0 eq skiz return
                push -1 add
                recurse
        })
        .with_input(bfe_vec![1500]);
        assert_identical_proofs(&looping);
    }

    fn assert_identical_proofs(program: &TestableProgram) {
        for cache in [CacheDecision::NoCache, CacheDecision::Cache] {
            let cpu = prove_with(GpuMode::Off, cache, program);
            let before = num_successful_gpu_operations();
            let gpu = prove_with(GpuMode::CpuReference, cache, program);
            // Main and auxiliary table: interpolation + coset hashing without
            // cache; evaluation + row hashing with cache.
            let gpu_operations = num_successful_gpu_operations() - before;
            let expected = if cache == CacheDecision::Cache { 4 } else { 2 };
            assert!(
                gpu_operations >= expected,
                "GPU paths not taken ({cache:?})"
            );
            assert_eq!(cpu, gpu, "proofs differ ({cache:?})");
        }
    }
}
