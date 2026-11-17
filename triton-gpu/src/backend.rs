//! The low-level backend interface: one method per GPU kernel.
//!
//! Every backend (CPU reference, CUDA, Metal) implements [`Backend`]. The
//! high-level operations in [`crate::Gpu`] are written once on top of it, so
//! the logic that is verified with the CPU reference is the same logic that
//! drives the GPUs.

use crate::Result;

/// Number of B-field elements per Tip5 sponge state.
pub const STATE_SIZE: usize = 16;
/// Number of B-field elements per digest.
pub const DIGEST_LEN: usize = 5;

/// A plan for one NTT pass. Mirrors `TgNttPass` in `kernels/common/tg_core.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NttPass {
    pub log_n: u32,
    pub stage: u32,
    pub k: u32,
    pub c: u32,
    pub lob: u32,
}

/// Where the elements of a batch of columns live in a buffer. Mirrors
/// `TgLayout`. Element `i` of virtual column `v` is at
/// `offset + (v / ncomp) * col_stride + v % ncomp + i * ncomp`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub col_stride: u64,
    pub ncomp: u32,
    /// Valid elements per column; reads past it yield zero.
    pub len: u32,
}

/// Fused pre-processing of the first NTT pass. Mirrors `TgPrescale`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Prescale {
    pub enabled: u32,
    pub has_rand: u32,
    pub zerofier: u64,
}

/// Device-side buffers of trace randomizers: flat coefficients, per-column
/// offsets (in u64s) and per-column lengths (in field elements).
pub struct Randomizers<'a, B: Backend + ?Sized> {
    pub coefficients: &'a B::Buffer,
    pub offsets: &'a B::Buffer,
    pub lengths: &'a B::Buffer,
}

/// A device buffer of `u64`s (raw Montgomery B-field elements).
pub trait DeviceBuffer {
    /// Length in `u64`s.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One method per kernel. All buffer arguments carry an `offset` in `u64`s.
///
/// GPU buffers are mutated through shared references: they are device memory,
/// ordering is guaranteed by the backend's single in-order queue. Methods are
/// asynchronous unless they say otherwise.
pub trait Backend {
    type Buffer: DeviceBuffer;

    /// Human-readable device name.
    fn name(&self) -> String;

    /// Maximum threads per threadgroup / block usable by the NTT pass kernel
    /// (a power of two, at most 1024).
    fn max_ntt_threads(&self) -> u32 {
        1024
    }

    /// Allocate `len` u64s. Contents are unspecified.
    fn alloc(&self, len: usize) -> Result<Self::Buffer>;
    /// Allocate `len` u32s (for per-column lengths).
    fn alloc_u32(&self, src: &[u32]) -> Result<Self::Buffer>;

    /// Copy host → device. May return before the copy is done, but `src` may
    /// be reused immediately.
    fn upload(&self, dst: &Self::Buffer, dst_offset: usize, src: &[u64]) -> Result<()>;
    /// Copy device → host. Blocks until all previously enqueued work is done.
    fn download(&self, src: &Self::Buffer, src_offset: usize, dst: &mut [u64]) -> Result<()>;
    /// Block until all enqueued work is done.
    fn synchronize(&self) -> Result<()>;

    /// Whether CPU and GPU share memory (unified memory, e.g. Apple silicon).
    /// Such backends profit from the hybrid CPU + GPU pipeline.
    fn unified_memory(&self) -> bool {
        false
    }

    /// A host pointer to the buffer's memory, if the CPU can access it
    /// directly. Accesses must not race with GPU work using the buffer:
    /// [`Self::wait`] for that work's ticket first.
    fn host_ptr(&self, _buf: &Self::Buffer) -> Option<*mut u64> {
        None
    }

    /// Start all enqueued work without waiting for it. Returns a ticket for
    /// [`Self::wait`].
    fn submit(&self) -> Result<u64> {
        self.synchronize()?;
        Ok(0)
    }

    /// Block until the work of `ticket`, and all work submitted before it,
    /// is done.
    fn wait(&self, _ticket: u64) -> Result<()> {
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn ntt_pass(
        &self,
        src: (&Self::Buffer, usize, Layout),
        dst: (&Self::Buffer, usize, Layout),
        num_vcols: u32,
        pass: NttPass,
        twiddles: &Self::Buffer,
        pre: Prescale,
        offset_powers: Option<&Self::Buffer>,
        rand: Option<&Randomizers<'_, Self>>,
    ) -> Result<()>;

    fn bitrev(
        &self,
        src: (&Self::Buffer, usize, Layout),
        dst: (&Self::Buffer, usize, Layout),
        num_vcols: u32,
        log_n: u32,
        scale: Option<u64>,
    ) -> Result<()>;

    fn powers(&self, out: &Self::Buffer, base: u64, n: u64) -> Result<()>;

    fn rows_init(&self, state: &Self::Buffer, num_rows: u64) -> Result<()>;
    fn rows_absorb(
        &self,
        state: &Self::Buffer,
        num_rows: u64,
        pos: u32,
        data: (&Self::Buffer, usize, Layout),
        num_vcols: u32,
    ) -> Result<()>;
    fn rows_finish(
        &self,
        state: &Self::Buffer,
        num_rows: u64,
        pos: u32,
        digests: &Self::Buffer,
    ) -> Result<()>;

    fn merkle_level(&self, nodes: &Self::Buffer, first: u64, count: u64) -> Result<()>;
    fn merkle_tail(&self, nodes: &Self::Buffer, m: u64) -> Result<()>;

    /// Tail threshold: levels with at most this many parents are done by one
    /// `merkle_tail` launch.
    fn merkle_tail_max(&self) -> u64 {
        1024
    }
}
