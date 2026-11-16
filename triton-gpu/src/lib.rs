//! # triton-gpu
//!
//! GPU kernels (Metal for Apple silicon, CUDA for NVIDIA) for the heavy parts
//! of the [Triton VM](https://github.com/TritonVM/triton-vm) prover:
//!
//! - batched NTT / iNTT of table columns ([`Gpu::ntt`], [`Gpu::intt`])
//! - low-degree extension onto a coset, with trace randomizers and zero
//!   padding fused into the transform ([`Gpu::lde`])
//! - Tip5 hashing of table rows ([`Gpu::hash_rows`])
//! - Merkle trees ([`Gpu::merkle_tree`])
//! - the memory-light main path of the prover's master table commitment:
//!   interpolants in, row digests out, coset by coset and column chunk by
//!   column chunk, without ever materializing the low-degree extended table
//!   ([`Gpu::lde_hash_cosets`])
//!
//! Every result is bit-identical to twenty-first / triton-vm's CPU code, so
//! proofs do not change. The kernel logic is shared between the backends
//! (`kernels/common`) and verified against twenty-first through the CPU
//! reference backend (feature `cpu`), which runs the very same code.
//!
//! ```no_run
//! use twenty_first::prelude::*;
//!
//! # fn main() -> triton_gpu::Result<()> {
//! let gpu = triton_gpu::best_available()?;          // Metal on a Mac, CUDA on NVIDIA
//! let n = 1 << 16;
//! let mut columns: Vec<BFieldElement> = (0..4 * n as u64).map(BFieldElement::new).collect();
//! gpu.intt(&mut columns, n)?;                // 4 columns at once, in place
//! gpu.ntt(&mut columns, n)?;
//! # Ok(()) }
//! ```

pub mod backend;
mod ffi;
mod field;

#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use num_traits::ConstOne;
use twenty_first::math::traits::{ModPowU32, PrimitiveRootOfUnity};
use twenty_first::prelude::*;

pub use backend::{Backend, DeviceBuffer, Layout, NttPass, Prescale, Randomizers};
use backend::{DIGEST_LEN, STATE_SIZE};
pub use field::GpuField;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no GPU backend available (enable feature \"metal\" or \"cuda\")")]
    NoBackend,
    #[error("GPU backend error: {0}")]
    Backend(String),
    #[error("invalid input: {0}")]
    Input(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Tuning knobs.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Upper bound for the device memory used by one column chunk, including
    /// scratch space. Smaller means lighter; larger means fewer launches.
    pub chunk_bytes: usize,

    /// Whether [`Gpu::lde_hash_cosets`] splits the work between CPU and GPU.
    pub hybrid: Hybrid,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chunk_bytes: 512 << 20,
            hybrid: Hybrid::Auto,
        }
    }
}

/// The hybrid CPU + GPU pipeline of [`Gpu::lde_hash_cosets`].
///
/// The CPU cores compute the low-degree extension (NTTs) of one column chunk
/// while the GPU hashes the rows of the previous chunk, through two buffers
/// in shared memory. Each processor does what it is best at, at the same
/// time, and no data is copied between them. Needs a backend whose buffers
/// the CPU can access ([`Backend::host_ptr`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hybrid {
    /// On for unified-memory GPUs (Apple silicon), off otherwise.
    Auto,
    /// Always, if the backend's buffers are accessible to the CPU.
    On,
    /// Never: everything on the GPU.
    Off,
}

/// The result of [`Gpu::merkle_tree`]: all nodes in twenty-first's layout
/// (`nodes[1]` is the root, the leafs are `nodes[n..2n]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuMerkleTree {
    pub nodes: Vec<Digest>,
}

impl GpuMerkleTree {
    pub fn root(&self) -> Digest {
        self.nodes[1]
    }
    pub fn num_leafs(&self) -> usize {
        self.nodes.len() / 2
    }
}

/// Twiddle tables per (log₂ length, inverse), kept on the device.
type TwiddleCache<B> = RefCell<HashMap<(u32, bool), Rc<<B as Backend>::Buffer>>>;

/// High-level prover operations on top of a [`Backend`].
pub struct Gpu<B: Backend> {
    backend: B,
    config: Config,
    twiddles: TwiddleCache<B>,
    dummy: B::Buffer,
}

/// A boxed choice of backend, picked at run time.
pub enum AnyGpu {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(Gpu<metal::MetalBackend>),
    #[cfg(feature = "cuda")]
    Cuda(Gpu<cuda::CudaBackend>),
    #[cfg(feature = "cpu")]
    Cpu(Gpu<cpu::CpuBackend>),
}

macro_rules! dispatch {
    ($self:ident, $g:ident => $e:expr) => {
        match $self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            AnyGpu::Metal($g) => $e,
            #[cfg(feature = "cuda")]
            AnyGpu::Cuda($g) => $e,
            #[cfg(feature = "cpu")]
            AnyGpu::Cpu($g) => $e,
        }
    };
}

impl AnyGpu {
    pub fn set_config(&mut self, config: Config) {
        dispatch!(self, g => g.config = config)
    }
    pub fn name(&self) -> String {
        dispatch!(self, g => g.backend.name())
    }
    pub fn ntt<FF: GpuField>(&self, t: &mut [FF], n: usize) -> Result<()> {
        dispatch!(self, g => g.ntt(t, n))
    }
    pub fn intt<FF: GpuField>(&self, t: &mut [FF], n: usize) -> Result<()> {
        dispatch!(self, g => g.intt(t, n))
    }
    pub fn lde<FF: GpuField>(
        &self,
        i: &[FF],
        m: usize,
        n: usize,
        o: BFieldElement,
        r: Option<(&[Vec<FF>], BFieldElement)>,
    ) -> Result<Vec<FF>> {
        dispatch!(self, g => g.lde(i, m, n, o, r))
    }
    pub fn lde_into<FF: GpuField>(
        &self,
        i: &[FF],
        m: usize,
        n: usize,
        o: BFieldElement,
        r: Option<(&[Vec<FF>], BFieldElement)>,
        out: &mut [std::mem::MaybeUninit<FF>],
    ) -> Result<()> {
        dispatch!(self, g => g.lde_into(i, m, n, o, r, out))
    }
    pub fn hash_rows<FF: GpuField>(&self, t: &[FF], rows: usize) -> Result<Vec<Digest>> {
        dispatch!(self, g => g.hash_rows(t, rows))
    }
    pub fn merkle_tree(&self, leafs: &[Digest]) -> Result<GpuMerkleTree> {
        dispatch!(self, g => g.merkle_tree(leafs))
    }
    pub fn lde_hash_cosets<FF: GpuField>(
        &self,
        i: &[FF],
        rows: usize,
        r: &[Vec<FF>],
        o: BFieldElement,
        gen: BFieldElement,
        c: usize,
    ) -> Result<Vec<Digest>> {
        dispatch!(self, g => g.lde_hash_cosets(i, rows, r, o, gen, c))
    }
}

/// A real GPU (Metal, then CUDA) if one is compiled in and present; never the
/// CPU reference backend.
#[allow(unreachable_code)]
pub fn best_hardware() -> Result<AnyGpu> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if let Ok(b) = metal::MetalBackend::new() {
        return Ok(AnyGpu::Metal(Gpu::new(b)?));
    }
    #[cfg(feature = "cuda")]
    if let Ok(b) = cuda::CudaBackend::new(0) {
        return Ok(AnyGpu::Cuda(Gpu::new(b)?));
    }
    Err(Error::NoBackend)
}

/// The fastest backend compiled in and present on this machine:
/// Metal, then CUDA, then the CPU reference.
#[allow(unreachable_code)]
pub fn best_available() -> Result<AnyGpu> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if let Ok(b) = metal::MetalBackend::new() {
        return Ok(AnyGpu::Metal(Gpu::new(b)?));
    }
    #[cfg(feature = "cuda")]
    if let Ok(b) = cuda::CudaBackend::new(0) {
        return Ok(AnyGpu::Cuda(Gpu::new(b)?));
    }
    #[cfg(feature = "cpu")]
    return Ok(AnyGpu::Cpu(Gpu::new(cpu::CpuBackend::default())?));
    Err(Error::NoBackend)
}

/// Raw u64 of a B-field element (Montgomery form, as stored).
#[inline]
fn raw(x: BFieldElement) -> u64 {
    x.raw_u64()
}

/// Split the `log_n` NTT stages into passes that fit a threadgroup.
pub fn plan_passes(log_n: u32, max_threads: u32) -> Vec<NttPass> {
    let max_tile_log = (max_threads.max(16).ilog2() + 1).min(11);
    let mut passes = vec![];
    let mut s = 0;
    while s < log_n {
        let rem = log_n - s;
        let (k, c) = if rem <= max_tile_log {
            (rem, 0)
        } else {
            (max_tile_log - 3, 3)
        };
        passes.push(NttPass {
            log_n,
            stage: s,
            k,
            c,
            lob: rem - k,
        });
        s += k;
    }
    passes
}

impl<B: Backend> Gpu<B> {
    pub fn new(backend: B) -> Result<Self> {
        Self::with_config(backend, Config::default())
    }

    pub fn with_config(backend: B, config: Config) -> Result<Self> {
        let dummy = backend.alloc(1)?;
        Ok(Self {
            backend,
            config,
            twiddles: RefCell::new(HashMap::new()),
            dummy,
        })
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    fn twiddles(&self, log_n: u32, inverse: bool) -> Result<Rc<B::Buffer>> {
        if let Some(t) = self.twiddles.borrow().get(&(log_n, inverse)) {
            return Ok(t.clone());
        }
        let n = 1usize << log_n;
        let mut omega = BFieldElement::primitive_root_of_unity(n as u64)
            .ok_or_else(|| Error::Input(format!("no root of unity of order {n}")))?;
        if inverse {
            omega = omega.inverse();
        }
        let half = (n / 2).max(1);
        let mut pw = Vec::with_capacity(half);
        let mut acc = BFieldElement::ONE;
        for _ in 0..half {
            pw.push(acc);
            acc *= omega;
        }
        let mut tw = vec![0u64; n.max(1)];
        let mut h = 1;
        while h < n {
            let step = n / (2 * h);
            for t in 0..h {
                tw[h + t] = raw(pw[t * step]);
            }
            h *= 2;
        }
        let buf = self.backend.alloc(tw.len())?;
        self.backend.upload(&buf, 0, &tw)?;
        let buf = Rc::new(buf);
        self.twiddles
            .borrow_mut()
            .insert((log_n, inverse), buf.clone());
        Ok(buf)
    }

    /// The core transform, all on the device.
    ///
    /// Reads `num_vcols` virtual columns from `src` (applying the optional
    /// fused prescale), transforms them (length 2^log_n), and writes natural
    /// order results to `dst`. `scratch` needs `num_vcols << log_n` u64s.
    #[allow(clippy::too_many_arguments)]
    pub fn transform(
        &self,
        src: (&B::Buffer, usize, Layout),
        dst: (&B::Buffer, usize, Layout),
        num_vcols: u32,
        log_n: u32,
        inverse: bool,
        pre: Prescale,
        offset_powers: Option<&B::Buffer>,
        rand: Option<&Randomizers<'_, B>>,
        scratch: &B::Buffer,
    ) -> Result<()> {
        if num_vcols == 0 {
            return Ok(());
        }
        let n = 1u64 << log_n;
        if log_n == 0 {
            if pre.enabled != 0 {
                return Err(Error::Input(
                    "prescale on length-1 transforms is not supported".into(),
                ));
            }
            return self.backend.bitrev(src, dst, num_vcols, 0, None);
        }
        assert!(
            scratch.len() as u64 >= n * num_vcols as u64,
            "scratch too small"
        );
        let tw = self.twiddles(log_n, inverse)?;
        let dense = Layout {
            col_stride: n,
            ncomp: 1,
            len: n as u32,
        };
        let op = offset_powers.unwrap_or(&self.dummy);
        for (i, pass) in plan_passes(log_n, self.backend.max_ntt_threads())
            .into_iter()
            .enumerate()
        {
            if i == 0 {
                self.backend.ntt_pass(
                    src,
                    (scratch, 0, dense),
                    num_vcols,
                    pass,
                    &tw,
                    pre,
                    Some(op),
                    rand,
                )?;
            } else {
                self.backend.ntt_pass(
                    (scratch, 0, dense),
                    (scratch, 0, dense),
                    num_vcols,
                    pass,
                    &tw,
                    Prescale::default(),
                    Some(&self.dummy),
                    None,
                )?;
            }
        }
        let scale = inverse.then(|| raw(BFieldElement::new(n).inverse()));
        self.backend
            .bitrev((scratch, 0, dense), dst, num_vcols, log_n, scale)
    }

    fn check_table(len: usize, n: usize) -> Result<(usize, u32)> {
        if n == 0 || !n.is_power_of_two() || n > (1 << 31) {
            return Err(Error::Input(format!(
                "column length {n} is not a power of two ≤ 2^31"
            )));
        }
        if !len.is_multiple_of(n) {
            return Err(Error::Input(format!(
                "table length {len} is not a multiple of column length {n}"
            )));
        }
        Ok((len / n, n.ilog2()))
    }

    /// Columns per chunk such that `per_col_u64s` per column fits the budget.
    fn chunk_cols(&self, per_col_u64s: usize, total: usize) -> usize {
        (self.config.chunk_bytes / (per_col_u64s * 8).max(1)).clamp(1, total.max(1))
    }

    fn ntt_impl<FF: GpuField>(&self, table: &mut [FF], n: usize, inverse: bool) -> Result<()> {
        let (num_cols, log_n) = Self::check_table(table.len(), n)?;
        if num_cols == 0 {
            return Ok(());
        }
        let nc = FF::NCOMP as usize;
        let flat = FF::as_u64s_mut(table);
        let cols = self.chunk_cols(2 * n * nc, num_cols);
        let buf = self.backend.alloc(cols * n * nc)?;
        let scratch = self.backend.alloc(cols * n * nc)?;
        for chunk in flat.chunks_mut(cols * n * nc) {
            let cc = chunk.len() / (n * nc);
            let layout = Layout {
                col_stride: (n * nc) as u64,
                ncomp: nc as u32,
                len: n as u32,
            };
            self.backend.upload(&buf, 0, chunk)?;
            self.transform(
                (&buf, 0, layout),
                (&buf, 0, layout),
                (cc * nc) as u32,
                log_n,
                inverse,
                Prescale::default(),
                None,
                None,
                &scratch,
            )?;
            self.backend.download(&buf, 0, chunk)?;
        }
        Ok(())
    }

    /// In-place NTT of every column of a column-major table (columns of
    /// length `n`). Same result as twenty-first's `ntt` on each column.
    pub fn ntt<FF: GpuField>(&self, table: &mut [FF], n: usize) -> Result<()> {
        self.ntt_impl(table, n, false)
    }

    /// In-place inverse NTT of every column. Same as twenty-first's `intt`.
    pub fn intt<FF: GpuField>(&self, table: &mut [FF], n: usize) -> Result<()> {
        self.ntt_impl(table, n, true)
    }

    /// Upload per-column randomizer polynomials for columns `[first, first+count)`.
    fn upload_randomizers<FF: GpuField>(
        &self,
        rands: &[Vec<FF>],
        first: usize,
        count: usize,
    ) -> Result<(B::Buffer, B::Buffer, B::Buffer)> {
        let nc = FF::NCOMP as usize;
        let mut coeffs: Vec<u64> = vec![];
        let mut offs = Vec::with_capacity(count);
        let mut lens = Vec::with_capacity(count);
        for r in &rands[first..first + count] {
            offs.push(coeffs.len() as u64);
            lens.push(
                u32::try_from(r.len()).map_err(|_| Error::Input("randomizer too long".into()))?,
            );
            coeffs.extend_from_slice(FF::as_u64s(r));
        }
        let _ = nc;
        let c = self.backend.alloc(coeffs.len().max(1))?;
        self.backend.upload(&c, 0, &coeffs)?;
        let o = self.backend.alloc(offs.len().max(1))?;
        self.backend.upload(&o, 0, &offs)?;
        let l = self.backend.alloc_u32(&lens)?;
        Ok((c, o, l))
    }

    /// Low-degree extension of interpolant columns onto the coset
    /// `offset · ⟨ω_n⟩`.
    ///
    /// `interpolants` is column-major with `m ≤ n` coefficients per column.
    /// With `randomizers = Some((polys, zerofier))`, column `j` additionally
    /// gets `zerofier · polys[j]` added, exactly like triton-vm's
    /// `evaluate_randomized_interpolant_into`. Returns the column-major
    /// codewords (length `n` per column).
    pub fn lde<FF: GpuField>(
        &self,
        interpolants: &[FF],
        m: usize,
        n: usize,
        offset: BFieldElement,
        randomizers: Option<(&[Vec<FF>], BFieldElement)>,
    ) -> Result<Vec<FF>> {
        let num_cols = interpolants.len().checked_div(m).unwrap_or(0);
        let mut out = vec![0u64; num_cols * n * FF::NCOMP as usize];
        self.lde_into_raw::<FF>(interpolants, m, n, offset, randomizers, &mut out)?;
        Ok(FF::from_u64s(out))
    }

    /// Like [`Self::lde`], writing the column-major codewords into `out`
    /// (which may be uninitialized memory: it is only written).
    pub fn lde_into<FF: GpuField>(
        &self,
        interpolants: &[FF],
        m: usize,
        n: usize,
        offset: BFieldElement,
        randomizers: Option<(&[Vec<FF>], BFieldElement)>,
        out: &mut [std::mem::MaybeUninit<FF>],
    ) -> Result<()> {
        let len = out.len() * FF::NCOMP as usize;
        // SAFETY: FF is NCOMP plain u64s; the buffer is only written, never read.
        let raw = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u64, len) };
        self.lde_into_raw::<FF>(interpolants, m, n, offset, randomizers, raw)
    }

    fn lde_into_raw<FF: GpuField>(
        &self,
        interpolants: &[FF],
        m: usize,
        n: usize,
        offset: BFieldElement,
        randomizers: Option<(&[Vec<FF>], BFieldElement)>,
        out: &mut [u64],
    ) -> Result<()> {
        let (_, log_n) = Self::check_table(n, n)?;
        if m == 0 || m > n || !interpolants.len().is_multiple_of(m) {
            return Err(Error::Input(
                "interpolant length must be a multiple of m, with 0 < m ≤ n".into(),
            ));
        }
        let num_cols = interpolants.len() / m;
        if let Some((r, _)) = randomizers {
            if r.len() != num_cols || r.iter().any(|p| p.len() > n) {
                return Err(Error::Input(
                    "need one randomizer of length ≤ n per column".into(),
                ));
            }
        }
        let nc = FF::NCOMP as usize;
        if out.len() != num_cols * n * nc {
            return Err(Error::Input("output buffer has the wrong length".into()));
        }
        let src_flat = FF::as_u64s(interpolants);
        let cols = self.chunk_cols(m * nc + 2 * n * nc, num_cols);
        let src = self.backend.alloc(cols * m * nc)?;
        let dst = self.backend.alloc(cols * n * nc)?;
        let scratch = self.backend.alloc(cols * n * nc)?;
        let powers = self.backend.alloc(n)?;
        self.backend.powers(&powers, raw(offset), n as u64)?;
        for first in (0..num_cols).step_by(cols) {
            let cc = cols.min(num_cols - first);
            self.backend
                .upload(&src, 0, &src_flat[first * m * nc..(first + cc) * m * nc])?;
            let sl = Layout {
                col_stride: (m * nc) as u64,
                ncomp: nc as u32,
                len: m as u32,
            };
            let dl = Layout {
                col_stride: (n * nc) as u64,
                ncomp: nc as u32,
                len: n as u32,
            };
            let rb = match randomizers {
                Some((r, _)) => Some(self.upload_randomizers(r, first, cc)?),
                None => None,
            };
            let rand = rb.as_ref().map(|(c, o, l)| Randomizers {
                coefficients: c,
                offsets: o,
                lengths: l,
            });
            let pre = Prescale {
                enabled: 1,
                has_rand: rand.is_some() as u32,
                zerofier: randomizers.map_or(0, |(_, z)| raw(z)),
            };
            self.transform(
                (&src, 0, sl),
                (&dst, 0, dl),
                (cc * nc) as u32,
                log_n,
                false,
                pre,
                Some(&powers),
                rand.as_ref(),
                &scratch,
            )?;
            self.backend
                .download(&dst, 0, &mut out[first * n * nc..(first + cc) * n * nc])?;
        }
        Ok(())
    }

    /// `Tip5::hash_varlen` of every row of a column-major table with
    /// `num_rows` rows: the same as triton-vm's
    /// `hash_rows_of_column_major_table` / twenty-first's
    /// `Tip5::hash_rows_of_columns`.
    pub fn hash_rows<FF: GpuField>(&self, table: &[FF], num_rows: usize) -> Result<Vec<Digest>> {
        if num_rows == 0 || !table.len().is_multiple_of(num_rows) {
            return Err(Error::Input(
                "table length must be a multiple of num_rows > 0".into(),
            ));
        }
        let nc = FF::NCOMP as usize;
        let num_cols = table.len() / num_rows;
        let flat = FF::as_u64s(table);
        let mut hasher = RowHasher::new(self, num_rows)?;
        let cols = self.chunk_cols(num_rows * nc, num_cols);
        let buf = self.backend.alloc(cols * num_rows * nc)?;
        for chunk in flat.chunks(cols * num_rows * nc) {
            let cc = chunk.len() / (num_rows * nc);
            self.backend.upload(&buf, 0, chunk)?;
            let l = Layout {
                col_stride: (num_rows * nc) as u64,
                ncomp: nc as u32,
                len: num_rows as u32,
            };
            hasher.absorb((&buf, 0, l), (cc * nc) as u32)?;
        }
        hasher.finish()
    }

    /// The Merkle tree over the given leafs (a power of two of them), the same
    /// as twenty-first's `MerkleTree::par_new`.
    pub fn merkle_tree(&self, leafs: &[Digest]) -> Result<GpuMerkleTree> {
        let n = leafs.len();
        if n == 0 || !n.is_power_of_two() {
            return Err(Error::Input(
                "number of leafs must be a power of two".into(),
            ));
        }
        let nodes = self.backend.alloc(2 * n * DIGEST_LEN)?;
        let flat: Vec<u64> = leafs.iter().flat_map(|d| d.0.map(raw)).collect();
        self.backend.upload(&nodes, n * DIGEST_LEN, &flat)?;
        let mut m = n as u64 / 2;
        let tail = self.backend.merkle_tail_max();
        while m > tail {
            self.backend.merkle_level(&nodes, m, m)?;
            m /= 2;
        }
        if m >= 1 {
            self.backend.merkle_tail(&nodes, m)?;
        }
        let mut out = vec![0u64; 2 * n * DIGEST_LEN];
        self.backend.download(&nodes, 0, &mut out)?;
        let mut nodes: Vec<Digest> = out
            .chunks_exact(DIGEST_LEN)
            .map(|c| Digest::new(std::array::from_fn(|i| BFieldElement::from_raw_u64(c[i]))))
            .collect();
        nodes[0] = Digest::default();
        Ok(GpuMerkleTree { nodes })
    }

    /// The prover's commitment path for one master table, memory-light.
    ///
    /// Equivalent to triton-vm's `MasterTable::hash_all_fri_domain_rows`
    /// without cached LDE: for every coset of the FRI domain (offset
    /// `fri_offset · fri_generator^i`, length `num_rows`), all columns are
    /// low-degree extended with their trace randomizers and every row is
    /// hashed. Only the digests ever leave the GPU; the extended table is
    /// never stored, on the device or the host. Device memory is bounded by
    /// [`Config::chunk_bytes`] plus 128 bytes per row of sponge state.
    ///
    /// `interpolants`: column-major interpolant coefficients (`num_rows` per
    /// column), i.e., the trace table after `intt`. `randomizers`: one trace
    /// randomizer polynomial per column. Returns the digests of all FRI
    /// domain rows in FRI domain order.
    pub fn lde_hash_cosets<FF: GpuField>(
        &self,
        interpolants: &[FF],
        num_rows: usize,
        randomizers: &[Vec<FF>],
        fri_offset: BFieldElement,
        fri_generator: BFieldElement,
        num_cosets: usize,
    ) -> Result<Vec<Digest>> {
        let (num_cols, log_n) = Self::check_table(interpolants.len(), num_rows)?;
        if randomizers.len() != num_cols || randomizers.iter().any(|r| r.len() > num_rows) {
            return Err(Error::Input(
                "need one randomizer of length ≤ num_rows per column".into(),
            ));
        }
        let hybrid = match self.config.hybrid {
            Hybrid::On => true,
            Hybrid::Off => false,
            Hybrid::Auto => self.backend.unified_memory(),
        };
        if hybrid && self.backend.host_ptr(&self.dummy).is_some() {
            return self.lde_hash_cosets_hybrid(
                interpolants,
                num_rows,
                randomizers,
                fri_offset,
                fri_generator,
                num_cosets,
            );
        }

        let nc = FF::NCOMP as usize;
        let flat = FF::as_u64s(interpolants);
        let col_u64s = num_rows * nc;
        // Per column: interpolant + codeword + scratch.
        let cols = self.chunk_cols(3 * col_u64s, num_cols);
        let resident = cols == num_cols;
        let src = self.backend.alloc(cols * col_u64s)?;
        let dst = self.backend.alloc(cols * col_u64s)?;
        let scratch = self.backend.alloc(cols * col_u64s)?;
        let powers = self.backend.alloc(num_rows)?;
        let layout = Layout {
            col_stride: col_u64s as u64,
            ncomp: nc as u32,
            len: num_rows as u32,
        };

        let mut rand_cache = None;
        let mut all = vec![Digest::default(); num_rows * num_cosets];
        for coset in 0..num_cosets {
            let coset_offset = fri_offset * fri_generator.mod_pow_u32(coset as u32);
            let zerofier = coset_offset.mod_pow_u32(num_rows as u32) - BFieldElement::ONE;
            self.backend
                .powers(&powers, raw(coset_offset), num_rows as u64)?;
            let mut hasher = RowHasher::new(self, num_rows)?;
            for first in (0..num_cols).step_by(cols) {
                let cc = cols.min(num_cols - first);
                if !resident || coset == 0 {
                    self.backend.upload(
                        &src,
                        0,
                        &flat[first * col_u64s..(first + cc) * col_u64s],
                    )?;
                    rand_cache = Some(self.upload_randomizers(randomizers, first, cc)?);
                }
                let (c, o, l) = rand_cache.as_ref().unwrap();
                let rand = Randomizers {
                    coefficients: c,
                    offsets: o,
                    lengths: l,
                };
                let pre = Prescale {
                    enabled: 1,
                    has_rand: 1,
                    zerofier: raw(zerofier),
                };
                self.transform(
                    (&src, 0, layout),
                    (&dst, 0, layout),
                    (cc * nc) as u32,
                    log_n,
                    false,
                    pre,
                    Some(&powers),
                    Some(&rand),
                    &scratch,
                )?;
                hasher.absorb((&dst, 0, layout), (cc * nc) as u32)?;
            }
            for (j, d) in hasher.finish()?.into_iter().enumerate() {
                all[coset + j * num_cosets] = d;
            }
        }
        Ok(all)
    }
}

impl<B: Backend> Gpu<B> {
    /// See [`Hybrid`]. Same inputs and outputs as [`Gpu::lde_hash_cosets`].
    fn lde_hash_cosets_hybrid<FF: GpuField>(
        &self,
        interpolants: &[FF],
        num_rows: usize,
        randomizers: &[Vec<FF>],
        fri_offset: BFieldElement,
        fri_generator: BFieldElement,
        num_cosets: usize,
    ) -> Result<Vec<Digest>> {
        use rayon::prelude::*;

        let num_cols = interpolants.len() / num_rows;
        let nc = FF::NCOMP as usize;
        let col_u64s = num_rows * nc;
        // Two slots: the CPU fills one while the GPU hashes the other.
        let cols = self.chunk_cols(2 * col_u64s, num_cols);
        let slots = [
            self.backend.alloc(cols * col_u64s)?,
            self.backend.alloc(cols * col_u64s)?,
        ];
        let layout = Layout {
            col_stride: col_u64s as u64,
            ncomp: nc as u32,
            len: num_rows as u32,
        };

        // Per slot: the ticket of the GPU work reading it, and the digests
        // to collect once that work is done.
        let mut tickets: [Option<u64>; 2] = [None, None];
        type Finished<Buf> = Option<(usize, Buf, Buf)>; // coset, digests, state
        let mut finished: [Finished<B::Buffer>; 2] = [None, None];
        let mut all = vec![Digest::default(); num_rows * num_cosets];

        let collect = |coset: usize, digests: &B::Buffer, all: &mut [Digest]| {
            let ptr = self.backend.host_ptr(digests).expect("host-visible buffer");
            // SAFETY: the GPU work writing the digests is done (waited for).
            let raw = unsafe { std::slice::from_raw_parts(ptr, num_rows * DIGEST_LEN) };
            for (j, c) in raw.chunks_exact(DIGEST_LEN).enumerate() {
                all[coset + j * num_cosets] =
                    Digest::new(std::array::from_fn(|i| BFieldElement::from_raw_u64(c[i])));
            }
        };

        let mut item = 0;
        for coset in 0..num_cosets {
            let coset_offset = fri_offset * fri_generator.mod_pow_u32(coset as u32);
            let zerofier = coset_offset.mod_pow_u32(num_rows as u32) - BFieldElement::ONE;
            let state = self.backend.alloc(num_rows * STATE_SIZE)?;
            self.backend.rows_init(&state, num_rows as u64)?;
            let mut pos = 0;
            let mut state = Some(state);
            for first in (0..num_cols).step_by(cols) {
                let s = item % 2;
                item += 1;
                if let Some(t) = tickets[s].take() {
                    self.backend.wait(t)?;
                }
                if let Some((c, digests, _state)) = finished[s].take() {
                    collect(c, &digests, &mut all);
                }

                // CPU: low-degree extend this chunk straight into the slot,
                // all cores, while the GPU still hashes the previous chunk.
                let cc = cols.min(num_cols - first);
                let ptr = self
                    .backend
                    .host_ptr(&slots[s])
                    .expect("host-visible buffer");
                // SAFETY: the slot holds `cc` columns of FF; no GPU work uses it
                // (its ticket was waited for above).
                let out =
                    unsafe { std::slice::from_raw_parts_mut(ptr.cast::<FF>(), cc * num_rows) };
                out.par_chunks_mut(num_rows)
                    .enumerate()
                    .for_each(|(j, column)| {
                        let col = first + j;
                        FF::cpu_coset_lde(
                            &interpolants[col * num_rows..(col + 1) * num_rows],
                            &randomizers[col],
                            zerofier,
                            coset_offset,
                            column,
                        );
                    });

                // GPU: hash the rows of this chunk.
                let st = state.as_ref().unwrap();
                let vcols = (cc * nc) as u32;
                self.backend.rows_absorb(
                    st,
                    num_rows as u64,
                    pos,
                    (&slots[s], 0, layout),
                    vcols,
                )?;
                pos = (pos + vcols) % 10;
                if first + cc == num_cols {
                    let digests = self.backend.alloc(num_rows * DIGEST_LEN)?;
                    self.backend
                        .rows_finish(st, num_rows as u64, pos, &digests)?;
                    finished[s] = Some((coset, digests, state.take().unwrap()));
                }
                tickets[s] = Some(self.backend.submit()?);
            }
        }
        for s in 0..2 {
            if let Some(t) = tickets[s].take() {
                self.backend.wait(t)?;
            }
            if let Some((c, digests, _state)) = finished[s].take() {
                collect(c, &digests, &mut all);
            }
        }
        Ok(all)
    }
}

/// Streaming row hasher: absorbs a table column chunk by column chunk.
pub struct RowHasher<'g, B: Backend> {
    gpu: &'g Gpu<B>,
    state: B::Buffer,
    num_rows: usize,
    pos: u32,
}

impl<'g, B: Backend> RowHasher<'g, B> {
    pub fn new(gpu: &'g Gpu<B>, num_rows: usize) -> Result<Self> {
        let state = gpu.backend.alloc(num_rows * STATE_SIZE)?;
        gpu.backend.rows_init(&state, num_rows as u64)?;
        Ok(Self {
            gpu,
            state,
            num_rows,
            pos: 0,
        })
    }

    /// Absorb `num_vcols` virtual (B-field) columns, in order.
    pub fn absorb(&mut self, data: (&B::Buffer, usize, Layout), num_vcols: u32) -> Result<()> {
        self.gpu.backend.rows_absorb(
            &self.state,
            self.num_rows as u64,
            self.pos,
            data,
            num_vcols,
        )?;
        self.pos = (self.pos + num_vcols) % 10;
        Ok(())
    }

    pub fn finish(self) -> Result<Vec<Digest>> {
        let digests = self.gpu.backend.alloc(self.num_rows * DIGEST_LEN)?;
        self.gpu
            .backend
            .rows_finish(&self.state, self.num_rows as u64, self.pos, &digests)?;
        let mut out = vec![0u64; self.num_rows * DIGEST_LEN];
        self.gpu.backend.download(&digests, 0, &mut out)?;
        Ok(out
            .chunks_exact(DIGEST_LEN)
            .map(|c| Digest::new(std::array::from_fn(|i| BFieldElement::from_raw_u64(c[i]))))
            .collect())
    }
}
