//! CUDA backend (NVIDIA GPUs).
//!
//! All work is enqueued on one non-blocking stream; device memory comes from
//! the stream-ordered pool (`cudaMallocAsync`), so allocations and frees
//! never stall the device. The host blocks only when it reads results.

use std::ffi::{c_char, c_void, CStr};
use std::ptr;

use crate::backend::{Backend, DeviceBuffer, Layout, NttPass, Prescale, Randomizers};
use crate::ffi::{cuda as k, cuda_mem as m};
use crate::{Error, Result};

fn check(code: i32, what: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::Backend(format!("CUDA error {code} in {what}")))
    }
}

pub struct CudaBuffer {
    ptr: *mut c_void,
    len: usize,
    stream: *mut c_void,
}

impl CudaBuffer {
    fn at(&self, offset: usize) -> *mut u64 {
        assert!(offset <= self.len);
        if self.ptr.is_null() {
            return ptr::null_mut();
        }
        // SAFETY: in bounds of the allocation.
        unsafe { (self.ptr as *mut u64).add(offset) }
    }
}

impl DeviceBuffer for CudaBuffer {
    fn len(&self) -> usize {
        self.len
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        // Stream-ordered free: safe even if kernels using it are still queued.
        unsafe { m::tg_cuda_free(self.stream, self.ptr) };
    }
}

pub struct CudaBackend {
    stream: *mut c_void,
    name: String,
}

impl CudaBackend {
    /// Open CUDA device `device` and create a non-blocking stream on it.
    pub fn new(device: i32) -> Result<Self> {
        let mut stream = ptr::null_mut();
        let mut name = [0 as c_char; 256];
        check(
            unsafe { m::tg_cuda_init(device, &mut stream, name.as_mut_ptr(), name.len() as i32) },
            "init",
        )?;
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Ok(Self { stream, name })
    }

    /// Page-lock a host slice for the lifetime of the returned guard, which
    /// turns uploads and downloads of it into asynchronous DMA at full PCIe
    /// bandwidth. Worth it for large tables that are transferred repeatedly.
    pub fn pin<'a, T>(&self, data: &'a mut [T]) -> Result<Pinned<'a, T>> {
        let bytes = std::mem::size_of_val(data) as u64;
        check(
            unsafe { m::tg_cuda_pin(data.as_mut_ptr() as *mut c_void, bytes) },
            "pin",
        )?;
        Ok(Pinned { data })
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        unsafe {
            m::tg_cuda_sync(self.stream);
            m::tg_cuda_destroy(self.stream);
        }
    }
}

/// A page-locked host slice; unpinned on drop.
pub struct Pinned<'a, T> {
    pub data: &'a mut [T],
}

impl<T> Drop for Pinned<'_, T> {
    fn drop(&mut self) {
        unsafe { m::tg_cuda_unpin(self.data.as_mut_ptr() as *mut c_void) };
    }
}

impl Backend for CudaBackend {
    type Buffer = CudaBuffer;

    fn name(&self) -> String {
        format!("CUDA: {}", self.name)
    }

    fn alloc(&self, len: usize) -> Result<CudaBuffer> {
        let mut p = ptr::null_mut();
        check(
            unsafe { m::tg_cuda_malloc(self.stream, &mut p, (len * 8) as u64) },
            "malloc",
        )?;
        Ok(CudaBuffer {
            ptr: p,
            len,
            stream: self.stream,
        })
    }

    fn alloc_u32(&self, src: &[u32]) -> Result<CudaBuffer> {
        let buf = self.alloc(src.len().div_ceil(2).max(1))?;
        check(
            unsafe {
                m::tg_cuda_htod(
                    self.stream,
                    buf.ptr,
                    src.as_ptr() as *const c_void,
                    (src.len() * 4) as u64,
                )
            },
            "htod",
        )?;
        Ok(buf)
    }

    fn upload(&self, dst: &CudaBuffer, off: usize, src: &[u64]) -> Result<()> {
        assert!(off + src.len() <= dst.len);
        check(
            unsafe {
                m::tg_cuda_htod(
                    self.stream,
                    dst.at(off) as *mut c_void,
                    src.as_ptr() as *const c_void,
                    (src.len() * 8) as u64,
                )
            },
            "htod",
        )
    }

    fn download(&self, src: &CudaBuffer, off: usize, dst: &mut [u64]) -> Result<()> {
        assert!(off + dst.len() <= src.len);
        check(
            unsafe {
                m::tg_cuda_dtoh(
                    self.stream,
                    dst.as_mut_ptr() as *mut c_void,
                    src.at(off) as *const c_void,
                    (dst.len() * 8) as u64,
                )
            },
            "dtoh",
        )
    }

    fn synchronize(&self) -> Result<()> {
        check(unsafe { m::tg_cuda_sync(self.stream) }, "sync")
    }

    fn ntt_pass(
        &self,
        src: (&CudaBuffer, usize, Layout),
        dst: (&CudaBuffer, usize, Layout),
        num_vcols: u32,
        pass: NttPass,
        tw: &CudaBuffer,
        pre: Prescale,
        op: Option<&CudaBuffer>,
        rand: Option<&Randomizers<'_, Self>>,
    ) -> Result<()> {
        let op = op.map_or(ptr::null(), |b| b.at(0) as *const u64);
        let (rc, ro, rl) = rand.map_or((ptr::null(), ptr::null(), ptr::null()), |r| {
            (
                r.coefficients.at(0) as *const u64,
                r.offsets.at(0) as *const u64,
                r.lengths.at(0) as *const u32,
            )
        });
        check(
            unsafe {
                k::ntt_pass(
                    self.stream,
                    src.0.at(src.1),
                    src.2,
                    dst.0.at(dst.1),
                    dst.2,
                    num_vcols,
                    pass,
                    tw.at(0),
                    pre,
                    op,
                    rc,
                    ro,
                    rl,
                )
            },
            "ntt_pass",
        )
    }

    fn bitrev(
        &self,
        src: (&CudaBuffer, usize, Layout),
        dst: (&CudaBuffer, usize, Layout),
        num_vcols: u32,
        log_n: u32,
        scale: Option<u64>,
    ) -> Result<()> {
        check(
            unsafe {
                k::bitrev(
                    self.stream,
                    src.0.at(src.1),
                    src.2,
                    dst.0.at(dst.1),
                    dst.2,
                    num_vcols,
                    log_n,
                    scale.is_some() as u32,
                    scale.unwrap_or(0),
                )
            },
            "bitrev",
        )
    }

    fn powers(&self, out: &CudaBuffer, base: u64, n: u64) -> Result<()> {
        check(
            unsafe { k::powers(self.stream, out.at(0), base, n) },
            "powers",
        )
    }

    fn rows_init(&self, state: &CudaBuffer, num_rows: u64) -> Result<()> {
        check(
            unsafe { k::rows_init(self.stream, state.at(0), num_rows) },
            "rows_init",
        )
    }

    fn rows_absorb(
        &self,
        state: &CudaBuffer,
        num_rows: u64,
        pos: u32,
        data: (&CudaBuffer, usize, Layout),
        num_vcols: u32,
    ) -> Result<()> {
        check(
            unsafe {
                k::rows_absorb(
                    self.stream,
                    state.at(0),
                    num_rows,
                    pos,
                    data.0.at(data.1),
                    data.2,
                    num_vcols,
                )
            },
            "rows_absorb",
        )
    }

    fn rows_finish(
        &self,
        state: &CudaBuffer,
        num_rows: u64,
        pos: u32,
        digests: &CudaBuffer,
    ) -> Result<()> {
        check(
            unsafe { k::rows_finish(self.stream, state.at(0), num_rows, pos, digests.at(0)) },
            "rows_finish",
        )
    }

    fn merkle_level(&self, nodes: &CudaBuffer, first: u64, count: u64) -> Result<()> {
        check(
            unsafe { k::merkle_level(self.stream, nodes.at(0), first, count) },
            "merkle_level",
        )
    }

    fn merkle_tail(&self, nodes: &CudaBuffer, m: u64) -> Result<()> {
        check(
            unsafe { k::merkle_tail(self.stream, nodes.at(0), m) },
            "merkle_tail",
        )
    }
}
