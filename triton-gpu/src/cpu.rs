//! CPU reference backend: runs the shared kernel code (kernels/common) on the
//! CPU, emulating the GPU launch geometry. Slow, but it executes exactly the
//! index mappings and arithmetic the GPU kernels execute, which is what the
//! equivalence tests check against twenty-first.

use std::cell::UnsafeCell;
use std::ptr;

use crate::backend::{Backend, DeviceBuffer, Layout, NttPass, Prescale, Randomizers};
use crate::ffi::cpu as k;
use crate::{Error, Result};

pub struct CpuBuffer {
    data: Box<[UnsafeCell<u64>]>,
}

impl CpuBuffer {
    fn ptr(&self, offset: usize) -> *mut u64 {
        assert!(offset <= self.data.len());
        // SAFETY: UnsafeCell<u64> has the same layout as u64; offset is in bounds.
        unsafe { (self.data.as_ptr() as *mut u64).add(offset) }
    }
}

impl DeviceBuffer for CpuBuffer {
    fn len(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CpuBackend {
    /// Emulated threadgroup size limit; smaller values force multi-pass NTT
    /// plans at small sizes (useful for tests).
    pub max_threads: u32,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self { max_threads: 1024 }
    }
}

fn check(code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::Backend(format!("cpu kernel error {code}")))
    }
}

impl Backend for CpuBackend {
    type Buffer = CpuBuffer;

    fn name(&self) -> String {
        "CPU reference (shared kernel code)".into()
    }

    fn max_ntt_threads(&self) -> u32 {
        self.max_threads
    }

    fn alloc(&self, len: usize) -> Result<CpuBuffer> {
        Ok(CpuBuffer {
            data: (0..len).map(|_| UnsafeCell::new(0)).collect(),
        })
    }

    fn alloc_u32(&self, src: &[u32]) -> Result<CpuBuffer> {
        let buf = self.alloc(src.len().div_ceil(2))?;
        // SAFETY: the buffer holds at least src.len() u32s.
        unsafe { ptr::copy_nonoverlapping(src.as_ptr(), buf.ptr(0) as *mut u32, src.len()) };
        Ok(buf)
    }

    fn upload(&self, dst: &CpuBuffer, off: usize, src: &[u64]) -> Result<()> {
        assert!(off + src.len() <= dst.len());
        unsafe { ptr::copy_nonoverlapping(src.as_ptr(), dst.ptr(off), src.len()) };
        Ok(())
    }

    fn download(&self, src: &CpuBuffer, off: usize, dst: &mut [u64]) -> Result<()> {
        assert!(off + dst.len() <= src.len());
        unsafe { ptr::copy_nonoverlapping(src.ptr(off), dst.as_mut_ptr(), dst.len()) };
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn ntt_pass(
        &self,
        src: (&CpuBuffer, usize, Layout),
        dst: (&CpuBuffer, usize, Layout),
        num_vcols: u32,
        pass: NttPass,
        tw: &CpuBuffer,
        pre: Prescale,
        offset_powers: Option<&CpuBuffer>,
        rand: Option<&Randomizers<'_, Self>>,
    ) -> Result<()> {
        let op = offset_powers.map_or(ptr::null(), |b| b.ptr(0) as *const u64);
        let (rc, ro, rl) = rand.map_or((ptr::null(), ptr::null(), ptr::null()), |r| {
            (
                r.coefficients.ptr(0) as *const u64,
                r.offsets.ptr(0) as *const u64,
                r.lengths.ptr(0) as *const u32,
            )
        });
        check(unsafe {
            k::ntt_pass(
                ptr::null_mut(),
                src.0.ptr(src.1),
                src.2,
                dst.0.ptr(dst.1),
                dst.2,
                num_vcols,
                pass,
                tw.ptr(0),
                pre,
                op,
                rc,
                ro,
                rl,
            )
        })
    }

    fn bitrev(
        &self,
        src: (&CpuBuffer, usize, Layout),
        dst: (&CpuBuffer, usize, Layout),
        num_vcols: u32,
        log_n: u32,
        scale: Option<u64>,
    ) -> Result<()> {
        check(unsafe {
            k::bitrev(
                ptr::null_mut(),
                src.0.ptr(src.1),
                src.2,
                dst.0.ptr(dst.1),
                dst.2,
                num_vcols,
                log_n,
                scale.is_some() as u32,
                scale.unwrap_or(0),
            )
        })
    }

    fn powers(&self, out: &CpuBuffer, base: u64, n: u64) -> Result<()> {
        check(unsafe { k::powers(ptr::null_mut(), out.ptr(0), base, n) })
    }

    fn rows_init(&self, state: &CpuBuffer, num_rows: u64) -> Result<()> {
        check(unsafe { k::rows_init(ptr::null_mut(), state.ptr(0), num_rows) })
    }

    fn rows_absorb(
        &self,
        state: &CpuBuffer,
        num_rows: u64,
        pos: u32,
        data: (&CpuBuffer, usize, Layout),
        num_vcols: u32,
    ) -> Result<()> {
        check(unsafe {
            k::rows_absorb(
                ptr::null_mut(),
                state.ptr(0),
                num_rows,
                pos,
                data.0.ptr(data.1),
                data.2,
                num_vcols,
            )
        })
    }

    fn rows_finish(
        &self,
        state: &CpuBuffer,
        num_rows: u64,
        pos: u32,
        digests: &CpuBuffer,
    ) -> Result<()> {
        check(unsafe {
            k::rows_finish(ptr::null_mut(), state.ptr(0), num_rows, pos, digests.ptr(0))
        })
    }

    fn merkle_level(&self, nodes: &CpuBuffer, first: u64, count: u64) -> Result<()> {
        check(unsafe { k::merkle_level(ptr::null_mut(), nodes.ptr(0), first, count) })
    }

    fn merkle_tail(&self, nodes: &CpuBuffer, m: u64) -> Result<()> {
        check(unsafe { k::merkle_tail(ptr::null_mut(), nodes.ptr(0), m) })
    }
}
