//! Metal backend (Apple silicon).
//!
//! - Unified memory: every buffer is `StorageModeShared`, so CPU and GPU use
//!   the same physical pages. "Upload" and "download" are plain memcpys in
//!   RAM, there is no bus transfer, and no second copy of the table exists in
//!   separate GPU memory.
//! - Batching: kernels are encoded into one command buffer and one compute
//!   encoder that are committed only when the host needs a result (download /
//!   synchronize). A whole NTT, LDE or Merkle tree is a single submission.
//! - The shader library is compiled from source at startup, so the crate
//!   builds without Xcode's command-line Metal compiler.

use std::cell::RefCell;
use std::ffi::c_void;
use std::ptr;

use metal::{
    Buffer, CommandBuffer, CommandQueue, CompileOptions, ComputeCommandEncoder,
    ComputeCommandEncoderRef, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
use objc::rc::autoreleasepool;

use crate::backend::{Backend, DeviceBuffer, Layout, NttPass, Prescale, Randomizers};
use crate::{Error, Result};

/// The full Metal source: shared headers + Metal kernels, includes inlined.
pub fn shader_source() -> String {
    let parts = [
        include_str!("../kernels/common/tg_prelude.h"),
        include_str!("../kernels/common/tg_arith.h"),
        include_str!("../kernels/common/tg_tip5_tables.h"),
        include_str!("../kernels/common/tg_tip5_mds.h"),
        include_str!("../kernels/common/tg_core.h"),
        include_str!("../kernels/metal/triton_gpu.metal"),
    ];
    let mut src = String::from("#define TG_METAL 1\n");
    for p in parts {
        for line in p.lines() {
            let t = line.trim_start();
            if t.starts_with("#include \"") || t.starts_with("#pragma once") {
                continue;
            }
            src.push_str(line);
            src.push('\n');
        }
    }
    src
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NttArgs {
    sl: Layout,
    dl: Layout,
    p: NttPass,
    pre: Prescale,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct BitrevArgs {
    sl: Layout,
    dl: Layout,
    log_n: u32,
    do_scale: u32,
    scale: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct RowsArgs {
    l: Layout,
    num_rows: u64,
    pos: u32,
    num_vcols: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct PowArgs {
    base: u64,
    n: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct MerkleArgs {
    first: u64,
    count: u64,
}

const _: () = assert!(std::mem::size_of::<NttArgs>() == 72);
const _: () = assert!(std::mem::size_of::<BitrevArgs>() == 48);
const _: () = assert!(std::mem::size_of::<RowsArgs>() == 32);

pub struct MetalBuffer {
    buf: Buffer,
    len: usize,
}

impl DeviceBuffer for MetalBuffer {
    fn len(&self) -> usize {
        self.len
    }
}

struct Pipelines {
    ntt: ComputePipelineState,
    bitrev: ComputePipelineState,
    powers: ComputePipelineState,
    rows_absorb: ComputePipelineState,
    rows_finish: ComputePipelineState,
    merkle_level: ComputePipelineState,
    merkle_tail: ComputePipelineState,
}

struct Pending {
    cb: CommandBuffer,
    enc: ComputeCommandEncoder,
}

pub struct MetalBackend {
    device: Device,
    queue: CommandQueue,
    p: Pipelines,
    pending: RefCell<Option<Pending>>,
    /// Committed command buffers that may still run, with their tickets.
    in_flight: RefCell<std::collections::VecDeque<(u64, CommandBuffer)>>,
    next_ticket: std::cell::Cell<u64>,
    dummy: Buffer,
}

const HASH_THREADS: u64 = 128;
const TAIL_THREADS: u64 = 256;

fn options() -> MTLResourceOptions {
    MTLResourceOptions::StorageModeShared
        | MTLResourceOptions::CPUCacheModeDefaultCache
        | MTLResourceOptions::HazardTrackingModeTracked
}

fn size(w: u64, h: u64) -> MTLSize {
    MTLSize::new(w, h, 1)
}

impl MetalBackend {
    pub fn new() -> Result<Self> {
        autoreleasepool(|| {
            let device =
                Device::system_default().ok_or_else(|| Error::Backend("no Metal device".into()))?;
            let opts = CompileOptions::new();
            opts.set_fast_math_enabled(false);
            let lib = device
                .new_library_with_source(&shader_source(), &opts)
                .map_err(|e| Error::Backend(format!("Metal shader compilation failed: {e}")))?;
            let pipe = |name: &str| -> Result<ComputePipelineState> {
                let f = lib.get_function(name, None).map_err(Error::Backend)?;
                device
                    .new_compute_pipeline_state_with_function(&f)
                    .map_err(Error::Backend)
            };
            let p = Pipelines {
                ntt: pipe("tg_ntt_pass_k")?,
                bitrev: pipe("tg_bitrev_k")?,
                powers: pipe("tg_powers_k")?,
                rows_absorb: pipe("tg_rows_absorb_k")?,
                rows_finish: pipe("tg_rows_finish_k")?,
                merkle_level: pipe("tg_merkle_level_k")?,
                merkle_tail: pipe("tg_merkle_tail_k")?,
            };
            let queue = device.new_command_queue();
            let dummy = device.new_buffer(16, options());
            Ok(Self {
                device,
                queue,
                p,
                pending: RefCell::new(None),
                in_flight: RefCell::new(std::collections::VecDeque::new()),
                next_ticket: std::cell::Cell::new(1),
                dummy,
            })
        })
    }

    /// Encode into the open command buffer (opening one if needed).
    fn encode(&self, f: impl FnOnce(&ComputeCommandEncoderRef)) {
        let mut pending = self.pending.borrow_mut();
        if pending.is_none() {
            *pending = Some(autoreleasepool(|| {
                let cb = self.queue.new_command_buffer().to_owned();
                let enc = cb.new_compute_command_encoder().to_owned();
                Pending { cb, enc }
            }));
        }
        f(&pending.as_ref().unwrap().enc);
    }

    /// Commit the open command buffer, if any, without waiting.
    fn commit(&self) -> u64 {
        let ticket = self.next_ticket.get();
        if let Some(p) = self.pending.borrow_mut().take() {
            p.enc.end_encoding();
            p.cb.commit();
            self.in_flight.borrow_mut().push_back((ticket, p.cb));
            self.next_ticket.set(ticket + 1);
            return ticket;
        }
        ticket - 1
    }

    /// Wait for all committed command buffers up to and including `ticket`.
    fn wait_for(&self, ticket: u64) -> Result<()> {
        loop {
            let front = {
                let mut q = self.in_flight.borrow_mut();
                match q.front() {
                    Some((t, _)) if *t <= ticket => q.pop_front(),
                    _ => None,
                }
            };
            let Some((_, cb)) = front else { return Ok(()) };
            cb.wait_until_completed();
            if cb.status() != MTLCommandBufferStatus::Completed {
                return Err(Error::Backend(format!(
                    "Metal command buffer ended with status {:?}",
                    cb.status()
                )));
            }
        }
    }

    /// Commit everything and wait for all of it.
    fn flush(&self) -> Result<()> {
        self.commit();
        self.wait_for(u64::MAX)
    }

    fn threads(pso: &ComputePipelineState, want: u64) -> u64 {
        want.min(pso.max_total_threads_per_threadgroup()).max(1)
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }
}

fn bytes<T>(enc: &ComputeCommandEncoderRef, index: u64, v: &T) {
    enc.set_bytes(
        index,
        std::mem::size_of::<T>() as u64,
        v as *const T as *const c_void,
    );
}

impl Backend for MetalBackend {
    type Buffer = MetalBuffer;

    fn name(&self) -> String {
        format!("Metal: {}", self.device.name())
    }

    fn max_ntt_threads(&self) -> u32 {
        let m = self.p.ntt.max_total_threads_per_threadgroup().min(1024) as u32;
        1 << m.ilog2()
    }

    fn merkle_tail_max(&self) -> u64 {
        1024
    }

    fn alloc(&self, len: usize) -> Result<MetalBuffer> {
        let buf = self.device.new_buffer((len.max(1) * 8) as u64, options());
        Ok(MetalBuffer { buf, len })
    }

    fn alloc_u32(&self, src: &[u32]) -> Result<MetalBuffer> {
        let b = self.alloc(src.len().div_ceil(2).max(1))?;
        unsafe { ptr::copy_nonoverlapping(src.as_ptr(), b.buf.contents() as *mut u32, src.len()) };
        Ok(b)
    }

    fn upload(&self, dst: &MetalBuffer, off: usize, src: &[u64]) -> Result<()> {
        assert!(off + src.len() <= dst.len);
        self.flush()?; // the GPU may still be reading the buffer
        unsafe {
            ptr::copy_nonoverlapping(
                src.as_ptr(),
                (dst.buf.contents() as *mut u64).add(off),
                src.len(),
            )
        };
        Ok(())
    }

    fn download(&self, src: &MetalBuffer, off: usize, dst: &mut [u64]) -> Result<()> {
        assert!(off + dst.len() <= src.len);
        self.flush()?;
        unsafe {
            ptr::copy_nonoverlapping(
                (src.buf.contents() as *const u64).add(off),
                dst.as_mut_ptr(),
                dst.len(),
            )
        };
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        self.flush()
    }

    fn unified_memory(&self) -> bool {
        true
    }

    fn host_ptr(&self, buf: &MetalBuffer) -> Option<*mut u64> {
        Some(buf.buf.contents() as *mut u64)
    }

    fn submit(&self) -> Result<u64> {
        Ok(self.commit())
    }

    fn wait(&self, ticket: u64) -> Result<()> {
        self.wait_for(ticket)
    }

    fn ntt_pass(
        &self,
        src: (&MetalBuffer, usize, Layout),
        dst: (&MetalBuffer, usize, Layout),
        num_vcols: u32,
        pass: NttPass,
        tw: &MetalBuffer,
        pre: Prescale,
        op: Option<&MetalBuffer>,
        rand: Option<&Randomizers<'_, Self>>,
    ) -> Result<()> {
        let tile = 1u64 << (pass.k + pass.c);
        let tiles = (1u64 << pass.log_n) / tile;
        let args = NttArgs {
            sl: src.2,
            dl: dst.2,
            p: pass,
            pre,
        };
        self.encode(|e| {
            e.set_compute_pipeline_state(&self.p.ntt);
            e.set_buffer(0, Some(&src.0.buf), (src.1 * 8) as u64);
            e.set_buffer(1, Some(&dst.0.buf), (dst.1 * 8) as u64);
            bytes(e, 2, &args);
            e.set_buffer(3, Some(&tw.buf), 0);
            e.set_buffer(4, Some(op.map_or(&self.dummy, |b| &b.buf)), 0);
            match rand {
                Some(r) => {
                    e.set_buffer(5, Some(&r.coefficients.buf), 0);
                    e.set_buffer(6, Some(&r.offsets.buf), 0);
                    e.set_buffer(7, Some(&r.lengths.buf), 0);
                }
                None => {
                    for i in 5..8 {
                        e.set_buffer(i, Some(&self.dummy), 0);
                    }
                }
            }
            e.set_threadgroup_memory_length(0, tile * 8);
            e.dispatch_thread_groups(size(tiles, num_vcols as u64), size(tile / 2, 1));
        });
        Ok(())
    }

    fn bitrev(
        &self,
        src: (&MetalBuffer, usize, Layout),
        dst: (&MetalBuffer, usize, Layout),
        num_vcols: u32,
        log_n: u32,
        scale: Option<u64>,
    ) -> Result<()> {
        let n = 1u64 << log_n;
        let args = BitrevArgs {
            sl: src.2,
            dl: dst.2,
            log_n,
            do_scale: scale.is_some() as u32,
            scale: scale.unwrap_or(0),
        };
        let t = Self::threads(&self.p.bitrev, 256);
        self.encode(|e| {
            e.set_compute_pipeline_state(&self.p.bitrev);
            e.set_buffer(0, Some(&src.0.buf), (src.1 * 8) as u64);
            e.set_buffer(1, Some(&dst.0.buf), (dst.1 * 8) as u64);
            bytes(e, 2, &args);
            e.dispatch_thread_groups(size(n.div_ceil(t), num_vcols as u64), size(t, 1));
        });
        Ok(())
    }

    fn powers(&self, out: &MetalBuffer, base: u64, n: u64) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let chunks = n.div_ceil(64);
        let t = Self::threads(&self.p.powers, 256);
        let args = PowArgs { base, n };
        self.encode(|e| {
            e.set_compute_pipeline_state(&self.p.powers);
            e.set_buffer(0, Some(&out.buf), 0);
            bytes(e, 1, &args);
            e.dispatch_thread_groups(size(chunks.div_ceil(t), 1), size(t, 1));
        });
        Ok(())
    }

    fn rows_init(&self, state: &MetalBuffer, num_rows: u64) -> Result<()> {
        // Shared memory: zero it from the CPU. The state buffer belongs to
        // one row hasher, which initializes it before any GPU work uses it,
        // so there is nothing to wait for (and the pipeline keeps running).
        unsafe {
            ptr::write_bytes(
                state.buf.contents() as *mut u64,
                0,
                (num_rows as usize) * 16,
            )
        };
        Ok(())
    }

    fn rows_absorb(
        &self,
        state: &MetalBuffer,
        num_rows: u64,
        pos: u32,
        data: (&MetalBuffer, usize, Layout),
        num_vcols: u32,
    ) -> Result<()> {
        if num_rows == 0 || num_vcols == 0 {
            return Ok(());
        }
        let args = RowsArgs {
            l: data.2,
            num_rows,
            pos,
            num_vcols,
        };
        let t = Self::threads(&self.p.rows_absorb, HASH_THREADS);
        self.encode(|e| {
            e.set_compute_pipeline_state(&self.p.rows_absorb);
            e.set_buffer(0, Some(&state.buf), 0);
            e.set_buffer(1, Some(&data.0.buf), (data.1 * 8) as u64);
            bytes(e, 2, &args);
            e.dispatch_thread_groups(size(num_rows.div_ceil(t), 1), size(t, 1));
        });
        Ok(())
    }

    fn rows_finish(
        &self,
        state: &MetalBuffer,
        num_rows: u64,
        pos: u32,
        digests: &MetalBuffer,
    ) -> Result<()> {
        if num_rows == 0 {
            return Ok(());
        }
        let args = RowsArgs {
            l: Layout {
                col_stride: 0,
                ncomp: 1,
                len: 0,
            },
            num_rows,
            pos,
            num_vcols: 0,
        };
        let t = Self::threads(&self.p.rows_finish, HASH_THREADS);
        self.encode(|e| {
            e.set_compute_pipeline_state(&self.p.rows_finish);
            e.set_buffer(0, Some(&state.buf), 0);
            e.set_buffer(1, Some(&digests.buf), 0);
            bytes(e, 2, &args);
            e.dispatch_thread_groups(size(num_rows.div_ceil(t), 1), size(t, 1));
        });
        Ok(())
    }

    fn merkle_level(&self, nodes: &MetalBuffer, first: u64, count: u64) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let args = MerkleArgs { first, count };
        let t = Self::threads(&self.p.merkle_level, HASH_THREADS);
        self.encode(|e| {
            e.set_compute_pipeline_state(&self.p.merkle_level);
            e.set_buffer(0, Some(&nodes.buf), 0);
            bytes(e, 1, &args);
            e.dispatch_thread_groups(size(count.div_ceil(t), 1), size(t, 1));
        });
        Ok(())
    }

    fn merkle_tail(&self, nodes: &MetalBuffer, m: u64) -> Result<()> {
        if m == 0 {
            return Ok(());
        }
        let args = MerkleArgs { first: m, count: 0 };
        let t = Self::threads(&self.p.merkle_tail, TAIL_THREADS.min(m));
        self.encode(|e| {
            e.set_compute_pipeline_state(&self.p.merkle_tail);
            e.set_buffer(0, Some(&nodes.buf), 0);
            bytes(e, 1, &args);
            e.dispatch_thread_groups(size(1, 1), size(t, 1));
        });
        Ok(())
    }
}

impl Drop for MetalBackend {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
