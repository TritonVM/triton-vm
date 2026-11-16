//! Bindings to the C ABI in `kernels/common/tg_api.h`, shared by the CPU
//! reference and the CUDA backend (with symbol prefixes `tg_cpu_` and
//! `tg_cuda_`).

#![allow(dead_code)]

macro_rules! tg_api {
    ($modname:ident, $ntt_pass:ident, $bitrev:ident, $powers:ident, $rows_init:ident,
     $rows_absorb:ident, $rows_finish:ident, $merkle_level:ident, $merkle_tail:ident) => {
        pub mod $modname {
            use crate::backend::{Layout, NttPass, Prescale};
            use std::ffi::c_void;
            extern "C" {
                #[link_name = stringify!($ntt_pass)]
                pub fn ntt_pass(
                    stream: *mut c_void,
                    src: *const u64,
                    src_layout: Layout,
                    dst: *mut u64,
                    dst_layout: Layout,
                    num_vcols: u32,
                    pass: NttPass,
                    twiddles: *const u64,
                    pre: Prescale,
                    offset_powers: *const u64,
                    rand: *const u64,
                    rand_off: *const u64,
                    rand_len: *const u32,
                ) -> i32;
                #[link_name = stringify!($bitrev)]
                pub fn bitrev(
                    stream: *mut c_void,
                    src: *const u64,
                    src_layout: Layout,
                    dst: *mut u64,
                    dst_layout: Layout,
                    num_vcols: u32,
                    log_n: u32,
                    do_scale: u32,
                    scale: u64,
                ) -> i32;
                #[link_name = stringify!($powers)]
                pub fn powers(stream: *mut c_void, out: *mut u64, base: u64, n: u64) -> i32;
                #[link_name = stringify!($rows_init)]
                pub fn rows_init(stream: *mut c_void, state: *mut u64, num_rows: u64) -> i32;
                #[link_name = stringify!($rows_absorb)]
                pub fn rows_absorb(
                    stream: *mut c_void,
                    state: *mut u64,
                    num_rows: u64,
                    pos: u32,
                    data: *const u64,
                    layout: Layout,
                    num_vcols: u32,
                ) -> i32;
                #[link_name = stringify!($rows_finish)]
                pub fn rows_finish(
                    stream: *mut c_void,
                    state: *const u64,
                    num_rows: u64,
                    pos: u32,
                    digests: *mut u64,
                ) -> i32;
                #[link_name = stringify!($merkle_level)]
                pub fn merkle_level(
                    stream: *mut c_void,
                    nodes: *mut u64,
                    first: u64,
                    count: u64,
                ) -> i32;
                #[link_name = stringify!($merkle_tail)]
                pub fn merkle_tail(stream: *mut c_void, nodes: *mut u64, m: u64) -> i32;
            }
        }
    };
}

#[cfg(feature = "cpu")]
tg_api!(
    cpu,
    tg_cpu_ntt_pass,
    tg_cpu_bitrev,
    tg_cpu_powers,
    tg_cpu_rows_init,
    tg_cpu_rows_absorb,
    tg_cpu_rows_finish,
    tg_cpu_merkle_level,
    tg_cpu_merkle_tail
);

#[cfg(feature = "cuda")]
tg_api!(
    cuda,
    tg_cuda_ntt_pass,
    tg_cuda_bitrev,
    tg_cuda_powers,
    tg_cuda_rows_init,
    tg_cuda_rows_absorb,
    tg_cuda_rows_finish,
    tg_cuda_merkle_level,
    tg_cuda_merkle_tail
);

#[cfg(feature = "cuda")]
pub mod cuda_mem {
    use std::ffi::{c_char, c_void};
    extern "C" {
        pub fn tg_cuda_init(
            device: i32,
            stream_out: *mut *mut c_void,
            name: *mut c_char,
            name_len: i32,
        ) -> i32;
        pub fn tg_cuda_destroy(stream: *mut c_void) -> i32;
        pub fn tg_cuda_malloc(stream: *mut c_void, ptr: *mut *mut c_void, bytes: u64) -> i32;
        pub fn tg_cuda_free(stream: *mut c_void, ptr: *mut c_void) -> i32;
        pub fn tg_cuda_htod(
            stream: *mut c_void,
            dst: *mut c_void,
            src: *const c_void,
            bytes: u64,
        ) -> i32;
        pub fn tg_cuda_dtoh(
            stream: *mut c_void,
            dst: *mut c_void,
            src: *const c_void,
            bytes: u64,
        ) -> i32;
        pub fn tg_cuda_sync(stream: *mut c_void) -> i32;
        pub fn tg_cuda_pin(ptr: *mut c_void, bytes: u64) -> i32;
        pub fn tg_cuda_unpin(ptr: *mut c_void) -> i32;
    }
}
