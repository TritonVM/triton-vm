// C ABI shared by the CUDA backend and the CPU reference backend. The Rust
// side (src/ffi.rs) binds to these. All pointers are device pointers for
// CUDA, host pointers for the CPU reference. `stream` is a cudaStream_t for
// CUDA and ignored by the CPU reference.
#pragma once

#ifndef TG_API_PREFIX
#error "define TG_API_PREFIX (tg_cpu_ or tg_cuda_) before including tg_api.h"
#endif
#define TG_CAT2(a, b) a##b
#define TG_CAT(a, b) TG_CAT2(a, b)
#define TG_API(name) TG_CAT(TG_API_PREFIX, name)

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned long long tg_u64;
typedef unsigned int tg_u32;

typedef struct { tg_u32 log_n, stage, k, c, lob; } tg_ntt_pass_t;
typedef struct { tg_u64 col_stride; tg_u32 ncomp; tg_u32 len; } tg_layout_t;
typedef struct { tg_u32 enabled; tg_u32 has_rand; tg_u64 zerofier; } tg_prescale_t;

// Returns 0 on success, a backend error code otherwise.
int TG_API(ntt_pass)(void *stream,
                const tg_u64 *src, tg_layout_t src_layout,
                tg_u64 *dst, tg_layout_t dst_layout,
                tg_u32 num_vcols, tg_ntt_pass_t pass, const tg_u64 *twiddles,
                tg_prescale_t pre, const tg_u64 *offset_powers,
                const tg_u64 *rand, const tg_u64 *rand_off, const tg_u32 *rand_len);

int TG_API(bitrev)(void *stream,
              const tg_u64 *src, tg_layout_t src_layout,
              tg_u64 *dst, tg_layout_t dst_layout,
              tg_u32 num_vcols, tg_u32 log_n, tg_u32 do_scale, tg_u64 scale);

int TG_API(powers)(void *stream, tg_u64 *out, tg_u64 base, tg_u64 n);

int TG_API(rows_init)(void *stream, tg_u64 *state, tg_u64 num_rows);
int TG_API(rows_absorb)(void *stream, tg_u64 *state, tg_u64 num_rows, tg_u32 pos,
                   const tg_u64 *data, tg_layout_t layout, tg_u32 num_vcols);
int TG_API(rows_finish)(void *stream, const tg_u64 *state, tg_u64 num_rows, tg_u32 pos,
                   tg_u64 *digests);

// nodes[i] for i in [first, first + count), one level.
int TG_API(merkle_level)(void *stream, tg_u64 *nodes, tg_u64 first, tg_u64 count);
// All levels from parents [m, 2m) up to the root, in one launch (m <= 1024).
int TG_API(merkle_tail)(void *stream, tg_u64 *nodes, tg_u64 m);

#ifdef __cplusplus
}
#endif
