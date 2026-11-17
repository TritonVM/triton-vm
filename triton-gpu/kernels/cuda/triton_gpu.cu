// CUDA backend for triton-gpu.
//
// GPU features used:
//  - multi-stage NTT passes in shared memory (radix 2^k, k up to 11 per pass,
//    one global round-trip per 8-11 butterfly layers), coalesced 2^c-element
//    runs on load/store
//  - fused kernels: coset scaling + trace randomizer + zero padding happen in
//    the first NTT pass's loads; the iNTT's 1/n scaling happens in the
//    bit-reversal gather
//  - Tip5 round constants in __constant__ memory (uniform reads, broadcast),
//    split-and-lookup table copied into shared memory per block (divergent
//    byte lookups without constant-cache serialization)
//  - __launch_bounds__ to keep register allocation in check for Tip5's 16-word
//    state; read-only (__restrict__ const) inputs for the non-coherent cache
//  - everything is enqueued on one non-blocking stream: no host round-trips
//    between kernels; the host only syncs when it reads results
//  - streaming sponge state (structure-of-arrays) so huge tables are hashed
//    in column chunks with bounded device memory
#define TG_CUDA
#include "../common/tg_core.h"
#define TG_API_PREFIX tg_cuda_
#include "../common/tg_api.h"

#include <cuda_runtime.h>

static __device__ __forceinline__ TgNttPass to_pass(tg_ntt_pass_t p) { TgNttPass r = {p.log_n, p.stage, p.k, p.c, p.lob}; return r; }
static __host__ __device__ __forceinline__ TgLayout to_layout(tg_layout_t l) { TgLayout r = {l.col_stride, l.ncomp, l.len}; return r; }
static __device__ __forceinline__ TgPrescale to_pre(tg_prescale_t p) { TgPrescale r = {p.enabled, p.has_rand, p.zerofier}; return r; }

#define TG_HASH_THREADS 128
#define TG_MAX_GRID_Y 65535u

static int tg_err() {
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}

// ------------------------------------------------------------------ NTT ----

extern __shared__ u64 tg_ntt_sm[];

__global__ void __launch_bounds__(1024)
k_ntt_pass(const u64 *src, tg_layout_t sl_, u64 *dst, tg_layout_t dl_, u32 vcol_base,
           tg_ntt_pass_t p_, const u64 *__restrict__ tw, tg_prescale_t pre_,
           const u64 *__restrict__ offset_powers, const u64 *__restrict__ rand,
           const u64 *__restrict__ rand_off, const u32 *__restrict__ rand_len)
{
    const TgNttPass p = to_pass(p_);
    const TgLayout sl = to_layout(sl_), dl = to_layout(dl_);
    const TgPrescale pre = to_pre(pre_);
    const u32 tile = blockIdx.x;
    const u32 v = vcol_base + blockIdx.y;
    const u32 tid = threadIdx.x;
    const u32 threads = blockDim.x;

    #pragma unroll
    for (u32 h = 0; h < 2; ++h) {
        u32 le = tid + h * threads;
        u64 gi = tg_ntt_tile_index(le, tile, p);
        tg_ntt_sm[le] = tg_ntt_load(src, sl, v, gi, pre, offset_powers, rand, rand_off, rand_len);
    }
    __syncthreads();
    for (u32 j = 0; j < p.k; ++j) {
        tg_ntt_butterfly(tg_ntt_sm, tid, j, tile, p, tw);
        __syncthreads();
    }
    const u64 doff = tg_col_off(v, dl.ncomp, dl.col_stride);
    #pragma unroll
    for (u32 h = 0; h < 2; ++h) {
        u32 le = tid + h * threads;
        u64 gi = tg_ntt_tile_index(le, tile, p);
        dst[doff + gi * dl.ncomp] = tg_ntt_sm[le];
    }
}

__global__ void k_bitrev(const u64 *__restrict__ src, tg_layout_t sl_, u64 *__restrict__ dst, tg_layout_t dl_,
                         u32 vcol_base, u32 log_n, u32 do_scale, u64 scale)
{
    const TgLayout sl = to_layout(sl_), dl = to_layout(dl_);
    const u64 n = U64C(1) << log_n;
    const u32 v = vcol_base + blockIdx.y;
    const u64 so = tg_col_off(v, sl.ncomp, sl.col_stride), d = tg_col_off(v, dl.ncomp, dl.col_stride);
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) {
        u64 x = src[so + (u64)tg_bitrev((u32)i, log_n) * sl.ncomp];
        dst[d + i * dl.ncomp] = do_scale ? bf_mul(x, scale) : x;
    }
}

#define TG_POW_CHUNK 64
__global__ void k_powers(u64 *out, u64 base, u64 n) {
    u64 start = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * TG_POW_CHUNK;
    if (start >= n) return;
    u64 acc = bf_pow(base, start);
    u64 end = start + TG_POW_CHUNK < n ? start + TG_POW_CHUNK : n;
    for (u64 i = start; i < end; ++i) { out[i] = acc; acc = bf_mul(acc, base); }
}

extern "C" int TG_API(ntt_pass)(void *stream, const tg_u64 *src, tg_layout_t sl, tg_u64 *dst, tg_layout_t dl,
                           tg_u32 num_vcols, tg_ntt_pass_t p, const tg_u64 *tw, tg_prescale_t pre,
                           const tg_u64 *offset_powers, const tg_u64 *rand, const tg_u64 *rand_off,
                           const tg_u32 *rand_len) {
    const u32 tile_len = 1u << (p.k + p.c);
    const u64 tiles = (U64C(1) << p.log_n) / tile_len;
    for (u32 base = 0; base < num_vcols; base += TG_MAX_GRID_Y) {
        u32 cnt = num_vcols - base < TG_MAX_GRID_Y ? num_vcols - base : TG_MAX_GRID_Y;
        dim3 grid((unsigned)tiles, cnt, 1);
        k_ntt_pass<<<grid, tile_len / 2, tile_len * sizeof(u64), (cudaStream_t)stream>>>(
            src, sl, dst, dl, base, p, tw, pre, offset_powers, rand, rand_off, rand_len);
    }
    return tg_err();
}

extern "C" int TG_API(bitrev)(void *stream, const tg_u64 *src, tg_layout_t sl, tg_u64 *dst, tg_layout_t dl,
                         tg_u32 num_vcols, tg_u32 log_n, tg_u32 do_scale, tg_u64 scale) {
    const u64 n = U64C(1) << log_n;
    const u32 threads = 256;
    u64 blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    for (u32 base = 0; base < num_vcols; base += TG_MAX_GRID_Y) {
        u32 cnt = num_vcols - base < TG_MAX_GRID_Y ? num_vcols - base : TG_MAX_GRID_Y;
        k_bitrev<<<dim3((unsigned)blocks, cnt, 1), threads, 0, (cudaStream_t)stream>>>(
            src, sl, dst, dl, base, log_n, do_scale, scale);
    }
    return tg_err();
}

extern "C" int TG_API(powers)(void *stream, tg_u64 *out, tg_u64 base, tg_u64 n) {
    if (n == 0) return 0;
    u64 chunks = (n + TG_POW_CHUNK - 1) / TG_POW_CHUNK;
    u32 threads = 256;
    k_powers<<<(unsigned)((chunks + threads - 1) / threads), threads, 0, (cudaStream_t)stream>>>(out, base, n);
    return tg_err();
}

// ------------------------------------------------------------ Row hashing --

#define TG_LOAD_LUT_TO_SHARED(lut)                                              \
    __shared__ u8 lut[256];                                                     \
    for (u32 i_ = threadIdx.x; i_ < 256; i_ += blockDim.x) { lut[i_] = TG_TIP5_LOOKUP_TABLE[i_]; } \
    __syncthreads();

__global__ void k_rows_init(u64 *state, u64 total) {
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (u64)gridDim.x * blockDim.x) state[i] = 0;
}

__global__ void __launch_bounds__(TG_HASH_THREADS)
k_rows_absorb(u64 *state, u64 num_rows, u32 pos, const u64 *__restrict__ data, tg_layout_t l, u32 num_vcols) {
    TG_LOAD_LUT_TO_SHARED(lut);
    u64 row = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) tg_rows_absorb(state, row, num_rows, pos, data, to_layout(l), num_vcols, lut);
}

__global__ void __launch_bounds__(TG_HASH_THREADS)
k_rows_finish(const u64 *state, u64 num_rows, u32 pos, u64 *digests) {
    TG_LOAD_LUT_TO_SHARED(lut);
    u64 row = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) tg_rows_finish(state, row, num_rows, pos, digests, lut);
}

extern "C" int TG_API(rows_init)(void *stream, tg_u64 *state, tg_u64 num_rows) {
    u64 total = num_rows * TG_STATE;
    if (total == 0) return 0;
    u64 blocks = (total + 255) / 256; if (blocks > 65535) blocks = 65535;
    k_rows_init<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(state, total);
    return tg_err();
}

extern "C" int TG_API(rows_absorb)(void *stream, tg_u64 *state, tg_u64 num_rows, tg_u32 pos, const tg_u64 *data,
                              tg_layout_t l, tg_u32 num_vcols) {
    if (num_rows == 0 || num_vcols == 0) return 0;
    u64 blocks = (num_rows + TG_HASH_THREADS - 1) / TG_HASH_THREADS;
    k_rows_absorb<<<(unsigned)blocks, TG_HASH_THREADS, 0, (cudaStream_t)stream>>>(state, num_rows, pos, data, l, num_vcols);
    return tg_err();
}

extern "C" int TG_API(rows_finish)(void *stream, const tg_u64 *state, tg_u64 num_rows, tg_u32 pos, tg_u64 *digests) {
    if (num_rows == 0) return 0;
    u64 blocks = (num_rows + TG_HASH_THREADS - 1) / TG_HASH_THREADS;
    k_rows_finish<<<(unsigned)blocks, TG_HASH_THREADS, 0, (cudaStream_t)stream>>>(state, num_rows, pos, digests);
    return tg_err();
}

// ------------------------------------------------------------ Merkle tree --

__global__ void __launch_bounds__(TG_HASH_THREADS) k_merkle_level(u64 *nodes, u64 first, u64 count) {
    TG_LOAD_LUT_TO_SHARED(lut);
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) tg_merkle_node(nodes, first + i, lut);
}

// The top of the tree in a single block: one launch instead of ~10 tiny ones.
// 256 threads: Tip5's state needs the registers; more threads would spill.
#define TG_TAIL_THREADS 256
__global__ void __launch_bounds__(TG_TAIL_THREADS) k_merkle_tail(u64 *nodes, u64 m) {
    TG_LOAD_LUT_TO_SHARED(lut);
    for (; m >= 1; m >>= 1) {
        for (u64 i = threadIdx.x; i < m; i += blockDim.x) tg_merkle_node(nodes, m + i, lut);
        __threadfence_block();
        __syncthreads();
    }
}

extern "C" int TG_API(merkle_level)(void *stream, tg_u64 *nodes, tg_u64 first, tg_u64 count) {
    if (count == 0) return 0;
    u64 blocks = (count + TG_HASH_THREADS - 1) / TG_HASH_THREADS;
    k_merkle_level<<<(unsigned)blocks, TG_HASH_THREADS, 0, (cudaStream_t)stream>>>(nodes, first, count);
    return tg_err();
}

extern "C" int TG_API(merkle_tail)(void *stream, tg_u64 *nodes, tg_u64 m) {
    if (m == 0) return 0;
    unsigned threads = m < TG_TAIL_THREADS ? (unsigned)m : TG_TAIL_THREADS;
    k_merkle_tail<<<1, threads, 0, (cudaStream_t)stream>>>(nodes, m);
    return tg_err();
}

// ------------------------------------------------- memory and streams -----

extern "C" int tg_cuda_init(int device, void **stream_out, char *name, int name_len) {
    if (cudaSetDevice(device) != cudaSuccess) return tg_err();
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return tg_err();
    int i = 0;
    for (; i < name_len - 1 && prop.name[i]; ++i) name[i] = prop.name[i];
    if (name_len > 0) name[i] = 0;
    cudaStream_t s;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) return tg_err();
    *stream_out = (void *)s;
    return 0;
}

extern "C" int tg_cuda_destroy(void *stream) { cudaStreamDestroy((cudaStream_t)stream); return tg_err(); }

extern "C" int tg_cuda_malloc(void *stream, void **ptr, tg_u64 bytes) {
    // Stream-ordered allocation: no device-wide sync, memory comes from a pool.
    if (bytes == 0) { *ptr = nullptr; return 0; }
    return (int)cudaMallocAsync(ptr, bytes, (cudaStream_t)stream);
}

extern "C" int tg_cuda_free(void *stream, void *ptr) {
    if (!ptr) return 0;
    return (int)cudaFreeAsync(ptr, (cudaStream_t)stream);
}

extern "C" int tg_cuda_htod(void *stream, void *dst, const void *src, tg_u64 bytes) {
    if (bytes == 0) return 0;
    return (int)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream);
}

extern "C" int tg_cuda_dtoh(void *stream, void *dst, const void *src, tg_u64 bytes) {
    if (bytes == 0) return 0;
    int e = (int)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    if (e) return e;
    return (int)cudaStreamSynchronize((cudaStream_t)stream);
}

extern "C" int tg_cuda_sync(void *stream) { return (int)cudaStreamSynchronize((cudaStream_t)stream); }

// Page-lock a host range so that copies to/from it are true async DMA.
extern "C" int tg_cuda_pin(void *ptr, tg_u64 bytes) { return (int)cudaHostRegister(ptr, bytes, cudaHostRegisterDefault); }
extern "C" int tg_cuda_unpin(void *ptr) { return (int)cudaHostUnregister(ptr); }
