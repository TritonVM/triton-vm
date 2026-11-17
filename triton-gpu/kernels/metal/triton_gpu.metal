// Metal backend for triton-gpu (Apple silicon).
//
// The Rust host concatenates tg_prelude.h, tg_tip5_tables.h, tg_tip5_mds.h,
// tg_core.h and this file (with TG_METAL defined) and compiles the result at
// startup, so no Xcode command-line tools are needed.
//
// GPU features used:
//  - unified memory: buffers are MTLStorageModeShared, so the CPU and GPU
//    read and write the same pages; there is no upload/download copy over a
//    bus like on a discrete GPU
//  - multi-stage NTT passes in threadgroup memory (radix 2^k, k up to 11)
//  - fused coset/randomizer/zero-padding prescale and fused iNTT scaling
//  - Tip5 round constants in the constant address space, lookup table in
//    threadgroup memory
//  - the host encodes whole pipelines (all NTT passes, hashing, Merkle
//    levels) into one command buffer and commits once
#ifndef TG_METAL
#define TG_METAL
#endif
#include "../common/tg_core.h"

struct TgNttArgs {
    TgLayout sl;
    TgLayout dl;
    TgNttPass p;
    TgPrescale pre;
};

struct TgBitrevArgs {
    TgLayout sl;
    TgLayout dl;
    uint log_n;
    uint do_scale;
    ulong scale;
};

struct TgRowsArgs {
    TgLayout l;
    ulong num_rows;
    uint pos;
    uint num_vcols;
};

struct TgPowArgs {
    ulong base;
    ulong n;
};

struct TgMerkleArgs {
    ulong first;
    ulong count;
};

kernel void tg_ntt_pass_k(
    device const ulong *src         [[buffer(0)]],
    device ulong *dst               [[buffer(1)]],
    constant TgNttArgs &a           [[buffer(2)]],
    device const ulong *tw          [[buffer(3)]],
    device const ulong *op          [[buffer(4)]],
    device const ulong *rand        [[buffer(5)]],
    device const ulong *rand_off    [[buffer(6)]],
    device const uint *rand_len     [[buffer(7)]],
    threadgroup ulong *sm           [[threadgroup(0)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint3 tpg                       [[threads_per_threadgroup]])
{
    const uint tile = tgid.x;
    const uint v = tgid.y;
    const uint threads = tpg.x;
    for (uint h = 0; h < 2; ++h) {
        uint le = tid + h * threads;
        ulong gi = tg_ntt_tile_index(le, tile, a.p);
        sm[le] = tg_ntt_load(src, a.sl, v, gi, a.pre, op, rand, rand_off, rand_len);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < a.p.k; ++j) {
        tg_ntt_butterfly(sm, tid, j, tile, a.p, tw);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const ulong doff = tg_col_off(v, a.dl.ncomp, a.dl.col_stride);
    for (uint h = 0; h < 2; ++h) {
        uint le = tid + h * threads;
        ulong gi = tg_ntt_tile_index(le, tile, a.p);
        dst[doff + gi * a.dl.ncomp] = sm[le];
    }
}

kernel void tg_bitrev_k(
    device const ulong *src         [[buffer(0)]],
    device ulong *dst               [[buffer(1)]],
    constant TgBitrevArgs &a        [[buffer(2)]],
    uint2 gid                       [[thread_position_in_grid]])
{
    const ulong n = 1ul << a.log_n;
    const ulong i = gid.x;
    if (i >= n) { return; }
    const uint v = gid.y;
    ulong so = tg_col_off(v, a.sl.ncomp, a.sl.col_stride);
    ulong d = tg_col_off(v, a.dl.ncomp, a.dl.col_stride);
    ulong x = src[so + (ulong)tg_bitrev((uint)i, a.log_n) * a.sl.ncomp];
    dst[d + i * a.dl.ncomp] = a.do_scale ? bf_mul(x, a.scale) : x;
}

#define TG_POW_CHUNK 64
kernel void tg_powers_k(
    device ulong *out               [[buffer(0)]],
    constant TgPowArgs &a           [[buffer(1)]],
    uint gid                        [[thread_position_in_grid]])
{
    ulong start = (ulong)gid * TG_POW_CHUNK;
    if (start >= a.n) { return; }
    ulong acc = bf_pow(a.base, start);
    ulong end = min(start + TG_POW_CHUNK, a.n);
    for (ulong i = start; i < end; ++i) { out[i] = acc; acc = bf_mul(acc, a.base); }
}

#define TG_LOAD_LUT(lut, tid, tpg)                                                   \
    threadgroup uchar lut[256];                                                      \
    for (uint i_ = tid; i_ < 256; i_ += tpg) { lut[i_] = TG_TIP5_LOOKUP_TABLE[i_]; } \
    threadgroup_barrier(mem_flags::mem_threadgroup);

kernel void tg_rows_absorb_k(
    device ulong *state             [[buffer(0)]],
    device const ulong *data        [[buffer(1)]],
    constant TgRowsArgs &a          [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tpg                        [[threads_per_threadgroup]])
{
    TG_LOAD_LUT(lut, tid, tpg);
    if ((ulong)gid < a.num_rows) {
        tg_rows_absorb(state, gid, a.num_rows, a.pos, data, a.l, a.num_vcols, lut);
    }
}

kernel void tg_rows_finish_k(
    device const ulong *state       [[buffer(0)]],
    device ulong *digests           [[buffer(1)]],
    constant TgRowsArgs &a          [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tpg                        [[threads_per_threadgroup]])
{
    TG_LOAD_LUT(lut, tid, tpg);
    if ((ulong)gid < a.num_rows) {
        tg_rows_finish(state, gid, a.num_rows, a.pos, digests, lut);
    }
}

kernel void tg_merkle_level_k(
    device ulong *nodes             [[buffer(0)]],
    constant TgMerkleArgs &a        [[buffer(1)]],
    uint gid                        [[thread_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tpg                        [[threads_per_threadgroup]])
{
    TG_LOAD_LUT(lut, tid, tpg);
    if ((ulong)gid < a.count) { tg_merkle_node(nodes, a.first + gid, lut); }
}

// Top of the tree in one threadgroup: parents [m, 2m) up to the root.
kernel void tg_merkle_tail_k(
    device ulong *nodes             [[buffer(0)]],
    constant TgMerkleArgs &a        [[buffer(1)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint3 tpg                       [[threads_per_threadgroup]])
{
    TG_LOAD_LUT(lut, tid, tpg.x);
    for (ulong m = a.first; m >= 1; m >>= 1) {
        for (ulong i = tid; i < m; i += tpg.x) { tg_merkle_node(nodes, m + i, lut); }
        threadgroup_barrier(mem_flags::mem_device);
    }
}
