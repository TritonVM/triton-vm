// Shared kernel logic for Metal, CUDA and the CPU reference.
//
// Every function here is written per *thread*: the backends only map their
// thread / threadgroup indices onto these functions and place the barriers.
// The CPU reference runs the very same functions in loops, which is how the
// arithmetic and all index mappings are checked against twenty-first on any
// machine, bit for bit.
//
// Field elements are kept in twenty-first's raw Montgomery representation
// (`BFieldElement::raw_u64`). A `&[BFieldElement]` is uploaded as-is: no
// conversion on the way in or out.
#pragma once

#include "tg_prelude.h"
#include "tg_arith.h"
#include "tg_tip5_tables.h"
#include "tg_tip5_mds.h"

// ---------------------------------------------------------------- B-field --

#define TG_P U64C(0xffffffff00000001)
#define TG_EPS U64C(0xffffffff)          // 2^64 - P = 2^32 - 1
#define TG_ONE_RAW U64C(0xffffffff)      // BFieldElement::ONE in Montgomery form

// Exactly twenty-first's `Add`: a - (P - b), corrected on borrow. The exact
// sequence matters, since Tip5 relies on it to repair degenerate values.
TG_FN u64 bf_add(u64 a, u64 b) {
    u64 t = TG_P - b;
    u64 x1 = a - t;
    return (a < t) ? x1 + TG_P : x1;
}

TG_FN u64 bf_sub(u64 a, u64 b) {
    u64 x1 = a - b;
    return (a < b) ? x1 - TG_EPS : x1;
}

// Exactly twenty-first's `BFieldElement::montyred`, on a 128-bit (xh:xl).
TG_FN u64 bf_montyred(u64 xl, u64 xh) {
    u64 a = xl + (xl << 32);
    u64 e = (a < xl) ? 1 : 0;
    u64 b = a - (a >> 32) - e;
    u64 r = xh - b;
    return (xh < b) ? r - TG_EPS : r;
}

TG_FN u64 bf_mul(u64 a, u64 b) {
    TgU128 p = tg_mul_wide(a, b);   // 32-bit pieces on Metal, see tg_arith.h
    return bf_montyred(p.lo, p.hi);
}

TG_FN u64 bf_pow(u64 base, u64 exp) {
    u64 acc = TG_ONE_RAW;
    while (exp != 0) {
        if (exp & 1) { acc = bf_mul(acc, base); }
        base = bf_mul(base, base);
        exp >>= 1;
    }
    return acc;
}

// ------------------------------------------------------------------- Tip5 --

#define TG_STATE 16
#define TG_RATE 10
#define TG_DIGEST 5

TG_FN u64 tg_split_and_lookup(u64 x, TG_LUT_SPACE const u8 *lut) {
    u64 r = 0;
    TG_UNROLL
    for (int i = 0; i < 8; ++i) {
        r |= ((u64)lut[(x >> (8 * i)) & 0xff]) << (8 * i);
    }
    return r;
}

TG_FN void tg_tip5_round(TG_THREAD u64 *s, int round, TG_LUT_SPACE const u8 *lut) {
    // S-box layer: 4 split-and-lookup, 12 power maps x^7.
    TG_UNROLL
    for (int i = 0; i < 4; ++i) { s[i] = tg_split_and_lookup(s[i], lut); }
    TG_UNROLL
    for (int i = 4; i < TG_STATE; ++i) {
        u64 sq = bf_mul(s[i], s[i]);
        u64 qu = bf_mul(sq, sq);
        s[i] = bf_mul(s[i], bf_mul(sq, qu));
    }

    // MDS layer, exactly as `Tip5::mds_generated`.
    u64 lo[TG_STATE], hi[TG_STATE], lo_out[TG_STATE], hi_out[TG_STATE];
    TG_UNROLL
    for (int i = 0; i < TG_STATE; ++i) { hi[i] = s[i] >> 32; lo[i] = s[i] & U64C(0xffffffff); }
    tg_tip5_mds_generated_function(lo, lo_out);
    tg_tip5_mds_generated_function(hi, hi_out);
    TG_UNROLL
    for (int r = 0; r < TG_STATE; ++r) {
        // s = (lo >> 4) + (hi << 28) as a 128-bit integer
        u64 a = lo_out[r] >> 4;
        u64 b_lo = hi_out[r] << 28;
        u64 s_lo = a + b_lo;
        u64 s_hi = (hi_out[r] >> 36) + ((s_lo < a) ? 1 : 0);
        u64 res = s_lo + ((s_hi << 32) - s_hi);   // s_hi * (2^32 - 1), no multiply
        s[r] = (res < s_lo) ? res + U64C(0xffffffff) : res;
    }

    // Round constants (constant memory: uniform across all threads).
    TG_UNROLL
    for (int i = 0; i < TG_STATE; ++i) {
        s[i] = bf_add(s[i], TG_TIP5_ROUND_CONSTANTS[round * TG_STATE + i]);
    }
}

TG_FN void tg_tip5_permutation(TG_THREAD u64 *s, TG_LUT_SPACE const u8 *lut) {
    // Rounds are deliberately not unrolled: one round's code is reused, which
    // keeps register pressure and instruction cache footprint low.
    for (int r = 0; r < 5; ++r) { tg_tip5_round(s, r, lut); }
}

// ---------------------------------------------------------- Column layout --
//
// A batch of columns is a set of "virtual" B-field columns. Element `i` of
// virtual column `v` lives at `base[col_off(v) + i * ncomp]`, where
// `col_off(v) = (v / ncomp) * col_stride + v % ncomp`. B-field columns have
// ncomp = 1; X-field columns are 3 interleaved B-field columns (ncomp = 3).

TG_FN u64 tg_col_off(u32 v, u32 ncomp, u64 col_stride) {
    return (u64)(v / ncomp) * col_stride + (u64)(v % ncomp);
}

// -------------------------------------------------------------------- NTT --
//
// Decimation-in-frequency, radix 2^k per pass in shared memory, natural-order
// input, bit-reversed output; a final pass gathers into natural order. Since
// field arithmetic is exact and every operation keeps values canonical, the
// result equals twenty-first's `ntt` / `intt` bit for bit.
//
// A pass covers DIF stages [s, s+k). Its tile holds 2^k "butterfly" positions
// times 2^C consecutive low indices, so that global loads come in runs of 2^C
// contiguous elements (coalescing). lob = L - s - k is the number of index
// bits below the pass's bits.
//
// Twiddle table `tw` (length n): tw[h + t] = omega_n^(t * n / (2h)) for the
// stage with half-distance h, so each stage reads a contiguous range.

struct TgNttPass {
    u32 log_n;      // L
    u32 stage;      // s
    u32 k;          // stages in this pass
    u32 c;          // log2 of consecutive low indices per tile
    u32 lob;        // L - s - k
};

struct TgLayout {
    u64 col_stride;  // distance between columns, in u64s
    u32 ncomp;       // 1 for B-field, 3 for X-field
    u32 len;         // number of valid elements per column (rest reads as 0)
};

// Optional fused pre-processing applied while loading the first pass:
//   x_i = (src_i + [i < rand_len] * rand_i * zerofier) * offset^i
// which is exactly triton-vm's `evaluate_randomized_interpolant_into` (with
// zero padding when the source is shorter than the transform).
struct TgPrescale {
    u32 enabled;
    u32 has_rand;
    u64 zerofier;     // raw
};

TG_FN u64 tg_ntt_tile_index(u32 le, u32 tile, TgNttPass p) {
    u32 cmask = (1u << p.c) - 1u;
    u64 c = le & cmask;
    u64 e = le >> p.c;
    u32 lo_groups_log = p.lob - p.c;
    u64 lo_base = ((u64)tile & ((U64C(1) << lo_groups_log) - 1)) << p.c;
    u64 high = (u64)tile >> lo_groups_log;
    return (high << (p.log_n - p.stage)) | (e << p.lob) | (lo_base + c);
}

TG_FN u64 tg_ntt_load(
    TG_GLOBAL const u64 *src, TgLayout sl, u32 v, u64 gi,
    TgPrescale pre, TG_GLOBAL const u64 *offset_powers,
    TG_GLOBAL const u64 *rand, TG_GLOBAL const u64 *rand_off, TG_GLOBAL const u32 *rand_len)
{
    u64 x = 0;
    if (gi < (u64)sl.len) { x = src[tg_col_off(v, sl.ncomp, sl.col_stride) + gi * sl.ncomp]; }
    if (pre.enabled) {
        if (pre.has_rand) {
            u32 col = v / sl.ncomp;
            if (gi < (u64)rand_len[col]) {
                u64 r = rand[rand_off[col] + gi * sl.ncomp + v % sl.ncomp];
                x = bf_add(x, bf_mul(r, pre.zerofier));
            }
        }
        x = bf_mul(x, offset_powers[gi]);
    }
    return x;
}

// One butterfly of local stage j for thread `tid` of a tile.
TG_FN void tg_ntt_butterfly(
    TG_SHARED u64 *sm, u32 tid, u32 j, u32 tile, TgNttPass p, TG_GLOBAL const u64 *tw)
{
    u32 cmask = (1u << p.c) - 1u;
    u32 c = tid & cmask;
    u32 bb = tid >> p.c;
    u32 sh = p.k - 1u - j;
    u32 he = 1u << sh;
    u32 e0 = ((bb >> sh) << (sh + 1u)) | (bb & (he - 1u));
    u32 i0 = (e0 << p.c) | c;
    u32 i1 = ((e0 + he) << p.c) | c;

    u32 lo_groups_log = p.lob - p.c;
    u64 lo_base = ((u64)tile & ((U64C(1) << lo_groups_log) - 1)) << p.c;
    u64 t = ((u64)(e0 & (he - 1u)) << p.lob) | (lo_base + c);
    u64 h = U64C(1) << (sh + p.lob);

    u64 a = sm[i0];
    u64 b = sm[i1];
    sm[i0] = bf_add(a, b);
    sm[i1] = bf_mul(bf_sub(a, b), tw[h + t]);
}

TG_FN u32 tg_bitrev(u32 i, u32 log_n) {
    if (log_n == 0) { return 0; }
    u32 r = 0;
#if defined(TG_METAL)
    r = reverse_bits(i);
#elif defined(TG_CUDA)
    r = __brev(i);
#else
    for (int b = 0; b < 32; ++b) { r |= ((i >> b) & 1u) << (31 - b); }
#endif
    return r >> (32u - log_n);
}

// ------------------------------------------------------------ Row hashing --
//
// Streaming `Tip5::hash_varlen` of every row of a column-major table. One
// thread per row; the sponge state lives in a structure-of-arrays buffer
// (state[i * num_rows + row]) between launches, so a table can be absorbed in
// column chunks of any size: memory stays bounded by the chunk, not the table.
// `pos` (the next rate slot to overwrite) is uniform across rows and tracked
// by the host.

TG_FN void tg_rows_load_state(TG_THREAD u64 *s, TG_GLOBAL const u64 *state, u64 row, u64 num_rows) {
    TG_UNROLL
    for (int i = 0; i < TG_STATE; ++i) { s[i] = state[(u64)i * num_rows + row]; }
}

TG_FN void tg_rows_store_state(TG_THREAD const u64 *s, TG_GLOBAL u64 *state, u64 row, u64 num_rows) {
    TG_UNROLL
    for (int i = 0; i < TG_STATE; ++i) { state[(u64)i * num_rows + row] = s[i]; }
}

TG_FN void tg_rows_absorb(
    TG_GLOBAL u64 *state, u64 row, u64 num_rows, u32 pos,
    TG_GLOBAL const u64 *data, TgLayout l, u32 num_vcols, TG_LUT_SPACE const u8 *lut)
{
    u64 s[TG_STATE];
    tg_rows_load_state(s, state, row, num_rows);
    for (u32 v = 0; v < num_vcols; ++v) {
        const u64 x = data[tg_col_off(v, l.ncomp, l.col_stride) + row * l.ncomp];
        TG_UNROLL
        for (u32 i = 0; i < TG_RATE; ++i) { if (i == pos) { s[i] = x; } }
        pos += 1;
        if (pos == TG_RATE) { tg_tip5_permutation(s, lut); pos = 0; }
    }
    tg_rows_store_state(s, state, row, num_rows);
}

TG_FN void tg_rows_finish(
    TG_GLOBAL const u64 *state, u64 row, u64 num_rows, u32 pos,
    TG_GLOBAL u64 *digests, TG_LUT_SPACE const u8 *lut)
{
    u64 s[TG_STATE];
    tg_rows_load_state(s, state, row, num_rows);
    TG_UNROLL
    for (u32 i = 0; i < TG_RATE; ++i) {
        if (i == pos) { s[i] = TG_ONE_RAW; } else if (i > pos) { s[i] = 0; }
    }
    tg_tip5_permutation(s, lut);
    TG_UNROLL
    for (int i = 0; i < TG_DIGEST; ++i) { digests[row * TG_DIGEST + i] = s[i]; }
}

// ------------------------------------------------------------ Merkle tree --
//
// twenty-first layout: nodes[1] is the root, nodes[i] = hash_pair(nodes[2i],
// nodes[2i+1]), leafs at [n, 2n). Each node is 5 consecutive u64s.

TG_FN void tg_merkle_node(TG_GLOBAL u64 *nodes, u64 i, TG_LUT_SPACE const u8 *lut) {
    u64 s[TG_STATE];
    TG_UNROLL
    for (int k = 0; k < TG_DIGEST; ++k) {
        s[k] = nodes[(2 * i) * TG_DIGEST + k];
        s[TG_DIGEST + k] = nodes[(2 * i + 1) * TG_DIGEST + k];
    }
    TG_UNROLL
    for (int k = TG_RATE; k < TG_STATE; ++k) { s[k] = TG_ONE_RAW; }  // fixed-length domain
    tg_tip5_permutation(s, lut);
    TG_UNROLL
    for (int k = 0; k < TG_DIGEST; ++k) { nodes[i * TG_DIGEST + k] = s[k]; }
}
