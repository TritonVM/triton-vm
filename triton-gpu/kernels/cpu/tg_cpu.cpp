// CPU reference backend: runs the shared per-thread kernel logic in loops that
// emulate the GPU launch geometry (tiles, threads, barriers). Used to verify
// the kernels against twenty-first on machines without a GPU, and as a
// functional fallback.
#define TG_CPU
#include "../common/tg_core.h"
#define TG_API_PREFIX tg_cpu_
#include "../common/tg_api.h"

#include <vector>

static TgNttPass to_pass(tg_ntt_pass_t p) { TgNttPass r = {p.log_n, p.stage, p.k, p.c, p.lob}; return r; }
static TgLayout to_layout(tg_layout_t l) { TgLayout r = {l.col_stride, l.ncomp, l.len}; return r; }
static TgPrescale to_pre(tg_prescale_t p) { TgPrescale r = {p.enabled, p.has_rand, p.zerofier}; return r; }

extern "C" int TG_API(ntt_pass)(void *, const tg_u64 *src, tg_layout_t sl_, tg_u64 *dst, tg_layout_t dl_,
                           tg_u32 num_vcols, tg_ntt_pass_t p_, const tg_u64 *tw, tg_prescale_t pre_,
                           const tg_u64 *offset_powers, const tg_u64 *rand, const tg_u64 *rand_off,
                           const tg_u32 *rand_len) {
    TgNttPass p = to_pass(p_);
    TgLayout sl = to_layout(sl_), dl = to_layout(dl_);
    TgPrescale pre = to_pre(pre_);
    const u32 tile_len = 1u << (p.k + p.c);
    const u32 threads = tile_len / 2;
    const u64 n = U64C(1) << p.log_n;
    const u64 tiles = n / tile_len;
    std::vector<u64> sm(tile_len);
    for (u32 v = 0; v < num_vcols; ++v) {
        u64 doff = tg_col_off(v, dl.ncomp, dl.col_stride);
        for (u64 tile = 0; tile < tiles; ++tile) {
            for (u32 tid = 0; tid < threads; ++tid) {
                for (u32 h = 0; h < 2; ++h) {
                    u32 le = tid + h * threads;
                    u64 gi = tg_ntt_tile_index(le, (u32)tile, p);
                    sm[le] = tg_ntt_load(src, sl, v, gi, pre, offset_powers, rand, rand_off, rand_len);
                }
            }
            for (u32 j = 0; j < p.k; ++j) {
                for (u32 tid = 0; tid < threads; ++tid) { tg_ntt_butterfly(sm.data(), tid, j, (u32)tile, p, tw); }
            }
            for (u32 tid = 0; tid < threads; ++tid) {
                for (u32 h = 0; h < 2; ++h) {
                    u32 le = tid + h * threads;
                    u64 gi = tg_ntt_tile_index(le, (u32)tile, p);
                    dst[doff + gi * dl.ncomp] = sm[le];
                }
            }
        }
    }
    return 0;
}

extern "C" int TG_API(bitrev)(void *, const tg_u64 *src, tg_layout_t sl_, tg_u64 *dst, tg_layout_t dl_,
                         tg_u32 num_vcols, tg_u32 log_n, tg_u32 do_scale, tg_u64 scale) {
    TgLayout sl = to_layout(sl_), dl = to_layout(dl_);
    const u64 n = U64C(1) << log_n;
    for (u32 v = 0; v < num_vcols; ++v) {
        u64 so = tg_col_off(v, sl.ncomp, sl.col_stride), d = tg_col_off(v, dl.ncomp, dl.col_stride);
        for (u64 i = 0; i < n; ++i) {
            u64 x = src[so + (u64)tg_bitrev((u32)i, log_n) * sl.ncomp];
            dst[d + i * dl.ncomp] = do_scale ? bf_mul(x, scale) : x;
        }
    }
    return 0;
}

extern "C" int TG_API(powers)(void *, tg_u64 *out, tg_u64 base, tg_u64 n) {
    u64 acc = TG_ONE_RAW;
    for (u64 i = 0; i < n; ++i) { out[i] = acc; acc = bf_mul(acc, base); }
    return 0;
}

extern "C" int TG_API(rows_init)(void *, tg_u64 *state, tg_u64 num_rows) {
    for (u64 i = 0; i < num_rows * TG_STATE; ++i) { state[i] = 0; }
    return 0;
}

extern "C" int TG_API(rows_absorb)(void *, tg_u64 *state, tg_u64 num_rows, tg_u32 pos, const tg_u64 *data,
                              tg_layout_t l, tg_u32 num_vcols) {
    for (u64 r = 0; r < num_rows; ++r) {
        tg_rows_absorb(state, r, num_rows, pos, data, to_layout(l), num_vcols, TG_TIP5_LOOKUP_TABLE);
    }
    return 0;
}

extern "C" int TG_API(rows_finish)(void *, const tg_u64 *state, tg_u64 num_rows, tg_u32 pos, tg_u64 *digests) {
    for (u64 r = 0; r < num_rows; ++r) { tg_rows_finish(state, r, num_rows, pos, digests, TG_TIP5_LOOKUP_TABLE); }
    return 0;
}

extern "C" int TG_API(merkle_level)(void *, tg_u64 *nodes, tg_u64 first, tg_u64 count) {
    for (u64 i = first; i < first + count; ++i) { tg_merkle_node(nodes, i, TG_TIP5_LOOKUP_TABLE); }
    return 0;
}

extern "C" int TG_API(merkle_tail)(void *, tg_u64 *nodes, tg_u64 m) {
    for (; m >= 1; m /= 2) {
        for (u64 i = m; i < 2 * m; ++i) { tg_merkle_node(nodes, i, TG_TIP5_LOOKUP_TABLE); }
    }
    return 0;
}
