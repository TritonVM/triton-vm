// 64-bit integer multiplication built from 32-bit pieces.
//
// Apple GPUs have no native 64-bit integer multiplier: a u64 * u64 or a
// 64-bit mulhi is emulated with many 32-bit instructions. With
// TG_WIDE_MUL_32 defined (always on for Metal), the full 128-bit product is
// formed from exactly four 32x32 -> 64 products (each one native mul + one
// native mulhi), and multiplications by the small constants of Tip5's MDS
// matrix use three native 32-bit instructions.
//
// The CPU reference can be built with TG_WIDE_MUL_32 too (environment
// variable TRITON_GPU_WIDE_MUL_32=1), which is how this path is verified
// against twenty-first on machines without an Apple GPU.
#pragma once

#include "tg_prelude.h"

#if defined(TG_METAL) && !defined(TG_WIDE_MUL_32)
  #define TG_WIDE_MUL_32 1
#endif

#if defined(TG_METAL)
  #define TG_MULHI32(a, b) mulhi((u32)(a), (u32)(b))
#elif defined(TG_CUDA)
  #define TG_MULHI32(a, b) __umulhi((u32)(a), (u32)(b))
#else
  #define TG_MULHI32(a, b) ((u32)(((u64)(u32)(a) * (u64)(u32)(b)) >> 32))
#endif

struct TgU128 {
    u64 lo;
    u64 hi;
};

// x * k mod 2^64 for a 32-bit k: three native 32-bit instructions.
TG_FN u64 tg_mul_u64_u32(u64 x, u32 k) {
    u32 x0 = (u32)x;
    u32 x1 = (u32)(x >> 32);
    u32 lo = x0 * k;
    u32 hi = TG_MULHI32(x0, k) + x1 * k;   // wrapping, mod 2^32
    return ((u64)hi << 32) | (u64)lo;
}

// The full 128-bit product a * b.
TG_FN TgU128 tg_mul_wide(u64 a, u64 b) {
    TgU128 r;
#if defined(TG_WIDE_MUL_32)
    u32 a0 = (u32)a, a1 = (u32)(a >> 32);
    u32 b0 = (u32)b, b1 = (u32)(b >> 32);
    u32 p00l = a0 * b0, p00h = TG_MULHI32(a0, b0);
    u32 p01l = a0 * b1, p01h = TG_MULHI32(a0, b1);
    u32 p10l = a1 * b0, p10h = TG_MULHI32(a1, b0);
    u32 p11l = a1 * b1, p11h = TG_MULHI32(a1, b1);
    // middle column: < 3 * 2^32, no overflow
    u64 mid = (u64)p00h + (u64)p01l + (u64)p10l;
    r.lo = ((u64)(u32)mid << 32) | (u64)p00l;
    // the product is < 2^128, so this cannot overflow
    r.hi = ((u64)p11h << 32) + (u64)p11l + (u64)p01h + (u64)p10h + (mid >> 32);
#else
    r.lo = a * b;
    r.hi = TG_MULHI(a, b);
#endif
    return r;
}

// x * c mod 2^64 for a compile-time constant c. With TG_WIDE_MUL_32 and
// |c| < 2^32 (all of Tip5's MDS constants), uses the 32-bit path; the
// branches fold away after inlining.
TG_FN u64 tg_mul_const(u64 x, u64 c) {
#if defined(TG_WIDE_MUL_32)
    if ((c >> 32) == 0) {
        return tg_mul_u64_u32(x, (u32)c);
    }
    u64 neg = U64C(0) - c;
    if ((neg >> 32) == 0) {
        return U64C(0) - tg_mul_u64_u32(x, (u32)neg);
    }
#endif
    return x * c;
}
