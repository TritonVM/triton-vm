// Backend prelude: maps the shared kernel code onto Metal Shading Language,
// CUDA C++, or plain C++ (the CPU reference used to verify the kernel logic).
//
// Exactly one of TG_METAL, TG_CUDA, TG_CPU must be defined before inclusion.
#pragma once

#if defined(TG_METAL)
  #include <metal_stdlib>
  using namespace metal;
  typedef ulong u64;
  typedef uint u32;
  typedef uchar u8;
  #define TG_FN inline
  #define TG_MFN inline
  #define TG_UNROLL _Pragma("unroll")
  #define TG_THREAD thread
  #define TG_SHARED threadgroup
  #define TG_GLOBAL device
  #define U64C(x) x##ul
  #define TG_MULHI(a, b) mulhi((a), (b))
  #define TG_CONST_ARRAY(type, name, n) constant type name[n]
  #define TG_LUT_ARRAY(name, n) constant u8 name[n]
  #define TG_LUT_SPACE threadgroup

#elif defined(TG_CUDA)
  typedef unsigned long long u64;
  typedef unsigned int u32;
  typedef unsigned char u8;
  #define TG_FN __device__ __forceinline__
  #define TG_MFN __device__ __forceinline__
  #define TG_UNROLL _Pragma("unroll")
  #define TG_THREAD
  #define TG_SHARED
  #define TG_GLOBAL
  #define U64C(x) x##ull
  #define TG_MULHI(a, b) __umul64hi((a), (b))
  #define TG_CONST_ARRAY(type, name, n) __constant__ type name[n]
  #define TG_LUT_ARRAY(name, n) __constant__ u8 name[n]
  #define TG_LUT_SPACE

#elif defined(TG_CPU)
  #include <cstdint>
  typedef unsigned long long u64;
  typedef unsigned int u32;
  typedef unsigned char u8;
  #define TG_FN static inline
  #define TG_MFN inline
  #define TG_UNROLL _Pragma("GCC unroll 16")
  #define TG_THREAD
  #define TG_SHARED
  #define TG_GLOBAL
  #define U64C(x) x##ull
  #if defined(_MSC_VER)
    // MSVC has no 128-bit integers: use the 32-bit-piece multiplication.
    #define TG_WIDE_MUL_32 1
    #define TG_MULHI(a, b) 0
  #else
    #define TG_MULHI(a, b) ((u64)(((unsigned __int128)(a) * (unsigned __int128)(b)) >> 64))
  #endif
  #define TG_CONST_ARRAY(type, name, n) static const type name[n]
  #define TG_LUT_ARRAY(name, n) static const u8 name[n]
  #define TG_LUT_SPACE

#else
  #error "define one of TG_METAL, TG_CUDA, TG_CPU"
#endif
