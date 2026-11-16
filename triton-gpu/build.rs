use std::env;
use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let common = root.join("kernels/common");
    for f in [
        "tg_prelude.h",
        "tg_arith.h",
        "tg_core.h",
        "tg_api.h",
        "tg_tip5_tables.h",
        "tg_tip5_mds.h",
    ] {
        println!("cargo:rerun-if-changed={}", common.join(f).display());
    }

    if env::var_os("CARGO_FEATURE_CPU").is_some() {
        let src = root.join("kernels/cpu/tg_cpu.cpp");
        println!("cargo:rerun-if-changed={}", src.display());
        // TRITON_GPU_WIDE_MUL_32=1 builds the CPU reference with the 32-bit-piece
        // multiplication that the Metal backend uses, to verify it here.
        println!("cargo:rerun-if-env-changed=TRITON_GPU_WIDE_MUL_32");
        let mut build = cc::Build::new();
        if env::var("TRITON_GPU_WIDE_MUL_32").is_ok_and(|v| v == "1") {
            build.define("TG_WIDE_MUL_32", "1");
        }
        build
            .cpp(true)
            .std("c++17")
            .opt_level(3)
            .flag_if_supported("-march=native")
            .file(src)
            .compile("triton_gpu_cpu");
    }

    // TRITON_GPU_SKIP_CUDA_BUILD=1 type-checks the Rust side without nvcc (for `cargo check`).
    println!("cargo:rerun-if-env-changed=TRITON_GPU_SKIP_CUDA_BUILD");
    if env::var_os("CARGO_FEATURE_CUDA").is_some()
        && env::var_os("TRITON_GPU_SKIP_CUDA_BUILD").is_none()
    {
        let src = root.join("kernels/cuda/triton_gpu.cu");
        println!("cargo:rerun-if-changed={}", src.display());
        println!("cargo:rerun-if-env-changed=TRITON_GPU_CUDA_ARCH");
        // Default: fat binary for Turing, Ampere, Ada, Hopper + PTX for newer GPUs.
        let archs = env::var("TRITON_GPU_CUDA_ARCH").unwrap_or_else(|_| "75,80,86,89,90".into());
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .cudart("static")
            .opt_level(3)
            .flag("-std=c++17")
            .flag("-lineinfo");
        build.flag("-Xptxas=-O3");
        let mut last = String::new();
        for a in archs.split(',').map(str::trim).filter(|a| !a.is_empty()) {
            build.flag(format!("-gencode=arch=compute_{a},code=sm_{a}"));
            last = a.to_string();
        }
        if !last.is_empty() {
            build.flag(format!("-gencode=arch=compute_{last},code=compute_{last}"));
        }
        build.file(src).compile("triton_gpu_cuda");
        if let Ok(cuda) = env::var("CUDA_PATH").or_else(|_| env::var("CUDA_HOME")) {
            println!("cargo:rustc-link-search=native={cuda}/lib64");
            println!("cargo:rustc-link-search=native={cuda}/lib");
        }
    }

    if env::var_os("CARGO_FEATURE_METAL").is_some() {
        println!(
            "cargo:rerun-if-changed={}",
            root.join("kernels/metal/triton_gpu.metal").display()
        );
    }
}
