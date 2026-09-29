//! This module contains various configuration options for Triton VM. In
//! general, the configuration options impact performance only. In particular,
//! any configuration provides the same completeness and soundness guarantees.
//!
//! The default configuration is sane and should provide the best performance
//! for most compilation targets. If you are generation Triton VM proofs on some
//! “unusual” system, you might want to try a few different options.
//!
//! # Time / Memory Trade-Offs
//!
//! Parts of the [proof generation](crate::stark::Stark::prove) process can
//! trade time for memory. This module provides ways to control these
//! trade-offs. Additionally, and with lower precedence, they can be controlled
//! via the following environment variables:
//!
//! - `TVM_LDE_TRACE`: Set to `cache` to cache the low-degree extended trace.
//!   Set to `no_cache` to not cache it. If unset (or set to anything else),
//!   Triton VM will make an automatic decision based on free memory.

use std::cell::RefCell;

use arbitrary::Arbitrary;

pub const ENV_VAR_LDE_CACHE: &str = "TVM_LDE_TRACE";
pub const ENV_VAR_LDE_CACHE_WITH_CACHE: &str = "cache";
pub const ENV_VAR_LDE_CACHE_NO_CACHE: &str = "no_cache";

thread_local! {
    pub(crate) static CONFIG: RefCell<Config> = RefCell::new(Config::default());
}

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub enum CacheDecision {
    #[default]
    Cache,
    NoCache,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
struct Config {
    /// Whether to cache the [low-degree extended trace][lde] when [proving].
    /// `None` means the decision is made automatically, based on free memory.
    /// Can be accessed via [`cache_lde_trace`].
    ///
    /// [lde]: crate::table::master_table::MasterTable::maybe_low_degree_extend_all_columns
    /// [proving]: crate::stark::Stark::prove
    pub cache_lde_trace_overwrite: Option<CacheDecision>,
}

impl Config {
    pub fn new() -> Self {
        let maybe_overwrite = std::env::var(ENV_VAR_LDE_CACHE).map(|s| s.to_ascii_lowercase());
        let cache_lde_trace_overwrite = match maybe_overwrite {
            Ok(t) if t == ENV_VAR_LDE_CACHE_WITH_CACHE => Some(CacheDecision::Cache),
            Ok(f) if f == ENV_VAR_LDE_CACHE_NO_CACHE => Some(CacheDecision::NoCache),
            _ => None,
        };

        Self {
            cache_lde_trace_overwrite,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

/// Overwrite the automatic decision whether to cache the [low-degree extended
/// trace][lde] when [proving]. Takes precedence over the environment variable
/// `TVM_LDE_TRACE`.
///
/// Caching the low-degree extended trace improves proving speed but requires
/// more memory. It is generally recommended to cache the trace. Triton VM will
/// make an automatic decision based on free memory. Use this function if you
/// know your requirements better.
///
/// [lde]: crate::arithmetic_domain::ArithmeticDomain::low_degree_extension
/// [proving]: crate::stark::Stark::prove
pub fn overwrite_lde_trace_caching_to(decision: CacheDecision) {
    CONFIG.with_borrow_mut(|config| config.cache_lde_trace_overwrite = Some(decision));
}

/// Should the [low-degree extended trace][lde] be cached? `None` means the
/// decision is made automatically, based on free memory.
///
/// [lde]: crate::table::master_table::MasterTable::maybe_low_degree_extend_all_columns
pub(crate) fn cache_lde_trace() -> Option<CacheDecision> {
    CONFIG.with_borrow(|config| config.cache_lde_trace_overwrite)
}

/// Decide whether to cache the [low-degree extended trace][lde], based on the
/// memory available to this process. Caching is decided for if at least
/// `required_bytes` are available. `None` if the available memory cannot be
/// determined.
///
/// [lde]: crate::table::master_table::MasterTable::maybe_low_degree_extend_all_columns
pub(crate) fn automatic_lde_trace_caching(required_bytes: u64) -> Option<CacheDecision> {
    let decision = if required_bytes <= available_memory()? {
        CacheDecision::Cache
    } else {
        CacheDecision::NoCache
    };

    Some(decision)
}

/// The number of bytes of memory available to this process: the system's
/// available memory, or less if the process's control group or any of its
/// ancestors limits the memory further. `None` if unknown, which is the case
/// on platforms other than Linux and Android.
///
/// Notably, a failing (or succeeding) allocation is not a good indicator: with
/// memory overcommitment, which is the default on Linux, allocating much more
/// than is available succeeds, and the process is killed later, when the
/// memory is actually used.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn available_memory() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let system_available = parse_mem_available(&meminfo)?;

    let control_group_available = std::fs::read_to_string("/proc/self/cgroup")
        .ok()
        .and_then(|cgroup| control_group_available_memory(&cgroup));

    Some(control_group_available.map_or(system_available, |c| c.min(system_available)))
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn available_memory() -> Option<u64> {
    None
}

/// The value of `MemAvailable` in the given contents of `/proc/meminfo`, in
/// bytes.
#[cfg(any(target_os = "linux", target_os = "android", test))]
fn parse_mem_available(meminfo: &str) -> Option<u64> {
    let kibibytes = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))?
        .trim()
        .strip_suffix("kB")?
        .trim()
        .parse::<u64>()
        .ok()?;

    kibibytes.checked_mul(1024)
}

/// The memory, in bytes, that the limits of the control group (v2) given by
/// the contents of `/proc/self/cgroup`, and of all its ancestors, leave
/// available. `None` if there is no limit, or if it cannot be determined.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn control_group_available_memory(cgroup: &str) -> Option<u64> {
    let path = cgroup.lines().find_map(|line| line.strip_prefix("0::"))?;
    let mut directory = std::path::Path::new("/sys/fs/cgroup").join(path.trim_start_matches('/'));

    let mut available = None;
    loop {
        let read = |file| std::fs::read_to_string(directory.join(file)).ok();
        let limit = read("memory.max").and_then(|max| max.trim().parse::<u64>().ok());
        let usage = read("memory.current").and_then(|current| current.trim().parse::<u64>().ok());
        if let (Some(limit), Some(usage)) = (limit, usage) {
            let headroom = limit.saturating_sub(usage);
            available = Some(available.map_or(headroom, |a: u64| a.min(headroom)));
        }
        if !directory.pop() || !directory.starts_with("/sys/fs/cgroup") {
            break;
        }
    }

    available
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::example_programs::FIBONACCI_SEQUENCE;
    use crate::prelude::*;
    use crate::shared_tests::TestableProgram;
    use crate::tests::test;

    #[macro_rules_attr::apply(test)]
    fn triton_vm_can_generate_valid_proof_with_just_in_time_lde() {
        overwrite_lde_trace_caching_to(CacheDecision::NoCache);
        prove_and_verify_a_triton_vm_program();
    }

    #[macro_rules_attr::apply(test)]
    fn triton_vm_can_generate_valid_proof_with_cached_lde_trace() {
        overwrite_lde_trace_caching_to(CacheDecision::Cache);
        prove_and_verify_a_triton_vm_program();
    }

    #[macro_rules_attr::apply(test)]
    fn mem_available_can_be_parsed() {
        let meminfo = "MemTotal:       65536000 kB\n\
                       MemFree:         1024000 kB\n\
                       MemAvailable:   32768000 kB\n\
                       Buffers:          512000 kB\n";
        assert_eq!(Some(32_768_000 * 1024), parse_mem_available(meminfo));
        assert_eq!(None, parse_mem_available("MemTotal: 65536000 kB\n"));
        assert_eq!(None, parse_mem_available("MemAvailable: lots\n"));
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[macro_rules_attr::apply(test)]
    fn automatic_lde_trace_caching_depends_on_required_memory() {
        assert_eq!(Some(CacheDecision::Cache), automatic_lde_trace_caching(0));
        assert_eq!(
            Some(CacheDecision::NoCache),
            automatic_lde_trace_caching(u64::MAX)
        );
    }

    fn prove_and_verify_a_triton_vm_program() {
        TestableProgram::new(FIBONACCI_SEQUENCE.clone())
            .with_input(PublicInput::from(bfe_array![100]))
            .prove_and_verify();
    }
}
