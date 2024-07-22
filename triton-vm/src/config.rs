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
    /// Can be accessed via [`Config::cache_lde_trace`].
    ///
    /// [lde]: crate::table::master_table::MasterTable::low_degree_extend_all_columns
    /// [proving]: crate::stark::Stark::prove
    pub cache_lde_trace_overwrite: Option<CacheDecision>,
}

impl Config {
    pub fn new() -> Self {
        let maybe_overwrite = std::env::var("TVM_LDE_TRACE").map(|s| s.to_ascii_lowercase());
        let cache_lde_trace_overwrite = match maybe_overwrite {
            Ok(t) if &t == "cache" => Some(CacheDecision::Cache),
            Ok(f) if &f == "no_cache" => Some(CacheDecision::NoCache),
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

/// Overwrite the automatic decision whether to cache the [low-degree extended trace][lde] when
/// [proving]. Takes precedence over the environment variable `TVM_LDE_TRACE`.
///
/// Caching the low-degree extended trace improves proving speed but requires more memory. It is
/// generally recommended to cache the trace. Triton VM will make an automatic decision based on
/// free memory. Use this function if you know your requirements better.
///
/// [lde]: crate::table::master_table::MasterTable::low_degree_extend_all_columns
/// [proving]: crate::stark::Stark::prove
pub fn overwrite_lde_trace_caching_to(decision: CacheDecision) {
    CONFIG.with_borrow_mut(|config| config.cache_lde_trace_overwrite = Some(decision));
}

/// Should the [low-degree extended trace][lde] be cached? `None` means the
/// decision is made automatically, based on free memory.
///
/// [lde]: crate::table::master_table::MasterTable::low_degree_extend_all_columns
pub(crate) fn cache_lde_trace() -> Option<CacheDecision> {
    CONFIG.with_borrow(|config| config.cache_lde_trace_overwrite)
}

#[cfg(test)]
mod tests {
    use crate::example_programs::FIBONACCI_SEQUENCE;
    use crate::prelude::*;
    use crate::shared_tests::prove_and_verify;
    use crate::shared_tests::ProgramAndInput;
    use crate::shared_tests::DEFAULT_LOG2_FRI_EXPANSION_FACTOR_FOR_TESTS;

    use super::*;

    #[test]
    fn triton_vm_can_generate_valid_proof_with_just_in_time_lde() {
        overwrite_lde_trace_caching_to(CacheDecision::NoCache);
        prove_and_verify_a_triton_vm_program();
    }

    #[test]
    fn triton_vm_can_generate_valid_proof_with_cached_lde_trace() {
        overwrite_lde_trace_caching_to(CacheDecision::Cache);
        prove_and_verify_a_triton_vm_program();
    }

    fn prove_and_verify_a_triton_vm_program() {
        let program_and_input = ProgramAndInput::new(FIBONACCI_SEQUENCE.clone())
            .with_input(PublicInput::from(bfe_array![100]));
        prove_and_verify(
            program_and_input,
            DEFAULT_LOG2_FRI_EXPANSION_FACTOR_FOR_TESTS,
        );
    }
}
