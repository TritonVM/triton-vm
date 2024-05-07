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
    use assert2::assert;
    use twenty_first::prelude::*;

    use crate::example_programs::FIBONACCI_SEQUENCE;
    use crate::prelude::*;
    use crate::profiler::TritonProfiler;
    use crate::shared_tests::prove_with_low_security_level;

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
        let stdin = PublicInput::from(bfe_array![100]);
        let secret_in = NonDeterminism::default();

        let mut profiler = Some(TritonProfiler::new("Prove Fib 100"));
        let (stark, claim, proof) =
            prove_with_low_security_level(&FIBONACCI_SEQUENCE, stdin, secret_in, &mut profiler);
        assert!(let Ok(()) = stark.verify(&claim, &proof, &mut None));

        let mut profiler = profiler.unwrap();
        let report = profiler.report();
        println!("{report}");
    }
}
