//! Constants and magic numbers for Triton VM
//!
//! This module centralizes all constants to improve code readability and maintainability.
//! Addresses Audit Suggestion E: Move constants to separate file/module.

/// FRI-related constants
pub mod fri {
    /// Default log2 of FRI expansion factor for security
    pub const DEFAULT_LOG2_FRI_EXPANSION_FACTOR: usize = 2;

    /// Low security expansion factor for testing
    pub const LOW_SECURITY_LOG2_FRI_EXPANSION_FACTOR: usize = 2;

    /// Default security level (bits)
    pub const DEFAULT_SECURITY_LEVEL: usize = 160;

    /// Low security level for testing
    pub const LOW_SECURITY_LEVEL: usize = 32;
}

/// VM execution constants
pub mod vm {
    /// Number of cosets used in trace domain
    pub const NUM_COSETS: usize = 8;

    /// Ratio between randomized trace length and working domain length
    pub const RANDOMIZED_TRACE_LEN_TO_WORKING_DOMAIN_LEN_RATIO: usize = 8;

    /// Number of quotient segments
    pub const NUM_QUOTIENT_SEGMENTS: usize = 4;

    /// Number of deep codeword components
    pub const NUM_DEEP_CODEWORD_COMPONENTS: usize = 3;
}

/// Table-related constants
pub mod table {
    /// Minimum required table length
    pub const MIN_TABLE_LENGTH: usize = 1;

    /// Default auxiliary column initial value
    pub const DEFAULT_AUX_COLUMN_INITIAL: i32 = 0;

    /// Maximum allowed lookup multiplicity to prevent attacks
    pub const MAX_LOOKUP_MULTIPLICITY: u64 = u32::MAX as u64;

    /// Maximum clock jump difference multiplicity
    pub const MAX_CLK_JUMP_DIFF_MULTIPLICITY: u64 = 1_000_000;
}

/// Cryptographic constants
pub mod crypto {
    /// Extension degree for XField elements
    pub const X_FIELD_EXTENSION_DEGREE: usize = 3;

    /// Number of out-of-domain rows for verification
    pub const NUM_OUT_OF_DOMAIN_ROWS: usize = 2;
}

/// Error handling constants
pub mod error {
    /// Maximum error message length
    pub const MAX_ERROR_MESSAGE_LEN: usize = 256;

    /// Default error context when none provided
    pub const DEFAULT_ERROR_CONTEXT: &str = "unknown operation";
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn constants_are_reasonable() {
        assert!(fri::DEFAULT_SECURITY_LEVEL > fri::LOW_SECURITY_LEVEL);
        assert!(fri::DEFAULT_LOG2_FRI_EXPANSION_FACTOR > 0);
        assert!(vm::NUM_COSETS.is_power_of_two());
        assert!(table::MIN_TABLE_LENGTH > 0);
    }
}
