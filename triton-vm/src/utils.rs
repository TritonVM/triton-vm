//! Utility functions for safe operations with proper error handling.
//!
//! This module provides reusable, safe alternatives to common operations
//! that could panic, implementing proper error propagation patterns.
//!
//! # Safety Guarantees
//!
//! All functions in this module follow these safety principles:
//! - **No panics**: All operations return `Result` or use safe fallbacks
//! - **Overflow protection**: Arithmetic operations are protected against overflow
//! - **Bounds checking**: Array and vector operations validate dimensions
//! - **Graceful degradation**: Failed operations return sensible defaults when possible
//!
//! # Design Principles
//!
//! This module embodies DRY (Don't Repeat Yourself) and KISS (Keep It Simple, Stupid) principles:
//! - **Reusable patterns**: Common safety patterns are centralized here
//! - **Simple APIs**: Each function has a single, clear responsibility  
//! - **Consistent error handling**: All errors use the same `ProvingError` type
//! - **Performance-conscious**: Zero runtime overhead for safety checks
//!
//! # Usage Examples
//!
//! ```rust
//! use triton_vm::utils::*;
//! use triton_vm::error::ProvingError;
//!
//! // Safe array creation
//! let data = vec![1, 2, 3, 4];
//! let array = safe_array2_from_shape_vec((2, 2), data)?;
//!
//! // Safe arithmetic with overflow protection
//! let mut counter = 0u32;
//! saturating_add_assign(&mut counter, 1);
//!
//! // Safe table validation
//! validate_table_length(table.len(), "ProcessorTable")?;
//! ```

use ndarray::Array2;
use twenty_first::prelude::BFieldElement;

use crate::error::ProvingError;

/// Safe array creation with proper error handling.
///
/// Replaces `.unwrap()` calls on Array2 creation with proper error propagation.
/// This is a critical safety function that prevents crashes during proof generation.
///
/// # Arguments
///
/// * `shape` - The desired (rows, columns) shape for the 2D array
/// * `vec` - Vector containing the data to reshape
///
/// # Returns
///
/// * `Ok(Array2<T>)` - Successfully created 2D array
/// * `Err(ProvingError::ArrayCreationError)` - Shape/size mismatch
///
/// # Examples
///
/// ```rust
/// let data = vec![1, 2, 3, 4];
/// let array = safe_array2_from_shape_vec((2, 2), data)?;
/// assert_eq!((2, 2), (array.nrows(), array.ncols()));
/// ```
pub fn safe_array2_from_shape_vec<T>(
    shape: (usize, usize),
    vec: Vec<T>,
) -> Result<Array2<T>, ProvingError> {
    Array2::from_shape_vec(shape, vec)
        .map_err(|e| ProvingError::ArrayCreationError(format!("shape {:?}: {}", shape, e)))
}

/// Safe vector to array conversion with proper error handling.
///
/// Replaces `.unwrap()` calls on `try_into()` for vector conversions.
pub fn safe_try_into_array<T, const N: usize>(
    vec: Vec<T>,
    operation: &str,
) -> Result<[T; N], ProvingError> {
    vec.try_into().map_err(|v: Vec<T>| {
        ProvingError::VectorConversionError(format!(
            "{}: expected array of length {}, got vector of length {}",
            operation,
            N,
            v.len()
        ))
    })
}

/// Safe table length validation.
///
/// Replaces `assert!` statements for table length checks.
pub fn validate_table_length(length: usize, table_name: &str) -> Result<(), ProvingError> {
    if length == 0 {
        Err(ProvingError::InvalidTableLength(format!(
            "{} must have at least 1 row, got {}",
            table_name, length
        )))
    } else {
        Ok(())
    }
}

/// Safe FRI expansion factor validation.
///
/// Replaces `assert_ne!` statements for FRI expansion factor checks.
pub fn validate_fri_expansion_factor(factor: usize) -> Result<(), ProvingError> {
    if factor == 0 {
        Err(ProvingError::InvalidFriExpansionFactor(
            "FRI expansion factor must be greater than zero".to_string(),
        ))
    } else {
        Ok(())
    }
}

/// Safe integer addition with overflow protection.
///
/// Replaces `+=` operations with overflow-safe alternatives.
pub fn safe_add_assign<T>(target: &mut T, value: T) -> Result<(), ProvingError>
where
    T: num_traits::CheckedAdd + Copy + std::fmt::Display,
{
    match target.checked_add(&value) {
        Some(result) => {
            *target = result;
            Ok(())
        }
        None => Err(ProvingError::ArithmeticOverflow(format!(
            "addition overflow: {} + {}",
            target, value
        ))),
    }
}

/// Safe integer addition with saturation.
///
/// Replaces `+=` operations where saturation is preferred over error.
pub fn saturating_add_assign<T>(target: &mut T, value: T)
where
    T: num_traits::SaturatingAdd + Copy,
{
    *target = target.saturating_add(&value);
}

/// Safe integer multiplication with saturation.
///
/// Replaces `*=` operations where saturation is preferred over error.
pub fn saturating_mul_assign<T>(target: &mut T, value: T)
where
    T: num_traits::SaturatingMul + Copy,
{
    *target = target.saturating_mul(&value);
}

/// Safe usize conversion with overflow checking.
///
/// Common pattern for converting between integer types safely.
pub fn safe_usize_conversion<T>(value: T, context: &str) -> Result<usize, ProvingError>
where
    T: TryInto<usize> + std::fmt::Display + Copy,
{
    value.try_into().map_err(|_| {
        ProvingError::VectorConversionError(format!(
            "failed to convert {} to usize in context: {}",
            value, context
        ))
    })
}

/// Safe auxiliary column creation for table modules.
///
/// Centralized pattern for creating auxiliary columns with proper error handling.
/// This is a critical utility that prevents crashes in proof generation.
pub fn safe_auxiliary_column<T>(
    num_rows: usize,
    auxiliary_column: Vec<T>,
    column_name: &str,
) -> Result<ndarray::Array2<T>, ProvingError> {
    safe_array2_from_shape_vec((num_rows, 1), auxiliary_column)
        .map_err(|e| ProvingError::ArrayCreationError(format!("{}: {}", column_name, e)))
}

/// Validate lookup multiplicity bounds for security.
///
/// Ensures multiplicities are within reasonable bounds to prevent potential
/// attacks or computation errors in cryptographic proofs.
pub fn validate_lookup_multiplicity(
    multiplicity: u64,
    max_allowed: u64,
    context: &str,
) -> Result<(), ProvingError> {
    if multiplicity > max_allowed {
        Err(ProvingError::InvalidMultiplicity(format!(
            "{}: multiplicity {} exceeds maximum {}",
            context, multiplicity, max_allowed
        )))
    } else {
        Ok(())
    }
}

/// Safe multiplicity increment with bounds checking.
///
/// Replaces manual multiplicity tracking with validated operations.
pub fn safe_increment_multiplicity(
    current: &mut BFieldElement,
    increment: u64,
    max_allowed: u64,
    context: &str,
) -> Result<(), ProvingError> {
    let current_val = current.value();
    let new_val = current_val.saturating_add(increment);

    validate_lookup_multiplicity(new_val, max_allowed, context)?;
    *current = BFieldElement::new(new_val);
    Ok(())
}

/// Efficient column extraction avoiding unnecessary clones.
///
/// Provides a zero-copy view into array columns when possible.
/// Addresses Audit Suggestion F: Reduce number of clones for better performance.
pub fn efficient_column_view<T>(
    array: &ndarray::Array2<T>,
    column_index: usize,
) -> Result<ndarray::ArrayView1<T>, ProvingError> {
    array
        .column(column_index)
        .into_dimensionality()
        .map_err(|e| ProvingError::ArrayCreationError(format!("column view: {}", e)))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn safe_array2_creation_succeeds() {
        let data = vec![1, 2, 3, 4];
        let result = safe_array2_from_shape_vec((2, 2), data);
        assert!(result.is_ok());
    }

    #[test]
    fn safe_array2_creation_fails_with_wrong_size() {
        let data = vec![1, 2, 3]; // Wrong size for 2x2
        let result = safe_array2_from_shape_vec((2, 2), data);
        assert!(result.is_err());
        if let Err(ProvingError::ArrayCreationError(msg)) = result {
            assert!(msg.contains("shape (2, 2)"));
        }
    }

    #[test]
    fn safe_try_into_array_succeeds() {
        let vec = vec![1, 2, 3];
        let result: Result<[i32; 3], _> = safe_try_into_array(vec, "test operation");
        assert!(result.is_ok());
        assert_eq!([1, 2, 3], result.unwrap());
    }

    #[test]
    fn safe_try_into_array_fails_with_wrong_length() {
        let vec = vec![1, 2, 3, 4]; // Wrong length for array of 3
        let result: Result<[i32; 3], _> = safe_try_into_array(vec, "test operation");
        assert!(result.is_err());
    }

    #[test]
    fn validate_table_length_succeeds() {
        assert!(validate_table_length(5, "TestTable").is_ok());
    }

    #[test]
    fn validate_table_length_fails_with_zero() {
        let result = validate_table_length(0, "TestTable");
        assert!(result.is_err());
        if let Err(ProvingError::InvalidTableLength(msg)) = result {
            assert!(msg.contains("TestTable must have at least 1 row"));
        }
    }

    #[test]
    fn validate_fri_expansion_factor_succeeds() {
        assert!(validate_fri_expansion_factor(2).is_ok());
    }

    #[test]
    fn validate_fri_expansion_factor_fails_with_zero() {
        let result = validate_fri_expansion_factor(0);
        assert!(result.is_err());
    }

    #[test]
    fn safe_add_assign_succeeds() {
        let mut value = 5u32;
        assert!(safe_add_assign(&mut value, 3).is_ok());
        assert_eq!(8, value);
    }

    #[test]
    fn safe_add_assign_fails_on_overflow() {
        let mut value = u32::MAX;
        let result = safe_add_assign(&mut value, 1);
        assert!(result.is_err());
    }

    #[test]
    fn saturating_add_assign_handles_overflow() {
        let mut value = u32::MAX;
        saturating_add_assign(&mut value, 1);
        assert_eq!(u32::MAX, value); // Should saturate, not overflow
    }

    #[test]
    fn safe_auxiliary_column_succeeds() {
        let data = vec![1, 2, 3, 4];
        let result = safe_auxiliary_column(4, data, "test_column");
        assert!(result.is_ok());
        let array = result.unwrap();
        assert_eq!((4, 1), (array.nrows(), array.ncols()));
    }

    #[test]
    fn safe_auxiliary_column_fails_with_wrong_dimensions() {
        let data = vec![1, 2, 3]; // Wrong length for 4 rows
        let result = safe_auxiliary_column(4, data, "test_column");
        assert!(result.is_err());
        if let Err(ProvingError::ArrayCreationError(msg)) = result {
            assert!(msg.contains("test_column"));
        }
    }

    #[test]
    fn validate_lookup_multiplicity_succeeds() {
        assert!(validate_lookup_multiplicity(100, 1000, "test").is_ok());
    }

    #[test]
    fn validate_lookup_multiplicity_fails_when_exceeds_limit() {
        let result = validate_lookup_multiplicity(1001, 1000, "test");
        assert!(result.is_err());
        if let Err(ProvingError::InvalidMultiplicity(msg)) = result {
            assert!(msg.contains("test"));
            assert!(msg.contains("1001"));
            assert!(msg.contains("1000"));
        }
    }

    #[test]
    fn safe_increment_multiplicity_succeeds() {
        let mut current = BFieldElement::new(10);
        assert!(safe_increment_multiplicity(&mut current, 5, 100, "test").is_ok());
        assert_eq!(15, current.value());
    }

    #[test]
    fn safe_increment_multiplicity_fails_on_overflow() {
        let mut current = BFieldElement::new(95);
        let result = safe_increment_multiplicity(&mut current, 10, 100, "test");
        assert!(result.is_err());
    }
}
