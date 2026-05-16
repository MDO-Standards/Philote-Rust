//! Error types for Philote operations
//!
//! This module defines the error types used throughout the Philote library.
//! All errors implement the standard [`std::error::Error`] trait using the
//! [`thiserror`] crate for ergonomic error handling.
//!
//! # Error Categories
//!
//! - **Variable Errors**: [`VariableNotFound`], [`InvalidVariableType`], [`ShapeMismatch`]
//! - **Lifecycle Errors**: [`DisciplineNotInitialized`], [`SetupNotCalled`]
//! - **Communication Errors**: [`GrpcError`], [`ProtobufError`]
//! - **Data Errors**: [`ArrayError`], [`IndexOutOfBounds`]
//! - **Configuration Errors**: [`InvalidOption`], [`ConfigurationError`]
//!
//! [`VariableNotFound`]: PhiloteError::VariableNotFound
//! [`InvalidVariableType`]: PhiloteError::InvalidVariableType
//! [`ShapeMismatch`]: PhiloteError::ShapeMismatch
//! [`DisciplineNotInitialized`]: PhiloteError::DisciplineNotInitialized
//! [`SetupNotCalled`]: PhiloteError::SetupNotCalled
//! [`GrpcError`]: PhiloteError::GrpcError
//! [`ProtobufError`]: PhiloteError::ProtobufError
//! [`ArrayError`]: PhiloteError::ArrayError
//! [`IndexOutOfBounds`]: PhiloteError::IndexOutOfBounds
//! [`InvalidOption`]: PhiloteError::InvalidOption
//! [`ConfigurationError`]: PhiloteError::ConfigurationError
//!
//! # Example
//!
//! ```rust
//! use philote_mdo::PhiloteError;
//!
//! fn find_variable(name: &str) -> Result<f64, PhiloteError> {
//!     if name == "x" {
//!         Ok(42.0)
//!     } else {
//!         Err(PhiloteError::VariableNotFound(name.to_string()))
//!     }
//! }
//! ```

use thiserror::Error;

/// Error types for Philote operations
#[derive(Error, Debug)]
pub enum PhiloteError {
    #[error("Variable '{0}' not found")]
    VariableNotFound(String),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Invalid variable type: {0}")]
    InvalidVariableType(String),

    #[error("Discipline not initialized")]
    DisciplineNotInitialized,

    #[error("Setup not called")]
    SetupNotCalled,

    #[error("Option '{name}' not found or has invalid type")]
    InvalidOption { name: String },

    #[error("Array index out of bounds: {index} >= {size}")]
    IndexOutOfBounds { index: usize, size: usize },

    #[error("gRPC communication error: {0}")]
    GrpcError(Box<tonic::Status>),

    #[error("Protocol buffer error: {0}")]
    ProtobufError(#[from] prost::DecodeError),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Array operation error: {0}")]
    ArrayError(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Operation cancelled")]
    Cancelled,
}

impl PhiloteError {
    /// Create an array operation error with a custom message
    pub fn array_error<S: Into<String>>(msg: S) -> Self {
        PhiloteError::ArrayError(msg.into())
    }

    /// Create a "not implemented" error for features not yet supported
    pub fn not_implemented<S: Into<String>>(feature: S) -> Self {
        PhiloteError::NotImplemented(feature.into())
    }

    /// Create a configuration error with a custom message
    pub fn config_error<S: Into<String>>(msg: S) -> Self {
        PhiloteError::ConfigurationError(msg.into())
    }
}

impl From<tonic::Status> for PhiloteError {
    fn from(status: tonic::Status) -> Self {
        PhiloteError::GrpcError(Box::new(status))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_error_helper() {
        let err = PhiloteError::array_error("test message");
        assert!(matches!(err, PhiloteError::ArrayError(_)));
        assert_eq!(err.to_string(), "Array operation error: test message");
    }

    #[test]
    fn test_not_implemented_helper() {
        let err = PhiloteError::not_implemented("test feature");
        assert!(matches!(err, PhiloteError::NotImplemented(_)));
        assert_eq!(err.to_string(), "Not implemented: test feature");
    }

    #[test]
    fn test_config_error_helper() {
        let err = PhiloteError::config_error("test config");
        assert!(matches!(err, PhiloteError::ConfigurationError(_)));
        assert_eq!(err.to_string(), "Configuration error: test config");
    }

    #[test]
    fn test_variable_not_found_display() {
        let err = PhiloteError::VariableNotFound("x".to_string());
        assert_eq!(err.to_string(), "Variable 'x' not found");
    }

    #[test]
    fn test_shape_mismatch_display() {
        let err = PhiloteError::ShapeMismatch {
            expected: vec![2, 3],
            actual: vec![3, 2],
        };
        assert_eq!(
            err.to_string(),
            "Shape mismatch: expected [2, 3], got [3, 2]"
        );
    }

    #[test]
    fn test_invalid_variable_type_display() {
        let err = PhiloteError::InvalidVariableType("unknown".to_string());
        assert_eq!(err.to_string(), "Invalid variable type: unknown");
    }

    #[test]
    fn test_discipline_not_initialized() {
        let err = PhiloteError::DisciplineNotInitialized;
        assert_eq!(err.to_string(), "Discipline not initialized");
    }

    #[test]
    fn test_setup_not_called() {
        let err = PhiloteError::SetupNotCalled;
        assert_eq!(err.to_string(), "Setup not called");
    }

    #[test]
    fn test_invalid_option_display() {
        let err = PhiloteError::InvalidOption {
            name: "test_opt".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Option 'test_opt' not found or has invalid type"
        );
    }

    #[test]
    fn test_index_out_of_bounds_display() {
        let err = PhiloteError::IndexOutOfBounds { index: 5, size: 3 };
        assert_eq!(err.to_string(), "Array index out of bounds: 5 >= 3");
    }

    #[test]
    fn test_from_tonic_status() {
        let status = tonic::Status::internal("test error");
        let err: PhiloteError = status.into();
        assert!(matches!(err, PhiloteError::GrpcError(_)));
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: PhiloteError = io_err.into();
        assert!(matches!(err, PhiloteError::IoError(_)));
        assert!(err.to_string().contains("file not found"));
    }
}
