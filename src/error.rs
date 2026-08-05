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
    /// A variable name was referenced that the discipline never declared.
    ///
    /// Raised when looking a variable up in the registry, when resolving the two
    /// halves of a partials pair, and when a streamed chunk names a variable that
    /// is not part of the discipline. The payload is the offending name.
    ///
    /// Maps to `INVALID_ARGUMENT`.
    #[error("Variable '{0}' not found")]
    VariableNotFound(String),

    /// An array's shape does not match the shape that was declared for it.
    ///
    /// Raised when building an array from flat data whose element count does not
    /// match the requested shape, and when checking a set of arrays against the
    /// discipline's declared variable shapes.
    ///
    /// Maps to `INVALID_ARGUMENT`.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// The shape the variable was declared with.
        expected: Vec<usize>,
        /// The shape that was actually supplied.
        actual: Vec<usize>,
    },

    /// A variable carries a type that is not valid in the context it appears in.
    ///
    /// Raised when a wire message's `VariableType` discriminant is unrecognized,
    /// or when a valid type is used where it is not allowed (for example an
    /// output where an input is expected). The payload describes the type seen.
    ///
    /// Maps to `INVALID_ARGUMENT`.
    #[error("Invalid variable type: {0}")]
    InvalidVariableType(String),

    /// A discipline was used before it had been initialized.
    ///
    /// Maps to `INVALID_ARGUMENT`, since it is the caller that sequenced the
    /// calls in the wrong order.
    #[error("Discipline not initialized")]
    DisciplineNotInitialized,

    /// An operation that needs variable metadata ran before that metadata was
    /// fetched.
    ///
    /// Raised by the client when a compute call is made before `setup` /
    /// `get_variable_definitions`: array reconstruction needs declared shapes, so
    /// the call could not produce correct results.
    ///
    /// Maps to `INVALID_ARGUMENT`.
    #[error("Setup not called")]
    SetupNotCalled,

    /// A discipline option was requested that is not declared, or whose stored
    /// value has a different type than the one requested.
    ///
    /// Maps to `INVALID_ARGUMENT`.
    #[error("Option '{name}' not found or has invalid type")]
    InvalidOption {
        /// Name of the option that could not be read.
        name: String,
    },

    /// A streamed chunk addresses elements past the end of its target array.
    ///
    /// Raised while applying a chunk to an array: the chunk's inclusive end index
    /// is not a valid position in an array of the declared length.
    ///
    /// Maps to `INVALID_ARGUMENT`.
    #[error("Array index out of bounds: {index} >= {size}")]
    IndexOutOfBounds {
        /// The out-of-range element index that was addressed.
        index: usize,
        /// Number of elements in the target array.
        size: usize,
    },

    /// A gRPC call failed, or a [`tonic::Status`] was converted into a
    /// `PhiloteError`.
    ///
    /// [`to_status`](PhiloteError::to_status) preserves the wrapped status's own
    /// code and message rather than remapping it, so proxying a downstream
    /// Philote call does not misattribute the failure.
    #[error("gRPC communication error: {0}")]
    GrpcError(Box<tonic::Status>),

    /// A protocol buffer message could not be decoded.
    ///
    /// Maps to `INTERNAL`. Decode failures a peer causes should be reported as
    /// [`Validation`](PhiloteError::Validation) instead, so that a caller's
    /// malformed input is not reported as a server fault.
    #[error("Protocol buffer error: {0}")]
    ProtobufError(#[from] prost::DecodeError),

    /// An underlying I/O operation failed (socket setup, file access, and so on).
    ///
    /// Maps to `INTERNAL`.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// An array operation failed on the server side.
    ///
    /// This is the variant a discipline's own `compute` raises when it fails, so
    /// it maps to `INTERNAL` rather than `INVALID_ARGUMENT` — the caller is not
    /// at fault. Construct it with
    /// [`array_error`](PhiloteError::array_error).
    #[error("Array operation error: {0}")]
    ArrayError(String),

    /// A requested feature or RPC is not supported by this discipline.
    ///
    /// Maps to `UNIMPLEMENTED`. Construct it with
    /// [`not_implemented`](PhiloteError::not_implemented); the payload names the
    /// missing feature.
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// The server or client is misconfigured — for example a malformed endpoint
    /// passed to the client.
    ///
    /// Maps to `INTERNAL`. Construct it with
    /// [`config_error`](PhiloteError::config_error).
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// The operation was cancelled before it completed.
    ///
    /// Maps to `CANCELLED`.
    #[error("Operation cancelled")]
    Cancelled,

    /// An input failed validation.
    ///
    /// Maps to `INVALID_ARGUMENT` on the wire, matching Philote-Python's use of
    /// `PhiloteValidationError`.
    #[error("{context}: {message}")]
    Validation {
        /// The operation that rejected the input, used to prefix the message.
        context: String,
        /// What was wrong with the input.
        message: String,
    },
}

impl PhiloteError {
    /// Create a validation error tagged with the operation that rejected the input.
    pub fn validation<C: Into<String>, M: Into<String>>(context: C, message: M) -> Self {
        PhiloteError::Validation {
            context: context.into(),
            message: message.into(),
        }
    }

    /// Convert to a gRPC [`Status`](tonic::Status).
    ///
    /// Client-supplied errors (validation failures, unknown variables, bad types,
    /// shape mismatches, malformed chunks) map to `INVALID_ARGUMENT`; everything
    /// else is `INTERNAL`. Philote-Python draws the same distinction via
    /// `context.abort`.
    ///
    /// Note that [`ArrayError`](PhiloteError::ArrayError) stays `INTERNAL`: it is
    /// the variant a discipline's own `compute` raises when it fails. Decode
    /// failures that a peer causes use
    /// [`Validation`](PhiloteError::Validation) instead, so the two are not
    /// conflated.
    ///
    /// A [`GrpcError`](PhiloteError::GrpcError) keeps the code of the status it
    /// wraps, so a server that proxies a downstream Philote call does not flatten
    /// that peer's `INVALID_ARGUMENT` into an `INTERNAL`.
    pub fn to_status(&self) -> tonic::Status {
        match self {
            PhiloteError::Validation { .. }
            | PhiloteError::VariableNotFound(_)
            | PhiloteError::InvalidVariableType(_)
            | PhiloteError::InvalidOption { .. }
            | PhiloteError::ShapeMismatch { .. }
            | PhiloteError::IndexOutOfBounds { .. }
            | PhiloteError::SetupNotCalled
            | PhiloteError::DisciplineNotInitialized => {
                tonic::Status::invalid_argument(self.to_string())
            }
            PhiloteError::Cancelled => tonic::Status::cancelled(self.to_string()),
            PhiloteError::NotImplemented(_) => tonic::Status::unimplemented(self.to_string()),
            PhiloteError::GrpcError(status) => {
                tonic::Status::new(status.code(), status.message().to_string())
            }
            _ => tonic::Status::internal(self.to_string()),
        }
    }
}

impl From<PhiloteError> for tonic::Status {
    fn from(err: PhiloteError) -> Self {
        err.to_status()
    }
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

    use tonic::Code;

    #[test]
    fn client_caused_errors_map_to_invalid_argument() {
        let cases: Vec<PhiloteError> = vec![
            PhiloteError::validation("ctx", "msg"),
            PhiloteError::VariableNotFound("x".into()),
            PhiloteError::InvalidVariableType("bad".into()),
            PhiloteError::InvalidOption { name: "opt".into() },
            PhiloteError::ShapeMismatch {
                expected: vec![2],
                actual: vec![3],
            },
            PhiloteError::IndexOutOfBounds { index: 5, size: 3 },
            PhiloteError::SetupNotCalled,
            PhiloteError::DisciplineNotInitialized,
        ];
        for err in cases {
            assert_eq!(
                err.to_status().code(),
                Code::InvalidArgument,
                "{err} should be INVALID_ARGUMENT"
            );
        }
    }

    #[test]
    fn server_side_failures_map_to_internal() {
        // A discipline's own compute failure is not the caller's fault.
        assert_eq!(
            PhiloteError::array_error("compute blew up")
                .to_status()
                .code(),
            Code::Internal
        );
        assert_eq!(
            PhiloteError::config_error("bad config").to_status().code(),
            Code::Internal
        );
    }

    #[test]
    fn cancelled_and_not_implemented_keep_their_own_codes() {
        assert_eq!(PhiloteError::Cancelled.to_status().code(), Code::Cancelled);
        assert_eq!(
            PhiloteError::not_implemented("apply_linear")
                .to_status()
                .code(),
            Code::Unimplemented
        );
    }

    #[test]
    fn a_wrapped_status_keeps_its_original_code() {
        // Matters when a server proxies a downstream Philote call: flattening the
        // peer's INVALID_ARGUMENT into INTERNAL would blame the wrong side.
        let err = PhiloteError::from(tonic::Status::invalid_argument("upstream said no"));
        let status = err.to_status();
        assert_eq!(status.code(), Code::InvalidArgument);
        assert_eq!(status.message(), "upstream said no");
    }

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
