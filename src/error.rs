use thiserror::Error;

#[derive(Error, Debug)]
pub enum PhiloteError {
    #[error("Variable '{0}' not found")]
    VariableNotFound(String),
    
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { 
        expected: Vec<usize>, 
        actual: Vec<usize> 
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
    GrpcError(#[from] tonic::Status),
    
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
}

impl PhiloteError {
    pub fn array_error<S: Into<String>>(msg: S) -> Self {
        PhiloteError::ArrayError(msg.into())
    }
    
    pub fn not_implemented<S: Into<String>>(feature: S) -> Self {
        PhiloteError::NotImplemented(feature.into())
    }
    
    pub fn config_error<S: Into<String>>(msg: S) -> Self {
        PhiloteError::ConfigurationError(msg.into())
    }
}