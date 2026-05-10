//! Philote-Rust: A high-performance library for building MDO analysis servers
//!
//! This library provides a type-safe, async implementation for creating distributed
//! analysis services in Multidisciplinary Design Optimization (MDO) frameworks using
//! gRPC and Protocol Buffers.
//!
//! # Overview
//!
//! Philote enables computational disciplines to be exposed as remote services that can
//! be integrated into MDO frameworks. It supports both explicit and implicit analysis
//! types with automatic gradient computation capabilities.
//!
//! # Key Features
//!
//! - **Type-safe gRPC communication** using Protocol Buffers
//! - **Async/await support** with Tokio runtime for high-performance concurrent operations
//! - **Flexible discipline types** for different analysis patterns
//! - **Efficient data streaming** for large arrays
//! - **Comprehensive error handling** with domain-specific error types
//!
//! # Quick Start
//!
//! ## Creating a Discipline Server
//!
//! ```rust,no_run
//! use async_trait::async_trait;
//! use philote::{
//!     traits::{Discipline, ExplicitDiscipline},
//!     server::ExplicitServer,
//!     ArrayMap, Result,
//! };
//! use ndarray::ArrayD;
//! use std::collections::HashMap;
//!
//! struct MyDiscipline;
//!
//! #[async_trait]
//! impl ExplicitDiscipline for MyDiscipline {
//!     async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
//!         let mut outputs = HashMap::new();
//!         // Your computation logic here
//!         Ok(outputs)
//!     }
//! }
//!
//! impl Discipline for MyDiscipline {
//!     fn name(&self) -> &str { "MyDiscipline" }
//!     fn version(&self) -> &str { "1.0.0" }
//!     // Implement other required methods...
//! #   fn add_input(&mut self, _: &str, _: &[usize], _: &str) -> Result<()> { Ok(()) }
//! #   fn add_output(&mut self, _: &str, _: &[usize], _: &str) -> Result<()> { Ok(()) }
//! #   fn add_option(&mut self, _: &str, _: &str) -> Result<()> { Ok(()) }
//! #   fn set_options(&mut self, _: &HashMap<String, serde_json::Value>) -> Result<()> { Ok(()) }
//! #   fn setup(&mut self) -> Result<()> { Ok(()) }
//! #   fn declare_partials(&mut self, _: &str, _: &str) -> Result<()> { Ok(()) }
//! #   fn get_variable_definitions(&self) -> Result<Vec<philote::philote_info::VariableMetaData>> { Ok(vec![]) }
//! #   fn get_partials_definitions(&self) -> Result<Vec<(String, String)>> { Ok(vec![]) }
//! #   fn get_available_options(&self) -> Result<HashMap<String, String>> { Ok(HashMap::new()) }
//! }
//! ```
//!
//! ## Connecting with a Client
//!
//! ```rust,no_run
//! use philote::client::ExplicitClient;
//!
//! # async fn example() -> philote::Result<()> {
//! let mut client = ExplicitClient::connect("http://localhost:50051").await?;
//! let info = client.get_info().await?;
//! println!("Connected to: {} v{}", info.name, info.version);
//! # Ok(())
//! # }
//! ```
//!
//! # Module Organization
//!
//! - [`client`] - Client implementations for connecting to Philote servers
//! - [`server`] - Server implementations for hosting disciplines
//! - [`traits`] - Core trait definitions for disciplines
//! - [`types`] - Data structures and type conversions
//! - [`error`] - Error types and handling
//! - [`utils`] - Utility functions for array operations
//!
//! # Type Aliases
//!
//! - [`Result<T>`](Result) - Standard result type using [`PhiloteError`]
//! - [`ArrayMap`] - Map of variable names to N-dimensional arrays
//! - [`PartialMap`] - Map of (output, input) tuples to partial derivative arrays

use ndarray::ArrayD;
use std::collections::HashMap;

/// Protocol buffer definitions generated from the Philote MDO specification
pub mod philote_info {
    tonic::include_proto!("philote");
}

pub mod client;
pub mod error;
pub mod server;
pub mod traits;
pub mod types;
pub mod utils;

pub use error::PhiloteError;
pub use traits::{Discipline, ExplicitDiscipline, ImplicitDiscipline};
pub use types::{ArrayData, VariableData};

/// Standard result type for Philote operations
pub type Result<T> = std::result::Result<T, PhiloteError>;

/// Map of variable names to their N-dimensional array values
pub type ArrayMap = HashMap<String, ArrayD<f64>>;

/// Map of (output variable, input variable) pairs to their partial derivative arrays
pub type PartialMap = HashMap<(String, String), ArrayD<f64>>;

pub type DiscreteMap = HashMap<String, prost_types::Value>;
