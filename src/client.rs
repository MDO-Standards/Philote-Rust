//! Client implementations for connecting to Philote servers
//!
//! This module provides gRPC clients for communicating with Philote discipline servers.
//! Clients handle connection management, request serialization, and response deserialization.
//!
//! # Client Types
//!
//! - [`DisciplineClient`] - Base client for discipline metadata and info
//! - [`ExplicitClient`] - Client for explicit discipline computations
//! - [`ImplicitClient`] - Client for implicit discipline with residuals
//!
//! # Example
//!
//! ```rust,no_run
//! use philote_mdo::client::ExplicitClient;
//!
//! # async fn example() -> philote_mdo::Result<()> {
//! // Connect to a Philote server
//! let mut client = ExplicitClient::connect("http://localhost:50051").await?;
//!
//! // Get discipline information
//! let info = client.get_info().await?;
//! println!("Connected to {} v{}", info.name, info.version);
//!
//! // Run setup, then fetch metadata. Both are required before computing: array
//! // responses are decoded using the declared shapes.
//! client.setup().await?;
//! let variables = client.get_variable_definitions().await?;
//! client.get_partial_definitions().await?;
//! println!("Discipline has {} variables", variables.len());
//! # Ok(())
//! # }
//! ```

pub mod base;
pub mod explicit;
pub mod implicit;

pub use base::{variable_shape_meta, DisciplineClient};
pub use explicit::ExplicitClient;
pub use implicit::ImplicitClient;
