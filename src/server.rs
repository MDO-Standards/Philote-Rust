//! Server implementations for hosting Philote disciplines
//!
//! This module provides gRPC server implementations that expose Rust disciplines
//! as remote services. Servers handle incoming requests, deserialize inputs,
//! execute discipline computations, and serialize responses.
//!
//! # Server Types
//!
//! - [`ExplicitServer`] - Server for hosting explicit disciplines
//! - [`ImplicitServer`] - Server for hosting implicit disciplines
//!
//! # Example
//!
//! ```rust,no_run
//! use philote_mdo::server::ExplicitServer;
//! use philote_mdo::traits::{Discipline, ExplicitDiscipline};
//! use async_trait::async_trait;
//! use std::net::SocketAddr;
//! use tonic::transport::Server;
//! # use philote_mdo::{ArrayMap, Result};
//! # use std::collections::HashMap;
//! # use philote_mdo::philote_info::{VariableMetaData, explicit_service_server::ExplicitServiceServer};
//!
//! # struct MyDiscipline;
//! # #[async_trait]
//! # impl ExplicitDiscipline for MyDiscipline {
//! #     async fn compute(&self, _: &ArrayMap) -> Result<ArrayMap> { Ok(HashMap::new()) }
//! # }
//! # impl Discipline for MyDiscipline {
//! #     fn name(&self) -> &str { "Test" }
//! #     fn add_input(&mut self, _: &str, _: &[usize], _: &str) -> Result<()> { Ok(()) }
//! #     fn add_output(&mut self, _: &str, _: &[usize], _: &str) -> Result<()> { Ok(()) }
//! #     fn add_option(&mut self, _: &str, _: &str) -> Result<()> { Ok(()) }
//! #     fn set_options(&mut self, _: &HashMap<String, serde_json::Value>) -> Result<()> { Ok(()) }
//! #     fn setup(&mut self) -> Result<()> { Ok(()) }
//! #     fn declare_partials(&mut self, _: &str, _: &str) -> Result<()> { Ok(()) }
//! #     fn get_variable_definitions(&self) -> Result<Vec<VariableMetaData>> { Ok(vec![]) }
//! #     fn get_partials_definitions(&self) -> Result<Vec<(String, String)>> { Ok(vec![]) }
//! #     fn get_available_options(&self) -> Result<HashMap<String, String>> { Ok(HashMap::new()) }
//! # }
//!
//! # async fn example() -> Result<()> {
//! // Create a discipline
//! let discipline = MyDiscipline;
//!
//! // Wrap in server
//! let server = ExplicitServer::new(discipline).with_verbose(true);
//!
//! // Start gRPC server
//! let addr: SocketAddr = "127.0.0.1:50051".parse().unwrap();
//! Server::builder()
//!     .add_service(ExplicitServiceServer::new(server))
//!     .serve(addr)
//!     .await
//!     .unwrap();
//! # Ok(())
//! # }
//! ```

pub mod base;
pub mod explicit;
pub mod implicit;

pub use base::DisciplineServer;
pub use explicit::ExplicitServer;
pub use implicit::ImplicitServer;
