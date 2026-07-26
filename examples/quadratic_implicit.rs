//! Serves the implicit quadratic discipline over gRPC.
//!
//! Mirrors Philote-Python's `examples/quadratic_implicit.py`.
//!
//! ```text
//! cargo run --example quadratic_implicit
//! ```

use std::sync::Arc;

use philote_mdo::examples::QuadraticImplicit;
use philote_mdo::philote_info::{
    discipline_service_server::DisciplineServiceServer,
    implicit_service_server::ImplicitServiceServer,
};
use philote_mdo::server::ImplicitServer;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "127.0.0.1:50051".parse()?;
    let server = Arc::new(ImplicitServer::new(QuadraticImplicit::new()).with_verbose(true));

    println!("Quadratic implicit server listening on {addr}");

    Server::builder()
        .add_service(DisciplineServiceServer::from_arc(server.clone()))
        .add_service(ImplicitServiceServer::from_arc(server))
        .serve(addr)
        .await?;

    Ok(())
}
