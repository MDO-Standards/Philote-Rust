//! Serves the paraboloid discipline over gRPC.
//!
//! Mirrors Philote-Python's `examples/parabaloid_explicit.py`, so a client from
//! either implementation can drive either server.
//!
//! ```text
//! cargo run --example paraboloid_server
//! ```

use std::sync::Arc;

use philote_mdo::examples::Paraboloid;
use philote_mdo::philote_info::{
    discipline_service_server::DisciplineServiceServer,
    explicit_service_server::ExplicitServiceServer,
};
use philote_mdo::server::ExplicitServer;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "127.0.0.1:50051".parse()?;
    let server = Arc::new(ExplicitServer::new(Paraboloid::new()).with_verbose(true));

    println!("Paraboloid server listening on {addr}");

    Server::builder()
        .add_service(DisciplineServiceServer::from_arc(server.clone()))
        .add_service(ExplicitServiceServer::from_arc(server))
        .serve(addr)
        .await?;

    Ok(())
}
