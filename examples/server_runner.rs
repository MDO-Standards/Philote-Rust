//! Serves the Rosenbrock discipline, showing option handling over gRPC.
//!
//! The `dimension` option is set by the client before `Setup`, which decides the
//! shape of `x`. Pair this with `cargo run --example client_example`.
//!
//! ```text
//! cargo run --example server_runner
//! ```

use std::sync::Arc;

use philote_mdo::examples::Rosenbrock;
use philote_mdo::philote_info::{
    discipline_service_server::DisciplineServiceServer,
    explicit_service_server::ExplicitServiceServer,
};
use philote_mdo::server::ExplicitServer;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `with_verbose(true)` emits `tracing` events; install a subscriber such as
    // `tracing-subscriber` in your own binary to see them.
    let addr = "127.0.0.1:50051".parse()?;
    let server = Arc::new(ExplicitServer::new(Rosenbrock::new()).with_verbose(true));

    println!("Rosenbrock server listening on {addr}");
    println!("Run `cargo run --example client_example` in another terminal.");

    Server::builder()
        .add_service(DisciplineServiceServer::from_arc(server.clone()))
        .add_service(ExplicitServiceServer::from_arc(server))
        .serve(addr)
        .await?;

    Ok(())
}
