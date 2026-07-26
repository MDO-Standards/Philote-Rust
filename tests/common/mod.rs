//! Shared harness for tests that need a real gRPC server.
//!
//! Each [`TestServer`] binds an ephemeral loopback port, so tests run concurrently
//! without colliding on a fixed port.

#![allow(dead_code)]

use std::net::SocketAddr;
use std::sync::Arc;

use philote_mdo::philote_info::{
    discipline_service_server::DisciplineServiceServer,
    explicit_service_server::ExplicitServiceServer, implicit_service_server::ImplicitServiceServer,
};
use philote_mdo::server::{ExplicitServer, ImplicitServer};
use philote_mdo::traits::{ExplicitDiscipline, ImplicitDiscipline};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;

/// A running server on an ephemeral port.
pub struct TestServer {
    addr: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    handle: Option<JoinHandle<()>>,
}

impl TestServer {
    /// The `http://` endpoint clients should connect to.
    pub fn endpoint(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Stop the server and wait for it to wind down.
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

impl Drop for TestServer {
    /// Signal shutdown for tests that return early (including on panic) without
    /// calling [`TestServer::shutdown`].
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
    }
}

async fn bind() -> (TcpListener, SocketAddr) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind an ephemeral loopback port");
    let addr = listener.local_addr().expect("resolve the bound address");
    (listener, addr)
}

/// Serve an explicit discipline.
///
/// Registers both `DisciplineService` and `ExplicitService` from one instance.
pub async fn spawn_explicit<D: ExplicitDiscipline + 'static>(discipline: D) -> TestServer {
    let (listener, addr) = bind().await;
    let (tx, rx) = oneshot::channel();
    let server = Arc::new(ExplicitServer::new(discipline));

    let handle = tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(DisciplineServiceServer::from_arc(server.clone()))
            .add_service(ExplicitServiceServer::from_arc(server))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                let _ = rx.await;
            })
            .await;
    });

    TestServer {
        addr,
        shutdown: Some(tx),
        handle: Some(handle),
    }
}

/// Serve an implicit discipline.
///
/// Registers both `DisciplineService` and `ImplicitService` from one instance.
pub async fn spawn_implicit<D: ImplicitDiscipline + 'static>(discipline: D) -> TestServer {
    let (listener, addr) = bind().await;
    let (tx, rx) = oneshot::channel();
    let server = Arc::new(ImplicitServer::new(discipline));

    let handle = tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(DisciplineServiceServer::from_arc(server.clone()))
            .add_service(ImplicitServiceServer::from_arc(server))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                let _ = rx.await;
            })
            .await;
    });

    TestServer {
        addr,
        shutdown: Some(tx),
        handle: Some(handle),
    }
}

/// Connect an explicit client and run the standard handshake.
///
/// `setup` plus both metadata fetches are required before computing, since array
/// responses are decoded using the declared shapes.
pub async fn connected_explicit_client(server: &TestServer) -> philote_mdo::client::ExplicitClient {
    let mut client = philote_mdo::client::ExplicitClient::connect(server.endpoint())
        .await
        .expect("connect to the test server");
    client.setup().await.expect("run setup");
    client
        .get_variable_definitions()
        .await
        .expect("fetch variable definitions");
    client
        .get_partial_definitions()
        .await
        .expect("fetch partials definitions");
    client
}

/// Connect an implicit client and run the standard handshake.
pub async fn connected_implicit_client(server: &TestServer) -> philote_mdo::client::ImplicitClient {
    let mut client = philote_mdo::client::ImplicitClient::connect(server.endpoint())
        .await
        .expect("connect to the test server");
    client.setup().await.expect("run setup");
    client
        .get_variable_definitions()
        .await
        .expect("fetch variable definitions");
    client
        .get_partial_definitions()
        .await
        .expect("fetch partials definitions");
    client
}
