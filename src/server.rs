//! Server implementations for hosting Philote disciplines
//!
//! This module provides gRPC server implementations that expose Rust disciplines
//! as remote services. Servers handle incoming requests, decode inputs, execute
//! discipline computations, and encode responses.
//!
//! # Server types
//!
//! - [`ExplicitServer`] — hosts an [`ExplicitDiscipline`](crate::traits::ExplicitDiscipline)
//! - [`ImplicitServer`] — hosts an [`ImplicitDiscipline`](crate::traits::ImplicitDiscipline)
//!
//! Both also implement `DisciplineService`, so one instance registers as two
//! services (see the example below).
//!
//! # Example
//!
//! ```rust,no_run
//! use philote_mdo::examples::Paraboloid;
//! use philote_mdo::philote_info::{
//!     discipline_service_server::DisciplineServiceServer,
//!     explicit_service_server::ExplicitServiceServer,
//! };
//! use philote_mdo::server::ExplicitServer;
//! use std::sync::Arc;
//! use tonic::transport::Server;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let server = Arc::new(ExplicitServer::new(Paraboloid::default()));
//!
//! Server::builder()
//!     .add_service(DisciplineServiceServer::from_arc(server.clone()))
//!     .add_service(ExplicitServiceServer::from_arc(server))
//!     .serve("127.0.0.1:50051".parse()?)
//!     .await?;
//! # Ok(())
//! # }
//! ```

pub mod base;
pub mod explicit;
pub mod implicit;

pub use base::DisciplineServer;
pub use explicit::ExplicitServer;
pub use implicit::ImplicitServer;

/// Forward every `DisciplineService` RPC to the wrapped [`DisciplineServer`].
///
/// Both `ExplicitServer` and `ImplicitServer` expose the shared discipline RPCs in
/// addition to their own service; this generates that pass-through so the two
/// server types do not each carry a hand-written copy.
#[macro_export]
#[doc(hidden)]
macro_rules! impl_discipline_service_delegate {
    ($server:ident, $bound:ident) => {
        #[tonic::async_trait]
        impl<D: $crate::traits::$bound + 'static>
            $crate::philote_info::discipline_service_server::DisciplineService for $server<D>
        {
            // Every method below dispatches through UFCS so the generated impl does
            // not depend on `DisciplineService` being imported at the call site.
            async fn get_info(
                &self,
                request: tonic::Request<()>,
            ) -> std::result::Result<
                tonic::Response<$crate::philote_info::DisciplineProperties>,
                tonic::Status,
            > {
                <$crate::server::DisciplineServer<D> as $crate::philote_info::discipline_service_server::DisciplineService>::get_info(&self.base, request).await
            }

            async fn set_stream_options(
                &self,
                request: tonic::Request<$crate::philote_info::StreamOptions>,
            ) -> std::result::Result<tonic::Response<()>, tonic::Status> {
                <$crate::server::DisciplineServer<D> as $crate::philote_info::discipline_service_server::DisciplineService>::set_stream_options(&self.base, request).await
            }

            async fn get_available_options(
                &self,
                request: tonic::Request<()>,
            ) -> std::result::Result<
                tonic::Response<$crate::philote_info::OptionsList>,
                tonic::Status,
            > {
                <$crate::server::DisciplineServer<D> as $crate::philote_info::discipline_service_server::DisciplineService>::get_available_options(&self.base, request).await
            }

            async fn set_options(
                &self,
                request: tonic::Request<$crate::philote_info::DisciplineOptions>,
            ) -> std::result::Result<tonic::Response<()>, tonic::Status> {
                <$crate::server::DisciplineServer<D> as $crate::philote_info::discipline_service_server::DisciplineService>::set_options(&self.base, request).await
            }

            async fn setup(
                &self,
                request: tonic::Request<()>,
            ) -> std::result::Result<tonic::Response<()>, tonic::Status> {
                <$crate::server::DisciplineServer<D> as $crate::philote_info::discipline_service_server::DisciplineService>::setup(&self.base, request).await
            }

            type GetVariableDefinitionsStream = <$crate::server::DisciplineServer<D> as $crate::philote_info::discipline_service_server::DisciplineService>::GetVariableDefinitionsStream;

            async fn get_variable_definitions(
                &self,
                request: tonic::Request<()>,
            ) -> std::result::Result<
                tonic::Response<Self::GetVariableDefinitionsStream>,
                tonic::Status,
            > {
                <$crate::server::DisciplineServer<D> as $crate::philote_info::discipline_service_server::DisciplineService>::get_variable_definitions(&self.base, request).await
            }

            type GetPartialDefinitionsStream = <$crate::server::DisciplineServer<D> as $crate::philote_info::discipline_service_server::DisciplineService>::GetPartialDefinitionsStream;

            async fn get_partial_definitions(
                &self,
                request: tonic::Request<()>,
            ) -> std::result::Result<
                tonic::Response<Self::GetPartialDefinitionsStream>,
                tonic::Status,
            > {
                <$crate::server::DisciplineServer<D> as $crate::philote_info::discipline_service_server::DisciplineService>::get_partial_definitions(&self.base, request).await
            }

            async fn set_variable_shapes(
                &self,
                request: tonic::Request<
                    tonic::Streaming<$crate::philote_info::VariableMetaData>,
                >,
            ) -> std::result::Result<tonic::Response<()>, tonic::Status> {
                <$crate::server::DisciplineServer<D> as $crate::philote_info::discipline_service_server::DisciplineService>::set_variable_shapes(&self.base, request).await
            }
        }
    };
}
