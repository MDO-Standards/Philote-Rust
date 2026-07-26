use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_stream::Stream;
use tonic::{Request, Response, Status, Streaming};

use crate::philote_info::{
    explicit_service_server::ExplicitService, VariableMessage, VariableType,
};
use crate::server::base::DisciplineServer;
use crate::traits::ExplicitDiscipline;
use crate::{wire, ArrayMap, DiscreteMap};

/// Serves an [`ExplicitDiscipline`] over gRPC.
///
/// Implements both `ExplicitService` and `DisciplineService`, so a single instance
/// can back both service registrations.
pub struct ExplicitServer<D: ExplicitDiscipline + 'static> {
    base: DisciplineServer<D>,
}

impl<D: ExplicitDiscipline + 'static> ExplicitServer<D> {
    pub fn new(discipline: D) -> Self {
        Self {
            base: DisciplineServer::new(discipline),
        }
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.base = self.base.with_verbose(verbose);
        self
    }

    pub fn discipline(&self) -> &Arc<RwLock<D>> {
        self.base.discipline()
    }

    /// Replace the served discipline.
    pub async fn set_discipline(&self, discipline: D) {
        self.base.set_discipline(discipline).await
    }

    /// Read the request stream into a fresh input map plus discrete inputs.
    async fn receive_inputs(
        &self,
        request: Request<Streaming<VariableMessage>>,
    ) -> std::result::Result<(ArrayMap, DiscreteMap), Status> {
        let mut inputs = self.base.preallocate_inputs().await?;
        let mut discrete_inputs = self.base.seeded_discrete_inputs().await;

        self.base
            .process_variable_message_stream(
                request.into_inner(),
                &mut inputs,
                None,
                &mut discrete_inputs,
            )
            .await?;

        Ok((inputs, discrete_inputs))
    }
}

type MessageStream =
    Pin<Box<dyn Stream<Item = std::result::Result<VariableMessage, Status>> + Send>>;

fn into_stream(messages: Vec<VariableMessage>) -> MessageStream {
    Box::pin(tokio_stream::iter(
        messages.into_iter().map(Ok).collect::<Vec<_>>(),
    ))
}

#[tonic::async_trait]
impl<D: ExplicitDiscipline + 'static> ExplicitService for ExplicitServer<D> {
    type ComputeFunctionStream = MessageStream;

    async fn compute_function(
        &self,
        request: Request<Streaming<VariableMessage>>,
    ) -> std::result::Result<Response<Self::ComputeFunctionStream>, Status> {
        self.base.log_if_verbose("ComputeFunction called").await;

        let (inputs, discrete_inputs) = self.receive_inputs(request).await?;
        let has_discrete = self.base.has_discrete().await;

        let discipline = self.base.discipline().read().await;
        let (outputs, discrete_outputs) = if has_discrete {
            discipline
                .compute_with_discrete(&inputs, &discrete_inputs)
                .await?
        } else {
            (discipline.compute(&inputs).await?, HashMap::new())
        };
        drop(discipline);

        let messages = wire::assemble_output_messages(
            &outputs,
            VariableType::KOutput,
            &discrete_outputs,
            self.base.chunk_size().await,
        );

        Ok(Response::new(into_stream(messages)))
    }

    type ComputeGradientStream = MessageStream;

    async fn compute_gradient(
        &self,
        request: Request<Streaming<VariableMessage>>,
    ) -> std::result::Result<Response<Self::ComputeGradientStream>, Status> {
        self.base.log_if_verbose("ComputeGradient called").await;

        let (inputs, discrete_inputs) = self.receive_inputs(request).await?;
        let has_discrete = self.base.has_discrete().await;

        let discipline = self.base.discipline().read().await;
        let partials = if has_discrete {
            discipline
                .compute_partials_with_discrete(&inputs, &discrete_inputs)
                .await?
        } else {
            discipline.compute_partials(&inputs).await?
        };
        drop(discipline);

        let messages = wire::assemble_partial_messages(&partials, self.base.chunk_size().await);

        Ok(Response::new(into_stream(messages)))
    }
}

crate::impl_discipline_service_delegate!(ExplicitServer, ExplicitDiscipline);
