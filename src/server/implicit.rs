use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_stream::Stream;
use tonic::{Request, Response, Status, Streaming};

use crate::philote_info::{
    implicit_service_server::ImplicitService, VariableMessage, VariableType,
};
use crate::server::base::DisciplineServer;
use crate::traits::ImplicitDiscipline;
use crate::{wire, ArrayMap, DiscreteMap};

/// Serves an [`ImplicitDiscipline`] over gRPC.
///
/// Implements both `ImplicitService` and `DisciplineService`, so a single instance
/// can back both service registrations.
pub struct ImplicitServer<D: ImplicitDiscipline + 'static> {
    base: DisciplineServer<D>,
}

impl<D: ImplicitDiscipline + 'static> ImplicitServer<D> {
    /// Wrap an implicit discipline for serving.
    ///
    /// Marks the registry implicit, so outputs get their residual twins even if
    /// the discipline built its registry with `VariableRegistry::default()`.
    pub fn new(mut discipline: D) -> Self {
        discipline.registry_mut().mark_implicit();
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

    /// Read a request stream carrying inputs only.
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

    /// Read a request stream carrying both inputs and current output values.
    async fn receive_inputs_and_outputs(
        &self,
        request: Request<Streaming<VariableMessage>>,
    ) -> std::result::Result<(ArrayMap, ArrayMap, DiscreteMap), Status> {
        let mut inputs = self.base.preallocate_inputs().await?;
        let mut outputs = self.base.preallocate_outputs().await?;
        let mut discrete_inputs = self.base.seeded_discrete_inputs().await;

        self.base
            .process_variable_message_stream(
                request.into_inner(),
                &mut inputs,
                Some(&mut outputs),
                &mut discrete_inputs,
            )
            .await?;

        Ok((inputs, outputs, discrete_inputs))
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
impl<D: ImplicitDiscipline + 'static> ImplicitService for ImplicitServer<D> {
    type ComputeResidualsStream = MessageStream;

    async fn compute_residuals(
        &self,
        request: Request<Streaming<VariableMessage>>,
    ) -> std::result::Result<Response<Self::ComputeResidualsStream>, Status> {
        self.base.log_if_verbose("ComputeResiduals called").await;

        let (inputs, outputs, discrete_inputs) = self.receive_inputs_and_outputs(request).await?;
        let has_discrete = self.base.has_discrete().await;

        let discipline = self.base.discipline().read().await;
        let (residuals, discrete_outputs) = if has_discrete {
            discipline
                .compute_residuals_with_discrete(&inputs, &outputs, &discrete_inputs)
                .await?
        } else {
            (
                discipline.compute_residuals(&inputs, &outputs).await?,
                HashMap::new(),
            )
        };
        drop(discipline);

        let messages = wire::assemble_output_messages(
            &residuals,
            VariableType::KResidual,
            &discrete_outputs,
            self.base.chunk_size().await,
        );

        Ok(Response::new(into_stream(messages)))
    }

    type SolveResidualsStream = MessageStream;

    async fn solve_residuals(
        &self,
        request: Request<Streaming<VariableMessage>>,
    ) -> std::result::Result<Response<Self::SolveResidualsStream>, Status> {
        self.base.log_if_verbose("SolveResiduals called").await;

        let (inputs, discrete_inputs) = self.receive_inputs(request).await?;
        let has_discrete = self.base.has_discrete().await;

        let discipline = self.base.discipline().read().await;
        let (outputs, discrete_outputs) = if has_discrete {
            discipline
                .solve_residuals_with_discrete(&inputs, &discrete_inputs)
                .await?
        } else {
            (discipline.solve_residuals(&inputs).await?, HashMap::new())
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

    type ComputeResidualGradientsStream = MessageStream;

    async fn compute_residual_gradients(
        &self,
        request: Request<Streaming<VariableMessage>>,
    ) -> std::result::Result<Response<Self::ComputeResidualGradientsStream>, Status> {
        self.base
            .log_if_verbose("ComputeResidualGradients called")
            .await;

        let (inputs, outputs, discrete_inputs) = self.receive_inputs_and_outputs(request).await?;
        let has_discrete = self.base.has_discrete().await;

        let discipline = self.base.discipline().read().await;
        let partials = if has_discrete {
            discipline
                .residual_partials_with_discrete(&inputs, &outputs, &discrete_inputs)
                .await?
        } else {
            discipline.residual_partials(&inputs, &outputs).await?
        };
        drop(discipline);

        let messages = wire::assemble_partial_messages(&partials, self.base.chunk_size().await);

        Ok(Response::new(into_stream(messages)))
    }
}

crate::impl_discipline_service_delegate!(ImplicitServer, ImplicitDiscipline);
