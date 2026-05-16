use std::collections::HashMap;
use std::pin::Pin;
use tokio_stream::Stream;
use tonic::{Request, Response, Status, Streaming};

use crate::philote_info::{
    implicit_service_server::ImplicitService, variable_message::Payload, Array, DiscreteVariable,
    VariableMessage, VariableType,
};
use crate::server::base::DisciplineServer;
use crate::traits::ImplicitDiscipline;
use crate::types::{ArrayChunker, ArrayData};
use crate::utils::chunk_arrays_for_streaming;
use crate::{ArrayMap, DiscreteMap, PartialMap};

pub struct ImplicitServer<D: ImplicitDiscipline + 'static> {
    base: DisciplineServer<D>,
}

impl<D: ImplicitDiscipline + 'static> ImplicitServer<D> {
    pub fn new(discipline: D) -> Self {
        Self {
            base: DisciplineServer::new(discipline),
        }
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.base = self.base.with_verbose(verbose);
        self
    }

    async fn log_if_verbose(&self, message: &str) {
        if self.base.verbose() {
            tracing::info!("{}", message);
        }
    }

    async fn stream_outputs_as_variable_messages(
        &self,
        arrays: &ArrayMap,
        var_type: VariableType,
        discrete_outputs: &DiscreteMap,
    ) -> Pin<Box<dyn Stream<Item = std::result::Result<VariableMessage, Status>> + Send>> {
        let stream_options = self.base.stream_options().read().await;
        let chunk_size = stream_options.max_double_per_slice;
        drop(stream_options);

        let chunks = chunk_arrays_for_streaming(arrays, var_type, chunk_size);

        let mut messages: Vec<std::result::Result<VariableMessage, Status>> = chunks
            .into_iter()
            .map(|chunk| {
                Ok(VariableMessage {
                    payload: Some(Payload::Continuous(Array::from(chunk))),
                })
            })
            .collect();

        for (name, value) in discrete_outputs {
            messages.push(Ok(VariableMessage {
                payload: Some(Payload::Discrete(DiscreteVariable {
                    name: name.clone(),
                    r#type: VariableType::KDiscreteOutput.into(),
                    value: Some(value.clone()),
                })),
            }));
        }

        Box::pin(tokio_stream::iter(messages))
    }

    async fn stream_partials_as_variable_messages(
        &self,
        partials: &PartialMap,
    ) -> Pin<Box<dyn Stream<Item = std::result::Result<VariableMessage, Status>> + Send>> {
        let stream_options = self.base.stream_options().read().await;
        let chunk_size = stream_options.max_double_per_slice;
        drop(stream_options);

        let mut chunks = Vec::new();

        for ((func_name, var_name), array) in partials {
            let flat_data = crate::utils::create_flattened_view(array);
            let array_chunks = ArrayChunker::new(chunk_size).chunk_array(
                func_name,
                &flat_data,
                VariableType::KPartial,
            );

            let partial_chunks: Vec<ArrayData> = array_chunks
                .into_iter()
                .map(|mut chunk| {
                    chunk.subname = Some(var_name.clone());
                    chunk
                })
                .collect();

            chunks.extend(partial_chunks);
        }

        let messages: Vec<std::result::Result<VariableMessage, Status>> = chunks
            .into_iter()
            .map(|chunk| {
                Ok(VariableMessage {
                    payload: Some(Payload::Continuous(Array::from(chunk))),
                })
            })
            .collect();

        Box::pin(tokio_stream::iter(messages))
    }

    async fn process_input_and_output_streams(
        &self,
        request_stream: Streaming<VariableMessage>,
    ) -> std::result::Result<(ArrayMap, ArrayMap, DiscreteMap), Status> {
        let (mut inputs, mut flat_inputs) = self
            .base
            .preallocate_inputs()
            .await
            .map_err(|e| Status::internal(format!("Failed to preallocate inputs: {}", e)))?;

        let (mut outputs, mut flat_outputs) = self
            .base
            .preallocate_outputs()
            .await
            .map_err(|e| Status::internal(format!("Failed to preallocate outputs: {}", e)))?;

        let mut discrete_inputs: DiscreteMap = HashMap::new();

        self.base
            .process_variable_message_stream(
                request_stream,
                &mut flat_inputs,
                Some(&mut flat_outputs),
                &mut discrete_inputs,
            )
            .await
            .map_err(|e| Status::internal(format!("Failed to process input stream: {}", e)))?;

        for (name, flat_data) in flat_inputs {
            if let Some(array) = inputs.get_mut(&name) {
                for (i, &value) in flat_data.iter().enumerate() {
                    if let Some(elem) = array.get_mut(i) {
                        *elem = value;
                    }
                }
            }
        }

        for (name, flat_data) in flat_outputs {
            if let Some(array) = outputs.get_mut(&name) {
                for (i, &value) in flat_data.iter().enumerate() {
                    if let Some(elem) = array.get_mut(i) {
                        *elem = value;
                    }
                }
            }
        }

        Ok((inputs, outputs, discrete_inputs))
    }
}

#[tonic::async_trait]
impl<D: ImplicitDiscipline + 'static> ImplicitService for ImplicitServer<D> {
    type ComputeResidualsStream =
        Pin<Box<dyn Stream<Item = std::result::Result<VariableMessage, Status>> + Send>>;

    async fn compute_residuals(
        &self,
        request: Request<Streaming<VariableMessage>>,
    ) -> std::result::Result<Response<Self::ComputeResidualsStream>, Status> {
        self.log_if_verbose("ComputeResiduals called").await;

        let input_stream = request.into_inner();
        let (inputs, outputs, discrete_inputs) =
            self.process_input_and_output_streams(input_stream).await?;

        let discipline = self.base.discipline().read().await;
        let has_discrete = !discrete_inputs.is_empty()
            || !discipline
                .get_discrete_variable_definitions()
                .unwrap_or_default()
                .is_empty();

        let (residuals, discrete_outputs) = if has_discrete {
            discipline
                .compute_residuals_with_discrete(&inputs, &outputs, &discrete_inputs)
                .await
                .map_err(|e| Status::internal(format!("Compute residuals failed: {}", e)))?
        } else {
            let residuals = discipline
                .compute_residuals(&inputs, &outputs)
                .await
                .map_err(|e| Status::internal(format!("Compute residuals failed: {}", e)))?;
            (residuals, HashMap::new())
        };
        drop(discipline);

        let residual_stream = self
            .stream_outputs_as_variable_messages(
                &residuals,
                VariableType::KResidual,
                &discrete_outputs,
            )
            .await;

        Ok(Response::new(residual_stream))
    }

    type SolveResidualsStream =
        Pin<Box<dyn Stream<Item = std::result::Result<VariableMessage, Status>> + Send>>;

    async fn solve_residuals(
        &self,
        request: Request<Streaming<VariableMessage>>,
    ) -> std::result::Result<Response<Self::SolveResidualsStream>, Status> {
        self.log_if_verbose("SolveResiduals called").await;

        let (mut inputs, mut flat_inputs) = self
            .base
            .preallocate_inputs()
            .await
            .map_err(|e| Status::internal(format!("Failed to preallocate inputs: {}", e)))?;

        let mut discrete_inputs: DiscreteMap = HashMap::new();

        let input_stream = request.into_inner();
        self.base
            .process_variable_message_stream(
                input_stream,
                &mut flat_inputs,
                None,
                &mut discrete_inputs,
            )
            .await
            .map_err(|e| Status::internal(format!("Failed to process input stream: {}", e)))?;

        for (name, flat_data) in flat_inputs {
            if let Some(array) = inputs.get_mut(&name) {
                for (i, &value) in flat_data.iter().enumerate() {
                    if let Some(elem) = array.get_mut(i) {
                        *elem = value;
                    }
                }
            }
        }

        let discipline = self.base.discipline().read().await;
        let has_discrete = !discrete_inputs.is_empty()
            || !discipline
                .get_discrete_variable_definitions()
                .unwrap_or_default()
                .is_empty();

        let (outputs, discrete_outputs) = if has_discrete {
            discipline
                .solve_residuals_with_discrete(&inputs, &discrete_inputs)
                .await
                .map_err(|e| Status::internal(format!("Solve residuals failed: {}", e)))?
        } else {
            let outputs = discipline
                .solve_residuals(&inputs)
                .await
                .map_err(|e| Status::internal(format!("Solve residuals failed: {}", e)))?;
            (outputs, HashMap::new())
        };
        drop(discipline);

        let output_stream = self
            .stream_outputs_as_variable_messages(&outputs, VariableType::KOutput, &discrete_outputs)
            .await;

        Ok(Response::new(output_stream))
    }

    type ComputeResidualGradientsStream =
        Pin<Box<dyn Stream<Item = std::result::Result<VariableMessage, Status>> + Send>>;

    async fn compute_residual_gradients(
        &self,
        request: Request<Streaming<VariableMessage>>,
    ) -> std::result::Result<Response<Self::ComputeResidualGradientsStream>, Status> {
        self.log_if_verbose("ComputeResidualGradients called").await;

        let input_stream = request.into_inner();
        let (inputs, outputs, discrete_inputs) =
            self.process_input_and_output_streams(input_stream).await?;

        let discipline = self.base.discipline().read().await;
        let has_discrete = !discrete_inputs.is_empty()
            || !discipline
                .get_discrete_variable_definitions()
                .unwrap_or_default()
                .is_empty();

        let partials = if has_discrete {
            discipline
                .residual_partials_with_discrete(&inputs, &outputs, &discrete_inputs)
                .await
                .map_err(|e| {
                    Status::internal(format!("Compute residual gradients failed: {}", e))
                })?
        } else {
            discipline
                .residual_partials(&inputs, &outputs)
                .await
                .map_err(|e| {
                    Status::internal(format!("Compute residual gradients failed: {}", e))
                })?
        };
        drop(discipline);

        let partial_stream = self.stream_partials_as_variable_messages(&partials).await;

        Ok(Response::new(partial_stream))
    }
}

#[tonic::async_trait]
impl<D: ImplicitDiscipline + 'static>
    crate::philote_info::discipline_service_server::DisciplineService for ImplicitServer<D>
{
    async fn get_info(
        &self,
        request: Request<()>,
    ) -> std::result::Result<Response<crate::philote_info::DisciplineProperties>, Status> {
        self.base.get_info(request).await
    }

    async fn set_stream_options(
        &self,
        request: Request<crate::philote_info::StreamOptions>,
    ) -> std::result::Result<Response<()>, Status> {
        self.base.set_stream_options(request).await
    }

    async fn get_available_options(
        &self,
        request: Request<()>,
    ) -> std::result::Result<Response<crate::philote_info::OptionsList>, Status> {
        self.base.get_available_options(request).await
    }

    async fn set_options(
        &self,
        request: Request<crate::philote_info::DisciplineOptions>,
    ) -> std::result::Result<Response<()>, Status> {
        self.base.set_options(request).await
    }

    async fn setup(&self, request: Request<()>) -> std::result::Result<Response<()>, Status> {
        self.base.setup(request).await
    }

    type GetVariableDefinitionsStream = <DisciplineServer<D> as crate::philote_info::discipline_service_server::DisciplineService>::GetVariableDefinitionsStream;

    async fn get_variable_definitions(
        &self,
        request: Request<()>,
    ) -> std::result::Result<Response<Self::GetVariableDefinitionsStream>, Status> {
        self.base.get_variable_definitions(request).await
    }

    type GetPartialDefinitionsStream = <DisciplineServer<D> as crate::philote_info::discipline_service_server::DisciplineService>::GetPartialDefinitionsStream;

    async fn get_partial_definitions(
        &self,
        request: Request<()>,
    ) -> std::result::Result<Response<Self::GetPartialDefinitionsStream>, Status> {
        self.base.get_partial_definitions(request).await
    }

    async fn set_variable_shapes(
        &self,
        request: Request<Streaming<crate::philote_info::VariableMetaData>>,
    ) -> std::result::Result<Response<()>, Status> {
        self.base.set_variable_shapes(request).await
    }
}
