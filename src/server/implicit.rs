use std::pin::Pin;
use tokio_stream::Stream;
use tonic::{Request, Response, Status, Streaming};

use crate::philote_info::{implicit_service_server::ImplicitService, Array, VariableType};
use crate::server::base::DisciplineServer;
use crate::traits::ImplicitDiscipline;
use crate::types::ArrayData;
use crate::utils::chunk_arrays_for_streaming;
use crate::{ArrayMap, PartialMap};

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

    async fn stream_arrays_as_chunks(
        &self,
        arrays: &ArrayMap,
        var_type: VariableType,
    ) -> Pin<Box<dyn Stream<Item = std::result::Result<Array, Status>> + Send>> {
        let stream_options = self.base.stream_options().read().await;
        let chunk_size = stream_options.max_double_per_slice;
        drop(stream_options);

        let chunks = chunk_arrays_for_streaming(arrays, var_type, chunk_size);
        let array_stream = chunks.into_iter().map(|chunk| Ok(Array::from(chunk)));

        Box::pin(tokio_stream::iter(array_stream))
    }

    async fn stream_partials_as_chunks(
        &self,
        partials: &PartialMap,
    ) -> Pin<Box<dyn Stream<Item = std::result::Result<Array, Status>> + Send>> {
        let stream_options = self.base.stream_options().read().await;
        let chunk_size = stream_options.max_double_per_slice;
        drop(stream_options);

        let mut chunks = Vec::new();

        for ((func_name, var_name), array) in partials {
            let flat_data = crate::utils::create_flattened_view(array);
            let array_chunks = crate::types::ArrayChunker::new(chunk_size).chunk_array(
                func_name,
                &flat_data,
                VariableType::KPartial,
            );

            // Set the subname for partials
            let partial_chunks: Vec<ArrayData> = array_chunks
                .into_iter()
                .map(|mut chunk| {
                    chunk.subname = Some(var_name.clone());
                    chunk
                })
                .collect();

            chunks.extend(partial_chunks);
        }

        let array_stream = chunks.into_iter().map(|chunk| Ok(Array::from(chunk)));

        Box::pin(tokio_stream::iter(array_stream))
    }

    async fn process_input_and_output_streams(
        &self,
        request_stream: Streaming<Array>,
    ) -> std::result::Result<(ArrayMap, ArrayMap), Status> {
        // Preallocate arrays
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

        // Process input stream (which may contain both inputs and outputs for implicit disciplines)
        self.base
            .process_input_stream(request_stream, &mut flat_inputs, Some(&mut flat_outputs))
            .await
            .map_err(|e| Status::internal(format!("Failed to process input stream: {}", e)))?;

        // Reconstruct input arrays from flat data
        for (name, flat_data) in flat_inputs {
            if let Some(array) = inputs.get_mut(&name) {
                if flat_data.len() != array.len() {
                    return Err(Status::internal(format!(
                        "Size mismatch for input variable {}: expected {}, got {}",
                        name,
                        array.len(),
                        flat_data.len()
                    )));
                }

                for (i, &value) in flat_data.iter().enumerate() {
                    if let Some(elem) = array.get_mut(i) {
                        *elem = value;
                    }
                }
            }
        }

        // Reconstruct output arrays from flat data
        for (name, flat_data) in flat_outputs {
            if let Some(array) = outputs.get_mut(&name) {
                if flat_data.len() != array.len() {
                    return Err(Status::internal(format!(
                        "Size mismatch for output variable {}: expected {}, got {}",
                        name,
                        array.len(),
                        flat_data.len()
                    )));
                }

                for (i, &value) in flat_data.iter().enumerate() {
                    if let Some(elem) = array.get_mut(i) {
                        *elem = value;
                    }
                }
            }
        }

        Ok((inputs, outputs))
    }
}

#[tonic::async_trait]
impl<D: ImplicitDiscipline + 'static> ImplicitService for ImplicitServer<D> {
    type ComputeResidualsStream =
        Pin<Box<dyn Stream<Item = std::result::Result<Array, Status>> + Send>>;

    async fn compute_residuals(
        &self,
        request: Request<Streaming<Array>>,
    ) -> std::result::Result<Response<Self::ComputeResidualsStream>, Status> {
        self.log_if_verbose("ComputeResiduals called").await;

        let input_stream = request.into_inner();
        let (inputs, outputs) = self.process_input_and_output_streams(input_stream).await?;

        // Call the compute_residuals function
        let discipline = self.base.discipline().read().await;
        let residuals = discipline
            .compute_residuals(&inputs, &outputs)
            .await
            .map_err(|e| Status::internal(format!("Compute residuals failed: {}", e)))?;
        drop(discipline);

        // Stream residuals back to client
        let residual_stream = self
            .stream_arrays_as_chunks(&residuals, VariableType::KResidual)
            .await;

        Ok(Response::new(residual_stream))
    }

    type SolveResidualsStream =
        Pin<Box<dyn Stream<Item = std::result::Result<Array, Status>> + Send>>;

    async fn solve_residuals(
        &self,
        request: Request<Streaming<Array>>,
    ) -> std::result::Result<Response<Self::SolveResidualsStream>, Status> {
        self.log_if_verbose("SolveResiduals called").await;

        // Preallocate input arrays
        let (mut inputs, mut flat_inputs) = self
            .base
            .preallocate_inputs()
            .await
            .map_err(|e| Status::internal(format!("Failed to preallocate inputs: {}", e)))?;

        // Process input stream
        let input_stream = request.into_inner();
        self.base
            .process_input_stream(input_stream, &mut flat_inputs, None)
            .await
            .map_err(|e| Status::internal(format!("Failed to process input stream: {}", e)))?;

        // Reconstruct arrays from flat data
        for (name, flat_data) in flat_inputs {
            if let Some(array) = inputs.get_mut(&name) {
                if flat_data.len() != array.len() {
                    return Err(Status::internal(format!(
                        "Size mismatch for variable {}: expected {}, got {}",
                        name,
                        array.len(),
                        flat_data.len()
                    )));
                }

                for (i, &value) in flat_data.iter().enumerate() {
                    if let Some(elem) = array.get_mut(i) {
                        *elem = value;
                    }
                }
            }
        }

        // Call the solve_residuals function
        let discipline = self.base.discipline().read().await;
        let outputs = discipline
            .solve_residuals(&inputs)
            .await
            .map_err(|e| Status::internal(format!("Solve residuals failed: {}", e)))?;
        drop(discipline);

        // Stream outputs back to client
        let output_stream = self
            .stream_arrays_as_chunks(&outputs, VariableType::KOutput)
            .await;

        Ok(Response::new(output_stream))
    }

    type ComputeResidualGradientsStream =
        Pin<Box<dyn Stream<Item = std::result::Result<Array, Status>> + Send>>;

    async fn compute_residual_gradients(
        &self,
        request: Request<Streaming<Array>>,
    ) -> std::result::Result<Response<Self::ComputeResidualGradientsStream>, Status> {
        self.log_if_verbose("ComputeResidualGradients called").await;

        let input_stream = request.into_inner();
        let (inputs, outputs) = self.process_input_and_output_streams(input_stream).await?;

        // Call the residual_partials function
        let discipline = self.base.discipline().read().await;
        let partials = discipline
            .residual_partials(&inputs, &outputs)
            .await
            .map_err(|e| Status::internal(format!("Compute residual gradients failed: {}", e)))?;
        drop(discipline);

        // Stream partials back to client
        let partial_stream = self.stream_partials_as_chunks(&partials).await;

        Ok(Response::new(partial_stream))
    }
}

// Delegate DisciplineService methods to the base server
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
}
