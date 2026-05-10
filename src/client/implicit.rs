use std::collections::HashMap;
use std::time::Duration;
use tokio_stream::StreamExt;
use tonic::transport::Channel;
use tonic::{Request, Streaming};

use crate::client::base::DisciplineClient;
use crate::philote_info::{
    implicit_service_client::ImplicitServiceClient, variable_message::Payload, Array,
    DiscreteVariable, VariableMessage, VariableType,
};
use crate::types::{ArrayChunker, ArrayData, StreamOptions};
use crate::{ArrayMap, DiscreteMap, PartialMap, PhiloteError, Result};

pub struct ImplicitClient {
    base_client: DisciplineClient,
    implicit_client: ImplicitServiceClient<Channel>,
    stream_options: StreamOptions,
    rpc_timeout: Option<Duration>,
}

impl ImplicitClient {
    pub async fn connect<T>(dst: T) -> Result<Self>
    where
        T: std::convert::TryInto<tonic::transport::Endpoint> + Clone,
        T::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        let base_client = DisciplineClient::connect(dst.clone()).await?;

        let channel = tonic::transport::Endpoint::new(dst)
            .map_err(|e| PhiloteError::config_error(format!("Invalid endpoint: {:?}", e)))?
            .connect()
            .await
            .map_err(|e| PhiloteError::config_error(format!("Failed to connect: {}", e)))?;

        let implicit_client = ImplicitServiceClient::new(channel);

        Ok(Self {
            base_client,
            implicit_client,
            stream_options: StreamOptions::default(),
            rpc_timeout: None,
        })
    }

    pub fn with_stream_options(mut self, options: StreamOptions) -> Self {
        self.stream_options = options;
        self.base_client = self.base_client.with_stream_options(options);
        self
    }

    pub fn with_rpc_timeout(mut self, timeout: Duration) -> Self {
        self.rpc_timeout = Some(timeout);
        self.base_client = self.base_client.with_rpc_timeout(timeout);
        self
    }

    fn make_request<T>(&self, inner: T) -> Request<T> {
        let mut req = Request::new(inner);
        if let Some(timeout) = self.rpc_timeout {
            req.set_timeout(timeout);
        }
        req
    }

    // Delegate base client methods
    pub async fn get_info(&mut self) -> Result<crate::philote_info::DisciplineProperties> {
        self.base_client.get_info().await
    }

    pub async fn set_stream_options(&mut self, options: StreamOptions) -> Result<()> {
        self.stream_options = options;
        self.base_client.set_stream_options(options).await
    }

    pub async fn get_available_options(&mut self) -> Result<HashMap<String, String>> {
        self.base_client.get_available_options().await
    }

    pub async fn set_options(&mut self, options: HashMap<String, serde_json::Value>) -> Result<()> {
        self.base_client.set_options(options).await
    }

    pub async fn setup(&mut self) -> Result<()> {
        self.base_client.setup().await
    }

    pub async fn get_variable_definitions(
        &mut self,
    ) -> Result<Vec<crate::philote_info::VariableMetaData>> {
        self.base_client.get_variable_definitions().await
    }

    pub async fn get_partial_definitions(
        &mut self,
    ) -> Result<Vec<crate::philote_info::PartialsMetaData>> {
        self.base_client.get_partial_definitions().await
    }

    pub async fn get_dynamic_variables(
        &mut self,
    ) -> Result<Vec<crate::philote_info::VariableMetaData>> {
        self.base_client.get_dynamic_variables().await
    }

    pub async fn send_variable_shapes(
        &mut self,
        shapes: Vec<crate::philote_info::VariableMetaData>,
    ) -> Result<()> {
        self.base_client.send_variable_shapes(shapes).await
    }

    // Implicit-specific methods
    pub async fn compute_residuals(
        &mut self,
        inputs: &ArrayMap,
        outputs: &ArrayMap,
    ) -> Result<ArrayMap> {
        let (residuals, _discrete) = self
            .compute_residuals_with_discrete(inputs, outputs, &HashMap::new())
            .await?;
        Ok(residuals)
    }

    pub async fn compute_residuals_with_discrete(
        &mut self,
        inputs: &ArrayMap,
        outputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        let input_stream =
            self.create_combined_variable_message_stream(inputs, outputs, discrete_inputs);

        let response = self
            .implicit_client
            .compute_residuals(self.make_request(input_stream))
            .await?;

        let output_stream = response.into_inner();
        self.process_variable_message_output_stream(output_stream).await
    }

    pub async fn solve_residuals(&mut self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let (outputs, _discrete) = self
            .solve_residuals_with_discrete(inputs, &HashMap::new())
            .await?;
        Ok(outputs)
    }

    pub async fn solve_residuals_with_discrete(
        &mut self,
        inputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        let input_stream =
            self.create_variable_message_stream(inputs, discrete_inputs, VariableType::KInput);

        let response = self
            .implicit_client
            .solve_residuals(self.make_request(input_stream))
            .await?;

        let output_stream = response.into_inner();
        self.process_variable_message_output_stream(output_stream).await
    }

    pub async fn compute_residual_gradients(
        &mut self,
        inputs: &ArrayMap,
        outputs: &ArrayMap,
    ) -> Result<PartialMap> {
        self.compute_residual_gradients_with_discrete(inputs, outputs, &HashMap::new())
            .await
    }

    pub async fn compute_residual_gradients_with_discrete(
        &mut self,
        inputs: &ArrayMap,
        outputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<PartialMap> {
        let input_stream =
            self.create_combined_variable_message_stream(inputs, outputs, discrete_inputs);

        let response = self
            .implicit_client
            .compute_residual_gradients(self.make_request(input_stream))
            .await?;

        let partial_stream = response.into_inner();
        self.process_partial_stream(partial_stream).await
    }

    fn create_variable_message_stream(
        &self,
        inputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
        var_type: VariableType,
    ) -> impl futures_util::Stream<Item = VariableMessage> + Send {
        let chunker = ArrayChunker::new(self.stream_options.max_double_per_slice);
        let mut messages = Vec::new();

        for (name, array) in inputs {
            let flat_data = crate::utils::create_flattened_view(array);
            let chunks = chunker.chunk_array(name, &flat_data, var_type);

            for chunk in chunks {
                messages.push(VariableMessage {
                    payload: Some(Payload::Continuous(Array::from(chunk))),
                });
            }
        }

        for (name, value) in discrete_inputs {
            messages.push(VariableMessage {
                payload: Some(Payload::Discrete(DiscreteVariable {
                    name: name.clone(),
                    r#type: VariableType::KDiscreteInput.into(),
                    value: Some(value.clone()),
                })),
            });
        }

        tokio_stream::iter(messages)
    }

    fn create_combined_variable_message_stream(
        &self,
        inputs: &ArrayMap,
        outputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> impl futures_util::Stream<Item = VariableMessage> + Send {
        let chunker = ArrayChunker::new(self.stream_options.max_double_per_slice);
        let mut messages = Vec::new();

        for (name, array) in inputs {
            let flat_data = crate::utils::create_flattened_view(array);
            let chunks = chunker.chunk_array(name, &flat_data, VariableType::KInput);

            for chunk in chunks {
                messages.push(VariableMessage {
                    payload: Some(Payload::Continuous(Array::from(chunk))),
                });
            }
        }

        for (name, array) in outputs {
            let flat_data = crate::utils::create_flattened_view(array);
            let chunks = chunker.chunk_array(name, &flat_data, VariableType::KOutput);

            for chunk in chunks {
                messages.push(VariableMessage {
                    payload: Some(Payload::Continuous(Array::from(chunk))),
                });
            }
        }

        for (name, value) in discrete_inputs {
            messages.push(VariableMessage {
                payload: Some(Payload::Discrete(DiscreteVariable {
                    name: name.clone(),
                    r#type: VariableType::KDiscreteInput.into(),
                    value: Some(value.clone()),
                })),
            });
        }

        tokio_stream::iter(messages)
    }

    async fn process_variable_message_output_stream(
        &self,
        mut stream: Streaming<VariableMessage>,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        let mut array_chunks: HashMap<String, Vec<ArrayData>> = HashMap::new();
        let mut discrete_outputs: DiscreteMap = HashMap::new();

        while let Some(msg) = stream.next().await {
            let var_msg = msg?;
            match var_msg.payload {
                Some(Payload::Continuous(array)) => {
                    let array_data = ArrayData::try_from(array)?;
                    array_chunks
                        .entry(array_data.name.clone())
                        .or_default()
                        .push(array_data);
                }
                Some(Payload::Discrete(discrete_var)) => {
                    if let Some(value) = discrete_var.value {
                        discrete_outputs.insert(discrete_var.name, value);
                    }
                }
                None => {}
            }
        }

        let mut outputs = HashMap::new();
        for (name, chunks) in array_chunks {
            let array = self.reconstruct_array_from_chunks(&chunks)?;
            outputs.insert(name, array);
        }

        Ok((outputs, discrete_outputs))
    }

    async fn process_partial_stream(
        &self,
        mut stream: Streaming<VariableMessage>,
    ) -> Result<PartialMap> {
        let mut partial_chunks: HashMap<(String, String), Vec<ArrayData>> = HashMap::new();

        while let Some(msg) = stream.next().await {
            let var_msg = msg?;
            match var_msg.payload {
                Some(Payload::Continuous(array)) => {
                    let array_data = ArrayData::try_from(array)?;
                    if let Some(subname) = &array_data.subname {
                        let key = (array_data.name.clone(), subname.clone());
                        partial_chunks.entry(key).or_default().push(array_data);
                    }
                }
                _ => {}
            }
        }

        let mut partials = HashMap::new();
        for (key, chunks) in partial_chunks {
            let array = self.reconstruct_array_from_chunks(&chunks)?;
            partials.insert(key, array);
        }

        Ok(partials)
    }

    fn reconstruct_array_from_chunks(&self, chunks: &[ArrayData]) -> Result<ndarray::ArrayD<f64>> {
        if chunks.is_empty() {
            return Err(PhiloteError::array_error("No chunks to reconstruct"));
        }

        let mut sorted_chunks = chunks.to_vec();
        sorted_chunks.sort_by_key(|c| c.start);

        let total_size = sorted_chunks.last().unwrap().end + 1;
        let mut data = vec![0.0; total_size];

        for chunk in &sorted_chunks {
            let chunk_len = chunk.data.len();
            let expected_len = chunk.end - chunk.start + 1;

            if chunk_len != expected_len {
                return Err(PhiloteError::array_error(format!(
                    "Chunk size mismatch: expected {}, got {}",
                    expected_len, chunk_len
                )));
            }

            for (i, &value) in chunk.data.iter().enumerate() {
                data[chunk.start + i] = value;
            }
        }

        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[data.len()]), data)
            .map_err(|e| PhiloteError::array_error(format!("Failed to create array: {}", e)))
    }
}

impl Clone for ImplicitClient {
    fn clone(&self) -> Self {
        Self {
            base_client: self.base_client.clone(),
            implicit_client: self.implicit_client.clone(),
            stream_options: self.stream_options,
            rpc_timeout: self.rpc_timeout,
        }
    }
}
