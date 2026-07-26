use std::collections::HashMap;
use std::time::Duration;
use tonic::transport::Channel;

use crate::client::base::DisciplineClient;
use crate::philote_info::{
    explicit_service_client::ExplicitServiceClient, PartialsMetaData, VariableMetaData,
    VariableType,
};
use crate::types::StreamOptions;
use crate::{wire, ArrayMap, DiscreteMap, PartialMap, PhiloteError, Result};

/// Client for an explicit discipline server.
pub struct ExplicitClient {
    base: DisciplineClient,
    explicit: ExplicitServiceClient<Channel>,
}

impl ExplicitClient {
    pub async fn connect<T>(dst: T) -> Result<Self>
    where
        T: std::convert::TryInto<tonic::transport::Endpoint>,
        T::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        let channel = tonic::transport::Endpoint::new(dst)
            .map_err(|e| PhiloteError::config_error(format!("Invalid endpoint: {:?}", e)))?
            .connect()
            .await
            .map_err(|e| PhiloteError::config_error(format!("Failed to connect: {}", e)))?;

        Ok(Self {
            base: DisciplineClient::from_channel(channel.clone()),
            explicit: ExplicitServiceClient::new(channel),
        })
    }

    pub fn with_stream_options(mut self, options: StreamOptions) -> Self {
        self.base = self.base.with_stream_options(options);
        self
    }

    pub fn with_rpc_timeout(mut self, timeout: Duration) -> Self {
        self.base = self.base.with_rpc_timeout(timeout);
        self
    }

    /// Access the underlying discipline client.
    pub fn base(&self) -> &DisciplineClient {
        &self.base
    }

    // --- Delegated discipline RPCs ---

    pub async fn get_info(&mut self) -> Result<crate::philote_info::DisciplineProperties> {
        self.base.get_info().await
    }

    pub async fn set_stream_options(&mut self, options: StreamOptions) -> Result<()> {
        self.base.set_stream_options(options).await
    }

    pub async fn get_available_options(&mut self) -> Result<HashMap<String, String>> {
        self.base.get_available_options().await
    }

    pub async fn set_options(&mut self, options: HashMap<String, serde_json::Value>) -> Result<()> {
        self.base.set_options(options).await
    }

    pub async fn setup(&mut self) -> Result<()> {
        self.base.setup().await
    }

    pub async fn get_variable_definitions(&mut self) -> Result<Vec<VariableMetaData>> {
        self.base.get_variable_definitions().await
    }

    pub async fn get_partial_definitions(&mut self) -> Result<Vec<PartialsMetaData>> {
        self.base.get_partial_definitions().await
    }

    pub async fn get_dynamic_variables(&mut self) -> Result<Vec<VariableMetaData>> {
        self.base.get_dynamic_variables().await
    }

    pub async fn send_variable_shapes(&mut self, shapes: Vec<VariableMetaData>) -> Result<()> {
        self.base.send_variable_shapes(shapes).await
    }

    pub fn var_meta(&self) -> &[VariableMetaData] {
        self.base.var_meta()
    }

    // --- Explicit RPCs ---

    /// Evaluate the discipline.
    pub async fn compute_function(&mut self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let (outputs, _) = self
            .compute_function_with_discrete(inputs, &HashMap::new())
            .await?;
        Ok(outputs)
    }

    /// Evaluate the discipline, passing and receiving discrete variables.
    pub async fn compute_function_with_discrete(
        &mut self,
        inputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        self.base.require_metadata()?;

        let messages = wire::assemble_input_messages(
            inputs,
            None,
            discrete_inputs,
            self.base.stream_options().max_double_per_slice,
        );

        let response = self
            .explicit
            .compute_function(self.base.make_request(tokio_stream::iter(messages)))
            .await?;

        let mut stream = response.into_inner();
        wire::recover_arrays(&mut stream, self.base.var_meta(), VariableType::KOutput).await
    }

    /// Evaluate the discipline's gradients.
    pub async fn compute_gradient(&mut self, inputs: &ArrayMap) -> Result<PartialMap> {
        self.compute_gradient_with_discrete(inputs, &HashMap::new())
            .await
    }

    /// Evaluate the discipline's gradients, passing discrete variables.
    pub async fn compute_gradient_with_discrete(
        &mut self,
        inputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<PartialMap> {
        self.base.require_metadata()?;

        let messages = wire::assemble_input_messages(
            inputs,
            None,
            discrete_inputs,
            self.base.stream_options().max_double_per_slice,
        );

        let response = self
            .explicit
            .compute_gradient(self.base.make_request(tokio_stream::iter(messages)))
            .await?;

        let mut stream = response.into_inner();
        wire::recover_partials(&mut stream, self.base.var_meta(), self.base.partials_meta()).await
    }
}

impl Clone for ExplicitClient {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            explicit: self.explicit.clone(),
        }
    }
}
