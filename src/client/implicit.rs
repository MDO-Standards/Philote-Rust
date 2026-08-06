use std::collections::HashMap;
use std::time::Duration;
use tonic::transport::Channel;

use crate::client::base::DisciplineClient;
use crate::philote_info::{
    implicit_service_client::ImplicitServiceClient, PartialsMetaData, VariableMetaData,
    VariableType,
};
use crate::types::StreamOptions;
use crate::{wire, ArrayMap, DiscreteMap, PartialMap, PhiloteError, Result};

/// Client for an implicit discipline server.
pub struct ImplicitClient {
    base: DisciplineClient,
    implicit: ImplicitServiceClient<Channel>,
}

impl ImplicitClient {
    /// Connect to an implicit discipline server at `dst`.
    ///
    /// One channel backs both the shared `DisciplineService` client and the
    /// `ImplicitService` client.
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
            implicit: ImplicitServiceClient::new(channel),
        })
    }

    /// Set the local streaming options; see
    /// [`DisciplineClient::with_stream_options`].
    pub fn with_stream_options(mut self, options: StreamOptions) -> Self {
        self.base = self.base.with_stream_options(options);
        self
    }

    /// Apply a deadline to every RPC; see [`DisciplineClient::with_rpc_timeout`].
    ///
    /// The implicit RPCs build their requests through the base client, so they carry
    /// the same deadline.
    pub fn with_rpc_timeout(mut self, timeout: Duration) -> Self {
        self.base = self.base.with_rpc_timeout(timeout);
        self
    }

    /// Access the underlying discipline client.
    pub fn base(&self) -> &DisciplineClient {
        &self.base
    }

    // --- Delegated discipline RPCs ---

    /// Fetch the server's discipline properties; see [`DisciplineClient::get_info`].
    pub async fn get_info(&mut self) -> Result<crate::philote_info::DisciplineProperties> {
        self.base.get_info().await
    }

    /// Negotiate streaming options with the server; see
    /// [`DisciplineClient::set_stream_options`].
    pub async fn set_stream_options(&mut self, options: StreamOptions) -> Result<()> {
        self.base.set_stream_options(options).await
    }

    /// List the options the discipline accepts; see
    /// [`DisciplineClient::get_available_options`].
    pub async fn get_available_options(&mut self) -> Result<HashMap<String, String>> {
        self.base.get_available_options().await
    }

    /// Set discipline options on the server; see [`DisciplineClient::set_options`].
    pub async fn set_options(&mut self, options: HashMap<String, serde_json::Value>) -> Result<()> {
        self.base.set_options(options).await
    }

    /// Run the discipline's setup on the server; see [`DisciplineClient::setup`].
    ///
    /// This discards the cached metadata, so
    /// [`get_variable_definitions`](Self::get_variable_definitions) must be called
    /// again before any residual or solve call.
    pub async fn setup(&mut self) -> Result<()> {
        self.base.setup().await
    }

    /// Fetch and cache variable definitions; see
    /// [`DisciplineClient::get_variable_definitions`].
    ///
    /// Required before [`compute_residuals`](Self::compute_residuals),
    /// [`solve_residuals`](Self::solve_residuals), or
    /// [`compute_residual_gradients`](Self::compute_residual_gradients), which decode
    /// responses against the cached shapes.
    pub async fn get_variable_definitions(&mut self) -> Result<Vec<VariableMetaData>> {
        self.base.get_variable_definitions().await
    }

    /// Fetch and cache partials definitions; see
    /// [`DisciplineClient::get_partial_definitions`].
    pub async fn get_partial_definitions(&mut self) -> Result<Vec<PartialsMetaData>> {
        self.base.get_partial_definitions().await
    }

    /// Fetch variable definitions and return those with a dynamic shape; see
    /// [`DisciplineClient::get_dynamic_variables`].
    pub async fn get_dynamic_variables(&mut self) -> Result<Vec<VariableMetaData>> {
        self.base.get_dynamic_variables().await
    }

    /// Send resolved shapes for dynamic variables; see
    /// [`DisciplineClient::send_variable_shapes`].
    ///
    /// Also refreshes the cached shapes — including the residual entries that mirror
    /// each output — so no re-fetch is needed afterwards.
    pub async fn send_variable_shapes(&mut self, shapes: Vec<VariableMetaData>) -> Result<()> {
        self.base.send_variable_shapes(shapes).await
    }

    /// Cached continuous variable metadata; see [`DisciplineClient::var_meta`].
    ///
    /// Empty until [`get_variable_definitions`](Self::get_variable_definitions) runs.
    pub fn var_meta(&self) -> &[VariableMetaData] {
        self.base.var_meta()
    }

    // --- Implicit RPCs ---

    /// Evaluate the residuals at the given inputs and outputs.
    pub async fn compute_residuals(
        &mut self,
        inputs: &ArrayMap,
        outputs: &ArrayMap,
    ) -> Result<ArrayMap> {
        let (residuals, _) = self
            .compute_residuals_with_discrete(inputs, outputs, &HashMap::new())
            .await?;
        Ok(residuals)
    }

    /// Evaluate the residuals, passing and receiving discrete variables.
    ///
    /// The response is filtered to `kResidual`. A residual carries the same name as
    /// its output, so type is the only thing distinguishing them.
    pub async fn compute_residuals_with_discrete(
        &mut self,
        inputs: &ArrayMap,
        outputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        self.base.require_metadata()?;

        let messages = wire::assemble_input_messages(
            inputs,
            Some(outputs),
            discrete_inputs,
            self.base.stream_options().max_double_per_slice,
        );

        let response = self
            .implicit
            .compute_residuals(self.base.make_request(tokio_stream::iter(messages)))
            .await?;

        let mut stream = response.into_inner();
        wire::recover_arrays(&mut stream, self.base.var_meta(), VariableType::KResidual).await
    }

    /// Solve the discipline for outputs that zero the residuals.
    pub async fn solve_residuals(&mut self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let (outputs, _) = self
            .solve_residuals_with_discrete(inputs, &HashMap::new())
            .await?;
        Ok(outputs)
    }

    /// Solve the discipline, passing and receiving discrete variables.
    pub async fn solve_residuals_with_discrete(
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
            .implicit
            .solve_residuals(self.base.make_request(tokio_stream::iter(messages)))
            .await?;

        let mut stream = response.into_inner();
        wire::recover_arrays(&mut stream, self.base.var_meta(), VariableType::KOutput).await
    }

    /// Evaluate the residual gradients.
    pub async fn compute_residual_gradients(
        &mut self,
        inputs: &ArrayMap,
        outputs: &ArrayMap,
    ) -> Result<PartialMap> {
        self.compute_residual_gradients_with_discrete(inputs, outputs, &HashMap::new())
            .await
    }

    /// Evaluate the residual gradients, passing discrete variables.
    pub async fn compute_residual_gradients_with_discrete(
        &mut self,
        inputs: &ArrayMap,
        outputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<PartialMap> {
        self.base.require_metadata()?;

        let messages = wire::assemble_input_messages(
            inputs,
            Some(outputs),
            discrete_inputs,
            self.base.stream_options().max_double_per_slice,
        );

        let response = self
            .implicit
            .compute_residual_gradients(self.base.make_request(tokio_stream::iter(messages)))
            .await?;

        let mut stream = response.into_inner();
        wire::recover_partials(&mut stream, self.base.var_meta(), self.base.partials_meta()).await
    }
}

impl Clone for ImplicitClient {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            implicit: self.implicit.clone(),
        }
    }
}
