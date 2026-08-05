use std::collections::HashMap;
use std::time::Duration;
use tonic::transport::Channel;
use tonic::Request;

use crate::discrete::json_map_to_struct;
use crate::philote_info::{
    discipline_service_client::DisciplineServiceClient, DisciplineOptions, DisciplineProperties,
    PartialsMetaData, StreamOptions as ProtoStreamOptions, VariableMetaData, VariableType,
};
use crate::types::StreamOptions;
use crate::{PhiloteError, Result};

/// Build a [`VariableMetaData`] message for resolving a dynamic variable's shape.
///
/// Pass the result to [`DisciplineClient::send_variable_shapes`].
pub fn variable_shape_meta(
    name: &str,
    shape: &[usize],
    var_type: VariableType,
) -> VariableMetaData {
    VariableMetaData {
        r#type: var_type.into(),
        name: name.to_string(),
        shape: shape.iter().map(|&d| d as i64).collect(),
        units: String::new(),
        dynamic_shape: true,
    }
}

/// Client for the RPCs shared by explicit and implicit disciplines.
///
/// Variable and partials metadata fetched from the server is cached, because both
/// are needed to reconstruct correctly shaped arrays from the flat chunk stream.
pub struct DisciplineClient {
    client: DisciplineServiceClient<Channel>,
    stream_options: StreamOptions,
    rpc_timeout: Option<Duration>,
    var_meta: Vec<VariableMetaData>,
    discrete_meta: Vec<VariableMetaData>,
    partials_meta: Vec<PartialsMetaData>,
    /// Whether `get_variable_definitions` has completed. Tracked separately from
    /// `var_meta` being non-empty, since a discipline may declare only discrete
    /// variables.
    metadata_fetched: bool,
}

impl DisciplineClient {
    /// Connect to a discipline server at `dst` (for example `"http://host:50051"`).
    ///
    /// Fails with [`PhiloteError::ConfigurationError`] if the endpoint is malformed
    /// or the connection cannot be established.
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

        Ok(Self::from_channel(channel))
    }

    /// Build a client from an existing channel.
    pub fn from_channel(channel: Channel) -> Self {
        Self {
            client: DisciplineServiceClient::new(channel),
            stream_options: StreamOptions::default(),
            rpc_timeout: None,
            var_meta: Vec::new(),
            discrete_meta: Vec::new(),
            partials_meta: Vec::new(),
            metadata_fetched: false,
        }
    }

    /// Set the local streaming options used to chunk outgoing arrays.
    ///
    /// This only configures the client; use
    /// [`set_stream_options`](Self::set_stream_options) to also inform the server.
    pub fn with_stream_options(mut self, options: StreamOptions) -> Self {
        self.stream_options = options;
        self
    }

    /// Apply a deadline to every RPC issued by this client.
    ///
    /// Without one, a stalled server blocks the call indefinitely.
    pub fn with_rpc_timeout(mut self, timeout: Duration) -> Self {
        self.rpc_timeout = Some(timeout);
        self
    }

    /// The streaming options currently in effect for this client.
    pub fn stream_options(&self) -> &StreamOptions {
        &self.stream_options
    }

    /// Cached continuous variable metadata, populated by
    /// [`get_variable_definitions`](Self::get_variable_definitions).
    pub fn var_meta(&self) -> &[VariableMetaData] {
        &self.var_meta
    }

    /// Cached discrete variable metadata.
    pub fn discrete_meta(&self) -> &[VariableMetaData] {
        &self.discrete_meta
    }

    /// Cached partials metadata, populated by
    /// [`get_partial_definitions`](Self::get_partial_definitions).
    pub fn partials_meta(&self) -> &[PartialsMetaData] {
        &self.partials_meta
    }

    /// Error if variable metadata has not been fetched yet.
    ///
    /// Array reconstruction needs declared shapes, so a compute call before
    /// `setup`/`get_variable_definitions` cannot produce correct results.
    /// Philote-Python silently returns empty maps in this situation; erroring is a
    /// deliberate divergence.
    ///
    /// This tracks whether the fetch happened, not whether it returned anything: a
    /// discipline may legitimately declare only discrete variables, leaving the
    /// continuous metadata empty.
    pub(crate) fn require_metadata(&self) -> Result<()> {
        if !self.metadata_fetched {
            return Err(PhiloteError::SetupNotCalled);
        }
        Ok(())
    }

    pub(crate) fn make_request<T>(&self, inner: T) -> Request<T> {
        let mut req = Request::new(inner);
        if let Some(timeout) = self.rpc_timeout {
            req.set_timeout(timeout);
        }
        req
    }

    /// Fetch the server's discipline properties (name, version, and capabilities).
    pub async fn get_info(&mut self) -> Result<DisciplineProperties> {
        let response = self.client.get_info(self.make_request(())).await?;
        Ok(response.into_inner())
    }

    /// Negotiate streaming options with the server.
    ///
    /// The local copy is updated only after the server accepts, so both sides chunk
    /// with the same limits.
    pub async fn set_stream_options(&mut self, options: StreamOptions) -> Result<()> {
        let proto_options = ProtoStreamOptions::from(options);
        self.client
            .set_stream_options(self.make_request(proto_options))
            .await?;
        self.stream_options = options;
        Ok(())
    }

    /// List the options the discipline accepts, mapped to their declared type name
    /// (`"bool"`, `"int"`, `"float"`, `"str"`, or `"dict"`).
    ///
    /// Errors with [`PhiloteError::InvalidVariableType`] if the server reports a
    /// type code outside that set.
    pub async fn get_available_options(&mut self) -> Result<HashMap<String, String>> {
        let response = self
            .client
            .get_available_options(self.make_request(()))
            .await?;
        let options_list = response.into_inner();

        let mut options_map = HashMap::new();
        for (option_name, data_type_int) in
            options_list.options.iter().zip(options_list.r#type.iter())
        {
            // Canonical names, matching Philote-Python's vocabulary.
            let type_str = match data_type_int {
                0 => "bool",
                1 => "int",
                2 => "float",
                3 => "str",
                4 => "dict",
                other => {
                    return Err(PhiloteError::InvalidVariableType(format!(
                        "Unknown data type: {other}"
                    )))
                }
            };
            options_map.insert(option_name.clone(), type_str.to_string());
        }

        Ok(options_map)
    }

    /// Set discipline options on the server.
    ///
    /// Options that change variable shapes only take effect once
    /// [`setup`](Self::setup) is called again.
    pub async fn set_options(&mut self, options: HashMap<String, serde_json::Value>) -> Result<()> {
        let discipline_options = DisciplineOptions {
            options: Some(json_map_to_struct(&options)),
        };
        self.client
            .set_options(self.make_request(discipline_options))
            .await?;
        Ok(())
    }

    /// Run the discipline's setup on the server.
    ///
    /// The server rebuilds its metadata from scratch, so the local cache is
    /// dropped: reusing it (for example after changing an option that resizes a
    /// variable) would decode later responses against stale shapes.
    pub async fn setup(&mut self) -> Result<()> {
        self.client.setup(self.make_request(())).await?;
        self.invalidate_metadata();
        Ok(())
    }

    /// Drop all cached metadata.
    fn invalidate_metadata(&mut self) {
        self.var_meta.clear();
        self.discrete_meta.clear();
        self.partials_meta.clear();
        self.metadata_fetched = false;
    }

    /// Fetch and cache variable definitions.
    ///
    /// Discrete metadata is separated from continuous metadata, so array
    /// preallocation never sees a discrete variable.
    pub async fn get_variable_definitions(&mut self) -> Result<Vec<VariableMetaData>> {
        let response = self
            .client
            .get_variable_definitions(self.make_request(()))
            .await?;
        let mut stream = response.into_inner();

        self.var_meta.clear();
        self.discrete_meta.clear();

        while let Some(meta) = stream.message().await? {
            let var_type = VariableType::try_from(meta.r#type).map_err(|_| {
                PhiloteError::InvalidVariableType(format!("Invalid type: {}", meta.r#type))
            })?;
            match var_type {
                VariableType::KDiscreteInput | VariableType::KDiscreteOutput => {
                    self.discrete_meta.push(meta)
                }
                _ => self.var_meta.push(meta),
            }
        }

        self.metadata_fetched = true;
        Ok(self.var_meta.clone())
    }

    /// Fetch and cache partials definitions.
    pub async fn get_partial_definitions(&mut self) -> Result<Vec<PartialsMetaData>> {
        let response = self
            .client
            .get_partial_definitions(self.make_request(()))
            .await?;
        let mut stream = response.into_inner();

        self.partials_meta.clear();
        while let Some(partial_meta) = stream.message().await? {
            self.partials_meta.push(partial_meta);
        }

        Ok(self.partials_meta.clone())
    }

    /// Cached variables that were declared with a dynamic shape.
    pub fn dynamic_variables(&self) -> Vec<VariableMetaData> {
        self.var_meta
            .iter()
            .filter(|v| v.dynamic_shape)
            .cloned()
            .collect()
    }

    /// Fetch variable definitions and return those with a dynamic shape.
    pub async fn get_dynamic_variables(&mut self) -> Result<Vec<VariableMetaData>> {
        self.get_variable_definitions().await?;
        Ok(self.dynamic_variables())
    }

    /// Send resolved shapes for dynamic variables and update the local cache.
    ///
    /// Updating the cache matters: subsequent responses are decoded into arrays
    /// preallocated from it, so a stale cache would produce zero-length arrays.
    pub async fn send_variable_shapes(&mut self, shapes: Vec<VariableMetaData>) -> Result<()> {
        let stream = tokio_stream::iter(shapes.clone());
        self.client
            .set_variable_shapes(self.make_request(stream))
            .await?;

        // Declared partial shapes were derived from the previous variable shapes,
        // so they no longer describe what the server will send. Clearing them makes
        // `recover_partials` re-derive from the updated variable metadata.
        for partial in self.partials_meta.iter_mut() {
            partial.shape.clear();
        }

        for meta in shapes {
            for var in self.var_meta.iter_mut() {
                let matches_target = var.name == meta.name && var.r#type == meta.r#type;
                // An implicit output's residual shares its name and shape.
                let matches_residual = var.name == meta.name
                    && meta.r#type == i32::from(VariableType::KOutput)
                    && var.r#type == i32::from(VariableType::KResidual);

                if matches_target || matches_residual {
                    var.shape.clone_from(&meta.shape);
                }
            }
        }

        Ok(())
    }
}

impl Clone for DisciplineClient {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            stream_options: self.stream_options,
            rpc_timeout: self.rpc_timeout,
            var_meta: self.var_meta.clone(),
            discrete_meta: self.discrete_meta.clone(),
            partials_meta: self.partials_meta.clone(),
            metadata_fetched: self.metadata_fetched,
        }
    }
}
