use std::collections::{BTreeMap, HashMap};
use std::time::Duration;
use tonic::transport::Channel;
use tonic::Request;

use crate::philote_info::{
    discipline_service_client::DisciplineServiceClient, DisciplineOptions, DisciplineProperties,
    PartialsMetaData, StreamOptions as ProtoStreamOptions, VariableMetaData,
};
use crate::types::StreamOptions;
use crate::{PhiloteError, Result};

pub struct DisciplineClient {
    client: DisciplineServiceClient<Channel>,
    stream_options: StreamOptions,
    rpc_timeout: Option<Duration>,
}

impl DisciplineClient {
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

        let client = DisciplineServiceClient::new(channel);

        Ok(Self {
            client,
            stream_options: StreamOptions::default(),
            rpc_timeout: None,
        })
    }

    pub fn with_stream_options(mut self, options: StreamOptions) -> Self {
        self.stream_options = options;
        self
    }

    pub fn with_rpc_timeout(mut self, timeout: Duration) -> Self {
        self.rpc_timeout = Some(timeout);
        self
    }

    pub fn stream_options(&self) -> &StreamOptions {
        &self.stream_options
    }

    fn make_request<T>(&self, inner: T) -> Request<T> {
        let mut req = Request::new(inner);
        if let Some(timeout) = self.rpc_timeout {
            req.set_timeout(timeout);
        }
        req
    }

    pub async fn get_info(&mut self) -> Result<DisciplineProperties> {
        let response = self.client.get_info(self.make_request(())).await?;
        Ok(response.into_inner())
    }

    pub async fn set_stream_options(&mut self, options: StreamOptions) -> Result<()> {
        let proto_options = ProtoStreamOptions::from(options);
        self.client
            .set_stream_options(self.make_request(proto_options))
            .await?;
        self.stream_options = options;
        Ok(())
    }

    pub async fn get_available_options(&mut self) -> Result<HashMap<String, String>> {
        let response = self.client.get_available_options(self.make_request(())).await?;
        let options_list = response.into_inner();

        let mut options_map = HashMap::new();

        for (option_name, data_type_int) in
            options_list.options.iter().zip(options_list.r#type.iter())
        {
            let type_str = match data_type_int {
                0 => "bool",
                1 => "int",
                2 => "double",
                3 => "string",
                4 => "struct",
                _ => {
                    return Err(PhiloteError::InvalidVariableType(format!(
                        "Unknown data type: {}",
                        data_type_int
                    )))
                }
            };

            options_map.insert(option_name.clone(), type_str.to_string());
        }

        Ok(options_map)
    }

    pub async fn set_options(
        &mut self,
        options: HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        let fields: BTreeMap<String, prost_types::Value> = options
            .into_iter()
            .map(|(k, v)| (k, json_to_proto_value(&v)))
            .collect();

        let struct_value = Some(prost_types::Struct { fields });

        let discipline_options = DisciplineOptions {
            options: struct_value,
        };

        self.client
            .set_options(self.make_request(discipline_options))
            .await?;
        Ok(())
    }

    pub async fn setup(&mut self) -> Result<()> {
        self.client.setup(self.make_request(())).await?;
        Ok(())
    }

    pub async fn get_variable_definitions(&mut self) -> Result<Vec<VariableMetaData>> {
        let response = self
            .client
            .get_variable_definitions(self.make_request(()))
            .await?;
        let mut stream = response.into_inner();

        let mut variables = Vec::new();
        while let Some(var_meta) = stream.message().await? {
            variables.push(var_meta);
        }

        Ok(variables)
    }

    pub async fn get_partial_definitions(&mut self) -> Result<Vec<PartialsMetaData>> {
        let response = self
            .client
            .get_partial_definitions(self.make_request(()))
            .await?;
        let mut stream = response.into_inner();

        let mut partials = Vec::new();
        while let Some(partial_meta) = stream.message().await? {
            partials.push(partial_meta);
        }

        Ok(partials)
    }

    pub async fn get_dynamic_variables(&mut self) -> Result<Vec<VariableMetaData>> {
        let all_vars = self.get_variable_definitions().await?;
        Ok(all_vars.into_iter().filter(|v| v.dynamic_shape).collect())
    }

    pub async fn send_variable_shapes(&mut self, shapes: Vec<VariableMetaData>) -> Result<()> {
        let stream = tokio_stream::iter(shapes);
        self.client
            .set_variable_shapes(self.make_request(stream))
            .await?;
        Ok(())
    }
}

impl Clone for DisciplineClient {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            stream_options: self.stream_options,
            rpc_timeout: self.rpc_timeout,
        }
    }
}

fn json_to_proto_value(v: &serde_json::Value) -> prost_types::Value {
    use prost_types::value::Kind;
    let kind = match v {
        serde_json::Value::Null => Some(Kind::NullValue(0)),
        serde_json::Value::Bool(b) => Some(Kind::BoolValue(*b)),
        serde_json::Value::Number(n) => Some(Kind::NumberValue(n.as_f64().unwrap_or(0.0))),
        serde_json::Value::String(s) => Some(Kind::StringValue(s.clone())),
        serde_json::Value::Array(arr) => {
            let values: Vec<prost_types::Value> = arr.iter().map(json_to_proto_value).collect();
            Some(Kind::ListValue(prost_types::ListValue { values }))
        }
        serde_json::Value::Object(map) => {
            let fields: BTreeMap<String, prost_types::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), json_to_proto_value(v)))
                .collect();
            Some(Kind::StructValue(prost_types::Struct { fields }))
        }
    };
    prost_types::Value { kind }
}
