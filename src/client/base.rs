use std::collections::{BTreeMap, HashMap};
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
        })
    }

    pub fn with_stream_options(mut self, options: StreamOptions) -> Self {
        self.stream_options = options;
        self
    }

    pub async fn get_info(&mut self) -> Result<DisciplineProperties> {
        let response = self.client.get_info(Request::new(())).await?;
        Ok(response.into_inner())
    }

    pub async fn set_stream_options(&mut self, options: StreamOptions) -> Result<()> {
        let proto_options = ProtoStreamOptions::from(options);
        self.client
            .set_stream_options(Request::new(proto_options))
            .await?;
        self.stream_options = options;
        Ok(())
    }

    pub async fn get_available_options(&mut self) -> Result<HashMap<String, String>> {
        let response = self.client.get_available_options(Request::new(())).await?;
        let options_list = response.into_inner();

        let mut options_map = HashMap::new();

        for (option_name, data_type_int) in
            options_list.options.iter().zip(options_list.r#type.iter())
        {
            let type_str = match data_type_int {
                0 => "bool",   // kBool
                1 => "int",    // kInt
                2 => "double", // kDouble
                3 => "string", // kString
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
        _options: HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        // For now, just create an empty struct - proper conversion would need more work
        let struct_value = Some(prost_types::Struct {
            fields: BTreeMap::new(),
        });

        let discipline_options = DisciplineOptions {
            options: struct_value,
        };

        self.client
            .set_options(Request::new(discipline_options))
            .await?;
        Ok(())
    }

    pub async fn setup(&mut self) -> Result<()> {
        self.client.setup(Request::new(())).await?;
        Ok(())
    }

    pub async fn get_variable_definitions(&mut self) -> Result<Vec<VariableMetaData>> {
        let response = self
            .client
            .get_variable_definitions(Request::new(()))
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
            .get_partial_definitions(Request::new(()))
            .await?;
        let mut stream = response.into_inner();

        let mut partials = Vec::new();
        while let Some(partial_meta) = stream.message().await? {
            partials.push(partial_meta);
        }

        Ok(partials)
    }
}

impl Clone for DisciplineClient {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            stream_options: self.stream_options,
        }
    }
}
