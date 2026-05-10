use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use crate::philote_info::{
    discipline_service_server::DisciplineService, variable_message::Payload, DataType,
    DisciplineOptions, DisciplineProperties, OptionsList, PartialsMetaData,
    StreamOptions as ProtoStreamOptions, VariableMessage, VariableMetaData, VariableType,
};
use crate::traits::Discipline;
use crate::types::{ArrayData, StreamOptions};
use crate::utils::preallocate_arrays;
use crate::{ArrayMap, DiscreteMap, PhiloteError, Result};

pub struct DisciplineServer<D: Discipline + 'static> {
    discipline: Arc<RwLock<D>>,
    stream_options: Arc<RwLock<StreamOptions>>,
    verbose: bool,
}

impl<D: Discipline + 'static> DisciplineServer<D> {
    pub fn new(discipline: D) -> Self {
        Self {
            discipline: Arc::new(RwLock::new(discipline)),
            stream_options: Arc::new(RwLock::new(StreamOptions::default())),
            verbose: false,
        }
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn set_discipline(&self, discipline: D) -> impl std::future::Future<Output = ()> + Send {
        let discipline_lock = Arc::clone(&self.discipline);
        async move {
            let mut d = discipline_lock.write().await;
            *d = discipline;
        }
    }

    async fn log_if_verbose(&self, message: &str) {
        if self.verbose {
            tracing::info!("{}", message);
        }
    }

    pub fn discipline(&self) -> &Arc<RwLock<D>> {
        &self.discipline
    }

    pub fn stream_options(&self) -> &Arc<RwLock<StreamOptions>> {
        &self.stream_options
    }

    pub fn verbose(&self) -> bool {
        self.verbose
    }

    pub async fn preallocate_inputs(&self) -> Result<(ArrayMap, HashMap<String, Vec<f64>>)> {
        let discipline = self.discipline.read().await;
        let var_definitions = discipline.get_variable_definitions()?;

        let inputs = preallocate_arrays(&var_definitions, Some(VariableType::KInput))?;
        let flat_inputs: HashMap<String, Vec<f64>> = inputs
            .iter()
            .map(|(name, array)| (name.clone(), crate::utils::create_flattened_view(array)))
            .collect();

        Ok((inputs, flat_inputs))
    }

    pub async fn preallocate_outputs(&self) -> Result<(ArrayMap, HashMap<String, Vec<f64>>)> {
        let discipline = self.discipline.read().await;
        let var_definitions = discipline.get_variable_definitions()?;

        let outputs = preallocate_arrays(&var_definitions, Some(VariableType::KOutput))?;
        let flat_outputs: HashMap<String, Vec<f64>> = outputs
            .iter()
            .map(|(name, array)| (name.clone(), crate::utils::create_flattened_view(array)))
            .collect();

        Ok((outputs, flat_outputs))
    }

    pub async fn preallocate_partials(&self) -> Result<crate::PartialMap> {
        let discipline = self.discipline.read().await;
        let var_definitions = discipline.get_variable_definitions()?;
        let partial_definitions = discipline.get_partials_definitions()?;

        crate::utils::preallocate_partials(&var_definitions, &partial_definitions)
    }

    pub async fn process_variable_message_stream(
        &self,
        mut request_stream: Streaming<VariableMessage>,
        flat_inputs: &mut HashMap<String, Vec<f64>>,
        mut flat_outputs: Option<&mut HashMap<String, Vec<f64>>>,
        discrete_inputs: &mut DiscreteMap,
    ) -> Result<()> {
        while let Some(msg) = request_stream.message().await? {
            match msg.payload {
                Some(Payload::Continuous(array)) => {
                    let array_data = ArrayData::try_from(array)?;

                    match array_data.var_type {
                        VariableType::KInput => {
                            if let Some(input_vec) = flat_inputs.get_mut(&array_data.name) {
                                let start = array_data.start;
                                let end = array_data.end;

                                if end >= input_vec.len() {
                                    return Err(PhiloteError::IndexOutOfBounds {
                                        index: end,
                                        size: input_vec.len(),
                                    });
                                }

                                for (i, &value) in array_data.data.iter().enumerate() {
                                    if start + i <= end && start + i < input_vec.len() {
                                        input_vec[start + i] = value;
                                    }
                                }
                            }
                        }
                        VariableType::KOutput => {
                            if let Some(flat_outputs) = &mut flat_outputs {
                                if let Some(output_vec) = flat_outputs.get_mut(&array_data.name) {
                                    let start = array_data.start;
                                    let end = array_data.end;

                                    if end >= output_vec.len() {
                                        return Err(PhiloteError::IndexOutOfBounds {
                                            index: end,
                                            size: output_vec.len(),
                                        });
                                    }

                                    for (i, &value) in array_data.data.iter().enumerate() {
                                        if start + i <= end && start + i < output_vec.len() {
                                            output_vec[start + i] = value;
                                        }
                                    }
                                }
                            }
                        }
                        _ => {
                            return Err(PhiloteError::InvalidVariableType(format!(
                                "Unexpected variable type in input stream: {:?}",
                                array_data.var_type
                            )));
                        }
                    }
                }
                Some(Payload::Discrete(discrete_var)) => {
                    if let Some(value) = discrete_var.value {
                        discrete_inputs.insert(discrete_var.name, value);
                    }
                }
                None => {}
            }
        }

        Ok(())
    }
}

#[tonic::async_trait]
impl<D: Discipline + 'static> DisciplineService for DisciplineServer<D> {
    async fn get_info(
        &self,
        _request: Request<()>,
    ) -> std::result::Result<Response<DisciplineProperties>, Status> {
        self.log_if_verbose("GetInfo called").await;

        let discipline = self.discipline.read().await;
        let properties = discipline.get_properties();

        Ok(Response::new(properties))
    }

    async fn set_stream_options(
        &self,
        request: Request<ProtoStreamOptions>,
    ) -> std::result::Result<Response<()>, Status> {
        self.log_if_verbose("SetStreamOptions called").await;

        let proto_options = request.into_inner();
        let stream_options = StreamOptions::from(proto_options);

        let mut options = self.stream_options.write().await;
        *options = stream_options;

        Ok(Response::new(()))
    }

    async fn get_available_options(
        &self,
        _request: Request<()>,
    ) -> std::result::Result<Response<OptionsList>, Status> {
        self.log_if_verbose("GetAvailableOptions called").await;

        let discipline = self.discipline.read().await;
        let options_map = discipline
            .get_available_options()
            .map_err(|e| Status::internal(format!("Failed to get options: {}", e)))?;

        let mut options = Vec::new();
        let mut types = Vec::new();

        for (name, type_str) in options_map {
            options.push(name);

            let data_type = match type_str.as_str() {
                "bool" => DataType::KBool,
                "int" => DataType::KInt,
                "float" | "double" => DataType::KDouble,
                "str" | "string" => DataType::KString,
                "struct" => DataType::KStruct,
                _ => {
                    return Err(Status::invalid_argument(format!(
                        "Invalid option type: {}",
                        type_str
                    )))
                }
            };

            types.push(data_type.into());
        }

        let options_list = OptionsList {
            options,
            r#type: types,
        };

        Ok(Response::new(options_list))
    }

    async fn set_options(
        &self,
        request: Request<DisciplineOptions>,
    ) -> std::result::Result<Response<()>, Status> {
        self.log_if_verbose("SetOptions called").await;

        let options_proto = request.into_inner();

        let options = if let Some(struct_val) = options_proto.options {
            convert_proto_struct_to_json(&struct_val)
        } else {
            HashMap::new()
        };

        let mut discipline = self.discipline.write().await;
        discipline
            .set_options(&options)
            .map_err(|e| Status::internal(format!("Failed to set options: {}", e)))?;

        Ok(Response::new(()))
    }

    async fn setup(&self, _request: Request<()>) -> std::result::Result<Response<()>, Status> {
        self.log_if_verbose("Setup called").await;

        let mut discipline = self.discipline.write().await;

        discipline
            .configure()
            .map_err(|e| Status::internal(format!("Configure failed: {}", e)))?;

        discipline
            .setup()
            .map_err(|e| Status::internal(format!("Setup failed: {}", e)))?;

        discipline
            .setup_partials()
            .map_err(|e| Status::internal(format!("Setup partials failed: {}", e)))?;

        Ok(Response::new(()))
    }

    type GetVariableDefinitionsStream = std::pin::Pin<
        Box<dyn tokio_stream::Stream<Item = std::result::Result<VariableMetaData, Status>> + Send>,
    >;

    async fn get_variable_definitions(
        &self,
        _request: Request<()>,
    ) -> std::result::Result<Response<Self::GetVariableDefinitionsStream>, Status> {
        self.log_if_verbose("GetVariableDefinitions called").await;

        let discipline = self.discipline.read().await;
        let mut var_definitions = discipline
            .get_variable_definitions()
            .map_err(|e| Status::internal(format!("Failed to get variable definitions: {}", e)))?;

        let discrete_definitions = discipline
            .get_discrete_variable_definitions()
            .map_err(|e| {
                Status::internal(format!(
                    "Failed to get discrete variable definitions: {}",
                    e
                ))
            })?;

        var_definitions.extend(discrete_definitions);

        let stream = tokio_stream::iter(var_definitions.into_iter().map(Ok).collect::<Vec<_>>());

        Ok(Response::new(Box::pin(stream)))
    }

    type GetPartialDefinitionsStream = std::pin::Pin<
        Box<dyn tokio_stream::Stream<Item = std::result::Result<PartialsMetaData, Status>> + Send>,
    >;

    async fn get_partial_definitions(
        &self,
        _request: Request<()>,
    ) -> std::result::Result<Response<Self::GetPartialDefinitionsStream>, Status> {
        self.log_if_verbose("GetPartialDefinitions called").await;

        let discipline = self.discipline.read().await;
        let partial_definitions = discipline
            .get_partials_definitions()
            .map_err(|e| Status::internal(format!("Failed to get partial definitions: {}", e)))?;

        let partials_meta: Vec<PartialsMetaData> = partial_definitions
            .into_iter()
            .map(|(name, subname)| PartialsMetaData {
                name,
                subname,
                shape: vec![],
            })
            .collect();

        let stream = tokio_stream::iter(partials_meta.into_iter().map(Ok).collect::<Vec<_>>());

        Ok(Response::new(Box::pin(stream)))
    }

    async fn set_variable_shapes(
        &self,
        request: Request<Streaming<VariableMetaData>>,
    ) -> std::result::Result<Response<()>, Status> {
        self.log_if_verbose("SetVariableShapes called").await;

        let mut stream = request.into_inner();
        while let Some(_var_meta) = stream.next().await {
            // Dynamic shape resolution: update stored variable metadata
            // For now, acknowledge the shapes without modifying internal state
            // since the discipline handles its own variable definitions
        }

        Ok(Response::new(()))
    }
}

fn convert_proto_struct_to_json(
    s: &prost_types::Struct,
) -> HashMap<String, serde_json::Value> {
    let mut map = HashMap::new();
    for (key, value) in &s.fields {
        map.insert(key.clone(), convert_proto_value_to_json(value));
    }
    map
}

fn convert_proto_value_to_json(v: &prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;
    match &v.kind {
        Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::NumberValue(n)) => serde_json::json!(*n),
        Some(Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(Kind::StructValue(s)) => {
            let map: serde_json::Map<String, serde_json::Value> = s
                .fields
                .iter()
                .map(|(k, v)| (k.clone(), convert_proto_value_to_json(v)))
                .collect();
            serde_json::Value::Object(map)
        }
        Some(Kind::ListValue(list)) => {
            let arr: Vec<serde_json::Value> =
                list.values.iter().map(convert_proto_value_to_json).collect();
            serde_json::Value::Array(arr)
        }
        None => serde_json::Value::Null,
    }
}
