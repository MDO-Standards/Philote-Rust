use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status, Streaming};

use crate::discrete::struct_to_json_map;
use crate::philote_info::{
    discipline_service_server::DisciplineService, DataType, DisciplineOptions,
    DisciplineProperties, OptionsList, PartialsMetaData, StreamOptions as ProtoStreamOptions,
    VariableMessage, VariableMetaData, VariableType,
};
use crate::traits::Discipline;
use crate::types::StreamOptions;
use crate::utils::{calculate_partial_shape, preallocate_arrays};
use crate::{wire, ArrayMap, DiscreteMap, PhiloteError, Result};

/// Serves the `DisciplineService` RPCs shared by explicit and implicit disciplines.
pub struct DisciplineServer<D: Discipline + 'static> {
    discipline: Arc<RwLock<D>>,
    stream_options: Arc<RwLock<StreamOptions>>,
    verbose: bool,
}

impl<D: Discipline + 'static> DisciplineServer<D> {
    /// Wrap a discipline for serving.
    ///
    /// Runs [`initialize`](Discipline::initialize) so options declared there are
    /// advertised even when the discipline was built with `#[derive(Default)]`.
    /// `Setup` deliberately preserves the option list, so this happens once.
    pub fn new(mut discipline: D) -> Self {
        if let Err(err) = discipline.initialize() {
            tracing::warn!("discipline initialize() failed: {err}");
        }
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

    pub(crate) async fn log_if_verbose(&self, message: &str) {
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

    /// The configured maximum number of doubles per chunk.
    pub(crate) async fn chunk_size(&self) -> usize {
        self.stream_options.read().await.max_double_per_slice
    }

    /// Whether the discipline declared any discrete variables.
    pub(crate) async fn has_discrete(&self) -> bool {
        !self
            .discipline
            .read()
            .await
            .registry()
            .discrete_meta()
            .is_empty()
    }

    /// Discrete inputs seeded with their declared defaults.
    pub(crate) async fn seeded_discrete_inputs(&self) -> DiscreteMap {
        self.discipline
            .read()
            .await
            .registry()
            .discrete_input_defaults()
    }

    fn preallocate(&self, discipline: &D, var_type: VariableType) -> Result<ArrayMap> {
        discipline.registry().assert_shapes_resolved()?;
        preallocate_arrays(discipline.registry().var_meta(), Some(var_type))
    }

    /// Allocate zeroed arrays for every declared input.
    pub async fn preallocate_inputs(&self) -> Result<ArrayMap> {
        let discipline = self.discipline.read().await;
        self.preallocate(&discipline, VariableType::KInput)
    }

    /// Allocate zeroed arrays for every declared output.
    pub async fn preallocate_outputs(&self) -> Result<ArrayMap> {
        let discipline = self.discipline.read().await;
        self.preallocate(&discipline, VariableType::KOutput)
    }

    /// Allocate zeroed arrays for every declared partial.
    pub async fn preallocate_partials(&self) -> Result<crate::PartialMap> {
        let discipline = self.discipline.read().await;
        crate::utils::preallocate_partials(
            discipline.registry().var_meta(),
            discipline.registry().partials_meta(),
        )
    }

    /// Read a request stream into preallocated input (and optionally output) maps.
    pub async fn process_variable_message_stream(
        &self,
        mut request_stream: Streaming<VariableMessage>,
        inputs: &mut ArrayMap,
        outputs: Option<&mut ArrayMap>,
        discrete_inputs: &mut DiscreteMap,
    ) -> Result<()> {
        wire::receive_request_stream(&mut request_stream, inputs, outputs, discrete_inputs).await
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
        Ok(Response::new(discipline.get_properties()))
    }

    async fn set_stream_options(
        &self,
        request: Request<ProtoStreamOptions>,
    ) -> std::result::Result<Response<()>, Status> {
        self.log_if_verbose("SetStreamOptions called").await;

        let proto_options = request.into_inner();
        if proto_options.num_double <= 0 {
            return Err(Status::invalid_argument(format!(
                "SetStreamOptions: num_double must be positive, got {}",
                proto_options.num_double
            )));
        }

        let mut options = self.stream_options.write().await;
        *options = StreamOptions::from(proto_options);

        Ok(Response::new(()))
    }

    async fn get_available_options(
        &self,
        _request: Request<()>,
    ) -> std::result::Result<Response<OptionsList>, Status> {
        self.log_if_verbose("GetAvailableOptions called").await;

        let discipline = self.discipline.read().await;
        let options_map = discipline.get_available_options().map_err(Status::from)?;

        let mut options = Vec::with_capacity(options_map.len());
        let mut types = Vec::with_capacity(options_map.len());

        for (name, type_str) in options_map {
            let data_type = match type_str.as_str() {
                "bool" => DataType::KBool,
                "int" => DataType::KInt,
                // "double" is accepted as an alias for the canonical "float".
                "float" | "double" => DataType::KDouble,
                // "string" is accepted as an alias for the canonical "str".
                "str" | "string" => DataType::KString,
                // "struct" is accepted as an alias for the canonical "dict".
                "dict" | "struct" => DataType::KStruct,
                other => {
                    return Err(Status::invalid_argument(format!(
                        "option '{name}' has invalid type '{other}'"
                    )))
                }
            };
            options.push(name);
            types.push(data_type.into());
        }

        Ok(Response::new(OptionsList {
            options,
            r#type: types,
        }))
    }

    async fn set_options(
        &self,
        request: Request<DisciplineOptions>,
    ) -> std::result::Result<Response<()>, Status> {
        self.log_if_verbose("SetOptions called").await;

        let options = request
            .into_inner()
            .options
            .as_ref()
            .map(struct_to_json_map)
            .unwrap_or_default();

        let mut discipline = self.discipline.write().await;
        discipline.set_options(&options).map_err(Status::from)?;

        Ok(Response::new(()))
    }

    async fn setup(&self, _request: Request<()>) -> std::result::Result<Response<()>, Status> {
        self.log_if_verbose("Setup called").await;

        let mut discipline = self.discipline.write().await;

        // Drop metadata from any previous Setup so repeated calls do not accumulate
        // duplicate definitions.
        discipline.registry_mut().clear();

        discipline.configure().map_err(Status::from)?;
        discipline.setup().map_err(Status::from)?;
        discipline.setup_partials().map_err(Status::from)?;

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
        let mut definitions = discipline
            .get_variable_definitions()
            .map_err(Status::from)?;
        definitions.extend(
            discipline
                .get_discrete_variable_definitions()
                .map_err(Status::from)?,
        );

        let stream = tokio_stream::iter(definitions.into_iter().map(Ok).collect::<Vec<_>>());
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
        let var_meta = discipline.registry().var_meta();

        let shape_of = |name: &str| -> Option<Vec<usize>> {
            var_meta
                .iter()
                .find(|v| v.name == name)
                .map(|v| v.shape.iter().map(|&d| d as usize).collect())
        };

        let partials_meta: Vec<PartialsMetaData> = discipline
            .get_partials_definitions()
            .map_err(Status::from)?
            .into_iter()
            .map(|(name, subname)| {
                // Report the derived shape when both variables are known. Philote-Python
                // always leaves this empty and lets the client derive it; sending it is
                // additive and harmless to clients that ignore it.
                let shape = match (shape_of(&name), shape_of(&subname)) {
                    (Some(func), Some(var)) if !func.is_empty() && !var.is_empty() => {
                        calculate_partial_shape(&func, &var)
                            .into_iter()
                            .map(|d| d as i64)
                            .collect()
                    }
                    _ => Vec::new(),
                };
                PartialsMetaData {
                    name,
                    subname,
                    shape,
                }
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
        let mut discipline = self.discipline.write().await;

        while let Some(meta) = stream.message().await? {
            let var_type = VariableType::try_from(meta.r#type).map_err(|_| {
                PhiloteError::InvalidVariableType(format!(
                    "SetVariableShapes: invalid type {} for '{}'",
                    meta.r#type, meta.name
                ))
            })?;
            let shape: Vec<usize> = meta.shape.iter().map(|&d| d as usize).collect();

            discipline
                .set_variable_shape(&meta.name, var_type, &shape)
                .map_err(Status::from)?;
        }

        Ok(Response::new(()))
    }
}
