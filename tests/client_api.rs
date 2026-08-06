//! Client-side API surface: the implicit client and the shared discipline client.
//!
//! The other integration suites drive the explicit client; these tests exercise the
//! implicit client's delegating methods and discrete paths, plus the connection,
//! configuration, and caching behaviour of the [`DisciplineClient`] both clients
//! are built on.

mod common;

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use philote_mdo::client::{variable_shape_meta, DisciplineClient, ImplicitClient};
use philote_mdo::discrete::{json_to_value, value_to_json};
use philote_mdo::examples::{scalar, vector, QuadraticImplicit};
use philote_mdo::philote_info::{
    discipline_service_server::{DisciplineService, DisciplineServiceServer},
    implicit_service_client::ImplicitServiceClient,
    DisciplineOptions, DisciplineProperties, OptionsList, PartialsMetaData,
    StreamOptions as ProtoStreamOptions, VariableMetaData, VariableType,
};
use philote_mdo::registry::VariableRegistry;
use philote_mdo::traits::{Discipline, ImplicitDiscipline};
use philote_mdo::types::StreamOptions;
use philote_mdo::{impl_registry, ArrayMap, DiscreteMap, PartialMap, PhiloteError, Result};
use tonic::{Code, Request, Response, Status, Streaming};

fn status_of(err: PhiloteError) -> Status {
    match err {
        PhiloteError::GrpcError(status) => *status,
        other => panic!("expected a gRPC error, got {other:?}"),
    }
}

fn quadratic_inputs(a: f64, b: f64, c: f64) -> ArrayMap {
    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), scalar(a));
    inputs.insert("b".to_string(), scalar(b));
    inputs.insert("c".to_string(), scalar(c));
    inputs
}

// --- Test disciplines ---

/// Implicit discipline whose variable size comes from an option, so `SetOptions`
/// followed by `Setup` changes the declared shapes. `R(x) = x - 2·a`.
struct ConfigurableImplicit {
    registry: VariableRegistry,
    size: usize,
}

impl Default for ConfigurableImplicit {
    fn default() -> Self {
        Self {
            registry: VariableRegistry::new(true),
            size: 1,
        }
    }
}

impl Discipline for ConfigurableImplicit {
    impl_registry!(registry);

    fn name(&self) -> &str {
        "ConfigurableImplicit"
    }

    fn initialize(&mut self) -> Result<()> {
        self.add_option("size", "int")?;
        self.add_option("label", "str")?;
        self.add_option("gain", "float")?;
        self.add_option("verbose", "bool")?;
        self.add_option("extra", "dict")
    }

    fn set_options(&mut self, options: &HashMap<String, serde_json::Value>) -> Result<()> {
        if let Some(value) = options.get("size") {
            let size = value.as_i64().ok_or_else(|| {
                PhiloteError::validation(
                    "set_options",
                    format!("'size' must be an integer: {value}"),
                )
            })?;
            self.size = size as usize;
        }
        Ok(())
    }

    fn setup(&mut self) -> Result<()> {
        let size = self.size;
        self.add_input("a", &[size], "")?;
        self.add_output("x", &[size], "")
    }
}

#[async_trait]
impl ImplicitDiscipline for ConfigurableImplicit {
    async fn compute_residuals(&self, inputs: &ArrayMap, outputs: &ArrayMap) -> Result<ArrayMap> {
        let mut residuals = HashMap::new();
        residuals.insert("x".to_string(), &outputs["x"] - &(&inputs["a"] * 2.0));
        Ok(residuals)
    }

    async fn solve_residuals(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let mut outputs = HashMap::new();
        outputs.insert("x".to_string(), &inputs["a"] * 2.0);
        Ok(outputs)
    }
}

/// Implicit discipline that scales its residual by a discrete input and echoes the
/// scale back as a discrete output. `R(x) = scale·(x - a)`.
#[derive(Default)]
struct ScaledImplicit {
    registry: VariableRegistry,
}

impl ScaledImplicit {
    fn scale(discrete_inputs: &DiscreteMap) -> f64 {
        discrete_inputs
            .get("scale")
            .map(|v| value_to_json(v).as_f64().unwrap_or(1.0))
            .unwrap_or(1.0)
    }

    fn echo(scale: f64) -> DiscreteMap {
        let mut discrete_outputs = HashMap::new();
        discrete_outputs.insert(
            "scale_used".to_string(),
            json_to_value(&serde_json::json!(scale)),
        );
        discrete_outputs
    }
}

impl Discipline for ScaledImplicit {
    impl_registry!(registry);

    fn name(&self) -> &str {
        "ScaledImplicit"
    }

    fn setup(&mut self) -> Result<()> {
        self.add_input("a", &[1], "")?;
        self.add_output("x", &[1], "")?;
        // Default of 2 applies when the client sends nothing.
        self.add_discrete_input("scale", Some(json_to_value(&serde_json::json!(2))))?;
        self.add_discrete_output("scale_used", None)
    }

    fn setup_partials(&mut self) -> Result<()> {
        self.declare_partials("x", "a")?;
        self.declare_partials("x", "x")
    }
}

#[async_trait]
impl ImplicitDiscipline for ScaledImplicit {
    async fn compute_residuals(&self, _inputs: &ArrayMap, _outputs: &ArrayMap) -> Result<ArrayMap> {
        unreachable!("the server uses the discrete path when discrete variables exist")
    }

    async fn solve_residuals(&self, _inputs: &ArrayMap) -> Result<ArrayMap> {
        unreachable!("the server uses the discrete path when discrete variables exist")
    }

    async fn compute_residuals_with_discrete(
        &self,
        inputs: &ArrayMap,
        outputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        let scale = Self::scale(discrete_inputs);

        let mut residuals = HashMap::new();
        residuals.insert("x".to_string(), (&outputs["x"] - &inputs["a"]) * scale);
        Ok((residuals, Self::echo(scale)))
    }

    async fn solve_residuals_with_discrete(
        &self,
        inputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        let scale = Self::scale(discrete_inputs);

        // The residual is zero at x = a regardless of scale; the scale is folded
        // into the answer so the test can see which value was used.
        let mut outputs = HashMap::new();
        outputs.insert("x".to_string(), &inputs["a"] * scale);
        Ok((outputs, Self::echo(scale)))
    }

    async fn residual_partials_with_discrete(
        &self,
        _inputs: &ArrayMap,
        _outputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<PartialMap> {
        let scale = Self::scale(discrete_inputs);

        let mut partials: PartialMap = HashMap::new();
        partials.insert(("x".to_string(), "a".to_string()), scalar(-scale));
        partials.insert(("x".to_string(), "x".to_string()), scalar(scale));
        Ok(partials)
    }
}

/// Implicit discipline with client-set shapes. `R(x) = x - 2·a`, element-wise.
#[derive(Default)]
struct DynamicImplicit {
    registry: VariableRegistry,
}

impl Discipline for DynamicImplicit {
    impl_registry!(registry);

    fn setup(&mut self) -> Result<()> {
        self.add_dynamic_input("a", "m")?;
        self.add_dynamic_output("x", "m")
    }
}

#[async_trait]
impl ImplicitDiscipline for DynamicImplicit {
    async fn compute_residuals(&self, inputs: &ArrayMap, outputs: &ArrayMap) -> Result<ArrayMap> {
        let mut residuals = HashMap::new();
        residuals.insert("x".to_string(), &outputs["x"] - &(&inputs["a"] * 2.0));
        Ok(residuals)
    }

    async fn solve_residuals(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let mut outputs = HashMap::new();
        outputs.insert("x".to_string(), &inputs["a"] * 2.0);
        Ok(outputs)
    }
}

/// Implicit discipline that stalls, so a client deadline is the only thing that
/// ends the call.
#[derive(Default)]
struct StalledImplicit {
    registry: VariableRegistry,
}

impl Discipline for StalledImplicit {
    impl_registry!(registry);

    fn setup(&mut self) -> Result<()> {
        self.add_input("a", &[1], "")?;
        self.add_output("x", &[1], "")
    }
}

#[async_trait]
impl ImplicitDiscipline for StalledImplicit {
    async fn compute_residuals(&self, _inputs: &ArrayMap, outputs: &ArrayMap) -> Result<ArrayMap> {
        let mut residuals = HashMap::new();
        residuals.insert("x".to_string(), outputs["x"].clone());
        Ok(residuals)
    }

    async fn solve_residuals(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        tokio::time::sleep(Duration::from_secs(2)).await;
        let mut outputs = HashMap::new();
        outputs.insert("x".to_string(), inputs["a"].clone());
        Ok(outputs)
    }
}

// --- Connecting ---

#[tokio::test]
async fn a_malformed_endpoint_is_rejected_by_the_implicit_client() {
    match ImplicitClient::connect("definitely not a url").await {
        Err(PhiloteError::ConfigurationError(_)) => {}
        Err(other) => panic!("expected a configuration error, got {other:?}"),
        Ok(_) => panic!("a malformed endpoint should not connect"),
    }
}

#[tokio::test]
async fn a_closed_port_is_reported_as_a_configuration_error() {
    // Port 1 on loopback has nothing listening.
    match ImplicitClient::connect("http://127.0.0.1:1").await {
        Err(PhiloteError::ConfigurationError(_)) => {}
        Err(other) => panic!("expected a configuration error, got {other:?}"),
        Ok(_) => panic!("connecting to a closed port should fail"),
    }
}

#[tokio::test]
async fn the_shared_discipline_client_is_usable_on_its_own() {
    // DisciplineClient is public API: a caller that only needs metadata should not
    // have to pick an explicit or implicit client.
    let server = common::spawn_implicit(QuadraticImplicit::new()).await;
    let mut client = DisciplineClient::connect(server.endpoint())
        .await
        .expect("connect the shared client");

    client.setup().await.unwrap();

    let properties = client.get_info().await.unwrap();
    assert_eq!(properties.name, "QuadraticImplicit");

    let vars = client.get_variable_definitions().await.unwrap();
    // a, b, c, x, plus the residual twin of x.
    assert_eq!(vars.len(), 5);
    assert_eq!(client.get_partial_definitions().await.unwrap().len(), 4);
    assert!(
        client.dynamic_variables().is_empty(),
        "QuadraticImplicit declares fixed shapes"
    );

    server.shutdown().await;
}

// --- Delegated discipline RPCs on the implicit client ---

#[tokio::test]
async fn the_implicit_client_reports_the_disciplines_properties() {
    let server = common::spawn_implicit(QuadraticImplicit::new()).await;
    let mut client = common::connected_implicit_client(&server).await;

    let properties = client.get_info().await.unwrap();
    assert_eq!(properties.name, "QuadraticImplicit");
    assert!(properties.differentiable);
    assert!(properties.provides_gradients);
    assert!(properties.continuous);
    assert!(!properties.version.is_empty());

    server.shutdown().await;
}

#[tokio::test]
async fn the_implicit_client_lists_every_declared_option_type() {
    let server = common::spawn_implicit(ConfigurableImplicit::default()).await;
    let mut client = ImplicitClient::connect(server.endpoint()).await.unwrap();

    let options = client.get_available_options().await.unwrap();
    assert_eq!(options.len(), 5);
    assert_eq!(options["size"], "int");
    assert_eq!(options["label"], "str");
    assert_eq!(options["gain"], "float");
    assert_eq!(options["verbose"], "bool");
    assert_eq!(options["extra"], "dict");

    server.shutdown().await;
}

#[tokio::test]
async fn options_set_through_the_implicit_client_resize_the_variables() {
    let server = common::spawn_implicit(ConfigurableImplicit::default()).await;
    let mut client = ImplicitClient::connect(server.endpoint()).await.unwrap();

    let mut options = HashMap::new();
    options.insert("size".to_string(), serde_json::json!(4));
    client.set_options(options).await.unwrap();
    client.setup().await.unwrap();

    let vars = client.get_variable_definitions().await.unwrap();
    let a = vars.iter().find(|v| v.name == "a").unwrap();
    assert_eq!(a.shape, vec![4], "options must apply before setup");
    client.get_partial_definitions().await.unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), vector(&[1.0, 2.0, 3.0, 4.0]));

    let outputs = client.solve_residuals(&inputs).await.unwrap();
    assert_eq!(outputs["x"].shape(), &[4]);
    assert_eq!(outputs["x"][[3]], 8.0);

    server.shutdown().await;
}

#[tokio::test]
async fn an_invalid_option_value_surfaces_through_the_implicit_client() {
    let server = common::spawn_implicit(ConfigurableImplicit::default()).await;
    let mut client = ImplicitClient::connect(server.endpoint()).await.unwrap();

    let mut options = HashMap::new();
    options.insert("size".to_string(), serde_json::json!("four"));

    let err = client.set_options(options).await.unwrap_err();
    assert_eq!(status_of(err).code(), Code::InvalidArgument);

    server.shutdown().await;
}

#[tokio::test]
async fn re_running_setup_invalidates_the_implicit_clients_metadata_cache() {
    // Setup rebuilds the server's metadata, so a client that kept its cache would
    // decode residuals against shapes that no longer exist.
    let server = common::spawn_implicit(ConfigurableImplicit::default()).await;
    let mut client = common::connected_implicit_client(&server).await;

    let mut options = HashMap::new();
    options.insert("size".to_string(), serde_json::json!(3));
    client.set_options(options).await.unwrap();
    client.setup().await.unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), vector(&[1.0, 2.0, 3.0]));

    let err = client.solve_residuals(&inputs).await.unwrap_err();
    assert!(
        matches!(err, PhiloteError::SetupNotCalled),
        "expected a stale-cache rejection, got: {err}"
    );

    // Re-fetching the definitions makes the client usable at the new size.
    let vars = client.get_variable_definitions().await.unwrap();
    assert_eq!(vars.iter().find(|v| v.name == "a").unwrap().shape, vec![3]);
    let outputs = client.solve_residuals(&inputs).await.unwrap();
    assert_eq!(outputs["x"].shape(), &[3]);
    assert_eq!(outputs["x"][[2]], 6.0);

    server.shutdown().await;
}

#[tokio::test]
async fn every_implicit_rpc_is_rejected_before_metadata_is_fetched() {
    // Responses are decoded against the declared shapes, so computing without them
    // cannot produce a correct result.
    let server = common::spawn_implicit(QuadraticImplicit::new()).await;
    let mut client = ImplicitClient::connect(server.endpoint()).await.unwrap();
    client.setup().await.unwrap();

    let inputs = quadratic_inputs(1.0, -3.0, 2.0);
    let mut outputs = HashMap::new();
    outputs.insert("x".to_string(), scalar(2.0));

    let solve_err = client.solve_residuals(&inputs).await.unwrap_err();
    assert!(matches!(solve_err, PhiloteError::SetupNotCalled));

    let residual_err = client
        .compute_residuals(&inputs, &outputs)
        .await
        .unwrap_err();
    assert!(matches!(residual_err, PhiloteError::SetupNotCalled));

    let gradient_err = client
        .compute_residual_gradients(&inputs, &outputs)
        .await
        .unwrap_err();
    assert!(matches!(gradient_err, PhiloteError::SetupNotCalled));

    server.shutdown().await;
}

#[tokio::test]
async fn residual_gradients_without_partials_metadata_name_the_missing_call() {
    // `get_partial_definitions` is easy to forget, and without it the Jacobian has
    // nothing to scatter into.
    let server = common::spawn_implicit(QuadraticImplicit::new()).await;
    let mut client = ImplicitClient::connect(server.endpoint()).await.unwrap();
    client.setup().await.unwrap();
    client.get_variable_definitions().await.unwrap();
    // deliberately skip get_partial_definitions

    let mut outputs = HashMap::new();
    outputs.insert("x".to_string(), scalar(2.0));

    let err = client
        .compute_residual_gradients(&quadratic_inputs(1.0, -3.0, 2.0), &outputs)
        .await
        .unwrap_err();
    assert!(matches!(err, PhiloteError::Validation { .. }));
    assert!(
        err.to_string().contains("get_partial_definitions"),
        "error should name the missing call, got: {err}"
    );

    server.shutdown().await;
}

// --- Streaming options and deadlines ---

/// Count the response messages the server sends for a `SolveResiduals` call,
/// bypassing the client's reassembly.
async fn count_solve_response_messages(endpoint: &str, inputs: &ArrayMap) -> usize {
    let mut raw = ImplicitServiceClient::connect(endpoint.to_string())
        .await
        .expect("connect the generated client");
    let messages = philote_mdo::wire::assemble_input_messages(inputs, None, &HashMap::new(), 1000);

    let response = raw
        .solve_residuals(tokio_stream::iter(messages))
        .await
        .expect("solve residuals");
    let mut stream = response.into_inner();

    let mut count = 0;
    while stream
        .message()
        .await
        .expect("read a response message")
        .is_some()
    {
        count += 1;
    }
    count
}

#[tokio::test]
async fn set_stream_options_changes_the_chunking_seen_on_the_wire() {
    let server = common::spawn_implicit(ConfigurableImplicit::default()).await;
    let mut client = ImplicitClient::connect(server.endpoint()).await.unwrap();

    let mut options = HashMap::new();
    options.insert("size".to_string(), serde_json::json!(8));
    client.set_options(options).await.unwrap();
    client.setup().await.unwrap();
    client.get_variable_definitions().await.unwrap();
    client.get_partial_definitions().await.unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), vector(&[1.0; 8]));

    // The default chunk size of 1000 fits the whole 8-element output in one message.
    assert_eq!(
        count_solve_response_messages(&server.endpoint(), &inputs).await,
        1
    );

    client
        .set_stream_options(StreamOptions {
            max_double_per_slice: 3,
        })
        .await
        .unwrap();

    // The server now splits the same array into ceil(8 / 3) = 3 messages.
    assert_eq!(
        count_solve_response_messages(&server.endpoint(), &inputs).await,
        3
    );
    assert_eq!(client.base().stream_options().max_double_per_slice, 3);

    // The client reassembles the chunked response transparently.
    let outputs = client.solve_residuals(&inputs).await.unwrap();
    assert_eq!(outputs["x"].shape(), &[8]);
    assert!(outputs["x"].iter().all(|v| *v == 2.0));

    server.shutdown().await;
}

#[tokio::test]
async fn a_rejected_stream_option_leaves_the_local_chunk_size_unchanged() {
    // The local copy is updated only after the server accepts, so both peers keep
    // chunking with the same limit.
    let server = common::spawn_implicit(QuadraticImplicit::new()).await;
    let mut client = ImplicitClient::connect(server.endpoint()).await.unwrap();

    let before = client.base().stream_options().max_double_per_slice;

    let err = client
        .set_stream_options(StreamOptions {
            max_double_per_slice: 0,
        })
        .await
        .unwrap_err();
    assert_eq!(status_of(err).code(), Code::InvalidArgument);
    assert_eq!(
        client.base().stream_options().max_double_per_slice,
        before,
        "a rejected chunk size must not be adopted locally"
    );

    server.shutdown().await;
}

#[tokio::test]
async fn with_stream_options_splits_outgoing_arrays() {
    // Requests are chunked with the local option, so a small limit exercises the
    // multi-chunk send path end to end.
    let server = common::spawn_implicit(ConfigurableImplicit::default()).await;
    let mut client = ImplicitClient::connect(server.endpoint())
        .await
        .unwrap()
        .with_stream_options(StreamOptions {
            max_double_per_slice: 3,
        });
    assert_eq!(client.base().stream_options().max_double_per_slice, 3);

    let mut options = HashMap::new();
    options.insert("size".to_string(), serde_json::json!(8));
    client.set_options(options).await.unwrap();
    client.setup().await.unwrap();
    client.get_variable_definitions().await.unwrap();
    client.get_partial_definitions().await.unwrap();

    let values: Vec<f64> = (0..8).map(|i| i as f64 + 1.0).collect();
    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), vector(&values));

    let outputs = client.solve_residuals(&inputs).await.unwrap();
    for (i, value) in values.iter().enumerate() {
        assert_eq!(outputs["x"][[i]], value * 2.0);
    }

    // Residuals travel with the outputs, so this sends two chunked arrays at once.
    let residuals = client.compute_residuals(&inputs, &outputs).await.unwrap();
    assert!(
        residuals["x"].iter().all(|v| *v == 0.0),
        "the solved outputs should zero the residuals: {:?}",
        residuals["x"]
    );

    server.shutdown().await;
}

#[tokio::test]
async fn an_rpc_timeout_ends_a_stalled_call() {
    let server = common::spawn_implicit(StalledImplicit::default()).await;
    let mut client = ImplicitClient::connect(server.endpoint())
        .await
        .unwrap()
        .with_rpc_timeout(Duration::from_millis(100));
    client.setup().await.unwrap();
    client.get_variable_definitions().await.unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), scalar(1.0));

    let started = std::time::Instant::now();
    let err = client.solve_residuals(&inputs).await.unwrap_err();
    let code = status_of(err).code();
    assert!(
        matches!(code, Code::DeadlineExceeded | Code::Cancelled),
        "expected the deadline to end the call, got {code:?}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "the call should have been cut short, not run to completion"
    );

    // Without a deadline the same call completes, so the failure above came from
    // the timeout rather than the discipline itself.
    let mut patient = common::connected_implicit_client(&server).await;
    let outputs = patient.solve_residuals(&inputs).await.unwrap();
    assert_eq!(outputs["x"][[0]], 1.0);

    server.shutdown().await;
}

// --- Caching and cloning ---

#[tokio::test]
async fn cloning_an_implicit_client_keeps_the_fetched_metadata() {
    // A clone shares the channel and the cache, so it can compute immediately;
    // re-running the handshake on every clone would be wasted round trips.
    let server = common::spawn_implicit(QuadraticImplicit::new()).await;
    let mut client = common::connected_implicit_client(&server)
        .await
        .with_stream_options(StreamOptions {
            max_double_per_slice: 5,
        });

    let mut clone = client.clone();
    assert_eq!(clone.base().stream_options().max_double_per_slice, 5);
    assert_eq!(clone.var_meta().len(), client.var_meta().len());

    let inputs = quadratic_inputs(1.0, -3.0, 2.0);
    let cloned_outputs = clone.solve_residuals(&inputs).await.unwrap();
    assert_eq!(cloned_outputs["x"][[0]], 2.0);

    // The original is unaffected by the clone's use.
    let outputs = client.solve_residuals(&inputs).await.unwrap();
    assert_eq!(outputs["x"][[0]], 2.0);

    server.shutdown().await;
}

// --- Dynamic shapes through the implicit client ---

#[tokio::test]
async fn the_implicit_client_lists_and_resolves_dynamic_variables() {
    let server = common::spawn_implicit(DynamicImplicit::default()).await;
    let mut client = ImplicitClient::connect(server.endpoint()).await.unwrap();
    client.setup().await.unwrap();

    // Input, output, and the output's residual twin are all unresolved.
    let dynamic = client.get_dynamic_variables().await.unwrap();
    assert_eq!(dynamic.len(), 3);
    assert!(dynamic.iter().all(|v| v.shape.is_empty()));

    client
        .send_variable_shapes(vec![
            variable_shape_meta("a", &[3], VariableType::KInput),
            variable_shape_meta("x", &[3], VariableType::KOutput),
        ])
        .await
        .unwrap();

    // The cached copies — including the residual, which was never named in the
    // request — carry the resolved shape.
    for var in client.base().dynamic_variables() {
        assert_eq!(var.shape, vec![3], "{} was left unresolved", var.name);
    }

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), vector(&[1.0, 2.0, 3.0]));

    let outputs = client.solve_residuals(&inputs).await.unwrap();
    assert_eq!(outputs["x"].shape(), &[3]);
    assert_eq!(outputs["x"][[2]], 6.0);

    let residuals = client.compute_residuals(&inputs, &outputs).await.unwrap();
    assert_eq!(residuals["x"].shape(), &[3]);
    assert!(residuals["x"].iter().all(|v| *v == 0.0));

    server.shutdown().await;
}

// --- Discrete variables through the implicit client ---

#[tokio::test]
async fn discrete_inputs_reach_the_implicit_residuals() {
    let server = common::spawn_implicit(ScaledImplicit::default()).await;
    let mut client = common::connected_implicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), scalar(1.0));
    let mut outputs = HashMap::new();
    outputs.insert("x".to_string(), scalar(3.0));

    let mut discrete_inputs: DiscreteMap = HashMap::new();
    discrete_inputs.insert("scale".to_string(), json_to_value(&serde_json::json!(5)));

    let (residuals, discrete_outputs) = client
        .compute_residuals_with_discrete(&inputs, &outputs, &discrete_inputs)
        .await
        .unwrap();

    // R = scale * (x - a) = 5 * (3 - 1)
    assert_eq!(residuals["x"][[0]], 10.0);
    assert_eq!(
        value_to_json(&discrete_outputs["scale_used"])
            .as_f64()
            .unwrap(),
        5.0
    );

    // Without a discrete value the declared default of 2 applies.
    let residuals = client.compute_residuals(&inputs, &outputs).await.unwrap();
    assert_eq!(residuals["x"][[0]], 4.0);

    server.shutdown().await;
}

#[tokio::test]
async fn solving_round_trips_discrete_values() {
    let server = common::spawn_implicit(ScaledImplicit::default()).await;
    let mut client = common::connected_implicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), scalar(4.0));

    let mut discrete_inputs: DiscreteMap = HashMap::new();
    discrete_inputs.insert("scale".to_string(), json_to_value(&serde_json::json!(3)));

    let (outputs, discrete_outputs) = client
        .solve_residuals_with_discrete(&inputs, &discrete_inputs)
        .await
        .unwrap();

    assert_eq!(outputs["x"][[0]], 12.0);
    assert_eq!(
        value_to_json(&discrete_outputs["scale_used"])
            .as_f64()
            .unwrap(),
        3.0
    );

    server.shutdown().await;
}

#[tokio::test]
async fn discrete_inputs_flow_through_residual_gradients() {
    let server = common::spawn_implicit(ScaledImplicit::default()).await;
    let mut client = common::connected_implicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), scalar(1.0));
    let mut outputs = HashMap::new();
    outputs.insert("x".to_string(), scalar(1.0));

    let mut discrete_inputs: DiscreteMap = HashMap::new();
    discrete_inputs.insert("scale".to_string(), json_to_value(&serde_json::json!(7)));

    let partials = client
        .compute_residual_gradients_with_discrete(&inputs, &outputs, &discrete_inputs)
        .await
        .unwrap();

    assert_eq!(partials[&("x".to_string(), "x".to_string())][[0]], 7.0);
    assert_eq!(partials[&("x".to_string(), "a".to_string())][[0]], -7.0);

    // The declared default of 2 is used when nothing is sent.
    let partials = client
        .compute_residual_gradients(&inputs, &outputs)
        .await
        .unwrap();
    assert_eq!(partials[&("x".to_string(), "x".to_string())][[0]], 2.0);

    server.shutdown().await;
}

#[tokio::test]
async fn an_undeclared_discrete_input_is_rejected_by_the_implicit_server() {
    // The client does not filter discrete names against the metadata, so a typo
    // reaches the server and must be reported rather than silently ignored.
    let server = common::spawn_implicit(ScaledImplicit::default()).await;
    let mut client = common::connected_implicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), scalar(1.0));

    let mut discrete_inputs: DiscreteMap = HashMap::new();
    discrete_inputs.insert("scal".to_string(), json_to_value(&serde_json::json!(5)));

    let err = client
        .solve_residuals_with_discrete(&inputs, &discrete_inputs)
        .await
        .unwrap_err();
    let status = status_of(err);
    assert_eq!(status.code(), Code::InvalidArgument);
    assert!(
        status.message().contains("scal"),
        "the offending name should be reported: {}",
        status.message()
    );

    server.shutdown().await;
}

// --- Metadata a client cannot interpret ---

/// A server that answers with type codes this client does not know, standing in
/// for a peer built against a newer protocol.
struct BogusMetadataServer;

type MetaStream = std::pin::Pin<
    Box<dyn tokio_stream::Stream<Item = std::result::Result<VariableMetaData, Status>> + Send>,
>;
type PartialsStream = std::pin::Pin<
    Box<dyn tokio_stream::Stream<Item = std::result::Result<PartialsMetaData, Status>> + Send>,
>;

#[tonic::async_trait]
impl DisciplineService for BogusMetadataServer {
    async fn get_info(
        &self,
        _request: Request<()>,
    ) -> std::result::Result<Response<DisciplineProperties>, Status> {
        Ok(Response::new(DisciplineProperties::default()))
    }

    async fn set_stream_options(
        &self,
        _request: Request<ProtoStreamOptions>,
    ) -> std::result::Result<Response<()>, Status> {
        Ok(Response::new(()))
    }

    async fn get_available_options(
        &self,
        _request: Request<()>,
    ) -> std::result::Result<Response<OptionsList>, Status> {
        Ok(Response::new(OptionsList {
            options: vec!["mystery".to_string()],
            // 9 is outside the DataType enum this client knows.
            r#type: vec![9],
        }))
    }

    async fn set_options(
        &self,
        _request: Request<DisciplineOptions>,
    ) -> std::result::Result<Response<()>, Status> {
        Ok(Response::new(()))
    }

    async fn setup(&self, _request: Request<()>) -> std::result::Result<Response<()>, Status> {
        Ok(Response::new(()))
    }

    type GetVariableDefinitionsStream = MetaStream;

    async fn get_variable_definitions(
        &self,
        _request: Request<()>,
    ) -> std::result::Result<Response<Self::GetVariableDefinitionsStream>, Status> {
        let meta = VariableMetaData {
            // 42 is outside the VariableType enum this client knows.
            r#type: 42,
            name: "mystery".to_string(),
            shape: vec![1],
            units: String::new(),
            dynamic_shape: false,
        };
        Ok(Response::new(Box::pin(tokio_stream::iter(vec![Ok(meta)]))))
    }

    type GetPartialDefinitionsStream = PartialsStream;

    async fn get_partial_definitions(
        &self,
        _request: Request<()>,
    ) -> std::result::Result<Response<Self::GetPartialDefinitionsStream>, Status> {
        Ok(Response::new(Box::pin(tokio_stream::iter(vec![]))))
    }

    async fn set_variable_shapes(
        &self,
        _request: Request<Streaming<VariableMetaData>>,
    ) -> std::result::Result<Response<()>, Status> {
        Ok(Response::new(()))
    }
}

/// Serve [`BogusMetadataServer`]; the returned sender keeps it alive.
async fn spawn_bogus_metadata_server() -> (String, tokio::sync::oneshot::Sender<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();

    tokio::spawn(async move {
        let _ = tonic::transport::Server::builder()
            .add_service(DisciplineServiceServer::new(BogusMetadataServer))
            .serve_with_incoming_shutdown(
                tokio_stream::wrappers::TcpListenerStream::new(listener),
                async {
                    let _ = rx.await;
                },
            )
            .await;
    });

    (format!("http://{addr}"), tx)
}

#[tokio::test]
async fn an_unknown_option_type_is_reported_rather_than_mislabelled() {
    let (endpoint, _keep_alive) = spawn_bogus_metadata_server().await;
    let mut client = DisciplineClient::connect(endpoint).await.unwrap();

    let err = client.get_available_options().await.unwrap_err();
    match err {
        PhiloteError::InvalidVariableType(message) => {
            assert!(
                message.contains('9'),
                "the unknown code should be reported: {message}"
            );
        }
        other => panic!("expected an invalid type error, got {other:?}"),
    }
}

#[tokio::test]
async fn an_unknown_variable_type_is_reported_rather_than_cached() {
    let (endpoint, _keep_alive) = spawn_bogus_metadata_server().await;
    let mut client = DisciplineClient::connect(endpoint).await.unwrap();

    let err = client.get_variable_definitions().await.unwrap_err();
    assert!(
        matches!(err, PhiloteError::InvalidVariableType(_)),
        "expected an invalid type error, got {err:?}"
    );
    // Nothing usable was cached, so a later compute still reports missing metadata.
    assert!(client.var_meta().is_empty());
}

// --- Explicit client configuration, discrete gradients, and cloning ---
//
// The other suites drive the explicit client with default settings; these cover the
// builder-style configuration, the discrete gradient path, and cloning.

use philote_mdo::client::ExplicitClient;
use philote_mdo::examples::{Paraboloid, Rosenbrock};
use philote_mdo::traits::ExplicitDiscipline;

/// Explicit discipline that stalls, so a client deadline is the only thing that
/// ends the call.
#[derive(Default)]
struct StalledExplicit {
    registry: VariableRegistry,
}

impl Discipline for StalledExplicit {
    impl_registry!(registry);

    fn setup(&mut self) -> Result<()> {
        self.add_input("x", &[1], "")?;
        self.add_output("y", &[1], "")
    }
}

#[async_trait]
impl ExplicitDiscipline for StalledExplicit {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        tokio::time::sleep(Duration::from_secs(2)).await;
        let mut outputs = HashMap::new();
        outputs.insert("y".to_string(), inputs["x"].clone());
        Ok(outputs)
    }
}

/// Explicit discipline whose value *and* derivatives depend on a discrete input:
/// `y = gain·Σ xᵢ²`, so `dy/dxᵢ = 2·gain·xᵢ`.
#[derive(Default)]
struct GainedSquare {
    registry: VariableRegistry,
}

impl GainedSquare {
    fn gain(discrete_inputs: &DiscreteMap) -> f64 {
        discrete_inputs
            .get("gain")
            .map(|v| value_to_json(v).as_f64().unwrap_or(1.0))
            .unwrap_or(1.0)
    }

    fn echo(gain: f64) -> DiscreteMap {
        let mut discrete_outputs = HashMap::new();
        discrete_outputs.insert(
            "gain_used".to_string(),
            json_to_value(&serde_json::json!(gain)),
        );
        discrete_outputs
    }
}

impl Discipline for GainedSquare {
    impl_registry!(registry);

    fn name(&self) -> &str {
        "GainedSquare"
    }

    fn setup(&mut self) -> Result<()> {
        self.add_input("x", &[3], "")?;
        self.add_output("y", &[1], "")?;
        // Default of 2 applies when the client sends nothing.
        self.add_discrete_input("gain", Some(json_to_value(&serde_json::json!(2))))?;
        self.add_discrete_output("gain_used", None)
    }

    fn setup_partials(&mut self) -> Result<()> {
        self.declare_partials("y", "x")
    }
}

#[async_trait]
impl ExplicitDiscipline for GainedSquare {
    async fn compute(&self, _inputs: &ArrayMap) -> Result<ArrayMap> {
        unreachable!("the server uses the discrete path when discrete variables exist")
    }

    async fn compute_with_discrete(
        &self,
        inputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        let gain = Self::gain(discrete_inputs);
        let sum_of_squares: f64 = inputs["x"].iter().map(|v| v * v).sum();

        let mut outputs = HashMap::new();
        outputs.insert("y".to_string(), scalar(gain * sum_of_squares));
        Ok((outputs, Self::echo(gain)))
    }

    async fn compute_partials_with_discrete(
        &self,
        inputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<PartialMap> {
        let gain = Self::gain(discrete_inputs);

        let mut partials: PartialMap = HashMap::new();
        partials.insert(
            ("y".to_string(), "x".to_string()),
            &inputs["x"] * (2.0 * gain),
        );
        Ok(partials)
    }
}

/// The Rosenbrock value and gradient at `x = [0.5, 0.6, ..., 1.2]`, computed
/// independently of the discipline so a mangled array cannot agree by accident.
const ROSEN_INPUT: [f64; 8] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
const ROSEN_VALUE: f64 = 45.36;
const ROSEN_GRADIENT: [f64; 8] = [-71.0, -12.4, -19.4, -21.6, -16.6, -2.0, 24.6, -2.0];

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "{what}: expected {expected}, got {actual}"
    );
}

#[tokio::test]
async fn with_stream_options_chunks_an_explicit_round_trip() {
    // An 8-element input at 3 doubles per slice cannot travel in one message, so a
    // broken split or reassembly shows up as zeroed-out trailing elements.
    let server = common::spawn_explicit(Rosenbrock::new()).await;
    let mut client = ExplicitClient::connect(server.endpoint())
        .await
        .unwrap()
        .with_stream_options(StreamOptions {
            max_double_per_slice: 3,
        });
    assert_eq!(client.base().stream_options().max_double_per_slice, 3);

    let mut options = HashMap::new();
    options.insert("dimension".to_string(), serde_json::json!(8));
    client.set_options(options).await.unwrap();
    client.setup().await.unwrap();
    client.get_variable_definitions().await.unwrap();
    client.get_partial_definitions().await.unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vector(&ROSEN_INPUT));

    // Only the request is chunked so far; the server still replies in one message.
    let outputs = client.compute_function(&inputs).await.unwrap();
    assert_close(outputs["f"][[0]], ROSEN_VALUE, "f with a chunked request");

    // Ask the server to chunk too, so the 8-element gradient comes back in three
    // messages and the response is reassembled from a partial-array stream.
    client
        .set_stream_options(StreamOptions {
            max_double_per_slice: 3,
        })
        .await
        .unwrap();

    let partials = client.compute_gradient(&inputs).await.unwrap();
    let df_dx = &partials[&("f".to_string(), "x".to_string())];
    assert_eq!(df_dx.shape(), &[8]);
    for (i, expected) in ROSEN_GRADIENT.iter().enumerate() {
        assert_close(df_dx[[i]], *expected, &format!("df/dx[{i}]"));
    }

    // A client left on the default 1000-double slice sees the same values, so the
    // chunk size changes only the framing.
    let mut whole = common::connected_explicit_client(&server).await;
    let unchunked = whole.compute_function(&inputs).await.unwrap();
    assert_close(unchunked["f"][[0]], outputs["f"][[0]], "f without chunking");

    server.shutdown().await;
}

#[tokio::test]
async fn an_rpc_timeout_ends_a_stalled_explicit_call() {
    let server = common::spawn_explicit(StalledExplicit::default()).await;
    let mut client = ExplicitClient::connect(server.endpoint())
        .await
        .unwrap()
        .with_rpc_timeout(Duration::from_millis(100));
    client.setup().await.unwrap();
    client.get_variable_definitions().await.unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), scalar(1.0));

    let started = std::time::Instant::now();
    let err = client.compute_function(&inputs).await.unwrap_err();
    let code = status_of(err).code();
    assert!(
        matches!(code, Code::DeadlineExceeded | Code::Cancelled),
        "expected the deadline to end the call, got {code:?}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "the call should have been cut short, not run to completion"
    );

    // Without a deadline the same call completes, so the failure above came from
    // the timeout rather than the discipline itself.
    let mut patient = common::connected_explicit_client(&server).await;
    let outputs = patient.compute_function(&inputs).await.unwrap();
    assert_eq!(outputs["y"][[0]], 1.0);

    server.shutdown().await;
}

#[tokio::test]
async fn discrete_inputs_reach_the_explicit_gradients() {
    let server = common::spawn_explicit(GainedSquare::default()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vector(&[1.0, 2.0, 3.0]));

    let mut discrete_inputs: DiscreteMap = HashMap::new();
    discrete_inputs.insert("gain".to_string(), json_to_value(&serde_json::json!(5)));

    // The value call echoes the gain back, confirming which one the server used.
    let (outputs, discrete_outputs) = client
        .compute_function_with_discrete(&inputs, &discrete_inputs)
        .await
        .unwrap();
    assert_eq!(outputs["y"][[0]], 70.0); // 5 * (1 + 4 + 9)
    assert_eq!(
        value_to_json(&discrete_outputs["gain_used"])
            .as_f64()
            .unwrap(),
        5.0
    );

    // dy/dx_i = 2 * gain * x_i
    let partials = client
        .compute_gradient_with_discrete(&inputs, &discrete_inputs)
        .await
        .unwrap();
    let dy_dx = &partials[&("y".to_string(), "x".to_string())];
    assert_eq!(dy_dx.shape(), &[3]);
    assert_eq!(dy_dx[[0]], 10.0);
    assert_eq!(dy_dx[[1]], 20.0);
    assert_eq!(dy_dx[[2]], 30.0);

    // Sending no discrete values falls back to the declared default of 2.
    let partials = client.compute_gradient(&inputs).await.unwrap();
    let dy_dx = &partials[&("y".to_string(), "x".to_string())];
    assert_eq!(dy_dx[[0]], 4.0);
    assert_eq!(dy_dx[[2]], 12.0);

    server.shutdown().await;
}

#[tokio::test]
async fn explicit_gradients_are_rejected_before_metadata_is_fetched() {
    // The Jacobian is scattered into arrays sized from the cached metadata, so a
    // gradient call without it cannot produce a correct result.
    let server = common::spawn_explicit(GainedSquare::default()).await;
    let mut client = ExplicitClient::connect(server.endpoint()).await.unwrap();
    client.setup().await.unwrap();
    // deliberately skip get_variable_definitions

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vector(&[1.0, 2.0, 3.0]));

    let mut discrete_inputs: DiscreteMap = HashMap::new();
    discrete_inputs.insert("gain".to_string(), json_to_value(&serde_json::json!(5)));

    let err = client
        .compute_gradient_with_discrete(&inputs, &discrete_inputs)
        .await
        .unwrap_err();
    assert!(
        matches!(err, PhiloteError::SetupNotCalled),
        "expected missing metadata to be reported, got {err:?}"
    );

    let err = client.compute_gradient(&inputs).await.unwrap_err();
    assert!(matches!(err, PhiloteError::SetupNotCalled));

    server.shutdown().await;
}

#[tokio::test]
async fn cloning_an_explicit_client_keeps_the_fetched_metadata() {
    // A clone shares the channel and the cache, so it can compute immediately;
    // re-running the handshake on every clone would be wasted round trips.
    let server = common::spawn_explicit(Paraboloid::new()).await;
    let mut client = common::connected_explicit_client(&server)
        .await
        .with_stream_options(StreamOptions {
            max_double_per_slice: 5,
        });

    let mut clone = client.clone();
    assert_eq!(clone.base().stream_options().max_double_per_slice, 5);
    assert_eq!(clone.var_meta().len(), client.var_meta().len());

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), scalar(1.0));
    inputs.insert("y".to_string(), scalar(2.0));

    // No setup or metadata fetch on the clone: the inherited cache is enough.
    let cloned_outputs = clone.compute_function(&inputs).await.unwrap();
    assert_eq!(cloned_outputs["f_xy"][[0]], 39.0);

    // The partials cache came across too, so the Jacobian can still be scattered.
    let partials = clone.compute_gradient(&inputs).await.unwrap();
    assert_eq!(partials[&("f_xy".to_string(), "x".to_string())][[0]], -2.0);
    assert_eq!(partials[&("f_xy".to_string(), "y".to_string())][[0]], 13.0);

    // The original is unaffected by the clone's use.
    let outputs = client.compute_function(&inputs).await.unwrap();
    assert_eq!(outputs["f_xy"][[0]], 39.0);

    server.shutdown().await;
}
