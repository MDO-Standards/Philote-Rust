//! End-to-end coverage over a real socket.
//!
//! Ports Philote-Python's `tests/test_integration.py`, plus the explicit and
//! implicit client/server suites.

mod common;

use async_trait::async_trait;
use philote_mdo::examples::{scalar, vector, Paraboloid, QuadraticImplicit, Rosenbrock};
use philote_mdo::philote_info::VariableType;
use philote_mdo::registry::VariableRegistry;
use philote_mdo::traits::{Discipline, ExplicitDiscipline};
use philote_mdo::types::StreamOptions;
use philote_mdo::{impl_registry, ArrayMap, PhiloteError, Result};
use std::collections::HashMap;

fn paraboloid_inputs(x: f64, y: f64) -> ArrayMap {
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), scalar(x));
    inputs.insert("y".to_string(), scalar(y));
    inputs
}

// --- Discipline service ---

#[tokio::test]
async fn get_info_reports_discipline_properties() {
    let server = common::spawn_explicit(Paraboloid::new()).await;
    let mut client = philote_mdo::client::ExplicitClient::connect(server.endpoint())
        .await
        .unwrap();

    let info = client.get_info().await.unwrap();
    assert_eq!(info.name, "Paraboloid");
    assert_eq!(info.version, env!("CARGO_PKG_VERSION"));
    assert!(info.continuous);
    assert!(info.differentiable);
    assert!(info.provides_gradients);

    server.shutdown().await;
}

#[tokio::test]
async fn variable_definitions_carry_shape_and_units() {
    let server = common::spawn_explicit(Paraboloid::new()).await;
    let client = common::connected_explicit_client(&server).await;

    let vars = client.var_meta().to_vec();
    assert_eq!(vars.len(), 3);

    let f = vars.iter().find(|v| v.name == "f_xy").unwrap();
    assert_eq!(f.units, "m**2");
    assert_eq!(f.shape, vec![1]);
    assert_eq!(f.r#type, i32::from(VariableType::KOutput));

    let x = vars.iter().find(|v| v.name == "x").unwrap();
    assert_eq!(x.units, "m");
    assert_eq!(x.r#type, i32::from(VariableType::KInput));

    server.shutdown().await;
}

#[tokio::test]
async fn partial_definitions_are_reported() {
    let server = common::spawn_explicit(Paraboloid::new()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let partials = client.get_partial_definitions().await.unwrap();
    assert_eq!(partials.len(), 2);
    assert!(partials.iter().all(|p| p.name == "f_xy"));
    // Shapes are derived server-side; Philote-Python leaves this empty.
    assert!(partials.iter().all(|p| p.shape == vec![1]));

    server.shutdown().await;
}

#[tokio::test]
async fn setup_is_idempotent() {
    // Re-running Setup must not accumulate duplicate metadata.
    let server = common::spawn_explicit(Paraboloid::new()).await;
    let mut client = common::connected_explicit_client(&server).await;

    for _ in 0..3 {
        client.setup().await.unwrap();
    }
    let vars = client.get_variable_definitions().await.unwrap();
    assert_eq!(vars.len(), 3, "Setup should not duplicate definitions");

    server.shutdown().await;
}

// --- Explicit path ---

#[tokio::test]
async fn paraboloid_computes_over_the_wire() {
    let server = common::spawn_explicit(Paraboloid::new()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let outputs = client
        .compute_function(&paraboloid_inputs(1.0, 2.0))
        .await
        .unwrap();
    assert_eq!(outputs["f_xy"][[0]], 39.0);

    server.shutdown().await;
}

#[tokio::test]
async fn paraboloid_computes_gradients_over_the_wire() {
    let server = common::spawn_explicit(Paraboloid::new()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let partials = client
        .compute_gradient(&paraboloid_inputs(1.0, 2.0))
        .await
        .unwrap();

    assert_eq!(partials[&("f_xy".to_string(), "x".to_string())][[0]], -2.0);
    assert_eq!(partials[&("f_xy".to_string(), "y".to_string())][[0]], 13.0);

    server.shutdown().await;
}

#[tokio::test]
async fn large_arrays_survive_chunked_transport() {
    let server = common::spawn_explicit(Rosenbrock::new()).await;
    let mut client = philote_mdo::client::ExplicitClient::connect(server.endpoint())
        .await
        .unwrap();

    let n = 250usize;
    let mut options = HashMap::new();
    options.insert("dimension".to_string(), serde_json::json!(n));
    client.set_options(options).await.unwrap();
    client.setup().await.unwrap();
    client.get_variable_definitions().await.unwrap();
    client.get_partial_definitions().await.unwrap();

    // Force many chunks per array.
    client
        .set_stream_options(StreamOptions {
            max_double_per_slice: 7,
        })
        .await
        .unwrap();

    let values: Vec<f64> = (0..n).map(|_| 1.0).collect();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vector(&values));

    let outputs = client.compute_function(&inputs).await.unwrap();
    assert_eq!(outputs["f"][[0]], 0.0, "Rosenbrock is zero at all ones");

    let partials = client.compute_gradient(&inputs).await.unwrap();
    let grad = &partials[&("f".to_string(), "x".to_string())];
    assert_eq!(grad.shape(), &[n]);
    assert!(grad.iter().all(|v| *v == 0.0));

    server.shutdown().await;
}

// --- Implicit path ---

#[tokio::test]
async fn quadratic_solves_over_the_wire() {
    let server = common::spawn_implicit(QuadraticImplicit::new()).await;
    let mut client = common::connected_implicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), scalar(1.0));
    inputs.insert("b".to_string(), scalar(-3.0));
    inputs.insert("c".to_string(), scalar(2.0));

    let outputs = client.solve_residuals(&inputs).await.unwrap();
    assert_eq!(outputs["x"][[0]], 2.0);

    server.shutdown().await;
}

#[tokio::test]
async fn quadratic_residuals_are_distinct_from_outputs() {
    // A residual shares its output's name, so only the variable type separates
    // them on the wire. Getting this wrong silently merges the two.
    let server = common::spawn_implicit(QuadraticImplicit::new()).await;
    let mut client = common::connected_implicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), scalar(1.0));
    inputs.insert("b".to_string(), scalar(-3.0));
    inputs.insert("c".to_string(), scalar(2.0));

    let mut outputs = HashMap::new();
    outputs.insert("x".to_string(), scalar(0.0));

    // R(0) = c = 2, and must not be confused with the output value 0.
    let residuals = client.compute_residuals(&inputs, &outputs).await.unwrap();
    assert_eq!(residuals["x"][[0]], 2.0);

    server.shutdown().await;
}

#[tokio::test]
async fn quadratic_residual_gradients_over_the_wire() {
    let server = common::spawn_implicit(QuadraticImplicit::new()).await;
    let mut client = common::connected_implicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), scalar(1.0));
    inputs.insert("b".to_string(), scalar(-3.0));
    inputs.insert("c".to_string(), scalar(2.0));
    let mut outputs = HashMap::new();
    outputs.insert("x".to_string(), scalar(2.0));

    let partials = client
        .compute_residual_gradients(&inputs, &outputs)
        .await
        .unwrap();

    assert_eq!(partials[&("x".to_string(), "a".to_string())][[0]], 4.0);
    assert_eq!(partials[&("x".to_string(), "b".to_string())][[0]], 2.0);
    assert_eq!(partials[&("x".to_string(), "c".to_string())][[0]], 1.0);
    assert_eq!(partials[&("x".to_string(), "x".to_string())][[0]], 1.0);

    server.shutdown().await;
}

/// Implicit discipline whose registry comes from `Default`, i.e. without the
/// implicit flag set at construction.
#[derive(Default)]
struct DefaultBuiltImplicit {
    registry: VariableRegistry,
}

impl Discipline for DefaultBuiltImplicit {
    impl_registry!(registry);

    fn setup(&mut self) -> Result<()> {
        self.add_input("a", &[1], "")?;
        self.add_output("x", &[1], "")
    }
}

#[async_trait]
impl philote_mdo::traits::ImplicitDiscipline for DefaultBuiltImplicit {
    async fn compute_residuals(&self, inputs: &ArrayMap, outputs: &ArrayMap) -> Result<ArrayMap> {
        let mut residuals = HashMap::new();
        residuals.insert("x".to_string(), &outputs["x"] - &inputs["a"]);
        Ok(residuals)
    }

    async fn solve_residuals(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let mut outputs = HashMap::new();
        outputs.insert("x".to_string(), inputs["a"].clone());
        Ok(outputs)
    }
}

#[tokio::test]
async fn an_implicit_server_supplies_residuals_for_a_default_built_registry() {
    // VariableRegistry::default() is not implicit, so without the server marking it
    // the output would get no residual twin and ComputeResiduals would fail.
    let server = common::spawn_implicit(DefaultBuiltImplicit::default()).await;
    let mut client = common::connected_implicit_client(&server).await;

    let residuals = client
        .var_meta()
        .iter()
        .filter(|v| v.r#type == i32::from(VariableType::KResidual))
        .count();
    assert_eq!(residuals, 1, "the server should mark the registry implicit");

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), scalar(4.0));
    let mut outputs = HashMap::new();
    outputs.insert("x".to_string(), scalar(1.0));

    let residual = client.compute_residuals(&inputs, &outputs).await.unwrap();
    assert_eq!(residual["x"][[0]], -3.0);

    server.shutdown().await;
}

// --- Options ---

/// Declares one option of every supported type and records what it received.
#[derive(Default)]
struct OptionEcho {
    registry: VariableRegistry,
    seen: std::sync::Arc<std::sync::Mutex<HashMap<String, serde_json::Value>>>,
}

impl Discipline for OptionEcho {
    impl_registry!(registry);

    fn initialize(&mut self) -> Result<()> {
        self.add_option("flag", "bool")?;
        self.add_option("count", "int")?;
        self.add_option("scale", "float")?;
        self.add_option("label", "str")?;
        self.add_option("config", "dict")
    }

    fn set_options(&mut self, options: &HashMap<String, serde_json::Value>) -> Result<()> {
        *self.seen.lock().unwrap() = options.clone();
        Ok(())
    }

    fn setup(&mut self) -> Result<()> {
        self.add_input("x", &[1], "")?;
        self.add_output("y", &[1], "")
    }
}

#[async_trait]
impl ExplicitDiscipline for OptionEcho {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let mut outputs = HashMap::new();
        outputs.insert("y".to_string(), inputs["x"].clone());
        Ok(outputs)
    }
}

#[tokio::test]
async fn the_server_runs_initialize_so_options_are_advertised() {
    // Options are declared in initialize(), which a discipline built with
    // #[derive(Default)] never runs itself. The server must run it, or the
    // discipline appears to have no options at all.
    let server = common::spawn_explicit(OptionEcho::default()).await;
    let mut client = philote_mdo::client::ExplicitClient::connect(server.endpoint())
        .await
        .unwrap();

    let options = client.get_available_options().await.unwrap();
    assert_eq!(options.len(), 5, "initialize() was not run by the server");

    server.shutdown().await;
}

#[tokio::test]
async fn every_option_type_is_advertised() {
    let mut discipline = OptionEcho::default();
    discipline.initialize().unwrap();

    let server = common::spawn_explicit(discipline).await;
    let mut client = philote_mdo::client::ExplicitClient::connect(server.endpoint())
        .await
        .unwrap();

    let options = client.get_available_options().await.unwrap();
    assert_eq!(options["flag"], "bool");
    assert_eq!(options["count"], "int");
    assert_eq!(options["scale"], "float");
    assert_eq!(options["label"], "str");
    assert_eq!(options["config"], "dict");

    server.shutdown().await;
}

#[tokio::test]
async fn struct_valued_options_round_trip() {
    // Exercises the prost_types::Struct <-> serde_json path.
    let mut discipline = OptionEcho::default();
    discipline.initialize().unwrap();
    let seen = discipline.seen.clone();

    let server = common::spawn_explicit(discipline).await;
    let mut client = philote_mdo::client::ExplicitClient::connect(server.endpoint())
        .await
        .unwrap();

    let mut options = HashMap::new();
    options.insert("flag".to_string(), serde_json::json!(true));
    options.insert("count".to_string(), serde_json::json!(5));
    options.insert("scale".to_string(), serde_json::json!(1.5));
    options.insert("label".to_string(), serde_json::json!("hello"));
    options.insert(
        "config".to_string(),
        serde_json::json!({"nested": {"list": [1, 2, 3]}, "on": false}),
    );
    client.set_options(options.clone()).await.unwrap();

    let received = seen.lock().unwrap().clone();
    assert_eq!(received["flag"], serde_json::json!(true));
    assert_eq!(received["count"], serde_json::json!(5));
    assert_eq!(received["scale"], serde_json::json!(1.5));
    assert_eq!(received["label"], serde_json::json!("hello"));
    assert_eq!(
        received["config"],
        serde_json::json!({"nested": {"list": [1, 2, 3]}, "on": false})
    );

    server.shutdown().await;
}

#[tokio::test]
async fn options_reach_the_discipline_before_setup() {
    let server = common::spawn_explicit(Rosenbrock::new()).await;
    let mut client = philote_mdo::client::ExplicitClient::connect(server.endpoint())
        .await
        .unwrap();

    let mut options = HashMap::new();
    options.insert("dimension".to_string(), serde_json::json!(6));
    client.set_options(options).await.unwrap();
    client.setup().await.unwrap();

    let vars = client.get_variable_definitions().await.unwrap();
    let x = vars.iter().find(|v| v.name == "x").unwrap();
    assert_eq!(x.shape, vec![6], "options must apply before setup");

    server.shutdown().await;
}

#[tokio::test]
async fn an_invalid_option_value_surfaces_as_invalid_argument() {
    let server = common::spawn_explicit(Rosenbrock::new()).await;
    let mut client = philote_mdo::client::ExplicitClient::connect(server.endpoint())
        .await
        .unwrap();

    let mut options = HashMap::new();
    options.insert("dimension".to_string(), serde_json::json!("lots"));

    let err = client.set_options(options).await.unwrap_err();
    let status = match err {
        PhiloteError::GrpcError(status) => *status,
        other => panic!("expected a gRPC error, got {other:?}"),
    };
    assert_eq!(status.code(), tonic::Code::InvalidArgument);

    server.shutdown().await;
}
