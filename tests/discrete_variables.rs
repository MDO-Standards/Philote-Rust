//! Discrete variables, end to end.
//!
//! Ports Philote-Python's `tests/test_discrete_variables.py` and
//! `tests/test_discrete_integration.py`. Discrete variables travel as
//! `google.protobuf.Value`, multiplexed with continuous arrays in the same stream.

mod common;

use async_trait::async_trait;
use philote_mdo::discrete::{json_to_value, value_to_json};
use philote_mdo::examples::{scalar, Paraboloid};
use philote_mdo::philote_info::VariableType;
use philote_mdo::registry::VariableRegistry;
use philote_mdo::traits::{Discipline, ExplicitDiscipline};
use philote_mdo::{impl_registry, ArrayMap, DiscreteMap, PartialMap, PhiloteError, Result};
use std::collections::HashMap;
use tonic::Code;

/// Scales `x` by a discrete `factor`, and echoes a discrete label back.
#[derive(Default)]
struct ScaledParaboloid {
    registry: VariableRegistry,
}

impl Discipline for ScaledParaboloid {
    impl_registry!(registry);

    fn name(&self) -> &str {
        "ScaledParaboloid"
    }

    fn setup(&mut self) -> Result<()> {
        self.add_input("x", &[1], "")?;
        self.add_output("y", &[1], "")?;
        // Default of 2 applies when the client sends nothing.
        self.add_discrete_input("factor", Some(json_to_value(&serde_json::json!(2))))?;
        self.add_discrete_output("label", None)
    }

    fn setup_partials(&mut self) -> Result<()> {
        self.declare_partials("y", "x")
    }
}

#[async_trait]
impl ExplicitDiscipline for ScaledParaboloid {
    async fn compute(&self, _inputs: &ArrayMap) -> Result<ArrayMap> {
        unreachable!("the server uses the discrete path when discrete variables exist")
    }

    async fn compute_with_discrete(
        &self,
        inputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        let factor = discrete_inputs
            .get("factor")
            .map(|v| value_to_json(v).as_f64().unwrap_or(1.0))
            .unwrap_or(1.0);

        let mut outputs = HashMap::new();
        outputs.insert("y".to_string(), &inputs["x"] * factor);

        let mut discrete_outputs = HashMap::new();
        discrete_outputs.insert(
            "label".to_string(),
            json_to_value(&serde_json::json!(format!("scaled by {factor}"))),
        );

        Ok((outputs, discrete_outputs))
    }

    async fn compute_partials_with_discrete(
        &self,
        _inputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<PartialMap> {
        let factor = discrete_inputs
            .get("factor")
            .map(|v| value_to_json(v).as_f64().unwrap_or(1.0))
            .unwrap_or(1.0);

        let mut partials: PartialMap = HashMap::new();
        partials.insert(("y".to_string(), "x".to_string()), scalar(factor));
        Ok(partials)
    }
}

fn discrete(value: serde_json::Value) -> prost_types::Value {
    json_to_value(&value)
}

fn status_of(err: PhiloteError) -> tonic::Status {
    match err {
        PhiloteError::GrpcError(status) => *status,
        other => panic!("expected a gRPC error, got {other:?}"),
    }
}

// --- Declaration ---

#[test]
fn declares_discrete_variables_separately_from_continuous() {
    let mut d = ScaledParaboloid::default();
    d.setup().unwrap();

    assert_eq!(d.get_variable_definitions().unwrap().len(), 2);

    let discrete_vars = d.get_discrete_variable_definitions().unwrap();
    assert_eq!(discrete_vars.len(), 2);
    assert_eq!(
        discrete_vars
            .iter()
            .find(|v| v.name == "factor")
            .unwrap()
            .r#type,
        i32::from(VariableType::KDiscreteInput)
    );
    assert_eq!(
        discrete_vars
            .iter()
            .find(|v| v.name == "label")
            .unwrap()
            .r#type,
        i32::from(VariableType::KDiscreteOutput)
    );
}

#[test]
fn stores_declared_defaults() {
    let mut d = ScaledParaboloid::default();
    d.setup().unwrap();

    let defaults = d.registry().discrete_input_defaults();
    assert_eq!(defaults.len(), 1);
    assert_eq!(value_to_json(&defaults["factor"]), serde_json::json!(2));
}

// --- Value round trips ---

#[test]
fn every_value_kind_round_trips() {
    let cases = vec![
        serde_json::Value::Null,
        serde_json::json!(true),
        serde_json::json!(false),
        serde_json::json!(42),
        serde_json::json!(-17),
        serde_json::json!(2.5),
        serde_json::json!("text"),
        serde_json::json!([1, "two", false, null]),
        serde_json::json!({"nested": {"deep": [1, 2, 3]}, "flag": true}),
    ];

    for case in cases {
        let recovered = value_to_json(&json_to_value(&case));
        assert_eq!(recovered, case, "round trip failed for {case}");
    }
}

#[test]
fn whole_numbers_come_back_as_integers() {
    let recovered = value_to_json(&json_to_value(&serde_json::json!(7)));
    assert!(recovered.is_i64(), "expected an integer, got {recovered}");
}

// --- End to end ---

#[tokio::test]
async fn discrete_input_reaches_the_discipline() {
    let server = common::spawn_explicit(ScaledParaboloid::default()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), scalar(3.0));

    let mut discrete_inputs: DiscreteMap = HashMap::new();
    discrete_inputs.insert("factor".to_string(), discrete(serde_json::json!(10)));

    let (outputs, discrete_outputs) = client
        .compute_function_with_discrete(&inputs, &discrete_inputs)
        .await
        .unwrap();

    assert_eq!(outputs["y"][[0]], 30.0);
    assert_eq!(
        value_to_json(&discrete_outputs["label"]),
        serde_json::json!("scaled by 10")
    );

    server.shutdown().await;
}

#[tokio::test]
async fn declared_default_applies_when_the_client_sends_nothing() {
    let server = common::spawn_explicit(ScaledParaboloid::default()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), scalar(3.0));

    // No discrete inputs sent: the declared default of 2 should be used.
    let outputs = client.compute_function(&inputs).await.unwrap();
    assert_eq!(outputs["y"][[0]], 6.0);

    server.shutdown().await;
}

#[tokio::test]
async fn discrete_inputs_flow_through_gradients() {
    let server = common::spawn_explicit(ScaledParaboloid::default()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), scalar(3.0));

    let mut discrete_inputs: DiscreteMap = HashMap::new();
    discrete_inputs.insert("factor".to_string(), discrete(serde_json::json!(7)));

    let partials = client
        .compute_gradient_with_discrete(&inputs, &discrete_inputs)
        .await
        .unwrap();

    assert_eq!(partials[&("y".to_string(), "x".to_string())][[0]], 7.0);

    server.shutdown().await;
}

#[tokio::test]
async fn discrete_metadata_is_kept_out_of_the_continuous_cache() {
    // Discrete metadata carries no shape, so leaking it into the continuous cache
    // would make array preallocation build a bogus zero-length entry.
    let server = common::spawn_explicit(ScaledParaboloid::default()).await;
    let client = common::connected_explicit_client(&server).await;

    for var in client.var_meta() {
        assert!(
            var.name == "x" || var.name == "y",
            "{} should not be in the continuous cache",
            var.name
        );
    }
    assert_eq!(client.base().discrete_meta().len(), 2);

    server.shutdown().await;
}

/// Declares nothing but discrete variables — no continuous arrays at all.
#[derive(Default)]
struct DiscreteOnly {
    registry: VariableRegistry,
}

impl Discipline for DiscreteOnly {
    impl_registry!(registry);

    fn setup(&mut self) -> Result<()> {
        self.add_discrete_input("word", Some(json_to_value(&serde_json::json!("hi"))))?;
        self.add_discrete_output("shout", None)
    }
}

#[async_trait]
impl ExplicitDiscipline for DiscreteOnly {
    async fn compute(&self, _inputs: &ArrayMap) -> Result<ArrayMap> {
        unreachable!("discrete path is used")
    }

    async fn compute_with_discrete(
        &self,
        _inputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        let word = discrete_inputs
            .get("word")
            .map(|v| value_to_json(v).as_str().unwrap_or("").to_string())
            .unwrap_or_default();

        let mut discrete_outputs = HashMap::new();
        discrete_outputs.insert(
            "shout".to_string(),
            json_to_value(&serde_json::json!(word.to_uppercase())),
        );
        Ok((HashMap::new(), discrete_outputs))
    }
}

#[tokio::test]
async fn a_discipline_with_only_discrete_variables_is_usable() {
    // Continuous metadata is legitimately empty here, so the client's
    // "did you fetch metadata?" guard must not infer from emptiness.
    let server = common::spawn_explicit(DiscreteOnly::default()).await;
    let mut client = common::connected_explicit_client(&server).await;

    assert!(client.var_meta().is_empty());

    let (_, discrete_outputs) = client
        .compute_function_with_discrete(&HashMap::new(), &HashMap::new())
        .await
        .expect("a discrete-only discipline should be reachable");

    // The declared default of "hi" was applied server-side.
    assert_eq!(
        value_to_json(&discrete_outputs["shout"]),
        serde_json::json!("HI")
    );

    server.shutdown().await;
}

#[tokio::test]
async fn structured_discrete_values_survive_the_round_trip() {
    let server = common::spawn_explicit(ScaledParaboloid::default()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), scalar(1.0));

    // A nested structure is accepted even though this discipline reads it as a
    // number; the point is that the payload survives transport.
    let mut discrete_inputs: DiscreteMap = HashMap::new();
    discrete_inputs.insert(
        "factor".to_string(),
        discrete(serde_json::json!({"a": [1, 2], "b": "c"})),
    );

    let (_, discrete_outputs) = client
        .compute_function_with_discrete(&inputs, &discrete_inputs)
        .await
        .unwrap();

    assert!(discrete_outputs.contains_key("label"));

    server.shutdown().await;
}

// --- Undeclared discrete inputs ---

#[tokio::test]
async fn an_undeclared_discrete_input_is_rejected() {
    // The client does not filter discrete names against the fetched metadata, so a
    // typo reaches the server; the server must reject it rather than quietly build
    // a map entry the discipline will never read.
    let server = common::spawn_explicit(ScaledParaboloid::default()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), scalar(3.0));

    let mut discrete_inputs: DiscreteMap = HashMap::new();
    discrete_inputs.insert("factr".to_string(), discrete(serde_json::json!(10)));

    let err = client
        .compute_function_with_discrete(&inputs, &discrete_inputs)
        .await
        .expect_err("an undeclared discrete input should be rejected");
    let status = status_of(err);
    assert_eq!(status.code(), Code::InvalidArgument);
    assert!(
        status.message().contains("factr"),
        "the offending name should be named: {}",
        status.message()
    );

    server.shutdown().await;
}

#[tokio::test]
async fn a_discipline_without_discrete_variables_rejects_discrete_values() {
    // Regression guard: this used to be silently dropped, so a client sending
    // discrete data to a purely continuous discipline saw a successful compute.
    let server = common::spawn_explicit(Paraboloid::new()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), scalar(1.0));
    inputs.insert("y".to_string(), scalar(2.0));

    let mut discrete_inputs: DiscreteMap = HashMap::new();
    discrete_inputs.insert("mode".to_string(), discrete(serde_json::json!("fast")));

    let err = client
        .compute_function_with_discrete(&inputs, &discrete_inputs)
        .await
        .expect_err("a discipline with no discrete variables should reject discrete values");
    assert_eq!(status_of(err).code(), Code::InvalidArgument);

    // Without the discrete value the same call succeeds, so the rejection is
    // specific to the discrete payload.
    let outputs = client.compute_function(&inputs).await.unwrap();
    assert_eq!(outputs["f_xy"][[0]], 39.0);

    server.shutdown().await;
}

/// Declares a discrete input with no default value.
#[derive(Default)]
struct DiscreteWithoutDefault {
    registry: VariableRegistry,
}

impl Discipline for DiscreteWithoutDefault {
    impl_registry!(registry);

    fn setup(&mut self) -> Result<()> {
        self.add_input("x", &[1], "")?;
        self.add_output("y", &[1], "")?;
        // No default: the declared set must come from the metadata, not the
        // defaults map, or this name would look undeclared on arrival.
        self.add_discrete_input("gain", None)?;
        self.add_discrete_output("used", None)
    }
}

#[async_trait]
impl ExplicitDiscipline for DiscreteWithoutDefault {
    async fn compute(&self, _inputs: &ArrayMap) -> Result<ArrayMap> {
        unreachable!("the server uses the discrete path when discrete variables exist")
    }

    async fn compute_with_discrete(
        &self,
        inputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        let gain = discrete_inputs
            .get("gain")
            .map(|v| value_to_json(v).as_f64().unwrap_or(1.0))
            .unwrap_or(1.0);

        let mut outputs = HashMap::new();
        outputs.insert("y".to_string(), &inputs["x"] * gain);

        let mut discrete_outputs = HashMap::new();
        discrete_outputs.insert("used".to_string(), json_to_value(&serde_json::json!(gain)));
        Ok((outputs, discrete_outputs))
    }
}

#[tokio::test]
async fn a_declared_discrete_input_without_a_default_is_accepted() {
    let server = common::spawn_explicit(DiscreteWithoutDefault::default()).await;
    let mut client = common::connected_explicit_client(&server).await;

    assert_eq!(client.base().discrete_meta().len(), 2);

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), scalar(4.0));

    let mut discrete_inputs: DiscreteMap = HashMap::new();
    discrete_inputs.insert("gain".to_string(), discrete(serde_json::json!(3)));

    let (outputs, discrete_outputs) = client
        .compute_function_with_discrete(&inputs, &discrete_inputs)
        .await
        .expect("a declared discrete input without a default should be accepted");

    assert_eq!(outputs["y"][[0]], 12.0);
    assert_eq!(
        value_to_json(&discrete_outputs["used"]),
        serde_json::json!(3)
    );

    server.shutdown().await;
}

#[test]
fn a_discrete_input_without_a_default_is_still_declared() {
    let mut d = DiscreteWithoutDefault::default();
    d.setup().unwrap();

    assert!(d.registry().discrete_input_defaults().is_empty());
    assert!(d.registry().discrete_input_names().contains("gain"));
}
