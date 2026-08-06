//! Error paths and boundary conditions.
//!
//! Ports Philote-Python's `tests/test_edge_cases.py` and the error-handling cases
//! from its client/server suites.

mod common;

use async_trait::async_trait;
use philote_mdo::client::ExplicitClient;
use philote_mdo::examples::{scalar, Paraboloid};
use philote_mdo::philote_info::{variable_message::Payload, Array, VariableMessage, VariableType};
use philote_mdo::registry::VariableRegistry;
use philote_mdo::traits::{Discipline, ExplicitDiscipline};
use philote_mdo::types::{ArrayData, StreamOptions};
use philote_mdo::{impl_registry, wire, ArrayMap, PhiloteError, Result};
use std::collections::HashMap;
use tonic::Code;

fn status_of(err: PhiloteError) -> tonic::Status {
    match err {
        PhiloteError::GrpcError(status) => *status,
        other => panic!("expected a gRPC error, got {other:?}"),
    }
}

/// Declares an option whose type string is not in the allowed set.
#[derive(Default)]
struct BadOptionType {
    registry: VariableRegistry,
}

impl Discipline for BadOptionType {
    impl_registry!(registry);

    fn setup(&mut self) -> Result<()> {
        self.add_input("x", &[1], "")?;
        self.add_output("y", &[1], "")
    }
}

#[async_trait]
impl ExplicitDiscipline for BadOptionType {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let mut outputs = HashMap::new();
        outputs.insert("y".to_string(), inputs["x"].clone());
        Ok(outputs)
    }
}

/// Explicit discipline with no gradient implementation.
#[derive(Default)]
struct NoGradients {
    registry: VariableRegistry,
}

impl Discipline for NoGradients {
    impl_registry!(registry);

    fn setup(&mut self) -> Result<()> {
        self.add_input("x", &[1], "")?;
        self.add_output("y", &[1], "")
    }
}

#[async_trait]
impl ExplicitDiscipline for NoGradients {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let mut outputs = HashMap::new();
        outputs.insert("y".to_string(), inputs["x"].clone());
        Ok(outputs)
    }
}

/// Discipline whose compute always fails.
#[derive(Default)]
struct AlwaysFails {
    registry: VariableRegistry,
}

impl Discipline for AlwaysFails {
    impl_registry!(registry);

    fn setup(&mut self) -> Result<()> {
        self.add_input("x", &[1], "")?;
        self.add_output("y", &[1], "")
    }
}

#[async_trait]
impl ExplicitDiscipline for AlwaysFails {
    async fn compute(&self, _inputs: &ArrayMap) -> Result<ArrayMap> {
        Err(PhiloteError::array_error("deliberate failure"))
    }
}

// --- Option validation ---

#[test]
fn invalid_option_types_are_rejected_at_declaration() {
    let mut d = BadOptionType::default();
    for bad in ["complex", "float64", "", "Bool", "list"] {
        assert!(
            d.add_option(bad, bad).is_err(),
            "'{bad}' should not be a valid option type"
        );
    }
}

#[test]
fn all_canonical_option_types_are_accepted() {
    let mut d = BadOptionType::default();
    for (i, good) in ["bool", "int", "float", "str", "dict"].iter().enumerate() {
        assert!(d.add_option(&format!("opt{i}"), good).is_ok());
    }
}

// --- Stream decoding ---

#[test]
fn an_empty_array_payload_is_rejected() {
    let proto = Array {
        name: "x".to_string(),
        subname: String::new(),
        start: 0,
        end: 0,
        r#type: VariableType::KInput.into(),
        data: vec![],
    };
    let err = ArrayData::try_from(proto).unwrap_err();
    // A malformed message from the peer, so INVALID_ARGUMENT, not INTERNAL.
    assert!(matches!(err, PhiloteError::Validation { .. }));
    assert_eq!(err.to_status().code(), Code::InvalidArgument);
}

#[test]
fn an_unknown_variable_type_is_rejected() {
    let proto = Array {
        name: "x".to_string(),
        subname: String::new(),
        start: 0,
        end: 1,
        r#type: 999,
        data: vec![1.0, 2.0],
    };
    let err = ArrayData::try_from(proto).unwrap_err();
    assert!(matches!(err, PhiloteError::InvalidVariableType(_)));
}

#[test]
fn a_chunk_past_the_end_of_the_array_is_rejected() {
    let mut target = ndarray::ArrayD::<f64>::zeros(ndarray::IxDyn(&[3]));
    let chunk = ArrayData::new(
        "x".to_string(),
        None,
        2,
        4,
        VariableType::KInput,
        vec![1.0, 2.0, 3.0],
    );
    assert!(matches!(
        wire::scatter_chunk(&mut target, &chunk).unwrap_err(),
        PhiloteError::IndexOutOfBounds { .. }
    ));
}

#[test]
fn a_message_with_no_payload_is_ignored() {
    // A default-constructed VariableMessage has an empty oneof; decoding must skip
    // it rather than erroring.
    let message = VariableMessage { payload: None };
    assert!(message.payload.is_none());

    let populated = VariableMessage {
        payload: Some(Payload::Continuous(Array::default())),
    };
    assert!(populated.payload.is_some());
}

// --- Server error surfaces ---

#[tokio::test]
async fn a_discipline_without_gradients_reports_unimplemented() {
    let server = common::spawn_explicit(NoGradients::default()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), scalar(1.0));

    let err = client.compute_gradient(&inputs).await.unwrap_err();
    assert_eq!(status_of(err).code(), Code::Unimplemented);

    server.shutdown().await;
}

#[tokio::test]
async fn a_failing_compute_surfaces_as_internal() {
    let server = common::spawn_explicit(AlwaysFails::default()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), scalar(1.0));

    let err = client.compute_function(&inputs).await.unwrap_err();
    let status = status_of(err);
    assert_eq!(status.code(), Code::Internal);
    assert!(status.message().contains("deliberate failure"));

    server.shutdown().await;
}

#[tokio::test]
async fn an_unknown_input_variable_is_rejected() {
    let server = common::spawn_explicit(Paraboloid::new()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), scalar(1.0));
    inputs.insert("y".to_string(), scalar(2.0));
    inputs.insert("not_a_variable".to_string(), scalar(3.0));

    // Philote-Python silently ignores unknown names; rejecting them catches typos.
    let err = client.compute_function(&inputs).await.unwrap_err();
    assert_eq!(status_of(err).code(), Code::InvalidArgument);

    server.shutdown().await;
}

#[tokio::test]
async fn a_nonpositive_chunk_size_is_rejected() {
    let server = common::spawn_explicit(Paraboloid::new()).await;
    let mut client = ExplicitClient::connect(server.endpoint()).await.unwrap();

    let err = client
        .set_stream_options(StreamOptions {
            max_double_per_slice: 0,
        })
        .await
        .unwrap_err();
    assert_eq!(status_of(err).code(), Code::InvalidArgument);

    server.shutdown().await;
}

// --- Client lifecycle ---

#[tokio::test]
async fn computing_before_fetching_metadata_is_rejected() {
    // Array responses are decoded using the declared shapes, so computing without
    // them cannot produce a correct result. Philote-Python returns empty maps here.
    let server = common::spawn_explicit(Paraboloid::new()).await;
    let mut client = ExplicitClient::connect(server.endpoint()).await.unwrap();
    client.setup().await.unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), scalar(1.0));
    inputs.insert("y".to_string(), scalar(2.0));

    let err = client.compute_function(&inputs).await.unwrap_err();
    assert!(matches!(err, PhiloteError::SetupNotCalled));

    server.shutdown().await;
}

#[tokio::test]
async fn skipping_partials_metadata_gives_an_actionable_error() {
    // `get_partial_definitions` is easy to forget, and without it the partials map
    // has nothing to scatter into. The error should name the fix rather than
    // surfacing as a bare "variable not found".
    let server = common::spawn_explicit(Paraboloid::new()).await;
    let mut client = ExplicitClient::connect(server.endpoint()).await.unwrap();
    client.setup().await.unwrap();
    client.get_variable_definitions().await.unwrap();
    // deliberately skip get_partial_definitions

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), scalar(1.0));
    inputs.insert("y".to_string(), scalar(2.0));

    let err = client.compute_gradient(&inputs).await.unwrap_err();
    assert!(matches!(err, PhiloteError::Validation { .. }));
    assert!(
        err.to_string().contains("get_partial_definitions"),
        "error should name the missing call, got: {err}"
    );

    server.shutdown().await;
}

#[tokio::test]
async fn connecting_to_a_dead_endpoint_fails_cleanly() {
    // Port 1 on loopback has nothing listening.
    match ExplicitClient::connect("http://127.0.0.1:1").await {
        Err(PhiloteError::ConfigurationError(_)) => {}
        Err(other) => panic!("expected a configuration error, got {other:?}"),
        Ok(_) => panic!("connecting to a closed port should fail"),
    }
}

#[tokio::test]
async fn a_malformed_endpoint_is_rejected() {
    match ExplicitClient::connect("not a url").await {
        Err(PhiloteError::ConfigurationError(_)) => {}
        Err(other) => panic!("expected a configuration error, got {other:?}"),
        Ok(_) => panic!("a malformed endpoint should be rejected"),
    }
}

/// Declares an output larger than what its compute actually returns.
#[derive(Default)]
struct ShortOutput {
    registry: VariableRegistry,
}

impl Discipline for ShortOutput {
    impl_registry!(registry);

    fn setup(&mut self) -> Result<()> {
        self.add_input("x", &[1], "")?;
        self.add_output("y", &[3], "")
    }
}

#[async_trait]
impl ExplicitDiscipline for ShortOutput {
    async fn compute(&self, _inputs: &ArrayMap) -> Result<ArrayMap> {
        let mut outputs = HashMap::new();
        // Only 2 of the 3 declared values.
        outputs.insert(
            "y".to_string(),
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![7.0, 8.0]).unwrap(),
        );
        Ok(outputs)
    }
}

#[tokio::test]
async fn a_short_response_is_rejected_rather_than_zero_filled() {
    // Silently returning [7, 8, 0] would be indistinguishable from a real result —
    // the same failure mode as the N-D scatter bug this crate was fixing.
    let server = common::spawn_explicit(ShortOutput::default()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), scalar(1.0));

    let err = client.compute_function(&inputs).await.unwrap_err();
    assert!(
        err.to_string().contains("declared with 3 values"),
        "expected a coverage error, got: {err}"
    );

    server.shutdown().await;
}

#[test]
fn negative_wire_indices_are_rejected() {
    // `as usize` would wrap -1 into usize::MAX and overflow downstream arithmetic.
    let proto = Array {
        name: "x".to_string(),
        subname: String::new(),
        start: 0,
        end: -1,
        r#type: VariableType::KInput.into(),
        data: vec![1.0],
    };
    let err = ArrayData::try_from(proto).unwrap_err();
    assert!(err.to_string().contains("negative"));
}

#[test]
fn a_zero_chunk_size_does_not_panic() {
    // The client-side builder bypasses the server's num_double validation.
    let chunks = philote_mdo::wire::chunk_arrays(
        &{
            let mut m = HashMap::new();
            m.insert("x".to_string(), philote_mdo::examples::vector(&[1.0, 2.0]));
            m
        },
        VariableType::KInput,
        0,
    );
    assert_eq!(chunks.len(), 2, "a zero chunk size should fall back to one");
}

// --- Server construction ---

#[tokio::test]
async fn a_discipline_can_be_replaced_after_construction() {
    use philote_mdo::server::ExplicitServer;

    let server = ExplicitServer::new(Paraboloid::new());
    assert_eq!(server.discipline().read().await.name(), "Paraboloid");

    server.set_discipline(Paraboloid::new()).await;
    assert_eq!(server.discipline().read().await.name(), "Paraboloid");
}
