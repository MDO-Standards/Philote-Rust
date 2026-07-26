//! Client-resolved variable shapes, end to end.
//!
//! Ports the non-OpenMDAO cases from Philote-Python's `tests/test_dynamic_shapes.py`.
//! Before this work the server accepted the `SetVariableShapes` stream and discarded
//! it, so nothing here was reachable.

mod common;

use async_trait::async_trait;
use philote_mdo::client::variable_shape_meta;
use philote_mdo::examples::{vector, FlexibleDiscipline};
use philote_mdo::philote_info::VariableType;
use philote_mdo::registry::VariableRegistry;
use philote_mdo::traits::{Discipline, ImplicitDiscipline};
use philote_mdo::{impl_registry, ArrayMap, PhiloteError, Result};
use std::collections::HashMap;
use tonic::Code;

/// Implicit discipline with a dynamically shaped output, to exercise residual twins.
#[derive(Default)]
struct DynamicImplicit {
    registry: VariableRegistry,
}

impl DynamicImplicit {
    fn new() -> Self {
        Self {
            registry: VariableRegistry::new(true),
        }
    }
}

impl Discipline for DynamicImplicit {
    impl_registry!(registry);

    fn setup(&mut self) -> Result<()> {
        self.add_dynamic_input("a", "")?;
        self.add_dynamic_output("x", "")
    }
}

#[async_trait]
impl ImplicitDiscipline for DynamicImplicit {
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

fn status_of(err: PhiloteError) -> tonic::Status {
    match err {
        PhiloteError::GrpcError(status) => *status,
        other => panic!("expected a gRPC error, got {other:?}"),
    }
}

// --- Registry-level behaviour ---

#[test]
fn dynamic_variables_are_reported_with_the_flag_set() {
    let mut d = FlexibleDiscipline::new();
    d.setup().unwrap();

    for var in d.get_variable_definitions().unwrap() {
        assert!(var.dynamic_shape);
        assert!(var.shape.is_empty());
    }
}

#[test]
fn dynamic_implicit_output_gets_a_dynamic_residual() {
    let mut d = DynamicImplicit::new();
    d.setup().unwrap();

    let residual = d
        .get_variable_definitions()
        .unwrap()
        .into_iter()
        .find(|v| v.r#type == i32::from(VariableType::KResidual))
        .expect("implicit outputs get a residual twin");
    assert_eq!(residual.name, "x");
    assert!(residual.dynamic_shape);
}

// --- RPC-level behaviour ---

#[tokio::test]
async fn client_can_resolve_shapes() {
    let server = common::spawn_explicit(FlexibleDiscipline::new()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let dynamic = client.get_dynamic_variables().await.unwrap();
    assert_eq!(dynamic.len(), 2);

    client
        .send_variable_shapes(vec![
            variable_shape_meta("x", &[3], VariableType::KInput),
            variable_shape_meta("y", &[3], VariableType::KOutput),
        ])
        .await
        .unwrap();

    // The local cache must reflect the resolved shapes, or responses decode into
    // zero-length arrays.
    for var in client.var_meta() {
        assert_eq!(var.shape, vec![3], "{} was not updated locally", var.name);
    }

    server.shutdown().await;
}

#[tokio::test]
async fn resolving_a_static_variable_is_rejected() {
    let server = common::spawn_explicit(philote_mdo::examples::Paraboloid::new()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let err = client
        .send_variable_shapes(vec![variable_shape_meta("x", &[4], VariableType::KInput)])
        .await
        .unwrap_err();

    assert_eq!(status_of(err).code(), Code::InvalidArgument);
    server.shutdown().await;
}

#[tokio::test]
async fn resolving_an_unknown_variable_is_rejected() {
    let server = common::spawn_explicit(FlexibleDiscipline::new()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let err = client
        .send_variable_shapes(vec![variable_shape_meta(
            "nope",
            &[3],
            VariableType::KInput,
        )])
        .await
        .unwrap_err();

    assert_eq!(status_of(err).code(), Code::InvalidArgument);
    server.shutdown().await;
}

#[tokio::test]
async fn a_zero_dimension_shape_is_rejected() {
    let server = common::spawn_explicit(FlexibleDiscipline::new()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let err = client
        .send_variable_shapes(vec![variable_shape_meta("x", &[0], VariableType::KInput)])
        .await
        .unwrap_err();

    assert_eq!(status_of(err).code(), Code::InvalidArgument);
    server.shutdown().await;
}

#[tokio::test]
async fn computing_before_resolving_shapes_is_rejected() {
    let server = common::spawn_explicit(FlexibleDiscipline::new()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vector(&[1.0, 2.0, 3.0]));

    let err = client.compute_function(&inputs).await.unwrap_err();
    assert_eq!(status_of(err).code(), Code::InvalidArgument);

    server.shutdown().await;
}

// --- End to end ---

#[tokio::test]
async fn computes_after_shapes_are_resolved() {
    let server = common::spawn_explicit(FlexibleDiscipline::new()).await;
    let mut client = common::connected_explicit_client(&server).await;

    client
        .send_variable_shapes(vec![
            variable_shape_meta("x", &[3], VariableType::KInput),
            variable_shape_meta("y", &[3], VariableType::KOutput),
        ])
        .await
        .unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vector(&[1.0, 2.0, 3.0]));

    let outputs = client.compute_function(&inputs).await.unwrap();
    assert_eq!(outputs["y"].shape(), &[3]);
    assert_eq!(outputs["y"][[0]], 2.0);
    assert_eq!(outputs["y"][[1]], 4.0);
    assert_eq!(outputs["y"][[2]], 6.0);

    server.shutdown().await;
}

#[tokio::test]
async fn computes_gradients_after_shapes_are_resolved() {
    let server = common::spawn_explicit(FlexibleDiscipline::new()).await;
    let mut client = common::connected_explicit_client(&server).await;

    client
        .send_variable_shapes(vec![
            variable_shape_meta("x", &[3], VariableType::KInput),
            variable_shape_meta("y", &[3], VariableType::KOutput),
        ])
        .await
        .unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vector(&[1.0, 2.0, 3.0]));

    let partials = client.compute_gradient(&inputs).await.unwrap();
    let jac = &partials[&("y".to_string(), "x".to_string())];

    assert_eq!(jac.shape(), &[3, 3]);
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(jac[[i, j]], if i == j { 2.0 } else { 0.0 });
        }
    }

    server.shutdown().await;
}

#[tokio::test]
async fn a_second_resolution_can_change_the_shape() {
    let server = common::spawn_explicit(FlexibleDiscipline::new()).await;
    let mut client = common::connected_explicit_client(&server).await;

    for n in [2usize, 5] {
        client
            .send_variable_shapes(vec![
                variable_shape_meta("x", &[n], VariableType::KInput),
                variable_shape_meta("y", &[n], VariableType::KOutput),
            ])
            .await
            .unwrap();

        let values: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), vector(&values));

        let outputs = client.compute_function(&inputs).await.unwrap();
        assert_eq!(outputs["y"].shape(), &[n]);
        assert_eq!(outputs["y"][[n - 1]], values[n - 1] * 2.0);
    }

    server.shutdown().await;
}

#[tokio::test]
async fn resolving_an_implicit_output_also_resolves_its_residual() {
    let server = common::spawn_implicit(DynamicImplicit::new()).await;
    let mut client = common::connected_implicit_client(&server).await;

    client
        .send_variable_shapes(vec![
            variable_shape_meta("a", &[2], VariableType::KInput),
            variable_shape_meta("x", &[2], VariableType::KOutput),
        ])
        .await
        .unwrap();

    // The residual was never named in the request, but must be resolved too.
    let residual = client
        .var_meta()
        .iter()
        .find(|v| v.r#type == i32::from(VariableType::KResidual))
        .unwrap();
    assert_eq!(residual.shape, vec![2]);

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), vector(&[3.0, 4.0]));

    let outputs = client.solve_residuals(&inputs).await.unwrap();
    assert_eq!(outputs["x"].shape(), &[2]);

    let residuals = client.compute_residuals(&inputs, &outputs).await.unwrap();
    assert_eq!(residuals["x"].shape(), &[2]);
    assert_eq!(residuals["x"][[0]], 0.0);
    assert_eq!(residuals["x"][[1]], 0.0);

    server.shutdown().await;
}

#[tokio::test]
async fn re_resolving_shapes_after_fetching_partials_stays_consistent() {
    // The server reports a derived partial shape, which the client caches. Resolving
    // shapes again makes that cached shape wrong, so it has to be invalidated —
    // otherwise the Jacobian is decoded against the previous dimensions.
    let server = common::spawn_explicit(FlexibleDiscipline::new()).await;
    let mut client = common::connected_explicit_client(&server).await;

    client
        .send_variable_shapes(vec![
            variable_shape_meta("x", &[3], VariableType::KInput),
            variable_shape_meta("y", &[3], VariableType::KOutput),
        ])
        .await
        .unwrap();
    // Fetch partials while the shapes say [3], baking [3, 3] into the cache.
    client.get_partial_definitions().await.unwrap();

    client
        .send_variable_shapes(vec![
            variable_shape_meta("x", &[5], VariableType::KInput),
            variable_shape_meta("y", &[5], VariableType::KOutput),
        ])
        .await
        .unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vector(&[1.0, 2.0, 3.0, 4.0, 5.0]));

    let partials = client.compute_gradient(&inputs).await.unwrap();
    assert_eq!(
        partials[&("y".to_string(), "x".to_string())].shape(),
        &[5, 5]
    );

    server.shutdown().await;
}

#[tokio::test]
async fn re_running_setup_invalidates_the_metadata_cache() {
    // Setup rebuilds server-side metadata, so a client that reuses its cache would
    // decode later responses against shapes that no longer exist.
    let server = common::spawn_explicit(philote_mdo::examples::Rosenbrock::new()).await;
    let mut client = philote_mdo::client::ExplicitClient::connect(server.endpoint())
        .await
        .unwrap();

    let mut options = HashMap::new();
    options.insert("dimension".to_string(), serde_json::json!(5));
    client.set_options(options).await.unwrap();
    client.setup().await.unwrap();
    client.get_variable_definitions().await.unwrap();
    client.get_partial_definitions().await.unwrap();

    // Resize, then re-run setup without re-fetching.
    let mut options = HashMap::new();
    options.insert("dimension".to_string(), serde_json::json!(10));
    client.set_options(options).await.unwrap();
    client.setup().await.unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vector(&[1.0; 10]));

    let err = client.compute_function(&inputs).await.unwrap_err();
    assert!(
        matches!(err, PhiloteError::SetupNotCalled),
        "expected a stale-cache rejection, got: {err}"
    );

    // Re-fetching makes the client usable again.
    client.get_variable_definitions().await.unwrap();
    client.get_partial_definitions().await.unwrap();
    let outputs = client.compute_function(&inputs).await.unwrap();
    assert_eq!(outputs["f"][[0]], 0.0);

    server.shutdown().await;
}

#[tokio::test]
async fn static_shape_disciplines_are_unaffected() {
    // Regression guard: adding dynamic-shape support must not disturb the normal path.
    let server = common::spawn_explicit(philote_mdo::examples::Paraboloid::new()).await;
    let mut client = common::connected_explicit_client(&server).await;

    assert!(client.get_dynamic_variables().await.unwrap().is_empty());

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), philote_mdo::examples::scalar(1.0));
    inputs.insert("y".to_string(), philote_mdo::examples::scalar(2.0));

    let outputs = client.compute_function(&inputs).await.unwrap();
    assert_eq!(outputs["f_xy"][[0]], 39.0);

    server.shutdown().await;
}
