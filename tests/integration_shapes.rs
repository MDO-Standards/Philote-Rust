//! End-to-end coverage for N-dimensional variables.
//!
//! Before the `wire::scatter_chunk` fix, the server wrote received data into
//! preallocated arrays with a flat `usize` index. That only addresses rank-1
//! arrays, so every variable of rank >= 2 arrived as zeros (and panicked in debug
//! builds). These tests drive real gRPC traffic with 2-D and 3-D variables.

mod common;

use async_trait::async_trait;
use ndarray::ArrayD;
use philote_mdo::registry::VariableRegistry;
use philote_mdo::traits::{Discipline, ExplicitDiscipline};
use philote_mdo::types::StreamOptions;
use philote_mdo::{impl_registry, ArrayMap, PartialMap, Result};
use std::collections::HashMap;

/// Takes a [2,3] input and returns its transpose, scaled by 10.
#[derive(Default)]
struct Transposer {
    registry: VariableRegistry,
}

impl Discipline for Transposer {
    impl_registry!(registry);

    fn name(&self) -> &str {
        "Transposer"
    }

    fn setup(&mut self) -> Result<()> {
        self.add_input("m", &[2, 3], "")?;
        self.add_output("mt", &[3, 2], "")
    }

    fn setup_partials(&mut self) -> Result<()> {
        self.declare_partials("mt", "m")
    }
}

#[async_trait]
impl ExplicitDiscipline for Transposer {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let m = &inputs["m"];
        let mut out: ArrayD<f64> = ArrayD::zeros(ndarray::IxDyn(&[3, 2]));
        for i in 0..2 {
            for j in 0..3 {
                out[[j, i]] = m[[i, j]] * 10.0;
            }
        }
        let mut outputs = HashMap::new();
        outputs.insert("mt".to_string(), out);
        Ok(outputs)
    }

    async fn compute_partials(&self, inputs: &ArrayMap) -> Result<PartialMap> {
        // Shape check only: fill the [3,2,2,3] Jacobian with the input's trace so a
        // wrong decode is visible in the values as well as the shape.
        let sum: f64 = inputs["m"].iter().sum();
        let mut partials: PartialMap = HashMap::new();
        partials.insert(
            ("mt".to_string(), "m".to_string()),
            ArrayD::from_elem(ndarray::IxDyn(&[3, 2, 2, 3]), sum),
        );
        Ok(partials)
    }
}

/// Sums a [2,2,2] input into a scalar.
#[derive(Default)]
struct CubeSum {
    registry: VariableRegistry,
}

impl Discipline for CubeSum {
    impl_registry!(registry);

    fn setup(&mut self) -> Result<()> {
        self.add_input("c", &[2, 2, 2], "")?;
        self.add_output("s", &[1], "")
    }
}

#[async_trait]
impl ExplicitDiscipline for CubeSum {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let sum: f64 = inputs["c"].iter().sum();
        let mut outputs = HashMap::new();
        outputs.insert("s".to_string(), philote_mdo::examples::scalar(sum));
        Ok(outputs)
    }
}

fn matrix_2x3() -> ArrayD<f64> {
    ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()
}

#[tokio::test]
async fn two_dimensional_variables_survive_a_round_trip() {
    let server = common::spawn_explicit(Transposer::default()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("m".to_string(), matrix_2x3());

    let outputs = client.compute_function(&inputs).await.unwrap();
    let mt = &outputs["mt"];

    assert_eq!(mt.shape(), &[3, 2], "output shape must be preserved");
    // Transpose of [[1,2,3],[4,5,6]] scaled by 10.
    assert_eq!(mt[[0, 0]], 10.0);
    assert_eq!(mt[[0, 1]], 40.0);
    assert_eq!(mt[[1, 0]], 20.0);
    assert_eq!(mt[[1, 1]], 50.0);
    assert_eq!(mt[[2, 0]], 30.0);
    assert_eq!(mt[[2, 1]], 60.0);

    server.shutdown().await;
}

#[tokio::test]
async fn two_dimensional_input_reaches_the_discipline_intact() {
    // The clearest statement of the original defect: the discipline used to see
    // all zeros. Summing the input proves every element arrived.
    let server = common::spawn_explicit(CubeSum::default()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert(
        "c".to_string(),
        ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap(),
    );

    let outputs = client.compute_function(&inputs).await.unwrap();
    assert_eq!(outputs["s"][[0]], 36.0, "3-D input was not received intact");

    server.shutdown().await;
}

#[tokio::test]
async fn n_dimensional_round_trip_survives_chunking() {
    // Force multiple chunks per array so reassembly is exercised too.
    let server = common::spawn_explicit(Transposer::default()).await;
    let mut client = common::connected_explicit_client(&server).await;
    client
        .set_stream_options(StreamOptions {
            max_double_per_slice: 2,
        })
        .await
        .unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("m".to_string(), matrix_2x3());

    let outputs = client.compute_function(&inputs).await.unwrap();
    assert_eq!(outputs["mt"].shape(), &[3, 2]);
    assert_eq!(outputs["mt"][[2, 1]], 60.0);

    server.shutdown().await;
}

#[tokio::test]
async fn partials_keep_their_derived_shape() {
    let server = common::spawn_explicit(Transposer::default()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let mut inputs = HashMap::new();
    inputs.insert("m".to_string(), matrix_2x3());

    let partials = client.compute_gradient(&inputs).await.unwrap();
    let jac = &partials[&("mt".to_string(), "m".to_string())];

    // d(mt)/d(m) concatenates the output and input shapes.
    assert_eq!(jac.shape(), &[3, 2, 2, 3]);
    assert_eq!(jac[[0, 0, 0, 0]], 21.0); // sum of 1..6
    assert_eq!(jac[[2, 1, 1, 2]], 21.0);

    server.shutdown().await;
}
