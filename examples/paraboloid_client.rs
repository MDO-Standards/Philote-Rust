//! Drives a paraboloid server.
//!
//! Mirrors Philote-Python's `examples/parabaloid_client.py`. Point it at either a
//! Rust or a Python paraboloid server to compare results across implementations.
//!
//! ```text
//! cargo run --example paraboloid_client [endpoint]
//! ```

use std::collections::HashMap;

use philote_mdo::client::ExplicitClient;
use philote_mdo::examples::scalar;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let endpoint = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "http://127.0.0.1:50051".to_string());

    let mut client = ExplicitClient::connect(endpoint.clone()).await?;
    println!("Connected to {endpoint}");

    let info = client.get_info().await?;
    println!("Discipline: {} v{}", info.name, info.version);

    // Metadata must be fetched before computing: responses are decoded using the
    // declared shapes.
    client.setup().await?;
    client.get_variable_definitions().await?;
    client.get_partial_definitions().await?;

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), scalar(1.0));
    inputs.insert("y".to_string(), scalar(2.0));

    let outputs = client.compute_function(&inputs).await?;
    println!("f_xy(1, 2) = {}", outputs["f_xy"][[0]]);

    let partials = client.compute_gradient(&inputs).await?;
    println!(
        "df/dx = {}, df/dy = {}",
        partials[&("f_xy".to_string(), "x".to_string())][[0]],
        partials[&("f_xy".to_string(), "y".to_string())][[0]],
    );

    Ok(())
}
