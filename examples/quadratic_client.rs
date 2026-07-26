//! Drives an implicit quadratic server.
//!
//! Mirrors Philote-Python's `examples/quadratic_client.py`.
//!
//! ```text
//! cargo run --example quadratic_client [endpoint]
//! ```

use std::collections::HashMap;

use philote_mdo::client::ImplicitClient;
use philote_mdo::examples::scalar;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let endpoint = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "http://127.0.0.1:50051".to_string());

    let mut client = ImplicitClient::connect(endpoint.clone()).await?;
    println!("Connected to {endpoint}");

    let info = client.get_info().await?;
    println!("Discipline: {} v{}", info.name, info.version);

    client.setup().await?;
    client.get_variable_definitions().await?;
    client.get_partial_definitions().await?;

    // x^2 - 3x + 2 = 0, whose positive root is 2.
    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), scalar(1.0));
    inputs.insert("b".to_string(), scalar(-3.0));
    inputs.insert("c".to_string(), scalar(2.0));

    let outputs = client.solve_residuals(&inputs).await?;
    println!("solved x = {}", outputs["x"][[0]]);

    let residuals = client.compute_residuals(&inputs, &outputs).await?;
    println!("residual at the solution = {}", residuals["x"][[0]]);

    let partials = client.compute_residual_gradients(&inputs, &outputs).await?;
    println!(
        "dR/da = {}, dR/db = {}, dR/dc = {}, dR/dx = {}",
        partials[&("x".to_string(), "a".to_string())][[0]],
        partials[&("x".to_string(), "b".to_string())][[0]],
        partials[&("x".to_string(), "c".to_string())][[0]],
        partials[&("x".to_string(), "x".to_string())][[0]],
    );

    Ok(())
}
