//! Full client walkthrough against the `server_runner` example.
//!
//! Shows the complete handshake: connect, inspect, configure options, run setup,
//! fetch metadata, then compute.
//!
//! ```text
//! cargo run --example server_runner    # in one terminal
//! cargo run --example client_example   # in another
//! ```

use std::collections::HashMap;

use philote_mdo::client::ExplicitClient;
use philote_mdo::examples::vector;
use philote_mdo::types::StreamOptions;

const DIMENSION: usize = 4;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let endpoint = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "http://127.0.0.1:50051".to_string());

    let mut client = ExplicitClient::connect(endpoint.clone()).await?;
    println!("Connected to {endpoint}");

    let info = client.get_info().await?;
    println!(
        "Discipline: {} v{} (differentiable: {}, provides gradients: {})",
        info.name, info.version, info.differentiable, info.provides_gradients
    );

    // Smaller chunks than the default 1000, to exercise streaming.
    client
        .set_stream_options(StreamOptions {
            max_double_per_slice: 2,
        })
        .await?;

    let available = client.get_available_options().await?;
    println!("Available options: {available:?}");

    let mut options = HashMap::new();
    options.insert("dimension".to_string(), serde_json::json!(DIMENSION));
    client.set_options(options).await?;

    // Setup must follow the options, since they determine the variable shapes.
    client.setup().await?;

    let variables = client.get_variable_definitions().await?;
    println!("\nVariables:");
    for var in &variables {
        println!("  {} shape={:?}", var.name, var.shape);
    }
    client.get_partial_definitions().await?;

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vector(&[0.5, 1.7, -0.3, 2.1]));

    let outputs = client.compute_function(&inputs).await?;
    println!("\nf(x) = {}", outputs["f"][[0]]);

    let partials = client.compute_gradient(&inputs).await?;
    println!(
        "df/dx = {:?}",
        partials[&("f".to_string(), "x".to_string())]
    );

    Ok(())
}
