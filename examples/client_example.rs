use ndarray::ArrayD;
use std::collections::HashMap;

use philote_mdo::{client::ExplicitClient, types::StreamOptions, ArrayMap, PhiloteError, Result};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Philote Rust Client Example");

    // This example demonstrates how to use the ExplicitClient
    // Note: This would connect to a running Philote server

    let server_address = "http://localhost:50051";
    println!("📡 Connecting to server at: {}", server_address);

    // Try to connect to the server
    let mut client = match ExplicitClient::connect(server_address).await {
        Ok(client) => {
            println!("✅ Connected successfully!");
            client
        }
        Err(e) => {
            println!("❌ Failed to connect to server: {}", e);
            println!("💡 To run this example, you need a Philote server running.");
            println!("   You can start one using the paraboloid example as a server.");
            return Err(e);
        }
    };

    // Configure streaming options
    let stream_options = StreamOptions {
        max_double_per_slice: 1000,
    };
    client = client.with_stream_options(stream_options);

    println!("⚙️ Getting discipline information...");

    // Get discipline properties
    match client.get_info().await {
        Ok(properties) => {
            println!("📋 Discipline Properties:");
            println!("   - Name: {}", properties.name);
            println!("   - Version: {}", properties.version);
            println!("   - Continuous: {}", properties.continuous);
            println!("   - Differentiable: {}", properties.differentiable);
            println!("   - Provides Gradients: {}", properties.provides_gradients);
        }
        Err(e) => {
            println!("❌ Failed to get discipline info: {}", e);
            return Err(e);
        }
    }

    // Get available options
    match client.get_available_options().await {
        Ok(options) => {
            println!("🔧 Available Options:");
            for (name, type_str) in options {
                println!("   - {}: {}", name, type_str);
            }
        }
        Err(e) => {
            println!("❌ Failed to get available options: {}", e);
        }
    }

    // Setup the discipline
    println!("🔄 Setting up discipline...");
    if let Err(e) = client.setup().await {
        println!("❌ Failed to setup discipline: {}", e);
        return Err(e);
    }

    // Get variable definitions
    match client.get_variable_definitions().await {
        Ok(variables) => {
            println!("📊 Variable Definitions:");
            for var in variables {
                println!("   - {}: {:?} ({})", var.name, var.shape, var.units);
            }
        }
        Err(e) => {
            println!("❌ Failed to get variable definitions: {}", e);
        }
    }

    // Get partial definitions (if available)
    match client.get_partial_definitions().await {
        Ok(partials) => {
            println!("📈 Partial Definitions:");
            for partial in partials {
                println!(
                    "   - {}/{}: {:?}",
                    partial.name, partial.subname, partial.shape
                );
            }
        }
        Err(e) => {
            println!("❌ Failed to get partial definitions: {}", e);
        }
    }

    // Create input arrays for computation
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), ArrayD::from_elem(vec![1], 2.0));
    inputs.insert("y".to_string(), ArrayD::from_elem(vec![1], -1.0));

    println!("🧮 Computing function with inputs: x=2.0, y=-1.0");

    // Call compute function
    match client.compute_function(&inputs).await {
        Ok(outputs) => {
            println!("✅ Computation successful!");
            for (name, array) in outputs {
                println!("   Output {}: {}", name, array[[0]]);
            }

            // Also compute gradients
            println!("📊 Computing gradients...");
            match client.compute_gradient(&inputs).await {
                Ok(partials) => {
                    println!("✅ Gradient computation successful!");
                    for ((func, var), array) in partials {
                        println!("   ∂{}/∂{}: {}", func, var, array[[0]]);
                    }
                }
                Err(e) => {
                    println!("❌ Failed to compute gradients: {}", e);
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to compute function: {}", e);
            return Err(e);
        }
    }

    println!("🎉 Client example completed successfully!");

    Ok(())
}

// Example of how to start a server for testing
pub async fn start_test_server() -> Result<()> {
    use philote_mdo::{
        philote_info::{VariableMetaData, VariableType},
        server::ExplicitServer,
        traits::{Discipline, ExplicitDiscipline},
    };
    use std::net::SocketAddr;
    use tonic::transport::Server;

    // This is a simplified version of our paraboloid discipline
    struct TestParaboloid {
        variables: Vec<VariableMetaData>,
        partials: Vec<(String, String)>,
        options: HashMap<String, String>,
    }

    impl TestParaboloid {
        fn new() -> Self {
            Self {
                variables: Vec::new(),
                partials: Vec::new(),
                options: HashMap::new(),
            }
        }
    }

    impl Discipline for TestParaboloid {
        fn name(&self) -> &str {
            "TestParaboloid"
        }
        fn version(&self) -> &str {
            "1.0.0"
        }
        fn is_continuous(&self) -> bool {
            true
        }
        fn is_differentiable(&self) -> bool {
            true
        }
        fn provides_gradients(&self) -> bool {
            true
        }

        fn add_input(&mut self, name: &str, shape: &[usize], units: &str) -> Result<()> {
            let var_meta = VariableMetaData {
                r#type: VariableType::KInput as i32,
                name: name.to_string(),
                shape: shape.iter().map(|&s| s as i64).collect(),
                units: units.to_string(),
                dynamic_shape: false,
            };
            self.variables.push(var_meta);
            Ok(())
        }

        fn add_output(&mut self, name: &str, shape: &[usize], units: &str) -> Result<()> {
            let var_meta = VariableMetaData {
                r#type: VariableType::KOutput as i32,
                name: name.to_string(),
                shape: shape.iter().map(|&s| s as i64).collect(),
                units: units.to_string(),
                dynamic_shape: false,
            };
            self.variables.push(var_meta);
            Ok(())
        }

        fn add_option(&mut self, name: &str, option_type: &str) -> Result<()> {
            self.options
                .insert(name.to_string(), option_type.to_string());
            Ok(())
        }

        fn set_options(&mut self, _options: &HashMap<String, serde_json::Value>) -> Result<()> {
            Ok(())
        }

        fn setup(&mut self) -> Result<()> {
            self.variables.clear();
            self.add_input("x", &[1], "")?;
            self.add_input("y", &[1], "")?;
            self.add_output("f", &[1], "")?;
            Ok(())
        }

        fn declare_partials(&mut self, func: &str, var: &str) -> Result<()> {
            self.partials.push((func.to_string(), var.to_string()));
            Ok(())
        }

        fn setup_partials(&mut self) -> Result<()> {
            self.declare_partials("f", "x")?;
            self.declare_partials("f", "y")?;
            Ok(())
        }

        fn get_variable_definitions(&self) -> Result<Vec<VariableMetaData>> {
            Ok(self.variables.clone())
        }

        fn get_partials_definitions(&self) -> Result<Vec<(String, String)>> {
            Ok(self.partials.clone())
        }

        fn get_available_options(&self) -> Result<HashMap<String, String>> {
            Ok(self.options.clone())
        }
    }

    #[async_trait::async_trait]
    impl ExplicitDiscipline for TestParaboloid {
        async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
            let x = inputs
                .get("x")
                .ok_or_else(|| PhiloteError::VariableNotFound("x".to_string()))?;
            let y = inputs
                .get("y")
                .ok_or_else(|| PhiloteError::VariableNotFound("y".to_string()))?;

            let x_val = x[[0]];
            let y_val = y[[0]];
            let f_val = (x_val - 3.0).powi(2) + x_val * y_val + (y_val + 4.0).powi(2) - 3.0;

            let mut outputs = HashMap::new();
            outputs.insert("f".to_string(), ArrayD::from_elem(vec![1], f_val));
            Ok(outputs)
        }

        async fn compute_partials(&self, inputs: &ArrayMap) -> Result<philote_mdo::PartialMap> {
            let x = inputs
                .get("x")
                .ok_or_else(|| PhiloteError::VariableNotFound("x".to_string()))?;
            let y = inputs
                .get("y")
                .ok_or_else(|| PhiloteError::VariableNotFound("y".to_string()))?;

            let x_val = x[[0]];
            let y_val = y[[0]];

            let df_dx = 2.0 * (x_val - 3.0) + y_val;
            let df_dy = x_val + 2.0 * (y_val + 4.0);

            let mut partials = HashMap::new();
            partials.insert(
                ("f".to_string(), "x".to_string()),
                ArrayD::from_elem(vec![1], df_dx),
            );
            partials.insert(
                ("f".to_string(), "y".to_string()),
                ArrayD::from_elem(vec![1], df_dy),
            );
            Ok(partials)
        }
    }

    let mut discipline = TestParaboloid::new();
    discipline.setup()?;
    discipline.setup_partials()?;

    let server_impl = ExplicitServer::new(discipline).with_verbose(true);

    let addr: SocketAddr = "127.0.0.1:50051"
        .parse()
        .map_err(|e| PhiloteError::config_error(format!("Invalid address: {}", e)))?;

    println!("🚀 Starting test server on {}", addr);

    Server::builder()
        .add_service(
            philote_mdo::philote_info::explicit_service_server::ExplicitServiceServer::new(
                server_impl,
            ),
        )
        .serve(addr)
        .await
        .map_err(|e| PhiloteError::config_error(format!("Server error: {}", e)))?;

    Ok(())
}
