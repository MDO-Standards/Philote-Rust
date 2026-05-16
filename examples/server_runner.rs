use async_trait::async_trait;
use ndarray::ArrayD;
use std::collections::HashMap;
use std::net::SocketAddr;
use tonic::transport::Server;

use philote_mdo::{
    philote_info::{
        discipline_service_server::DisciplineServiceServer,
        explicit_service_server::ExplicitServiceServer, VariableMetaData, VariableType,
    },
    server::ExplicitServer,
    traits::{Discipline, ExplicitDiscipline},
    ArrayMap, PartialMap, PhiloteError, Result,
};

// Simple Paraboloid discipline for server example
pub struct ParaboloidDiscipline {
    variables: Vec<VariableMetaData>,
    partials: Vec<(String, String)>,
    options: HashMap<String, String>,
}

impl ParaboloidDiscipline {
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            partials: Vec::new(),
            options: HashMap::new(),
        }
    }
}

impl Discipline for ParaboloidDiscipline {
    fn name(&self) -> &str {
        "ParaboloidServer"
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

    fn initialize(&mut self) -> Result<()> {
        self.add_option("a", "double")?;
        self.add_option("b", "double")?;
        Ok(())
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

#[async_trait]
impl ExplicitDiscipline for ParaboloidDiscipline {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let x = inputs
            .get("x")
            .ok_or_else(|| PhiloteError::VariableNotFound("x".to_string()))?;

        let y = inputs
            .get("y")
            .ok_or_else(|| PhiloteError::VariableNotFound("y".to_string()))?;

        if x.len() != 1 || y.len() != 1 {
            return Err(PhiloteError::array_error("Expected scalar inputs"));
        }

        let x_val = x[[0]];
        let y_val = y[[0]];

        // f = (x - 3)^2 + x*y + (y + 4)^2 - 3
        let f_val = (x_val - 3.0).powi(2) + x_val * y_val + (y_val + 4.0).powi(2) - 3.0;

        println!("🧮 Computing: x={}, y={} => f={}", x_val, y_val, f_val);

        let mut outputs = HashMap::new();
        let f_array = ArrayD::from_elem(vec![1], f_val);
        outputs.insert("f".to_string(), f_array);

        Ok(outputs)
    }

    async fn compute_partials(&self, inputs: &ArrayMap) -> Result<PartialMap> {
        let x = inputs
            .get("x")
            .ok_or_else(|| PhiloteError::VariableNotFound("x".to_string()))?;

        let y = inputs
            .get("y")
            .ok_or_else(|| PhiloteError::VariableNotFound("y".to_string()))?;

        if x.len() != 1 || y.len() != 1 {
            return Err(PhiloteError::array_error("Expected scalar inputs"));
        }

        let x_val = x[[0]];
        let y_val = y[[0]];

        // df/dx = 2*(x - 3) + y
        let df_dx = 2.0 * (x_val - 3.0) + y_val;

        // df/dy = x + 2*(y + 4)
        let df_dy = x_val + 2.0 * (y_val + 4.0);

        println!("📊 Computing gradients: df/dx={}, df/dy={}", df_dx, df_dy);

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

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Starting Philote Rust Server Example");

    // Create and initialize the discipline
    let mut discipline = ParaboloidDiscipline::new();
    discipline.initialize()?;
    discipline.setup()?;
    discipline.setup_partials()?;

    // Create the server
    let server = ExplicitServer::new(discipline).with_verbose(true);

    // Define the address
    let addr: SocketAddr = "127.0.0.1:50051"
        .parse()
        .map_err(|e| PhiloteError::config_error(format!("Invalid address: {}", e)))?;

    println!("🌐 Server listening on: {}", addr);
    println!("📡 Ready to accept client connections!");
    println!("💡 You can now run the client_example to test the connection.");

    // Start the gRPC server with both ExplicitService and DisciplineService
    // We need to use std::sync::Arc to share ownership between the two services
    let server_arc = std::sync::Arc::new(server);

    Server::builder()
        .add_service(ExplicitServiceServer::from_arc(server_arc.clone()))
        .add_service(DisciplineServiceServer::from_arc(server_arc))
        .serve(addr)
        .await
        .map_err(|e| PhiloteError::config_error(format!("Server failed: {}", e)))?;

    Ok(())
}
