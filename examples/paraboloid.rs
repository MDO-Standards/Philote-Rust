use async_trait::async_trait;
use ndarray::ArrayD;
use std::collections::HashMap;

use philote::{
    philote_info::{VariableMetaData, VariableType},
    server::ExplicitServer,
    traits::{Discipline, ExplicitDiscipline},
    ArrayMap, PartialMap, PhiloteError, Result,
};

pub struct Paraboloid {
    variables: Vec<VariableMetaData>,
    partials: Vec<(String, String)>,
    options: HashMap<String, String>,
}

impl Paraboloid {
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            partials: Vec::new(),
            options: HashMap::new(),
        }
    }
}

impl Discipline for Paraboloid {
    fn name(&self) -> &str {
        "Paraboloid"
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
        // For this example, we'll use default values
        Ok(())
    }

    fn setup(&mut self) -> Result<()> {
        // Clear existing variables
        self.variables.clear();

        // Add inputs
        self.add_input("x", &[1], "")?;
        self.add_input("y", &[1], "")?;

        // Add output
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
impl ExplicitDiscipline for Paraboloid {
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
    // Create the discipline
    let mut paraboloid = Paraboloid::new();
    paraboloid.initialize()?;
    paraboloid.setup()?;
    paraboloid.setup_partials()?;

    // Create the server
    let server = ExplicitServer::new(paraboloid).with_verbose(true);

    println!("✅ Paraboloid discipline server created successfully!");
    println!("🚀 The Rust port of Philote-MDO is working!");

    // Test the discipline directly
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), ArrayD::from_elem(vec![1], 2.0));
    inputs.insert("y".to_string(), ArrayD::from_elem(vec![1], -1.0));

    let discipline = server.discipline().read().await;
    let outputs = discipline.compute(&inputs).await?;
    let partials = discipline.compute_partials(&inputs).await?;

    println!("\nTest computation:");
    println!("Inputs: x = {}, y = {}", inputs["x"][[0]], inputs["y"][[0]]);
    println!("Output: f = {}", outputs["f"][[0]]);
    println!(
        "Partials: df/dx = {}, df/dy = {}",
        partials[&("f".to_string(), "x".to_string())][[0]],
        partials[&("f".to_string(), "y".to_string())][[0]]
    );

    Ok(())
}
