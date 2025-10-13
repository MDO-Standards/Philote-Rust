use async_trait::async_trait;
use std::collections::HashMap;

use crate::philote_info::{DisciplineProperties, VariableMetaData, VariableType};
use crate::{ArrayMap, PartialMap, PhiloteError, Result};

pub trait Discipline: Send + Sync {
    fn name(&self) -> &str {
        "UnnamedDiscipline"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn is_continuous(&self) -> bool {
        true
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn provides_gradients(&self) -> bool {
        false
    }

    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    fn add_input(&mut self, name: &str, shape: &[usize], units: &str) -> Result<()>;

    fn add_output(&mut self, name: &str, shape: &[usize], units: &str) -> Result<()>;

    fn add_option(&mut self, name: &str, option_type: &str) -> Result<()>;

    fn set_options(&mut self, options: &HashMap<String, serde_json::Value>) -> Result<()>;

    fn setup(&mut self) -> Result<()>;

    fn setup_partials(&mut self) -> Result<()> {
        Ok(())
    }

    fn declare_partials(&mut self, func: &str, var: &str) -> Result<()>;

    fn get_variable_definitions(&self) -> Result<Vec<VariableMetaData>>;

    fn get_partials_definitions(&self) -> Result<Vec<(String, String)>>;

    fn get_properties(&self) -> DisciplineProperties {
        DisciplineProperties {
            continuous: self.is_continuous(),
            differentiable: self.is_differentiable(),
            provides_gradients: self.provides_gradients(),
            name: self.name().to_string(),
            version: self.version().to_string(),
        }
    }

    fn get_available_options(&self) -> Result<HashMap<String, String>>;
}

#[async_trait]
pub trait ExplicitDiscipline: Discipline {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap>;

    async fn compute_partials(&self, _inputs: &ArrayMap) -> Result<PartialMap> {
        Err(PhiloteError::not_implemented("compute_partials"))
    }
}

#[async_trait]
pub trait ImplicitDiscipline: Discipline {
    async fn compute_residuals(&self, inputs: &ArrayMap, outputs: &ArrayMap) -> Result<ArrayMap>;

    async fn solve_residuals(&self, inputs: &ArrayMap) -> Result<ArrayMap>;

    async fn residual_partials(
        &self,
        _inputs: &ArrayMap,
        _outputs: &ArrayMap,
    ) -> Result<PartialMap> {
        Err(PhiloteError::not_implemented("residual_partials"))
    }

    async fn apply_linear(
        &self,
        _inputs: &ArrayMap,
        _outputs: &ArrayMap,
        _mode: &str,
    ) -> Result<ArrayMap> {
        Err(PhiloteError::not_implemented("apply_linear"))
    }
}

#[derive(Debug, Clone)]
pub struct VariableInfo {
    pub name: String,
    pub var_type: VariableType,
    pub shape: Vec<usize>,
    pub units: String,
}

impl VariableInfo {
    pub fn new(name: String, var_type: VariableType, shape: Vec<usize>, units: String) -> Self {
        Self {
            name,
            var_type,
            shape,
            units,
        }
    }

    pub fn input(name: String, shape: Vec<usize>, units: String) -> Self {
        Self::new(name, VariableType::KInput, shape, units)
    }

    pub fn output(name: String, shape: Vec<usize>, units: String) -> Self {
        Self::new(name, VariableType::KOutput, shape, units)
    }

    pub fn residual(name: String, shape: Vec<usize>, units: String) -> Self {
        Self::new(name, VariableType::KResidual, shape, units)
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

impl From<VariableInfo> for VariableMetaData {
    fn from(info: VariableInfo) -> Self {
        VariableMetaData {
            r#type: info.var_type.into(),
            name: info.name,
            shape: info.shape.into_iter().map(|s| s as i64).collect(),
            units: info.units,
        }
    }
}
