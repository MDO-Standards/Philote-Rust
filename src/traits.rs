//! Core trait definitions for computational disciplines
//!
//! This module defines the fundamental traits that all Philote disciplines must implement.
//! Disciplines represent computational analysis components in MDO frameworks.
//!
//! # Trait Hierarchy
//!
//! ```text
//! Discipline (base trait)
//!     ├── ExplicitDiscipline (for direct input-output mappings)
//!     └── ImplicitDiscipline (for residual-based formulations)
//! ```
//!
//! # Discipline Types
//!
//! ## Explicit Disciplines
//!
//! Explicit disciplines compute outputs directly from inputs: `y = f(x)`.
//! They implement the [`ExplicitDiscipline`] trait and provide a [`compute`] method.
//!
//! **Example:**
//! ```rust
//! use async_trait::async_trait;
//! use philote::{traits::{Discipline, ExplicitDiscipline}, ArrayMap, Result};
//! use std::collections::HashMap;
//! # use philote::philote_info::VariableMetaData;
//!
//! struct SimpleAnalysis;
//!
//! #[async_trait]
//! impl ExplicitDiscipline for SimpleAnalysis {
//!     async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
//!         // Compute outputs from inputs
//!         Ok(HashMap::new())
//!     }
//! }
//!
//! impl Discipline for SimpleAnalysis {
//!     fn name(&self) -> &str { "SimpleAnalysis" }
//!     // ... implement other required methods
//! #   fn add_input(&mut self, _: &str, _: &[usize], _: &str) -> Result<()> { Ok(()) }
//! #   fn add_output(&mut self, _: &str, _: &[usize], _: &str) -> Result<()> { Ok(()) }
//! #   fn add_option(&mut self, _: &str, _: &str) -> Result<()> { Ok(()) }
//! #   fn set_options(&mut self, _: &HashMap<String, serde_json::Value>) -> Result<()> { Ok(()) }
//! #   fn setup(&mut self) -> Result<()> { Ok(()) }
//! #   fn declare_partials(&mut self, _: &str, _: &str) -> Result<()> { Ok(()) }
//! #   fn get_variable_definitions(&self) -> Result<Vec<VariableMetaData>> { Ok(vec![]) }
//! #   fn get_partials_definitions(&self) -> Result<Vec<(String, String)>> { Ok(vec![]) }
//! #   fn get_available_options(&self) -> Result<HashMap<String, String>> { Ok(HashMap::new()) }
//! }
//! ```
//!
//! ## Implicit Disciplines
//!
//! Implicit disciplines solve for outputs that satisfy residual equations: `R(x, y) = 0`.
//! They implement the [`ImplicitDiscipline`] trait with methods for computing and solving residuals.
//!
//! # Lifecycle
//!
//! Disciplines follow a specific initialization lifecycle:
//!
//! 1. [`initialize`] - Set up options and initial configuration
//! 2. [`setup`] - Define inputs and outputs
//! 3. [`setup_partials`] - Declare partial derivatives (if applicable)
//! 4. Ready for computation
//!
//! [`compute`]: ExplicitDiscipline::compute
//! [`initialize`]: Discipline::initialize
//! [`setup`]: Discipline::setup
//! [`setup_partials`]: Discipline::setup_partials

use async_trait::async_trait;
use std::collections::HashMap;

use crate::philote_info::{DisciplineProperties, VariableMetaData, VariableType};
use crate::{ArrayMap, DiscreteMap, PartialMap, PhiloteError, Result};

/// Base trait for all computational disciplines
///
/// This trait defines the common interface that all disciplines must implement,
/// regardless of whether they are explicit or implicit. It handles discipline
/// metadata, variable definitions, and lifecycle management.
///
/// # Required Methods
///
/// Implementors must provide:
/// - Variable management: [`add_input`], [`add_output`]
/// - Configuration: [`add_option`], [`set_options`]
/// - Initialization: [`setup`]
/// - Metadata: [`get_variable_definitions`], [`get_partials_definitions`], [`get_available_options`]
/// - Partial derivatives: [`declare_partials`]
///
/// # Provided Methods
///
/// Default implementations are provided for:
/// - Identity methods: [`name`], [`version`]
/// - Property flags: [`is_continuous`], [`is_differentiable`], [`provides_gradients`]
/// - Optional lifecycle: [`initialize`], [`setup_partials`]
///
/// [`add_input`]: Discipline::add_input
/// [`add_output`]: Discipline::add_output
/// [`add_option`]: Discipline::add_option
/// [`set_options`]: Discipline::set_options
/// [`setup`]: Discipline::setup
/// [`get_variable_definitions`]: Discipline::get_variable_definitions
/// [`get_partials_definitions`]: Discipline::get_partials_definitions
/// [`get_available_options`]: Discipline::get_available_options
/// [`declare_partials`]: Discipline::declare_partials
/// [`name`]: Discipline::name
/// [`version`]: Discipline::version
/// [`is_continuous`]: Discipline::is_continuous
/// [`is_differentiable`]: Discipline::is_differentiable
/// [`provides_gradients`]: Discipline::provides_gradients
/// [`initialize`]: Discipline::initialize
/// [`setup_partials`]: Discipline::setup_partials
pub trait Discipline: Send + Sync {
    /// Returns the name of this discipline
    fn name(&self) -> &str {
        "UnnamedDiscipline"
    }

    /// Returns the version string of this discipline
    fn version(&self) -> &str {
        "0.1.0"
    }

    /// Returns whether this discipline's outputs are continuous functions of inputs
    fn is_continuous(&self) -> bool {
        true
    }

    /// Returns whether this discipline is differentiable
    fn is_differentiable(&self) -> bool {
        false
    }

    /// Returns whether this discipline provides analytical gradients
    fn provides_gradients(&self) -> bool {
        false
    }

    /// Initialize the discipline with options and configuration
    ///
    /// Called before `setup()` to perform any initial configuration.
    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    /// Add an input variable to this discipline
    ///
    /// # Arguments
    ///
    /// * `name` - Variable name
    /// * `shape` - Array dimensions
    /// * `units` - Physical units (empty string if dimensionless)
    fn add_input(&mut self, name: &str, shape: &[usize], units: &str) -> Result<()>;

    /// Add an output variable to this discipline
    ///
    /// # Arguments
    ///
    /// * `name` - Variable name
    /// * `shape` - Array dimensions
    /// * `units` - Physical units (empty string if dimensionless)
    fn add_output(&mut self, name: &str, shape: &[usize], units: &str) -> Result<()>;

    /// Add a configuration option to this discipline
    ///
    /// # Arguments
    ///
    /// * `name` - Option name
    /// * `option_type` - Type descriptor (e.g., "double", "int", "string")
    fn add_option(&mut self, name: &str, option_type: &str) -> Result<()>;

    /// Set configuration options from a JSON value map
    fn set_options(&mut self, options: &HashMap<String, serde_json::Value>) -> Result<()>;

    /// Set up the discipline by defining all inputs and outputs
    ///
    /// This is called after `initialize()` and should configure all variables.
    fn setup(&mut self) -> Result<()>;

    /// Set up partial derivative declarations
    ///
    /// Called after `setup()` for disciplines that provide gradients.
    fn setup_partials(&mut self) -> Result<()> {
        Ok(())
    }

    fn configure(&mut self) -> Result<()> {
        Ok(())
    }

    fn add_discrete_input(&mut self, _name: &str) -> Result<()> {
        Ok(())
    }

    fn add_discrete_output(&mut self, _name: &str) -> Result<()> {
        Ok(())
    }

    fn get_discrete_variable_definitions(&self) -> Result<Vec<VariableMetaData>> {
        Ok(vec![])
    }

    /// Declare a partial derivative of an output with respect to an input
    ///
    /// # Arguments
    ///
    /// * `func` - Output variable name
    /// * `var` - Input variable name
    fn declare_partials(&mut self, func: &str, var: &str) -> Result<()>;

    /// Get metadata for all variables (inputs, outputs, residuals)
    fn get_variable_definitions(&self) -> Result<Vec<VariableMetaData>>;

    /// Get a list of all declared partial derivatives as (output, input) pairs
    fn get_partials_definitions(&self) -> Result<Vec<(String, String)>>;

    /// Get the discipline properties for client introspection
    fn get_properties(&self) -> DisciplineProperties {
        DisciplineProperties {
            continuous: self.is_continuous(),
            differentiable: self.is_differentiable(),
            provides_gradients: self.provides_gradients(),
            name: self.name().to_string(),
            version: self.version().to_string(),
        }
    }

    /// Get available configuration options as a map of name to type
    fn get_available_options(&self) -> Result<HashMap<String, String>>;
}

/// Trait for explicit disciplines with direct input-output mappings
///
/// Explicit disciplines compute outputs directly from inputs: `y = f(x)`.
/// The computation is performed asynchronously to support long-running analyses.
///
/// # Required Methods
///
/// * [`compute`] - Calculate outputs from inputs
///
/// # Optional Methods
///
/// * [`compute_partials`] - Calculate partial derivatives (gradients)
///
/// [`compute`]: ExplicitDiscipline::compute
/// [`compute_partials`]: ExplicitDiscipline::compute_partials
#[async_trait]
pub trait ExplicitDiscipline: Discipline {
    /// Compute outputs from inputs
    ///
    /// # Arguments
    ///
    /// * `inputs` - Map of input variable names to their array values
    ///
    /// # Returns
    ///
    /// Map of output variable names to their computed array values
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap>;

    /// Compute partial derivatives of outputs with respect to inputs
    ///
    /// # Arguments
    ///
    /// * `inputs` - Map of input variable names to their array values
    ///
    /// # Returns
    ///
    /// Map of (output, input) tuples to their partial derivative arrays
    ///
    /// # Default Implementation
    ///
    /// Returns a "not implemented" error. Override to provide analytical gradients.
    async fn compute_partials(&self, _inputs: &ArrayMap) -> Result<PartialMap> {
        Err(PhiloteError::not_implemented("compute_partials"))
    }

    async fn compute_with_discrete(
        &self,
        inputs: &ArrayMap,
        _discrete_inputs: &DiscreteMap,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        let outputs = self.compute(inputs).await?;
        Ok((outputs, std::collections::HashMap::new()))
    }

    async fn compute_partials_with_discrete(
        &self,
        inputs: &ArrayMap,
        _discrete_inputs: &DiscreteMap,
    ) -> Result<PartialMap> {
        self.compute_partials(inputs).await
    }
}

/// Trait for implicit disciplines based on residual equations
///
/// Implicit disciplines solve for outputs that satisfy residual equations: `R(x, y) = 0`.
/// They require iterative solution techniques and are used for coupled systems.
///
/// # Required Methods
///
/// * [`compute_residuals`] - Evaluate residual equations
/// * [`solve_residuals`] - Solve for outputs that satisfy residuals
///
/// # Optional Methods
///
/// * [`residual_partials`] - Compute derivatives of residuals
/// * [`apply_linear`] - Apply linear operator for derivative computations
///
/// [`compute_residuals`]: ImplicitDiscipline::compute_residuals
/// [`solve_residuals`]: ImplicitDiscipline::solve_residuals
/// [`residual_partials`]: ImplicitDiscipline::residual_partials
/// [`apply_linear`]: ImplicitDiscipline::apply_linear
#[async_trait]
pub trait ImplicitDiscipline: Discipline {
    /// Compute residuals given inputs and proposed outputs
    ///
    /// # Arguments
    ///
    /// * `inputs` - Map of input variable names to values
    /// * `outputs` - Map of output variable names to proposed values
    ///
    /// # Returns
    ///
    /// Map of residual variable names to their computed values
    async fn compute_residuals(&self, inputs: &ArrayMap, outputs: &ArrayMap) -> Result<ArrayMap>;

    /// Solve for outputs that make residuals zero
    ///
    /// # Arguments
    ///
    /// * `inputs` - Map of input variable names to values
    ///
    /// # Returns
    ///
    /// Map of output variable names to solved values
    async fn solve_residuals(&self, inputs: &ArrayMap) -> Result<ArrayMap>;

    /// Compute partial derivatives of residuals
    ///
    /// # Arguments
    ///
    /// * `inputs` - Map of input variable names to values
    /// * `outputs` - Map of output variable names to values
    ///
    /// # Returns
    ///
    /// Map of (residual, variable) tuples to their partial derivative arrays
    ///
    /// # Default Implementation
    ///
    /// Returns a "not implemented" error. Override to provide analytical gradients.
    async fn residual_partials(
        &self,
        _inputs: &ArrayMap,
        _outputs: &ArrayMap,
    ) -> Result<PartialMap> {
        Err(PhiloteError::not_implemented("residual_partials"))
    }

    /// Apply linear operator for adjoint or forward derivative computation
    ///
    /// # Arguments
    ///
    /// * `inputs` - Map of input variable names to values
    /// * `outputs` - Map of output variable names to values
    /// * `mode` - Either "fwd" (forward) or "rev" (reverse/adjoint)
    ///
    /// # Returns
    ///
    /// Result of applying the linear operator
    ///
    /// # Default Implementation
    ///
    /// Returns a "not implemented" error. Override to support linear operators.
    async fn apply_linear(
        &self,
        _inputs: &ArrayMap,
        _outputs: &ArrayMap,
        _mode: &str,
    ) -> Result<ArrayMap> {
        Err(PhiloteError::not_implemented("apply_linear"))
    }

    async fn compute_residuals_with_discrete(
        &self,
        inputs: &ArrayMap,
        outputs: &ArrayMap,
        _discrete_inputs: &DiscreteMap,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        let residuals = self.compute_residuals(inputs, outputs).await?;
        Ok((residuals, std::collections::HashMap::new()))
    }

    async fn solve_residuals_with_discrete(
        &self,
        inputs: &ArrayMap,
        _discrete_inputs: &DiscreteMap,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        let outputs = self.solve_residuals(inputs).await?;
        Ok((outputs, std::collections::HashMap::new()))
    }

    async fn residual_partials_with_discrete(
        &self,
        inputs: &ArrayMap,
        outputs: &ArrayMap,
        _discrete_inputs: &DiscreteMap,
    ) -> Result<PartialMap> {
        self.residual_partials(inputs, outputs).await
    }
}

/// Helper struct for variable metadata
///
/// Provides a convenient way to build variable definitions with type safety.
#[derive(Debug, Clone)]
pub struct VariableInfo {
    /// Variable name
    pub name: String,
    /// Variable type (input, output, or residual)
    pub var_type: VariableType,
    /// Array shape/dimensions
    pub shape: Vec<usize>,
    /// Physical units
    pub units: String,
    pub dynamic_shape: bool,
}

impl VariableInfo {
    /// Create a new variable with the specified properties
    pub fn new(name: String, var_type: VariableType, shape: Vec<usize>, units: String) -> Self {
        Self {
            name,
            var_type,
            shape,
            units,
            dynamic_shape: false,
        }
    }

    /// Create an input variable
    pub fn input(name: String, shape: Vec<usize>, units: String) -> Self {
        Self::new(name, VariableType::KInput, shape, units)
    }

    /// Create an output variable
    pub fn output(name: String, shape: Vec<usize>, units: String) -> Self {
        Self::new(name, VariableType::KOutput, shape, units)
    }

    /// Create a residual variable
    pub fn residual(name: String, shape: Vec<usize>, units: String) -> Self {
        Self::new(name, VariableType::KResidual, shape, units)
    }

    /// Calculate the total size (number of elements) of this variable
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
            dynamic_shape: info.dynamic_shape,
        }
    }
}
