//! Core trait definitions for computational disciplines
//!
//! This module defines the traits that all Philote disciplines implement.
//! Disciplines represent computational analysis components in MDO frameworks.
//!
//! # Trait hierarchy
//!
//! ```text
//! Discipline (base trait)
//!     ├── ExplicitDiscipline (for direct input-output mappings)
//!     └── ImplicitDiscipline (for residual-based formulations)
//! ```
//!
//! # Implementing a discipline
//!
//! [`Discipline`] stores its metadata in a [`VariableRegistry`], so an implementor
//! only supplies the two accessors (via the [`impl_registry!`](crate::impl_registry)
//! macro) plus whatever lifecycle hooks it needs. Everything else — declaring
//! variables, resolving dynamic shapes, reporting definitions — is provided.
//!
//! ```rust
//! use async_trait::async_trait;
//! use ndarray::ArrayD;
//! use philote_mdo::{
//!     impl_registry, registry::VariableRegistry,
//!     traits::{Discipline, ExplicitDiscipline},
//!     ArrayMap, Result,
//! };
//! use std::collections::HashMap;
//!
//! #[derive(Default)]
//! struct Doubler {
//!     registry: VariableRegistry,
//! }
//!
//! impl Discipline for Doubler {
//!     impl_registry!(registry);
//!
//!     fn name(&self) -> &str { "Doubler" }
//!
//!     fn setup(&mut self) -> Result<()> {
//!         self.add_input("x", &[1], "")?;
//!         self.add_output("y", &[1], "")
//!     }
//! }
//!
//! #[async_trait]
//! impl ExplicitDiscipline for Doubler {
//!     async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
//!         let mut outputs = HashMap::new();
//!         outputs.insert("y".to_string(), &inputs["x"] * 2.0);
//!         Ok(outputs)
//!     }
//! }
//! ```
//!
//! # Lifecycle
//!
//! 1. [`initialize`](Discipline::initialize) — declare available options
//! 2. [`set_options`](Discipline::set_options) — receive option values from the client
//! 3. [`configure`](Discipline::configure), then [`setup`](Discipline::setup) —
//!    declare inputs and outputs
//! 4. [`setup_partials`](Discipline::setup_partials) — declare partial derivatives
//! 5. Ready for computation

use async_trait::async_trait;
use std::collections::HashMap;

use crate::philote_info::{DisciplineProperties, VariableMetaData, VariableType};
use crate::registry::VariableRegistry;
use crate::{ArrayMap, DiscreteMap, PartialMap, PhiloteError, Result};

/// Direction for [`ImplicitDiscipline::apply_linear`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearMode {
    /// Forward mode: `d_residuals += J * [d_inputs; d_outputs]`
    Fwd,
    /// Reverse (adjoint) mode: `[d_inputs; d_outputs] += Jᵀ * d_residuals`
    Rev,
}

/// Base trait for all computational disciplines.
///
/// Implementors must provide [`registry`](Self::registry) and
/// [`registry_mut`](Self::registry_mut) — use [`impl_registry!`](crate::impl_registry).
/// Every other method has a default implementation.
pub trait Discipline: Send + Sync {
    /// Immutable access to this discipline's metadata store.
    fn registry(&self) -> &VariableRegistry;

    /// Mutable access to this discipline's metadata store.
    fn registry_mut(&mut self) -> &mut VariableRegistry;

    /// The name of this discipline.
    fn name(&self) -> &str {
        "UnnamedDiscipline"
    }

    /// The version string of this discipline.
    fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }

    /// Whether outputs are continuous functions of the inputs.
    fn is_continuous(&self) -> bool {
        true
    }

    /// Whether this discipline is differentiable.
    fn is_differentiable(&self) -> bool {
        false
    }

    /// Whether this discipline provides analytical gradients.
    fn provides_gradients(&self) -> bool {
        false
    }

    /// Declare available options. Called once when the discipline is constructed.
    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    /// Apply option values received from the client.
    fn set_options(&mut self, _options: &HashMap<String, serde_json::Value>) -> Result<()> {
        Ok(())
    }

    /// Hook that runs immediately before [`setup`](Self::setup).
    fn configure(&mut self) -> Result<()> {
        Ok(())
    }

    /// Declare all inputs and outputs.
    ///
    /// Required rather than defaulted: a discipline that declares no variables
    /// would otherwise compile and then fail at the first compute call with a
    /// confusing "variable not found".
    fn setup(&mut self) -> Result<()>;

    /// Declare partial derivatives. Runs after [`setup`](Self::setup).
    fn setup_partials(&mut self) -> Result<()> {
        Ok(())
    }

    /// Declare a continuous input with a fixed shape.
    fn add_input(&mut self, name: &str, shape: &[usize], units: &str) -> Result<()> {
        self.registry_mut().add_input(name, shape, units)
    }

    /// Declare a continuous input whose shape the client sets at runtime.
    fn add_dynamic_input(&mut self, name: &str, units: &str) -> Result<()> {
        self.registry_mut().add_dynamic_input(name, units)
    }

    /// Declare a continuous output with a fixed shape.
    ///
    /// For an implicit discipline this also records a matching residual.
    fn add_output(&mut self, name: &str, shape: &[usize], units: &str) -> Result<()> {
        self.registry_mut().add_output(name, shape, units)
    }

    /// Declare a continuous output whose shape the client sets at runtime.
    fn add_dynamic_output(&mut self, name: &str, units: &str) -> Result<()> {
        self.registry_mut().add_dynamic_output(name, units)
    }

    /// Declare a discrete input, optionally with a default value.
    fn add_discrete_input(
        &mut self,
        name: &str,
        default: Option<prost_types::Value>,
    ) -> Result<()> {
        self.registry_mut().add_discrete_input(name, default)
    }

    /// Declare a discrete output, optionally with a default value.
    fn add_discrete_output(
        &mut self,
        name: &str,
        default: Option<prost_types::Value>,
    ) -> Result<()> {
        self.registry_mut().add_discrete_output(name, default)
    }

    /// Declare an available option and its type.
    fn add_option(&mut self, name: &str, option_type: &str) -> Result<()> {
        self.registry_mut().add_option(name, option_type)
    }

    /// Declare a partial derivative of `func` with respect to `var`.
    fn declare_partials(&mut self, func: &str, var: &str) -> Result<()> {
        self.registry_mut().declare_partials(func, var)
    }

    /// Resolve the shape of a variable declared with a dynamic shape.
    fn set_variable_shape(
        &mut self,
        name: &str,
        var_type: VariableType,
        shape: &[usize],
    ) -> Result<()> {
        self.registry_mut()
            .set_variable_shape(name, var_type, shape)
    }

    /// Metadata for all continuous variables.
    fn get_variable_definitions(&self) -> Result<Vec<VariableMetaData>> {
        Ok(self.registry().var_meta().to_vec())
    }

    /// Metadata for all discrete variables.
    fn get_discrete_variable_definitions(&self) -> Result<Vec<VariableMetaData>> {
        Ok(self.registry().discrete_meta().to_vec())
    }

    /// Declared partials as `(function, variable)` pairs.
    fn get_partials_definitions(&self) -> Result<Vec<(String, String)>> {
        Ok(self.registry().partials_meta().to_vec())
    }

    /// Available options as a map of name to type string.
    fn get_available_options(&self) -> Result<HashMap<String, String>> {
        Ok(self.registry().options_list().clone())
    }

    /// Discipline properties reported to a client via `GetInfo`.
    fn get_properties(&self) -> DisciplineProperties {
        DisciplineProperties {
            continuous: self.is_continuous(),
            differentiable: self.is_differentiable(),
            provides_gradients: self.provides_gradients(),
            name: self.name().to_string(),
            version: self.version().to_string(),
        }
    }
}

/// A discipline that maps inputs directly to outputs: `y = f(x)`.
#[async_trait]
pub trait ExplicitDiscipline: Discipline {
    /// Compute outputs from inputs.
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap>;

    /// Compute partial derivatives of outputs with respect to inputs.
    ///
    /// Returns a "not implemented" error by default; override to provide gradients.
    async fn compute_partials(&self, _inputs: &ArrayMap) -> Result<PartialMap> {
        Err(PhiloteError::not_implemented("compute_partials"))
    }

    /// Compute outputs from inputs, including discrete variables.
    ///
    /// Defaults to delegating to [`compute`](Self::compute) and returning no
    /// discrete outputs. Override when the discipline declares discrete variables.
    async fn compute_with_discrete(
        &self,
        inputs: &ArrayMap,
        _discrete_inputs: &DiscreteMap,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        Ok((self.compute(inputs).await?, HashMap::new()))
    }

    /// Compute partials, including discrete variables.
    async fn compute_partials_with_discrete(
        &self,
        inputs: &ArrayMap,
        _discrete_inputs: &DiscreteMap,
    ) -> Result<PartialMap> {
        self.compute_partials(inputs).await
    }
}

/// A discipline defined by residual equations: `R(x, y) = 0`.
#[async_trait]
pub trait ImplicitDiscipline: Discipline {
    /// Evaluate residuals for the given inputs and proposed outputs.
    async fn compute_residuals(&self, inputs: &ArrayMap, outputs: &ArrayMap) -> Result<ArrayMap>;

    /// Solve for the outputs that drive the residuals to zero.
    async fn solve_residuals(&self, inputs: &ArrayMap) -> Result<ArrayMap>;

    /// Compute partial derivatives of the residuals.
    ///
    /// Returns a "not implemented" error by default.
    async fn residual_partials(
        &self,
        _inputs: &ArrayMap,
        _outputs: &ArrayMap,
    ) -> Result<PartialMap> {
        Err(PhiloteError::not_implemented("residual_partials"))
    }

    /// Apply the linearized residual operator without forming the Jacobian.
    ///
    /// In [`Fwd`](LinearMode::Fwd) mode this accumulates into `d_residuals`; in
    /// [`Rev`](LinearMode::Rev) mode it accumulates into `d_inputs` and `d_outputs`.
    ///
    /// This is local API only — the Philote standard defines no matrix-free RPC, so
    /// nothing on the wire exercises it. The signature mirrors Philote-Python's
    /// `apply_linear` so a discipline ported between the two reads the same.
    async fn apply_linear(
        &self,
        _inputs: &ArrayMap,
        _outputs: &ArrayMap,
        _d_inputs: &mut ArrayMap,
        _d_outputs: &mut ArrayMap,
        _d_residuals: &mut ArrayMap,
        _mode: LinearMode,
    ) -> Result<()> {
        Err(PhiloteError::not_implemented("apply_linear"))
    }

    /// Evaluate residuals, including discrete variables.
    async fn compute_residuals_with_discrete(
        &self,
        inputs: &ArrayMap,
        outputs: &ArrayMap,
        _discrete_inputs: &DiscreteMap,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        Ok((
            self.compute_residuals(inputs, outputs).await?,
            HashMap::new(),
        ))
    }

    /// Solve the residuals, including discrete variables.
    async fn solve_residuals_with_discrete(
        &self,
        inputs: &ArrayMap,
        _discrete_inputs: &DiscreteMap,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        Ok((self.solve_residuals(inputs).await?, HashMap::new()))
    }

    /// Compute residual partials, including discrete variables.
    async fn residual_partials_with_discrete(
        &self,
        inputs: &ArrayMap,
        outputs: &ArrayMap,
        _discrete_inputs: &DiscreteMap,
    ) -> Result<PartialMap> {
        self.residual_partials(inputs, outputs).await
    }
}

/// Builder for [`VariableMetaData`].
#[derive(Debug, Clone)]
pub struct VariableInfo {
    /// Variable name.
    pub name: String,
    /// Variable type (input, output, or residual).
    pub var_type: VariableType,
    /// Array shape.
    pub shape: Vec<usize>,
    /// Physical units.
    pub units: String,
    /// Whether the client may set this variable's shape.
    pub dynamic_shape: bool,
}

impl VariableInfo {
    /// Create a variable with the given properties.
    pub fn new(name: String, var_type: VariableType, shape: Vec<usize>, units: String) -> Self {
        Self {
            name,
            var_type,
            shape,
            units,
            dynamic_shape: false,
        }
    }

    /// Create an input variable.
    pub fn input(name: String, shape: Vec<usize>, units: String) -> Self {
        Self::new(name, VariableType::KInput, shape, units)
    }

    /// Create an output variable.
    pub fn output(name: String, shape: Vec<usize>, units: String) -> Self {
        Self::new(name, VariableType::KOutput, shape, units)
    }

    /// Create a residual variable.
    pub fn residual(name: String, shape: Vec<usize>, units: String) -> Self {
        Self::new(name, VariableType::KResidual, shape, units)
    }

    /// Mark this variable as having a client-supplied shape.
    pub fn dynamic(mut self) -> Self {
        self.dynamic_shape = true;
        self.shape.clear();
        self
    }

    /// Total number of elements.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impl_registry;

    #[derive(Default)]
    struct Bare {
        registry: VariableRegistry,
    }

    impl Discipline for Bare {
        impl_registry!(registry);

        fn setup(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn defaults_report_sensible_properties() {
        let d = Bare::default();
        let props = d.get_properties();
        assert_eq!(props.name, "UnnamedDiscipline");
        assert_eq!(props.version, env!("CARGO_PKG_VERSION"));
        assert!(props.continuous);
        assert!(!props.differentiable);
        assert!(!props.provides_gradients);
    }

    #[test]
    fn provided_methods_delegate_to_registry() {
        let mut d = Bare::default();
        d.add_input("x", &[2], "m").unwrap();
        d.add_output("y", &[2], "").unwrap();
        d.declare_partials("y", "x").unwrap();
        d.add_option("a", "float").unwrap();
        d.add_discrete_input("n", None).unwrap();

        assert_eq!(d.get_variable_definitions().unwrap().len(), 2);
        assert_eq!(
            d.get_partials_definitions().unwrap(),
            vec![("y".into(), "x".into())]
        );
        assert_eq!(d.get_available_options().unwrap().len(), 1);
        assert_eq!(d.get_discrete_variable_definitions().unwrap().len(), 1);
    }

    #[test]
    fn variable_info_dynamic_clears_shape() {
        let info = VariableInfo::input("x".into(), vec![3], "m".into()).dynamic();
        assert!(info.dynamic_shape);
        assert!(info.shape.is_empty());
        let meta: VariableMetaData = info.into();
        assert!(meta.dynamic_shape);
        assert!(meta.shape.is_empty());
    }

    #[test]
    fn variable_info_size_is_product_of_shape() {
        assert_eq!(
            VariableInfo::output("y".into(), vec![2, 3], "".into()).size(),
            6
        );
    }
}
