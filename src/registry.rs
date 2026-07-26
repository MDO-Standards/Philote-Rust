//! Centralized variable, partials, and option metadata for a discipline
//!
//! [`VariableRegistry`] owns everything a discipline declares during `setup`.
//! Centralizing it here (rather than leaving storage to each implementor, as the
//! original trait did) is what makes server-side shape resolution, duplicate
//! detection, residual twins, and re-`Setup` clearing possible at all.
//!
//! This mirrors the state held on `philote_mdo.general.Discipline` in Philote-Python.

use std::collections::HashMap;

use crate::philote_info::{VariableMetaData, VariableType};
use crate::validation::{validate_name, validate_option_type, validate_shape};
use crate::{PhiloteError, Result};

/// Metadata store backing a [`Discipline`](crate::traits::Discipline).
#[derive(Debug, Default, Clone)]
pub struct VariableRegistry {
    var_meta: Vec<VariableMetaData>,
    discrete_meta: Vec<VariableMetaData>,
    discrete_defaults: HashMap<(String, i32), prost_types::Value>,
    partials_meta: Vec<(String, String)>,
    options_list: HashMap<String, String>,
    is_implicit: bool,
}

impl VariableRegistry {
    /// Create an empty registry.
    ///
    /// When `is_implicit` is true, [`add_output`](Self::add_output) additionally
    /// records a `kResidual` twin for every output, matching Philote-Python.
    pub fn new(is_implicit: bool) -> Self {
        Self {
            is_implicit,
            ..Default::default()
        }
    }

    /// Whether this registry belongs to an implicit discipline.
    pub fn is_implicit(&self) -> bool {
        self.is_implicit
    }

    /// Turn on residual twins for outputs declared from now on.
    ///
    /// [`ImplicitServer`](crate::server::ImplicitServer) calls this when it takes
    /// ownership of a discipline, so an implicit discipline built with
    /// `#[derive(Default)]` still gets its residuals.
    pub fn mark_implicit(&mut self) {
        self.is_implicit = true;
    }

    /// Drop all variable, partials, and discrete metadata.
    ///
    /// Called by the server before re-running `setup`, so repeated `Setup` RPCs do
    /// not accumulate duplicate definitions. Equivalent to Python's `_clear_data`.
    /// Declared options are preserved, since they are established in `initialize`.
    pub fn clear(&mut self) {
        self.var_meta.clear();
        self.discrete_meta.clear();
        self.discrete_defaults.clear();
        self.partials_meta.clear();
    }

    fn ensure_absent(&self, name: &str, var_type: VariableType, context: &str) -> Result<()> {
        let pool = match var_type {
            VariableType::KDiscreteInput | VariableType::KDiscreteOutput => &self.discrete_meta,
            _ => &self.var_meta,
        };
        if pool
            .iter()
            .any(|v| v.name == name && v.r#type == i32::from(var_type))
        {
            return Err(PhiloteError::validation(
                context,
                format!("'{name}' is already defined"),
            ));
        }
        Ok(())
    }

    fn push_var(
        &mut self,
        name: &str,
        var_type: VariableType,
        shape: Option<&[usize]>,
        units: &str,
    ) -> Result<()> {
        self.var_meta.push(VariableMetaData {
            r#type: var_type.into(),
            name: name.to_string(),
            shape: shape
                .map(|s| s.iter().map(|&d| d as i64).collect())
                .unwrap_or_default(),
            units: units.to_string(),
            dynamic_shape: shape.is_none(),
        });
        Ok(())
    }

    /// Declare a continuous input with a fixed shape.
    pub fn add_input(&mut self, name: &str, shape: &[usize], units: &str) -> Result<()> {
        validate_name(name, "add_input")?;
        validate_shape(shape, "add_input")?;
        self.ensure_absent(name, VariableType::KInput, "add_input")?;
        self.push_var(name, VariableType::KInput, Some(shape), units)
    }

    /// Declare a continuous input whose shape the client sets via `SetVariableShapes`.
    pub fn add_dynamic_input(&mut self, name: &str, units: &str) -> Result<()> {
        validate_name(name, "add_input")?;
        self.ensure_absent(name, VariableType::KInput, "add_input")?;
        self.push_var(name, VariableType::KInput, None, units)
    }

    /// Declare a continuous output with a fixed shape.
    ///
    /// For an implicit discipline this also records a same-named `kResidual` entry.
    pub fn add_output(&mut self, name: &str, shape: &[usize], units: &str) -> Result<()> {
        validate_name(name, "add_output")?;
        validate_shape(shape, "add_output")?;
        self.ensure_absent(name, VariableType::KOutput, "add_output")?;
        self.push_var(name, VariableType::KOutput, Some(shape), units)?;
        if self.is_implicit {
            self.push_var(name, VariableType::KResidual, Some(shape), units)?;
        }
        Ok(())
    }

    /// Declare a continuous output whose shape the client sets via `SetVariableShapes`.
    pub fn add_dynamic_output(&mut self, name: &str, units: &str) -> Result<()> {
        validate_name(name, "add_output")?;
        self.ensure_absent(name, VariableType::KOutput, "add_output")?;
        self.push_var(name, VariableType::KOutput, None, units)?;
        if self.is_implicit {
            self.push_var(name, VariableType::KResidual, None, units)?;
        }
        Ok(())
    }

    fn push_discrete(
        &mut self,
        name: &str,
        var_type: VariableType,
        default: Option<prost_types::Value>,
        context: &str,
    ) -> Result<()> {
        validate_name(name, context)?;
        self.ensure_absent(name, var_type, context)?;
        self.discrete_meta.push(VariableMetaData {
            r#type: var_type.into(),
            name: name.to_string(),
            shape: Vec::new(),
            units: String::new(),
            dynamic_shape: false,
        });
        if let Some(value) = default {
            self.discrete_defaults
                .insert((name.to_string(), var_type.into()), value);
        }
        Ok(())
    }

    /// Declare a discrete input, optionally with a default value.
    ///
    /// Unlike Philote-Python — which accepts a `default` argument and discards it —
    /// the default is stored and seeded into the input map on every compute.
    pub fn add_discrete_input(
        &mut self,
        name: &str,
        default: Option<prost_types::Value>,
    ) -> Result<()> {
        self.push_discrete(
            name,
            VariableType::KDiscreteInput,
            default,
            "add_discrete_input",
        )
    }

    /// Declare a discrete output, optionally with a default value.
    pub fn add_discrete_output(
        &mut self,
        name: &str,
        default: Option<prost_types::Value>,
    ) -> Result<()> {
        self.push_discrete(
            name,
            VariableType::KDiscreteOutput,
            default,
            "add_discrete_output",
        )
    }

    /// Declare an available option and its type.
    ///
    /// `option_type` must be one of
    /// [`VALID_OPTION_TYPES`](crate::validation::VALID_OPTION_TYPES).
    pub fn add_option(&mut self, name: &str, option_type: &str) -> Result<()> {
        validate_name(name, "add_option")?;
        validate_option_type(option_type, name)?;
        if self.options_list.contains_key(name) {
            return Err(PhiloteError::validation(
                "add_option",
                format!("option '{name}' is already defined"),
            ));
        }
        self.options_list
            .insert(name.to_string(), option_type.to_string());
        Ok(())
    }

    /// Declare a partial derivative of `func` with respect to `var`.
    pub fn declare_partials(&mut self, func: &str, var: &str) -> Result<()> {
        validate_name(func, "declare_partials (func)")?;
        validate_name(var, "declare_partials (var)")?;
        let pair = (func.to_string(), var.to_string());
        if !self.partials_meta.contains(&pair) {
            self.partials_meta.push(pair);
        }
        Ok(())
    }

    /// Resolve the shape of a variable previously declared as dynamic.
    ///
    /// Rejects unknown variables and variables that were declared with a fixed
    /// shape. When `var_type` is `kOutput`, the matching `kResidual` twin is updated
    /// too, mirroring `SetVariableShapes` in Philote-Python.
    pub fn set_variable_shape(
        &mut self,
        name: &str,
        var_type: VariableType,
        shape: &[usize],
    ) -> Result<()> {
        validate_shape(shape, "SetVariableShapes")?;
        let encoded: Vec<i64> = shape.iter().map(|&d| d as i64).collect();

        let idx = self
            .var_meta
            .iter()
            .position(|v| v.name == name && v.r#type == i32::from(var_type))
            .ok_or_else(|| PhiloteError::VariableNotFound(name.to_string()))?;

        if !self.var_meta[idx].dynamic_shape {
            return Err(PhiloteError::validation(
                "SetVariableShapes",
                format!("variable '{name}' was not declared with a dynamic shape"),
            ));
        }
        self.var_meta[idx].shape = encoded.clone();

        if var_type == VariableType::KOutput {
            if let Some(twin) = self
                .var_meta
                .iter_mut()
                .find(|v| v.name == name && v.r#type == i32::from(VariableType::KResidual))
            {
                twin.shape = encoded;
            }
        }
        Ok(())
    }

    /// Error if any dynamic variable still has no shape.
    pub fn assert_shapes_resolved(&self) -> Result<()> {
        for var in &self.var_meta {
            if var.dynamic_shape && var.shape.is_empty() {
                return Err(PhiloteError::validation(
                    "compute",
                    format!(
                        "variable '{}' has a dynamic shape that was never resolved; \
                         call SetVariableShapes before computing",
                        var.name
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Continuous variable metadata (inputs, outputs, residuals).
    pub fn var_meta(&self) -> &[VariableMetaData] {
        &self.var_meta
    }

    /// Discrete variable metadata.
    pub fn discrete_meta(&self) -> &[VariableMetaData] {
        &self.discrete_meta
    }

    /// Declared default values, keyed by `(name, variable type)`.
    ///
    /// The type is part of the key because a discrete input and a discrete output
    /// may share a name.
    pub fn discrete_defaults(&self) -> &HashMap<(String, i32), prost_types::Value> {
        &self.discrete_defaults
    }

    /// Declared defaults for discrete *inputs* only.
    ///
    /// This is what seeds the map handed to a discipline: an output's default is
    /// the discipline's to produce, so feeding it back in as an input would be
    /// wrong.
    pub fn discrete_input_defaults(&self) -> HashMap<String, prost_types::Value> {
        self.discrete_defaults
            .iter()
            .filter(|((_, var_type), _)| *var_type == i32::from(VariableType::KDiscreteInput))
            .map(|((name, _), value)| (name.clone(), value.clone()))
            .collect()
    }

    /// Declared partials as `(function, variable)` pairs.
    pub fn partials_meta(&self) -> &[(String, String)] {
        &self.partials_meta
    }

    /// Declared options as a map of name to type string.
    pub fn options_list(&self) -> &HashMap<String, String> {
        &self.options_list
    }
}

/// Implement the two required [`Discipline`](crate::traits::Discipline) accessors
/// by delegating to a [`VariableRegistry`] field.
///
/// ```
/// use philote_mdo::{impl_registry, registry::VariableRegistry, traits::Discipline};
///
/// struct MyDiscipline {
///     registry: VariableRegistry,
/// }
///
/// impl Discipline for MyDiscipline {
///     impl_registry!(registry);
///
///     fn setup(&mut self) -> philote_mdo::Result<()> {
///         self.add_input("x", &[1], "")
///     }
/// }
/// ```
#[macro_export]
macro_rules! impl_registry {
    ($field:ident) => {
        fn registry(&self) -> &$crate::registry::VariableRegistry {
            &self.$field
        }

        fn registry_mut(&mut self) -> &mut $crate::registry::VariableRegistry {
            &mut self.$field
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names_of(reg: &VariableRegistry, var_type: VariableType) -> Vec<String> {
        reg.var_meta()
            .iter()
            .filter(|v| v.r#type == i32::from(var_type))
            .map(|v| v.name.clone())
            .collect()
    }

    #[test]
    fn adds_input_with_shape_and_units() {
        let mut reg = VariableRegistry::new(false);
        reg.add_input("x", &[2, 3], "m").unwrap();
        let var = &reg.var_meta()[0];
        assert_eq!(var.name, "x");
        assert_eq!(var.shape, vec![2, 3]);
        assert_eq!(var.units, "m");
        assert!(!var.dynamic_shape);
    }

    #[test]
    fn rejects_duplicate_input() {
        let mut reg = VariableRegistry::new(false);
        reg.add_input("x", &[1], "").unwrap();
        assert!(reg.add_input("x", &[1], "").is_err());
    }

    #[test]
    fn rejects_duplicate_output() {
        let mut reg = VariableRegistry::new(false);
        reg.add_output("f", &[1], "").unwrap();
        assert!(reg.add_output("f", &[1], "").is_err());
    }

    #[test]
    fn allows_same_name_across_input_and_output() {
        let mut reg = VariableRegistry::new(false);
        reg.add_input("x", &[1], "").unwrap();
        assert!(reg.add_output("x", &[1], "").is_ok());
    }

    #[test]
    fn rejects_empty_name() {
        let mut reg = VariableRegistry::new(false);
        assert!(reg.add_input("", &[1], "").is_err());
    }

    #[test]
    fn rejects_zero_dimension_shape() {
        let mut reg = VariableRegistry::new(false);
        assert!(reg.add_input("x", &[0], "").is_err());
    }

    #[test]
    fn implicit_output_creates_residual_twin() {
        let mut reg = VariableRegistry::new(true);
        reg.add_output("x", &[3], "m").unwrap();
        assert_eq!(names_of(&reg, VariableType::KOutput), vec!["x"]);
        assert_eq!(names_of(&reg, VariableType::KResidual), vec!["x"]);
        let residual = reg
            .var_meta()
            .iter()
            .find(|v| v.r#type == i32::from(VariableType::KResidual))
            .unwrap();
        assert_eq!(residual.shape, vec![3]);
        assert_eq!(residual.units, "m");
    }

    #[test]
    fn explicit_output_creates_no_twin() {
        let mut reg = VariableRegistry::new(false);
        reg.add_output("x", &[3], "").unwrap();
        assert!(names_of(&reg, VariableType::KResidual).is_empty());
    }

    #[test]
    fn dynamic_input_has_empty_shape_and_flag() {
        let mut reg = VariableRegistry::new(false);
        reg.add_dynamic_input("x", "m").unwrap();
        let var = &reg.var_meta()[0];
        assert!(var.shape.is_empty());
        assert!(var.dynamic_shape);
    }

    #[test]
    fn dynamic_implicit_output_creates_dynamic_twin() {
        let mut reg = VariableRegistry::new(true);
        reg.add_dynamic_output("x", "").unwrap();
        let residual = reg
            .var_meta()
            .iter()
            .find(|v| v.r#type == i32::from(VariableType::KResidual))
            .unwrap();
        assert!(residual.dynamic_shape);
        assert!(residual.shape.is_empty());
    }

    #[test]
    fn sets_dynamic_shape() {
        let mut reg = VariableRegistry::new(false);
        reg.add_dynamic_input("x", "").unwrap();
        reg.set_variable_shape("x", VariableType::KInput, &[4])
            .unwrap();
        assert_eq!(reg.var_meta()[0].shape, vec![4]);
    }

    #[test]
    fn setting_output_shape_updates_residual_twin() {
        let mut reg = VariableRegistry::new(true);
        reg.add_dynamic_output("x", "").unwrap();
        reg.set_variable_shape("x", VariableType::KOutput, &[5])
            .unwrap();
        for var in reg.var_meta() {
            assert_eq!(var.shape, vec![5], "{} not updated", var.name);
        }
    }

    #[test]
    fn rejects_shape_for_static_variable() {
        let mut reg = VariableRegistry::new(false);
        reg.add_input("x", &[2], "").unwrap();
        let err = reg
            .set_variable_shape("x", VariableType::KInput, &[4])
            .unwrap_err();
        assert!(matches!(err, PhiloteError::Validation { .. }));
    }

    #[test]
    fn rejects_shape_for_unknown_variable() {
        let mut reg = VariableRegistry::new(false);
        let err = reg
            .set_variable_shape("nope", VariableType::KInput, &[4])
            .unwrap_err();
        assert!(matches!(err, PhiloteError::VariableNotFound(_)));
    }

    #[test]
    fn rejects_zero_shape_on_set() {
        let mut reg = VariableRegistry::new(false);
        reg.add_dynamic_input("x", "").unwrap();
        assert!(reg
            .set_variable_shape("x", VariableType::KInput, &[0])
            .is_err());
    }

    #[test]
    fn assert_shapes_resolved_flags_unresolved() {
        let mut reg = VariableRegistry::new(false);
        reg.add_dynamic_input("x", "").unwrap();
        let err = reg.assert_shapes_resolved().unwrap_err();
        assert!(err.to_string().contains('x'));

        reg.set_variable_shape("x", VariableType::KInput, &[2])
            .unwrap();
        assert!(reg.assert_shapes_resolved().is_ok());
    }

    #[test]
    fn assert_shapes_resolved_passes_for_static() {
        let mut reg = VariableRegistry::new(false);
        reg.add_input("x", &[2], "").unwrap();
        assert!(reg.assert_shapes_resolved().is_ok());
    }

    #[test]
    fn stores_discrete_defaults() {
        let mut reg = VariableRegistry::new(false);
        reg.add_discrete_input(
            "n",
            Some(crate::discrete::json_to_value(&serde_json::json!(7))),
        )
        .unwrap();
        assert_eq!(reg.discrete_meta().len(), 1);
        assert!(reg
            .discrete_defaults()
            .contains_key(&("n".to_string(), i32::from(VariableType::KDiscreteInput))));
    }

    #[test]
    fn discrete_without_default_stores_no_default() {
        let mut reg = VariableRegistry::new(false);
        reg.add_discrete_input("n", None).unwrap();
        assert!(reg.discrete_defaults().is_empty());
    }

    #[test]
    fn rejects_duplicate_discrete_input() {
        let mut reg = VariableRegistry::new(false);
        reg.add_discrete_input("n", None).unwrap();
        assert!(reg.add_discrete_input("n", None).is_err());
    }

    #[test]
    fn allows_same_name_across_discrete_input_and_output() {
        let mut reg = VariableRegistry::new(false);
        reg.add_discrete_input("n", None).unwrap();
        assert!(reg.add_discrete_output("n", None).is_ok());
    }

    #[test]
    fn rejects_duplicate_option() {
        let mut reg = VariableRegistry::new(false);
        reg.add_option("a", "float").unwrap();
        assert!(reg.add_option("a", "int").is_err());
    }

    #[test]
    fn rejects_invalid_option_type() {
        let mut reg = VariableRegistry::new(false);
        assert!(reg.add_option("a", "complex").is_err());
    }

    #[test]
    fn declare_partials_deduplicates() {
        let mut reg = VariableRegistry::new(false);
        reg.declare_partials("f", "x").unwrap();
        reg.declare_partials("f", "x").unwrap();
        assert_eq!(reg.partials_meta().len(), 1);
    }

    #[test]
    fn declare_partials_rejects_empty_names() {
        let mut reg = VariableRegistry::new(false);
        assert!(reg.declare_partials("", "x").is_err());
        assert!(reg.declare_partials("f", "").is_err());
    }

    #[test]
    fn clear_empties_metadata_but_keeps_options() {
        let mut reg = VariableRegistry::new(true);
        reg.add_option("a", "float").unwrap();
        reg.add_input("x", &[1], "").unwrap();
        reg.add_output("y", &[1], "").unwrap();
        reg.add_discrete_input("n", None).unwrap();
        reg.declare_partials("y", "x").unwrap();

        reg.clear();

        assert!(reg.var_meta().is_empty());
        assert!(reg.discrete_meta().is_empty());
        assert!(reg.partials_meta().is_empty());
        assert!(reg.discrete_defaults().is_empty());
        assert_eq!(reg.options_list().len(), 1);
    }

    #[test]
    fn clear_allows_redeclaration_without_duplicates() {
        let mut reg = VariableRegistry::new(false);
        reg.add_input("x", &[1], "").unwrap();
        reg.clear();
        reg.add_input("x", &[1], "").unwrap();
        assert_eq!(reg.var_meta().len(), 1);
    }
}
