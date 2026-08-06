//! Default trait method bodies.
//!
//! [`ExplicitDiscipline`] and [`ImplicitDiscipline`] both provide `*_with_discrete`
//! variants that default to delegating to the continuous method and reporting no
//! discrete outputs. A discipline with no discrete variables never overrides them,
//! so the defaults are what actually runs in the common case. The disciplines here
//! override only the required methods, so every call below exercises a default body.

use async_trait::async_trait;
use ndarray::ArrayD;
use philote_mdo::examples::{scalar, vector};
use philote_mdo::philote_info::{VariableMetaData, VariableType};
use philote_mdo::registry::VariableRegistry;
use philote_mdo::traits::{
    Discipline, ExplicitDiscipline, ImplicitDiscipline, LinearMode, VariableInfo,
};
use philote_mdo::{impl_registry, ArrayMap, DiscreteMap, PartialMap, PhiloteError, Result};
use std::collections::HashMap;

/// A discrete map that must be ignored by every default body.
fn discrete_inputs() -> DiscreteMap {
    let mut map = DiscreteMap::new();
    map.insert(
        "mode".to_string(),
        philote_mdo::discrete::json_to_value(&serde_json::json!("turbulent")),
    );
    map
}

fn inputs(values: &[f64]) -> ArrayMap {
    let mut map = ArrayMap::new();
    map.insert("x".to_string(), vector(values));
    map
}

// --- Explicit disciplines ---

/// `y = 2x`. Overrides `compute` and nothing else.
#[derive(Default)]
struct Doubler {
    registry: VariableRegistry,
}

impl Discipline for Doubler {
    impl_registry!(registry);

    fn name(&self) -> &str {
        "Doubler"
    }

    fn setup(&mut self) -> Result<()> {
        self.add_input("x", &[2], "")?;
        self.add_output("y", &[2], "")
    }
}

#[async_trait]
impl ExplicitDiscipline for Doubler {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let mut outputs = ArrayMap::new();
        outputs.insert("y".to_string(), &inputs["x"] * 2.0);
        Ok(outputs)
    }
}

/// `y = 2x` with analytic gradients. Overrides `compute` and `compute_partials`.
#[derive(Default)]
struct DifferentiableDoubler {
    registry: VariableRegistry,
}

impl Discipline for DifferentiableDoubler {
    impl_registry!(registry);

    fn setup(&mut self) -> Result<()> {
        self.add_input("x", &[2], "")?;
        self.add_output("y", &[2], "")
    }
}

#[async_trait]
impl ExplicitDiscipline for DifferentiableDoubler {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let mut outputs = ArrayMap::new();
        outputs.insert("y".to_string(), &inputs["x"] * 2.0);
        Ok(outputs)
    }

    /// Deliberately depends on the inputs, so a default that dropped them would
    /// produce the wrong numbers rather than merely a different code path.
    async fn compute_partials(&self, inputs: &ArrayMap) -> Result<PartialMap> {
        let mut partials = PartialMap::new();
        partials.insert(
            ("y".to_string(), "x".to_string()),
            inputs["x"].mapv(|v| 2.0 * v),
        );
        Ok(partials)
    }
}

#[tokio::test]
async fn compute_with_discrete_delegates_to_compute_and_returns_no_discrete_outputs() {
    let d = Doubler::default();
    let (outputs, discrete_outputs) = d
        .compute_with_discrete(&inputs(&[1.0, 2.0]), &discrete_inputs())
        .await
        .unwrap();

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs["y"], vector(&[2.0, 4.0]));
    assert!(
        discrete_outputs.is_empty(),
        "a discipline that overrides only `compute` has no discrete outputs to report"
    );
}

#[tokio::test]
async fn compute_partials_with_discrete_delegates_to_compute_partials() {
    let d = DifferentiableDoubler::default();
    let partials = d
        .compute_partials_with_discrete(&inputs(&[3.0, 4.0]), &discrete_inputs())
        .await
        .unwrap();

    assert_eq!(partials.len(), 1);
    // The inputs must reach `compute_partials` unchanged.
    assert_eq!(
        partials[&("y".to_string(), "x".to_string())],
        vector(&[6.0, 8.0])
    );
}

#[tokio::test]
async fn compute_partials_with_discrete_reports_not_implemented_when_gradients_are_absent() {
    let d = Doubler::default();
    let err = d
        .compute_partials_with_discrete(&inputs(&[1.0, 2.0]), &discrete_inputs())
        .await
        .unwrap_err();

    assert!(matches!(err, PhiloteError::NotImplemented(ref f) if f == "compute_partials"));
    assert_eq!(err.to_status().code(), tonic::Code::Unimplemented);
}

// --- Implicit disciplines ---

/// `R(x, y) = y - 2x`, solved by `y = 2x`. Overrides only the two required methods.
#[derive(Default)]
struct LinearImplicit {
    registry: VariableRegistry,
}

impl Discipline for LinearImplicit {
    impl_registry!(registry);

    fn setup(&mut self) -> Result<()> {
        self.add_input("x", &[1], "")?;
        self.add_output("y", &[1], "")
    }
}

#[async_trait]
impl ImplicitDiscipline for LinearImplicit {
    async fn compute_residuals(&self, inputs: &ArrayMap, outputs: &ArrayMap) -> Result<ArrayMap> {
        let mut residuals = ArrayMap::new();
        residuals.insert("y".to_string(), &outputs["y"] - &(&inputs["x"] * 2.0));
        Ok(residuals)
    }

    async fn solve_residuals(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let mut outputs = ArrayMap::new();
        outputs.insert("y".to_string(), &inputs["x"] * 2.0);
        Ok(outputs)
    }
}

/// Adds residual partials to [`LinearImplicit`]'s behaviour.
#[derive(Default)]
struct DifferentiableImplicit {
    registry: VariableRegistry,
}

impl Discipline for DifferentiableImplicit {
    impl_registry!(registry);

    fn setup(&mut self) -> Result<()> {
        self.add_input("x", &[1], "")?;
        self.add_output("y", &[1], "")
    }
}

#[async_trait]
impl ImplicitDiscipline for DifferentiableImplicit {
    async fn compute_residuals(&self, inputs: &ArrayMap, outputs: &ArrayMap) -> Result<ArrayMap> {
        let mut residuals = ArrayMap::new();
        residuals.insert("y".to_string(), &outputs["y"] - &(&inputs["x"] * 2.0));
        Ok(residuals)
    }

    async fn solve_residuals(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let mut outputs = ArrayMap::new();
        outputs.insert("y".to_string(), &inputs["x"] * 2.0);
        Ok(outputs)
    }

    /// Both maps feed the result, so a default that dropped either one would be
    /// visible in the numbers.
    async fn residual_partials(&self, inputs: &ArrayMap, outputs: &ArrayMap) -> Result<PartialMap> {
        let mut partials = PartialMap::new();
        partials.insert(
            ("y".to_string(), "x".to_string()),
            &inputs["x"] * 10.0 + &outputs["y"],
        );
        Ok(partials)
    }
}

fn guess(value: f64) -> ArrayMap {
    let mut map = ArrayMap::new();
    map.insert("y".to_string(), scalar(value));
    map
}

#[tokio::test]
async fn compute_residuals_with_discrete_delegates_and_returns_no_discrete_outputs() {
    let d = LinearImplicit::default();
    let (residuals, discrete_outputs) = d
        .compute_residuals_with_discrete(&inputs(&[3.0]), &guess(5.0), &discrete_inputs())
        .await
        .unwrap();

    // 5 - 2*3
    assert_eq!(residuals["y"], scalar(-1.0));
    assert!(discrete_outputs.is_empty());
}

#[tokio::test]
async fn solve_residuals_with_discrete_delegates_and_returns_no_discrete_outputs() {
    let d = LinearImplicit::default();
    let (outputs, discrete_outputs) = d
        .solve_residuals_with_discrete(&inputs(&[3.0]), &discrete_inputs())
        .await
        .unwrap();

    assert_eq!(outputs["y"], scalar(6.0));
    assert!(discrete_outputs.is_empty());
}

#[tokio::test]
async fn residual_partials_with_discrete_delegates_to_residual_partials() {
    let d = DifferentiableImplicit::default();
    let partials = d
        .residual_partials_with_discrete(&inputs(&[3.0]), &guess(5.0), &discrete_inputs())
        .await
        .unwrap();

    // Both the inputs and the outputs arrived: 10*3 + 5.
    assert_eq!(partials[&("y".to_string(), "x".to_string())], scalar(35.0));
}

#[tokio::test]
async fn residual_partials_with_discrete_reports_not_implemented_when_gradients_are_absent() {
    let d = LinearImplicit::default();
    let err = d
        .residual_partials_with_discrete(&inputs(&[3.0]), &guess(5.0), &discrete_inputs())
        .await
        .unwrap_err();

    assert!(matches!(err, PhiloteError::NotImplemented(ref f) if f == "residual_partials"));
    assert_eq!(err.to_status().code(), tonic::Code::Unimplemented);
}

#[tokio::test]
async fn apply_linear_is_unimplemented_by_default_and_leaves_the_seeds_untouched() {
    let d = LinearImplicit::default();

    for mode in [LinearMode::Fwd, LinearMode::Rev] {
        let mut d_inputs = inputs(&[1.0]);
        let mut d_outputs = guess(1.0);
        let mut d_residuals = guess(0.0);

        let err = d
            .apply_linear(
                &inputs(&[3.0]),
                &guess(6.0),
                &mut d_inputs,
                &mut d_outputs,
                &mut d_residuals,
                mode,
            )
            .await
            .unwrap_err();

        assert!(matches!(err, PhiloteError::NotImplemented(ref f) if f == "apply_linear"));
        assert_eq!(err.to_status().code(), tonic::Code::Unimplemented);
        // A failed call must not have accumulated anything.
        assert_eq!(d_inputs["x"], scalar(1.0));
        assert_eq!(d_outputs["y"], scalar(1.0));
        assert_eq!(d_residuals["y"], scalar(0.0));
    }
}

// --- Base discipline defaults ---

#[test]
fn set_options_accepts_and_ignores_option_values_by_default() {
    let mut d = Doubler::default();
    d.initialize().unwrap();

    let mut options = HashMap::new();
    options.insert("tolerance".to_string(), serde_json::json!(1e-6));
    d.set_options(&options).unwrap();

    // The default declares no options, so nothing was recorded either.
    assert!(d.get_available_options().unwrap().is_empty());
}

#[test]
fn lifecycle_hooks_default_to_doing_nothing_but_setup_still_declares_variables() {
    let mut d = Doubler::default();
    d.initialize().unwrap();
    d.configure().unwrap();
    d.setup().unwrap();
    d.setup_partials().unwrap();

    assert_eq!(d.get_variable_definitions().unwrap().len(), 2);
    assert!(d.get_partials_definitions().unwrap().is_empty());
    assert_eq!(d.get_properties().name, "Doubler");
}

#[test]
fn variable_info_residual_carries_the_residual_type_through_conversion() {
    let info = VariableInfo::residual("y".into(), vec![2, 3], "m/s".into());
    assert_eq!(info.var_type, VariableType::KResidual);
    assert_eq!(info.size(), 6);
    assert!(!info.dynamic_shape);

    let meta: VariableMetaData = info.into();
    assert_eq!(meta.r#type, i32::from(VariableType::KResidual));
    assert_eq!(meta.name, "y");
    assert_eq!(meta.shape, vec![2_i64, 3]);
    assert_eq!(meta.units, "m/s");
}

/// Not a default body, but the counterpart the defaults are measured against:
/// an override must win over the delegating default.
#[tokio::test]
async fn an_override_replaces_the_delegating_default() {
    struct Overriding {
        registry: VariableRegistry,
    }

    impl Discipline for Overriding {
        impl_registry!(registry);

        fn setup(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl ExplicitDiscipline for Overriding {
        async fn compute(&self, _inputs: &ArrayMap) -> Result<ArrayMap> {
            unreachable!("the discrete override must not fall through to `compute`")
        }

        async fn compute_with_discrete(
            &self,
            _inputs: &ArrayMap,
            discrete_inputs: &DiscreteMap,
        ) -> Result<(ArrayMap, DiscreteMap)> {
            let mut outputs = ArrayMap::new();
            outputs.insert("y".to_string(), ArrayD::zeros(vec![1]));
            Ok((outputs, discrete_inputs.clone()))
        }
    }

    let d = Overriding {
        registry: VariableRegistry::default(),
    };
    let (_, discrete_outputs) = d
        .compute_with_discrete(&ArrayMap::new(), &discrete_inputs())
        .await
        .unwrap();
    assert_eq!(discrete_outputs.len(), 1);
    assert!(discrete_outputs.contains_key("mode"));
}
