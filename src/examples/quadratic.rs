use async_trait::async_trait;
use std::collections::HashMap;

use crate::examples::scalar;
use crate::registry::VariableRegistry;
use crate::traits::{Discipline, ImplicitDiscipline, LinearMode};
use crate::{impl_registry, ArrayMap, PartialMap, Result};

/// Implicit solution of `a·x² + b·x + c = 0`.
///
/// Matches `philote_mdo.examples.QuadradicImplicit`. The residual is
/// `R(x) = a·x² + b·x + c`, solved with the positive quadratic root.
pub struct QuadraticImplicit {
    registry: VariableRegistry,
}

impl Default for QuadraticImplicit {
    fn default() -> Self {
        Self {
            registry: VariableRegistry::new(true),
        }
    }
}

impl QuadraticImplicit {
    /// Create the discipline with a registry configured for an implicit
    /// discipline, so each output gets a matching residual.
    ///
    /// `setup` declares the coefficients `a`, `b`, and `c` as inputs and the root
    /// `x` as the implicit output.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Discipline for QuadraticImplicit {
    impl_registry!(registry);

    fn name(&self) -> &str {
        "QuadraticImplicit"
    }

    fn is_differentiable(&self) -> bool {
        true
    }

    fn provides_gradients(&self) -> bool {
        true
    }

    fn setup(&mut self) -> Result<()> {
        self.add_input("a", &[1], "")?;
        self.add_input("b", &[1], "")?;
        self.add_input("c", &[1], "")?;
        // Implicit disciplines get a matching residual for free.
        self.add_output("x", &[1], "")
    }

    fn setup_partials(&mut self) -> Result<()> {
        self.declare_partials("x", "a")?;
        self.declare_partials("x", "b")?;
        self.declare_partials("x", "c")?;
        self.declare_partials("x", "x")
    }
}

#[async_trait]
impl ImplicitDiscipline for QuadraticImplicit {
    async fn compute_residuals(&self, inputs: &ArrayMap, outputs: &ArrayMap) -> Result<ArrayMap> {
        let (a, b, c) = (&inputs["a"], &inputs["b"], &inputs["c"]);
        let x = &outputs["x"];

        let mut residuals = HashMap::new();
        residuals.insert("x".to_string(), a * &x.mapv(|v| v * v) + b * x + c);
        Ok(residuals)
    }

    async fn solve_residuals(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let a = inputs["a"][[0]];
        let b = inputs["b"][[0]];
        let c = inputs["c"][[0]];

        let x = (-b + (b * b - 4.0 * a * c).sqrt()) / (2.0 * a);

        let mut outputs = HashMap::new();
        outputs.insert("x".to_string(), scalar(x));
        Ok(outputs)
    }

    async fn residual_partials(&self, inputs: &ArrayMap, outputs: &ArrayMap) -> Result<PartialMap> {
        let a = inputs["a"][[0]];
        let b = inputs["b"][[0]];
        let x = outputs["x"][[0]];

        let mut partials: PartialMap = HashMap::new();
        partials.insert(("x".to_string(), "a".to_string()), scalar(x * x));
        partials.insert(("x".to_string(), "b".to_string()), scalar(x));
        partials.insert(("x".to_string(), "c".to_string()), scalar(1.0));
        partials.insert(("x".to_string(), "x".to_string()), scalar(2.0 * a * x + b));
        Ok(partials)
    }

    async fn apply_linear(
        &self,
        inputs: &ArrayMap,
        outputs: &ArrayMap,
        d_inputs: &mut ArrayMap,
        d_outputs: &mut ArrayMap,
        d_residuals: &mut ArrayMap,
        mode: LinearMode,
    ) -> Result<()> {
        let a = inputs["a"][[0]];
        let b = inputs["b"][[0]];
        let x = outputs["x"][[0]];
        let dr_dx = 2.0 * a * x + b;

        match mode {
            LinearMode::Fwd => {
                if !d_residuals.contains_key("x") {
                    return Ok(());
                }
                let mut accumulated = 0.0;
                if let Some(dx) = d_outputs.get("x") {
                    accumulated += dr_dx * dx[[0]];
                }
                if let Some(da) = d_inputs.get("a") {
                    accumulated += x * x * da[[0]];
                }
                if let Some(db) = d_inputs.get("b") {
                    accumulated += x * db[[0]];
                }
                if let Some(dc) = d_inputs.get("c") {
                    accumulated += dc[[0]];
                }
                d_residuals.get_mut("x").expect("checked above")[[0]] += accumulated;
            }
            LinearMode::Rev => {
                let Some(dr) = d_residuals.get("x").map(|v| v[[0]]) else {
                    return Ok(());
                };
                if let Some(dx) = d_outputs.get_mut("x") {
                    dx[[0]] += dr_dx * dr;
                }
                if let Some(da) = d_inputs.get_mut("a") {
                    da[[0]] += x * x * dr;
                }
                if let Some(db) = d_inputs.get_mut("b") {
                    db[[0]] += x * dr;
                }
                if let Some(dc) = d_inputs.get_mut("c") {
                    dc[[0]] += dr;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn inputs(a: f64, b: f64, c: f64) -> ArrayMap {
        let mut map = HashMap::new();
        map.insert("a".to_string(), scalar(a));
        map.insert("b".to_string(), scalar(b));
        map.insert("c".to_string(), scalar(c));
        map
    }

    fn outputs(x: f64) -> ArrayMap {
        let mut map = HashMap::new();
        map.insert("x".to_string(), scalar(x));
        map
    }

    #[test]
    fn setup_creates_residual_twin() {
        use crate::philote_info::VariableType;

        let mut d = QuadraticImplicit::new();
        d.setup().unwrap();

        let vars = d.get_variable_definitions().unwrap();
        let residuals: Vec<_> = vars
            .iter()
            .filter(|v| v.r#type == i32::from(VariableType::KResidual))
            .collect();
        assert_eq!(residuals.len(), 1);
        assert_eq!(residuals[0].name, "x");
    }

    #[test]
    fn declares_four_partials() {
        let mut d = QuadraticImplicit::new();
        d.setup().unwrap();
        d.setup_partials().unwrap();
        assert_eq!(d.get_partials_definitions().unwrap().len(), 4);
    }

    #[tokio::test]
    async fn solves_known_root() {
        let d = QuadraticImplicit::new();
        // x^2 - 3x + 2 = 0 → roots 1 and 2; positive branch gives 2.
        let out = d.solve_residuals(&inputs(1.0, -3.0, 2.0)).await.unwrap();
        assert_relative_eq!(out["x"][[0]], 2.0);
    }

    #[tokio::test]
    async fn residual_is_zero_at_the_solution() {
        let d = QuadraticImplicit::new();
        let input = inputs(1.0, -3.0, 2.0);
        let solved = d.solve_residuals(&input).await.unwrap();
        let residuals = d.compute_residuals(&input, &solved).await.unwrap();
        assert_relative_eq!(residuals["x"][[0]], 0.0, epsilon = 1e-12);
    }

    #[tokio::test]
    async fn residual_is_nonzero_away_from_the_solution() {
        let d = QuadraticImplicit::new();
        let residuals = d
            .compute_residuals(&inputs(1.0, -3.0, 2.0), &outputs(0.0))
            .await
            .unwrap();
        // R(0) = c = 2
        assert_relative_eq!(residuals["x"][[0]], 2.0);
    }

    #[tokio::test]
    async fn computes_expected_residual_partials() {
        let d = QuadraticImplicit::new();
        let partials = d
            .residual_partials(&inputs(1.0, -3.0, 2.0), &outputs(2.0))
            .await
            .unwrap();

        assert_relative_eq!(partials[&("x".into(), "a".into())][[0]], 4.0);
        assert_relative_eq!(partials[&("x".into(), "b".into())][[0]], 2.0);
        assert_relative_eq!(partials[&("x".into(), "c".into())][[0]], 1.0);
        // dR/dx = 2ax + b = 4 - 3 = 1
        assert_relative_eq!(partials[&("x".into(), "x".into())][[0]], 1.0);
    }

    #[tokio::test]
    async fn residual_partials_match_finite_difference() {
        let d = QuadraticImplicit::new();
        let (a, b, c, x, h) = (1.5, -3.0, 2.0, 1.4, 1e-7);
        let base = d
            .compute_residuals(&inputs(a, b, c), &outputs(x))
            .await
            .unwrap()["x"][[0]];

        let analytic = d
            .residual_partials(&inputs(a, b, c), &outputs(x))
            .await
            .unwrap();

        let fd_a = (d
            .compute_residuals(&inputs(a + h, b, c), &outputs(x))
            .await
            .unwrap()["x"][[0]]
            - base)
            / h;
        let fd_x = (d
            .compute_residuals(&inputs(a, b, c), &outputs(x + h))
            .await
            .unwrap()["x"][[0]]
            - base)
            / h;

        assert_relative_eq!(
            analytic[&("x".into(), "a".into())][[0]],
            fd_a,
            epsilon = 1e-4
        );
        assert_relative_eq!(
            analytic[&("x".into(), "x".into())][[0]],
            fd_x,
            epsilon = 1e-4
        );
    }

    #[tokio::test]
    async fn apply_linear_forward_accumulates_into_residuals() {
        let d = QuadraticImplicit::new();
        let (a, b, c, x) = (1.0, -3.0, 2.0, 2.0);

        let mut d_inputs: ArrayMap = HashMap::new();
        d_inputs.insert("a".to_string(), scalar(1.0));
        let mut d_outputs: ArrayMap = HashMap::new();
        d_outputs.insert("x".to_string(), scalar(1.0));
        let mut d_residuals: ArrayMap = HashMap::new();
        d_residuals.insert("x".to_string(), scalar(0.0));

        d.apply_linear(
            &inputs(a, b, c),
            &outputs(x),
            &mut d_inputs,
            &mut d_outputs,
            &mut d_residuals,
            LinearMode::Fwd,
        )
        .await
        .unwrap();

        // dR/dx * 1 + x^2 * 1 = (2*1*2 - 3) + 4 = 5
        assert_relative_eq!(d_residuals["x"][[0]], 5.0);
    }

    #[tokio::test]
    async fn apply_linear_reverse_accumulates_into_inputs_and_outputs() {
        let d = QuadraticImplicit::new();
        let (a, b, c, x) = (1.0, -3.0, 2.0, 2.0);

        let mut d_inputs: ArrayMap = HashMap::new();
        d_inputs.insert("a".to_string(), scalar(0.0));
        d_inputs.insert("c".to_string(), scalar(0.0));
        let mut d_outputs: ArrayMap = HashMap::new();
        d_outputs.insert("x".to_string(), scalar(0.0));
        let mut d_residuals: ArrayMap = HashMap::new();
        d_residuals.insert("x".to_string(), scalar(1.0));

        d.apply_linear(
            &inputs(a, b, c),
            &outputs(x),
            &mut d_inputs,
            &mut d_outputs,
            &mut d_residuals,
            LinearMode::Rev,
        )
        .await
        .unwrap();

        assert_relative_eq!(d_outputs["x"][[0]], 1.0); // dR/dx = 1
        assert_relative_eq!(d_inputs["a"][[0]], 4.0); // x^2
        assert_relative_eq!(d_inputs["c"][[0]], 1.0);
    }

    #[tokio::test]
    async fn apply_linear_is_a_no_op_without_residual_seed() {
        let d = QuadraticImplicit::new();
        let mut d_inputs: ArrayMap = HashMap::new();
        d_inputs.insert("a".to_string(), scalar(0.0));
        let mut d_outputs: ArrayMap = HashMap::new();
        let mut d_residuals: ArrayMap = HashMap::new();

        d.apply_linear(
            &inputs(1.0, -3.0, 2.0),
            &outputs(2.0),
            &mut d_inputs,
            &mut d_outputs,
            &mut d_residuals,
            LinearMode::Rev,
        )
        .await
        .unwrap();

        assert_relative_eq!(d_inputs["a"][[0]], 0.0);
    }
}
