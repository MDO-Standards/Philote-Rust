use async_trait::async_trait;
use ndarray::ArrayD;
use std::collections::HashMap;

use crate::registry::VariableRegistry;
use crate::traits::{Discipline, ExplicitDiscipline};
use crate::{impl_registry, ArrayMap, PartialMap, PhiloteError, Result};

/// The Rosenbrock function over an `n`-dimensional input vector.
///
/// Matches `philote_mdo.examples.Rosenbrock`, including the `dimension` option.
/// Philote-Python delegates to `scipy.optimize.rosen`; the same formula is
/// implemented directly here rather than pulling in a numerics dependency.
pub struct Rosenbrock {
    registry: VariableRegistry,
    dimension: usize,
}

impl Default for Rosenbrock {
    fn default() -> Self {
        let mut discipline = Self {
            registry: VariableRegistry::new(false),
            dimension: 2,
        };
        discipline
            .initialize()
            .expect("declaring the dimension option cannot fail");
        discipline
    }
}

impl Rosenbrock {
    /// Create the discipline with the default input dimension of 2.
    ///
    /// The `dimension` option is declared here; set it before `setup` to size the
    /// input vector differently.
    pub fn new() -> Self {
        Self::default()
    }

    /// The configured input dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// `sum_{i} 100·(x_{i+1} - x_i²)² + (1 - x_i)²`
fn rosen(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| {
            let (a, b) = (w[0], w[1]);
            100.0 * (b - a * a).powi(2) + (1.0 - a).powi(2)
        })
        .sum()
}

/// Gradient of [`rosen`].
fn rosen_der(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut der = vec![0.0; n];
    for i in 0..n.saturating_sub(1) {
        let (a, b) = (x[i], x[i + 1]);
        der[i] += -400.0 * a * (b - a * a) - 2.0 * (1.0 - a);
        der[i + 1] += 200.0 * (b - a * a);
    }
    der
}

impl Discipline for Rosenbrock {
    impl_registry!(registry);

    fn name(&self) -> &str {
        "Rosenbrock"
    }

    fn is_differentiable(&self) -> bool {
        true
    }

    fn provides_gradients(&self) -> bool {
        true
    }

    fn initialize(&mut self) -> Result<()> {
        self.add_option("dimension", "int")
    }

    fn set_options(&mut self, options: &HashMap<String, serde_json::Value>) -> Result<()> {
        if let Some(value) = options.get("dimension") {
            let dimension = value.as_i64().ok_or_else(|| {
                PhiloteError::validation(
                    "set_options",
                    format!("'dimension' must be an integer, got {value}"),
                )
            })?;
            if dimension < 2 {
                return Err(PhiloteError::validation(
                    "set_options",
                    format!("'dimension' must be at least 2, got {dimension}"),
                ));
            }
            self.dimension = dimension as usize;
        }
        Ok(())
    }

    fn setup(&mut self) -> Result<()> {
        let dimension = self.dimension;
        self.add_input("x", &[dimension], "")?;
        self.add_output("f", &[1], "")
    }

    fn setup_partials(&mut self) -> Result<()> {
        self.declare_partials("f", "x")
    }
}

#[async_trait]
impl ExplicitDiscipline for Rosenbrock {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let x: Vec<f64> = inputs["x"].iter().copied().collect();

        let mut outputs = HashMap::new();
        outputs.insert(
            "f".to_string(),
            ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![rosen(&x)])
                .map_err(|e| PhiloteError::array_error(e.to_string()))?,
        );
        Ok(outputs)
    }

    async fn compute_partials(&self, inputs: &ArrayMap) -> Result<PartialMap> {
        let x: Vec<f64> = inputs["x"].iter().copied().collect();
        let der = rosen_der(&x);

        let mut partials: PartialMap = HashMap::new();
        partials.insert(
            ("f".to_string(), "x".to_string()),
            ArrayD::from_shape_vec(ndarray::IxDyn(&[der.len()]), der)
                .map_err(|e| PhiloteError::array_error(e.to_string()))?,
        );
        Ok(partials)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::vector;
    use approx::assert_relative_eq;

    fn inputs(values: &[f64]) -> ArrayMap {
        let mut map = HashMap::new();
        map.insert("x".to_string(), vector(values));
        map
    }

    #[test]
    fn declares_dimension_option() {
        let d = Rosenbrock::new();
        assert_eq!(
            d.get_available_options().unwrap().get("dimension"),
            Some(&"int".to_string())
        );
    }

    #[test]
    fn set_options_changes_input_shape() {
        let mut d = Rosenbrock::new();
        let mut options = HashMap::new();
        options.insert("dimension".to_string(), serde_json::json!(5));
        d.set_options(&options).unwrap();
        d.setup().unwrap();

        let x = d
            .get_variable_definitions()
            .unwrap()
            .into_iter()
            .find(|v| v.name == "x")
            .unwrap();
        assert_eq!(x.shape, vec![5]);
    }

    #[test]
    fn rejects_non_integer_dimension() {
        let mut d = Rosenbrock::new();
        let mut options = HashMap::new();
        options.insert("dimension".to_string(), serde_json::json!("many"));
        assert!(d.set_options(&options).is_err());
    }

    #[tokio::test]
    async fn minimum_is_zero_at_ones() {
        let d = Rosenbrock::new();
        let out = d.compute(&inputs(&[1.0, 1.0, 1.0])).await.unwrap();
        assert_relative_eq!(out["f"][[0]], 0.0);
    }

    #[tokio::test]
    async fn matches_known_value() {
        let d = Rosenbrock::new();
        // 100*(2 - 1)^2 + (1 - 1)^2 = 100
        let out = d.compute(&inputs(&[1.0, 2.0])).await.unwrap();
        assert_relative_eq!(out["f"][[0]], 100.0);
    }

    #[tokio::test]
    async fn gradient_vanishes_at_minimum() {
        let d = Rosenbrock::new();
        let partials = d.compute_partials(&inputs(&[1.0, 1.0, 1.0])).await.unwrap();
        for value in partials[&("f".to_string(), "x".to_string())].iter() {
            assert_relative_eq!(*value, 0.0);
        }
    }

    #[tokio::test]
    async fn gradient_matches_finite_difference() {
        let d = Rosenbrock::new();
        let x = [0.5, 1.7, -0.3, 2.1];
        let h = 1e-7;
        let base = d.compute(&inputs(&x)).await.unwrap()["f"][[0]];

        let partials = d.compute_partials(&inputs(&x)).await.unwrap();
        let analytic = &partials[&("f".to_string(), "x".to_string())];

        for i in 0..x.len() {
            let mut perturbed = x;
            perturbed[i] += h;
            let fd = (d.compute(&inputs(&perturbed)).await.unwrap()["f"][[0]] - base) / h;
            assert_relative_eq!(analytic[[i]], fd, epsilon = 1e-3);
        }
    }
}
