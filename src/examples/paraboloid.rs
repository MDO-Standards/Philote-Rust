use async_trait::async_trait;
use std::collections::HashMap;

use crate::registry::VariableRegistry;
use crate::traits::{Discipline, ExplicitDiscipline};
use crate::{impl_registry, ArrayMap, PartialMap, Result};

/// Two-dimensional paraboloid: `f_xy = (x - 3)² + x·y + (y + 4)² - 3`.
///
/// Matches `philote_mdo.examples.Paraboloid` in Philote-Python, including units,
/// so results are directly comparable across implementations.
#[derive(Default)]
pub struct Paraboloid {
    registry: VariableRegistry,
}

impl Paraboloid {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Discipline for Paraboloid {
    impl_registry!(registry);

    fn name(&self) -> &str {
        "Paraboloid"
    }

    fn is_differentiable(&self) -> bool {
        true
    }

    fn provides_gradients(&self) -> bool {
        true
    }

    fn setup(&mut self) -> Result<()> {
        self.add_input("x", &[1], "m")?;
        self.add_input("y", &[1], "m")?;
        self.add_output("f_xy", &[1], "m**2")
    }

    fn setup_partials(&mut self) -> Result<()> {
        self.declare_partials("f_xy", "x")?;
        self.declare_partials("f_xy", "y")
    }
}

#[async_trait]
impl ExplicitDiscipline for Paraboloid {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let x = &inputs["x"];
        let y = &inputs["y"];

        let f = (x - 3.0).mapv(|v| v * v) + x * y + (y + 4.0).mapv(|v| v * v) - 3.0;

        let mut outputs = HashMap::new();
        outputs.insert("f_xy".to_string(), f);
        Ok(outputs)
    }

    async fn compute_partials(&self, inputs: &ArrayMap) -> Result<PartialMap> {
        let x = &inputs["x"];
        let y = &inputs["y"];

        let mut partials: PartialMap = HashMap::new();
        partials.insert(("f_xy".to_string(), "x".to_string()), x * 2.0 - 6.0 + y);
        partials.insert(("f_xy".to_string(), "y".to_string()), y * 2.0 + 8.0 + x);
        Ok(partials)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::scalar;
    use approx::assert_relative_eq;

    fn inputs(x: f64, y: f64) -> ArrayMap {
        let mut map = HashMap::new();
        map.insert("x".to_string(), scalar(x));
        map.insert("y".to_string(), scalar(y));
        map
    }

    #[test]
    fn declares_expected_variables_and_partials() {
        let mut d = Paraboloid::new();
        d.setup().unwrap();
        d.setup_partials().unwrap();

        let vars = d.get_variable_definitions().unwrap();
        assert_eq!(vars.len(), 3);
        let f = vars.iter().find(|v| v.name == "f_xy").unwrap();
        assert_eq!(f.units, "m**2");
        assert_eq!(f.shape, vec![1]);

        assert_eq!(d.get_partials_definitions().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn computes_expected_value() {
        let d = Paraboloid::new();
        let out = d.compute(&inputs(1.0, 2.0)).await.unwrap();
        // (1-3)^2 + 1*2 + (2+4)^2 - 3 = 4 + 2 + 36 - 3 = 39
        assert_relative_eq!(out["f_xy"][[0]], 39.0);
    }

    #[tokio::test]
    async fn computes_expected_value_at_origin() {
        let d = Paraboloid::new();
        let out = d.compute(&inputs(0.0, 0.0)).await.unwrap();
        // 9 + 0 + 16 - 3 = 22
        assert_relative_eq!(out["f_xy"][[0]], 22.0);
    }

    #[tokio::test]
    async fn computes_expected_partials() {
        let d = Paraboloid::new();
        let partials = d.compute_partials(&inputs(1.0, 2.0)).await.unwrap();
        // df/dx = 2x - 6 + y = 2 - 6 + 2 = -2
        assert_relative_eq!(partials[&("f_xy".into(), "x".into())][[0]], -2.0);
        // df/dy = 2y + 8 + x = 4 + 8 + 1 = 13
        assert_relative_eq!(partials[&("f_xy".into(), "y".into())][[0]], 13.0);
    }

    #[tokio::test]
    async fn partials_match_finite_difference() {
        let d = Paraboloid::new();
        let (x, y, h) = (1.3, -2.7, 1e-6);
        let base = d.compute(&inputs(x, y)).await.unwrap()["f_xy"][[0]];
        let dx = (d.compute(&inputs(x + h, y)).await.unwrap()["f_xy"][[0]] - base) / h;
        let dy = (d.compute(&inputs(x, y + h)).await.unwrap()["f_xy"][[0]] - base) / h;

        let partials = d.compute_partials(&inputs(x, y)).await.unwrap();
        assert_relative_eq!(
            partials[&("f_xy".into(), "x".into())][[0]],
            dx,
            epsilon = 1e-5
        );
        assert_relative_eq!(
            partials[&("f_xy".into(), "y".into())][[0]],
            dy,
            epsilon = 1e-5
        );
    }
}
