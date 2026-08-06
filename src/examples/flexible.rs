use async_trait::async_trait;
use ndarray::ArrayD;
use std::collections::HashMap;

use crate::registry::VariableRegistry;
use crate::traits::{Discipline, ExplicitDiscipline};
use crate::{impl_registry, ArrayMap, PartialMap, Result};

/// Doubles every element of its input: `y = 2·x`.
///
/// Neither variable has a server-side shape; the client sets both via
/// `SetVariableShapes` before computing. Matches
/// `philote_mdo.examples.FlexibleDiscipline`.
#[derive(Default)]
pub struct FlexibleDiscipline {
    registry: VariableRegistry,
}

impl FlexibleDiscipline {
    /// Create the discipline with an empty variable registry.
    ///
    /// `setup` still has to run to declare `x` and `y`, and the client must set
    /// their shapes before computing.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Discipline for FlexibleDiscipline {
    impl_registry!(registry);

    fn name(&self) -> &str {
        "FlexibleDiscipline"
    }

    fn is_differentiable(&self) -> bool {
        true
    }

    fn provides_gradients(&self) -> bool {
        true
    }

    fn setup(&mut self) -> Result<()> {
        self.add_dynamic_input("x", "m")?;
        self.add_dynamic_output("y", "m")
    }

    fn setup_partials(&mut self) -> Result<()> {
        self.declare_partials("y", "x")
    }
}

#[async_trait]
impl ExplicitDiscipline for FlexibleDiscipline {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let mut outputs = HashMap::new();
        outputs.insert("y".to_string(), &inputs["x"] * 2.0);
        Ok(outputs)
    }

    async fn compute_partials(&self, inputs: &ArrayMap) -> Result<PartialMap> {
        let n = inputs["x"].len();

        // dy/dx is 2·I for an element-wise doubling.
        let mut jacobian = ArrayD::zeros(ndarray::IxDyn(&[n, n]));
        for i in 0..n {
            jacobian[[i, i]] = 2.0;
        }

        let mut partials: PartialMap = HashMap::new();
        partials.insert(("y".to_string(), "x".to_string()), jacobian);
        Ok(partials)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::vector;
    use approx::assert_relative_eq;

    fn vector_input(values: &[f64]) -> ArrayMap {
        let mut map = HashMap::new();
        map.insert("x".to_string(), vector(values));
        map
    }

    #[test]
    fn declares_both_variables_as_dynamic() {
        let mut d = FlexibleDiscipline::new();
        d.setup().unwrap();

        let vars = d.get_variable_definitions().unwrap();
        assert_eq!(vars.len(), 2);
        for var in vars {
            assert!(var.dynamic_shape, "{} should be dynamic", var.name);
            assert!(var.shape.is_empty());
            assert_eq!(var.units, "m");
        }
    }

    #[test]
    fn shapes_are_unresolved_until_set() {
        use crate::philote_info::VariableType;

        let mut d = FlexibleDiscipline::new();
        d.setup().unwrap();
        assert!(d.registry().assert_shapes_resolved().is_err());

        d.set_variable_shape("x", VariableType::KInput, &[3])
            .unwrap();
        d.set_variable_shape("y", VariableType::KOutput, &[3])
            .unwrap();
        assert!(d.registry().assert_shapes_resolved().is_ok());
    }

    #[tokio::test]
    async fn doubles_its_input() {
        let d = FlexibleDiscipline::new();
        let out = d.compute(&vector_input(&[1.0, 2.0, 3.0])).await.unwrap();
        assert_eq!(out["y"].shape(), &[3]);
        assert_relative_eq!(out["y"][[0]], 2.0);
        assert_relative_eq!(out["y"][[1]], 4.0);
        assert_relative_eq!(out["y"][[2]], 6.0);
    }

    #[tokio::test]
    async fn jacobian_is_twice_the_identity() {
        let d = FlexibleDiscipline::new();
        let partials = d
            .compute_partials(&vector_input(&[1.0, 2.0, 3.0]))
            .await
            .unwrap();
        let jac = &partials[&("y".to_string(), "x".to_string())];

        assert_eq!(jac.shape(), &[3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 2.0 } else { 0.0 };
                assert_relative_eq!(jac[[i, j]], expected);
            }
        }
    }
}
