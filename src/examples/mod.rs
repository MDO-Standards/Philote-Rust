//! Ready-to-run example disciplines
//!
//! These mirror `philote_mdo/examples/` in Philote-Python, using the same formulas,
//! variable names, units, and options, so numeric results can be compared directly
//! across the two implementations.

mod flexible;
mod paraboloid;
mod quadratic;
mod rosenbrock;

pub use flexible::FlexibleDiscipline;
pub use paraboloid::Paraboloid;
pub use quadratic::QuadraticImplicit;
pub use rosenbrock::Rosenbrock;

use ndarray::ArrayD;

/// Build a one-element array, the shape Philote uses for scalar variables.
pub fn scalar(value: f64) -> ArrayD<f64> {
    ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![value]).expect("1-element array")
}

/// Build a one-dimensional array from a slice.
pub fn vector(values: &[f64]) -> ArrayD<f64> {
    ArrayD::from_shape_vec(ndarray::IxDyn(&[values.len()]), values.to_vec())
        .expect("length matches the slice")
}
