use ndarray::ArrayD;
use std::collections::HashMap;

pub mod philote_info {
    tonic::include_proto!("philote");
}

pub mod client;
pub mod error;
pub mod server;
pub mod traits;
pub mod types;
pub mod utils;

pub use error::PhiloteError;
pub use traits::{Discipline, ExplicitDiscipline, ImplicitDiscipline};
pub use types::{ArrayData, VariableData};

pub type Result<T> = std::result::Result<T, PhiloteError>;
pub type ArrayMap = HashMap<String, ArrayD<f64>>;
pub type PartialMap = HashMap<(String, String), ArrayD<f64>>;
