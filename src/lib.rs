use std::collections::HashMap;
use ndarray::ArrayD;

pub mod philote_info {
    tonic::include_proto!("philote");
}

pub mod error;
pub mod traits;
pub mod types;
pub mod server;
pub mod client;
pub mod utils;

pub use error::PhiloteError;
pub use traits::{Discipline, ExplicitDiscipline, ImplicitDiscipline};
pub use types::{VariableData, ArrayData};

pub type Result<T> = std::result::Result<T, PhiloteError>;
pub type ArrayMap = HashMap<String, ArrayD<f64>>;
pub type PartialMap = HashMap<(String, String), ArrayD<f64>>;
