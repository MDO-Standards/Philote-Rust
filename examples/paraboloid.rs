//! Runs the paraboloid discipline locally, without a server.
//!
//! Useful for checking discipline logic in isolation. See `paraboloid_server` and
//! `paraboloid_client` for the networked version.
//!
//! ```text
//! cargo run --example paraboloid
//! ```

use std::collections::HashMap;

use philote_mdo::examples::{scalar, Paraboloid};
use philote_mdo::traits::{Discipline, ExplicitDiscipline};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut discipline = Paraboloid::new();
    discipline.setup()?;
    discipline.setup_partials()?;

    println!("Variables:");
    for var in discipline.get_variable_definitions()? {
        println!("  {} shape={:?} units='{}'", var.name, var.shape, var.units);
    }

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), scalar(1.0));
    inputs.insert("y".to_string(), scalar(2.0));

    let outputs = discipline.compute(&inputs).await?;
    println!("\nf_xy(1, 2) = {}", outputs["f_xy"][[0]]);

    let partials = discipline.compute_partials(&inputs).await?;
    println!(
        "df/dx = {}, df/dy = {}",
        partials[&("f_xy".to_string(), "x".to_string())][[0]],
        partials[&("f_xy".to_string(), "y".to_string())][[0]],
    );

    Ok(())
}
