//! Input validation helpers
//!
//! These mirror the validation layer in Philote-Python (`philote_mdo/utils/validation.py`)
//! so that both implementations reject the same inputs with comparable messages.
//!
//! # Deliberate omissions
//!
//! Python additionally provides `validate_units`, `validate_is_dict`, and
//! `validate_numpy_array`. Those exist to recover type guarantees that a dynamically
//! typed language cannot express statically; in Rust the corresponding parameters are
//! already `&str`, `&HashMap<..>`, and `ArrayD<f64>`, so the checks would be
//! unreachable. They are intentionally not ported.

use crate::{PhiloteError, Result};

/// Option type names accepted by `add_option`.
///
/// These are the canonical names used by Philote-Python. The server additionally
/// accepts the aliases `double`, `string`, and `struct` when mapping to the wire
/// [`DataType`](crate::philote_info::DataType) enum.
pub const VALID_OPTION_TYPES: [&str; 5] = ["bool", "int", "float", "str", "dict"];

/// Validates that a variable or option name is non-empty.
pub fn validate_name(name: &str, context: &str) -> Result<()> {
    if name.is_empty() {
        return Err(PhiloteError::validation(
            context,
            "'name' must not be empty",
        ));
    }
    Ok(())
}

/// Validates that a shape is non-empty and has strictly positive dimensions.
pub fn validate_shape(shape: &[usize], context: &str) -> Result<()> {
    if shape.is_empty() {
        return Err(PhiloteError::validation(
            context,
            "'shape' must not be empty",
        ));
    }
    for (i, &dim) in shape.iter().enumerate() {
        if dim == 0 {
            return Err(PhiloteError::validation(
                context,
                format!("all elements of 'shape' must be positive, but element {i} is 0"),
            ));
        }
    }
    Ok(())
}

/// Validates that an option type is one of [`VALID_OPTION_TYPES`].
pub fn validate_option_type(type_str: &str, name: &str) -> Result<()> {
    if !VALID_OPTION_TYPES.contains(&type_str) {
        return Err(PhiloteError::validation(
            "add_option",
            format!(
                "invalid type '{type_str}' for option '{name}'. Allowed types are: {:?}",
                VALID_OPTION_TYPES
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_non_empty_name() {
        assert!(validate_name("x", "add_input").is_ok());
    }

    #[test]
    fn rejects_empty_name() {
        let err = validate_name("", "add_input").unwrap_err();
        assert!(matches!(err, PhiloteError::Validation { .. }));
        assert!(err.to_string().contains("add_input"));
    }

    #[test]
    fn accepts_positive_shape() {
        assert!(validate_shape(&[2, 3], "add_input").is_ok());
    }

    #[test]
    fn rejects_empty_shape() {
        assert!(validate_shape(&[], "add_input").is_err());
    }

    #[test]
    fn rejects_zero_dimension() {
        let err = validate_shape(&[2, 0], "add_input").unwrap_err();
        assert!(err.to_string().contains("element 1"));
    }

    #[test]
    fn accepts_all_canonical_option_types() {
        for t in VALID_OPTION_TYPES {
            assert!(validate_option_type(t, "opt").is_ok(), "{t} rejected");
        }
    }

    #[test]
    fn rejects_unknown_option_type() {
        let err = validate_option_type("complex", "opt").unwrap_err();
        assert!(err.to_string().contains("complex"));
    }
}
