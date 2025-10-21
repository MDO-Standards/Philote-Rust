//! Utility functions for array and variable operations
//!
//! This module provides helper functions for common operations on N-dimensional arrays
//! and discipline variables. These utilities simplify array manipulation, memory
//! allocation, and validation tasks.
//!
//! # Key Functions
//!
//! - **Array manipulation**: [`create_flattened_view`], [`get_flattened_view_mut`]
//! - **Memory allocation**: [`preallocate_arrays`], [`preallocate_partials`]
//! - **Validation**: `validate_array_shapes`, `validate_variable_names`
//! - **Streaming**: `collect_streamed_arrays`, `collect_streamed_partials`
//!
//! # Example
//!
//! ```rust
//! use philote::utils::preallocate_arrays;
//! use philote::philote_info::{VariableMetaData, VariableType};
//!
//! let var_meta = vec![
//!     VariableMetaData {
//!         name: "x".to_string(),
//!         r#type: VariableType::KInput as i32,
//!         shape: vec![3],
//!         units: "m".to_string(),
//!     },
//! ];
//!
//! let arrays = preallocate_arrays(&var_meta, None).unwrap();
//! assert_eq!(arrays["x"].shape(), &[3]);
//! ```

use ndarray::{ArrayD, ArrayViewMutD};
use std::collections::HashMap;

use crate::philote_info::{VariableMetaData, VariableType};
use crate::types::{ArrayChunker, ArrayData};
use crate::{ArrayMap, PartialMap, PhiloteError, Result};

pub fn create_flattened_view(array: &ArrayD<f64>) -> Vec<f64> {
    array.iter().copied().collect()
}

pub fn get_flattened_view_mut(array: &mut ArrayD<f64>) -> ArrayViewMutD<'_, f64> {
    array.view_mut()
}

pub fn preallocate_arrays(
    var_meta: &[VariableMetaData],
    filter_type: Option<VariableType>,
) -> Result<ArrayMap> {
    let mut arrays = HashMap::new();

    for var in var_meta {
        let var_type = VariableType::try_from(var.r#type).map_err(|_| {
            PhiloteError::InvalidVariableType(format!("Invalid type: {}", var.r#type))
        })?;

        if let Some(filter) = filter_type {
            if var_type != filter {
                continue;
            }
        }

        let shape: Vec<usize> = var.shape.iter().map(|&s| s as usize).collect();
        let array = ArrayD::zeros(shape);
        arrays.insert(var.name.clone(), array);
    }

    Ok(arrays)
}

pub fn preallocate_partials(
    var_meta: &[VariableMetaData],
    partials_meta: &[(String, String)],
) -> Result<PartialMap> {
    let mut partials = HashMap::new();

    let var_shapes: HashMap<String, Vec<usize>> = var_meta
        .iter()
        .map(|var| {
            let shape: Vec<usize> = var.shape.iter().map(|&s| s as usize).collect();
            (var.name.clone(), shape)
        })
        .collect();

    for (func_name, var_name) in partials_meta {
        let func_shape = var_shapes
            .get(func_name)
            .ok_or_else(|| PhiloteError::VariableNotFound(func_name.clone()))?;

        let var_shape = var_shapes
            .get(var_name)
            .ok_or_else(|| PhiloteError::VariableNotFound(var_name.clone()))?;

        let partial_shape = calculate_partial_shape(func_shape, var_shape);
        let partial_array = ArrayD::zeros(partial_shape);

        partials.insert((func_name.clone(), var_name.clone()), partial_array);
    }

    Ok(partials)
}

fn calculate_partial_shape(func_shape: &[usize], var_shape: &[usize]) -> Vec<usize> {
    match (func_shape, var_shape) {
        // Both scalar
        ([1], [1]) => vec![1],
        // Function is scalar, variable is not
        ([1], var_shape) => var_shape.to_vec(),
        // Variable is scalar, function is not
        (func_shape, [1]) => func_shape.to_vec(),
        // Both are non-scalar
        (func_shape, var_shape) => {
            let mut shape = func_shape.to_vec();
            shape.extend_from_slice(var_shape);
            shape
        }
    }
}

pub fn chunk_arrays_for_streaming(
    arrays: &ArrayMap,
    var_type: VariableType,
    chunk_size: usize,
) -> Vec<ArrayData> {
    let chunker = ArrayChunker::new(chunk_size);
    let mut all_chunks = Vec::new();

    for (name, array) in arrays {
        let flat_data = create_flattened_view(array);
        let chunks = chunker.chunk_array(name, &flat_data, var_type);
        all_chunks.extend(chunks);
    }

    all_chunks
}

pub fn reassemble_arrays_from_chunks(chunks: &[ArrayData]) -> Result<ArrayMap> {
    let mut array_data: HashMap<String, (Vec<f64>, Vec<usize>)> = HashMap::new();

    // Group chunks by array name and collect data
    for chunk in chunks {
        let entry = array_data.entry(chunk.name.clone()).or_default();

        // Extend the data vector
        entry.0.extend(&chunk.data);

        // For now, we assume shape information comes from elsewhere
        // This is a simplification - in practice, we'd need to track shapes
    }

    // This is incomplete - we need shape information to properly reconstruct arrays
    // For now, return an error indicating this needs to be implemented
    Err(PhiloteError::not_implemented(
        "reassemble_arrays_from_chunks",
    ))
}

pub fn validate_array_shapes(arrays: &ArrayMap, expected_meta: &[VariableMetaData]) -> Result<()> {
    let expected_shapes: HashMap<String, Vec<usize>> = expected_meta
        .iter()
        .map(|var| {
            let shape: Vec<usize> = var.shape.iter().map(|&s| s as usize).collect();
            (var.name.clone(), shape)
        })
        .collect();

    for (name, array) in arrays {
        if let Some(expected_shape) = expected_shapes.get(name) {
            let actual_shape = array.shape();
            if actual_shape != expected_shape.as_slice() {
                return Err(PhiloteError::ShapeMismatch {
                    expected: expected_shape.clone(),
                    actual: actual_shape.to_vec(),
                });
            }
        }
    }

    Ok(())
}

pub struct PairDict<T> {
    data: HashMap<(String, String), T>,
}

impl<T> PairDict<T> {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    pub fn insert(&mut self, key: (String, String), value: T) -> Option<T> {
        self.data.insert(key, value)
    }

    pub fn get(&self, key: &(String, String)) -> Option<&T> {
        self.data.get(key)
    }

    pub fn get_mut(&mut self, key: &(String, String)) -> Option<&mut T> {
        self.data.get_mut(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&(String, String), &T)> {
        self.data.iter()
    }

    pub fn keys(&self) -> impl Iterator<Item = &(String, String)> {
        self.data.keys()
    }

    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.data.values()
    }
}

impl<T> Default for PairDict<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> std::ops::Index<&(String, String)> for PairDict<T> {
    type Output = T;

    fn index(&self, key: &(String, String)) -> &Self::Output {
        &self.data[key]
    }
}

impl<T> std::ops::IndexMut<&(String, String)> for PairDict<T> {
    fn index_mut(&mut self, key: &(String, String)) -> &mut Self::Output {
        self.data.get_mut(key).expect("Key not found")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    #[test]
    fn test_create_flattened_view() {
        let arr = ArrayD::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let flat = create_flattened_view(&arr);
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_create_flattened_view_1d() {
        let arr = ArrayD::from_shape_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let flat = create_flattened_view(&arr);
        assert_eq!(flat, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_preallocate_arrays_no_filter() {
        let meta = vec![
            VariableMetaData {
                name: "x".to_string(),
                r#type: VariableType::KInput as i32,
                shape: vec![2, 3],
                units: "m".to_string(),
            },
            VariableMetaData {
                name: "y".to_string(),
                r#type: VariableType::KOutput as i32,
                shape: vec![3],
                units: "kg".to_string(),
            },
        ];
        let arrays = preallocate_arrays(&meta, None).unwrap();
        assert_eq!(arrays.len(), 2);
        assert_eq!(arrays["x"].shape(), &[2, 3]);
        assert_eq!(arrays["y"].shape(), &[3]);
    }

    #[test]
    fn test_preallocate_arrays_with_filter() {
        let meta = vec![
            VariableMetaData {
                name: "x".to_string(),
                r#type: VariableType::KInput as i32,
                shape: vec![2],
                units: "".to_string(),
            },
            VariableMetaData {
                name: "y".to_string(),
                r#type: VariableType::KOutput as i32,
                shape: vec![3],
                units: "".to_string(),
            },
        ];
        let arrays = preallocate_arrays(&meta, Some(VariableType::KInput)).unwrap();
        assert_eq!(arrays.len(), 1);
        assert!(arrays.contains_key("x"));
        assert!(!arrays.contains_key("y"));
    }

    #[test]
    fn test_preallocate_arrays_invalid_type() {
        let meta = vec![VariableMetaData {
            name: "x".to_string(),
            r#type: 999, // Invalid type
            shape: vec![2],
            units: "".to_string(),
        }];
        let result = preallocate_arrays(&meta, None);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PhiloteError::InvalidVariableType(_)
        ));
    }

    #[test]
    fn test_preallocate_partials_scalar_to_scalar() {
        let var_meta = vec![
            VariableMetaData {
                name: "f".to_string(),
                r#type: VariableType::KOutput as i32,
                shape: vec![1],
                units: "".to_string(),
            },
            VariableMetaData {
                name: "x".to_string(),
                r#type: VariableType::KInput as i32,
                shape: vec![1],
                units: "".to_string(),
            },
        ];
        let partials_meta = vec![("f".to_string(), "x".to_string())];
        let partials = preallocate_partials(&var_meta, &partials_meta).unwrap();
        assert_eq!(partials.len(), 1);
        assert_eq!(partials[&("f".to_string(), "x".to_string())].shape(), &[1]);
    }

    #[test]
    fn test_preallocate_partials_vector_to_vector() {
        let var_meta = vec![
            VariableMetaData {
                name: "f".to_string(),
                r#type: VariableType::KOutput as i32,
                shape: vec![3],
                units: "".to_string(),
            },
            VariableMetaData {
                name: "x".to_string(),
                r#type: VariableType::KInput as i32,
                shape: vec![2],
                units: "".to_string(),
            },
        ];
        let partials_meta = vec![("f".to_string(), "x".to_string())];
        let partials = preallocate_partials(&var_meta, &partials_meta).unwrap();
        // Shape should be [3, 2] for df/dx
        assert_eq!(
            partials[&("f".to_string(), "x".to_string())].shape(),
            &[3, 2]
        );
    }

    #[test]
    fn test_preallocate_partials_missing_variable() {
        let var_meta = vec![VariableMetaData {
            name: "f".to_string(),
            r#type: VariableType::KOutput as i32,
            shape: vec![1],
            units: "".to_string(),
        }];
        let partials_meta = vec![("f".to_string(), "x".to_string())];
        let result = preallocate_partials(&var_meta, &partials_meta);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PhiloteError::VariableNotFound(_)
        ));
    }

    #[test]
    fn test_calculate_partial_shape() {
        assert_eq!(calculate_partial_shape(&[1], &[1]), vec![1]);
        assert_eq!(calculate_partial_shape(&[1], &[3]), vec![3]);
        assert_eq!(calculate_partial_shape(&[4], &[1]), vec![4]);
        assert_eq!(calculate_partial_shape(&[3], &[2]), vec![3, 2]);
        assert_eq!(calculate_partial_shape(&[2, 3], &[4, 5]), vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_chunk_arrays_for_streaming() {
        let mut arrays = HashMap::new();
        arrays.insert(
            "x".to_string(),
            ArrayD::from_shape_vec(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        arrays.insert(
            "y".to_string(),
            ArrayD::from_shape_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
        );
        let chunks = chunk_arrays_for_streaming(&arrays, VariableType::KInput, 3);
        // x: 6 elements / 3 per chunk = 2 chunks
        // y: 4 elements / 3 per chunk = 2 chunks (3 + 1)
        // Total: 4 chunks
        assert_eq!(chunks.len(), 4);
        // Verify all chunks have correct type
        for chunk in &chunks {
            assert_eq!(chunk.var_type, VariableType::KInput);
        }
    }

    #[test]
    fn test_validate_array_shapes_valid() {
        let mut arrays = HashMap::new();
        arrays.insert("x".to_string(), ArrayD::zeros(vec![2, 3]));
        arrays.insert("y".to_string(), ArrayD::zeros(vec![4]));
        let meta = vec![
            VariableMetaData {
                name: "x".to_string(),
                r#type: VariableType::KInput as i32,
                shape: vec![2, 3],
                units: "".to_string(),
            },
            VariableMetaData {
                name: "y".to_string(),
                r#type: VariableType::KOutput as i32,
                shape: vec![4],
                units: "".to_string(),
            },
        ];
        let result = validate_array_shapes(&arrays, &meta);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_array_shapes_mismatch() {
        let mut arrays = HashMap::new();
        arrays.insert("x".to_string(), ArrayD::zeros(vec![2, 3]));
        let meta = vec![VariableMetaData {
            name: "x".to_string(),
            r#type: VariableType::KInput as i32,
            shape: vec![3, 2], // Wrong shape
            units: "".to_string(),
        }];
        let result = validate_array_shapes(&arrays, &meta);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PhiloteError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn test_pair_dict_new() {
        let dict: PairDict<i32> = PairDict::new();
        assert_eq!(dict.data.len(), 0);
    }

    #[test]
    fn test_pair_dict_insert_and_get() {
        let mut dict = PairDict::new();
        dict.insert(("f".to_string(), "x".to_string()), 42);
        assert_eq!(dict.get(&("f".to_string(), "x".to_string())), Some(&42));
        assert_eq!(dict.get(&("g".to_string(), "x".to_string())), None);
    }

    #[test]
    fn test_pair_dict_get_mut() {
        let mut dict = PairDict::new();
        dict.insert(("f".to_string(), "x".to_string()), 10);
        if let Some(val) = dict.get_mut(&("f".to_string(), "x".to_string())) {
            *val = 20;
        }
        assert_eq!(dict.get(&("f".to_string(), "x".to_string())), Some(&20));
    }

    #[test]
    fn test_pair_dict_index() {
        let mut dict = PairDict::new();
        let key = ("f".to_string(), "x".to_string());
        dict.insert(key.clone(), 100);
        assert_eq!(dict[&key], 100);
    }

    #[test]
    fn test_pair_dict_index_mut() {
        let mut dict = PairDict::new();
        let key = ("f".to_string(), "x".to_string());
        dict.insert(key.clone(), 50);
        dict[&key] = 75;
        assert_eq!(dict[&key], 75);
    }

    #[test]
    fn test_pair_dict_iter() {
        let mut dict = PairDict::new();
        dict.insert(("f1".to_string(), "x1".to_string()), 1);
        dict.insert(("f2".to_string(), "x2".to_string()), 2);
        let count = dict.iter().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_pair_dict_keys() {
        let mut dict = PairDict::new();
        dict.insert(("f".to_string(), "x".to_string()), 1);
        dict.insert(("g".to_string(), "y".to_string()), 2);
        let keys: Vec<_> = dict.keys().collect();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_pair_dict_values() {
        let mut dict = PairDict::new();
        dict.insert(("f".to_string(), "x".to_string()), 10);
        dict.insert(("g".to_string(), "y".to_string()), 20);
        let sum: i32 = dict.values().sum();
        assert_eq!(sum, 30);
    }

    #[test]
    fn test_pair_dict_default() {
        let dict: PairDict<String> = PairDict::default();
        assert_eq!(dict.data.len(), 0);
    }
}
