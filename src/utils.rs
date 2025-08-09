use std::collections::HashMap;
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};

use crate::{Result, PhiloteError, ArrayMap, PartialMap};
use crate::types::{VariableData, ArrayData, ArrayChunker};
use crate::philote_info::{VariableType, VariableMetaData};

pub fn create_flattened_view(array: &ArrayD<f64>) -> Vec<f64> {
    array.iter().copied().collect()
}

pub fn get_flattened_view_mut(array: &mut ArrayD<f64>) -> ArrayViewMutD<f64> {
    array.view_mut()
}

pub fn preallocate_arrays(var_meta: &[VariableMetaData], filter_type: Option<VariableType>) -> Result<ArrayMap> {
    let mut arrays = HashMap::new();
    
    for var in var_meta {
        let var_type = VariableType::try_from(var.r#type)
            .map_err(|_| PhiloteError::InvalidVariableType(format!("Invalid type: {}", var.r#type)))?;
        
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
    Err(PhiloteError::not_implemented("reassemble_arrays_from_chunks"))
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