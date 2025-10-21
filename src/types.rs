//! Data types and Protocol Buffer conversions
//!
//! This module provides Rust data structures for working with discipline variables
//! and arrays, along with conversions to/from Protocol Buffer messages.
//!
//! # Key Types
//!
//! - [`VariableData`] - Complete variable with name, data array, units, and type
//! - [`ArrayData`] - Chunked array data for streaming transmission
//! - [`StreamOptions`] - Configuration for array streaming behavior
//! - [`ArrayChunker`] - Utility for splitting large arrays into chunks
//!
//! # Protocol Buffer Conversion
//!
//! Types implement `From` and `TryFrom` traits for seamless conversion between
//! Rust structures and Protocol Buffer messages, enabling efficient gRPC communication.

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};

use crate::philote_info::{Array, PartialsMetaData, VariableType};
use crate::{PhiloteError, Result};

#[derive(Debug, Clone)]
pub struct VariableData {
    pub name: String,
    pub data: ArrayD<f64>,
    pub units: String,
    pub var_type: VariableType,
}

impl VariableData {
    pub fn new(name: String, data: ArrayD<f64>, units: String, var_type: VariableType) -> Self {
        Self {
            name,
            data,
            units,
            var_type,
        }
    }

    pub fn zeros(name: String, shape: &[usize], units: String, var_type: VariableType) -> Self {
        let data = ArrayD::zeros(shape);
        Self::new(name, data, units, var_type)
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn view(&self) -> ArrayViewD<'_, f64> {
        self.data.view()
    }

    pub fn view_mut(&mut self) -> ArrayViewMutD<'_, f64> {
        self.data.view_mut()
    }

    pub fn flatten(&self) -> Vec<f64> {
        self.data.iter().copied().collect()
    }

    pub fn from_flat(
        name: String,
        flat_data: &[f64],
        shape: &[usize],
        units: String,
        var_type: VariableType,
    ) -> Result<Self> {
        let expected_size: usize = shape.iter().product();
        if flat_data.len() != expected_size {
            return Err(PhiloteError::ShapeMismatch {
                expected: vec![expected_size],
                actual: vec![flat_data.len()],
            });
        }

        let data = ArrayD::from_shape_vec(shape, flat_data.to_vec()).map_err(|e| {
            PhiloteError::array_error(format!("Failed to create array from flat data: {}", e))
        })?;

        Ok(Self::new(name, data, units, var_type))
    }
}

#[derive(Debug, Clone)]
pub struct ArrayData {
    pub name: String,
    pub subname: Option<String>,
    pub start: usize,
    pub end: usize,
    pub var_type: VariableType,
    pub data: Vec<f64>,
}

impl ArrayData {
    pub fn new(
        name: String,
        subname: Option<String>,
        start: usize,
        end: usize,
        var_type: VariableType,
        data: Vec<f64>,
    ) -> Self {
        Self {
            name,
            subname,
            start,
            end,
            var_type,
            data,
        }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

impl From<ArrayData> for Array {
    fn from(array_data: ArrayData) -> Self {
        Array {
            name: array_data.name,
            subname: array_data.subname.unwrap_or_default(),
            start: array_data.start as i64,
            end: array_data.end as i64,
            r#type: array_data.var_type.into(),
            data: array_data.data,
        }
    }
}

impl TryFrom<Array> for ArrayData {
    type Error = PhiloteError;

    fn try_from(array: Array) -> Result<Self> {
        let var_type = VariableType::try_from(array.r#type).map_err(|_| {
            PhiloteError::InvalidVariableType(format!("Invalid type: {}", array.r#type))
        })?;

        if array.data.is_empty() {
            return Err(PhiloteError::array_error("Array contains no data"));
        }

        let subname = if array.subname.is_empty() {
            None
        } else {
            Some(array.subname)
        };

        Ok(ArrayData::new(
            array.name,
            subname,
            array.start as usize,
            array.end as usize,
            var_type,
            array.data,
        ))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StreamOptions {
    pub max_double_per_slice: usize,
    pub max_int_per_slice: usize,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            max_double_per_slice: 1000,
            max_int_per_slice: 1000,
        }
    }
}

impl From<crate::philote_info::StreamOptions> for StreamOptions {
    fn from(opts: crate::philote_info::StreamOptions) -> Self {
        Self {
            max_double_per_slice: opts.num_double as usize,
            max_int_per_slice: 1000, // Default since it's not in the proto
        }
    }
}

impl From<StreamOptions> for crate::philote_info::StreamOptions {
    fn from(opts: StreamOptions) -> Self {
        crate::philote_info::StreamOptions {
            num_double: opts.max_double_per_slice as i64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PartialsInfo {
    pub name: String,
    pub subname: String,
    pub shape: Vec<usize>,
}

impl PartialsInfo {
    pub fn new(name: String, subname: String, shape: Vec<usize>) -> Self {
        Self {
            name,
            subname,
            shape,
        }
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

impl From<PartialsInfo> for PartialsMetaData {
    fn from(info: PartialsInfo) -> Self {
        PartialsMetaData {
            name: info.name,
            subname: info.subname,
            shape: info.shape.into_iter().map(|s| s as i64).collect(),
        }
    }
}

pub struct ArrayChunker {
    chunk_size: usize,
}

impl ArrayChunker {
    pub fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }

    pub fn chunk_array(&self, name: &str, data: &[f64], var_type: VariableType) -> Vec<ArrayData> {
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < data.len() {
            let end = std::cmp::min(start + self.chunk_size, data.len()) - 1;
            let chunk_data = data[start..=end].to_vec();

            chunks.push(ArrayData::new(
                name.to_string(),
                None,
                start,
                end,
                var_type,
                chunk_data,
            ));

            start = end + 1;
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    // VariableData tests
    #[test]
    fn test_variable_data_new() {
        let data = ArrayD::from_elem(vec![2, 3], 1.0);
        let var = VariableData::new("x".to_string(), data, "m".to_string(), VariableType::KInput);
        assert_eq!(var.name, "x");
        assert_eq!(var.units, "m");
        assert_eq!(var.shape(), &[2, 3]);
        assert_eq!(var.size(), 6);
    }

    #[test]
    fn test_variable_data_zeros() {
        let var = VariableData::zeros(
            "y".to_string(),
            &[3, 4],
            "kg".to_string(),
            VariableType::KOutput,
        );
        assert_eq!(var.name, "y");
        assert_eq!(var.shape(), &[3, 4]);
        assert_eq!(var.size(), 12);
        assert_eq!(var.flatten(), vec![0.0; 12]);
    }

    #[test]
    fn test_variable_data_flatten() {
        let data = ArrayD::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let var = VariableData::new("x".to_string(), data, "".to_string(), VariableType::KInput);
        let flat = var.flatten();
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_variable_data_from_flat() {
        let flat_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let var = VariableData::from_flat(
            "z".to_string(),
            &flat_data,
            &[2, 3],
            "".to_string(),
            VariableType::KOutput,
        )
        .unwrap();
        assert_eq!(var.shape(), &[2, 3]);
        assert_eq!(var.flatten(), flat_data);
    }

    #[test]
    fn test_variable_data_from_flat_shape_mismatch() {
        let flat_data = vec![1.0, 2.0, 3.0];
        let result = VariableData::from_flat(
            "z".to_string(),
            &flat_data,
            &[2, 3], // Expects 6 elements, not 3
            "".to_string(),
            VariableType::KOutput,
        );
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PhiloteError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn test_variable_data_view_mut() {
        let data = ArrayD::from_elem(vec![2, 2], 0.0);
        let mut var =
            VariableData::new("x".to_string(), data, "".to_string(), VariableType::KInput);
        {
            let mut view = var.view_mut();
            view[[0, 0]] = 5.0;
        }
        assert_eq!(var.data[[0, 0]], 5.0);
    }

    // ArrayData tests
    #[test]
    fn test_array_data_new() {
        let data = vec![1.0, 2.0, 3.0];
        let arr = ArrayData::new(
            "x".to_string(),
            None,
            0,
            2,
            VariableType::KInput,
            data.clone(),
        );
        assert_eq!(arr.name, "x");
        assert_eq!(arr.subname, None);
        assert_eq!(arr.start, 0);
        assert_eq!(arr.end, 2);
        assert_eq!(arr.size(), 3);
        assert_eq!(arr.data, data);
    }

    #[test]
    fn test_array_data_with_subname() {
        let arr = ArrayData::new(
            "f".to_string(),
            Some("x".to_string()),
            0,
            4,
            VariableType::KPartial,
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        );
        assert_eq!(arr.subname, Some("x".to_string()));
    }

    #[test]
    fn test_array_data_to_proto() {
        let arr = ArrayData::new(
            "x".to_string(),
            Some("sub".to_string()),
            0,
            2,
            VariableType::KInput,
            vec![1.0, 2.0, 3.0],
        );
        let proto: Array = arr.into();
        assert_eq!(proto.name, "x");
        assert_eq!(proto.subname, "sub");
        assert_eq!(proto.start, 0);
        assert_eq!(proto.end, 2);
        assert_eq!(proto.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_array_data_from_proto() {
        let proto = Array {
            name: "y".to_string(),
            subname: "".to_string(),
            start: 5,
            end: 9,
            r#type: VariableType::KOutput.into(),
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        };
        let arr = ArrayData::try_from(proto).unwrap();
        assert_eq!(arr.name, "y");
        assert_eq!(arr.subname, None);
        assert_eq!(arr.start, 5);
        assert_eq!(arr.end, 9);
        assert_eq!(arr.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_array_data_from_proto_empty_data() {
        let proto = Array {
            name: "x".to_string(),
            subname: "".to_string(),
            start: 0,
            end: 0,
            r#type: VariableType::KInput.into(),
            data: vec![],
        };
        let result = ArrayData::try_from(proto);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PhiloteError::ArrayError(_)));
    }

    #[test]
    fn test_array_data_from_proto_invalid_type() {
        let proto = Array {
            name: "x".to_string(),
            subname: "".to_string(),
            start: 0,
            end: 2,
            r#type: 999, // Invalid type
            data: vec![1.0, 2.0, 3.0],
        };
        let result = ArrayData::try_from(proto);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PhiloteError::InvalidVariableType(_)
        ));
    }

    // StreamOptions tests
    #[test]
    fn test_stream_options_default() {
        let opts = StreamOptions::default();
        assert_eq!(opts.max_double_per_slice, 1000);
        assert_eq!(opts.max_int_per_slice, 1000);
    }

    #[test]
    fn test_stream_options_to_proto() {
        let opts = StreamOptions {
            max_double_per_slice: 500,
            max_int_per_slice: 250,
        };
        let proto: crate::philote_info::StreamOptions = opts.into();
        assert_eq!(proto.num_double, 500);
    }

    #[test]
    fn test_stream_options_from_proto() {
        let proto = crate::philote_info::StreamOptions { num_double: 750 };
        let opts: StreamOptions = proto.into();
        assert_eq!(opts.max_double_per_slice, 750);
        assert_eq!(opts.max_int_per_slice, 1000); // Default
    }

    #[test]
    fn test_stream_options_copy() {
        let opts1 = StreamOptions::default();
        let opts2 = opts1; // Should copy, not move
        assert_eq!(opts1.max_double_per_slice, opts2.max_double_per_slice);
    }

    // PartialsInfo tests
    #[test]
    fn test_partials_info_new() {
        let info = PartialsInfo::new("f".to_string(), "x".to_string(), vec![3, 4]);
        assert_eq!(info.name, "f");
        assert_eq!(info.subname, "x");
        assert_eq!(info.shape, vec![3, 4]);
        assert_eq!(info.size(), 12);
    }

    #[test]
    fn test_partials_info_size_scalar() {
        let info = PartialsInfo::new("f".to_string(), "x".to_string(), vec![1]);
        assert_eq!(info.size(), 1);
    }

    #[test]
    fn test_partials_info_to_proto() {
        let info = PartialsInfo::new("df".to_string(), "dx".to_string(), vec![2, 3]);
        let proto: PartialsMetaData = info.into();
        assert_eq!(proto.name, "df");
        assert_eq!(proto.subname, "dx");
        assert_eq!(proto.shape, vec![2, 3]);
    }

    // ArrayChunker tests
    #[test]
    fn test_array_chunker_exact_chunks() {
        let chunker = ArrayChunker::new(3);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let chunks = chunker.chunk_array("x", &data, VariableType::KInput);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[0].end, 2);
        assert_eq!(chunks[0].data, vec![1.0, 2.0, 3.0]);
        assert_eq!(chunks[1].start, 3);
        assert_eq!(chunks[1].end, 5);
        assert_eq!(chunks[1].data, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_array_chunker_partial_last_chunk() {
        let chunker = ArrayChunker::new(4);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let chunks = chunker.chunk_array("y", &data, VariableType::KOutput);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(chunks[1].data, vec![5.0]);
        assert_eq!(chunks[1].start, 4);
        assert_eq!(chunks[1].end, 4);
    }

    #[test]
    fn test_array_chunker_single_chunk() {
        let chunker = ArrayChunker::new(10);
        let data = vec![1.0, 2.0, 3.0];
        let chunks = chunker.chunk_array("z", &data, VariableType::KInput);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].data, data);
    }

    #[test]
    fn test_array_chunker_single_element() {
        let chunker = ArrayChunker::new(5);
        let data = vec![42.0];
        let chunks = chunker.chunk_array("a", &data, VariableType::KInput);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[0].end, 0);
        assert_eq!(chunks[0].data, vec![42.0]);
    }

    #[test]
    fn test_array_chunker_preserves_name_and_type() {
        let chunker = ArrayChunker::new(2);
        let data = vec![1.0, 2.0, 3.0];
        let chunks = chunker.chunk_array("test", &data, VariableType::KPartial);
        for chunk in chunks {
            assert_eq!(chunk.name, "test");
            assert_eq!(chunk.var_type, VariableType::KPartial);
        }
    }
}
