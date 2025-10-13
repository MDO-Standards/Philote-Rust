use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};

use crate::{Result, PhiloteError};
use crate::philote_info::{Array, VariableType, PartialsMetaData};

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
    
    pub fn from_flat(name: String, flat_data: &[f64], shape: &[usize], units: String, var_type: VariableType) -> Result<Self> {
        let expected_size: usize = shape.iter().product();
        if flat_data.len() != expected_size {
            return Err(PhiloteError::ShapeMismatch {
                expected: vec![expected_size],
                actual: vec![flat_data.len()],
            });
        }
        
        let data = ArrayD::from_shape_vec(shape, flat_data.to_vec())
            .map_err(|e| PhiloteError::array_error(format!("Failed to create array from flat data: {}", e)))?;
        
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
        let var_type = VariableType::try_from(array.r#type)
            .map_err(|_| PhiloteError::InvalidVariableType(format!("Invalid type: {}", array.r#type)))?;
        
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