use std::collections::HashMap;
use tonic::transport::Channel;
use tonic::{Request, Streaming};
use tokio_stream::StreamExt;

use crate::{Result, PhiloteError, ArrayMap, PartialMap};
use crate::client::base::DisciplineClient;
use crate::types::{ArrayData, ArrayChunker, StreamOptions};
use crate::philote_info::{
    implicit_service_client::ImplicitServiceClient,
    Array, VariableType,
};

pub struct ImplicitClient {
    base_client: DisciplineClient,
    implicit_client: ImplicitServiceClient<Channel>,
    stream_options: StreamOptions,
}

impl ImplicitClient {
    pub async fn connect<T>(dst: T) -> Result<Self>
    where
        T: std::convert::TryInto<tonic::transport::Endpoint> + Clone,
        T::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        let base_client = DisciplineClient::connect(dst.clone()).await?;
        
        let channel = tonic::transport::Endpoint::new(dst)
            .map_err(|e| PhiloteError::config_error(format!("Invalid endpoint: {:?}", e)))?
            .connect()
            .await
            .map_err(|e| PhiloteError::config_error(format!("Failed to connect: {}", e)))?;
        
        let implicit_client = ImplicitServiceClient::new(channel);
        
        Ok(Self {
            base_client,
            implicit_client,
            stream_options: StreamOptions::default(),
        })
    }
    
    pub fn with_stream_options(mut self, options: StreamOptions) -> Self {
        self.stream_options = options;
        self.base_client = self.base_client.with_stream_options(options);
        self
    }
    
    // Delegate base client methods
    pub async fn get_info(&mut self) -> Result<crate::philote_info::DisciplineProperties> {
        self.base_client.get_info().await
    }
    
    pub async fn set_stream_options(&mut self, options: StreamOptions) -> Result<()> {
        self.stream_options = options;
        self.base_client.set_stream_options(options).await
    }
    
    pub async fn get_available_options(&mut self) -> Result<HashMap<String, String>> {
        self.base_client.get_available_options().await
    }
    
    pub async fn set_options(&mut self, options: HashMap<String, serde_json::Value>) -> Result<()> {
        self.base_client.set_options(options).await
    }
    
    pub async fn setup(&mut self) -> Result<()> {
        self.base_client.setup().await
    }
    
    pub async fn get_variable_definitions(&mut self) -> Result<Vec<crate::philote_info::VariableMetaData>> {
        self.base_client.get_variable_definitions().await
    }
    
    pub async fn get_partial_definitions(&mut self) -> Result<Vec<crate::philote_info::PartialsMetaData>> {
        self.base_client.get_partial_definitions().await
    }
    
    // Implicit-specific methods
    pub async fn compute_residuals(&mut self, inputs: &ArrayMap, outputs: &ArrayMap) -> Result<ArrayMap> {
        // Create combined input stream with both inputs and outputs
        let input_stream = self.create_combined_stream(inputs, outputs).await?;
        
        // Call the compute residuals function
        let response = self.implicit_client
            .compute_residuals(Request::new(input_stream))
            .await?;
        
        // Process the residual stream
        let residual_stream = response.into_inner();
        let residuals = self.process_output_stream(residual_stream).await?;
        
        Ok(residuals)
    }
    
    pub async fn solve_residuals(&mut self, inputs: &ArrayMap) -> Result<ArrayMap> {
        // Convert inputs to streaming arrays
        let input_stream = self.create_input_stream(inputs, VariableType::KInput).await?;
        
        // Call the solve residuals function
        let response = self.implicit_client
            .solve_residuals(Request::new(input_stream))
            .await?;
        
        // Process the output stream
        let output_stream = response.into_inner();
        let outputs = self.process_output_stream(output_stream).await?;
        
        Ok(outputs)
    }
    
    pub async fn compute_residual_gradients(&mut self, inputs: &ArrayMap, outputs: &ArrayMap) -> Result<PartialMap> {
        // Create combined input stream with both inputs and outputs
        let input_stream = self.create_combined_stream(inputs, outputs).await?;
        
        // Call the compute residual gradients function
        let response = self.implicit_client
            .compute_residual_gradients(Request::new(input_stream))
            .await?;
        
        // Process the partial stream
        let partial_stream = response.into_inner();
        let partials = self.process_partial_stream(partial_stream).await?;
        
        Ok(partials)
    }
    
    async fn create_input_stream(
        &self,
        inputs: &ArrayMap,
        var_type: VariableType,
    ) -> Result<impl futures_util::Stream<Item = Array> + Send> {
        let chunker = ArrayChunker::new(self.stream_options.max_double_per_slice);
        let mut all_chunks = Vec::new();
        
        for (name, array) in inputs {
            let flat_data = crate::utils::create_flattened_view(array);
            let chunks = chunker.chunk_array(name, &flat_data, var_type);
            
            for chunk in chunks {
                all_chunks.push(Array::from(chunk));
            }
        }
        
        Ok(tokio_stream::iter(all_chunks))
    }
    
    async fn create_combined_stream(
        &self,
        inputs: &ArrayMap,
        outputs: &ArrayMap,
    ) -> Result<impl futures_util::Stream<Item = Array> + Send> {
        let chunker = ArrayChunker::new(self.stream_options.max_double_per_slice);
        let mut all_chunks = Vec::new();
        
        // Add input chunks
        for (name, array) in inputs {
            let flat_data = crate::utils::create_flattened_view(array);
            let chunks = chunker.chunk_array(name, &flat_data, VariableType::KInput);
            
            for chunk in chunks {
                all_chunks.push(Array::from(chunk));
            }
        }
        
        // Add output chunks
        for (name, array) in outputs {
            let flat_data = crate::utils::create_flattened_view(array);
            let chunks = chunker.chunk_array(name, &flat_data, VariableType::KOutput);
            
            for chunk in chunks {
                all_chunks.push(Array::from(chunk));
            }
        }
        
        Ok(tokio_stream::iter(all_chunks))
    }
    
    async fn process_output_stream(
        &self,
        mut stream: Streaming<Array>,
    ) -> Result<ArrayMap> {
        let mut array_chunks: HashMap<String, Vec<ArrayData>> = HashMap::new();
        
        while let Some(array_msg) = stream.next().await {
            let array = array_msg?;
            let array_data = ArrayData::try_from(array)?;
            
            array_chunks.entry(array_data.name.clone())
                .or_default()
                .push(array_data);
        }
        
        // Reconstruct arrays from chunks
        let mut outputs = HashMap::new();
        for (name, chunks) in array_chunks {
            let array = self.reconstruct_array_from_chunks(&chunks)?;
            outputs.insert(name, array);
        }
        
        Ok(outputs)
    }
    
    async fn process_partial_stream(
        &self,
        mut stream: Streaming<Array>,
    ) -> Result<PartialMap> {
        let mut partial_chunks: HashMap<(String, String), Vec<ArrayData>> = HashMap::new();
        
        while let Some(array_msg) = stream.next().await {
            let array = array_msg?;
            let array_data = ArrayData::try_from(array)?;
            
            if let Some(subname) = &array_data.subname {
                let key = (array_data.name.clone(), subname.clone());
                partial_chunks.entry(key).or_default().push(array_data);
            }
        }
        
        // Reconstruct partials from chunks
        let mut partials = HashMap::new();
        for (key, chunks) in partial_chunks {
            let array = self.reconstruct_array_from_chunks(&chunks)?;
            partials.insert(key, array);
        }
        
        Ok(partials)
    }
    
    fn reconstruct_array_from_chunks(
        &self,
        chunks: &[ArrayData],
    ) -> Result<ndarray::ArrayD<f64>> {
        if chunks.is_empty() {
            return Err(PhiloteError::array_error("No chunks to reconstruct"));
        }
        
        // Sort chunks by start index
        let mut sorted_chunks = chunks.to_vec();
        sorted_chunks.sort_by_key(|c| c.start);
        
        // Calculate total size
        let total_size = sorted_chunks.last().unwrap().end + 1;
        let mut data = vec![0.0; total_size];
        
        // Fill in data from chunks
        for chunk in &sorted_chunks {
            let chunk_len = chunk.data.len();
            let expected_len = chunk.end - chunk.start + 1;
            
            if chunk_len != expected_len {
                return Err(PhiloteError::array_error(format!(
                    "Chunk size mismatch: expected {}, got {}",
                    expected_len, chunk_len
                )));
            }
            
            for (i, &value) in chunk.data.iter().enumerate() {
                data[chunk.start + i] = value;
            }
        }
        
        // For now, assume 1D arrays - in a full implementation, we'd need shape info
        Ok(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[data.len()]), data)
            .map_err(|e| PhiloteError::array_error(format!("Failed to create array: {}", e)))?)
    }
}

impl Clone for ImplicitClient {
    fn clone(&self) -> Self {
        Self {
            base_client: self.base_client.clone(),
            implicit_client: self.implicit_client.clone(),
            stream_options: self.stream_options.clone(),
        }
    }
}