//! Wire encoding and decoding for variable streams
//!
//! Every RPC that moves array data does the same two things: split arrays into
//! [`Array`] chunks for transmission, and scatter received chunks back into
//! preallocated N-dimensional arrays. This module owns both directions so the
//! client and server implementations share one encoding.
//!
//! # Chunk index convention
//!
//! `Array.start` and `Array.end` are **inclusive** on both ends, so a chunk carries
//! `end - start + 1` values. This matches the Philote standard and Philote-Python's
//! explicit server. Note that Philote-Python's *implicit* server emits an exclusive
//! `end`; that is a defect on the Python side and is deliberately not reproduced
//! here. See `tests/interop_notes.rs`.
//!
//! # Shape recovery
//!
//! The `Array` message carries no shape, only a flat index range. Both peers
//! therefore reconstruct N-D shape from the declared
//! [`VariableMetaData`], never from the chunks themselves.

use std::collections::{HashMap, HashSet};

use ndarray::ArrayD;
use tokio_stream::StreamExt;
use tonic::Streaming;

use crate::philote_info::{
    variable_message::Payload, Array, DiscreteVariable, PartialsMetaData, VariableMessage,
    VariableMetaData, VariableType,
};
use crate::types::{ArrayChunker, ArrayData};
use crate::utils::{calculate_partial_shape, create_flattened_view, preallocate_arrays};
use crate::{ArrayMap, DiscreteMap, PartialMap, PhiloteError, Result};

/// Write one received chunk into a preallocated array, in row-major flat order.
///
/// The array keeps its declared N-dimensional shape; the chunk addresses a flat
/// range within it.
///
/// Indexing an `ArrayD` with a single `usize` only works for rank-1 arrays — for
/// higher ranks `NdIndex` requires an index whose rank matches the array's, so a
/// flat index silently misses (and trips a `debug_assert` in debug builds). This
/// function goes through the flat slice instead, which is correct at any rank.
pub fn scatter_chunk(target: &mut ArrayD<f64>, chunk: &ArrayData) -> Result<()> {
    let expected = chunk
        .end
        .checked_sub(chunk.start)
        .map(|span| span + 1)
        .ok_or_else(|| {
            // Defensive: `ArrayData::try_from` already rejects end < start, so a
            // chunk off the wire never reaches here. Kept because `scatter_chunk`
            // is public and can be called with a hand-built `ArrayData`.
            PhiloteError::validation(
                "scatter_chunk",
                format!(
                    "chunk for '{}' has end {} before start {}",
                    chunk.name, chunk.end, chunk.start
                ),
            )
        })?;

    if chunk.data.len() != expected {
        return Err(PhiloteError::validation(
            "scatter_chunk",
            format!(
                "chunk size mismatch for '{}': indices {}..={} span {} values, got {}",
                chunk.name,
                chunk.start,
                chunk.end,
                expected,
                chunk.data.len()
            ),
        ));
    }

    let len = target.len();
    if chunk.end >= len {
        return Err(PhiloteError::IndexOutOfBounds {
            index: chunk.end,
            size: len,
        });
    }

    match target.as_slice_mut() {
        Some(slice) => slice[chunk.start..=chunk.end].copy_from_slice(&chunk.data),
        // Non-contiguous layout: fall back to iteration in logical order.
        None => {
            for (elem, &value) in target
                .iter_mut()
                .skip(chunk.start)
                .take(expected)
                .zip(chunk.data.iter())
            {
                *elem = value;
            }
        }
    }
    Ok(())
}

/// Split every array in `arrays` into chunks tagged with `var_type`.
pub fn chunk_arrays(
    arrays: &ArrayMap,
    var_type: VariableType,
    chunk_size: usize,
) -> Vec<ArrayData> {
    let chunker = ArrayChunker::new(chunk_size);
    let mut chunks = Vec::new();
    for (name, array) in arrays {
        let flat = create_flattened_view(array);
        chunks.extend(chunker.chunk_array(name, &flat, var_type));
    }
    chunks
}

/// Split a partials map into `kPartial` chunks carrying `name`/`subname`.
pub fn chunk_partials(partials: &PartialMap, chunk_size: usize) -> Vec<ArrayData> {
    let chunker = ArrayChunker::new(chunk_size);
    let mut chunks = Vec::new();
    for ((func, var), array) in partials {
        let flat = create_flattened_view(array);
        for mut chunk in chunker.chunk_array(func, &flat, VariableType::KPartial) {
            chunk.subname = Some(var.clone());
            chunks.push(chunk);
        }
    }
    chunks
}

fn continuous(chunk: ArrayData) -> VariableMessage {
    VariableMessage {
        payload: Some(Payload::Continuous(Array::from(chunk))),
    }
}

fn discrete(name: &str, value: prost_types::Value, var_type: VariableType) -> VariableMessage {
    VariableMessage {
        payload: Some(Payload::Discrete(DiscreteVariable {
            name: name.to_string(),
            r#type: var_type.into(),
            value: Some(value),
        })),
    }
}

/// Build the request messages a client sends for a compute call.
///
/// `outputs` is `Some` only for implicit RPCs, which send current output values
/// alongside the inputs.
pub fn assemble_input_messages(
    inputs: &ArrayMap,
    outputs: Option<&ArrayMap>,
    discrete_inputs: &DiscreteMap,
    chunk_size: usize,
) -> Vec<VariableMessage> {
    let mut messages: Vec<VariableMessage> = chunk_arrays(inputs, VariableType::KInput, chunk_size)
        .into_iter()
        .map(continuous)
        .collect();

    if let Some(outputs) = outputs {
        messages.extend(
            chunk_arrays(outputs, VariableType::KOutput, chunk_size)
                .into_iter()
                .map(continuous),
        );
    }

    for (name, value) in discrete_inputs {
        messages.push(discrete(name, value.clone(), VariableType::KDiscreteInput));
    }

    messages
}

/// Build the response messages a server sends for a compute call.
pub fn assemble_output_messages(
    arrays: &ArrayMap,
    var_type: VariableType,
    discrete_outputs: &DiscreteMap,
    chunk_size: usize,
) -> Vec<VariableMessage> {
    let mut messages: Vec<VariableMessage> = chunk_arrays(arrays, var_type, chunk_size)
        .into_iter()
        .map(continuous)
        .collect();

    for (name, value) in discrete_outputs {
        messages.push(discrete(name, value.clone(), VariableType::KDiscreteOutput));
    }

    messages
}

/// Build the response messages a server sends for a gradient call.
pub fn assemble_partial_messages(partials: &PartialMap, chunk_size: usize) -> Vec<VariableMessage> {
    chunk_partials(partials, chunk_size)
        .into_iter()
        .map(continuous)
        .collect()
}

/// Error if any declared variable was left partially or entirely unwritten.
///
/// A zeroed array is indistinguishable from a correctly computed one, so a short
/// or missing response would otherwise be reported as success. Over-long chunks
/// already fail loudly; this closes the other side of that asymmetry.
fn verify_fully_populated(arrays: &ArrayMap, written: &HashMap<String, usize>) -> Result<()> {
    for (name, array) in arrays {
        let received = written.get(name).copied().unwrap_or(0);
        if received != array.len() {
            return Err(PhiloteError::validation(
                "decode",
                format!(
                    "'{name}' was declared with {} values but the peer sent {received}",
                    array.len()
                ),
            ));
        }
    }
    Ok(())
}

/// Error if any declared partial was left partially or entirely unwritten.
///
/// The partials counterpart of [`verify_fully_populated`]; a silently zero-filled
/// Jacobian reads as a legitimately zero derivative, which is worse than an error.
fn verify_partials_fully_populated(
    partials: &PartialMap,
    written: &HashMap<(String, String), usize>,
) -> Result<()> {
    for (key, array) in partials {
        let received = written.get(key).copied().unwrap_or(0);
        if received != array.len() {
            return Err(PhiloteError::validation(
                "compute_gradient",
                format!(
                    "d({})/d({}) was declared with {} values but the server sent {received}",
                    key.0,
                    key.1,
                    array.len()
                ),
            ));
        }
    }
    Ok(())
}

/// Read a response stream into arrays of the declared shape, keeping only
/// `var_type`, plus any discrete values.
///
/// Filtering on `var_type` is what keeps an implicit discipline's residuals and
/// outputs apart: the standard gives a residual the same name as its output, so
/// grouping by name alone would merge them.
pub async fn recover_arrays(
    stream: &mut Streaming<VariableMessage>,
    var_meta: &[VariableMetaData],
    var_type: VariableType,
) -> Result<(ArrayMap, DiscreteMap)> {
    let mut arrays = preallocate_arrays(var_meta, Some(var_type))?;
    let mut discrete_outputs: DiscreteMap = HashMap::new();
    let mut written: HashMap<String, usize> = HashMap::new();

    while let Some(message) = stream.next().await {
        match message?.payload {
            Some(Payload::Continuous(array)) => {
                let chunk = ArrayData::try_from(array)?;
                // Reject rather than skip. `kInput` is the proto3 default, so a peer
                // that forgets to tag its response would otherwise leave every
                // output at zero and report success.
                if chunk.var_type != var_type {
                    return Err(PhiloteError::InvalidVariableType(format!(
                        "expected {:?} data for '{}', but the server sent {:?}",
                        var_type, chunk.name, chunk.var_type
                    )));
                }
                let target = arrays
                    .get_mut(&chunk.name)
                    .ok_or_else(|| PhiloteError::VariableNotFound(chunk.name.clone()))?;
                scatter_chunk(target, &chunk)?;
                *written.entry(chunk.name).or_default() += chunk.data.len();
            }
            Some(Payload::Discrete(var)) => {
                // Mirror the continuous branch: only genuine outputs count, so an
                // echoed input cannot masquerade as a computed result.
                if var.r#type != i32::from(VariableType::KDiscreteOutput) {
                    return Err(PhiloteError::InvalidVariableType(format!(
                        "expected a discrete output for '{}', but the server sent type {}",
                        var.name, var.r#type
                    )));
                }
                if let Some(value) = var.value {
                    discrete_outputs.insert(var.name, value);
                }
            }
            None => {}
        }
    }

    verify_fully_populated(&arrays, &written)?;
    Ok((arrays, discrete_outputs))
}

/// Read a gradient response stream into a partials map of the declared shapes.
pub async fn recover_partials(
    stream: &mut Streaming<VariableMessage>,
    var_meta: &[VariableMetaData],
    partials_meta: &[PartialsMetaData],
) -> Result<PartialMap> {
    let shapes: HashMap<&str, Vec<usize>> = var_meta
        .iter()
        .map(|v| {
            (
                v.name.as_str(),
                v.shape.iter().map(|&d| d as usize).collect(),
            )
        })
        .collect();

    let mut partials: PartialMap = HashMap::new();
    for meta in partials_meta {
        // Prefer the shape the server declared; fall back to deriving it from the
        // variable shapes, which is what Philote-Python always does.
        let shape: Vec<usize> = if meta.shape.is_empty() {
            let func = shapes
                .get(meta.name.as_str())
                .ok_or_else(|| PhiloteError::VariableNotFound(meta.name.clone()))?;
            let var = shapes
                .get(meta.subname.as_str())
                .ok_or_else(|| PhiloteError::VariableNotFound(meta.subname.clone()))?;
            calculate_partial_shape(func, var)
        } else {
            meta.shape.iter().map(|&d| d as usize).collect()
        };

        partials.insert(
            (meta.name.clone(), meta.subname.clone()),
            ArrayD::zeros(shape),
        );
    }

    let mut written: HashMap<(String, String), usize> = HashMap::new();

    while let Some(message) = stream.next().await {
        if let Some(Payload::Continuous(array)) = message?.payload {
            let chunk = ArrayData::try_from(array)?;
            // As in `recover_arrays`: reject rather than skip, so an untagged
            // response cannot masquerade as an all-zero Jacobian.
            if chunk.var_type != VariableType::KPartial {
                return Err(PhiloteError::InvalidVariableType(format!(
                    "expected partial derivative data for '{}', but the server sent {:?}",
                    chunk.name, chunk.var_type
                )));
            }
            let subname = chunk.subname.clone().ok_or_else(|| {
                PhiloteError::validation(
                    "compute_gradient",
                    format!("partial chunk for '{}' is missing its subname", chunk.name),
                )
            })?;
            let key = (chunk.name.clone(), subname);
            let target = partials.get_mut(&key).ok_or_else(|| {
                // Most often this means the caller skipped `get_partial_definitions`,
                // leaving `partials_meta` empty while the server streams partials.
                PhiloteError::validation(
                    "compute_gradient",
                    format!(
                        "server sent the partial d({})/d({}), which was not declared; \
                         call get_partial_definitions() before computing gradients",
                        key.0, key.1
                    ),
                )
            })?;
            scatter_chunk(target, &chunk)?;
            *written.entry(key).or_default() += chunk.data.len();
        }
    }

    verify_partials_fully_populated(&partials, &written)?;
    Ok(partials)
}

/// Read a request stream on the server, scattering into preallocated maps.
///
/// `outputs` is `Some` only for implicit RPCs. `declared_discrete` is the set of
/// discrete input names the discipline declared; a discrete value for any other
/// name is rejected, mirroring how an undeclared continuous variable is handled.
pub async fn receive_request_stream(
    stream: &mut Streaming<VariableMessage>,
    inputs: &mut ArrayMap,
    mut outputs: Option<&mut ArrayMap>,
    discrete_inputs: &mut DiscreteMap,
    declared_discrete: &HashSet<String>,
) -> Result<()> {
    while let Some(message) = stream.next().await {
        match message?.payload {
            Some(Payload::Continuous(array)) => {
                let chunk = ArrayData::try_from(array)?;
                let target = match chunk.var_type {
                    VariableType::KInput => inputs.get_mut(&chunk.name),
                    VariableType::KOutput => {
                        let Some(outputs) = outputs.as_mut() else {
                            // Only the implicit residual RPCs take output values.
                            return Err(PhiloteError::validation(
                                "compute",
                                format!(
                                    "received a value for output '{}', but this RPC \
                                     accepts inputs only",
                                    chunk.name
                                ),
                            ));
                        };
                        outputs.get_mut(&chunk.name)
                    }
                    other => {
                        return Err(PhiloteError::InvalidVariableType(format!(
                            "unexpected variable type in request stream: {other:?}"
                        )))
                    }
                }
                .ok_or_else(|| PhiloteError::VariableNotFound(chunk.name.clone()))?;
                scatter_chunk(target, &chunk)?;
            }
            Some(Payload::Discrete(var)) => {
                if var.r#type != i32::from(VariableType::KDiscreteInput) {
                    return Err(PhiloteError::InvalidVariableType(format!(
                        "expected a discrete input for '{}', but the client sent type {}",
                        var.name, var.r#type
                    )));
                }
                if !declared_discrete.contains(&var.name) {
                    // Silently dropping this would hide a client bug, and would also
                    // mean a discipline that declares no discrete variables accepts
                    // discrete values and ignores them.
                    return Err(PhiloteError::VariableNotFound(var.name));
                }
                if let Some(value) = var.value {
                    discrete_inputs.insert(var.name, value);
                }
            }
            None => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(name: &str, start: usize, end: usize, data: Vec<f64>) -> ArrayData {
        ArrayData::new(
            name.to_string(),
            None,
            start,
            end,
            VariableType::KInput,
            data,
        )
    }

    #[test]
    fn scatters_into_two_dimensional_array() {
        // Regression test: indexing an ArrayD with a flat usize silently fails for
        // rank >= 2, which used to zero out every non-1-D input.
        let mut target: ArrayD<f64> = ArrayD::zeros(vec![2, 3]);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        scatter_chunk(&mut target, &chunk("x", 0, 5, values.clone())).unwrap();

        assert_eq!(target.shape(), &[2, 3]);
        assert_eq!(target.iter().copied().collect::<Vec<_>>(), values);
        assert_eq!(target[[0, 0]], 1.0);
        assert_eq!(target[[1, 2]], 6.0);
    }

    #[test]
    fn scatters_into_three_dimensional_array() {
        let mut target: ArrayD<f64> = ArrayD::zeros(vec![2, 2, 2]);
        let values: Vec<f64> = (1..=8).map(f64::from).collect();
        scatter_chunk(&mut target, &chunk("x", 0, 7, values.clone())).unwrap();
        assert_eq!(target.iter().copied().collect::<Vec<_>>(), values);
    }

    #[test]
    fn multi_chunk_round_trip_preserves_shape_and_order() {
        let source =
            ArrayD::from_shape_vec(vec![3, 4], (1..=12).map(f64::from).collect::<Vec<_>>())
                .unwrap();
        let mut arrays = HashMap::new();
        arrays.insert("x".to_string(), source.clone());

        let chunks = chunk_arrays(&arrays, VariableType::KInput, 2);
        assert_eq!(chunks.len(), 6);

        let mut target: ArrayD<f64> = ArrayD::zeros(vec![3, 4]);
        for chunk in &chunks {
            scatter_chunk(&mut target, chunk).unwrap();
        }
        assert_eq!(target, source);
    }

    #[test]
    fn rejects_chunk_length_mismatch() {
        let mut target: ArrayD<f64> = ArrayD::zeros(vec![4]);
        let err = scatter_chunk(&mut target, &chunk("x", 0, 2, vec![1.0, 2.0])).unwrap_err();
        // A malformed chunk is the peer's fault, so it must map to INVALID_ARGUMENT.
        assert!(matches!(err, PhiloteError::Validation { .. }));
        assert!(err.to_string().contains("size mismatch"));
        assert_eq!(err.to_status().code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn rejects_out_of_bounds_chunk() {
        let mut target: ArrayD<f64> = ArrayD::zeros(vec![3]);
        let err = scatter_chunk(&mut target, &chunk("x", 2, 4, vec![1.0, 2.0, 3.0])).unwrap_err();
        assert!(matches!(
            err,
            PhiloteError::IndexOutOfBounds { index: 4, size: 3 }
        ));
    }

    #[test]
    fn chunker_uses_inclusive_end_indices() {
        let mut arrays = HashMap::new();
        arrays.insert(
            "x".to_string(),
            ArrayD::from_shape_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
        );
        let chunks = chunk_arrays(&arrays, VariableType::KInput, 2);
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[0].end, 1);
        assert_eq!(chunks[0].data.len(), 2);
        assert_eq!(chunks[2].start, 4);
        assert_eq!(chunks[2].end, 4);
    }

    #[test]
    fn scatters_into_a_non_contiguous_target_in_logical_order() {
        // A transposed owned array is non-contiguous, so `as_slice_mut` returns None
        // and the fallback iteration runs. That path must place values in row-major
        // *logical* order, not in memory order — writing memory order would silently
        // transpose the variable.
        let mut target: ArrayD<f64> = ArrayD::zeros(vec![2, 3]).reversed_axes();
        assert_eq!(target.shape(), &[3, 2]);
        assert!(
            target.as_slice_mut().is_none(),
            "this test is only meaningful for a non-contiguous target"
        );

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        scatter_chunk(&mut target, &chunk("x", 0, 5, values.clone())).unwrap();

        assert_eq!(target.iter().copied().collect::<Vec<_>>(), values);
        assert_eq!(target[[0, 0]], 1.0);
        assert_eq!(target[[0, 1]], 2.0);
        assert_eq!(target[[1, 0]], 3.0);
        assert_eq!(target[[2, 1]], 6.0);
    }

    #[test]
    fn scatters_a_partial_range_into_a_non_contiguous_target() {
        let mut target: ArrayD<f64> = ArrayD::zeros(vec![2, 3]).reversed_axes();
        // Flat indices 2..=3 are the logical elements [1, 0] and [1, 1].
        scatter_chunk(&mut target, &chunk("x", 2, 3, vec![7.0, 8.0])).unwrap();

        assert_eq!(
            target.iter().copied().collect::<Vec<_>>(),
            vec![0.0, 0.0, 7.0, 8.0, 0.0, 0.0]
        );
        assert_eq!(target[[1, 0]], 7.0);
        assert_eq!(target[[1, 1]], 8.0);
    }

    #[test]
    fn rejects_a_chunk_whose_end_precedes_its_start() {
        let mut target: ArrayD<f64> = ArrayD::zeros(vec![4]);
        let err = scatter_chunk(&mut target, &chunk("x", 3, 1, vec![1.0])).unwrap_err();
        assert!(matches!(err, PhiloteError::Validation { .. }));
        assert!(err.to_string().contains("end 1 before start 3"));
        assert_eq!(err.to_status().code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn partial_chunks_carry_subname() {
        let mut partials: PartialMap = HashMap::new();
        partials.insert(
            ("f".to_string(), "x".to_string()),
            ArrayD::from_shape_vec(vec![2], vec![1.0, 2.0]).unwrap(),
        );
        let chunks = chunk_partials(&partials, 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].name, "f");
        assert_eq!(chunks[0].subname, Some("x".to_string()));
        assert_eq!(chunks[0].var_type, VariableType::KPartial);
    }

    // --- Stream decoding ---
    //
    // `recover_arrays`, `recover_partials` and `receive_request_stream` all take a
    // concrete `tonic::Streaming`, which can only come off a real connection. The
    // harness below serves a scripted endpoint on an ephemeral loopback port so a
    // test can hand these functions any message sequence, including malformed ones
    // that the crate's own client and server would never produce.
    mod harness {
        use super::*;
        use std::net::SocketAddr;
        use std::pin::Pin;
        use tokio::sync::oneshot;
        use tokio_stream::Stream;
        use tonic::transport::Server;
        use tonic::{Request, Response, Status};

        use crate::philote_info::explicit_service_client::ExplicitServiceClient;
        use crate::philote_info::explicit_service_server::{
            ExplicitService, ExplicitServiceServer,
        };

        type MessageStream =
            Pin<Box<dyn Stream<Item = std::result::Result<VariableMessage, Status>> + Send>>;

        /// What the scripted endpoint does with a call.
        pub enum Behavior {
            /// Drain the request, then stream back exactly these messages.
            Reply(Vec<VariableMessage>),
            /// Decode the request with [`receive_request_stream`], then echo whatever
            /// landed so the caller can assert on it. A decode error becomes the
            /// RPC's status.
            Receive {
                inputs: Vec<(&'static str, Vec<usize>)>,
                outputs: Option<Vec<(&'static str, Vec<usize>)>>,
                declared_discrete: Vec<&'static str>,
            },
        }

        fn allocate(vars: &[(&'static str, Vec<usize>)]) -> ArrayMap {
            vars.iter()
                .map(|(name, shape)| (name.to_string(), ArrayD::zeros(shape.clone())))
                .collect()
        }

        struct Endpoint {
            behavior: Behavior,
        }

        impl Endpoint {
            async fn run(
                &self,
                request: Request<Streaming<VariableMessage>>,
            ) -> std::result::Result<Response<MessageStream>, Status> {
                let mut stream = request.into_inner();
                let messages = match &self.behavior {
                    Behavior::Reply(script) => {
                        while stream.next().await.transpose()?.is_some() {}
                        script.clone()
                    }
                    Behavior::Receive {
                        inputs,
                        outputs,
                        declared_discrete,
                    } => {
                        let mut inputs = allocate(inputs);
                        let mut outputs = outputs.as_ref().map(|vars| allocate(vars));
                        let mut discrete_inputs: DiscreteMap = HashMap::new();
                        let declared: HashSet<String> =
                            declared_discrete.iter().map(|n| n.to_string()).collect();

                        receive_request_stream(
                            &mut stream,
                            &mut inputs,
                            outputs.as_mut(),
                            &mut discrete_inputs,
                            &declared,
                        )
                        .await
                        .map_err(|err| err.to_status())?;

                        let mut received = inputs;
                        if let Some(outputs) = outputs {
                            received.extend(outputs);
                        }
                        assemble_output_messages(
                            &received,
                            VariableType::KOutput,
                            &discrete_inputs,
                            8,
                        )
                    }
                };
                Ok(Response::new(Box::pin(tokio_stream::iter(
                    messages.into_iter().map(Ok).collect::<Vec<_>>(),
                ))))
            }
        }

        #[tonic::async_trait]
        impl ExplicitService for Endpoint {
            type ComputeFunctionStream = MessageStream;
            type ComputeGradientStream = MessageStream;

            async fn compute_function(
                &self,
                request: Request<Streaming<VariableMessage>>,
            ) -> std::result::Result<Response<Self::ComputeFunctionStream>, Status> {
                self.run(request).await
            }

            async fn compute_gradient(
                &self,
                request: Request<Streaming<VariableMessage>>,
            ) -> std::result::Result<Response<Self::ComputeGradientStream>, Status> {
                self.run(request).await
            }
        }

        /// Keeps the server and the channel alive for as long as the response stream
        /// is being read.
        pub struct Connection {
            shutdown: Option<oneshot::Sender<()>>,
            _client: ExplicitServiceClient<tonic::transport::Channel>,
        }

        impl Drop for Connection {
            fn drop(&mut self) {
                if let Some(tx) = self.shutdown.take() {
                    let _ = tx.send(());
                }
            }
        }

        /// Serve `behavior`, send `request`, and return the live response stream.
        pub async fn exchange(
            behavior: Behavior,
            request: Vec<VariableMessage>,
        ) -> (
            Connection,
            std::result::Result<Streaming<VariableMessage>, Status>,
        ) {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind an ephemeral loopback port");
            let addr: SocketAddr = listener.local_addr().expect("resolve the bound address");
            let (tx, rx) = oneshot::channel();

            tokio::spawn(async move {
                let _ = Server::builder()
                    .add_service(ExplicitServiceServer::new(Endpoint { behavior }))
                    .serve_with_incoming_shutdown(
                        tokio_stream::wrappers::TcpListenerStream::new(listener),
                        async {
                            let _ = rx.await;
                        },
                    )
                    .await;
            });

            let mut client = ExplicitServiceClient::connect(format!("http://{addr}"))
                .await
                .expect("connect to the scripted endpoint");
            let response = client.compute_function(tokio_stream::iter(request)).await;

            (
                Connection {
                    shutdown: Some(tx),
                    _client: client,
                },
                response.map(|r| r.into_inner()),
            )
        }
    }

    use harness::{exchange, Behavior};
    use tonic::Status;

    fn meta(name: &str, shape: &[i64], var_type: VariableType) -> VariableMetaData {
        VariableMetaData {
            r#type: var_type.into(),
            name: name.to_string(),
            shape: shape.to_vec(),
            units: String::new(),
            dynamic_shape: false,
        }
    }

    fn typed_chunk(name: &str, var_type: VariableType, data: Vec<f64>) -> VariableMessage {
        continuous(ArrayData::new(
            name.to_string(),
            None,
            0,
            data.len() - 1,
            var_type,
            data,
        ))
    }

    fn flag(value: bool) -> prost_types::Value {
        crate::discrete::json_to_value(&serde_json::json!(value))
    }

    async fn replying(
        script: Vec<VariableMessage>,
    ) -> (harness::Connection, Streaming<VariableMessage>) {
        let (conn, stream) = exchange(Behavior::Reply(script), Vec::new()).await;
        (
            conn,
            stream.expect("the scripted endpoint accepted the call"),
        )
    }

    // --- recover_arrays ---

    #[tokio::test]
    async fn recovering_arrays_rejects_a_chunk_tagged_with_the_wrong_type() {
        // kInput is the proto3 default, so an untagged response would otherwise be
        // accepted and leave every output at zero.
        let (_conn, mut stream) =
            replying(vec![typed_chunk("y", VariableType::KInput, vec![1.0, 2.0])]).await;

        let err = recover_arrays(
            &mut stream,
            &[meta("y", &[2], VariableType::KOutput)],
            VariableType::KOutput,
        )
        .await
        .unwrap_err();

        assert!(matches!(err, PhiloteError::InvalidVariableType(_)));
        assert!(err.to_string().contains("KInput"));
        assert_eq!(err.to_status().code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn recovering_arrays_rejects_a_discrete_value_that_is_not_an_output() {
        let (_conn, mut stream) = replying(vec![discrete(
            "mode",
            flag(true),
            VariableType::KDiscreteInput,
        )])
        .await;

        let err = recover_arrays(&mut stream, &[], VariableType::KOutput)
            .await
            .unwrap_err();

        assert!(matches!(err, PhiloteError::InvalidVariableType(_)));
        assert!(err
            .to_string()
            .contains("expected a discrete output for 'mode'"));
    }

    #[tokio::test]
    async fn recovering_arrays_skips_a_message_with_no_payload() {
        // A peer may emit an empty message; it carries no data, so it must neither
        // fail the call nor count towards the values a variable expects.
        let (_conn, mut stream) = replying(vec![
            VariableMessage { payload: None },
            typed_chunk("y", VariableType::KOutput, vec![1.0, 2.0]),
            VariableMessage { payload: None },
        ])
        .await;

        let (arrays, discrete_outputs) = recover_arrays(
            &mut stream,
            &[meta("y", &[2], VariableType::KOutput)],
            VariableType::KOutput,
        )
        .await
        .unwrap();

        assert_eq!(
            arrays["y"],
            ArrayD::from_shape_vec(vec![2], vec![1.0, 2.0]).unwrap()
        );
        assert!(discrete_outputs.is_empty());
    }

    #[tokio::test]
    async fn recovering_arrays_collects_discrete_outputs_alongside_the_arrays() {
        let (_conn, mut stream) = replying(vec![
            typed_chunk("y", VariableType::KOutput, vec![4.0]),
            discrete("converged", flag(true), VariableType::KDiscreteOutput),
        ])
        .await;

        let (arrays, discrete_outputs) = recover_arrays(
            &mut stream,
            &[meta("y", &[1], VariableType::KOutput)],
            VariableType::KOutput,
        )
        .await
        .unwrap();

        assert_eq!(
            arrays["y"],
            ArrayD::from_shape_vec(vec![1], vec![4.0]).unwrap()
        );
        assert_eq!(
            crate::discrete::value_to_json(&discrete_outputs["converged"]),
            serde_json::json!(true)
        );
    }

    // --- recover_partials ---

    fn partials_meta(name: &str, subname: &str) -> PartialsMetaData {
        PartialsMetaData {
            name: name.to_string(),
            subname: subname.to_string(),
            shape: vec![1],
        }
    }

    #[tokio::test]
    async fn recovering_partials_rejects_a_chunk_that_is_not_a_partial() {
        let (_conn, mut stream) =
            replying(vec![typed_chunk("f", VariableType::KOutput, vec![1.0])]).await;

        let err = recover_partials(&mut stream, &[], &[partials_meta("f", "x")])
            .await
            .unwrap_err();

        assert!(matches!(err, PhiloteError::InvalidVariableType(_)));
        assert!(err
            .to_string()
            .contains("expected partial derivative data for 'f'"));
    }

    #[tokio::test]
    async fn recovering_partials_rejects_a_partial_chunk_without_a_subname() {
        // Without a subname there is no way to tell which derivative this is.
        let (_conn, mut stream) =
            replying(vec![typed_chunk("f", VariableType::KPartial, vec![1.0])]).await;

        let err = recover_partials(&mut stream, &[], &[partials_meta("f", "x")])
            .await
            .unwrap_err();

        assert!(matches!(err, PhiloteError::Validation { .. }));
        assert!(err.to_string().contains("missing its subname"));
    }

    #[tokio::test]
    async fn recovering_partials_rejects_a_partial_that_was_never_declared() {
        let mut chunk = ArrayData::new(
            "f".to_string(),
            Some("x".to_string()),
            0,
            0,
            VariableType::KPartial,
            vec![1.0],
        );
        chunk.subname = Some("x".to_string());
        let (_conn, mut stream) = replying(vec![continuous(chunk)]).await;

        // Empty `partials_meta` is what a caller that skipped
        // `get_partial_definitions` ends up with.
        let err = recover_partials(&mut stream, &[], &[]).await.unwrap_err();

        assert!(matches!(err, PhiloteError::Validation { .. }));
        assert!(err.to_string().contains("d(f)/d(x)"));
        assert!(err.to_string().contains("get_partial_definitions"));
    }

    #[tokio::test]
    async fn recovering_partials_skips_messages_that_carry_no_array() {
        // A gradient stream carries only partials; anything else (an empty message,
        // or a discrete value a peer tacked on) has no place to go and is ignored
        // rather than failing the call.
        let mut chunk = ArrayData::new(
            "f".to_string(),
            Some("x".to_string()),
            0,
            0,
            VariableType::KPartial,
            vec![3.0],
        );
        chunk.subname = Some("x".to_string());
        let (_conn, mut stream) = replying(vec![
            VariableMessage { payload: None },
            continuous(chunk),
            discrete("note", flag(true), VariableType::KDiscreteOutput),
        ])
        .await;

        let partials = recover_partials(&mut stream, &[], &[partials_meta("f", "x")])
            .await
            .unwrap();

        assert_eq!(
            partials[&("f".to_string(), "x".to_string())],
            ArrayD::from_shape_vec(vec![1], vec![3.0]).unwrap()
        );
    }

    #[tokio::test]
    async fn recovering_partials_rejects_a_jacobian_the_server_left_short() {
        // Regression: `recover_arrays` verified full population but
        // `recover_partials` did not, so a server that streamed only part of a
        // Jacobian yielded a silently zero-filled derivative reported as success.
        // A zero derivative is a legitimate value, which is what made this quiet.
        let mut chunk = ArrayData::new(
            "f".to_string(),
            Some("x".to_string()),
            0,
            1,
            VariableType::KPartial,
            vec![1.0, 2.0],
        );
        chunk.subname = Some("x".to_string());
        let (_conn, mut stream) = replying(vec![continuous(chunk)]).await;

        // Declared as 4 values; the server sent 2.
        let meta = PartialsMetaData {
            name: "f".to_string(),
            subname: "x".to_string(),
            shape: vec![4],
        };
        let err = recover_partials(&mut stream, &[], &[meta])
            .await
            .unwrap_err();

        assert!(matches!(err, PhiloteError::Validation { .. }));
        assert!(err.to_string().contains("d(f)/d(x)"));
        assert!(err.to_string().contains("4 values"));
    }

    #[tokio::test]
    async fn recovering_partials_rejects_a_jacobian_the_server_never_sent() {
        // The all-zero case: nothing arrives at all, which previously decoded as a
        // zero Jacobian and reported success.
        let (_conn, mut stream) = replying(vec![]).await;

        let err = recover_partials(&mut stream, &[], &[partials_meta("f", "x")])
            .await
            .unwrap_err();

        assert!(matches!(err, PhiloteError::Validation { .. }));
        assert!(err.to_string().contains("the server sent 0"));
    }

    // --- receive_request_stream ---

    /// Decode `request` against a discipline declaring input `x` (and optionally
    /// output `y` plus discrete inputs), returning the echoed result.
    async fn receive(
        request: Vec<VariableMessage>,
        outputs: Option<Vec<(&'static str, Vec<usize>)>>,
        declared_discrete: Vec<&'static str>,
    ) -> std::result::Result<(ArrayMap, DiscreteMap), Status> {
        let (_conn, stream) = exchange(
            Behavior::Receive {
                inputs: vec![("x", vec![2])],
                outputs,
                declared_discrete,
            },
            request,
        )
        .await;

        let mut stream = stream?;
        let mut arrays: ArrayMap = HashMap::new();
        let mut discrete_values: DiscreteMap = HashMap::new();
        while let Some(message) = stream.next().await {
            match message?.payload {
                Some(Payload::Continuous(array)) => {
                    let chunk = ArrayData::try_from(array).expect("decode the echoed chunk");
                    let target = arrays
                        .entry(chunk.name.clone())
                        .or_insert_with(|| ArrayD::zeros(vec![chunk.data.len()]));
                    scatter_chunk(target, &chunk).expect("scatter the echoed chunk");
                }
                Some(Payload::Discrete(var)) => {
                    discrete_values.insert(var.name, var.value.expect("a discrete value"));
                }
                None => {}
            }
        }
        Ok((arrays, discrete_values))
    }

    #[tokio::test]
    async fn a_request_stream_scatters_inputs_and_skips_empty_messages() {
        let (arrays, discrete_values) = receive(
            vec![
                VariableMessage { payload: None },
                typed_chunk("x", VariableType::KInput, vec![1.5, 2.5]),
            ],
            None,
            Vec::new(),
        )
        .await
        .expect("a well-formed request stream");

        assert_eq!(
            arrays["x"],
            ArrayD::from_shape_vec(vec![2], vec![1.5, 2.5]).unwrap()
        );
        assert!(discrete_values.is_empty());
    }

    #[tokio::test]
    async fn a_request_stream_accepts_output_values_only_for_implicit_rpcs() {
        let request = || {
            vec![
                typed_chunk("x", VariableType::KInput, vec![1.0, 2.0]),
                typed_chunk("y", VariableType::KOutput, vec![9.0]),
            ]
        };

        let (arrays, _) = receive(request(), Some(vec![("y", vec![1])]), Vec::new())
            .await
            .expect("an implicit RPC takes output values");
        assert_eq!(
            arrays["y"],
            ArrayD::from_shape_vec(vec![1], vec![9.0]).unwrap()
        );

        // The same stream against an RPC that takes inputs only must be rejected
        // rather than silently dropped.
        let status = receive(request(), None, Vec::new())
            .await
            .expect_err("an explicit RPC has nowhere to put output values");
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("accepts inputs only"));
    }

    #[tokio::test]
    async fn a_request_stream_rejects_a_chunk_that_is_neither_an_input_nor_an_output() {
        let status = receive(
            vec![typed_chunk("x", VariableType::KResidual, vec![1.0, 2.0])],
            None,
            Vec::new(),
        )
        .await
        .expect_err("residuals never travel upstream");

        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("unexpected variable type"));
        assert!(status.message().contains("KResidual"));
    }

    #[tokio::test]
    async fn a_request_stream_rejects_a_discrete_value_that_is_not_an_input() {
        let status = receive(
            vec![discrete("mode", flag(true), VariableType::KDiscreteOutput)],
            None,
            vec!["mode"],
        )
        .await
        .expect_err("a client may not send discrete outputs");

        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status
            .message()
            .contains("expected a discrete input for 'mode'"));
    }

    #[tokio::test]
    async fn a_request_stream_rejects_an_undeclared_discrete_input() {
        // Declared discrete inputs are accepted; anything else is a client bug and
        // must not be silently dropped.
        let (_, discrete_values) = receive(
            vec![discrete("mode", flag(true), VariableType::KDiscreteInput)],
            None,
            vec!["mode"],
        )
        .await
        .expect("a declared discrete input");
        assert_eq!(
            crate::discrete::value_to_json(&discrete_values["mode"]),
            serde_json::json!(true)
        );

        let status = receive(
            vec![discrete("speed", flag(true), VariableType::KDiscreteInput)],
            None,
            vec!["mode"],
        )
        .await
        .expect_err("an undeclared discrete input");
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("speed"));
    }
}
