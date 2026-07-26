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

use std::collections::HashMap;

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
            PhiloteError::array_error(format!(
                "chunk for '{}' has end {} before start {}",
                chunk.name, chunk.end, chunk.start
            ))
        })?;

    if chunk.data.len() != expected {
        return Err(PhiloteError::array_error(format!(
            "chunk size mismatch for '{}': indices {}..={} span {} values, got {}",
            chunk.name,
            chunk.start,
            chunk.end,
            expected,
            chunk.data.len()
        )));
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
                PhiloteError::array_error(format!(
                    "partial chunk for '{}' is missing its subname",
                    chunk.name
                ))
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
        }
    }

    Ok(partials)
}

/// Read a request stream on the server, scattering into preallocated maps.
///
/// `outputs` is `Some` only for implicit RPCs.
pub async fn receive_request_stream(
    stream: &mut Streaming<VariableMessage>,
    inputs: &mut ArrayMap,
    mut outputs: Option<&mut ArrayMap>,
    discrete_inputs: &mut DiscreteMap,
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
        assert!(matches!(err, PhiloteError::ArrayError(_)));
        assert!(err.to_string().contains("size mismatch"));
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
}
