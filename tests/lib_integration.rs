// Integration tests for the philote library
// These tests verify that different modules work together correctly

use ndarray::ArrayD;
use philote::{ArrayMap, PhiloteError};
use std::collections::HashMap;

#[test]
fn test_end_to_end_array_serialization() {
    // Create an array map similar to what would be used in client-server communication
    let mut arrays: ArrayMap = HashMap::new();
    arrays.insert(
        "x".to_string(),
        ArrayD::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    arrays.insert(
        "y".to_string(),
        ArrayD::from_shape_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
    );

    // Verify we can flatten and work with the data
    let flat_x = philote::utils::create_flattened_view(&arrays["x"]);
    assert_eq!(flat_x, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let flat_y = philote::utils::create_flattened_view(&arrays["y"]);
    assert_eq!(flat_y, vec![10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_array_chunking_and_metadata() {
    use philote::philote_info::VariableType;
    use philote::types::ArrayChunker;

    let chunker = ArrayChunker::new(100);
    let data = vec![1.0; 250];
    let chunks = chunker.chunk_array("test", &data, VariableType::KInput);

    // Should create 3 chunks: 100, 100, 50
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].data.len(), 100);
    assert_eq!(chunks[1].data.len(), 100);
    assert_eq!(chunks[2].data.len(), 50);

    // Verify metadata is preserved
    for chunk in &chunks {
        assert_eq!(chunk.name, "test");
        assert_eq!(chunk.var_type, VariableType::KInput);
    }
}

#[test]
fn test_variable_metadata_workflow() {
    use philote::philote_info::{VariableMetaData, VariableType};

    // Create variable metadata as would be done in a discipline
    let meta = vec![
        VariableMetaData {
            name: "x".to_string(),
            r#type: VariableType::KInput as i32,
            shape: vec![3],
            units: "m".to_string(),
            dynamic_shape: false,
        },
        VariableMetaData {
            name: "y".to_string(),
            r#type: VariableType::KOutput as i32,
            shape: vec![2, 2],
            units: "kg".to_string(),
            dynamic_shape: false,
        },
    ];

    // Preallocate arrays from metadata
    let arrays = philote::utils::preallocate_arrays(&meta, None).unwrap();
    assert_eq!(arrays.len(), 2);
    assert_eq!(arrays["x"].shape(), &[3]);
    assert_eq!(arrays["y"].shape(), &[2, 2]);

    // Validate shapes match
    let validation = philote::utils::validate_array_shapes(&arrays, &meta);
    assert!(validation.is_ok());
}

#[test]
fn test_partial_derivatives_workflow() {
    use philote::philote_info::{VariableMetaData, VariableType};

    // Set up variables for a simple function f(x, y) = x^2 + y
    let var_meta = vec![
        VariableMetaData {
            name: "f".to_string(),
            r#type: VariableType::KOutput as i32,
            shape: vec![1],
            units: "".to_string(),
            dynamic_shape: false,
        },
        VariableMetaData {
            name: "x".to_string(),
            r#type: VariableType::KInput as i32,
            shape: vec![1],
            units: "".to_string(),
            dynamic_shape: false,
        },
        VariableMetaData {
            name: "y".to_string(),
            r#type: VariableType::KInput as i32,
            shape: vec![1],
            units: "".to_string(),
            dynamic_shape: false,
        },
    ];

    // Define partial derivatives: df/dx and df/dy
    let partials_meta = vec![
        ("f".to_string(), "x".to_string()),
        ("f".to_string(), "y".to_string()),
    ];

    // Preallocate partial derivative arrays
    let partials = philote::utils::preallocate_partials(&var_meta, &partials_meta).unwrap();
    assert_eq!(partials.len(), 2);
    assert!(partials.contains_key(&("f".to_string(), "x".to_string())));
    assert!(partials.contains_key(&("f".to_string(), "y".to_string())));
}

#[test]
fn test_error_propagation() {
    use philote::philote_info::Array;
    use philote::types::ArrayData;

    // Test that invalid data is properly rejected
    let invalid_proto = Array {
        name: "test".to_string(),
        subname: "".to_string(),
        start: 0,
        end: 5,
        r#type: 999, // Invalid type
        data: vec![1.0, 2.0],
    };

    let result = ArrayData::try_from(invalid_proto);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        PhiloteError::InvalidVariableType(_)
    ));
}

#[test]
fn test_stream_options_roundtrip() {
    use philote::types::StreamOptions;

    let opts = StreamOptions {
        max_double_per_slice: 500,
    };

    // Convert to proto and back
    let proto: philote::philote_info::StreamOptions = opts.into();
    let opts2: StreamOptions = proto.into();

    assert_eq!(opts2.max_double_per_slice, 500);
}
