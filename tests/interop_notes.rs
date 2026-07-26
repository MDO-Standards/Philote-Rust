//! Executable notes on cross-implementation compatibility.
//!
//! These lock in wire-level choices where Philote-Rust and Philote-Python differ,
//! so a future "fix" cannot quietly adopt the other side's behaviour.

use philote_mdo::philote_info::{Array, VariableType};
use philote_mdo::types::{ArrayChunker, ArrayData};

/// `Array.start` and `Array.end` are inclusive on both ends.
///
/// This is what the Philote standard specifies and what Philote-Python's
/// *explicit* server emits (`explicit_server.py` sends `end = e - 1`).
///
/// Philote-Python's *implicit* server instead sends `end = e` (exclusive) at
/// `implicit_server.py:192,259,333`, while both of its clients decode
/// `e = arr.end + 1`. That is a defect on the Python side, not a convention this
/// implementation should match: reproducing it here would corrupt every
/// multi-chunk implicit transfer between two Rust peers.
///
/// Consequence, until Philote-Python is patched: a Rust client talking to a
/// Python implicit server will reject multi-chunk arrays with a chunk-size
/// mismatch. That failure is expected and is the correct behaviour.
#[test]
fn chunk_end_indices_are_inclusive() {
    let chunker = ArrayChunker::new(2);
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let chunks = chunker.chunk_array("x", &data, VariableType::KInput);

    assert_eq!(chunks.len(), 3);

    // Each chunk spans end - start + 1 values.
    for chunk in &chunks {
        assert_eq!(
            chunk.data.len(),
            chunk.end - chunk.start + 1,
            "chunk [{}..={}] carries {} values; end must be inclusive",
            chunk.start,
            chunk.end,
            chunk.data.len()
        );
    }

    // Chunk boundaries abut without gaps or overlap.
    assert_eq!((chunks[0].start, chunks[0].end), (0, 1));
    assert_eq!((chunks[1].start, chunks[1].end), (2, 3));
    assert_eq!((chunks[2].start, chunks[2].end), (4, 4));
}

/// A whole-array chunk reports `end = len - 1`, never `len`.
#[test]
fn a_single_chunk_reports_the_last_index_not_the_length() {
    let chunker = ArrayChunker::new(100);
    let chunks = chunker.chunk_array("x", &[1.0, 2.0, 3.0], VariableType::KInput);

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].start, 0);
    assert_eq!(chunks[0].end, 2, "3 values span indices 0..=2");
}

/// The inclusive convention survives conversion to the protobuf message.
#[test]
fn the_wire_message_preserves_inclusive_indices() {
    let chunk = ArrayData::new(
        "x".to_string(),
        None,
        4,
        7,
        VariableType::KInput,
        vec![1.0, 2.0, 3.0, 4.0],
    );
    let proto = Array::from(chunk);

    assert_eq!(proto.start, 4);
    assert_eq!(proto.end, 7);
    assert_eq!(proto.data.len(), 4);
    assert_eq!(
        proto.end - proto.start + 1,
        proto.data.len() as i64,
        "the encoded span must match the payload length"
    );
}

/// `DisciplineProperties.name` and `.version` are populated.
///
/// Philote-Python never sets either field. Sending them is additive: a client that
/// ignores them is unaffected, and one that reads them gets real values.
#[test]
fn discipline_properties_carry_name_and_version() {
    use philote_mdo::examples::Paraboloid;
    use philote_mdo::traits::Discipline;

    let properties = Paraboloid::new().get_properties();
    assert_eq!(properties.name, "Paraboloid");
    assert!(!properties.version.is_empty());
}
