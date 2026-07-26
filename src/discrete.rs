//! Conversions between `serde_json::Value` and `google.protobuf.Value`
//!
//! Discrete variables and discipline options both travel as protobuf `Value` /
//! `Struct` payloads. These helpers mirror `_python_to_value` / `_value_to_python`
//! in `philote_mdo/general/discipline_server.py`.

use std::collections::{BTreeMap, HashMap};

use prost_types::value::Kind;

/// Convert a JSON value into a protobuf `Value`.
pub fn json_to_value(v: &serde_json::Value) -> prost_types::Value {
    let kind = match v {
        serde_json::Value::Null => Kind::NullValue(0),
        serde_json::Value::Bool(b) => Kind::BoolValue(*b),
        serde_json::Value::Number(n) => Kind::NumberValue(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(s) => Kind::StringValue(s.clone()),
        serde_json::Value::Array(arr) => Kind::ListValue(prost_types::ListValue {
            values: arr.iter().map(json_to_value).collect(),
        }),
        serde_json::Value::Object(map) => Kind::StructValue(prost_types::Struct {
            fields: map
                .iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect(),
        }),
    };
    prost_types::Value { kind: Some(kind) }
}

/// Convert a protobuf `Value` into a JSON value.
///
/// Whole numbers are returned as JSON integers when the conversion is lossless,
/// matching Python's `_value_to_python`.
pub fn value_to_json(v: &prost_types::Value) -> serde_json::Value {
    match &v.kind {
        Some(Kind::NullValue(_)) | None => serde_json::Value::Null,
        Some(Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(Kind::NumberValue(n)) => {
            if n.fract() == 0.0 && n.is_finite() && *n >= i64::MIN as f64 && *n <= i64::MAX as f64 {
                serde_json::Value::from(*n as i64)
            } else {
                serde_json::json!(*n)
            }
        }
        Some(Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(Kind::ListValue(list)) => {
            serde_json::Value::Array(list.values.iter().map(value_to_json).collect())
        }
        Some(Kind::StructValue(s)) => serde_json::Value::Object(
            s.fields
                .iter()
                .map(|(k, v)| (k.clone(), value_to_json(v)))
                .collect(),
        ),
    }
}

/// Convert a protobuf `Struct` into a map of JSON values.
pub fn struct_to_json_map(s: &prost_types::Struct) -> HashMap<String, serde_json::Value> {
    s.fields
        .iter()
        .map(|(k, v)| (k.clone(), value_to_json(v)))
        .collect()
}

/// Convert a map of JSON values into a protobuf `Struct`.
pub fn json_map_to_struct(map: &HashMap<String, serde_json::Value>) -> prost_types::Struct {
    let fields: BTreeMap<String, prost_types::Value> = map
        .iter()
        .map(|(k, v)| (k.clone(), json_to_value(v)))
        .collect();
    prost_types::Struct { fields }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(v: serde_json::Value) -> serde_json::Value {
        value_to_json(&json_to_value(&v))
    }

    #[test]
    fn null_round_trips() {
        assert_eq!(round_trip(serde_json::Value::Null), serde_json::Value::Null);
    }

    #[test]
    fn bool_round_trips() {
        assert_eq!(round_trip(serde_json::json!(true)), serde_json::json!(true));
        assert_eq!(
            round_trip(serde_json::json!(false)),
            serde_json::json!(false)
        );
    }

    #[test]
    fn integer_round_trips_as_integer() {
        let out = round_trip(serde_json::json!(42));
        assert_eq!(out, serde_json::json!(42));
        assert!(out.is_i64(), "whole numbers should come back as integers");
    }

    #[test]
    fn negative_integer_round_trips() {
        assert_eq!(round_trip(serde_json::json!(-7)), serde_json::json!(-7));
    }

    #[test]
    fn float_round_trips() {
        let out = round_trip(serde_json::json!(1.5));
        assert_eq!(out.as_f64().unwrap(), 1.5);
    }

    #[test]
    fn string_round_trips() {
        assert_eq!(round_trip(serde_json::json!("hi")), serde_json::json!("hi"));
    }

    #[test]
    fn list_round_trips() {
        let v = serde_json::json!([1, "two", false, null]);
        assert_eq!(round_trip(v.clone()), v);
    }

    #[test]
    fn nested_struct_round_trips() {
        let v = serde_json::json!({"a": {"b": [1, 2]}, "c": "d"});
        assert_eq!(round_trip(v.clone()), v);
    }

    #[test]
    fn struct_map_round_trips() {
        let mut map = HashMap::new();
        map.insert("n".to_string(), serde_json::json!(3));
        map.insert("s".to_string(), serde_json::json!("x"));
        let recovered = struct_to_json_map(&json_map_to_struct(&map));
        assert_eq!(recovered, map);
    }
}
