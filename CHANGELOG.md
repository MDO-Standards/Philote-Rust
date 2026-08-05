# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **N-dimensional variables no longer arrive as zeros.** The server wrote received
  data into preallocated arrays with a flat `usize` index, which only addresses
  rank-1 arrays; for any variable of rank >= 2 every write was silently dropped in
  release builds and panicked in debug builds. Decoding now goes through the flat
  slice, which is correct at any rank.
- Array shape is preserved end to end. Clients previously rebuilt every response as
  a flat 1-D array, discarding declared dimensions. Arrays are now preallocated
  from variable metadata and scattered into, so a `[3, 2]` output arrives as
  `[3, 2]`.
- Implicit residuals are no longer conflated with outputs. Responses were grouped
  by variable name alone, and the standard gives a residual the same name as its
  output, so the two could overwrite one another. Decoding now filters on variable
  type.
- `SetVariableShapes` is implemented. It previously drained the request stream and
  discarded it, leaving dynamic shapes non-functional server-side.
- Repeated `Setup` calls no longer accumulate duplicate variable definitions.
- Errors caused by client input (validation failures, unknown variables, invalid
  types) now map to `INVALID_ARGUMENT` rather than `INTERNAL`.
- A truncated or missing response is rejected instead of decoding as zeros. Arrays
  are preallocated, so a peer that sends fewer values than declared would otherwise
  be indistinguishable from one that computed them correctly.
- A response chunk whose variable type does not match what was requested is now an
  error rather than being skipped. `kInput` is the proto3 default, so an untagged
  response would otherwise leave every output at zero and report success.
- The server runs `initialize()` when it takes ownership of a discipline. Options
  declared there were previously never advertised for a discipline built with
  `#[derive(Default)]`.
- `ImplicitServer` marks its discipline's registry implicit, so outputs get their
  residual twins even when the registry came from `VariableRegistry::default()`.
- Discrete defaults are keyed by name *and* type. A discrete output's default no
  longer leaks into (or overwrites) the discrete inputs handed to a discipline.
- `setup()` and `send_variable_shapes()` invalidate the client's cached metadata,
  so a resized discipline is not decoded against stale shapes.
- Negative `Array.start`/`Array.end` from a peer are rejected rather than wrapping
  through `as usize` into an arithmetic overflow.
- A chunk size of zero no longer panics; it falls back to one.
- A discrete value sent for an undeclared name is rejected with `VariableNotFound`
  instead of being inserted into the map and ignored. Continuous variables were
  already rejected this way; a discipline declaring no discrete variables at all
  now also refuses discrete values rather than silently dropping them.
- Malformed messages from a peer (an empty `Array` payload, negative indices,
  `end` before `start`, a partial chunk missing its `subname`) map to
  `INVALID_ARGUMENT` rather than `INTERNAL`. `ArrayError` is now reserved for a
  discipline's own compute failing, which correctly stays `INTERNAL`.
- `PhiloteError::GrpcError` preserves the status code it wraps instead of
  flattening it to `INTERNAL`, so a server proxying a downstream Philote call no
  longer reports the peer's `INVALID_ARGUMENT` as its own internal failure.
- `build.rs` falls back to a vendored `protoc`, so building the crate no longer
  requires one on `PATH`. This is what lets docs.rs build the documentation. Set
  `PROTOC` to override.

### Added

- `VariableRegistry` and the `impl_registry!` macro. Metadata storage moved out of
  each discipline into a shared registry, which is what makes shape resolution,
  duplicate detection, residual twins, and re-`Setup` clearing possible.
- `wire` module: the single public implementation of the Philote chunk format
  (encode, decode, and stream assembly), replacing the two divergent encoders that
  previously lived in `utils`.
- Discrete variable support: declared defaults, per-type metadata, and full
  `google.protobuf.Value` round trips including nested structures.
- Client-resolved shapes: `add_dynamic_input` / `add_dynamic_output`,
  `variable_shape_meta`, and automatic residual-twin resolution.
- Input validation (`validation` module) mirroring Philote-Python, with a
  `PhiloteError::Validation` variant.
- `examples` module with `Paraboloid`, `Rosenbrock`, `QuadraticImplicit`, and
  `FlexibleDiscipline`, matching Philote-Python's example disciplines.
- Example binaries: `paraboloid_server`, `paraboloid_client`, `quadratic_implicit`,
  and `quadratic_client`.
- Test coverage for the gRPC layer, which previously had none: 217 tests spanning
  wire encoding, dynamic shapes, discrete variables, edge cases, and end-to-end
  client/server round trips.
- `PartialsMetaData.shape` is now populated by the server.
- `VariableRegistry::discrete_input_names`, the declared-name set used to reject
  undeclared discrete inputs. `discrete_input_defaults` cannot serve this purpose
  because a discrete input may be declared without a default.
- `missing_docs` is now warned on, and every public item is documented. CI runs
  clippy with `-D warnings` over `--all-targets`, so this is enforced.
- `rust-version = "1.70"`, matching what the README already claimed.

### Changed

- **Breaking:** `Discipline` now requires only `registry()` and `registry_mut()`.
  `add_input`, `add_output`, `add_option`, `declare_partials`,
  `get_variable_definitions`, `get_partials_definitions`, and
  `get_available_options` became provided methods that delegate to the registry.
  Migrate by holding a `VariableRegistry` field and invoking `impl_registry!`.
- **Breaking:** `ImplicitDiscipline::apply_linear` now takes `d_inputs`,
  `d_outputs`, and `d_residuals` plus a `LinearMode`, matching Philote-Python.
  Reverse-mode accumulation was inexpressible in the previous signature.
- **Breaking:** `Discipline::setup` is required rather than defaulted. A discipline
  that declares no variables would otherwise compile and then fail at the first
  compute call with a confusing "variable not found".
- **Breaking:** clients must call `setup()` and `get_variable_definitions()` before
  computing; doing otherwise returns `SetupNotCalled` instead of silently
  producing empty results. Re-running `setup()` requires re-fetching.
- Option type names use Philote-Python's vocabulary (`bool`/`int`/`float`/`str`/
  `dict`); `double`, `string`, and `struct` are still accepted as aliases.
- Removed `utils::reassemble_arrays_from_chunks` (unimplemented) and
  `utils::chunk_arrays_for_streaming` (a duplicate encoder); both are superseded by
  the `wire` module, which is now the single implementation of the chunk format.

### Notes on interoperability with Philote-Python

Checked manually (not in CI) in both directions for the explicit path, with
identical numeric results. `tests/interop_notes.rs` pins the wire conventions that
check relied on, but it runs in-process and starts no Python. Two upstream Philote-Python defects are documented rather than reproduced (see
`tests/interop_notes.rs`): its `GetInfo` is a generator although the proto
declares it unary, so the call fails for any client including Python's own; and
its implicit server emits an exclusive `Array.end` while its explicit server and
both of its clients use an inclusive one.

## [0.1.0] - 2026-05-16

### Added

- Async gRPC-based discipline server and client implementation
- Support for explicit and implicit discipline types
- Streaming data transfer for variable arrays
- Protocol Buffer communication using Philote-MDO v0.8.0
- Comprehensive error handling with domain-specific error types
- Paraboloid example discipline

[Unreleased]: https://github.com/MDO-Standards/Philote-Rust/compare/v0.1.0...develop
[0.1.0]: https://github.com/MDO-Standards/Philote-Rust/releases/tag/v0.1.0
