[![Tests](https://github.com/MDO-Standards/Philote-Rust/actions/workflows/tests.yaml/badge.svg)](https://github.com/MDO-Standards/Philote-Rust/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/MDO-Standards/Philote-Rust/branch/main/graph/badge.svg)](https://codecov.io/gh/MDO-Standards/Philote-Rust)

<div align="center">
<img src="https://github.com/MDO-Standards/Philote-MDO/blob/main/doc/graphics/logos/philote.svg?raw=true" width="500">
</div>

# Philote-Rust

A Rust library for building and interacting with Philote MDO (Multidisciplinary Design Optimization) analysis servers using gRPC and Protocol Buffers.

## Overview

Philote-Rust provides a high-performance, type-safe implementation for creating distributed analysis services in MDO frameworks. It enables seamless integration of computational disciplines written in Rust with MDO frameworks, supporting both explicit and implicit analysis types.

### Key Features

- **Type-safe gRPC communication** using Protocol Buffers
- **Async/await** support with Tokio runtime
- **Flexible discipline types**:
  - Explicit disciplines (direct input-output mappings)
  - Implicit disciplines (residual-based formulations)
- **Analytic gradients** for both discipline types
- **Discrete variables** alongside continuous arrays
- **Client-resolved shapes** for disciplines sized at runtime
- **N-dimensional arrays**, chunked and streamed for large transfers
- **Comprehensive error handling** with custom error types

### Interoperability

This crate implements the [Philote-MDO standard](https://github.com/MDO-Standards/Philote-MDO)
(v0.8.0), so its clients and servers interoperate with other implementations of
the standard. Verified against Philote-Python: a Rust client drives a Python
paraboloid server and vice versa, with identical results.

Two caveats when talking to Philote-Python specifically:

- Its `GetInfo` RPC is implemented as a generator although the proto declares it
  unary, so the call fails with `Failed to serialize response!` for *any* client,
  including Python's own. Avoid `get_info` against a Python server.
- Its **implicit** server emits an exclusive `Array.end` while its explicit server
  and both of its clients use an inclusive one. This crate follows the standard
  (inclusive) everywhere, so a Rust client cannot decode responses from a Python
  implicit server until that is fixed upstream. The reverse direction — a Python
  client against a Rust implicit server — works correctly.

Both are upstream defects in Philote-Python, not divergences introduced here; see
`tests/interop_notes.rs`.

## Installation

Add Philote to your `Cargo.toml`:

```toml
[dependencies]
philote-mdo = { git = "https://github.com/MDO-Standards/Philote-Rust.git" }
```

### Prerequisites

- Rust 1.70 or later
- Protocol Buffers compiler (`protoc`) for building from source

On Ubuntu/Debian:
```bash
sudo apt-get install protobuf-compiler
```

On macOS:
```bash
brew install protobuf
```

## Quick Start

### Creating a Discipline Server

Here's an explicit discipline that computes a paraboloid function, served over
gRPC. A discipline stores its metadata in a `VariableRegistry`, so it only supplies
the two accessors — via the `impl_registry!` macro — plus the hooks it needs.

```rust
use async_trait::async_trait;
use ndarray::ArrayD;
use std::collections::HashMap;
use std::sync::Arc;
use philote_mdo::{
    impl_registry,
    philote_info::{
        discipline_service_server::DisciplineServiceServer,
        explicit_service_server::ExplicitServiceServer,
    },
    registry::VariableRegistry,
    server::ExplicitServer,
    traits::{Discipline, ExplicitDiscipline},
    ArrayMap, Result,
};
use tonic::transport::Server;

#[derive(Default)]
struct Paraboloid {
    registry: VariableRegistry,
}

impl Discipline for Paraboloid {
    impl_registry!(registry);

    fn name(&self) -> &str { "Paraboloid" }
    fn provides_gradients(&self) -> bool { true }

    fn setup(&mut self) -> Result<()> {
        self.add_input("x", &[1], "m")?;
        self.add_input("y", &[1], "m")?;
        self.add_output("f_xy", &[1], "m**2")
    }

    fn setup_partials(&mut self) -> Result<()> {
        self.declare_partials("f_xy", "x")?;
        self.declare_partials("f_xy", "y")
    }
}

#[async_trait]
impl ExplicitDiscipline for Paraboloid {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let x = inputs["x"][[0]];
        let y = inputs["y"][[0]];

        // f = (x - 3)^2 + x*y + (y + 4)^2 - 3
        let f = (x - 3.0).powi(2) + x * y + (y + 4.0).powi(2) - 3.0;

        let mut outputs = HashMap::new();
        outputs.insert("f_xy".to_string(), ArrayD::from_elem(vec![1], f));
        Ok(outputs)
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // One instance backs both services.
    let server = Arc::new(ExplicitServer::new(Paraboloid::default()));

    Server::builder()
        .add_service(DisciplineServiceServer::from_arc(server.clone()))
        .add_service(ExplicitServiceServer::from_arc(server))
        .serve("127.0.0.1:50051".parse()?)
        .await?;

    Ok(())
}
```

### Creating a Client

Connect to and interact with a Philote server:

```rust
use philote_mdo::client::ExplicitClient;
use ndarray::ArrayD;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> philote_mdo::Result<()> {
    let mut client = ExplicitClient::connect("http://localhost:50051").await?;

    let info = client.get_info().await?;
    println!("Connected to: {} v{}", info.name, info.version);

    // Run setup and fetch metadata before computing: array responses carry only a
    // flat index range, so the declared shapes are what restore their dimensions.
    client.setup().await?;
    client.get_variable_definitions().await?;
    client.get_partial_definitions().await?;

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), ArrayD::from_elem(vec![1], 2.0));
    inputs.insert("y".to_string(), ArrayD::from_elem(vec![1], -1.0));

    let outputs = client.compute_function(&inputs).await?;
    println!("Result: f_xy = {}", outputs["f_xy"][[0]]);

    let partials = client.compute_gradient(&inputs).await?;
    println!("df/dx = {}", partials[&("f_xy".to_string(), "x".to_string())][[0]]);

    Ok(())
}
```

## Examples

Ready-made disciplines live in `philote_mdo::examples`: `Paraboloid`,
`Rosenbrock` (integer `dimension` option), `QuadraticImplicit` (implicit, with
`apply_linear`), and `FlexibleDiscipline` (client-resolved shapes). They mirror
Philote-Python's `philote_mdo/examples/`, so results compare directly.

Runnable binaries in `examples/`:

| Example | Description |
| --- | --- |
| `paraboloid` | Runs the paraboloid locally, no server |
| `paraboloid_server` / `paraboloid_client` | Explicit discipline over gRPC |
| `quadratic_implicit` / `quadratic_client` | Implicit discipline over gRPC |
| `server_runner` / `client_example` | Full walkthrough, including options and chunked streaming |

Start a server and drive it from a second terminal:

```bash
cargo run --example paraboloid_server
cargo run --example paraboloid_client
```

## Architecture

### Core Components

- **Traits** (`traits.rs`) - Define the interfaces for disciplines
  - `Discipline` - Base trait for all analysis types
  - `ExplicitDiscipline` - For direct input-output mappings
  - `ImplicitDiscipline` - For residual-based formulations

- **Registry** (`registry.rs`) - `VariableRegistry`, the metadata store backing
  every discipline. Owning this centrally is what makes shape resolution,
  duplicate detection, residual twins, and re-`Setup` clearing possible.

- **Server** (`server/`) - gRPC server implementations
  - `ExplicitServer` - Serves explicit disciplines
  - `ImplicitServer` - Serves implicit disciplines

- **Client** (`client/`) - gRPC client implementations
  - `ExplicitClient` - Connects to explicit discipline servers
  - `ImplicitClient` - Connects to implicit discipline servers

- **Wire** (`wire.rs`) - Chunk encoding and decoding, shared by client and server
- **Discrete** (`discrete.rs`) - JSON ↔ protobuf `Value` conversions
- **Validation** (`validation.rs`) - Input validation helpers
- **Types** (`types.rs`) - Core data structures and conversions
- **Utils** (`utils.rs`) - Helper functions for array operations
- **Examples** (`examples/`) - Ready-to-run example disciplines

### Data Flow

```
Client Request → gRPC → Server → Discipline.compute() → Server → gRPC → Client Response
```

Array data is automatically chunked and streamed for efficient transfer of large datasets.

## Protocol Buffers

This library uses the [Philote MDO Protocol Buffers](https://github.com/MDO-Standards/Philote-MDO) specification. The proto definitions are included as a submodule.

To update proto definitions:
```bash
git submodule update --init --recursive
```

## Development

### Building

```bash
cargo build
```

### Testing

```bash
# Run all tests
cargo test

# Run with verbose output
cargo test -- --nocapture
```

### Linting and Formatting

```bash
# Check formatting
cargo fmt --check

# Run Clippy lints
cargo clippy -- -D warnings
```

## Project Structure

```
philote-rust/
├── src/
│   ├── client/         # Client implementations
│   ├── server/         # Server implementations
│   ├── examples/       # Example disciplines
│   ├── lib.rs          # Library entry point
│   ├── traits.rs       # Core trait definitions
│   ├── registry.rs     # Variable and option metadata store
│   ├── wire.rs         # Chunk encoding and decoding
│   ├── discrete.rs     # JSON <-> protobuf Value conversions
│   ├── validation.rs   # Input validation
│   ├── types.rs        # Data structures
│   ├── error.rs        # Error types
│   └── utils.rs        # Utility functions
├── examples/           # Runnable example binaries
├── tests/              # Integration tests
├── proto/              # Protocol buffer definitions (submodule)
└── Cargo.toml          # Package manifest
```

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `cargo test`
2. Code is formatted: `cargo fmt`
3. No Clippy warnings: `cargo clippy`
4. Add tests for new functionality

## Documentation

Protocol documentation is available at: https://mdo-standards.github.io/Philote-MDO/

## License

Copyright 2022-2025 Christopher A. Lupp

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

This work has been cleared for public release, distribution unlimited, case
number: AFRL-2023-1321. The views expressed are those of the author and do
not necessarily reflect the official policy or position of the Department of
the Air Force, the Department of Defense, or the U.S. government.
