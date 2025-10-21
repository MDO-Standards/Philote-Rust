[![Build](https://github.com/chrislupp/Philote-Rust/actions/workflows/build.yml/badge.svg)](https://github.com/chrislupp/Philote-Rust/actions/workflows/build.yml)
[![Test](https://github.com/chrislupp/Philote-Rust/actions/workflows/test.yml/badge.svg)](https://github.com/chrislupp/Philote-Rust/actions/workflows/test.yml)
[![Clippy](https://github.com/chrislupp/Philote-Rust/actions/workflows/clippy.yml/badge.svg)](https://github.com/chrislupp/Philote-Rust/actions/workflows/clippy.yml)
[![Format Check](https://github.com/chrislupp/Philote-Rust/actions/workflows/fmt.yml/badge.svg)](https://github.com/chrislupp/Philote-Rust/actions/workflows/fmt.yml)

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
- **Automatic gradient computation** support
- **Efficient array streaming** for large data transfers
- **Comprehensive error handling** with custom error types
- **Zero-copy views** for performance-critical operations

## Installation

Add Philote to your `Cargo.toml`:

```toml
[dependencies]
philote = { git = "https://github.com/chrislupp/Philote-Rust.git" }
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

Here's a simple example of creating an explicit discipline that computes a paraboloid function:

```rust
use async_trait::async_trait;
use ndarray::ArrayD;
use std::collections::HashMap;
use philote::{
    traits::{Discipline, ExplicitDiscipline},
    server::ExplicitServer,
    ArrayMap, PartialMap, Result,
};

struct Paraboloid {
    // Your discipline state
}

#[async_trait]
impl ExplicitDiscipline for Paraboloid {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let x = inputs["x"][[0]];
        let y = inputs["y"][[0]];

        // f = (x - 3)^2 + x*y + (y + 4)^2 - 3
        let f = (x - 3.0).powi(2) + x * y + (y + 4.0).powi(2) - 3.0;

        let mut outputs = HashMap::new();
        outputs.insert("f".to_string(), ArrayD::from_elem(vec![1], f));
        Ok(outputs)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let discipline = Paraboloid::new();
    let server = ExplicitServer::new(discipline);

    // Server is now ready to handle gRPC requests
    Ok(())
}
```

### Creating a Client

Connect to and interact with a Philote server:

```rust
use philote::client::ExplicitClient;
use ndarray::ArrayD;

#[tokio::main]
async fn main() -> Result<()> {
    // Connect to the server
    let mut client = ExplicitClient::connect("http://localhost:50051").await?;

    // Get discipline information
    let info = client.get_info().await?;
    println!("Connected to: {} v{}", info.name, info.version);

    // Prepare inputs
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), ArrayD::from_elem(vec![1], 2.0));
    inputs.insert("y".to_string(), ArrayD::from_elem(vec![1], -1.0));

    // Compute outputs
    let outputs = client.compute(inputs).await?;
    println!("Result: f = {}", outputs["f"][[0]]);

    Ok(())
}
```

## Examples

The repository includes several complete examples:

- **paraboloid.rs** - Simple explicit discipline demonstrating core functionality
- **server_runner.rs** - Full gRPC server setup with connection handling
- **client_example.rs** - Client usage patterns and error handling

Run an example:
```bash
cargo run --example paraboloid
```

## Architecture

### Core Components

- **Traits** (`traits.rs`) - Define the interfaces for disciplines
  - `Discipline` - Base trait for all analysis types
  - `ExplicitDiscipline` - For direct input-output mappings
  - `ImplicitDiscipline` - For residual-based formulations

- **Server** (`server/`) - gRPC server implementations
  - `ExplicitServer` - Serves explicit disciplines
  - `ImplicitServer` - Serves implicit disciplines

- **Client** (`client/`) - gRPC client implementations
  - `ExplicitClient` - Connects to explicit discipline servers
  - `ImplicitClient` - Connects to implicit discipline servers

- **Types** (`types.rs`) - Core data structures and conversions
- **Utils** (`utils.rs`) - Helper functions for array operations

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
│   ├── client/          # Client implementations
│   ├── server/          # Server implementations
│   ├── lib.rs          # Library entry point
│   ├── traits.rs       # Core trait definitions
│   ├── types.rs        # Data structures
│   ├── error.rs        # Error types
│   └── utils.rs        # Utility functions
├── examples/           # Usage examples
├── tests/             # Integration tests
├── proto/             # Protocol buffer definitions (submodule)
└── Cargo.toml         # Package manifest
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
