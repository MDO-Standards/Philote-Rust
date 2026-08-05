fn main() -> Result<(), Box<dyn std::error::Error>> {
    // tonic-build shells out to `protoc`. Rather than require every downstream
    // user (and docs.rs, which has no protobuf-compiler) to install it, fall back
    // to the vendored binary. An explicit PROTOC still wins, so distributions that
    // must build against their own toolchain can override.
    if std::env::var_os("PROTOC").is_none() {
        std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path()?);
    }

    tonic_build::configure()
        .compile_protos(&["proto/data.proto", "proto/disciplines.proto"], &["proto"])?;

    Ok(())
}
