# CRC-64-NVME Implementation

High-performance Rust implementation of CRC-64 using the NVMe polynomial.

**Translated from Bob Jenkins' public domain C++ implementation**

## Quick Start

```rust
use crc64_nvme::Crc;

let crc = Crc::new();
let data = b"hello world!";
let checksum = crc.compute(data, 0);
println!("CRC: 0x{:016x}", checksum);
```

## Build and Test

```bash
# Build the library
cargo build --release

# Run tests
cargo test --release

# Run examples
cargo run --release --example crc_example
cargo run --release --example concat_crossover
```

## Performance

- **16 GB/s** throughput on modern CPUs
- **~5-10 ns** for small inputs (1-16 bytes)
- **~9 ns** constant-time concatenation regardless of data size

See [CRC_BENCHMARKS.md](CRC_BENCHMARKS.md) for detailed comparisons.

## Documentation

- **[CRC_README.md](CRC_README.md)** - Complete usage guide and API documentation
- **[CRC_BENCHMARKS.md](CRC_BENCHMARKS.md)** - Performance comparisons
- **[examples/crc_example.rs](examples/crc_example.rs)** - Common usage patterns
- **[examples/concat_crossover.rs](examples/concat_crossover.rs)** - Performance analysis

## Key Features

✅ **SIMD Acceleration** - Uses PCLMULQDQ instructions
✅ **Small Value Optimization** - Fast paths for 1-16 byte inputs
✅ **O(log n) Concatenation** - Merge CRCs without recomputation
✅ **Zero Dependencies** - Only uses `std::arch::x86_64` intrinsics
✅ **Production Ready** - Comprehensive tests and documentation

## Requirements

- Rust 1.70+
- x86_64 CPU with PCLMULQDQ support (Intel 2010+, AMD 2011+)
- Compiler flags configured in `.cargo/config.toml` (already included)

## License

Public domain (maintains status of original C++ implementation).
Attribution to Bob Jenkins appreciated but not required.

## Credits

- **Original C++ Implementation**: Bob Jenkins (https://burtleburtle.net/bob/hash/crc.html)
- **Rust Translation**: Claude Sonnet 4.5
- **Performance Testing**: AMD Ryzen 7 5825U @ 16 GB/s
