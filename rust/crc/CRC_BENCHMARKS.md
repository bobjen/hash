# CRC-64-NVME Performance Benchmarks

Comparison of CRC-64-NVME implementations on AMD Ryzen 7 5825U.

## Test Date
2026-02-15

## Implementations Tested

1. **Bob Jenkins' C++ (reference)** - from https://github.com/bobjen/hash/cplus/crc
2. **Our Rust Implementation** - Claude 4.5's direct translation of Jenkins' C++
3. **crc64fast-nvme** - Popular Rust crate (now deprecated)

## Test Methodology

- All tests compiled with `-C target-cpu=native` (Rust) or `-O3 -mpclmul -msse4.1` (C++)
- Each size tested with 1M iterations (for small) or scaled appropriately

## Performance Results

| Size | Our Rust | crc64fast-nvme | Our Speedup |
|------|----------|----------------|-------------|
| 1 byte | 0.0029 µs (0.34 GB/s) | 0.0047 µs (0.21 GB/s) | **1.6x** |
| 2 bytes | 0.0034 µs (0.58 GB/s) | 0.0071 µs (0.28 GB/s) | **2.1x** |
| 4 bytes | 0.0036 µs (1.10 GB/s) | 0.0079 µs (0.51 GB/s) | **2.2x** |
| 8 bytes | 0.0044 µs (1.83 GB/s) | 0.0082 µs (0.97 GB/s) | **1.9x** |
| 16 bytes | 0.0059 µs (2.72 GB/s) | 0.0076 µs (2.11 GB/s) | **1.3x** |
| 32 bytes | 0.0079 µs (4.04 GB/s) | 0.0111 µs (2.88 GB/s) | **1.4x** |
| 64 bytes | 0.0089 µs (7.20 GB/s) | 0.0204 µs (3.14 GB/s) | **2.3x** |
| 128 bytes | 0.0094 µs (13.68 GB/s) | 0.0143 µs (8.98 GB/s) | **1.5x** |
| 256 bytes | 0.0176 µs (14.53 GB/s) | 0.0211 µs (12.12 GB/s) | **1.2x** |
| 512 bytes | 0.0356 µs (14.36 GB/s) | 0.0361 µs (14.19 GB/s) | **1.0x** |
| 1024 bytes | 0.0643 µs (15.93 GB/s) | 0.0672 µs (15.24 GB/s) | **1.05x** |
| 2048 bytes | 0.1267 µs (16.17 GB/s) | 0.1298 µs (15.78 GB/s) | **1.02x** |
| 4096 bytes | 0.2499 µs (16.39 GB/s) | 0.2544 µs (16.10 GB/s) | **1.02x** |
| 8192 bytes | 0.5014 µs (16.34 GB/s) | 0.5049 µs (16.23 GB/s) | **1.01x** |
| 16384 bytes | 1.007 µs (16.27 GB/s) | 1.085 µs (15.10 GB/s) | **1.08x** |
| 32768 bytes | 2.025 µs (16.18 GB/s) | 2.034 µs (16.11 GB/s) | **1.00x** |
| 65536 bytes | 4.022 µs (16.30 GB/s) | 4.119 µs (15.91 GB/s) | **1.02x** |

## C++ Reference Comparison

At 65536 bytes (where SIMD implementations converge):

| Implementation | Time | Throughput |
|----------------|------|------------|
| Bob Jenkins C++ (reference) | 3.80 µs | 17.24 GB/s |
| Our Rust translation | 4.02 µs | 16.30 GB/s |
| crc64fast-nvme | 4.12 µs | 15.91 GB/s |

Our Rust is within 6% of the original C++ reference at large sizes.

## Feature Comparison

| Feature | Our Rust | crc64fast-nvme | Bob Jenkins C++ |
|---------|----------|----------------|-----------------|
| SIMD Acceleration | ✅ | ✅ | ✅ |
| Small value optimization | ✅ | ❌ | ✅ |
| **O(log n) Concat** | ✅ | ❌ | ✅ |
| Incremental updates | ✅ | ✅ | ✅ |
| Zero dependencies | ✅ | ❌ (uses `crc` crate) | ✅ |
| Active maintenance | ✅ | ❌ (deprecated) | ✅ |

## Verification

`"hello world!" → 0xd9160d1fa8e418e3`

Use this to verify your port gives correct results.

It's fast and useful. You should give it a try.

---

**Test Platform:** AMD Ryzen 7 5825U
**Compiled with:** `RUSTFLAGS="-C target-cpu=native"`
**Rust Version:** 1.70+
**Date:** 2026-02-15
