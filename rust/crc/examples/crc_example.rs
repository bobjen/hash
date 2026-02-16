// Example: Using CRC-64-NVME for data integrity checking
//
// This demonstrates common patterns for using the high-performance CRC implementation.

use crc64_nvme::Crc;

fn main() {
    let crc = Crc::new();

    // Example 1: Simple checksum
    println!("=== Example 1: Simple Checksum ===");
    let data = b"Hello, World!";
    let checksum = crc.compute(data, 0);
    println!("Data: {:?}", std::str::from_utf8(data).unwrap());
    println!("CRC:  0x{:016x}\n", checksum);

    // Example 2: Verify data integrity
    println!("=== Example 2: Verify Integrity ===");
    let stored_crc = checksum;
    let received_data = b"Hello, World!";
    let computed_crc = crc.compute(received_data, 0);

    if computed_crc == stored_crc {
        println!("✓ Data integrity verified!");
    } else {
        println!("✗ Data corruption detected!");
    }
    println!();

    // Example 3: Fast concatenation
    println!("=== Example 3: Fast Concatenation ===");
    let part1 = b"Hello, ";
    let part2 = b"World!";

    let crc1 = crc.compute(part1, 0);
    let crc2 = crc.compute(part2, 0);

    // Merge CRCs in O(log n) time instead of O(n)
    let merged = crc.concat(
        0,                      // init_crc_ab
        0,                      // init_crc_a
        crc1,                   // final_crc_a
        part1.len() as u64,     // size_a
        0,                      // init_crc_b
        crc2,                   // final_crc_b
        part2.len() as u64,     // size_b
    );

    // Verify it matches direct computation
    let direct = crc.compute(b"Hello, World!", 0);

    println!("Part 1 CRC:    0x{:016x}", crc1);
    println!("Part 2 CRC:    0x{:016x}", crc2);
    println!("Merged CRC:    0x{:016x}", merged);
    println!("Direct CRC:    0x{:016x}", direct);
    println!("Match: {}\n", merged == direct);

    // Example 4: Chunked storage with periodic CRCs
    println!("=== Example 4: Chunked Storage (1KB intervals) ===");
    let large_data = vec![0x42u8; 10000];
    let chunk_size = 1024;
    let mut chunk_crcs = Vec::new();

    for (i, chunk) in large_data.chunks(chunk_size).enumerate() {
        let chunk_crc = crc.compute(chunk, 0);
        chunk_crcs.push(chunk_crc);
        if i < 3 {
            println!("Chunk {} CRC: 0x{:016x}", i, chunk_crc);
        }
    }

    let overhead = (chunk_crcs.len() * 8) as f64 / large_data.len() as f64 * 100.0;
    println!("...");
    println!("Total chunks: {}", chunk_crcs.len());
    println!("Storage overhead: {:.2}%\n", overhead);

    // Example 5: Parallel processing simulation
    println!("=== Example 5: Parallel Processing Pattern ===");
    let data = vec![0x55u8; 100000];
    let mid = data.len() / 2;

    // In real code, these would run in separate threads
    let crc_first_half = crc.compute(&data[..mid], 0);
    let crc_second_half = crc.compute(&data[mid..], 0);

    // Fast merge without recomputing entire dataset
    let parallel_result = crc.concat(
        0,
        0,
        crc_first_half,
        mid as u64,
        0,
        crc_second_half,
        (data.len() - mid) as u64,
    );

    // Verify matches single-threaded computation
    let single_result = crc.compute(&data, 0);

    println!("First half:    0x{:016x}", crc_first_half);
    println!("Second half:   0x{:016x}", crc_second_half);
    println!("Parallel CRC:  0x{:016x}", parallel_result);
    println!("Single CRC:    0x{:016x}", single_result);
    println!("Match: {}", parallel_result == single_result);
}
