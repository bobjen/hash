// Find the crossover point where compute() becomes slower than concat()
//
// This answers: "At what byte length N is it faster to use concat()
// rather than recomputing the CRC?"

use crc64_nvme::Crc;
use std::time::Instant;

fn main() {
    let crc = Crc::new();

    println!("Finding concat() vs compute() crossover point");
    println!("==============================================\n");

    // First, measure concat() performance
    let mut concat_times = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        for _ in 0..100_000 {
            let _ = crc.concat(0, 0, 0x1234567890abcdef, 1000, 0, 0xfedcba0987654321, 1000);
        }
        let elapsed = start.elapsed().as_secs_f64();
        concat_times.push(elapsed / 100_000.0 * 1_000_000_000.0); // ns per operation
    }

    let concat_avg = concat_times.iter().sum::<f64>() / concat_times.len() as f64;
    let concat_min = concat_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let concat_max = concat_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("concat() performance: avg={:.1} ns, min={:.1} ns, max={:.1} ns\n",
             concat_avg, concat_min, concat_max);

    // Now test compute() at various sizes
    println!("Size (bytes) | compute() time (ns) | Faster than concat?");
    println!("-------------|---------------------|-------------------");

    let sizes = [
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
        1024, 2048, 4096, 8192, 16384, 32768, 65536
    ];

    let mut crossover_size = None;

    for &size in &sizes {
        let data = vec![0x42u8; size];
        let iterations = if size <= 256 { 100_000 } else { 100_000 / (size / 256) };

        let start = Instant::now();
        let mut result = 0u64;
        for _ in 0..iterations {
            result = crc.compute(&data, result);
        }
        let elapsed = start.elapsed().as_secs_f64();
        let time_ns = (elapsed / iterations as f64) * 1_000_000_000.0;

        let faster = if time_ns < concat_avg { "YES" } else { "NO" };

        if crossover_size.is_none() && time_ns >= concat_avg {
            crossover_size = Some(size);
        }

        println!("{:12} | {:19.1} | {:^19} (result: {:x})",
                 size, time_ns, faster, result);
    }

    println!("\n==============================================");
    if let Some(crossover) = crossover_size {
        println!("CROSSOVER POINT: ~{} bytes", crossover);
        println!("\nConclusion:");
        println!("  - For N < {} bytes: compute() is faster", crossover);
        println!("  - For N >= {} bytes: concat() is faster", crossover);
    } else {
        println!("compute() is faster than concat() for all tested sizes!");
    }

    println!("\nPractical implications:");
    println!("  - Small updates (< {} bytes): just recompute", crossover_size.unwrap_or(65536));
    println!("  - Large updates (>= {} bytes): use concat()", crossover_size.unwrap_or(65536));
    println!("  - concat() advantage grows with data size (O(log n) vs O(n))");
}
