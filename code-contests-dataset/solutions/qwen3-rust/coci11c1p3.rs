use std::io::{self, Read};
use rayon::prelude::*;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    
    let mut lines = input.lines();
    let n: usize = lines.next().unwrap().parse().unwrap();
    let mut numbers: Vec<u32> = Vec::with_capacity(n);
    
    for _ in 0..n {
        numbers.push(lines.next().unwrap().parse().unwrap());
    }
    
    // For each bit position, count how many numbers have 0 and 1
    let mut bit_counts = vec![(0u64, 0u64); 20]; // 20 bits enough for numbers < 1_000_000
    
    for &num in &numbers {
        for i in 0..20 {
            if num & (1 << i) != 0 {
                bit_counts[i].1 += 1;
            } else {
                bit_counts[i].0 += 1;
            }
        }
    }
    
    // Calculate contribution of each bit position to total sum
    let result: u64 = bit_counts
        .par_iter()
        .enumerate()
        .map(|(i, &(zeros, ones))| {
            // For each bit position, the friendship value will be 1 if bits differ
            // Number of pairs with differing bits = zeros * ones
            zeros * ones * (1u64 << i)
        })
        .sum();
    
    println!("{}", result);
}