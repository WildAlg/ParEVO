// main.rs
use pareval_runner::{ParEvalBenchmark, run};
use rand::prelude::*;
use rayon::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024 * 1024;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Compute the prefix sum array of the vector x and return its sum.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [-7, 2, 1, 9, 4, 8]
   output: 15
*/
pub fn sum_of_prefix_sum(x: &[f64]) -> f64 {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_sum_of_prefix_sum(x: &[f64]) -> f64 {
    // Equivalent to:
    // std::vector<double> prefixSum(x.size());
    // std::inclusive_scan(x.begin(), x.end(), prefixSum.begin());
    // return std::accumulate(prefixSum.begin(), prefixSum.end(), 0.0);

    let mut prefix_sum = vec![0.0; x.len()];
    let mut current_sum = 0.0;
    
    // Inclusive scan
    for (i, &val) in x.iter().enumerate() {
        current_sum += val;
        prefix_sum[i] = current_sum;
    }

    // Accumulate
    prefix_sum.iter().sum()
}

fn fill_rand(slice: &mut [f64], min: f64, max: f64) {
    let mut rng = rand::thread_rng();
    for x in slice.iter_mut() {
        *x = rng.gen_range(min..max);
    }
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct Context {
    x: Vec<f64>,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Context {
            x: vec![0.0; driver_problem_size],
        }
    }

    fn reset(&mut self) {
        fill_rand(&mut self.x, -100.0, 100.0);
    }

    fn compute(&mut self) {
        let val = sum_of_prefix_sum(&self.x);
        // Prevent optimization away
        std::hint::black_box(val);
    }

    fn best(&mut self) {
        let val = correct_sum_of_prefix_sum(&self.x);
        std::hint::black_box(val);
    }

    fn validate(&mut self) -> bool {
        let mut input = vec![0.0; TEST_SIZE];
        
        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // set up input
            fill_rand(&mut input, -100.0, 100.0);

            // compute correct result
            let correct_result = correct_sum_of_prefix_sum(&input);

            // compute test result
            let test_result = sum_of_prefix_sum(&input);

            // check correctness
            let diff = (correct_result - test_result).abs();
            if diff > 1e-5 {
                return false;
            }
        }
        
        true
    }
}

// ============================================================================
//  4. Entry Point
// ============================================================================

fn main() {
    run::<Context>();
}