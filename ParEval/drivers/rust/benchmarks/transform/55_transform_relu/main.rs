// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1 << 20; // Default problem size
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code
// ============================================================================

/* Compute the ReLU function on every element of x. Elements less than zero become zero,
   while elements greater than zero stay the same.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [-1.8, 24.0, 1.2, 0.0, -5.1, -0.2, 4.5]
   output: [0, 24.0, 1.2, 0, 0, 0, 4.5]
*/
pub fn relu(x: &mut [f64]) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_relu(x: &mut [f64]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

// Helpers
fn fill_rand(slice: &mut [f64], min: f64, max: f64) {
    let mut rng = rand::thread_rng();
    for x in slice.iter_mut() {
        *x = rng.gen_range(min..max);
    }
}

fn fequal(a: &[f64], b: &[f64], epsilon: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= epsilon)
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct ReluContext {
    x: Vec<f64>,
}

impl ParEvalBenchmark for ReluContext {
    fn new() -> Self {
        let DRIVER_PROBLEM_SIZE = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            x: vec![0.0; DRIVER_PROBLEM_SIZE],
        }
    }

    fn reset(&mut self) {
        fill_rand(&mut self.x, -50.0, 50.0);
    }

    fn compute(&mut self) {
        relu(&mut self.x);
    }

    fn best(&mut self) {
        correct_relu(&mut self.x);
    }

    fn validate(&mut self) -> bool {
        const TEST_SIZE: usize = 1024;
        let mut input = vec![0.0; TEST_SIZE];
        let mut correct_result = vec![0.0; TEST_SIZE];
        let mut test_result = vec![0.0; TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // Set up input
            fill_rand(&mut input, -50.0, 50.0);

            // Compute correct result
            correct_result.copy_from_slice(&input);
            correct_relu(&mut correct_result);

            // Compute test result
            test_result.copy_from_slice(&input);
            relu(&mut test_result);

            // Verify
            if !fequal(&correct_result, &test_result, 1e-6) {
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
    run::<ReluContext>();
}