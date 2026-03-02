use pareval_runner::{ParEvalBenchmark, run};
use rand::prelude::*;
use rayon::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1_000_000;
const TEST_SIZE: usize = 2048;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Compute the prefix sum of the vector x into output.
   Use Rust Rayon to compute in parallel.
   Example:
   
   input: [1, 7, 4, 6, 6, 2]
   output: [1, 8, 12, 18, 24, 26]
*/
pub fn prefix_sum(x: &[f64], output: &mut [f64]) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_prefix_sum(x: &[f64], output: &mut [f64]) {
    let mut acc = 0.0;
    for (val, out) in x.iter().zip(output.iter_mut()) {
        acc += val;
        *out = acc;
    }
}

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

struct ScanContext {
    x: Vec<f64>,
    output: Vec<f64>,
}

impl ParEvalBenchmark for ScanContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            x: vec![0.0; driver_problem_size],
            output: vec![0.0; driver_problem_size],
        }
    }

    fn reset(&mut self) {
        fill_rand(&mut self.x, -100.0, 100.0);
    }

    fn compute(&mut self) {
        prefix_sum(&self.x, &mut self.output);
    }

    fn best(&mut self) {
        correct_prefix_sum(&self.x, &mut self.output);
    }

    fn validate(&mut self) -> bool {
        let mut input = vec![0.0; TEST_SIZE];
        let mut correct = vec![0.0; TEST_SIZE];
        let mut test = vec![0.0; TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            fill_rand(&mut input, -100.0, 100.0);
            
            // Clear outputs to ensure no stale data
            correct.fill(0.0);
            test.fill(0.0);

            // Compute correct result
            correct_prefix_sum(&input, &mut correct);

            // Compute test result
            prefix_sum(&input, &mut test);

            if !fequal(&correct, &test, 1e-6) {
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
    run::<ScanContext>();
}