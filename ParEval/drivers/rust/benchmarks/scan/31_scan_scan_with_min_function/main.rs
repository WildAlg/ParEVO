use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1_000_000;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Replace the i-th element of the vector x with the minimum value from indices 0 through i.
   Use Rust Rayon to compute in parallel.
   Examples:

   input: [8, 6, -1, 7, 3, 4, 4]
   output: [8, 6, -1, -1, -1, -1, -1]

   input: [5, 4, 6, 4, 3, 6, 1, 1]
   output: [5, 4, 4, 4, 3, 3, 1, 1]
*/
pub fn partial_minimums(x: &mut [f32]) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_partial_minimums(x: &mut [f32]) {
    let mut running_min = std::f32::MAX;
    for val in x.iter_mut() {
        if *val < running_min {
            running_min = *val;
        }
        *val = running_min;
    }
}

fn fill_rand(slice: &mut [f32], min: f32, max: f32) {
    let mut rng = rand::thread_rng();
    for x in slice.iter_mut() {
        *x = rng.gen_range(min..max);
    }
}

fn fequal(a: &[f32], b: &[f32], epsilon: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= epsilon)
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct Context {
    x: Vec<f32>,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            x: vec![0.0; driver_problem_size],
        }
    }

    fn reset(&mut self) {
        fill_rand(&mut self.x, -100.0, 100.0);
    }

    fn compute(&mut self) {
        partial_minimums(&mut self.x);
    }

    fn best(&mut self) {
        correct_partial_minimums(&mut self.x);
    }

    fn validate(&mut self) -> bool {
        const TEST_SIZE: usize = 1024;
        let mut input = vec![0.0; TEST_SIZE];
        let mut correct = vec![0.0; TEST_SIZE];
        let mut test = vec![0.0; TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            fill_rand(&mut input, -100.0, 100.0);

            correct.copy_from_slice(&input);
            correct_partial_minimums(&mut correct);

            test.copy_from_slice(&input);
            partial_minimums(&mut test);

            if !fequal(&correct, &test, 1e-3) {
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