// main.rs
use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024 * 1024;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* Return the value of the smallest odd number in the vector x.
   Use Rust Rayon to compute in parallel.
   Examples:

   input: [7, 9, 5, 2, 8, 16, 4, 1]
   output: 1

   input: [8, 36, 7, 2, 11]
   output: 7
*/
pub fn smallest_odd(x: &[i32]) -> i32 {
    // LLM_OUTPUT_HERE
}

// Sequential baseline equivalent to correctSmallestOdd in baseline.hpp
fn correct_smallest_odd(x: &[i32]) -> i32 {
    x.iter()
        .filter(|&&v| v % 2 != 0) // Check if number is odd
        .cloned()
        .min()
        .unwrap_or(i32::MAX) // Default to MAX if no odd numbers found
}

struct Context {
    x: Vec<i32>,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Context {
            x: vec![0; driver_problem_size],
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        for val in self.x.iter_mut() {
            // C++ uses fillRand(ctx->x, 0.0, 100.0)
            *val = rng.gen_range(0..100);
        }
    }

    fn compute(&mut self) {
        let _ = smallest_odd(&self.x);
    }

    fn best(&mut self) {
        let _ = correct_smallest_odd(&self.x);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        let mut x = vec![0; TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // Set up input
            for val in x.iter_mut() {
                *val = rng.gen_range(0..100);
            }

            // Compute correct result
            let correct = correct_smallest_odd(&x);

            // Compute test result
            let test = smallest_odd(&x);

            // Compare
            if correct != test {
                return false;
            }
        }
        true
    }
}

fn main() {
    run::<Context>();
}