use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024 * 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* Replace every element of x with the square of its value.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [5, 1, 2, -4, 8]
   output: [25, 1, 4, 16, 64]
*/
pub fn square_each(x: &mut [i32]) {
    // LLM_OUTPUT_HERE
}

fn correct_square_each(x: &mut [i32]) {
    for i in 0..x.len() {
        x[i] = x[i] * x[i];
    }
}

struct Context {
    x: Vec<i32>,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let DRIVER_PROBLEM_SIZE = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Context {
            x: vec![0; DRIVER_PROBLEM_SIZE],
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        for v in self.x.iter_mut() {
            // C++ fillRand(..., -50, 50) is inclusive
            *v = rng.gen_range(-50..=50);
        }
    }

    fn compute(&mut self) {
        square_each(&mut self.x);
    }

    fn best(&mut self) {
        correct_square_each(&mut self.x);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            let mut input = vec![0; 1024];
            for v in input.iter_mut() {
                *v = rng.gen_range(-50..=50);
            }

            let mut correct_result = input.clone();
            correct_square_each(&mut correct_result);

            let mut test_result = input.clone();
            square_each(&mut test_result);

            if correct_result != test_result {
                return false;
            }
        }
        true
    }
}

fn main() {
    run::<Context>();
}