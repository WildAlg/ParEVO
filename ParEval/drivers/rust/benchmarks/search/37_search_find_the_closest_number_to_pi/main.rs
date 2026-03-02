use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;
use std::f64::consts::PI;

// const DRIVER_PROBLEM_SIZE: usize = 1 << 20; // ~1 million elements
const TEST_SIZE: usize = 1024;

/* Return the index of the value in the vector x that is closest to the math constant PI.
   Use M_PI for the value of PI.
   Use Rust Rayon to search in parallel.
   Example:

   input: [9.18, 3.05, 7.24, 11.3, -166.49, 2.1]
   output: 1
*/
pub fn find_closest_to_pi(x: &[f64]) -> usize {
    // LLM_OUTPUT_HERE
}

fn correct_find_closest_to_pi(x: &[f64]) -> usize {
    if x.is_empty() {
        return 0;
    }
    let mut index = 0;
    let mut min = (x[0] - PI).abs();
    for i in 1..x.len() {
        let diff = (x[i] - PI).abs();
        if diff < min {
            min = diff;
            index = i;
        }
    }
    index
}

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
        let mut rng = rand::thread_rng();
        // fillRand(ctx->x, -10000.0, 10000.0);
        for val in self.x.iter_mut() {
            *val = rng.gen_range(-10000.0..10000.0);
        }
    }

    fn compute(&mut self) {
        let _idx = find_closest_to_pi(&self.x);
    }

    fn best(&mut self) {
        let _idx = correct_find_closest_to_pi(&self.x);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        let num_tries = 10;

        for _ in 0..num_tries {
            // set up input
            let mut input = vec![0.0; TEST_SIZE];
            // fillRand(input, 100.0, 1000.0);
            for val in input.iter_mut() {
                *val = rng.gen_range(100.0..1000.0);
            }
            // input[rand() % TEST_SIZE] = 10.0;
            let idx = rng.gen_range(0..TEST_SIZE);
            input[idx] = 10.0;

            // compute correct result
            let correct = correct_find_closest_to_pi(&input);

            // compute test result
            let test = find_closest_to_pi(&input);

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