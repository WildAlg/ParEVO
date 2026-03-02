use pareval_runner::{ParEvalBenchmark, run};
use rand::prelude::*;
use rayon::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1 << 20;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* Replace every element of the vector x with 1-1/x.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [2, 4, 1, 12, -2]
   output: [0.5, 0.75, 0, 0.91666666, 1.5]
*/
pub fn oneMinusInverse(x: &mut [f64]) {
    // LLM_OUTPUT_HERE
}

fn correct_one_minus_inverse(x: &mut [f64]) {
    for val in x.iter_mut() {
        *val = 1.0 - 1.0 / *val;
    }
}

fn fequal(a: &[f64], b: &[f64], epsilon: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(val_a, val_b)| (val_a - val_b).abs() <= epsilon)
}

struct Context {
    x: Vec<f64>,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let DRIVER_PROBLEM_SIZE = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Context {
            x: vec![0.0; DRIVER_PROBLEM_SIZE],
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        for val in self.x.iter_mut() {
            *val = rng.gen_range(-50.0..50.0);
        }
    }

    fn compute(&mut self) {
        oneMinusInverse(&mut self.x);
    }

    fn best(&mut self) {
        correct_one_minus_inverse(&mut self.x);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        let test_size = 1024;
        let mut input = vec![0.0; test_size];
        let mut correct = vec![0.0; test_size];
        let mut test = vec![0.0; test_size];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // Fill input with random values
            for val in input.iter_mut() {
                *val = rng.gen_range(-50.0..50.0);
            }

            // Compute correct result
            correct.copy_from_slice(&input);
            correct_one_minus_inverse(&mut correct);

            // Compute test result
            test.copy_from_slice(&input);
            oneMinusInverse(&mut test);

            // Compare
            if !fequal(&correct, &test, 1e-5) {
                return false;
            }
        }
        true
    }
}

fn main() {
    run::<Context>();
}