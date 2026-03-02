use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024 * 1024;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* Compute one iteration of a 3-point 1D jacobi stencil on `input`. Store the results in `output`.
   Each element of `input` will be averaged with its two neighbors and stored in the corresponding element of `output`.
   i.e. output[i] = (input[i-1]+input[i]+input[i+1])/3
   Replace with 0 when reading past the boundaries of `input`.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [9, -6, -1, 2, 3]
   output: [1, 2/3, -5/3, 4/3, 5/3]
*/
pub fn jacobi1D(input: &[f64], output: &mut [f64]) {
    // LLM_OUTPUT_HERE
}

fn correct_jacobi1d(input: &[f64], output: &mut [f64]) {
    let n = input.len();
    for i in 0..n {
        let mut sum = 0.0;
        if i > 0 {
            sum += input[i - 1];
        }
        if i < n - 1 {
            sum += input[i + 1];
        }
        sum += input[i];
        output[i] = sum / 3.0;
    }
}

fn fill_rand(slice: &mut [f64], min: f64, max: f64) {
    let mut rng = rand::thread_rng();
    for x in slice.iter_mut() {
        *x = rng.gen_range(min..max);
    }
}

struct Context {
    input: Vec<f64>,
    output: Vec<f64>,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let DRIVER_PROBLEM_SIZE = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            input: vec![0.0; DRIVER_PROBLEM_SIZE],
            output: vec![0.0; DRIVER_PROBLEM_SIZE],
        }
    }

    fn reset(&mut self) {
        fill_rand(&mut self.input, -100.0, 100.0);
        for x in self.output.iter_mut() {
            *x = 0.0;
        }
    }

    fn compute(&mut self) {
        jacobi1D(&self.input, &mut self.output);
    }

    fn best(&mut self) {
        correct_jacobi1d(&self.input, &mut self.output);
    }

    fn validate(&mut self) -> bool {
        let mut input = vec![0.0; TEST_SIZE];
        let mut correct = vec![0.0; TEST_SIZE];
        let mut test = vec![0.0; TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            fill_rand(&mut input, -100.0, 100.0);
            test.fill(0.0);
            correct.fill(0.0);

            correct_jacobi1d(&input, &mut correct);
            jacobi1D(&input, &mut test);

            // Validation skips the very first and last elements as per C++ driver logic
            for i in 1..TEST_SIZE - 1 {
                if (test[i] - correct[i]).abs() > 1e-4 {
                    return false;
                }
            }
        }
        true
    }
}

fn main() {
    run::<Context>();
}