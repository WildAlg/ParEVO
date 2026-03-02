use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* Compute one iteration of a 5-point 2D jacobi stencil on `input`. Store the results in `output`.
   Each element of `input` will be averaged with its four neighbors and stored in the corresponding element of `output`.
   i.e. output_{i,j} = (input_{i,j-1} + input_{i,j+1} + input_{i-1,j} + input_{i+1,j} + input_{i,j})/5
   Replace with 0 when reading past the boundaries of `input`.
   `input` and `output` are NxN grids stored in row-major.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [[3, 4, 1], [0, 1, 7], [5, 3, 2]]
   output: [[1.4, 1.8, 2.4],[1.8, 3, 2.2], [1.6, 2.2, 2.4]]
*/
pub fn jacobi2D(input: &[f64], output: &mut [f64], N: usize) {
    // LLM_OUTPUT_HERE
}

fn correct_jacobi_2d(input: &[f64], output: &mut [f64], N: usize) {
    for i in 0..N {
        for j in 0..N {
            let mut sum = 0.0;
            // Up
            if i > 0 {
                sum += input[(i - 1) * N + j];
            }
            // Down
            if i < N - 1 {
                sum += input[(i + 1) * N + j];
            }
            // Left
            if j > 0 {
                sum += input[i * N + (j - 1)];
            }
            // Right
            if j < N - 1 {
                sum += input[i * N + (j + 1)];
            }
            // Center
            sum += input[i * N + j];
            
            output[i * N + j] = sum / 5.0;
        }
    }
}

struct Context {
    input: Vec<f64>,
    output: Vec<f64>,
    n: usize,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let DRIVER_PROBLEM_SIZE = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Context {
            input: vec![0.0; DRIVER_PROBLEM_SIZE * DRIVER_PROBLEM_SIZE],
            output: vec![0.0; DRIVER_PROBLEM_SIZE * DRIVER_PROBLEM_SIZE],
            n: DRIVER_PROBLEM_SIZE,
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        for x in self.input.iter_mut() {
            *x = rng.gen_range(-100.0..100.0);
        }
        for x in self.output.iter_mut() {
            *x = 0.0;
        }
    }

    fn compute(&mut self) {
        jacobi2D(&self.input, &mut self.output, self.n);
    }

    fn best(&mut self) {
        correct_jacobi_2d(&self.input, &mut self.output, self.n);
    }

    fn validate(&mut self) -> bool {
        let n = TEST_SIZE;
        let mut input = vec![0.0; n * n];
        let mut correct = vec![0.0; n * n];
        let mut test = vec![0.0; n * n];
        
        let mut rng = rand::thread_rng();

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // Set up input
            for x in input.iter_mut() {
                *x = rng.gen_range(-100.0..100.0);
            }
            // Clear outputs
            correct.fill(0.0);
            test.fill(0.0);

            // Compute correct result
            correct_jacobi_2d(&input, &mut correct, n);

            // Compute test result
            jacobi2D(&input, &mut test, n);

            // Validate logic:
            // C++ validates (1..N-1) for both i and j.
            for i in 1..(n - 1) {
                for j in 1..(n - 1) {
                    let idx = i * n + j;
                    if (test[idx] - correct[idx]).abs() > 1e-6 {
                        return false;
                    }
                }
            }
        }
        true
    }
}

fn main() {
    run::<Context>();
}