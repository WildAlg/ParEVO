// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rand::prelude::*;
use rayon::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024;
const TEST_SIZE: usize = 512;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* Set every cell's value to 1 if it has exactly one neighbor that's a 1. Otherwise set it to 0.
   Note that we only consider neighbors and not input_{i,j} when computing output_{i,j}.
   input and output are NxN grids of ints in row-major.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [[0, 1, 1, 0],
           [1, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 1, 0, 0]
   output: [[0, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [1, 0, 1, 0]]
*/
pub fn cells_xor(input: &[i32], output: &mut [i32], n: usize) {
    // LLM_OUTPUT_HERE
}

fn correct_cells_xor(input: &[i32], output: &mut [i32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut count = 0;
            // Check Up
            if i > 0 && input[(i - 1) * n + j] == 1 {
                count += 1;
            }
            // Check Down
            if i < n - 1 && input[(i + 1) * n + j] == 1 {
                count += 1;
            }
            // Check Left
            if j > 0 && input[i * n + j - 1] == 1 {
                count += 1;
            }
            // Check Right
            if j < n - 1 && input[i * n + j + 1] == 1 {
                count += 1;
            }

            output[i * n + j] = if count == 1 { 1 } else { 0 };
        }
    }
}

struct Context {
    input: Vec<i32>,
    output: Vec<i32>,
    n: usize,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            input: vec![0; driver_problem_size * driver_problem_size],
            output: vec![0; driver_problem_size * driver_problem_size],
            n: driver_problem_size,
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        // fillRand(ctx->input, 0, 2) in C++ usually implies values [0, 2] inclusive for uniform_int_distribution
        // or [0, 2) for modulus based. Given the problem is binary (XOR), we generate 0 or 1.
        for x in self.input.iter_mut() {
            *x = rng.gen_range(0..2);
        }
        self.output.fill(0);
    }

    fn compute(&mut self) {
        cells_xor(&self.input, &mut self.output, self.n);
    }

    fn best(&mut self) {
        correct_cells_xor(&self.input, &mut self.output, self.n);
    }

    fn validate(&mut self) -> bool {
        let n = TEST_SIZE;
        let mut input = vec![0; n * n];
        let mut correct = vec![0; n * n];
        let mut test = vec![0; n * n];
        let mut rng = rand::thread_rng();

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // Set up input
            for x in input.iter_mut() {
                *x = rng.gen_range(0..2);
            }
            correct.fill(0);
            test.fill(0);

            // Compute correct result
            correct_cells_xor(&input, &mut correct, n);

            // Compute test result
            cells_xor(&input, &mut test, n);

            // Compare results. 
            // Note: C++ validation loop runs from 1 to TEST_SIZE - 1 exclusive,
            // effectively checking indices 1 to n-2 (skipping borders).
            for i in 1..(n - 1) {
                for j in 1..(n - 1) {
                    if test[i * n + j] != correct[i * n + j] {
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