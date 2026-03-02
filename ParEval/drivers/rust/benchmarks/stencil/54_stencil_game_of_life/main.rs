// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 2048;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Simulate one generation of Game of Life on `input`. Store the results in `output`.
   A cell is 1 if it is alive and 0 if it is dead.
   If a live cell has fewer than 2 live neighbors then it dies.
   If a live cell has 2 or 3 live neighbors then it lives on.
   If a live cell has more than 3 live neighbords then it dies.
   If a cell is dead and has exactly 3 live neighbors then it becomes alive.
   `input` and `output` are NxN grids stored in row-major.
   Use Rust Rayon to compute in parallel.
   Example:

   input:  [[0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0]]
   output: [[0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0]]
*/
pub fn gameOfLife(input: &[i32], output: &mut [i32], N: usize) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

#[inline(never)]
fn correct_game_of_life(input: &[i32], output: &mut [i32], N: usize) {
    for i in 0..N {
        for j in 0..N {
            let mut sum = 0;
            if i > 0 {
                sum += input[(i - 1) * N + j];
            }
            if i < N - 1 {
                sum += input[(i + 1) * N + j];
            }
            if j > 0 {
                sum += input[i * N + (j - 1)];
            }
            if j < N - 1 {
                sum += input[i * N + (j + 1)];
            }
            if i > 0 && j > 0 {
                sum += input[(i - 1) * N + (j - 1)];
            }
            if i > 0 && j < N - 1 {
                sum += input[(i - 1) * N + (j + 1)];
            }
            if i < N - 1 && j > 0 {
                sum += input[(i + 1) * N + (j - 1)];
            }
            if i < N - 1 && j < N - 1 {
                sum += input[(i + 1) * N + (j + 1)];
            }

            if input[i * N + j] == 1 {
                if sum < 2 {
                    output[i * N + j] = 0;
                } else if sum == 2 || sum == 3 {
                    output[i * N + j] = 1;
                } else {
                    output[i * N + j] = 0;
                }
            } else {
                if sum == 3 {
                    output[i * N + j] = 1;
                } else {
                    output[i * N + j] = 0;
                }
            }
        }
    }
}

fn fill_rand(slice: &mut [i32], min: i32, max: i32) {
    let mut rng = rand::thread_rng();
    for x in slice.iter_mut() {
        *x = rng.gen_range(min..max);
    }
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct GameOfLifeContext {
    input: Vec<i32>,
    output: Vec<i32>,
    n: usize,
}

impl ParEvalBenchmark for GameOfLifeContext {
    fn new() -> Self {
        let DRIVER_PROBLEM_SIZE = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            input: vec![0; DRIVER_PROBLEM_SIZE * DRIVER_PROBLEM_SIZE],
            output: vec![0; DRIVER_PROBLEM_SIZE * DRIVER_PROBLEM_SIZE],
            n: DRIVER_PROBLEM_SIZE,
        }
    }

    fn reset(&mut self) {
        fill_rand(&mut self.input, 0, 2);
        // std::fill(ctx->output.begin(), ctx->output.end(), 0);
        for x in self.output.iter_mut() {
            *x = 0;
        }
    }

    fn compute(&mut self) {
        gameOfLife(&self.input, &mut self.output, self.n);
    }

    fn best(&mut self) {
        correct_game_of_life(&self.input, &mut self.output, self.n);
    }

    fn validate(&mut self) -> bool {
        let n = TEST_SIZE;
        let mut input = vec![0; n * n];
        let mut correct = vec![0; n * n];
        let mut test = vec![0; n * n];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // set up input
            fill_rand(&mut input, 0, 2);
            // clear outputs
            for x in test.iter_mut() { *x = 0; }
            for x in correct.iter_mut() { *x = 0; }

            // compute correct result
            correct_game_of_life(&input, &mut correct, n);

            // compute test result
            gameOfLife(&input, &mut test, n);

            let mut is_correct = true;
            // Matches C++ validation loop: 1 to TEST_SIZE-1 (exclusive)
            for i in 1..(n - 1) {
                for j in 1..(n - 1) {
                    if test[i * n + j] != correct[i * n + j] {
                        is_correct = false;
                        break;
                    }
                }
                if !is_correct {
                    break;
                }
            }

            if !is_correct {
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
    run::<GameOfLifeContext>();
}