// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024 * 1024;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Return true if `val` is only in one of vectors x or y.
   Return false if it is in both or neither.
   Use Rust Rayon to search in parallel.
   Examples:

   input: x=[1,8,4,3,2], y=[3,4,4,1,1,7], val=7
   output: true

   input: x=[1,8,4,3,2], y=[3,4,4,1,1,7], val=1
   output: false
*/
pub fn xorContains(x: &[i32], y: &[i32], val: i32) -> bool {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_xor_contains(x: &[i32], y: &[i32], val: i32) -> bool {
    let found_in_x = x.contains(&val);
    let found_in_y = y.contains(&val);
    found_in_x ^ found_in_y
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct Context {
    x: Vec<i32>,
    y: Vec<i32>,
    val: i32,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Context {
            x: vec![0; driver_problem_size],
            y: vec![0; driver_problem_size],
            val: 0,
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        // fillRand(ctx->x, -10000, 10000);
        for v in self.x.iter_mut() {
            *v = rng.gen_range(-10000..=10000);
        }
        // fillRand(ctx->y, -10000, 10000);
        for v in self.y.iter_mut() {
            *v = rng.gen_range(-10000..=10000);
        }
        // ctx->val = rand() % 1000;
        self.val = rng.gen_range(0..1000);
    }

    fn compute(&mut self) {
        let _ = xorContains(&self.x, &self.y, self.val);
    }

    fn best(&mut self) {
        let _ = correct_xor_contains(&self.x, &self.y, self.val);
    }

    fn validate(&mut self) -> bool {
        const TEST_SIZE: usize = 1024;
        const NUM_TRIES: usize = 10;
        let mut rng = rand::thread_rng();

        for i in 0..NUM_TRIES {
            let mut x = vec![0; TEST_SIZE];
            let mut y = vec![0; TEST_SIZE];

            // fillRand(x, -100, 100);
            for v in x.iter_mut() {
                *v = rng.gen_range(-100..=100);
            }
            // fillRand(y, -100, 100);
            for v in y.iter_mut() {
                *v = rng.gen_range(-100..=100);
            }

            // int val = rand() % 200 - 100;
            // Matches range [-100, 99] usually, or similar
            let mut val = rng.gen_range(0..200) - 100;

            // if (i == 1) { x[...] = val; y[...] = val; }
            if i == 1 {
                let idx_x = rng.gen_range(0..x.len());
                x[idx_x] = val;
                let idx_y = rng.gen_range(0..y.len());
                y[idx_y] = val;
            }

            let correct = correct_xor_contains(&x, &y, val);
            let test = xorContains(&x, &y, val);

            if correct != test {
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