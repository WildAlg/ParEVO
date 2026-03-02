use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* Return the largest sum of any contiguous subarray in the vector x.
   i.e. if x=[−2, 1, −3, 4, −1, 2, 1, −5, 4] then [4, −1, 2, 1] is the contiguous
   subarray with the largest sum of 6.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [−2, 1, −3, 4, −1, 2, 1, −5, 4]
   output: 6
*/
pub fn maximum_subarray(x: &[i32]) -> i32 {
    // LLM_OUTPUT_HERE
}

fn correct_maximum_subarray(x: &[i32]) -> i32 {
    let mut largest_sum = i32::MIN;
    for i in 0..x.len() {
        let mut curr_sum = 0;
        for j in i..x.len() {
            curr_sum += x[j];
            if curr_sum > largest_sum {
                largest_sum = curr_sum;
            }
        }
    }
    largest_sum
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
        // fillRand(ctx->x, -100, 100);
        for v in self.x.iter_mut() {
            *v = rng.gen_range(-100..=100);
        }
    }

    fn compute(&mut self) {
        let _ = maximum_subarray(&self.x);
    }

    fn best(&mut self) {
        let _ = correct_maximum_subarray(&self.x);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        let mut x = vec![0; TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // set up input
            for v in x.iter_mut() {
                *v = rng.gen_range(-100..=100);
            }

            // compute correct result
            let correct = correct_maximum_subarray(&x);

            // compute test result
            let test = maximum_subarray(&x);

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