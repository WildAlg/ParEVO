use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1_000_000;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* Compute the reverse prefix sum of the vector x into output.
   Use Rust Rayon to compute in parallel.
   Examples:
   
   input: [1, 7, 4, 6, 6, 2]
   output: [2, 8, 14, 18, 25, 26]

   input: [3, 3, 7, 1, -2]
   output: [-2, -1, 6, 9, 12]
*/
pub fn reversePrefixSum(x: &[i32], output: &mut [i32]) {
    // LLM_OUTPUT_HERE
}

fn correct_reverse_prefix_sum(x: &[i32], output: &mut [i32]) {
    let mut sum = 0;
    // Iterate x in reverse order, accumulate sum, and store in output sequentially
    for (i, val) in x.iter().rev().enumerate() {
        sum += val;
        output[i] = sum;
    }
}

fn fill_rand(slice: &mut [i32], min: i32, max: i32) {
    let mut rng = rand::thread_rng();
    for x in slice.iter_mut() {
        *x = rng.gen_range(min..max);
    }
}

struct Context {
    x: Vec<i32>,
    output: Vec<i32>,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            x: vec![0; driver_problem_size],
            output: vec![0; driver_problem_size],
        }
    }

    fn reset(&mut self) {
        fill_rand(&mut self.x, -100, 100);
    }

    fn compute(&mut self) {
        reversePrefixSum(&self.x, &mut self.output);
    }

    fn best(&mut self) {
        correct_reverse_prefix_sum(&self.x, &mut self.output);
    }

    fn validate(&mut self) -> bool {
        let mut x = vec![0; TEST_SIZE];
        let mut correct = vec![0; TEST_SIZE];
        let mut test = vec![0; TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            fill_rand(&mut x, -100, 100);
            
            correct_reverse_prefix_sum(&x, &mut correct);
            reversePrefixSum(&x, &mut test);
            
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