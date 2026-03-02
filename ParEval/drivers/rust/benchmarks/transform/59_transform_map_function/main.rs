use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024 * 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

pub fn is_power_of_two(x: i32) -> bool {
    x > 0 && (x & (x - 1)) == 0
}

/* Apply the isPowerOfTwo function to every value in x and store the results in mask.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [8, 0, 9, 7, 15, 64, 3]
   output: [true, false, false, false, false, true, false]
*/
pub fn mapPowersOfTwo(x: &[i32], mask: &mut Vec<bool>) {
    // LLM_OUTPUT_HERE
}

fn correct_map_powers_of_two(x: &[i32], mask: &mut Vec<bool>) {
    mask.resize(x.len(), false);
    for (i, &val) in x.iter().enumerate() {
        mask[i] = is_power_of_two(val);
    }
}

struct Context {
    x: Vec<i32>,
    mask: Vec<bool>,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let DRIVER_PROBLEM_SIZE = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Context {
            x: vec![0; DRIVER_PROBLEM_SIZE],
            mask: vec![false; DRIVER_PROBLEM_SIZE],
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        // fillRand(ctx->x, 1, 1025) roughly implies range [1, 1025)
        for val in self.x.iter_mut() {
            *val = rng.gen_range(1..1025);
        }
    }

    fn compute(&mut self) {
        mapPowersOfTwo(&self.x, &mut self.mask);
    }

    fn best(&mut self) {
        correct_map_powers_of_two(&self.x, &mut self.mask);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            let n = 1024;
            let mut input = vec![0; n];
            for val in input.iter_mut() {
                *val = rng.gen_range(1..1025);
            }

            let mut correct_result = vec![false; n];
            correct_map_powers_of_two(&input, &mut correct_result);

            let mut test_result = vec![false; n];
            mapPowersOfTwo(&input, &mut test_result);

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