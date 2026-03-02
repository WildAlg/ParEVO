use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1 << 20; // 1048576
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* In the vector x negate the odd values and divide the even values by 2.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [16, 11, 12, 14, 1, 0, 5]
   output: [8, -11, 6, 7, -1, 0, -5]
*/
pub fn negate_odds_and_halve_evens(x: &mut [i32]) {
    // LLM_OUTPUT_HERE
}

fn correct_negate_odds_and_halve_evens(x: &mut [i32]) {
    for val in x.iter_mut() {
        if *val % 2 == 0 {
            *val /= 2;
        } else {
            *val = -*val;
        }
    }
}

fn fill_rand(slice: &mut [i32], min: i32, max: i32) {
    let mut rng = rand::thread_rng();
    for x in slice.iter_mut() {
        *x = rng.gen_range(min..=max);
    }
}

struct Context {
    x: Vec<i32>,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let DRIVER_PROBLEM_SIZE = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Context {
            x: vec![0; DRIVER_PROBLEM_SIZE],
        }
    }

    fn reset(&mut self) {
        fill_rand(&mut self.x, 1, 100);
    }

    fn compute(&mut self) {
        negate_odds_and_halve_evens(&mut self.x);
    }

    fn best(&mut self) {
        correct_negate_odds_and_halve_evens(&mut self.x);
    }

    fn validate(&mut self) -> bool {
        const TEST_SIZE: usize = 1024;
        let mut input = vec![0; TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            fill_rand(&mut input, 1, 100);

            let mut correct_result = input.clone();
            correct_negate_odds_and_halve_evens(&mut correct_result);

            let mut test_result = input.clone();
            negate_odds_and_halve_evens(&mut test_result);

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