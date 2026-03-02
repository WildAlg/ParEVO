use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1_000_000;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* Return the logical XOR reduction of the vector of bools x.
   Use Rust Rayon to reduce in parallel.
   Example:

   input: [false, false, false, true]
   output: true
*/
pub fn reduceLogicalXOR(x: &[bool]) -> bool {
    // LLM_OUTPUT_HERE
}

fn correct_reduce_logical_xor(x: &[bool]) -> bool {
    x.iter().fold(false, |acc, &val| acc ^ val)
}

struct Context {
    x: Vec<bool>,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Context {
            x: vec![false; driver_problem_size],
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        for val in self.x.iter_mut() {
            *val = rng.gen();
        }
    }

    fn compute(&mut self) {
        let _ = reduceLogicalXOR(&self.x);
    }

    fn best(&mut self) {
        let _ = correct_reduce_logical_xor(&self.x);
    }

    fn validate(&mut self) -> bool {
        const TEST_SIZE: usize = 1024;
        let mut x = vec![false; TEST_SIZE];
        let mut rng = rand::thread_rng();

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // Set up input
            for val in x.iter_mut() {
                *val = rng.gen();
            }

            // Compute correct result
            let correct = correct_reduce_logical_xor(&x);

            // Compute test result
            let test = reduceLogicalXOR(&x);

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