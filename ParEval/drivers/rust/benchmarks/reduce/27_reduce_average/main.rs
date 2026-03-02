use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1_000_000;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* Return the average of the vector x.
   Use Rust Rayon to compute in parallel.
   Examples:
		
	 input: [1, 8, 4, 5, 1]
   output: 3.8

   input: [2, 2, 2, 3]
   output: 2.25
*/
pub fn average(x: &[f64]) -> f64 {
    // LLM_OUTPUT_HERE
}

fn correct_average(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

fn fill_rand(slice: &mut [f64], min: f64, max: f64) {
    let mut rng = rand::thread_rng();
    for x in slice.iter_mut() {
        *x = rng.gen_range(min..max);
    }
}

struct Context {
    x: Vec<f64>,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            x: vec![0.0; driver_problem_size],
        }
    }

    fn reset(&mut self) {
        fill_rand(&mut self.x, 0.0, 100.0);
    }

    fn compute(&mut self) {
        let _val = average(&self.x);
    }

    fn best(&mut self) {
        let _val = correct_average(&self.x);
    }

    fn validate(&mut self) -> bool {
        let mut x = vec![0.0; TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            fill_rand(&mut x, 0.0, 100.0);

            let correct = correct_average(&x);
            let test = average(&x);

            if (correct - test).abs() > 1e-4 {
                return false;
            }
        }
        true
    }
}

fn main() {
    run::<Context>();
}