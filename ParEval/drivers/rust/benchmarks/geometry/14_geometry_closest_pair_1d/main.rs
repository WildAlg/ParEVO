use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 4096;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* Return the distance between the closest two elements in the vector x.
   Use Rust Rayon to compute in parallel.
   Example: 

   input: [7, 3, 9, 12, 31, 1]
   output: 2
*/
pub fn closest_pair(x: &[f64]) -> f64 {
    // LLM_OUTPUT_HERE
}

fn correct_closest_pair(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }

    let mut min_dist = f64::MAX;
    for i in 0..x.len() - 1 {
        for j in (i + 1)..x.len() {
            let dist = (x[i] - x[j]).abs();
            if dist < min_dist {
                min_dist = dist;
            }
        }
    }
    min_dist
}

struct Context {
    x: Vec<f64>,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Context {
            x: vec![0.0; driver_problem_size],
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        for val in self.x.iter_mut() {
            *val = rng.gen_range(-1000.0..1000.0);
        }
    }

    fn compute(&mut self) {
        let _ = closest_pair(&self.x);
    }

    fn best(&mut self) {
        let _ = correct_closest_pair(&self.x);
    }

    fn validate(&mut self) -> bool {
        const TEST_SIZE: usize = 1024;
        let mut rng = rand::thread_rng();
        let mut x = vec![0.0; TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // set up input
            for val in x.iter_mut() {
                *val = rng.gen_range(-1000.0..1000.0);
            }

            // compute correct result
            let correct = correct_closest_pair(&x);

            // compute test result
            let test = closest_pair(&x);

            // validate
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