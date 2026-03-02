// main.rs
use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024 * 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* Return the product of the vector x with every odd indexed element inverted.
   i.e. x_0 * 1/x_1 * x_2 * 1/x_3 * x_4 ...
   Use Rust Rayon to compute product in parallel.
   Example:

   input: [4, 2, 10, 4, 5]
   output: 25
*/
pub fn product_with_inverses(x: &[f64]) -> f64 {
    // LLM_OUTPUT_HERE
}

fn correct_product_with_inverses(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, &val)| if i % 2 != 0 { 1.0 / val } else { val })
        .product()
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
            *val = rng.gen_range(1.0..100.0);
        }
    }

    fn compute(&mut self) {
        let val = product_with_inverses(&self.x);
        std::hint::black_box(val);
    }

    fn best(&mut self) {
        let val = correct_product_with_inverses(&self.x);
        std::hint::black_box(val);
    }

    fn validate(&mut self) -> bool {
        const TEST_SIZE: usize = 1024;
        let mut x = vec![0.0; TEST_SIZE];
        let mut rng = rand::thread_rng();

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // set up input
            for val in x.iter_mut() {
                *val = rng.gen_range(1.0..100.0);
            }

            // compute correct result
            let correct = correct_product_with_inverses(&x);

            // compute test result
            let test = product_with_inverses(&x);

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