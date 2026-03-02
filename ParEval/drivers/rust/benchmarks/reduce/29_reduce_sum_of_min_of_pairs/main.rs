use pareval_runner::{ParEvalBenchmark, run};
use rand::prelude::*;
use rayon::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024 * 1024;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* Return the sum of the minimum value at each index of vectors x and y for all indices.
   i.e. sum = min(x_0, y_0) + min(x_1, y_1) + min(x_2, y_2) + ...
   Use Rust Rayon to sum in parallel.
   Example:

   input: x=[3, 4, 0, 2, 3], y=[2, 5, 3, 1, 7]
   output: 10
*/
pub fn sumOfMinimumElements(x: &[f64], y: &[f64]) -> f64 {
    // LLM_OUTPUT_HERE
}

fn correct_sum_of_minimum_elements(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| a.min(b))
        .sum()
}

fn fill_rand(slice: &mut [f64], min: f64, max: f64) {
    let mut rng = rand::thread_rng();
    for x in slice.iter_mut() {
        *x = rng.gen_range(min..max);
    }
}

struct Context {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            x: vec![0.0; driver_problem_size],
            y: vec![0.0; driver_problem_size],
        }
    }

    fn reset(&mut self) {
        fill_rand(&mut self.x, 0.0, 100.0);
        fill_rand(&mut self.y, 0.0, 100.0);
    }

    fn compute(&mut self) {
        let val = sumOfMinimumElements(&self.x, &self.y);
        std::hint::black_box(val);
    }

    fn best(&mut self) {
        let val = correct_sum_of_minimum_elements(&self.x, &self.y);
        std::hint::black_box(val);
    }

    fn validate(&mut self) -> bool {
        let mut x = vec![0.0; TEST_SIZE];
        let mut y = vec![0.0; TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            fill_rand(&mut x, 0.0, 100.0);
            fill_rand(&mut y, 0.0, 100.0);

            let correct = correct_sum_of_minimum_elements(&x, &y);
            let test = sumOfMinimumElements(&x, &y);

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