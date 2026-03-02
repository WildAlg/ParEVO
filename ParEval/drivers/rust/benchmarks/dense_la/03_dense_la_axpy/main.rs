use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1_000_000;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Compute z = alpha*x+y where x and y are vectors. Store the result in z.
   Use Rust Rayon to compute in parallel.
   Example:
   
   input: x=[1, -5, 2, 9] y=[0, 4, 1, -1] alpha=2
   output: z=[2, -6, 5, 17]
*/
pub fn axpy(alpha: f64, x: &[f64], y: &[f64], z: &mut [f64]) {
    // LLM_OUTPUT_HERE
    z.par_iter_mut()
        .zip(x.par_iter())
        .zip(y.par_iter())
        .for_each(|((z_val, x_val), y_val)| {
            *z_val = alpha * x_val + y_val;
        });
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_axpy(alpha: f64, x: &[f64], y: &[f64], z: &mut [f64]) {
    for i in 0..x.len() {
        z[i] = alpha * x[i] + y[i];
    }
}

fn fill_rand(slice: &mut [f64], min: f64, max: f64) {
    let mut rng = rand::thread_rng();
    for x in slice.iter_mut() {
        *x = rng.gen_range(min..max);
    }
}

fn fequal(a: &[f64], b: &[f64], epsilon: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= epsilon)
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct AxpyContext {
    alpha: f64,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
}

impl ParEvalBenchmark for AxpyContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            alpha: 2.0,
            x: vec![0.0; driver_problem_size],
            y: vec![0.0; driver_problem_size],
            z: vec![0.0; driver_problem_size],
        }
    }

    fn reset(&mut self) {
        fill_rand(&mut self.x, -1.0, 1.0);
        fill_rand(&mut self.y, -1.0, 1.0);
        // z does not need initialization as it is output
    }

    fn compute(&mut self) {
        axpy(self.alpha, &self.x, &self.y, &mut self.z);
    }

    fn best(&mut self) {
        correct_axpy(self.alpha, &self.x, &self.y, &mut self.z);
    }

    fn validate(&mut self) -> bool {
        let n = TEST_SIZE;
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut correct = vec![0.0; n];
        let mut test = vec![0.0; n];
        let alpha = 2.0;

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            fill_rand(&mut x, -1.0, 1.0);
            fill_rand(&mut y, -1.0, 1.0);

            // Compute correct result
            correct_axpy(alpha, &x, &y, &mut correct);

            // Compute test result
            axpy(alpha, &x, &y, &mut test);

            if !fequal(&correct, &test, 1e-6) {
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
    run::<AxpyContext>();
}