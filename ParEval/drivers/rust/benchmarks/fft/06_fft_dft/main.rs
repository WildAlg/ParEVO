// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;
use num_complex::Complex;
use std::f64::consts::PI;

// const DRIVER_PROBLEM_SIZE: usize = 2048;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* Compute the discrete fourier transform of x. Store the result in output.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [1, 4, 9, 16]
   output: [30+0i, -8-12i, -10-0i, -8+12i]
*/
pub fn dft(x: &[f64], output: &mut [num_complex::Complex<f64>]) {
    // LLM_OUTPUT_HERE
}

// Sequential Baseline
fn correct_dft(x: &[f64], output: &mut [Complex<f64>]) {
    let n = x.len();
    // Output is expected to be a slice of size N, pre-allocated by caller.
    for k in 0..n {
        let mut sum = Complex::new(0.0, 0.0);
        for idx in 0..n {
            let angle = 2.0 * PI * (idx as f64) * (k as f64) / (n as f64);
            // Euler's formula: e^(-i * angle) = cos(angle) - i*sin(angle)
            let c = Complex::new(angle.cos(), -angle.sin());
            sum += x[idx] * c;
        }
        output[k] = sum;
    }
}

struct DftContext {
    x: Vec<f64>,
    output: Vec<Complex<f64>>,
}

impl ParEvalBenchmark for DftContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            x: vec![0.0; driver_problem_size],
            output: vec![Complex::new(0.0, 0.0); driver_problem_size],
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        // fillRand(ctx->x, -1.0, 1.0);
        for val in self.x.iter_mut() {
            *val = rng.gen_range(-1.0..1.0);
        }
    }

    fn compute(&mut self) {
        dft(&self.x, &mut self.output);
    }

    fn best(&mut self) {
        correct_dft(&self.x, &mut self.output);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        let mut x = vec![0.0; TEST_SIZE];
        let mut correct = vec![Complex::new(0.0, 0.0); TEST_SIZE];
        let mut test = vec![Complex::new(0.0, 0.0); TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // Set up input
            for val in x.iter_mut() {
                *val = rng.gen_range(-1.0..1.0);
            }

            // Compute correct result
            correct_dft(&x, &mut correct);

            // Compute test result
            dft(&x, &mut test);

            // Validation check (tolerance 1e-4)
            for j in 0..TEST_SIZE {
                let diff_re = (correct[j].re - test[j].re).abs();
                let diff_im = (correct[j].im - test[j].im).abs();
                if diff_re > 1e-4 || diff_im > 1e-4 {
                    return false;
                }
            }
        }
        true
    }
}

fn main() {
    run::<DftContext>();
}