// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;
use num_complex::Complex;

// const DRIVER_PROBLEM_SIZE: usize = 1_000_000;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Sort the vector x of complex numbers by their magnitude in ascending order.
   Use Rust Rayon to sort in parallel.
   Example:
   
   input: [3.0-1.0i, 4.5+2.1i, 0.0-1.0i, 1.0-0.0i, 0.5+0.5i]
   output: [0.5+0.5i, 0.0-1.0i, 1.0-0.0i, 3.0-1.0i, 4.5+2.1i]
*/
pub fn sort_complex_by_magnitude(x: &mut Vec<num_complex::Complex<f64>>) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_sort_complex_by_magnitude(x: &mut Vec<Complex<f64>>) {
    // C++ std::sort is unstable. We use sort_unstable_by to match behavior.
    // We compare magnitudes. Since sqrt is monotonic, comparing squared magnitudes is sufficient and faster.
    x.sort_unstable_by(|a, b| {
        let mag_sq_a = a.norm_sqr();
        let mag_sq_b = b.norm_sqr();
        // Handle floating point comparison
        mag_sq_a.partial_cmp(&mag_sq_b).unwrap_or(std::cmp::Ordering::Equal)
    });
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct SortComplexContext {
    x: Vec<Complex<f64>>,
}

impl ParEvalBenchmark for SortComplexContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            x: vec![Complex::new(0.0, 0.0); driver_problem_size],
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        // Fill with random complex numbers with real/imag parts in [-100.0, 100.0]
        for val in self.x.iter_mut() {
            val.re = rng.gen_range(-100.0..100.0);
            val.im = rng.gen_range(-100.0..100.0);
        }
    }

    fn compute(&mut self) {
        sort_complex_by_magnitude(&mut self.x);
    }

    fn best(&mut self) {
        correct_sort_complex_by_magnitude(&mut self.x);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // 1. Setup Input
            let mut correct = vec![Complex::new(0.0, 0.0); TEST_SIZE];
            for val in correct.iter_mut() {
                val.re = rng.gen_range(-100.0..100.0);
                val.im = rng.gen_range(-100.0..100.0);
            }
            let mut test = correct.clone();

            // 2. Compute Correct Result
            correct_sort_complex_by_magnitude(&mut correct);

            // 3. Compute Test Result
            sort_complex_by_magnitude(&mut test);

            // 4. Validate
            if correct.len() != test.len() {
                return false;
            }

            // Check element-wise difference.
            // C++: std::abs(correct[i] - test[i]) > 1e-6
            // Rust: (c-t).norm_sqr() > (1e-6)^2 = 1e-12
            for (c, t) in correct.iter().zip(test.iter()) {
                if (c - t).norm_sqr() > 1e-12 {
                    return false;
                }
            }
        }
        true
    }
}

// ============================================================================
//  4. Entry Point
// ============================================================================

fn main() {
    run::<SortComplexContext>();
}