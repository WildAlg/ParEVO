// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rand::prelude::*;
use rayon::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1_000_000;
const SPARSE_LA_SPARSITY: f64 = 0.05; // 5% sparsity
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Element {
    pub index: usize,
    pub value: f64,
}

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Compute z = alpha*x+y where x and y are sparse vectors. Store the result in the dense vector z.
   Use Rust Rayon to compute in parallel.
   Example:
   
   input: x=[{5, 12}, {8, 3}, {12, -1}], y=[{3, 1}, {5, -2}, {7, 1}, {8, -3}], alpha=1
   output: z=[0, 0, 0, 1, 0, 10, 0, 1, 0, 0, 0, 0, -1]
*/
pub fn sparseAxpy(alpha: f64, x: &[Element], y: &[Element], z: &mut [f64]) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_sparse_axpy(alpha: f64, x: &[Element], y: &[Element], z: &mut [f64]) {
    let mut xi = 0;
    let mut yi = 0;

    // Both x and y are sorted by index
    while xi < x.len() && yi < y.len() {
        if x[xi].index < y[yi].index {
            z[x[xi].index] += alpha * x[xi].value;
            xi += 1;
        } else if x[xi].index > y[yi].index {
            z[y[yi].index] += y[yi].value;
            yi += 1;
        } else {
            z[x[xi].index] += alpha * x[xi].value + y[yi].value;
            xi += 1;
            yi += 1;
        }
    }

    while xi < x.len() {
        z[x[xi].index] += alpha * x[xi].value;
        xi += 1;
    }

    while yi < y.len() {
        z[y[yi].index] += y[yi].value;
        yi += 1;
    }
}

// Helper to fill a slice with random values
fn fill_rand_f64(slice: &mut [f64], min: f64, max: f64) {
    let mut rng = rand::thread_rng();
    for x in slice.iter_mut() {
        *x = rng.gen_range(min..max);
    }
}

// Helper to check floating point equality
fn fequal(a: &[f64], b: &[f64], epsilon: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= epsilon)
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct SparseAxpyContext {
    x: Vec<Element>,
    y: Vec<Element>,
    z: Vec<f64>,
    alpha: f64,
    n: usize,
}

impl ParEvalBenchmark for SparseAxpyContext {
    fn new() -> Self {
        let n = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        let n_vals = (n as f64 * SPARSE_LA_SPARSITY) as usize;
        
        Self {
            x: vec![Element { index: 0, value: 0.0 }; n_vals],
            y: vec![Element { index: 0, value: 0.0 }; n_vals],
            z: vec![0.0; n],
            alpha: 1.0,
            n,
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        
        // Random alpha (-1.0 to 1.0)
        self.alpha = rng.gen_range(-1.0..1.0);

        // Fill x
        for el in self.x.iter_mut() {
            el.index = rng.gen_range(0..self.n);
            el.value = rng.gen_range(-1.0..1.0);
        }

        // Fill y
        for el in self.y.iter_mut() {
            el.index = rng.gen_range(0..self.n);
            el.value = rng.gen_range(-1.0..1.0);
        }

        // Sort by index to maintain sparse vector invariant
        self.x.par_sort_by(|a, b| a.index.cmp(&b.index));
        self.y.par_sort_by(|a, b| a.index.cmp(&b.index));

        // Zero output vector
        for val in self.z.iter_mut() {
            *val = 0.0;
        }
    }

    fn compute(&mut self) {
        sparseAxpy(self.alpha, &self.x, &self.y, &mut self.z);
    }

    fn best(&mut self) {
        correct_sparse_axpy(self.alpha, &self.x, &self.y, &mut self.z);
    }

    fn validate(&mut self) -> bool {
        let n_vals = (TEST_SIZE as f64 * SPARSE_LA_SPARSITY) as usize;
        
        let mut x = vec![Element { index: 0, value: 0.0 }; n_vals];
        let mut y = vec![Element { index: 0, value: 0.0 }; n_vals];
        let mut correct = vec![0.0; TEST_SIZE];
        let mut test = vec![0.0; TEST_SIZE];
        let mut rng = rand::thread_rng();

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // Setup input
            let alpha = rng.gen_range(-1.0..1.0);

            for el in x.iter_mut() {
                el.index = rng.gen_range(0..TEST_SIZE);
                el.value = rng.gen_range(-1.0..1.0);
            }
            for el in y.iter_mut() {
                el.index = rng.gen_range(0..TEST_SIZE);
                el.value = rng.gen_range(-1.0..1.0);
            }

            // Sort
            x.sort_by(|a, b| a.index.cmp(&b.index));
            y.sort_by(|a, b| a.index.cmp(&b.index));

            // Reset outputs
            correct.fill(0.0);
            test.fill(0.0);

            // Compute correct result
            correct_sparse_axpy(alpha, &x, &y, &mut correct);

            // Compute test result
            sparseAxpy(alpha, &x, &y, &mut test);

            if !fequal(&correct, &test, 1e-4) {
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
    run::<SparseAxpyContext>();
}