// main.rs

use pareval_runner::{ParEvalBenchmark, run}; 
use rand::prelude::*;
use rayon::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Multiply the matrix A by the matrix B. Store the results in the matrix C.
   A is an MxK matrix, B is a KxN matrix, and C is a MxN matrix. The matrices are stored in row-major.
   Use Rust Rayon to compute in parallel.
   Example:

   input: A=[[1, -1, 2], [0, -2, 1]] B=[[4, 1], [-1, 0], [2, 2]]
   output: C=[[9, 5], [4, 2]]
*/
pub fn gemm(A: &[f64], B: &[f64], C: &mut [f64], M: usize, K: usize, N: usize) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_gemm(A: &[f64], B: &[f64], C: &mut [f64], M: usize, K: usize, N: usize) {
    for i in 0..M {
        for k in 0..K {
            for j in 0..N {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

// Helper: Fill slice with random values in [min, max)
fn fill_rand(slice: &mut [f64], min: f64, max: f64) {
    let mut rng = rand::thread_rng();
    for x in slice.iter_mut() {
        *x = rng.gen_range(min..max);
    }
}

// Helper: Fill slice with a constant value
fn fill_const(slice: &mut [f64], val: f64) {
    for x in slice.iter_mut() {
        *x = val;
    }
}

// Helper: Compare two slices with tolerance
fn fequal(a: &[f64], b: &[f64], epsilon: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= epsilon)
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct GemmContext {
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    m: usize,
    k: usize,
    n: usize,
}

impl ParEvalBenchmark for GemmContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        let m = driver_problem_size;
        let k = driver_problem_size / 4;
        let n = driver_problem_size / 2;

        Self {
            a: vec![0.0; m * k],
            b: vec![0.0; k * n],
            c: vec![0.0; m * n],
            m,
            k,
            n,
        }
    }

    fn reset(&mut self) {
        fill_rand(&mut self.a, -1.0, 1.0);
        fill_rand(&mut self.b, -1.0, 1.0);
        fill_const(&mut self.c, 0.0);
    }

    fn compute(&mut self) {
        gemm(&self.a, &self.b, &mut self.c, self.m, self.k, self.n);
    }

    fn best(&mut self) {
        correct_gemm(&self.a, &self.b, &mut self.c, self.m, self.k, self.n);
    }

    fn validate(&mut self) -> bool {
        // Use TEST_SIZE for validation dimensions (square matrices)
        let m = TEST_SIZE;
        let k = TEST_SIZE;
        let n = TEST_SIZE;

        let mut a = vec![0.0; m * k];
        let mut b = vec![0.0; k * n];
        let mut c_correct = vec![0.0; m * n];
        let mut c_test = vec![0.0; m * n];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // Setup input
            fill_rand(&mut a, -1.0, 1.0);
            fill_rand(&mut b, -1.0, 1.0);
            
            // Clear outputs
            fill_const(&mut c_correct, 0.0);
            fill_const(&mut c_test, 0.0);

            // Compute correct result
            correct_gemm(&a, &b, &mut c_correct, m, k, n);

            // Compute test result
            gemm(&a, &b, &mut c_test, m, k, n);

            // Compare
            if !fequal(&c_correct, &c_test, 1e-4) {
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
    run::<GemmContext>();
}