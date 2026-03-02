use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Multiply the matrix A by the vector x. Store the results in the vector y.
   A is an MxN matrix stored in row-major, x has N elements, and y has M elements.
   Use Rust Rayon to compute in parallel.
   Example:

   input: A=[[1, -1, 2], [0, -3, 1]] x=[2, 1, 0]
   output: y=[1, -3]
*/
pub fn gemv(A: &[f64], x: &[f64], y: &mut [f64], M: usize, N: usize) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_gemv(A: &[f64], x: &[f64], y: &mut [f64], M: usize, N: usize) {
    // A is MxN, x is N, y is M
    for i in 0..M {
        let mut val = 0.0;
        for j in 0..N {
            val += A[i * N + j] * x[j];
        }
        y[i] = val;
    }
}

// Helper to fill a slice with random values
fn fill_rand(slice: &mut [f64], min: f64, max: f64) {
    let mut rng = rand::thread_rng();
    for x in slice.iter_mut() {
        *x = rng.gen_range(min..max);
    }
}

// Helper to compare two slices
fn fequal(a: &[f64], b: &[f64], epsilon: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(val_a, val_b)| (val_a - val_b).abs() <= epsilon)
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct GemvContext {
    a: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
    m: usize,
    n: usize,
}

impl ParEvalBenchmark for GemvContext {
    fn new() -> Self {
        // init logic from C++:
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        let m = driver_problem_size / 2;
        let n = driver_problem_size;
        
        Self {
            a: vec![0.0; m * n],
            x: vec![0.0; n],
            y: vec![0.0; m],
            m,
            n,
        }
    }

    fn reset(&mut self) {
        fill_rand(&mut self.a, -10.0, 10.0);
        fill_rand(&mut self.x, -10.0, 10.0);
        // y is output, so strictly speaking doesn't need randomization, 
        // but resetting it or leaving it is fine as it gets overwritten.
    }

    fn compute(&mut self) {
        gemv(&self.a, &self.x, &mut self.y, self.m, self.n);
    }

    fn best(&mut self) {
        correct_gemv(&self.a, &self.x, &mut self.y, self.m, self.n);
    }

    fn validate(&mut self) -> bool {
        // C++ validate logic:
        // TEST_SIZE = 1024
        // Matrix is TEST_SIZE * TEST_SIZE
        const TEST_SIZE: usize = 1024;
        
        let mut a = vec![0.0; TEST_SIZE * TEST_SIZE];
        let mut x = vec![0.0; TEST_SIZE];
        let mut correct = vec![0.0; TEST_SIZE];
        let mut test = vec![0.0; TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            fill_rand(&mut a, -10.0, 10.0);
            fill_rand(&mut x, -10.0, 10.0);

            // Compute correct result
            correct_gemv(&a, &x, &mut correct, TEST_SIZE, TEST_SIZE);

            // Compute test result
            gemv(&a, &x, &mut test, TEST_SIZE, TEST_SIZE);

            // Compare
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
    run::<GemvContext>();
}