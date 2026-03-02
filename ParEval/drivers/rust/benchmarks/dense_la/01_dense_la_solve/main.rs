// main.rs
use pareval_runner::{ParEvalBenchmark, run};
use rand::prelude::*;
use rayon::prelude::*;

// Constants
// const DRIVER_PROBLEM_SIZE: usize = 1024;
const TEST_SIZE: usize = 512;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/// The LLM solution should be pasted here.
/// Task: Solve the linear system Ax = b for x.
/// A is Row-Major (NxN). x and b have N elements.
/// Input: A (read-only), b (read-only)
/// Output: Write result into x
#[allow(non_snake_case)]
pub fn solve_linear_system(A: &[f64], b: &[f64], x: &mut [f64], N: usize) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline & Helpers
// ============================================================================

/// Helper: Creates a random system Ax=b by generating A and a known x,
/// then computing b = A*x. Finally, it resets x to 0 so the solver has to find it.
fn create_random_linear_system(a: &mut [f64], b: &mut [f64], x: &mut [f64], n: usize) {
    let mut rng = rand::thread_rng();

    // 1. Fill A with random values
    for val in a.iter_mut() {
        *val = rng.gen_range(-10.0..10.0);
    }

    // 2. Generate a "true" X temporarily to calculate B
    // We use the `x` buffer temporarily for the true values
    let mut x_true = vec![0.0; n];
    for val in x_true.iter_mut() {
        *val = rng.gen_range(-10.0..10.0);
    }

    // 3. Compute b = A * x_true
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += a[i * n + j] * x_true[j];
        }
        b[i] = sum;
    }

    // 4. Reset x to 0.0 (the solver must compute the values)
    x.fill(0.0);
}

/// Baseline: Standard Gaussian Elimination with Partial Pivoting
fn correct_solve_linear_system(A: &[f64], b: &[f64], x: &mut [f64], N: usize) {
    // Since input A is immutable, we must make a working copy
    let mut mat = A.to_vec();
    // We augment the matrix with vector b essentially
    let mut rhs = b.to_vec();

    // Forward Elimination
    for k in 0..N {
        // 1. Partial Pivoting: Find max in column k
        let mut max_row = k;
        let mut max_val = mat[k * N + k].abs();

        for i in (k + 1)..N {
            let val = mat[i * N + k].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        // Swap rows if needed
        if max_row != k {
            for j in k..N {
                mat.swap(k * N + j, max_row * N + j);
            }
            rhs.swap(k, max_row);
        }

        let pivot = mat[k * N + k];
        // Singular check (basic)
        if pivot.abs() < 1e-12 {
            continue; 
        }

        // Eliminate rows below
        for i in (k + 1)..N {
            let factor = mat[i * N + k] / pivot;
            for j in (k + 1)..N {
                mat[i * N + j] -= factor * mat[k * N + j];
            }
            rhs[i] -= factor * rhs[k];
        }
    }

    // Back Substitution
    for i in (0..N).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..N {
            sum -= mat[i * N + j] * x[j];
        }
        x[i] = sum / mat[i * N + i];
    }
}

fn fequal(a: &[f64], b: &[f64], tolerance: f64) -> bool {
    if a.len() != b.len() { return false; }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tolerance)
}

fn has_nan(a: &[f64]) -> bool {
    a.iter().any(|x| x.is_nan())
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct LinearSolveContext {
    a: Vec<f64>,
    b: Vec<f64>,
    x: Vec<f64>,
    n: usize,
}

impl ParEvalBenchmark for LinearSolveContext {
    fn new() -> Self {
        let n = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            a: vec![0.0; n * n],
            b: vec![0.0; n],
            x: vec![0.0; n],
            n,
        }
    }

    fn reset(&mut self) {
        create_random_linear_system(&mut self.a, &mut self.b, &mut self.x, self.n);
    }

    fn compute(&mut self) {
        // Execute the parallel solution
        solve_linear_system(&self.a, &self.b, &mut self.x, self.n);
    }

    fn best(&mut self) {
        // Execute the sequential baseline
        correct_solve_linear_system(&self.a, &self.b, &mut self.x, self.n);
    }

    fn validate(&mut self) -> bool {
        let n = TEST_SIZE;
        
        // Validation buffers
        let mut a_input = vec![0.0; n * n];
        let mut b_input = vec![0.0; n];
        let mut x_correct = vec![0.0; n];
        let mut x_test = vec![0.0; n];

        for attempt in 0..MAX_VALIDATION_ATTEMPTS {
            // 1. Setup Input (A, b constructed from a known x)
            // Note: We use x_correct as a temp buffer inside create_random... 
            // but create_random... clears it to 0.0 at the end.
            create_random_linear_system(&mut a_input, &mut b_input, &mut x_correct, n);
            
            // 2. Compute Correct
            correct_solve_linear_system(&a_input, &b_input, &mut x_correct, n);

            // 3. Compute Test (reset output buffer first)
            x_test.fill(0.0);
            solve_linear_system(&a_input, &b_input, &mut x_test, n);

            // 4. Compare
            if has_nan(&x_test) || !fequal(&x_correct, &x_test, 1e-4) {
                eprintln!("Validation failed on attempt {}", attempt + 1);
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
    run::<LinearSolveContext>();
}