// main.rs
use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024;
const TEST_SIZE: usize = 128;
const MAX_VALIDATION_ATTEMPTS: usize = 5;
const SPARSE_LA_SPARSITY: f64 = 0.05;

#[derive(Clone, Copy, Debug)]
pub struct COOElement {
    pub row: usize,
    pub column: usize,
    pub value: f64,
}

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Compute y = alpha*A*x + beta*y where alpha and beta are scalars, x and y are vectors,
   and A is a sparse matrix stored in COO format.
   A has dimensions MxN, x has N values, and y has M values.
   Use Rust Rayon to parallelize.
   Example:

   input: alpha=0.5 beta=1.0 A=[{0,1,3}, {1,0,-1}] x=[-4, 2] y=[-1,1]
   output: y=[2, 3]
*/
pub fn spmv(alpha: f64, A: &[COOElement], x: &[f64], beta: f64, y: &mut [f64], M: usize, N: usize) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_spmv(alpha: f64, A: &[COOElement], x: &[f64], beta: f64, y: &mut [f64], M: usize, N: usize) {
    // First scale y by beta
    for element in y.iter_mut() {
        *element *= beta;
    }

    // Accumulate A*x into y
    for elem in A {
        if elem.row < M && elem.column < N {
            y[elem.row] += alpha * elem.value * x[elem.column];
        }
    }
}

// ============================================================================
//  3. Helpers
// ============================================================================

fn fill_rand(slice: &mut [f64], min: f64, max: f64) {
    let mut rng = rand::thread_rng();
    for x in slice.iter_mut() {
        *x = rng.gen_range(min..max);
    }
}

fn fill_rand_usize(slice: &mut [usize], min: usize, max: usize) {
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
//  4. Benchmark State Implementation
// ============================================================================

struct SpmvContext {
    a: Vec<COOElement>,
    x: Vec<f64>,
    y: Vec<f64>,
    alpha: f64,
    beta: f64,
    m: usize,
    n: usize,
}

impl ParEvalBenchmark for SpmvContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        SpmvContext {
            a: Vec::new(),
            x: Vec::new(),
            y: Vec::new(),
            alpha: 0.0,
            beta: 0.0,
            m: driver_problem_size,
            n: driver_problem_size,
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        self.m = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        self.n = self.m;

        self.alpha = rng.gen_range(-1.0..1.0);
        self.beta = rng.gen_range(-1.0..1.0);

        let n_vals = ((self.m * self.n) as f64 * SPARSE_LA_SPARSITY) as usize;

        let mut rows = vec![0usize; n_vals];
        let mut cols = vec![0usize; n_vals];
        let mut values = vec![0.0f64; n_vals];

        fill_rand_usize(&mut rows, 0, self.m);
        fill_rand_usize(&mut cols, 0, self.n);
        fill_rand(&mut values, -1.0, 1.0);

        self.x = vec![0.0; self.n];
        self.y = vec![0.0; self.m];
        fill_rand(&mut self.x, -1.0, 1.0);
        fill_rand(&mut self.y, -1.0, 1.0);

        self.a.clear();
        self.a.reserve(n_vals);
        for i in 0..n_vals {
            self.a.push(COOElement {
                row: rows[i],
                column: cols[i],
                value: values[i],
            });
        }

        // Sort A by row, then column
        self.a.sort_by(|a, b| {
            if a.row == b.row {
                a.column.cmp(&b.column)
            } else {
                a.row.cmp(&b.row)
            }
        });
    }

    fn compute(&mut self) {
        spmv(self.alpha, &self.a, &self.x, self.beta, &mut self.y, self.m, self.n);
    }

    fn best(&mut self) {
        correct_spmv(self.alpha, &self.a, &self.x, self.beta, &mut self.y, self.m, self.n);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        
        // Use TEST_SIZE for validation
        let m = TEST_SIZE;
        let n = TEST_SIZE;
        let n_vals = ((m * n) as f64 * SPARSE_LA_SPARSITY) as usize;

        let mut rows = vec![0usize; n_vals];
        let mut cols = vec![0usize; n_vals];
        let mut values = vec![0.0f64; n_vals];
        let mut x = vec![0.0; n];
        let mut y_correct = vec![0.0; m];
        let mut y_test = vec![0.0; m];
        let mut y_init = vec![0.0; m];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            let alpha = rng.gen_range(-1.0..1.0);
            let beta = rng.gen_range(-1.0..1.0);

            fill_rand_usize(&mut rows, 0, m);
            fill_rand_usize(&mut cols, 0, n);
            fill_rand(&mut values, -1.0, 1.0);
            fill_rand(&mut x, -1.0, 1.0);
            fill_rand(&mut y_init, -1.0, 1.0);

            // Set up A
            let mut a_vec = Vec::with_capacity(n_vals);
            for i in 0..n_vals {
                a_vec.push(COOElement {
                    row: rows[i],
                    column: cols[i],
                    value: values[i],
                });
            }
            a_vec.sort_by(|a, b| {
                if a.row == b.row {
                    a.column.cmp(&b.column)
                } else {
                    a.row.cmp(&b.row)
                }
            });

            // Initialize Y vectors
            y_correct.copy_from_slice(&y_init);
            y_test.copy_from_slice(&y_init);

            // Compute
            correct_spmv(alpha, &a_vec, &x, beta, &mut y_correct, m, n);
            spmv(alpha, &a_vec, &x, beta, &mut y_test, m, n);

            // Validate
            if !fequal(&y_correct, &y_test, 1e-4) {
                return false;
            }
        }

        true
    }
}

// ============================================================================
//  5. Entry Point
// ============================================================================

fn main() {
    run::<SpmvContext>();
}