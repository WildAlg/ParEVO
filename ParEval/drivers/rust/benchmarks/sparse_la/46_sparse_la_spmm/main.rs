// main.rs
use pareval_runner::{ParEvalBenchmark, run};
use rand::prelude::*;
use rayon::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024;
const SPARSE_LA_SPARSITY: f64 = 0.05;
const TEST_SIZE: usize = 128;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct COOElement {
    pub row: usize,
    pub column: usize,
    pub value: f64,
}

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Compute the matrix multiplication Y=AX. A is a sparse MxK matrix in COO format.
   X is a sparse KxN matrix in COO format. Y is a dense MxN matrix in row-major.
   Use Rust Rayon to compute in parallel.
   Example:

   input: A=[{0,0,-2}, {0,1,1}, {1,1,-1}] X=[{0,1,2}, {1,0,-1}]
   output: Y=[{-1,-4}, {1,0}]
*/
pub fn spmm(A: &[COOElement], X: &[COOElement], Y: &mut [f64], M: usize, K: usize, N: usize) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline
// ============================================================================

fn correct_spmm(A: &[COOElement], X: &[COOElement], Y: &mut [f64], _M: usize, _K: usize, N: usize) {
    // Fill Y with zeros
    Y.fill(0.0);

    // Compute Y = A * X
    // Naive implementation O(|A| * |X|)
    for a in A {
        for x in X {
            if a.column == x.row {
                Y[a.row * N + x.column] += a.value * x.value;
            }
        }
    }
}

// ============================================================================
//  3. Helpers
// ============================================================================

fn sort_coo_elements(elements: &mut [COOElement]) {
    elements.sort_by(|a, b| {
        if a.row != b.row {
            a.row.cmp(&b.row)
        } else {
            a.column.cmp(&b.column)
        }
    });
}

fn generate_random_coo(n_vals: usize, max_row: usize, max_col: usize) -> Vec<COOElement> {
    let mut rng = rand::thread_rng();
    let mut elements = Vec::with_capacity(n_vals);
    for _ in 0..n_vals {
        elements.push(COOElement {
            row: rng.gen_range(0..max_row),
            column: rng.gen_range(0..max_col),
            value: rng.gen_range(-1.0..1.0),
        });
    }
    elements
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

struct SpmmContext {
    a: Vec<COOElement>,
    x: Vec<COOElement>,
    y: Vec<f64>,
    m: usize,
    k: usize,
    n: usize,
}

impl ParEvalBenchmark for SpmmContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        let m = driver_problem_size;
        let k = driver_problem_size / 4;
        let n = driver_problem_size / 2;

        Self {
            a: Vec::new(),
            x: Vec::new(),
            y: vec![0.0; m * n],
            m,
            k,
            n,
        }
    }

    fn reset(&mut self) {
        let n_vals_a = (self.m as f64 * self.k as f64 * SPARSE_LA_SPARSITY) as usize;
        let n_vals_x = (self.k as f64 * self.n as f64 * SPARSE_LA_SPARSITY) as usize;

        self.a = generate_random_coo(n_vals_a, self.m, self.k);
        sort_coo_elements(&mut self.a);

        self.x = generate_random_coo(n_vals_x, self.k, self.n);
        sort_coo_elements(&mut self.x);

        self.y.fill(0.0);
    }

    fn compute(&mut self) {
        spmm(&self.a, &self.x, &mut self.y, self.m, self.k, self.n);
    }

    fn best(&mut self) {
        correct_spmm(&self.a, &self.x, &mut self.y, self.m, self.k, self.n);
    }

    fn validate(&mut self) -> bool {
        let m = TEST_SIZE;
        let k = TEST_SIZE;
        let n = TEST_SIZE;

        let n_vals_a = (m as f64 * k as f64 * SPARSE_LA_SPARSITY) as usize;
        let n_vals_x = (k as f64 * n as f64 * SPARSE_LA_SPARSITY) as usize;

        let mut correct_y = vec![0.0; m * n];
        let mut test_y = vec![0.0; m * n];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            let mut a = generate_random_coo(n_vals_a, m, k);
            sort_coo_elements(&mut a);

            let mut x = generate_random_coo(n_vals_x, k, n);
            sort_coo_elements(&mut x);

            // Compute correct result
            correct_spmm(&a, &x, &mut correct_y, m, k, n);

            // Compute test result
            test_y.fill(0.0);
            spmm(&a, &x, &mut test_y, m, k, n);

            if !fequal(&correct_y, &test_y, 1e-4) {
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
    run::<SpmmContext>();
}