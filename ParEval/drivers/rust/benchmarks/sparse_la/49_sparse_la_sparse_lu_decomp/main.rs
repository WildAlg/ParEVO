// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024;
const SPARSE_LA_SPARSITY: f64 = 0.05;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

#[derive(Clone, Copy, Debug)]
pub struct COOElement {
    pub row: usize,
    pub column: usize,
    pub value: f64,
}

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/// The LLM solution should be pasted here.
pub fn lu_factorize(A: &[COOElement], L: &mut Vec<f64>, U: &mut Vec<f64>, N: usize) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_lu_factorize(A: &[COOElement], L: &mut [f64], U: &mut [f64], N: usize) {
    // Convert sparse A to dense matrix for baseline computation
    let mut full_a = vec![0.0; N * N];
    for elem in A {
        if elem.row < N && elem.column < N {
            // In case of duplicate entries (same row, col), the later one in the slice overwrites the previous one.
            // C++ processes the sorted vector sequentially, so last one wins.
            full_a[elem.row * N + elem.column] = elem.value;
        }
    }

    // Standard Dense LU Factorization (Doolittle Algorithm)
    // Assumes L and U are pre-allocated and zero-initialized in entries where appropriate.
    // L has 1s on diagonal (set at end of loop).
    for i in 0..N {
        for j in 0..N {
            if j >= i {
                // Compute U[i][j]
                let mut sum = 0.0;
                for k in 0..i {
                    sum += L[i * N + k] * U[k * N + j];
                }
                U[i * N + j] = full_a[i * N + j] - sum;
            }

            if i > j {
                // Compute L[i][j]
                let mut sum = 0.0;
                for k in 0..j {
                    sum += L[i * N + k] * U[k * N + j];
                }
                let u_jj = U[j * N + j];
                if u_jj != 0.0 {
                    L[i * N + j] = (full_a[i * N + j] - sum) / u_jj;
                } else {
                    L[i * N + j] = 0.0;
                }
            }
        }
        // Set diagonal of L to 1
        L[i * N + i] = 1.0;
    }
}

// Helper to compare floating point slices
fn fequal(a: &[f64], b: &[f64], epsilon: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= epsilon)
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct SparseLuContext {
    a: Vec<COOElement>,
    l: Vec<f64>,
    u: Vec<f64>,
    n: usize,
}

impl ParEvalBenchmark for SparseLuContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            a: Vec::new(),
            l: vec![0.0; driver_problem_size * driver_problem_size],
            u: vec![0.0; driver_problem_size * driver_problem_size],
            n: driver_problem_size,
        }
    }

    fn reset(&mut self) {
        let n = self.n;
        let n_vals = (SPARSE_LA_SPARSITY * (n as f64 * n as f64)) as usize;
        let mut rng = rand::thread_rng();

        self.a.clear();
        self.a.reserve(n_vals);

        // Generate random sparse entries
        for _ in 0..n_vals {
            self.a.push(COOElement {
                row: rng.gen_range(0..n),
                column: rng.gen_range(0..n),
                value: rng.gen_range(-10.0..10.0),
            });
        }

        // Sort by row, then column
        self.a.sort_by(|x, y| x.row.cmp(&y.row).then(x.column.cmp(&y.column)));

        // Reset output buffers to 0.0
        self.l.fill(0.0);
        self.u.fill(0.0);
    }

    fn compute(&mut self) {
        lu_factorize(&self.a, &mut self.l, &mut self.u, self.n);
    }

    fn best(&mut self) {
        correct_lu_factorize(&self.a, &mut self.l, &mut self.u, self.n);
    }

    fn validate(&mut self) -> bool {
        let test_size = 64;
        let n_vals = (SPARSE_LA_SPARSITY * (test_size as f64 * test_size as f64)) as usize;
        let mut rng = rand::thread_rng();

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // 1. Setup Input
            let mut a_test = Vec::with_capacity(n_vals);
            for _ in 0..n_vals {
                a_test.push(COOElement {
                    row: rng.gen_range(0..test_size),
                    column: rng.gen_range(0..test_size),
                    value: rng.gen_range(-10.0..10.0),
                });
            }
            a_test.sort_by(|x, y| x.row.cmp(&y.row).then(x.column.cmp(&y.column)));

            // 2. Prepare buffers
            // Note: Parallel version uses &mut Vec, sequential uses slice.
            let mut l_correct = vec![0.0; test_size * test_size];
            let mut u_correct = vec![0.0; test_size * test_size];
            let mut l_test = vec![0.0; test_size * test_size];
            let mut u_test = vec![0.0; test_size * test_size];

            // 3. Compute Correct
            correct_lu_factorize(&a_test, &mut l_correct, &mut u_correct, test_size);

            // 4. Compute Test
            lu_factorize(&a_test, &mut l_test, &mut u_test, test_size);

            // 5. Compare
            if !fequal(&l_correct, &l_test, 1e-3) || !fequal(&u_correct, &u_test, 1e-3) {
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
    run::<SparseLuContext>();
}