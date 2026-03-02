// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024;
const TEST_SIZE: usize = 512;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Compute the highest node degree in the undirected graph. The graph is defined in the adjacency matrix A.
   A is an NxN adjacency matrix stored in row-major. A is an undirected graph.
   Use Rust Rayon to compute in parallel.
   Example:

	 input: [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 1, 0]]
   output: 3
*/
pub fn max_degree(a: &[i32], n: usize) -> usize {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_max_degree(a: &[i32], n: usize) -> usize {
    let mut max_deg = 0;
    for i in 0..n {
        let mut degree = 0;
        for j in 0..n {
            degree += a[i * n + j];
        }
        if degree > max_deg {
            max_deg = degree;
        }
    }
    max_deg as usize
}

// Helper to fill the graph randomly (undirected)
fn fill_random_undirected_graph(a: &mut [i32], n: usize) {
    // Reset to 0
    a.fill(0);

    let mut rng = rand::thread_rng();
    for i in 0..n {
        // Diagonal is already 0
        for j in (i + 1)..n {
            let val = rng.gen_range(0..2); // 0 or 1
            a[i * n + j] = val;
            a[j * n + i] = val;
        }
    }
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct GraphContext {
    a: Vec<i32>,
    n: usize,
}

impl ParEvalBenchmark for GraphContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            a: vec![0; driver_problem_size * driver_problem_size],
            n: driver_problem_size,
        }
    }

    fn reset(&mut self) {
        fill_random_undirected_graph(&mut self.a, self.n);
    }

    fn compute(&mut self) {
        let _ = max_degree(&self.a, self.n);
    }

    fn best(&mut self) {
        let _ = correct_max_degree(&self.a, self.n);
    }

    fn validate(&mut self) -> bool {
        let mut a = vec![0; TEST_SIZE * TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // set up input
            fill_random_undirected_graph(&mut a, TEST_SIZE);

            // compute correct result
            let correct = correct_max_degree(&a, TEST_SIZE);

            // compute test result
            let test = max_degree(&a, TEST_SIZE);

            if correct != test {
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
    run::<GraphContext>();
}