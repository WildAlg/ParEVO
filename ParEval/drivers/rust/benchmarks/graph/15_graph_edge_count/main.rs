// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024;
const TEST_SIZE: usize = 128;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Count the number of edges in the directed graph defined by the adjacency matrix A.
   A is an NxN adjacency matrix stored in row-major. A represents a directed graph.
   Use Rust Rayon to compute in parallel.
   Example:

	 input: [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 1, 0]]
   output: 6
*/
pub fn edge_count(A: &[i32], N: usize) -> usize {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_edge_count(A: &[i32], N: usize) -> usize {
    let mut count = 0;
    for i in 0..N {
        for j in 0..N {
            if A[i * N + j] == 1 {
                count += 1;
            }
        }
    }
    count
}

fn fill_rand_directed_graph(a: &mut [i32], n: usize) {
    let mut rng = rand::thread_rng();
    a.fill(0);
    for i in 0..n {
        for j in 0..n {
            if rng.gen::<u32>() % 2 == 0 {
                a[i * n + j] = 1;
            }
        }
    }
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct Context {
    a: Vec<i32>,
    n: usize,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            a: vec![0; driver_problem_size * driver_problem_size],
            n: driver_problem_size,
        }
    }

    fn reset(&mut self) {
        fill_rand_directed_graph(&mut self.a, self.n);
    }

    fn compute(&mut self) {
        let _ = edge_count(&self.a, self.n);
    }

    fn best(&mut self) {
        let _ = correct_edge_count(&self.a, self.n);
    }

    fn validate(&mut self) -> bool {
        let mut a = vec![0; TEST_SIZE * TEST_SIZE];
        
        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // set up input
            fill_rand_directed_graph(&mut a, TEST_SIZE);

            // compute correct result
            let correct = correct_edge_count(&a, TEST_SIZE);

            // compute test result
            let test = edge_count(&a, TEST_SIZE);

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
    run::<Context>();
}