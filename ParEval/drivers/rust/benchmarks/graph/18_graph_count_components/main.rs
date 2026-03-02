// main.rs

use pareval_runner::{ParEvalBenchmark, run}; 

use rand::prelude::*;
use rayon::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024;
const TEST_SIZE: usize = 512;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Count the number of connected components in the undirected graph defined by the adjacency matrix A.
   A is an NxN adjacency matrix stored in row-major. A is an undirected graph.
	 Use Rust Rayon to compute in parallel.
   Example:

	 input: [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
   output: 2
*/
pub fn component_count(A: &[i32], N: usize) -> usize {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn dfs(a: &[i32], node: usize, n: usize, visited: &mut [bool]) {
    visited[node] = true;
    for i in 0..n {
        if a[node * n + i] == 1 && !visited[i] {
            dfs(a, i, n, visited);
        }
    }
}

fn correct_component_count(A: &[i32], N: usize) -> usize {
    let mut visited = vec![false; N];
    let mut count = 0;
    for i in 0..N {
        if !visited[i] {
            dfs(A, i, N, &mut visited);
            count += 1;
        }
    }
    count
}

fn fill_random_undirected_graph(a: &mut [i32], n: usize) {
    let mut rng = rand::thread_rng();
    a.fill(0);
    for i in 0..n {
        // a[i*n + i] = 0; // Already zeroed
        for j in (i + 1)..n {
            let val = rng.gen_range(0..2);
            a[i * n + j] = val;
            a[j * n + i] = val;
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
        Context {
            a: vec![0; driver_problem_size * driver_problem_size],
            n: driver_problem_size,
        }
    }

    fn reset(&mut self) {
        fill_random_undirected_graph(&mut self.a, self.n);
    }

    fn compute(&mut self) {
        let _cc = component_count(&self.a, self.n);
    }

    fn best(&mut self) {
        let _cc = correct_component_count(&self.a, self.n);
    }

    fn validate(&mut self) -> bool {
        let n = TEST_SIZE;
        let mut a = vec![0; n * n];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            fill_random_undirected_graph(&mut a, n);

            let correct = correct_component_count(&a, n);
            let test = component_count(&a, n);

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