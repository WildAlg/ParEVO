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

/* Return the number of vertices in the largest component of the undirected graph defined by the adjacency matrix A.
   A is an NxN adjacency matrix stored in row-major. A is an undirected graph.
   Use Rust Rayon to compute in parallel.
   Example:

	 input: [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
   output: 2
*/
pub fn largest_component(A: &[i32], N: usize) -> usize {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn dfs(adj: &[i32], node: usize, n: usize, visited: &mut [bool], count: &mut usize) {
    visited[node] = true;
    *count += 1;
    for i in 0..n {
        if adj[node * n + i] == 1 && !visited[i] {
            dfs(adj, i, n, visited, count);
        }
    }
}

fn correct_largest_component(A: &[i32], N: usize) -> usize {
    let mut visited = vec![false; N];
    let mut max_count = 0;
    for i in 0..N {
        if !visited[i] {
            let mut count = 0;
            dfs(A, i, N, &mut visited, &mut count);
            if count > max_count {
                max_count = count;
            }
        }
    }
    max_count
}

fn fill_random_undirected_graph(adj: &mut [i32], n: usize) {
    let mut rng = rand::thread_rng();
    adj.fill(0);
    for i in 0..n {
        adj[i * n + i] = 0;
        for j in (i + 1)..n {
            let val = rng.gen_range(0..2);
            adj[i * n + j] = val;
            adj[j * n + i] = val;
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
        let val = largest_component(&self.a, self.n);
        std::hint::black_box(val);
    }

    fn best(&mut self) {
        let val = correct_largest_component(&self.a, self.n);
        std::hint::black_box(val);
    }

    fn validate(&mut self) -> bool {
        let mut a_test = vec![0; TEST_SIZE * TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            fill_random_undirected_graph(&mut a_test, TEST_SIZE);

            let correct = correct_largest_component(&a_test, TEST_SIZE);
            let test = largest_component(&a_test, TEST_SIZE);

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