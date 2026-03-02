// main.rs
use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;
use std::collections::VecDeque;

// const DRIVER_PROBLEM_SIZE: usize = 1024;
const TEST_SIZE: usize = 512;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Return the length of the shortest path from source to dest in the undirected graph defined by the adjacency matrix A.
   A is an NxN adjacency matrix stored in row-major. A is an undirected graph.
   Use Rust Rayon to compute in parallel.
   Example:

	 input: [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]], source=0, dest=3
   output: 2
*/
pub fn shortest_path_length(A: &[i32], N: usize, source: i32, dest: i32) -> i32 {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_shortest_path_length(A: &[i32], N: usize, source: i32, dest: i32) -> i32 {
    let mut visited = vec![false; N];
    let mut queue = VecDeque::new();

    visited[source as usize] = true;
    queue.push_back((source, 0));

    while let Some((current, path_length)) = queue.pop_front() {
        if current == dest {
            return path_length;
        }

        // Check all adjacent vertices
        // Since A is stored in row-major, the neighbors of `current` are in the row `current`.
        let row_offset = (current as usize) * N;
        for i in 0..N {
            if A[row_offset + i] != 0 && !visited[i] {
                visited[i] = true;
                queue.push_back((i as i32, path_length + 1));
            }
        }
    }

    i32::MAX
}

// ============================================================================
//  3. Helper Functions
// ============================================================================

fn random_connected_undirected_graph(A: &mut [i32], N: usize) {
    let mut rng = rand::thread_rng();
    
    // Clear matrix
    A.fill(0);

    // Create a path to ensure connectivity
    let mut nodes: Vec<usize> = (0..N).collect();
    nodes.shuffle(&mut rng);

    for i in 0..(N - 1) {
        let u = nodes[i];
        let v = nodes[i + 1];
        A[u * N + v] = 1;
        A[v * N + u] = 1;
    }

    // Add random edges to densify the graph
    // C++ Logic:
    // for (int i = 0; i < N; i += 1) {
    //     int numEdges = rand() % (N - 1);
    //     for (int j = 0; j < numEdges; j += 1) {
    //         int other = rand() % N;
    //         if (other != i) {
    //             A[i * N + other] = 1;
    //             A[other * N + i] = 1;
    //         }
    //     }
    // }
    if N > 1 {
        for i in 0..N {
            let num_edges = rng.gen_range(0..(N - 1));
            for _ in 0..num_edges {
                let other = rng.gen_range(0..N);
                if other != i {
                    A[i * N + other] = 1;
                    A[other * N + i] = 1;
                }
            }
        }
    }
}

// ============================================================================
//  4. Benchmark State Implementation
// ============================================================================

struct GraphContext {
    a: Vec<i32>,
    n: usize,
    source: i32,
    dest: i32,
}

impl ParEvalBenchmark for GraphContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            a: vec![0; driver_problem_size * driver_problem_size],
            n: driver_problem_size,
            source: 0,
            dest: 0,
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        random_connected_undirected_graph(&mut self.a, self.n);
        self.source = rng.gen_range(0..self.n as i32);
        
        // Ensure dest != source
        loop {
            self.dest = rng.gen_range(0..self.n as i32);
            if self.dest != self.source {
                break;
            }
        }
    }

    fn compute(&mut self) {
        let result = shortest_path_length(&self.a, self.n, self.source, self.dest);
        std::hint::black_box(result);
    }

    fn best(&mut self) {
        let result = correct_shortest_path_length(&self.a, self.n, self.source, self.dest);
        std::hint::black_box(result);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        let mut a_test = vec![0; TEST_SIZE * TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // Set up input
            random_connected_undirected_graph(&mut a_test, TEST_SIZE);
            let source = rng.gen_range(0..TEST_SIZE as i32);
            let mut dest;
            loop {
                dest = rng.gen_range(0..TEST_SIZE as i32);
                if dest != source {
                    break;
                }
            }

            // Compute correct result
            let mut correct = correct_shortest_path_length(&a_test, TEST_SIZE, source, dest);

            // Compute test result
            let mut test = shortest_path_length(&a_test, TEST_SIZE, source, dest);

            // Normalize "not found" or invalid results to -1 for comparison
            // C++ uses std::numeric_limits<int>::max() for not found
            if correct == i32::MAX || correct < 0 {
                correct = -1;
            }
            if test == i32::MAX || test < 0 {
                test = -1;
            }

            if correct != test {
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
    run::<GraphContext>();
}