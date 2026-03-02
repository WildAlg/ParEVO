// main.rs

use pareval_runner::{ParEvalBenchmark, run}; 

use rand::prelude::*;
use rayon::prelude::*;

// Constants for the problem sizes
// const DRIVER_PROBLEM_SIZE: usize = 1024;
const TEST_SIZE: usize = 512;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/// The LLM solution should be pasted here.
pub fn luFactorize(A: &mut [f64], N: usize) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_lu_factorize(A: &mut [f64], N: usize) {
    for k in 0..N {
        let pivot = A[k * N + k];
        let inv_pivot = 1.0 / pivot;

        // Compute L column
        for i in (k + 1)..N {
            A[i * N + k] *= inv_pivot;
        }

        // Update submatrix
        for i in (k + 1)..N {
            let val_ik = A[i * N + k];
            for j in (k + 1)..N {
                let val_kj = A[k * N + j];
                A[i * N + j] -= val_ik * val_kj;
            }
        }
    }
}

// Equivalent to fillRand in utilities.hpp
// Populates a slice with random values between min and max
fn fill_rand(slice: &mut [f64], min: f64, max: f64) {
    let mut rng = rand::thread_rng();
    for x in slice.iter_mut() {
        *x = rng.gen_range(min..max);
    }
}

// Equivalent to fequal in utilities.hpp
// Compares two slices with a tolerance epsilon
fn fequal(a: &[f64], b: &[f64], epsilon: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    // zip iterates both simultaneously; all() checks if the condition holds for every pair
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= epsilon)
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct LuDecompContext {
    a: Vec<f64>,
    n: usize,
}

impl ParEvalBenchmark for LuDecompContext {
    fn new() -> Self {
        let size_str = std::env!("DRIVER_PROBLEM_SIZE"); 

        let driver_problem_size: usize = size_str.parse().expect("Not a number!");
        // let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            a: vec![0.0; driver_problem_size * driver_problem_size],
            n: driver_problem_size,
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        // Fill with random double values between -10.0 and 10.0
        for x in self.a.iter_mut() {
            *x = rng.gen_range(-10.0..10.0);
        }
    }

    fn compute(&mut self) {
        // Execute the parallel solution (LLM generated)
        luFactorize(&mut self.a, self.n);
    }

    fn best(&mut self) {
        // Execute the sequential baseline
        correct_lu_factorize(&mut self.a, self.n);
    }

    // Assuming this is inside: impl ParEvalBenchmark for LuDecompContext { ... }
    fn validate(&mut self) -> bool {
        const TEST_SIZE: usize = 512;
        
        // In Rust, we allocate Vectors instead of using std::vector
        // We create fresh buffers for validation so we don't mess up the main benchmark state
        let n = TEST_SIZE;
        let mut a_input = vec![0.0; n * n];
        let mut a_correct = vec![0.0; n * n];
        let mut a_test = vec![0.0; n * n];

        // NOTE: MPI constructs (GET_RANK, BCAST, SYNC) are removed 
        // because Rayon runs in a single process shared-memory environment.
        // rank is always 0, and IS_ROOT is always true.

        for _trial_iter in 0..MAX_VALIDATION_ATTEMPTS {
            // 1. Set up input (equivalent to fillRand)
            fill_rand(&mut a_input, -10.0, 10.0);
            // No need for BCAST(A, DOUBLE); memory is local.

            // 2. Compute correct result (Sequential Baseline)
            a_correct.copy_from_slice(&a_input);
            correct_lu_factorize(&mut a_correct, n);

            // 3. Compute test result (Parallel Rayon Solution)
            a_test.copy_from_slice(&a_input);
            luFactorize(&mut a_test, n);
            
            // No need for SYNC(); Rust function calls are blocking by default.

            // 4. Compare
            // IS_ROOT(rank) is implicitly true in shared memory.
            if !fequal(&a_correct, &a_test, 1e-3) {
                // In C++, BCAST_PTR sends the failure to other nodes. 
                // Here, we just return false immediately.
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
    // Passes the Context type to the driver defined in lib.rs
    run::<LuDecompContext>();
}