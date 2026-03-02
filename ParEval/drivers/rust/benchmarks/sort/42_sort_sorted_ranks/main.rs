use pareval_runner::{ParEvalBenchmark, run};
use rand::prelude::*;
use rayon::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024 * 1024;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* For each value in the vector x compute its index in the sorted vector.
   Store the results in `ranks`.
   Use Rust Rayon to compute in parallel.
   Examples:

   input: [3.1, 2.8, 9.1, 0.4, 3.14]
   output: [2, 1, 4, 0, 3]
 
   input: [100, 7.6, 16.1, 18, 7.6]
   output: [4, 0, 1, 2, 3]
*/
pub fn ranks(x: &[f32], ranks: &mut [usize]) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_ranks(x: &[f32], ranks: &mut [usize]) {
    // Create indices [0, 1, ..., N-1]
    let mut indices: Vec<usize> = (0..x.len()).collect();

    // Sort indices based on the values in x.
    // std::sort is unstable, so we use sort_unstable_by for comparable behavior/performance.
    indices.sort_unstable_by(|&i1, &i2| {
        // Handle float comparison. We assume no NaNs in input as per benchmark standard.
        x[i1].partial_cmp(&x[i2]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Assign ranks
    for (i, &idx) in indices.iter().enumerate() {
        ranks[idx] = i;
    }
}

// Helper to fill a slice with random values
fn fill_rand(slice: &mut [f32], min: f32, max: f32) {
    let mut rng = rand::thread_rng();
    for val in slice.iter_mut() {
        *val = rng.gen_range(min..max);
    }
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct RanksContext {
    x: Vec<f32>,
    ranks: Vec<usize>,
}

impl ParEvalBenchmark for RanksContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            x: vec![0.0; driver_problem_size],
            ranks: vec![0; driver_problem_size],
        }
    }

    fn reset(&mut self) {
        fill_rand(&mut self.x, -100.0, 100.0);
    }

    fn compute(&mut self) {
        ranks(&self.x, &mut self.ranks);
    }

    fn best(&mut self) {
        correct_ranks(&self.x, &mut self.ranks);
    }

    fn validate(&mut self) -> bool {
        let n = TEST_SIZE;
        let mut x = vec![0.0; n];
        let mut correct = vec![0; n];
        let mut test = vec![0; n];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // Set up input
            fill_rand(&mut x, -100.0, 100.0);

            // Compute correct result
            correct_ranks(&x, &mut correct);

            // Compute test result
            ranks(&x, &mut test);

            // Compare
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
    run::<RanksContext>();
}