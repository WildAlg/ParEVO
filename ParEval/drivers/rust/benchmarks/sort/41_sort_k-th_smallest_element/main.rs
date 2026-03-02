// main.rs
use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1_000_000;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* Find the k-th smallest element of the vector x.
   Use Rust Rayon to compute in parallel.
   Example:
   
   input: x=[1, 7, 6, 0, 2, 2, 10, 6], k=4
   output: 6
*/
pub fn findKthSmallest(x: &[i32], k: i32) -> i32 {
    // LLM_OUTPUT_HERE
}

// Sequential baseline translated from baseline.hpp
// "correctFindKthSmallest"
fn correct_find_kth_smallest(x: &[i32], k: i32) -> i32 {
    let mut x_copy = x.to_vec();
    x_copy.sort_unstable(); // equivalent to std::sort
    x_copy[(k - 1) as usize]
}

struct Context {
    x: Vec<i32>,
    k: i32,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Context {
            x: vec![0; driver_problem_size],
            k: 1,
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        // fillRand(ctx->x, 0, 10000);
        for el in self.x.iter_mut() {
            *el = rng.gen_range(0..10000);
        }
        
        // C++ uses rand() % size, but baseline uses k-1.
        // If k=0, k-1 is invalid. Assuming 1-based k to prevent crash and match semantics.
        self.k = rng.gen_range(1..=self.x.len()) as i32;
    }

    fn compute(&mut self) {
        let _ = findKthSmallest(&self.x, self.k);
    }

    fn best(&mut self) {
        let _ = correct_find_kth_smallest(&self.x, self.k);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        let mut x = vec![0; TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // set up input
            for el in x.iter_mut() {
                *el = rng.gen_range(0..10000);
            }
            
            // generate k in [1, size]
            let k = rng.gen_range(1..=x.len()) as i32;

            // compute correct result
            let correct = correct_find_kth_smallest(&x, k);

            // compute test result
            let test = findKthSmallest(&x, k);

            if correct != test {
                return false;
            }
        }
        true
    }
}

fn main() {
    run::<Context>();
}