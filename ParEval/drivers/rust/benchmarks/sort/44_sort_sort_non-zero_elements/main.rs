use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1048576; // Default to ~1M elements
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* Sort the vector x in ascending order ignoring elements with value 0.
   Leave zero valued elements in-place.
	 Use Rust Rayon to sort x in parallel.
   Example:

   input: [8, 4, 0, 9, 8, 0, 1, -1, 7]
   output: [-1, 1, 0, 4, 7, 0, 8, 8, 9]
*/
pub fn sort_ignore_zero(x: &mut Vec<i32>) {
    // LLM_OUTPUT_HERE
}

fn correct_sort_ignore_zero(x: &mut Vec<i32>) {
    // Extract non-zero elements
    let mut non_zero_elements: Vec<i32> = x.iter().cloned().filter(|&n| n != 0).collect();
    
    // Sort them
    non_zero_elements.sort_unstable();

    // Place them back into the original vector, skipping zeros
    let mut non_zero_index = 0;
    for i in 0..x.len() {
        if x[i] != 0 {
            x[i] = non_zero_elements[non_zero_index];
            non_zero_index += 1;
        }
    }
}

fn fill_rand_with_zeroes(x: &mut Vec<i32>) {
    let mut rng = rand::thread_rng();
    for i in 0..x.len() {
        x[i] = rng.gen();
        // C++ logic: if (rand() % 5) { x[i] = 0; }
        // In C++, rand() % 5 returns 0, 1, 2, 3, 4.
        // Condition is true for 1, 2, 3, 4 (non-zero).
        // So 4/5 (80%) probability of setting to 0.
        if rng.gen_range(0..5) != 0 {
            x[i] = 0;
        }
    }
}

struct Context {
    x: Vec<i32>,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Context {
            x: vec![0; driver_problem_size],
        }
    }

    fn reset(&mut self) {
        fill_rand_with_zeroes(&mut self.x);
    }

    fn compute(&mut self) {
        sort_ignore_zero(&mut self.x);
    }

    fn best(&mut self) {
        correct_sort_ignore_zero(&mut self.x);
    }

    fn validate(&mut self) -> bool {
        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            const TEST_SIZE: usize = 1024;
            let mut input = vec![0; TEST_SIZE];
            fill_rand_with_zeroes(&mut input);

            let mut correct_result = input.clone();
            correct_sort_ignore_zero(&mut correct_result);

            let mut test_result = input.clone();
            sort_ignore_zero(&mut test_result);

            if correct_result != test_result {
                return false;
            }
        }
        true
    }
}

fn main() {
    run::<Context>();
}