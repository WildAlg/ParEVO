// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1_000_000;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Parallel Solution Placeholder
// ============================================================================

/* For each letter in the alphabet, count the number of strings in the vector s that start with that letter.
   Assume all strings are in lower case. Store the output in `bins` array.
   Use Rust Rayon to compute in parallel.
   Example:

   input: ["dog", "cat", "xray", "cow", "code", "type", "flower"]
   output: [0, 0, 3, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
*/
pub fn firstLetterCounts(s: &[String], bins: &mut [usize; 26]) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_first_letter_counts(s: &[String], bins: &mut [usize; 26]) {
    for string in s {
        let bytes = string.as_bytes();
        if !bytes.is_empty() {
            let c = bytes[0];
            // 'a' is 97 in ASCII
            let index = (c as usize).wrapping_sub(b'a' as usize);
            if index < 26 {
                bins[index] += 1;
            }
        }
    }
}

// ============================================================================
//  3. Helper Functions
// ============================================================================

fn fill_rand_string(s: &mut Vec<String>, min_len: usize, max_len: usize) {
    let mut rng = rand::thread_rng();
    for str_ref in s.iter_mut() {
        let len = rng.gen_range(min_len..=max_len);
        let new_str: String = (0..len)
            .map(|_| (b'a' + rng.gen_range(0..26)) as char)
            .collect();
        *str_ref = new_str;
    }
}

// ============================================================================
//  4. Benchmark State Implementation
// ============================================================================

struct Context {
    s: Vec<String>,
    bins: [usize; 26],
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Context {
            s: vec![String::new(); driver_problem_size],
            bins: [0; 26],
        }
    }

    fn reset(&mut self) {
        fill_rand_string(&mut self.s, 2, 10);
        self.bins = [0; 26];
    }

    fn compute(&mut self) {
        firstLetterCounts(&self.s, &mut self.bins);
    }

    fn best(&mut self) {
        correct_first_letter_counts(&self.s, &mut self.bins);
    }

    fn validate(&mut self) -> bool {
        const TEST_SIZE: usize = 1024;
        let mut rng = rand::thread_rng();

        // Allocations for validation
        let mut s = vec![String::new(); TEST_SIZE];
        let mut correct = [0usize; 26];
        let mut test = [0usize; 26];
        let mut first_letters = vec![0u8; TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // Set up input
            for j in 0..TEST_SIZE {
                first_letters[j] = b'a' + rng.gen_range(0..26);
            }

            fill_rand_string(&mut s, 2, 10);
            
            // Ensure every string has the pre-determined first letter
            for j in 0..TEST_SIZE {
                // We know fill_rand_string(..., 2, 10) ensures len >= 2.
                // We replace the first character.
                let first_char = first_letters[j] as char;
                s[j].replace_range(..1, &first_char.to_string());
            }

            correct = [0; 26];
            test = [0; 26];

            // Compute correct result
            correct_first_letter_counts(&s, &mut correct);

            // Compute test result
            firstLetterCounts(&s, &mut test);

            // Compare
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
    run::<Context>();
}