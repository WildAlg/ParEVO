use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1_000_000;

/* Return the index of the first even number in the vector x.
   Use Rust Rayon to parallelize the search.
   Examples:

   input: [7, 3, 9, 5, 5, 7, 2, 9, 12, 11]
   output: 6

   input: [3, 8, 9, 9, 3, 4, 8, 6]
   output: 1
*/
pub fn findFirstEven(x: &[i32]) -> usize {
    // LLM_OUTPUT_HERE
}

fn correct_find_first_even(x: &[i32]) -> usize {
    for (i, &val) in x.iter().enumerate() {
        if val % 2 == 0 {
            return i;
        }
    }
    x.len()
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
        let mut rng = rand::thread_rng();
        
        // Fill with random values between 1 and 20
        for v in self.x.iter_mut() {
            *v = rng.gen_range(1..=20);
        }

        // Make everything odd: x[i] = 2 * x[i] + 1
        for v in self.x.iter_mut() {
            *v = 2 * *v + 1;
        }

        // Make two values in the middle quadrants even
        let len = self.x.len();
        if len > 0 {
            let min = len / 4;
            let max = 3 * len / 4;
            
            // Ensure range is valid
            if max > min {
                let idx1 = rng.gen_range(min..max);
                let idx2 = rng.gen_range(min..max);
                self.x[idx1] += 1;
                self.x[idx2] += 1;
            }
        }
    }

    fn compute(&mut self) {
        let _idx = findFirstEven(&self.x);
    }

    fn best(&mut self) {
        let _idx = correct_find_first_even(&self.x);
    }

    fn validate(&mut self) -> bool {
        const TEST_SIZE: usize = 1024;
        const NUM_TRIES: usize = 10;
        let mut rng = rand::thread_rng();

        for i in 0..NUM_TRIES {
            let mut x = vec![0i32; TEST_SIZE];
            
            // fillRand(x, 1, 100);
            for v in x.iter_mut() {
                *v = rng.gen_range(1..=100);
            }

            // In specific iteration, make first 20 elements odd
            if i == 1 {
                for j in 0..20 {
                    x[j] = 2 * rng.gen_range(0..50) + 1;
                }
            }

            let correct = correct_find_first_even(&x);
            let test = findFirstEven(&x);

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