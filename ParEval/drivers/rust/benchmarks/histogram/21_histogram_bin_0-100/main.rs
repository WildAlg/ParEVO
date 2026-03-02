// main.rs
use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1_000_000;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* Vector x contains values between 0 and 100, inclusive. Count the number of
   values in [0,10), [10, 20), [20, 30), ... and store the counts in `bins`.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [7, 32, 95, 12, 39, 32, 11, 71, 70, 66]
   output: [1, 2, 0, 3, 0, 0, 1, 2, 0, 1]
*/
pub fn bins_by_10_count(x: &[f64], bins: &mut [usize; 10]) {
    // LLM_OUTPUT_HERE
}

fn correct_bins_by_10_count(x: &[f64], bins: &mut [usize; 10]) {
    for &val in x {
        let mut bin = (val / 10.0) as usize;
        // Clamp to the last bin if val is 100.0 (or slightly larger due to float inaccuracies)
        // C++: bin = std::min(bin, bins.size() - 1);
        if bin >= 10 {
            bin = 9;
        }
        bins[bin] += 1;
    }
}

struct HistogramContext {
    x: Vec<f64>,
    bins: [usize; 10],
}

impl ParEvalBenchmark for HistogramContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            x: vec![0.0; driver_problem_size],
            bins: [0; 10],
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        // Values between 0 and 100 inclusive
        for val in self.x.iter_mut() {
            *val = rng.gen_range(0.0..=100.0);
        }
        self.bins = [0; 10];
    }

    fn compute(&mut self) {
        bins_by_10_count(&self.x, &mut self.bins);
    }

    fn best(&mut self) {
        correct_bins_by_10_count(&self.x, &mut self.bins);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        let mut x = vec![0.0; TEST_SIZE];
        let mut correct = [0usize; 10];
        let mut test = [0usize; 10];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // 1. Setup input
            for val in x.iter_mut() {
                *val = rng.gen_range(0.0..=100.0);
            }
            // Clear bins
            correct = [0; 10];
            test = [0; 10];

            // 2. Compute correct result
            correct_bins_by_10_count(&x, &mut correct);

            // 3. Compute test result
            bins_by_10_count(&x, &mut test);

            // 4. Compare
            if correct != test {
                return false;
            }
        }
        true
    }
}

fn main() {
    run::<HistogramContext>();
}