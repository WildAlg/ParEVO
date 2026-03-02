// main.rs
use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024 * 1024;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;


pub fn countQuartiles(x: &[f64], bins: &mut [usize; 4]) {
    // LLM_OUTPUT_HERE
}

fn correct_count_quartiles(x: &[f64], bins: &mut [usize; 4]) {
    for &val in x {
        let frac = val - (val as i64 as f64);
        if frac < 0.25 {
            bins[0] += 1;
        } else if frac < 0.5 {
            bins[1] += 1;
        } else if frac < 0.75 {
            bins[2] += 1;
        } else {
            bins[3] += 1;
        }
    }
}

struct HistogramContext {
    x: Vec<f64>,
    bins: [usize; 4],
}

impl ParEvalBenchmark for HistogramContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            x: vec![0.0; driver_problem_size],
            bins: [0; 4],
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        // Fill x with random doubles between 0.0 and 100.0
        for val in self.x.iter_mut() {
            *val = rng.gen_range(0.0..100.0);
        }
        // Reset bins
        self.bins = [0; 4];
    }

    fn compute(&mut self) {
        countQuartiles(&self.x, &mut self.bins);
    }

    fn best(&mut self) {
        correct_count_quartiles(&self.x, &mut self.bins);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        let mut x = vec![0.0; TEST_SIZE];
        
        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            for val in x.iter_mut() {
                *val = rng.gen_range(0.0..100.0);
            }

            let mut correct_bins = [0usize; 4];
            let mut test_bins = [0usize; 4];

            correct_count_quartiles(&x, &mut correct_bins);
            countQuartiles(&x, &mut test_bins);

            if correct_bins != test_bins {
                return false;
            }
        }
        true
    }
}

fn main() {
    run::<HistogramContext>();
}