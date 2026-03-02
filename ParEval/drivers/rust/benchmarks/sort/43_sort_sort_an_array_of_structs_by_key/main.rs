// main.rs
use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024 * 1024;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Result {
    pub start_time: i32,
    pub duration: i32,
    pub value: f32,
}

impl Default for Result {
    fn default() -> Self {
        Self {
            start_time: 0,
            duration: 0,
            value: 0.0,
        }
    }
}

pub fn sort_by_start_time(results: &mut [Result]) {
    // LLM_OUTPUT_HERE
}

fn correct_sort_by_start_time(results: &mut [Result]) {
    // Using sort_by (stable) to act as the reference implementation
    results.sort_by(|a, b| a.start_time.cmp(&b.start_time));
}

struct SortStructsContext {
    results: Vec<Result>,
}

impl ParEvalBenchmark for SortStructsContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            results: vec![Result::default(); driver_problem_size],
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        for r in self.results.iter_mut() {
            // Mimic C++ fillRand ranges
            r.start_time = rng.gen_range(0..=100);
            r.duration = rng.gen_range(1..=10);
            r.value = rng.gen_range(-1.0..1.0);
        }
    }

    fn compute(&mut self) {
        sort_by_start_time(&mut self.results);
    }

    fn best(&mut self) {
        correct_sort_by_start_time(&mut self.results);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        let mut correct = vec![Result::default(); TEST_SIZE];
        let mut test = vec![Result::default(); TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // Fill with random data
            for i in 0..TEST_SIZE {
                let r = Result {
                    start_time: rng.gen_range(0..=100),
                    duration: rng.gen_range(1..=10),
                    value: rng.gen_range(-1.0..1.0),
                };
                correct[i] = r;
                test[i] = r;
            }

            // Compute correct result
            correct_sort_by_start_time(&mut correct);

            // Compute test result
            sort_by_start_time(&mut test);

            // Validate
            // We rely on PartialEq derived on the Result struct
            // This compares all fields; strict float equality is acceptable for simple moves/sorts
            if correct != test {
                return false;
            }
        }
        true
    }
}

fn main() {
    run::<SortStructsContext>();
}