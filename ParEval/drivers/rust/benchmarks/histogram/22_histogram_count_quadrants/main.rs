// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1_000_000;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

#[derive(Clone, Copy, Debug, Default)]
struct Point {
   x: f64,
   y: f64,
}

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

pub fn countQuadrants(points: &[Point], bins: &mut [usize; 4]) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_count_quadrants(points: &[Point], bins: &mut [usize; 4]) {
    for point in points {
        if point.x >= 0.0 && point.y >= 0.0 {
            bins[0] += 1;
        } else if point.x < 0.0 && point.y >= 0.0 {
            bins[1] += 1;
        } else if point.x < 0.0 && point.y < 0.0 {
            bins[2] += 1;
        } else if point.x >= 0.0 && point.y < 0.0 {
            bins[3] += 1;
        }
    }
}

// Helper to fill points with random data, equivalent to C++ initialization
fn fill_rand_points(points: &mut [Point]) {
    let mut rng = rand::thread_rng();
    for p in points.iter_mut() {
        p.x = rng.gen_range(-1.0..1.0);
        p.y = rng.gen_range(-1.0..1.0);
    }
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct Context {
    points: Vec<Point>,
    bins: [usize; 4],
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            points: vec![Point::default(); driver_problem_size],
            bins: [0; 4],
        }
    }

    fn reset(&mut self) {
        fill_rand_points(&mut self.points);
        self.bins = [0; 4];
    }

    fn compute(&mut self) {
        countQuadrants(&self.points, &mut self.bins);
    }

    fn best(&mut self) {
        correct_count_quadrants(&self.points, &mut self.bins);
    }

    fn validate(&mut self) -> bool {
        let mut points = vec![Point::default(); TEST_SIZE];
        let mut correct = [0usize; 4];
        let mut test = [0usize; 4];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // Set up input
            fill_rand_points(&mut points);
            
            // Reset bins
            correct = [0; 4];
            test = [0; 4];

            // Compute correct result
            correct_count_quadrants(&points, &mut correct);

            // Compute test result
            countQuadrants(&points, &mut test);

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
    run::<Context>();
}