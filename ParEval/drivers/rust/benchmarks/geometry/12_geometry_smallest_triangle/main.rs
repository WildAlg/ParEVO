// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rand::prelude::*;
use rayon::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

#[derive(Clone, Copy, Default, Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Return the area of the smallest triangle that can be formed by any 3 points.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [{0, 10}, {5, 5}, {1,0}, {-1, 1}, {-10, 0}]
   output: 5.5
*/
pub fn smallest_area(points: &[Point]) -> f64 {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_smallest_area(points: &[Point]) -> f64 {
    let n = points.len();
    if n < 3 {
        return 0.0;
    }

    let mut min_area = f64::MAX;

    for i in 0..n - 2 {
        let p1 = points[i];
        for j in (i + 1)..n - 1 {
            let p2 = points[j];
            for k in (j + 1)..n {
                let p3 = points[k];

                let val = p1.x * (p2.y - p3.y) +
                          p2.x * (p3.y - p1.y) +
                          p3.x * (p1.y - p2.y);
                let area = 0.5 * val.abs();

                if area < min_area {
                    min_area = area;
                }
            }
        }
    }
    min_area
}

// Helper to fill points randomly
fn fill_random_points(points: &mut [Point]) {
    let mut rng = rand::thread_rng();
    for p in points.iter_mut() {
        p.x = rng.gen_range(-1000.0..1000.0);
        p.y = rng.gen_range(-1000.0..1000.0);
    }
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct GeometryContext {
    points: Vec<Point>,
}

impl ParEvalBenchmark for GeometryContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            points: vec![Point::default(); driver_problem_size],
        }
    }

    fn reset(&mut self) {
        fill_random_points(&mut self.points);
    }

    fn compute(&mut self) {
        let _ = smallest_area(&self.points);
    }

    fn best(&mut self) {
        let _ = correct_smallest_area(&self.points);
    }

    fn validate(&mut self) -> bool {
        let mut points = vec![Point::default(); TEST_SIZE];
        
        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            fill_random_points(&mut points);

            let correct = correct_smallest_area(&points);
            let test = smallest_area(&points);

            if (correct - test).abs() > 1e-4 {
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
    run::<GeometryContext>();
}