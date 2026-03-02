// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rand::prelude::*;
use rayon::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 4096;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Return the distance between the closest two points in the vector points.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [{2, 3}, {12, 30}, {40, 50}, {5, 1}, {12, 10}, {3, 4}]
   output: 1.41421
*/
pub fn closestPair(points: &[Point]) -> f64 {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn dist(p1: &Point, p2: &Point) -> f64 {
    ((p2.x - p1.x).powi(2) + (p2.y - p1.y).powi(2)).sqrt()
}

fn correct_closest_pair(points: &[Point]) -> f64 {
    // The polygon needs to have at least two points
    if points.len() < 2 {
        return 0.0;
    }

    let mut min_dist = f64::MAX;
    for i in 0..points.len() - 1 {
        for j in (i + 1)..points.len() {
            let d = dist(&points[i], &points[j]);
            if d < min_dist {
                min_dist = d;
            }
        }
    }

    min_dist
}

fn fill_rand(points: &mut [Point], min: f64, max: f64) {
    let mut rng = rand::thread_rng();
    for p in points.iter_mut() {
        p.x = rng.gen_range(min..max);
        p.y = rng.gen_range(min..max);
    }
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct Context {
    points: Vec<Point>,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            points: vec![Point::default(); driver_problem_size],
        }
    }

    fn reset(&mut self) {
        fill_rand(&mut self.points, -1000.0, 1000.0);
    }

    fn compute(&mut self) {
        let _ = closestPair(&self.points);
    }

    fn best(&mut self) {
        let _ = correct_closest_pair(&self.points);
    }

    fn validate(&mut self) -> bool {
        const TEST_SIZE: usize = 1024;
        let mut points = vec![Point::default(); TEST_SIZE];
        
        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // set up input
            fill_rand(&mut points, -1000.0, 1000.0);

            // compute correct result
            let correct = correct_closest_pair(&points);

            // compute test result
            let test = closestPair(&points);

            // validate
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
    run::<Context>();
}