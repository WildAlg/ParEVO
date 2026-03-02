// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1_000_000;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/// Find the set of points that defined the smallest convex polygon that contains all the points in the vector points. 
/// Store the result in `hull`.
/// Use Rust Rayon to compute in parallel.
pub fn convexHull(points: &[Point], hull: &mut Vec<Point>) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_convex_hull(points: &[Point], hull: &mut Vec<Point>) {
    // The polygon needs to have at least three points
    if points.len() < 3 {
        *hull = points.to_vec();
        return;
    }

    let mut points_sorted = points.to_vec();

    // Sort by x, then y
    points_sorted.sort_unstable_by(|a, b| {
        a.x.partial_cmp(&b.x).unwrap()
            .then(a.y.partial_cmp(&b.y).unwrap())
    });

    let cross_product = |a: Point, b: Point, c: Point| -> bool {
        (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x) > 0.0
    };

    let mut upper_hull = Vec::new();
    let mut lower_hull = Vec::new();

    // Initial points
    upper_hull.push(points_sorted[0]);
    upper_hull.push(points_sorted[1]);

    let len = points_sorted.len();
    lower_hull.push(points_sorted[len - 1]);
    lower_hull.push(points_sorted[len - 2]);

    for i in 2..len {
        // Upper hull
        while upper_hull.len() > 1 && !cross_product(
            upper_hull[upper_hull.len() - 2],
            upper_hull[upper_hull.len() - 1],
            points_sorted[i]
        ) {
            upper_hull.pop();
        }
        upper_hull.push(points_sorted[i]);

        // Lower hull
        let lower_idx = len - i - 1;
        while lower_hull.len() > 1 && !cross_product(
            lower_hull[lower_hull.len() - 2],
            lower_hull[lower_hull.len() - 1],
            points_sorted[lower_idx]
        ) {
            lower_hull.pop();
        }
        lower_hull.push(points_sorted[lower_idx]);
    }

    // Merge: upperHull.insert(upperHull.end(), lowerHull.begin()+1, lowerHull.end()-1);
    if lower_hull.len() >= 2 {
        upper_hull.extend_from_slice(&lower_hull[1..lower_hull.len() - 1]);
    }

    *hull = upper_hull;
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct Context {
    points: Vec<Point>,
    hull: Vec<Point>,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Context {
            points: vec![Point { x: 0.0, y: 0.0 }; driver_problem_size],
            hull: Vec::new(),
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        // fillRand(ctx->x, -1000.0, 1000.0) for both x and y
        for p in self.points.iter_mut() {
            p.x = rng.gen_range(-1000.0..1000.0);
            p.y = rng.gen_range(-1000.0..1000.0);
        }
        self.hull.clear();
    }

    fn compute(&mut self) {
        convexHull(&self.points, &mut self.hull);
    }

    fn best(&mut self) {
        correct_convex_hull(&self.points, &mut self.hull);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // Set up input
            let mut points = vec![Point { x: 0.0, y: 0.0 }; TEST_SIZE];
            for p in points.iter_mut() {
                p.x = rng.gen_range(-1000.0..1000.0);
                p.y = rng.gen_range(-1000.0..1000.0);
            }

            // Compute correct result
            let mut correct = Vec::new();
            correct_convex_hull(&points, &mut correct);

            // Compute test result
            let mut test = Vec::new();
            convexHull(&points, &mut test);

            // Compare
            if test.len() != correct.len() {
                return false;
            }

            // Sort both hulls to compare, as points might be ordered differently on the hull perimeter
            // C++ sort: a.x < b.x || (a.x == b.x && a.y < b.y)
            let sort_cmp = |a: &Point, b: &Point| {
                a.x.partial_cmp(&b.x).unwrap()
                    .then(a.y.partial_cmp(&b.y).unwrap())
            };
            
            test.sort_by(sort_cmp);
            correct.sort_by(sort_cmp);

            for (t, c) in test.iter().zip(correct.iter()) {
                if (t.x - c.x).abs() > 1e-6 || (t.y - c.y).abs() > 1e-6 {
                    return false;
                }
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