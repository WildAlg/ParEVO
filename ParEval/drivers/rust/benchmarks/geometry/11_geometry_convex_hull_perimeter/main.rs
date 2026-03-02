// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1_000_000;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

// Helper function for Euclidean distance
fn dist(p1: &Point, p2: &Point) -> f64 {
    ((p2.x - p1.x).powi(2) + (p2.y - p1.y).powi(2)).sqrt()
}

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Return the perimeter of the smallest convex polygon that contains all the points in the vector points.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [{0, 3}, {1, 1}, {2, 2}, {4, 4}, {0, 0}, {1, 2}, {3, 1}, {3, 3}]
   output: 13.4477
*/
pub fn convex_hull_perimeter(points: &[Point]) -> f64 {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_convex_hull_perimeter(points: &[Point]) -> f64 {
    // The polygon needs to have at least three points
    if points.len() < 3 {
        return 0.0;
    }

    let mut points_sorted = points.to_vec();

    // Sort by x, then by y
    points_sorted.sort_by(|a, b| {
        a.x.partial_cmp(&b.x).unwrap().then(a.y.partial_cmp(&b.y).unwrap())
    });

    // Cross product logic to determine turn direction
    // Matches C++ logic: (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x) > 0
    let cross_product = |a: &Point, b: &Point, c: &Point| -> bool {
        (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x) > 0.0
    };

    let mut upper_hull = Vec::new();
    let mut lower_hull = Vec::new();

    // Init upper hull
    upper_hull.push(points_sorted[0]);
    upper_hull.push(points_sorted[1]);

    // Init lower hull
    lower_hull.push(points_sorted[points_sorted.len() - 1]);
    lower_hull.push(points_sorted[points_sorted.len() - 2]);

    for i in 2..points_sorted.len() {
        // Build upper hull
        while upper_hull.len() > 1
            && !cross_product(
                &upper_hull[upper_hull.len() - 2],
                &upper_hull[upper_hull.len() - 1],
                &points_sorted[i],
            )
        {
            upper_hull.pop();
        }
        upper_hull.push(points_sorted[i]);

        // Build lower hull
        let idx_lower = points_sorted.len() - i - 1;
        while lower_hull.len() > 1
            && !cross_product(
                &lower_hull[lower_hull.len() - 2],
                &lower_hull[lower_hull.len() - 1],
                &points_sorted[idx_lower],
            )
        {
            lower_hull.pop();
        }
        lower_hull.push(points_sorted[idx_lower]);
    }

    // Merge lower hull into upper hull
    // Equivalent to: upperHull.insert(upperHull.end(), lowerHull.begin()+1, lowerHull.end()-1);
    if lower_hull.len() > 2 {
        upper_hull.extend_from_slice(&lower_hull[1..lower_hull.len() - 1]);
    }

    // Calculate perimeter
    let mut perimeter = 0.0;
    for i in 0..upper_hull.len() - 1 {
        perimeter += dist(&upper_hull[i], &upper_hull[i + 1]);
    }
    // Close the polygon
    if !upper_hull.is_empty() {
        perimeter += dist(&upper_hull[0], &upper_hull[upper_hull.len() - 1]);
    }

    perimeter
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct ConvexHullContext {
    points: Vec<Point>,
}

impl ParEvalBenchmark for ConvexHullContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            points: vec![Point::default(); driver_problem_size],
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        // Fill random points between -1000.0 and 1000.0
        for p in self.points.iter_mut() {
            p.x = rng.gen_range(-1000.0..1000.0);
            p.y = rng.gen_range(-1000.0..1000.0);
        }
    }

    fn compute(&mut self) {
        // Prevent optimization by using the result
        let result = convex_hull_perimeter(&self.points);
        std::hint::black_box(result);
    }

    fn best(&mut self) {
        let result = correct_convex_hull_perimeter(&self.points);
        std::hint::black_box(result);
    }

    fn validate(&mut self) -> bool {
        let mut points = vec![Point::default(); TEST_SIZE];
        let mut rng = rand::thread_rng();

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // Set up input
            for p in points.iter_mut() {
                p.x = rng.gen_range(-1000.0..1000.0);
                p.y = rng.gen_range(-1000.0..1000.0);
            }

            // Compute correct result
            let correct = correct_convex_hull_perimeter(&points);

            // Compute test result
            let test = convex_hull_perimeter(&points);

            // Compare
            if (correct - test).abs() > 1e-6 {
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
    run::<ConvexHullContext>();
}