use pareval_runner::{ParEvalBenchmark, run};
use rand::prelude::*;
use rayon::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1_000_000;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

/* Count the number of pixels in image with each grayscale intensity.
   The vector `image` is a grayscale image with values 0-255.
   Store the results in `bins`.
   Use Rust Rayon to count in parallel.
   Example:
   
   input: image=[2, 116, 201, 11, 92, 92, 201, 4, 2]
   output: [0, 0, 2, 0, 1, ...]
*/
pub fn pixelCounts(image: &[i32], bins: &mut [usize; 256]) {
    // LLM_OUTPUT_HERE
}

fn correct_pixel_counts(image: &[i32], bins: &mut [usize; 256]) {
    for &pixel in image {
        bins[pixel as usize] += 1;
    }
}

struct HistogramContext {
    image: Vec<i32>,
    bins: [usize; 256],
}

impl ParEvalBenchmark for HistogramContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            image: vec![0; driver_problem_size],
            bins: [0; 256],
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        for x in self.image.iter_mut() {
            *x = rng.gen_range(0..=255);
        }
        self.bins = [0; 256];
    }

    fn compute(&mut self) {
        pixelCounts(&self.image, &mut self.bins);
    }

    fn best(&mut self) {
        correct_pixel_counts(&self.image, &mut self.bins);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        let mut image = vec![0; TEST_SIZE];
        let mut correct = [0; 256];
        let mut test = [0; 256];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // set up input
            for x in image.iter_mut() {
                *x = rng.gen_range(0..=255);
            }

            // reset bins
            correct = [0; 256];
            test = [0; 256];

            // compute correct result
            correct_pixel_counts(&image, &mut correct);

            // compute test result
            pixelCounts(&image, &mut test);

            // compare
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