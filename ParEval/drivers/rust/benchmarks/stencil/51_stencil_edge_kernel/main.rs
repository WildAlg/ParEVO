// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rand::prelude::*;
use rayon::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

const EDGE_KERNEL: [[i32; 3]; 3] = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]];

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Convolve the edge kernel with a grayscale image. Each pixel will be replaced with
   the dot product of itself and its neighbors with the edge kernel.
   Use a value of 0 for pixels outside the image's boundaries and clip outputs between 0 and 255.
   imageIn and imageOut are NxN grayscale images stored in row-major.
   Store the output of the computation in imageOut.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [[112, 118, 141, 152],
           [93, 101, 119, 203],
           [45, 17, 16, 232],
           [82, 31, 49, 101]]
   output: [[255, 255, 255, 255],
            [255, 147, 0, 255],
            [36, 0, 0, 255],
            [255, 39, 0, 255]]
*/
pub fn convolve_kernel(image_in: &[i32], image_out: &mut [i32], n: usize) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

fn correct_convolve_kernel(image_in: &[i32], image_out: &mut [i32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0;
            // Iterate over the 3x3 kernel
            for k in -1..=1 {
                for l in -1..=1 {
                    let x = i as isize + k;
                    let y = j as isize + l;

                    // Check bounds
                    if x >= 0 && x < n as isize && y >= 0 && y < n as isize {
                        let val = image_in[(x as usize) * n + (y as usize)];
                        let k_val = EDGE_KERNEL[(k + 1) as usize][(l + 1) as usize];
                        sum += val * k_val;
                    } 
                    // else: implicit 0 value for out-of-bounds pixels, so sum += 0
                }
            }

            // Clamp output
            if sum < 0 {
                image_out[i * n + j] = 0;
            } else if sum > 255 {
                image_out[i * n + j] = 255;
            } else {
                image_out[i * n + j] = sum;
            }
        }
    }
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct Context {
    input: Vec<i32>,
    output: Vec<i32>,
    n: usize,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Context {
            input: vec![0; driver_problem_size * driver_problem_size],
            output: vec![0; driver_problem_size * driver_problem_size],
            n: driver_problem_size,
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        // fillRand(ctx->input, 0, 255);
        for x in self.input.iter_mut() {
            *x = rng.gen_range(0..=255);
        }
        // std::fill(ctx->output.begin(), ctx->output.end(), 0);
        self.output.fill(0);
    }

    fn compute(&mut self) {
        convolve_kernel(&self.input, &mut self.output, self.n);
    }

    fn best(&mut self) {
        correct_convolve_kernel(&self.input, &mut self.output, self.n);
    }

    fn validate(&mut self) -> bool {
        let n = TEST_SIZE;
        let mut input = vec![0; n * n];
        let mut correct = vec![0; n * n];
        let mut test = vec![0; n * n];
        
        let mut rng = rand::thread_rng();

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // set up input
            for x in input.iter_mut() {
                *x = rng.gen_range(0..=255);
            }
            correct.fill(0);
            test.fill(0);

            // compute correct result
            correct_convolve_kernel(&input, &mut correct, n);

            // compute test result
            convolve_kernel(&input, &mut test, n);

            // compare
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