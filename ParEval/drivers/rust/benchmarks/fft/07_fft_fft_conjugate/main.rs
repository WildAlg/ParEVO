// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;
use num_complex::Complex;
use std::f64::consts::PI;

// Constants matching C++ driver/validation logic
// const DRIVER_PROBLEM_SIZE: usize = 1 << 20; // 2^20 approx 1 million elements
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Compute the fourier transform of x in-place. Return the imaginary conjugate of each value.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
   output: [{4,0}, {1,-2.41421}, {0,0}, {1,-0.414214}, {0,0}, {1,0.414214}, {0,0}, {1,2.41421}]
*/
pub fn fft_conjugate(x: &mut [num_complex::Complex<f64>]) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baselines
// ============================================================================

/// Iterative Cooley-Tukey implementation (corresponds to `correctFft` in baseline.hpp)
/// This is used for the "best" (performance reference) logic.
fn correct_fft(x: &mut [Complex<f64>]) {
    let n_total = x.len();
    let mut k = n_total;
    let theta_t = PI / (n_total as f64);
    let mut phi_t = Complex::new(theta_t.cos(), -theta_t.sin());
    
    // DFT (Decimation in Frequency)
    while k > 1 {
        let n = k;
        k >>= 1;
        phi_t = phi_t * phi_t;
        let mut t_curr = Complex::new(1.0, 0.0);
        
        for l in 0..k {
            for a in (l..n_total).step_by(n) {
                let b = a + k;
                let t_val = x[a] - x[b];
                x[a] = x[a] + x[b];
                x[b] = t_val * t_curr;
            }
            t_curr = t_curr * phi_t;
        }
    }

    // Decimate (Bit Reversal)
    let m = (n_total as f64).log2() as u32;
    for a in 0..n_total {
        let mut b = a as u32;
        // Reverse bits (32-bit integer reversal logic from C++)
        b = ((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1);
        b = ((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2);
        b = ((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4);
        b = ((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8);
        // Shift based on m
        b = ((b >> 16) | (b << 16)) >> (32 - m);
        
        let b = b as usize;
        if b > a {
            x.swap(a, b);
        }
    }

    // Conjugate
    for val in x.iter_mut() {
        *val = val.conj();
    }
}

fn fft_cooley_tukey_recursive(x: &mut [Complex<f64>]) {
    let n = x.len();
    if n <= 1 {
        // C++: "if (N <= 1) return;"
        // Note: The C++ function actually returns *before* the conjugate loop if N <= 1.
        // So for N=1, it does NOT conjugate. We must match that.
        return;
    }

    // divide
    let half = n / 2;
    let mut even = Vec::with_capacity(half);
    let mut odd = Vec::with_capacity(half);

    for i in 0..half {
        even.push(x[i * 2]);
        odd.push(x[i * 2 + 1]);
    }

    // conquer
    fft_cooley_tukey_recursive(&mut even);
    fft_cooley_tukey_recursive(&mut odd);

    // combine
    for k in 0..half {
        // C++: std::polar(1.0, -2 * M_PI * k / N)
        let angle = -2.0 * PI * (k as f64) / (n as f64);
        let t = Complex::from_polar(1.0, angle) * odd[k];
        x[k] = even[k] + t;
        x[k + half] = even[k] - t;
    }

    // conjugate (MUST happen at every level of recursion, just like C++)
    for val in x.iter_mut() {
        *val = val.conj();
    }
}

// ============================================================================
//  3. Benchmark State & Validator
// ============================================================================

struct Context {
    x: Vec<Complex<f64>>,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            x: vec![Complex::new(0.0, 0.0); driver_problem_size],
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        // fillRand -1.0 to 1.0 for real and imag
        // C++: fills separate real/imag vectors then combines. 
        // Equivalent here is just generating complex numbers directly.
        for val in self.x.iter_mut() {
            let re = rng.gen_range(-1.0..1.0);
            let im = rng.gen_range(-1.0..1.0);
            *val = Complex::new(re, im);
        }
    }

    fn compute(&mut self) {
        fft_conjugate(&mut self.x);
    }

    fn best(&mut self) {
        correct_fft(&mut self.x);
    }

    fn validate(&mut self) -> bool {
        // Validation loop matching C++ validate() logic
        let mut rng = rand::thread_rng();
        let n = TEST_SIZE;

        // buffers for validation
        let mut x_input = vec![Complex::new(0.0, 0.0); n];
        let mut x_correct = vec![Complex::new(0.0, 0.0); n];
        let mut x_test = vec![Complex::new(0.0, 0.0); n];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // set up input
            for k in 0..n {
                let re = rng.gen_range(-1.0..1.0);
                let im = rng.gen_range(-1.0..1.0);
                x_input[k] = Complex::new(re, im);
            }

            // compute correct result (using recursive method as per C++ validate)
            x_correct.copy_from_slice(&x_input);
            correct_fft(&mut x_correct);
            // fft_cooley_tukey_recursive(&mut x_correct);

            // compute test result
            x_test.copy_from_slice(&x_input);
            fft_conjugate(&mut x_test);

            // check correctness
            // C++: abs(real_diff) > 1e-3 || abs(imag_diff) > 1e-3
            let epsilon = 1e-3;
            for k in 0..n {
                let diff_re = (x_correct[k].re - x_test[k].re).abs();
                let diff_im = (x_correct[k].im - x_test[k].im).abs();
                if diff_re > epsilon || diff_im > epsilon {
                    return false;
                }
            }
        }

        true
    }
}

fn main() {
    run::<Context>();
}