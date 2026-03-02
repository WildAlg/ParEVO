// main.rs
use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;
use num_complex::Complex;
use std::f64::consts::PI;

// const DRIVER_PROBLEM_SIZE: usize = 1 << 20; // 2^20 = 1,048,576
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Compute the fourier transform of x. Store the result in output.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
   output: [{4,0}, {1,-2.42421}, {0,0}, {1,-0.414214}, {0,0}, {1,0.414214}, {0,0}, {1,2.41421}]
*/
pub fn fft(x: &[num_complex::Complex<f64>], output: &mut [num_complex::Complex<f64>]) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

/// Iterative FFT implementation corresponding to `correctFft` in baseline.hpp
fn correct_fft(x: &[Complex<f64>], output: &mut [Complex<f64>]) {
    assert_eq!(x.len(), output.len());
    output.copy_from_slice(x);

    let n_total = output.len();
    let mut k = n_total;
    let mut n;
    
    // In the C++ code: thetaT = PI / N (not 2*PI/N, note the logic inside loop doubles the angle)
    let theta_t = PI / (n_total as f64);
    let mut phi_t = Complex::new(theta_t.cos(), -theta_t.sin());
    let mut t_coeff;

    while k > 1 {
        n = k;
        k >>= 1;
        
        phi_t = phi_t * phi_t;
        t_coeff = Complex::new(1.0, 0.0);

        for l in 0..k {
            for a in (l..n_total).step_by(n) {
                let b = a + k;
                let t = output[a] - output[b];
                output[a] = output[a] + output[b];
                output[b] = t * t_coeff;
            }
            t_coeff = t_coeff * phi_t;
        }
    }

    // Decimate (Bit Reversal)
    let m = (n_total as f64).log2() as u32;
    for a in 0..n_total {
        let mut b = a as u32;
        // The C++ code manually reverses bits for a 32-bit integer.
        // Rust's reverse_bits() reverses all 32 bits.
        // Shifting right by (32 - m) aligns the reversed 'm' bits to the least significant positions.
        b = b.reverse_bits();
        b >>= 32 - m;
        let b = b as usize;

        if b > a {
            output.swap(a, b);
        }
    }
}

/// Recursive FFT implementation corresponding to `fftCooleyTookey` in baseline.hpp
/// Used for validation.
fn fft_cooley_tukey(x: &mut [Complex<f64>]) {
    let n = x.len();
    if n <= 1 {
        return;
    }

    // Divide
    // Note: This allocation pattern is inefficient but matches the C++ validation code structure.
    let mut even = Vec::with_capacity(n / 2);
    let mut odd = Vec::with_capacity(n / 2);

    for j in 0..(n / 2) {
        even.push(x[j * 2]);
        odd.push(x[j * 2 + 1]);
    }

    // Conquer
    fft_cooley_tukey(&mut even);
    fft_cooley_tukey(&mut odd);

    // Combine
    for k in 0..(n / 2) {
        let angle = -2.0 * PI * (k as f64) / (n as f64);
        let t = Complex::from_polar(1.0, angle) * odd[k];
        x[k] = even[k] + t;
        x[k + n / 2] = even[k] - t;
    }
}

// ============================================================================
//  3. Benchmark State Implementation
// ============================================================================

struct FftContext {
    x: Vec<Complex<f64>>,
    output: Vec<Complex<f64>>,
    real: Vec<f64>,
    imag: Vec<f64>,
}

impl ParEvalBenchmark for FftContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            x: vec![Complex::new(0.0, 0.0); driver_problem_size],
            output: vec![Complex::new(0.0, 0.0); driver_problem_size],
            real: vec![0.0; driver_problem_size],
            imag: vec![0.0; driver_problem_size],
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        // fillRand(..., -1.0, 1.0)
        for v in self.real.iter_mut() {
            *v = rng.gen_range(-1.0..1.0);
        }
        for v in self.imag.iter_mut() {
            *v = rng.gen_range(-1.0..1.0);
        }

        for i in 0..self.x.len() {
            self.x[i] = Complex::new(self.real[i], self.imag[i]);
        }
    }

    fn compute(&mut self) {
        fft(&self.x, &mut self.output);
    }

    fn best(&mut self) {
        correct_fft(&self.x, &mut self.output);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();

        // Buffers for validation
        let mut x_in = vec![Complex::new(0.0, 0.0); TEST_SIZE];
        let mut correct = vec![Complex::new(0.0, 0.0); TEST_SIZE];
        let mut test_out = vec![Complex::new(0.0, 0.0); TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // Set up input
            for j in 0..TEST_SIZE {
                let r = rng.gen_range(-1.0..1.0);
                let im = rng.gen_range(-1.0..1.0);
                x_in[j] = Complex::new(r, im);
            }

            // Compute correct result
            correct.copy_from_slice(&x_in);
            fft_cooley_tukey(&mut correct);

            // Compute test result
            fft(&x_in, &mut test_out);

            // Compare
            // C++ tolerance: 1e-4
            for k in 0..TEST_SIZE {
                let diff_r = (correct[k].re - test_out[k].re).abs();
                let diff_i = (correct[k].im - test_out[k].im).abs();

                if diff_r > 1e-4 || diff_i > 1e-4 {
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
    run::<FftContext>();
}