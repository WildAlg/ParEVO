// main.rs

use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;
use num_complex::Complex;
use std::f64::consts::PI;

// Problem size for the main benchmark. 
// Using 2^20 (approx 1 million elements) which is a standard size for FFT benchmarks
// and ensures bit-reversal logic (which assumes power of 2) works correctly.
// const DRIVER_PROBLEM_SIZE: usize = 1 << 20;
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Helper / Forward Declared Functions
// ============================================================================

/* fft. computes fourier transform in-place
   Translated from baseline.hpp
*/
pub fn fft(x: &mut [Complex<f64>]) {
    let n_len = x.len();
    let mut k = n_len;
    let mut n;
    
    // Using high precision PI from C++ source
    let pi_val = 3.14159265358979323846264338328;
    let theta_t = pi_val / (n_len as f64);
    
    let mut phi_t = Complex::new(theta_t.cos(), -theta_t.sin());
    let mut t_coeff; // Represents T in C++ code

    // DFT / Butterfly operations
    while k > 1 {
        n = k;
        k >>= 1;
        phi_t = phi_t * phi_t;
        t_coeff = Complex::new(1.0, 0.0);
        
        for _l in 0..k {
            for a in (_l..n_len).step_by(n) {
                let b = a + k;
                let t = x[a] - x[b];
                x[a] = x[a] + x[b];
                x[b] = t * t_coeff;
            }
            t_coeff = t_coeff * phi_t;
        }
    }

    // Decimate / Bit Reversal
    // The C++ code uses explicit 32-bit integer bit manipulation.
    // We replicate that logic here.
    let m = (n_len as f64).log2() as u32;
    for a in 0..n_len {
        let mut b = a as u32;
        // Reverse bits
        b = ((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1);
        b = ((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2);
        b = ((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4);
        b = ((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8);
        b = ((b >> 16) | (b << 16)) >> (32 - m);
        
        let b_idx = b as usize;
        if b_idx > a {
            x.swap(a, b_idx);
        }
    }
}

// ============================================================================
//  2. Generated Code Placeholder
// ============================================================================

/* Compute the inverse fourier transform of x in-place.
   Use Rust Rayon to compute in parallel.
   Example:
   
   input: [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
   output: [{0.5,0}, {0.125,0.301777}, {0,-0}, {0.125,0.0517767}, {0,-0}, {0.125,-0.0517767}, {0,-0}, {0.125,-0.301777}]
*/
pub fn ifft(x: &mut [num_complex::Complex<f64>]) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  3. Sequential Baseline (Reference)
// ============================================================================

fn correct_ifft(x: &mut [Complex<f64>]) {
    // conjugate the complex numbers
    for val in x.iter_mut() {
        *val = val.conj();
    }

    // forward fft
    fft(x);

    // conjugate again and scale
    let size = x.len() as f64;
    for val in x.iter_mut() {
        *val = val.conj() / size;
    }
}

fn fill_rand(slice: &mut [f64], min: f64, max: f64) {
    let mut rng = rand::thread_rng();
    for x in slice.iter_mut() {
        *x = rng.gen_range(min..max);
    }
}

// ============================================================================
//  4. Benchmark State Implementation
// ============================================================================

struct FftContext {
    x: Vec<Complex<f64>>,
    real: Vec<f64>,
    imag: Vec<f64>,
}

impl ParEvalBenchmark for FftContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            x: vec![Complex::default(); driver_problem_size],
            real: vec![0.0; driver_problem_size],
            imag: vec![0.0; driver_problem_size],
        }
    }

    fn reset(&mut self) {
        // fillRand(ctx->real, -1.0, 1.0);
        // fillRand(ctx->imag, -1.0, 1.0);
        fill_rand(&mut self.real, -1.0, 1.0);
        fill_rand(&mut self.imag, -1.0, 1.0);

        // Reconstruct x from real and imag parts
        for i in 0..self.x.len() {
            self.x[i] = Complex::new(self.real[i], self.imag[i]);
        }
    }

    fn compute(&mut self) {
        ifft(&mut self.x);
    }

    fn best(&mut self) {
        correct_ifft(&mut self.x);
    }

    fn validate(&mut self) -> bool {
        let mut x_input = vec![Complex::default(); TEST_SIZE];
        let mut real = vec![0.0; TEST_SIZE];
        let mut imag = vec![0.0; TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // set up input
            fill_rand(&mut real, -1.0, 1.0);
            fill_rand(&mut imag, -1.0, 1.0);

            for j in 0..TEST_SIZE {
                x_input[j] = Complex::new(real[j], imag[j]);
            }

            // compute correct result
            let mut correct = x_input.clone();
            correct_ifft(&mut correct);

            // compute test result
            let mut test = x_input.clone();
            ifft(&mut test);

            // validation check
            for j in 0..correct.len() {
                let diff_re = (correct[j].re - test[j].re).abs();
                let diff_im = (correct[j].im - test[j].im).abs();
                if diff_re > 1e-4 || diff_im > 1e-4 {
                    return false;
                }
            }
        }
        true
    }
}

// ============================================================================
//  5. Entry Point
// ============================================================================

fn main() {
    run::<FftContext>();
}