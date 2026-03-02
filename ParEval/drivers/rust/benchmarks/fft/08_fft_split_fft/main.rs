// main.rs
use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;
use num_complex::Complex;
use std::f64::consts::PI;

// const DRIVER_PROBLEM_SIZE: usize = 1 << 20; // 2^20 approx 1 million elements
const TEST_SIZE: usize = 1024;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

// ============================================================================
//  1. Generated Code Placeholder
// ============================================================================

/* Compute the fourier transform of x. Store real part of results in r and imaginary in i.
   Use Rust Rayon to compute in parallel.
   Example:

   input: [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
   output: r: [4, 1, 0, 1, 0, 1, 0, 1] i: [0, -2.41421, 0, -0.414214, 0, 0.414214, 0, 2.41421]
*/
pub fn fft(x: &[num_complex::Complex<f64>], r: &mut [f64], i: &mut [f64]) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  2. Sequential Baseline (Reference)
// ============================================================================

// Recursive Cooley-Tukey implementation for validation
fn fft_cooley_tukey(x: &mut [Complex<f64>]) {
    let n = x.len();
    if n <= 1 {
        return;
    }

    // Divide
    let half_n = n / 2;
    let mut even = Vec::with_capacity(half_n);
    let mut odd = Vec::with_capacity(half_n);

    for j in 0..half_n {
        even.push(x[j * 2]);
        odd.push(x[j * 2 + 1]);
    }

    // Conquer
    fft_cooley_tukey(&mut even);
    fft_cooley_tukey(&mut odd);

    // Combine
    for k in 0..half_n {
        let theta = -2.0 * PI * (k as f64) / (n as f64);
        let t = Complex::from_polar(1.0, theta) * odd[k];
        x[k] = even[k] + t;
        x[k + half_n] = even[k] - t;
    }
}

// Iterative implementation corresponding to `correctFft` in baseline.hpp
fn correct_fft(x: &[Complex<f64>], r: &mut [f64], i: &mut [f64]) {
    let mut x_copy = x.to_vec();
    let n_total = x_copy.len();
    let mut k = n_total;
    
    // DFT (Iterative Butterfly / Stockham-like structure before bit reversal?)
    // This matches the logic in baseline.hpp
    let theta_t = PI / (n_total as f64);
    let mut phi_t = Complex::new(theta_t.cos(), -theta_t.sin());
    
    while k > 1 {
        let n_curr = k;
        k >>= 1;
        phi_t = phi_t * phi_t;
        let mut t_coeff = Complex::new(1.0, 0.0);
        
        for _l in 0..k {
            for a in (_l..n_total).step_by(n_curr) {
                let b = a + k;
                let t = x_copy[a] - x_copy[b];
                x_copy[a] = x_copy[a] + x_copy[b];
                x_copy[b] = t * t_coeff;
            }
            t_coeff = t_coeff * phi_t;
        }
    }

    // Decimate (Bit Reversal)
    let m = (n_total as f64).log2() as u32;
    for a in 0..n_total {
        let mut b = a as u32;
        // Reverse bits (32-bit integer logic)
        b = ((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1);
        b = ((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2);
        b = ((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4);
        b = ((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8);
        // Shift down to correct width
        b = ((b >> 16) | (b << 16)) >> (32 - m);
        
        let b_idx = b as usize;
        if b_idx > a {
            x_copy.swap(a, b_idx);
        }
    }

    // Split into real and imaginary parts
    for j in 0..n_total {
        r[j] = x_copy[j].re;
        i[j] = x_copy[j].im;
    }
}

// Helpers
fn fill_rand(slice: &mut [f64], min: f64, max: f64) {
    let mut rng = rand::thread_rng();
    for x in slice.iter_mut() {
        *x = rng.gen_range(min..max);
    }
}

fn fequal(a: &[f64], b: &[f64], epsilon: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= epsilon)
}

// ============================================================================
//  3. Benchmark State Implementation
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
        // Re-use helper logic: fillRand(ctx->real/imag, -1.0, 1.0)
        let mut rng = rand::thread_rng();
        // The C++ logic fills `real` and `imag` vectors first, then populates `x`.
        // However, `real` and `imag` in the context are also output buffers.
        // We will use temporary vectors or just fill `real`/`imag` buffers temporarily to init `x`, 
        // then clear them or leave them as they will be overwritten by output.
        // The C++ reset fills ctx->real and ctx->imag, then sets ctx->x.
        
        fill_rand(&mut self.real, -1.0, 1.0);
        fill_rand(&mut self.imag, -1.0, 1.0);

        for i in 0..self.x.len() {
            self.x[i] = Complex::new(self.real[i], self.imag[i]);
        }
    }

    fn compute(&mut self) {
        fft(&self.x, &mut self.real, &mut self.imag);
    }

    fn best(&mut self) {
        correct_fft(&self.x, &mut self.real, &mut self.imag);
    }

    fn validate(&mut self) -> bool {
        let mut real = vec![0.0; TEST_SIZE];
        let mut imag = vec![0.0; TEST_SIZE];
        let mut correct_real = vec![0.0; TEST_SIZE];
        let mut correct_imag = vec![0.0; TEST_SIZE];
        let mut test_real = vec![0.0; TEST_SIZE];
        let mut test_imag = vec![0.0; TEST_SIZE];
        let mut x = vec![Complex::default(); TEST_SIZE];

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            // Set up input
            fill_rand(&mut real, -1.0, 1.0);
            fill_rand(&mut imag, -1.0, 1.0);

            for j in 0..TEST_SIZE {
                x[j] = Complex::new(real[j], imag[j]);
            }

            // Compute correct result
            // C++ uses fftCooleyTookey on a copy of x, then extracts real/imag
            let mut x_copy = x.clone();
            fft_cooley_tukey(&mut x_copy);
            for j in 0..TEST_SIZE {
                correct_real[j] = x_copy[j].re;
                correct_imag[j] = x_copy[j].im;
            }

            // Compute test result
            // Note: The signature for fft takes x as input (const ref in C++, slice in Rust)
            // and writes to buffers.
            fft(&x, &mut test_real, &mut test_imag);

            if !fequal(&correct_real, &test_real, 1e-4) || !fequal(&correct_imag, &test_imag, 1e-4) {
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
    run::<FftContext>();
}