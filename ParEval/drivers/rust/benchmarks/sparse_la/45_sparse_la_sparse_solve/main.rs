// main.rs
use pareval_runner::{ParEvalBenchmark, run};
use rand::prelude::*;
use rayon::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 512;
const SPARSE_LA_SPARSITY: f64 = 0.05;
const TEST_SIZE: usize = 128;
const MAX_VALIDATION_ATTEMPTS: usize = 5;

#[derive(Clone, Copy, Debug)]
pub struct COOElement {
   pub row: usize,
   pub column: usize,
   pub value: f64,
}

/* Solve the sparse linear system Ax=b for x.
   A is a sparse NxN matrix in COO format. x and b are dense vectors with N elements.
   Use Rust Rayon to compute in parallel.
   Example:
   
   input: A=[{0,0,1}, {0,1,1}, {1,1,-2}] b=[1,4]
   output: x=[3,-2]
*/
pub fn solveLinearSystem(A: &[COOElement], b: &[f64], x: &mut [f64], N: usize) {
    // LLM_OUTPUT_HERE
}

// ============================================================================
//  Sequential Baseline
// ============================================================================

fn correct_solve_linear_system(A: &[COOElement], b: &[f64], x: &mut [f64], N: usize) {
    // Dense matrix conversion
    let mut matrix = vec![vec![0.0; N]; N];
    let mut b_copy = b.to_vec();

    // Fill the matrix with the values from A
    for element in A {
        matrix[element.row][element.column] = element.value;
    }

    // Initialize x
    x.fill(0.0);

    // Perform Gaussian elimination
    for i in 0..N {
        // Find pivot
        let mut max_el = matrix[i][i].abs();
        let mut max_row = i;
        for k in (i + 1)..N {
            if matrix[k][i].abs() > max_el {
                max_el = matrix[k][i].abs();
                max_row = k;
            }
        }

        // Swap maximum row with current row (column by column)
        // We iterate k from i to n to match the C++ logic
        for k in i..N {
            let temp = matrix[max_row][k];
            matrix[max_row][k] = matrix[i][k];
            matrix[i][k] = temp;
        }
        b_copy.swap(max_row, i);

        // Make all rows below this one 0 in the current column
        for k in (i + 1)..N {
            if matrix[i][i].abs() < 1e-15 { continue; } // Avoid div by zero
            let c = -matrix[k][i] / matrix[i][i];

            for j in i..N {
                if i == j {
                    matrix[k][j] = 0.0;
                } else {
                    matrix[k][j] += c * matrix[i][j];
                }
            }
            b_copy[k] += c * b_copy[i];
        }
    }

    // Solve equation Ax=b for an upper triangular matrix A
    for i in (0..N).rev() {
        if matrix[i][i].abs() > 1e-15 {
            x[i] = b_copy[i] / matrix[i][i];
        } else {
            x[i] = 0.0;
        }
        for k in (0..i).rev() {
            b_copy[k] -= matrix[k][i] * x[i];
        }
    }
}

// ============================================================================
//  Helpers
// ============================================================================

fn fequal(a: &[f64], b: &[f64], epsilon: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= epsilon)
}

fn create_random_linear_system(
    n: usize,
    sparsity: f64,
    a_out: &mut Vec<COOElement>,
    b_out: &mut Vec<f64>,
    x_out: &mut Vec<f64>,
) {
    let mut rng = rand::thread_rng();
    let n_vals = ((n * n) as f64 * sparsity) as usize;

    // A setup
    a_out.clear();
    a_out.reserve(n_vals);
    for _ in 0..n_vals {
        a_out.push(COOElement {
            row: rng.gen_range(0..n),
            column: rng.gen_range(0..n),
            value: rng.gen_range(-10.0..10.0),
        });
    }
    
    // Sort COOElements
    a_out.sort_by(|a, b| a.row.cmp(&b.row).then(a.column.cmp(&b.column)));

    // Generate x target to compute valid b
    let mut x_target = vec![0.0; n];
    for val in x_target.iter_mut() {
        *val = rng.gen_range(-10.0..10.0);
    }

    // Compute b = A * x_target
    b_out.resize(n, 0.0);
    b_out.fill(0.0);
    for elem in a_out.iter() {
        b_out[elem.row] += elem.value * x_target[elem.column];
    }

    // Reset output x
    x_out.resize(n, 0.0);
    x_out.fill(0.0);
}

// ============================================================================
//  Benchmark Struct
// ============================================================================

struct Context {
    a: Vec<COOElement>,
    b: Vec<f64>,
    x: Vec<f64>,
    n: usize,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Self {
            a: Vec::new(),
            b: Vec::new(),
            x: Vec::new(),
            n: driver_problem_size,
        }
    }

    fn reset(&mut self) {
        create_random_linear_system(self.n, SPARSE_LA_SPARSITY, &mut self.a, &mut self.b, &mut self.x);
    }

    fn compute(&mut self) {
        solveLinearSystem(&self.a, &self.b, &mut self.x, self.n);
    }

    fn best(&mut self) {
        correct_solve_linear_system(&self.a, &self.b, &mut self.x, self.n);
    }

    fn validate(&mut self) -> bool {
        let n = TEST_SIZE;
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut x_correct = Vec::new();
        let mut x_test = Vec::new();
        let mut x_temp = Vec::new(); // Dummy buffer

        for _ in 0..MAX_VALIDATION_ATTEMPTS {
            create_random_linear_system(n, SPARSE_LA_SPARSITY, &mut a, &mut b, &mut x_temp);
            
            // Prepare outputs
            x_correct.resize(n, 0.0);
            x_correct.fill(0.0);
            x_test.resize(n, 0.0);
            x_test.fill(0.0);

            // Compute correct
            correct_solve_linear_system(&a, &b, &mut x_correct, n);

            // Compute test
            solveLinearSystem(&a, &b, &mut x_test, n);

            if !fequal(&x_correct, &x_test, 1e-3) {
                return false;
            }
        }
        true
    }
}

fn main() {
    run::<Context>();
}