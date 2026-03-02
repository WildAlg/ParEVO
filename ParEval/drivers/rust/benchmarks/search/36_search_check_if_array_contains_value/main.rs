use pareval_runner::{ParEvalBenchmark, run};
use rand::prelude::*;
use rayon::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1024 * 1024;

/* Return true if the vector x contains the value `target`. Return false otherwise.
   Use Rust Rayon to search in parallel.
   Examples:

   input: x=[1, 8, 2, 6, 4, 6], target=3
   output: false
   
   input: x=[1, 8, 2, 6, 4, 6], target=8
   output: true
*/
pub fn contains(x: &[i32], target: i32) -> bool {
    // LLM_OUTPUT_HERE
}

fn correct_contains(x: &[i32], target: i32) -> bool {
    x.contains(&target)
}

struct Context {
    x: Vec<i32>,
    target: i32,
}

impl ParEvalBenchmark for Context {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        Context {
            x: vec![0; driver_problem_size],
            target: 0,
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        // fillRand(ctx->x, -50, 50);
        for val in self.x.iter_mut() {
            *val = rng.gen_range(-50..50);
        }
        // ctx->target = (rand() % 200) - 100;
        self.target = rng.gen_range(-100..100);
    }

    fn compute(&mut self) {
        let _ = contains(&self.x, self.target);
    }

    fn best(&mut self) {
        let _ = correct_contains(&self.x, self.target);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        let num_tries = 10;
        
        for i in 0..num_tries {
            let mut input = vec![0; 1024];
            for val in input.iter_mut() {
                *val = rng.gen_range(-50..50);
            }
            
            let target = if i == 1 {
                input[rng.gen_range(0..input.len())]
            } else {
                rng.gen_range(-100..100)
            };
            
            let correct = correct_contains(&input, target);
            let test = contains(&input, target);
            
            if correct != test {
                return false;
            }
        }
        true
    }
}

fn main() {
    run::<Context>();
}