use std::time::Instant;
use clap::Parser;

/// The interface every benchmark (main.rs) must implement.
/// This mirrors the extern functions in cpu.cc.
pub trait ParEvalBenchmark {
    fn new() -> Self;
    fn reset(&mut self);
    fn compute(&mut self);  // The parallel solution
    fn best(&mut self);     // The sequential baseline
    fn validate(&mut self) -> bool;
}

#[derive(Parser)]
struct Args {
    #[arg(default_value_t = 1)]
    num_threads: usize,
}

pub fn run<B: ParEvalBenchmark>() {
    let args = Args::parse();

    // Set Rayon threads
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build_global()
        .unwrap();

    // Initialize
    let mut ctx = B::new();

    // Validate
    let is_valid = ctx.validate();
    println!("Validation: {}", if is_valid { "PASS" } else { "FAIL" });
    if !is_valid { return; }

    const NITER: u32 = 5;

    // Benchmark Parallel
    let mut total_time = 0.0;
    for _ in 0..NITER {
        ctx.reset(); // Reset state before run
        let start = Instant::now();
        ctx.compute();
        total_time += start.elapsed().as_secs_f64();
    }
    println!("Time: {:.15}", total_time / NITER as f64);

    // Benchmark Baseline
    let mut total_time_seq = 0.0;
    for _ in 0..NITER {
        ctx.reset();
        let start = Instant::now();
        ctx.best();
        total_time_seq += start.elapsed().as_secs_f64();
    }
    println!("BestSequential: {:.15}", total_time_seq / NITER as f64);
}