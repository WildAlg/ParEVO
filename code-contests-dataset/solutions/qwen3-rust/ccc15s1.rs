use std::io::{self, Read};
use rayon::prelude::*;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    
    let lines: Vec<&str> = input.lines().collect();
    let k: usize = lines[0].parse().unwrap();
    
    let mut stack = Vec::with_capacity(k);
    
    for i in 1..=k {
        let num: i32 = lines[i].parse().unwrap();
        if num == 0 {
            stack.pop();
        } else {
            stack.push(num);
        }
    }
    
    let sum = stack.into_par_iter().sum::<i32>();
    println!("{}", sum);
}