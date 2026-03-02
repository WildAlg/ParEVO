use std::io::{self, Read};
use rayon::prelude::*;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    
    let mut lines = input.lines();
    let burger_choice = lines.next().unwrap().parse::<i32>().unwrap();
    let side_choice = lines.next().unwrap().parse::<i32>().unwrap();
    let drink_choice = lines.next().unwrap().parse::<i32>().unwrap();
    let dessert_choice = lines.next().unwrap().parse::<i32>().unwrap();
    
    let burger_calories = [0, 461, 431, 420, 0];
    let side_calories = [0, 100, 57, 70, 0];
    let drink_calories = [0, 130, 160, 118, 0];
    let dessert_calories = [0, 167, 266, 75, 0];
    
    let total = burger_calories[burger_choice as usize] +
                side_calories[side_choice as usize] +
                drink_calories[drink_choice as usize] +
                dessert_calories[dessert_choice as usize];
                
    println!("Your total Calorie count is {}.", total);
}