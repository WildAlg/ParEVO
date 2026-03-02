use std::io::{self, Read};
use rayon::prelude::*;

#[derive(Clone)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);
        
        if root_x == root_y {
            return false;
        }
        
        if self.rank[root_x] < self.rank[root_y] {
            self.parent[root_x] = root_y;
        } else if self.rank[root_x] > self.rank[root_y] {
            self.parent[root_y] = root_x;
        } else {
            self.parent[root_y] = root_x;
            self.rank[root_x] += 1;
        }
        true
    }
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    
    let lines: Vec<&str> = input.lines().collect();
    let dims: Vec<usize> = lines[0].split_whitespace().map(|x| x.parse().unwrap()).collect();
    let n = dims[0];
    let m = dims[1];
    
    let grid: Vec<Vec<char>> = lines[1..=n].iter().map(|line| line.chars().collect()).collect();
    
    let mut uf = UnionFind::new(n * m);
    
    // For each cell, connect it to the cell it points to
    for i in 0..n {
        for j in 0..m {
            let current_idx = i * m + j;
            let dir = grid[i][j];
            
            let neighbor_idx = match dir {
                'N' => {
                    if i == 0 { continue; }
                    (i - 1) * m + j
                },
                'S' => {
                    if i == n - 1 { continue; }
                    (i + 1) * m + j
                },
                'E' => {
                    if j == m - 1 { continue; }
                    i * m + (j + 1)
                },
                'W' => {
                    if j == 0 { continue; }
                    i * m + (j - 1)
                },
                _ => continue,
            };
            
            uf.union(current_idx, neighbor_idx);
        }
    }
    
    // Count unique components
    let mut components = std::collections::HashSet::new();
    for i in 0..(n * m) {
        components.insert(uf.find(i));
    }
    
    println!("{}", components.len());
}