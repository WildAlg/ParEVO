use std::io::{self, BufRead};
use rayon::prelude::*;

#[derive(Clone)]
struct SectionCounts {
    section: usize,
    subsection: usize,
    subsubsection: usize,
}

impl SectionCounts {
    fn new() -> Self {
        SectionCounts {
            section: 0,
            subsection: 0,
            subsubsection: 0,
        }
    }
    
    fn next_section(&mut self) -> usize {
        self.section += 1;
        self.subsection = 0;
        self.subsubsection = 0;
        self.section
    }
    
    fn next_subsection(&mut self) -> usize {
        self.subsection += 1;
        self.subsubsection = 0;
        self.subsection
    }
    
    fn next_subsubsection(&mut self) -> usize {
        self.subsubsection += 1;
        self.subsubsection
    }
}

fn main() {
    let stdin = io::stdin();
    let lines: Vec<String> = stdin.lock().lines().map(|l| l.unwrap()).collect();
    
    let n = lines[0].parse::<usize>().unwrap();
    let mut counts = SectionCounts::new();
    let mut outputs = Vec::with_capacity(n);
    
    for i in 1..=n {
        let parts: Vec<&str> = lines[i].split_whitespace().collect();
        let command = parts[0];
        let title = parts[1];
        
        let output = match command {
            "section" => {
                let section_num = counts.next_section();
                format!("{} {}", section_num, title)
            }
            "subsection" => {
                let subsection_num = counts.next_subsection();
                format!("{}.{} {}", counts.section, subsection_num, title)
            }
            "subsubsection" => {
                let subsubsection_num = counts.next_subsubsection();
                format!("{}.{}.{} {}", counts.section, counts.subsection, subsubsection_num, title)
            }
            _ => panic!("Invalid command"),
        };
        
        outputs.push(output);
    }
    
    for output in outputs {
        println!("{}", output);
    }
}