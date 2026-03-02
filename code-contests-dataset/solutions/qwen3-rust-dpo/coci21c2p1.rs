use std::io::{self, Read};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    
    let lines: Vec<&str> = input.lines().collect();
    
    let mut section_counter = 0;
    let mut subsection_counters = vec![0i32; 101]; // 1-indexed, [0] unused
    let mut subsubsection_counters = vec![vec![0i32; 101]; 101]; // [section][subsection]
    
    for line in lines {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 { continue; }
        
        let command = parts[0];
        let title = parts[1];
        
        match command {
            "section" => {
                section_counter += 1;
                subsection_counters[section_counter as usize] = 0;
                subsubsection_counters[section_counter as usize].fill(0);
                println!("{} {}", section_counter, title);
            },
            "subsection" => {
                let section_id = section_counter;
                subsection_counters[section_id as usize] += 1;
                let subsection_id = subsection_counters[section_id as usize];
                println!("{}.{} {}", section_id, subsection_id, title);
            },
            "subsubsection" => {
                let section_id = section_counter;
                let subsection_id = subsection_counters[section_id as usize];
                subsubsection_counters[section_id as usize][subsection_id as usize] += 1;
                let subsubsection_id = subsubsection_counters[section_id as usize][subsection_id as usize];
                println!("{}.{}.{} {}", section_id, subsection_id, subsubsection_id, title);
            },
            _ => {}
        }
    }
}