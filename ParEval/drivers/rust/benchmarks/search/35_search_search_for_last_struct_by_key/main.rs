// main.rs
use pareval_runner::{ParEvalBenchmark, run};
use rayon::prelude::*;
use rand::prelude::*;

// const DRIVER_PROBLEM_SIZE: usize = 1_000_000;

#[derive(Clone, Debug)]
pub struct Book {
    pub title: String,
    pub pages: i32,
}

/* Return the index of the last Book item in the vector books where Book.pages is less than 100.
   Use Rust Rayon to search in parallel.
   Example:

   input: [{title="Green Eggs and Ham", pages=72}, {title="gulliver's travels", pages=362}, {title="Stories of Your Life", pages=54}, {title="Hamilton", pages=818}]
   output: 2
*/
pub fn find_last_short_book(books: &[Book]) -> usize {
    // LLM_OUTPUT_HERE
}

fn correct_find_last_short_book(books: &[Book]) -> usize {
    for (i, book) in books.iter().enumerate().rev() {
        if book.pages < 100 {
            return i;
        }
    }
    books.len()
}

struct SearchContext {
    books: Vec<Book>,
}

impl ParEvalBenchmark for SearchContext {
    fn new() -> Self {
        let driver_problem_size = std::env!("DRIVER_PROBLEM_SIZE").parse().expect("Invalid DRIVER_PROBLEM_SIZE env var");
        SearchContext {
            books: vec![Book { title: String::new(), pages: 0 }; driver_problem_size],
        }
    }

    fn reset(&mut self) {
        // Parallel initialization for speed on large sizes
        self.books.par_iter_mut().for_each(|book| {
            let mut rng = rand::thread_rng();
            // Random string length 5..15
            let len = rng.gen_range(5..15);
            let charset = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
            let title: String = (0..len)
                .map(|_| {
                    let idx = rng.gen_range(0..charset.len());
                    charset[idx] as char
                })
                .collect();
            book.title = title;
            // Random pages 101..1000
            book.pages = rng.gen_range(101..=1000);
        });

        // Force one short book in the first quarter of the vector
        // C++: ctx->pages[rand() % (max - min) + min] = 72; where max = size/4
        let len = self.books.len();
        if len > 0 {
            let mut rng = rand::thread_rng();
            let limit = len / 4;
            let idx = if limit > 0 { rng.gen_range(0..limit) } else { 0 };
            self.books[idx].pages = 72;
        }
    }

    fn compute(&mut self) {
        let idx = find_last_short_book(&self.books);
        std::hint::black_box(idx);
    }

    fn best(&mut self) {
        let idx = correct_find_last_short_book(&self.books);
        std::hint::black_box(idx);
    }

    fn validate(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        const TEST_SIZE: usize = 1024;

        for _ in 0..5 {
            let mut input = Vec::with_capacity(TEST_SIZE);
            for _ in 0..TEST_SIZE {
                input.push(Book {
                    title: "title".to_string(),
                    pages: rng.gen_range(1..=1000),
                });
            }
            // Ensure at least one book < 100 pages
            let idx = rng.gen_range(0..TEST_SIZE);
            input[idx].pages = 72;

            let correct = correct_find_last_short_book(&input);
            let test = find_last_short_book(&input);

            if correct != test {
                return false;
            }
        }
        true
    }
}

fn main() {
    run::<SearchContext>();
}