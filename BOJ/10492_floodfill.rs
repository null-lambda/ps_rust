use core::f64;
use iter::product;
use std::{
    collections::LinkedList,
    fmt::Display,
    io::{BufRead, Write},
    ops::{Index, IndexMut},
};

mod simple_io {
    pub struct InputAtOnce(std::str::SplitAsciiWhitespace<'static>);

    impl InputAtOnce {
        pub fn token(&mut self) -> &str {
            self.0.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(buf.into_boxed_str());
        InputAtOnce(buf.split_ascii_whitespace())
    }

    pub fn stdout_buf() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

struct Window {
    id: usize,
    bbox_original: (i32, i32, i32, i32),
    zoomed: bool,
}

impl Window {
    fn new(id: usize, bbox: (i32, i32, i32, i32)) -> Self {
        Self {
            id,
            bbox_original: bbox,
            zoomed: false,
        }
    }

    fn bbox(&self) -> (i32, i32, i32, i32) {
        if self.zoomed {
            (0, 0, 1024, 1024)
        } else {
            self.bbox_original
        }
    }
}

fn main() {
    let lines = std::io::stdin().lock().lines().flatten();
    let mut output = simple_io::stdout_buf();

    let mut id = 0;
    let mut windows: LinkedList<Window> = Default::default();
    let mut selected: Option<usize> = None;
    let mut mouse_pressed = false;

    for line in lines {
        let mut tokens = line.split_ascii_whitespace();
        let head = tokens.next().unwrap();
        let rest: Vec<i32> = tokens.into_iter().map(|x| x.parse().unwrap()).collect();

        match head {
            "ZZ" => break,
            "CR" => {
                windows.push_back(Window::new(id, (rest[0], rest[1], rest[2], rest[3])));
                id += 1;
            }

            _ => panic!(),
        }
    }
}
