use std::{cmp::Reverse, collections::BTreeSet, io::Write, iter};

mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let k: usize = input.value();

    let mut words: Vec<u32> = (0..n)
        .map(|_| {
            input
                .token()
                .bytes()
                .fold(0, |acc, b| acc | (1 << (b - b'a')))
        })
        .collect();
    words.sort_unstable_by_key(|w| 32 - w.count_ones());

    let ans = (0u32..1 << 26)
        .filter(|i| i.count_ones() == k as u32)
        .map(|i| words.iter().filter(|&&w| i & w == w).count())
        .max()
        .unwrap();
    writeln!(output, "{}", ans).unwrap();
}
