use std::{
    collections::{HashMap, HashSet},
    io::Write,
};

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

    pub fn stdin<'a>() -> InputAtOnce<'a> {
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
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let k: i32 = input.value();
    let mut dp = vec![];
    dp.push(HashSet::new());
    dp.push([k].into());
    for l in 2..=8 {
        let mut combined = HashSet::new();
        combined.insert(k * (10i32.pow(l as u32) - 1) / 9);
        for i in 1..l {
            let xs = &dp[i];
            let ys = &dp[l - i];

            for &x in xs {
                for &y in ys {
                    combined.insert(x + y);
                    combined.insert(x - y);
                    combined.insert(x * y);
                    if y != 0 {
                        combined.insert(x / y);
                    }
                }
            }
        }
        dp.push(combined);
    }

    let mut min_len = HashMap::new();
    for l in 1..=8 {
        for x in &dp[l] {
            min_len.entry(*x).or_insert(l);
        }
    }

    for _ in 0..input.value() {
        let x = input.value();
        if let Some(&l) = min_len.get(&x) {
            writeln!(output, "{}", l).unwrap();
        } else {
            writeln!(output, "NO").unwrap();
        }
    }
}
