use std::{io::Write, iter};

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

const P: u64 = 1_000_000_007;

fn pow(mut base: u64, mut exp: u64) -> u64 {
    let mut result = 1;
    while exp > 0 {
        if exp % 2 == 1 {
            result = result * base % P;
        }
        base = base * base % P;
        exp >>= 1;
    }
    result
}

fn mod_inv(n: u64) -> u64 {
    pow(n, P - 2)
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n_max = 1_000_001;
    let fact = iter::once(1)
        .chain((1..=n_max).scan(1, |acc, x| {
            *acc = *acc * x % P;
            Some(*acc)
        }))
        .collect::<Vec<_>>();
    let inv_fact = fact.iter().map(|&x| mod_inv(x)).collect::<Vec<_>>();
    let comb = |n: usize, k: usize| {
        if n < k {
            0
        } else {
            fact[n] * inv_fact[k] % P * inv_fact[n - k] % P
        }
    };
    let f = |n: usize, k: usize| -> u64 {
        (P + comb(n + 1, k + 1) + comb(n + 1, k - 1) - comb(n - 1, k - 1)) % P
    };

    for _ in 0..input.value() {
        let n = input.value::<usize>();
        let k = input.value::<usize>();
        let x: u64 = input.value();
        if f(n, k) == x {
            writeln!(output, "Correct").unwrap();
        } else {
            writeln!(output, "Incorrect").unwrap();
        }
    }
}
