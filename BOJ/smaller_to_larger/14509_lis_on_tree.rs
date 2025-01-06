use std::{collections::BTreeMap, io::Write};

mod simple_io {
    use std::string::*;

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

// Longest increasing subforest
#[derive(Default)]
struct LIF {
    freq: BTreeMap<u32, u32>,
    value: u32,
}

impl LIF {
    fn singleton(value: u32) -> Self {
        Self {
            freq: BTreeMap::new(),
            value,
        }
    }

    fn pull_from(&mut self, mut other: Self) {
        if self.freq.len() < other.freq.len() {
            std::mem::swap(&mut self.freq, &mut other.freq);
        }
        for (v, f) in other.freq {
            *self.freq.entry(v).or_default() += f;
        }
    }

    fn finalize(&mut self) {
        if let Some((&old, f)) = self.freq.range_mut(self.value..).next() {
            if self.value == old {
                return;
            }
            *f -= 1;
            if *f == 0 {
                self.freq.remove(&old);
            }
        }
        *self.freq.entry(self.value).or_default() += 1;
    }
}

pub fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();

    let mut parent = Vec::with_capacity(n);
    let mut dp = Vec::with_capacity(n);
    for _ in 0..n {
        dp.push(LIF::singleton(input.value::<u32>()));
        parent.push(input.value::<u32>().saturating_sub(1));
    }

    for u in (1..n).rev() {
        let p = parent[u as usize] as usize;

        dp[u].finalize();
        let dp_u = std::mem::take(&mut dp[u]);
        dp[p].pull_from(dp_u);
    }
    dp[0].finalize();

    let ans = dp[0].freq.values().sum::<u32>();
    writeln!(output, "{}", ans).unwrap();
}
