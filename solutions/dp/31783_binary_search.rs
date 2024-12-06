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

    let n: usize = input.value();
    let hs: Vec<i32> = (0..n).map(|_| input.value()).collect();
    let a: Vec<i64> = iter::once(0).chain((1..n).map(|_| input.value())).collect();
    let b: Vec<i32> = iter::once(0).chain((1..n).map(|_| input.value())).collect();

    let mut dp = vec![0; n];
    let mut acc = vec![0; n];
    for i in 1..n {
        let h_bound = hs[i] + b[i];
        let j = hs[..i].partition_point(|&h| h >= h_bound);
        dp[i] = acc[i - 1];
        if j > 0 && hs[j - 1] >= h_bound {
            dp[i] = dp[i].max(acc[j - 1] + a[i]);
        }

        acc[i] = acc[i - 1].max(dp[i]);
    }

    let ans = dp[n - 1];
    writeln!(output, "{}", ans).unwrap();
}
