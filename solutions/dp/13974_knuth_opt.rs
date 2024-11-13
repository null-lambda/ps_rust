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

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    for _ in 0..input.value() {
        let n: usize = input.value();
        let xs: Vec<u32> = iter::once(0).chain((0..n).map(|_| input.value())).collect();
        let acc: Vec<u32> = xs
            .iter()
            .scan(0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();

        // Knuth optimization
        // dp[s..=e] = max_{i in [s..=e]} ( dp[s..=i] + dp[i+1..=e] ) + acc[e] - acc[s-1]
        let mut dp = vec![vec![0u64; n + 1]; n + 1];
        let mut min_k = vec![vec![0; n + 1]; n + 1];

        for i in 1..=n {
            dp[i][i] = 0;
            min_k[i][i] = i;
        }
        for i in 1..=n - 1 {
            dp[i][i + 1] = (xs[i] + xs[i + 1]) as u64;
            min_k[i][i + 1] = i;
        }

        for gap in 2..=n - 1 {
            for i in 1..=n - gap {
                let j = i + gap;
                let (cost, k) = (min_k[i][j - 1]..=min_k[i + 1][j])
                    .map(|k| (dp[i][k] + dp[k + 1][j], k))
                    .min_by_key(|&(cost, _)| cost)
                    .unwrap();
                dp[i][j] = cost + (acc[j] - acc[i - 1]) as u64;
                min_k[i][j] = k;
            }
        }
        let ans = dp[1][n];
        writeln!(output, "{}", ans).unwrap();
    }
}
