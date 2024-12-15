use std::io::Write;

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

fn test_pair(lhs: u8, rhs: u8) -> bool {
    matches!(
        (lhs, rhs),
        (b'[' | b'*', b']' | b'*') | (b'(' | b'*', b')' | b'*') | (b'{' | b'*', b'}' | b'*')
    )
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for _ in 0..input.value() {
        let s = input.token().as_bytes();

        let n = s.len();
        let mut dp = vec![vec![0; n + 1]; n + 1];
        for i in 0..n + 1 {
            dp[i][i] = 0;
        }
        for i in 0..n {
            dp[i][i + 1] = 1;
        }

        for gap in 2..=n {
            for i in 0..=n - gap {
                let j = i + gap;
                dp[i][j] = dp[i + 1][j] + 1;
                for k in i + 1..j {
                    if test_pair(s[i], s[k]) {
                        dp[i][j] = dp[i][j].min(dp[i + 1][k] + dp[k + 1][j]);
                    }
                }
            }
        }

        let ans = dp[0][n];
        writeln!(output, "{}", ans).unwrap();
    }
}
