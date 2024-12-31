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

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let k: i32 = input.value();
    let hs: Vec<i32> = (0..n).map(|_| input.value()).collect();

    let mut dp = vec![vec![0; n]; 1 << n];
    for i in 0..n {
        dp[1 << i][i] = 1;
    }

    for mask in 1..1 << n {
        for i_last in 0..n {
            if (mask >> i_last) & 1 == 0 {
                continue;
            }

            for j in 0..n {
                if (mask >> j) & 1 == 1 || (hs[i_last] - hs[j]).abs() <= k {
                    continue;
                }
                dp[mask | 1 << j][j] += dp[mask][i_last];
            }
        }
    }

    let ans = (0..n).map(|i| dp[(1 << n) - 1][i]).sum::<u64>();
    writeln!(output, "{}", ans).unwrap();
}
