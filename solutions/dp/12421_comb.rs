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

    let a_bound = 30;
    let mut comb = vec![vec![0.0f64; a_bound + 1]; a_bound + 1];
    for a in 0..=a_bound {
        comb[a][0] = 1.0;
        for b in 1..=a {
            comb[a][b] = comb[a - 1][b - 1] + comb[a - 1][b];
        }
    }

    let binom = |a: usize, b: usize, p: f64| {
        debug_assert!(b <= a);

        if a == 0 {
            return 1.0;
        }
        comb[a][b] * p.powi(b as i32) * (1.0 - p).powi((a - b) as i32)
    };

    for i_tc in 1..=input.value() {
        let k_bound: usize = input.value();
        let n_bound: usize = input.value();
        let m_bound: usize = input.value();
        let h: usize = input.value();

        // dp[k][m][n]
        let mut dp = vec![vec![0.0; n_bound + 1]; m_bound + 1];
        dp[0][0] = 1.0;
        for k in 0..k_bound {
            let prev = dp;

            dp = vec![vec![0.0; n_bound + 1]; m_bound + 1];

            let k_remained = k_bound - k;
            let base = if k < h { 2 } else { 1 };

            let p = 1.0 / k_remained as f64;
            let mut n_trans = vec![vec![0.0; n_bound + 1]; n_bound + 1];
            for n in 0..=n_bound {
                for dn in 0..=n {
                    n_trans[n][dn] = binom(n, dn, p);
                }
            }

            let mut m_trans = vec![vec![0.0; m_bound + 1]; m_bound + 1];
            for m in 0..=m_bound {
                for dm in 0..=m {
                    m_trans[m][dm] = binom(m, dm, p);
                }
            }

            for dm in 0..=m_bound {
                for dn in 0..=n_bound {
                    if base + dm <= dn {
                        continue;
                    }
                    for m in 0..=m_bound - dm {
                        let m_remained = m_bound - m;
                        for n in 0..=n_bound - dn {
                            let n_remained = n_bound - n;
                            dp[m + dm][n + dn] +=
                                prev[m][n] * n_trans[n_remained][dn] * m_trans[m_remained][dm];
                        }
                    }
                }
            }
        }

        let ans = 1.0 - dp[m_bound][n_bound];
        writeln!(output, "Case #{}: {}", i_tc, ans).unwrap();
    }
}
