use std::{cmp::Reverse, io::Write, iter};

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

fn add(a: u64, b: u64) -> u64 {
    (a + b) % P
}

fn add_assign(a: &mut u64, b: u64) {
    *a = (*a + b) % P;
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let l: usize = input.value();

    let word: Vec<u8> = input.token().bytes().map(|b| b - b'A').collect();
    let grid: Vec<u8> = (0..n)
        .flat_map(|_| input.token().bytes().take(m))
        .map(|b| b - b'A')
        .collect();

    let mut dp = vec![0; n * m];

    for i in 0..n {
        for j in 0..m {
            if grid[i * m + j] == word[0] {
                dp[i * m + j] = 1;
            }
        }
    }

    for &c in &word[1..] {
        let prev = dp;
        dp = vec![0; n * m];

        let mut sum_row = vec![0; n];
        let mut sum_col = vec![0; m];

        for i in 0..n {
            for j in 0..m {
                add_assign(&mut sum_row[i], prev[i * m + j]);
                add_assign(&mut sum_col[j], prev[i * m + j]);
            }
        }

        let sum_total = sum_row.iter().copied().fold(0, add);

        for i in 0..n {
            for j in 0..m {
                if grid[i * m + j] != c {
                    continue;
                }

                add_assign(&mut dp[i * m + j], sum_total);

                let is = || {
                    iter::empty()
                        .chain((i > 0).then(|| i - 1))
                        .chain(iter::once(i))
                        .chain((i + 1 < n).then(|| i + 1))
                };
                let js = || {
                    iter::empty()
                        .chain((j > 0).then(|| j - 1))
                        .chain(iter::once(j))
                        .chain((j + 1 < m).then(|| j + 1))
                };

                for ni in is() {
                    add_assign(&mut dp[i * m + j], P - sum_row[ni]);
                }
                for nj in js() {
                    add_assign(&mut dp[i * m + j], P - sum_col[nj]);
                }

                for ni in is() {
                    for nj in js() {
                        add_assign(&mut dp[i * m + j], prev[ni * m + nj]);
                    }
                }

                for di in [-2, 2] {
                    for dj in [-2, 2] {
                        let ni = i as isize + di;
                        let nj = j as isize + dj;
                        if ni < 0 || ni >= n as isize || nj < 0 || nj >= m as isize {
                            continue;
                        }

                        let ni = ni as usize;
                        let nj = nj as usize;
                        add_assign(&mut dp[i * m + j], P - prev[ni * m + nj]);
                    }
                }
            }
        }
    }

    let ans = dp.iter().copied().fold(0, add);
    writeln!(output, "{}", ans).unwrap();
}
