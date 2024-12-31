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

const P: u32 = 1_000_000_007;
fn add_assign_mod(a: &mut u32, b: u32) {
    *a = (*a + b) % P
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let x: usize = input.value();
    let _y: usize = input.value();

    let grid: Vec<_> = (0..n).map(|_| input.token().as_bytes()).collect();

    let mut blank = vec![vec![true; m]; n];
    for i in 0..n {
        for j in 0..m {
            if grid[i][j] != b'A' {
                continue;
            }
            blank[i][j] = false;
            if i > 0 {
                blank[i - 1][j] = false;
            }
            if i < n - 1 {
                blank[i + 1][j] = false;
            }
            if j > 0 {
                blank[i][j - 1] = false;
            }
            if j < m - 1 {
                blank[i][j + 1] = false;
            }
        }
    }

    let mut mask_valid = vec![true; 1 << m * 2];
    for mask in 0..1 << m * 2 {
        for i in 0..m {
            if (mask >> 2 * i) & 0b11 == 3 {
                mask_valid[mask] = false;
            }
        }
    }

    let mut dp = vec![vec![0u32; x + 1]; 1 << m * 2];
    dp[0][0] = 1;

    let push = |mask: usize, cell: usize| (mask << 2 | cell) & ((1usize << m * 2) - 1);

    for i in 0..n {
        for j in 0..m {
            let prev = dp;
            dp = vec![vec![0; x + 1]; 1 << m * 2];
            for count in 0..=x {
                for prev_mask in 0..1 << m * 2 {
                    if !mask_valid[prev_mask] {
                        continue;
                    }

                    let left = if j > 0 { prev_mask & 0b11 } else { 0 };
                    let up = (prev_mask >> 2 * m - 2) & 0b11;
                    if up != 1 {
                        add_assign_mod(&mut dp[push(prev_mask, 0)][count], prev[prev_mask][count]);
                    }
                    if blank[i][j] && count < x {
                        match (left, up) {
                            (0, 0) => {
                                add_assign_mod(
                                    &mut dp[push(prev_mask, 1)][count + 1],
                                    prev[prev_mask][count],
                                );
                            }
                            (1..=2, _) | (_, 1..=2) => {
                                let mut mask = push(prev_mask, 2);
                                if left != 0 {
                                    mask = mask & !(0b11 << 2);
                                    mask = mask | (2 << 2);
                                }
                                add_assign_mod(&mut dp[mask][count + 1], prev[prev_mask][count]);
                            }
                            _ => panic!(),
                        }
                    }
                }
            }
        }
    }

    let mut ans = 0;
    'outer: for mask in 0..1 << 2 * m {
        for i in 0..m {
            if matches!((mask >> 2 * i) & 0b11, 1 | 3) {
                continue 'outer;
            }
        }

        add_assign_mod(&mut ans, dp[mask][x]);
    }
    writeln!(output, "{}", ans).unwrap();
}
