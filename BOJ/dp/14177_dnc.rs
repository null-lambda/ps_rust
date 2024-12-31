use std::{io::Write, ops::Range};

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

// Compute row minimum C(i) = min_j A(i, j)
// where opt(i) = argmin_j A(i, j) is monotonically increasing.
//
// Arguments:
// - naive: Compute (C(i), opt(i)) for a given range of j.
fn dnc_row_min<T>(
    res: &mut [T],
    naive: &impl Fn(usize, Range<usize>) -> (T, usize),
    i: Range<usize>,
    j: Range<usize>,
) {
    if i.start >= i.end {
        return;
    }
    let i_mid = i.start + i.end >> 1;
    let (res_mid, j_opt) = naive(i_mid, j.clone());

    res[i_mid] = res_mid;
    dnc_row_min(res, naive, i.start..i_mid, j.start..j_opt + 1);
    dnc_row_min(res, naive, i_mid + 1..i.end, j_opt..j.end);
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let k: usize = input.value();
    let mat: Vec<Vec<u32>> = (0..n)
        .map(|_| (0..n).map(|_| input.value()).collect())
        .collect();

    let mut row_sum = vec![vec![0; n]; n];
    for s in 0..n {
        for e in s + 1..n {
            row_sum[s][e] = row_sum[s][e - 1] + mat[s][e];
        }
    }

    let mut interval_cost_trans = vec![vec![0; n]; n]; // Transposed, for locality
    for e in 1..n {
        for s in (0..e).rev() {
            interval_cost_trans[e][s] = interval_cost_trans[e][s + 1] + row_sum[s][e];
        }
    }
    let interval_cost = |s: usize, e: usize| interval_cost_trans[e][s];

    let inf = 1 << 30;
    let mut dp = vec![inf; n + 1];
    dp[0] = 0;
    let mut prev = dp.clone();

    for _ in 0..k {
        std::mem::swap(&mut dp, &mut prev);

        let naive = |i: usize, j: Range<usize>| {
            (j.start..j.end.min(i))
                .map(|j| (prev[j] + interval_cost(j, i - 1), j))
                .min()
                .unwrap()
        };
        dnc_row_min(&mut dp, &naive, 1..n + 1, 0..n);
    }
    let ans = dp[n];
    writeln!(output, "{}", ans).unwrap();
}
