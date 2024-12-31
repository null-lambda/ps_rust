use std::{cmp::Reverse, collections::HashSet, io::Write, iter};

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

    let n: usize = input.value();

    let xs: Vec<i64> = (0..n).map(|_| input.value()).collect();
    let ys: Vec<i64> = (0..n).map(|_| input.value()).collect();
    let ds: Vec<i64> = (0..n).map(|i| -ys[i] - xs[i]).collect();
    let ds: Vec<Vec<i64>> = vec![ds[..n / 2].to_vec(), ds[n / 2..].to_vec()];

    let mut acc = vec![vec![0; 1 << (n / 2)]; 2];
    acc[0][0] = xs[..n / 2].iter().sum();
    acc[1][0] = xs[n / 2..].iter().sum();
    for g in 0..2 {
        for d in 0..n / 2 {
            for i in 0..1 << d {
                acc[g][i | (1 << d)] = acc[g][i] + ds[g][d];
            }
        }
    }

    let rev_digits = |i: usize, m: usize| {
        (0..m)
            .map(|j| (i >> j) & 1)
            .fold(0u64, |acc, x| (acc << 1) | x as u64)
    };
    let lex_key = |i: u64, j: u64| (i << (n / 2)) | j;

    let mut acc_agg = vec![vec![vec![]; n / 2 + 1]; 2];
    for g in 0..2 {
        for (i, &x) in acc[g].iter().enumerate() {
            acc_agg[g][i.count_ones() as usize].push((x, rev_digits(i, n / 2)));
        }
        for i in 0..n / 2 {
            acc_agg[g][i].sort_unstable();
            acc_agg[g][i].dedup_by_key(|(x, _)| *x);
        }
    }

    let mut ans = (i64::MAX, 0u64);
    for k0 in 0..=n / 2 {
        for &(a, i) in &acc_agg[0][k0] {
            let pair = &acc_agg[1][n / 2 - k0];

            let j0 = pair.partition_point(|&(x, _)| x < -a);
            if j0 > 0 {
                let (b, j) = pair[j0 - 1];
                ans = ans.min(((a + b).abs(), lex_key(i, j)));
            }
            if j0 < pair.len() {
                let (b, j) = pair[j0];
                ans = ans.min(((a + b).abs(), lex_key(i, j)));
            }
        }
    }

    for i in (0..n).rev() {
        let c = if (ans.1 >> i) & 1 == 1 { 2 } else { 1 };
        write!(output, "{} ", c).unwrap();
    }
    writeln!(output).unwrap();
}
