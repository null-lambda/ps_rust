use std::{collections::HashSet, io::Write};

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

fn partition_point<P>(
    mut left: f64,
    mut right: f64,
    mut max_iter: usize,
    threshold: f64,
    mut pred: P,
) -> f64
where
    P: FnMut(f64) -> bool,
{
    while right - left > threshold && max_iter > 0 {
        let mid = left + (right - left) * 0.5;
        if pred(mid) {
            left = mid;
        } else {
            right = mid;
        }
        max_iter -= 1;
    }
    left
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    assert!(n >= 4);
    let mut ls: Vec<f64> = (0..n).map(|_| input.value()).collect();
    ls.sort_unstable_by(|a, b| a.total_cmp(b).reverse());

    // println!("{:?}", ls);
    let l_sum = ls.iter().sum::<f64>();
    if 2.0 * ls[0] >= l_sum {
        writeln!(output, "NO CIRCLE").unwrap();
        return;
    }

    let theta = |r: f64, l: f64| {
        let mut res = (l * 0.5 / r).asin() * 2.0;
        if res.is_nan() {
            res = std::f64::consts::PI;
        }
        res
    };
    let sum_theta = |r| ls.iter().map(|&l| theta(r, l)).sum::<f64>();

    let r_lower = ls[0] * 0.5;
    let r_upper = 120.0;
    let inf = 1e6;
    let r_inner = partition_point(r_lower, inf, 1000, 1e-9, |r| {
        sum_theta(r) > std::f64::consts::PI * 2.0 - 1e-9
    });
    // println!("{:?}", r_inner);
    // println!("{:?}", sum_theta(r_inner));
    // println!("{:?}", sum_theta(52.0));

    if sum_theta(r_inner) > std::f64::consts::PI * 2.0 - 1e-9 {
        if r_inner <= r_upper + 1e-9 {
            writeln!(output, "{:.4}", r_inner).unwrap();
        } else {
            writeln!(output, "TOO BIG").unwrap();
        }
        return;
    }

    let sum_theta_res = |r| ls.iter().map(|&l| theta(r, l)).sum::<f64>() - theta(r, ls[0]);
    let r_outer = partition_point(r_lower, inf, 1000, 1e-9, |r| sum_theta_res(r) < 1e-9);
    // println!("{:?}", sum_theta_res(r_outer));

    if sum_theta_res(r_outer) > -1e-9 {
        if r_outer <= r_upper + 1e-9 {
            writeln!(output, "OUTSIDE").unwrap();
        } else {
            writeln!(output, "TOO BIG").unwrap();
        }
        return;
    }

    writeln!(output, "NO CIRCLE").unwrap();
}
