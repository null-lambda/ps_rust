use std::{collections::HashMap, io::Write};

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

fn ternary_search<F, K>(
    mut left: f64,
    mut right: f64,
    eps: f64,
    mut max_iter: usize,
    mut f: F,
) -> f64
where
    K: PartialOrd,
    F: FnMut(&f64) -> K,
{
    while right - left > eps && max_iter > 0 {
        let m1 = left + (right - left) * 3.0f64.recip();
        let m2 = right - (right - left) * 3.0f64.recip();
        if f(&m1) <= f(&m2) {
            right = m2;
        } else {
            left = m1;
        }
        max_iter -= 1;
    }
    (left + right) * 0.5
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let qs: Vec<[f64; 4]> = (0..n).map(|_| [0; 4].map(|_| input.value())).collect();

    let l = |&t: &f64| {
        let mut x_min = std::f64::MAX;
        let mut x_max = std::f64::MIN;
        let mut y_min = std::f64::MAX;
        let mut y_max = std::f64::MIN;
        for [px, py, vx, vy] in &qs {
            let x = px + vx * t;
            let y = py + vy * t;
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }

        (x_max - x_min).max(y_max - y_min)
    };

    let t = ternary_search(0.0, 1e4, 1e-15, 1000, l);
    writeln!(output, "{}", l(&t)).unwrap();
}
