use std::{cmp::Ordering, io::Write};

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

struct OrderedFloat<T>(T);

impl PartialEq for OrderedFloat<f64> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(&other) == Ordering::Equal
    }
}

impl Eq for OrderedFloat<f64> {}

impl PartialOrd for OrderedFloat<f64> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl Ord for OrderedFloat<f64> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let x: f64 = input.value();
    let y: f64 = input.value();
    let ga: Vec<f64> = (0..n).map(|_| input.value()).collect();
    let gb: Vec<f64> = (0..n).map(|_| input.value()).collect();
    let w: Vec<f64> = (0..n).map(|_| input.value()).collect();

    let a_max = w
        .iter()
        .zip(&ga)
        .map(|(x, y)| x / y)
        .min_by_key(|&z| OrderedFloat(z))
        .unwrap();
    let b_max = w
        .iter()
        .zip(&gb)
        .map(|(x, y)| x / y)
        .min_by_key(|&z| OrderedFloat(z))
        .unwrap();

    let b_from_a = |a: f64| {
        b_max.min(
            ga.iter()
                .zip(&gb)
                .zip(&w)
                .map(|((x, y), z)| (z - x * a) / y)
                .min_by_key(|&z| OrderedFloat(z))
                .unwrap(),
        )
    };
    let score = |a: f64| a * x + b_from_a(a) * y;

    let a_opt = ternary_search(0.0, a_max, 1e-9, 10000, |&x| -score(x));
    let b_opt = b_from_a(a_opt);
    let score_opt = score(a_opt);
    writeln!(output, "{}", score_opt).unwrap();
    writeln!(output, "{} {}", a_opt, b_opt).unwrap();
}
