use std::{collections::BinaryHeap, io::Write};

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

    for _ in 0..input.value() {
        let n: usize = input.value();
        let d: i64 = input.value();
        let mut xs: Vec<_> = (0..n).map(|_| input.value::<i64>()).collect();
        xs.sort_unstable();
        for i in 0..n {
            xs[i] -= i as i64 * d;
        }

        let mut breakpoints = BinaryHeap::<i64>::new();
        let mut intercept = 0;
        for x in xs {
            intercept += x.abs();
            if x > 0 {
                breakpoints.push(x);
                breakpoints.push(x);
            }
            breakpoints.pop();
        }
        breakpoints.push(0);

        let mut y_min = intercept;
        let mut abs_slope = 0;
        if let Some(mut prev) = breakpoints.pop() {
            while let Some(x) = breakpoints.pop() {
                abs_slope += 1;
                intercept += abs_slope * (x - prev);
                y_min = y_min.min(intercept);
                prev = x;
            }
        }
        writeln!(output, "{}", y_min).unwrap();
    }
}
