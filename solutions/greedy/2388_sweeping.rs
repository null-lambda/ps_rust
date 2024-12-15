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
    let m: usize = input.value();
    let mut xs: Vec<i64> = (0..n).map(|_| input.value()).collect();
    let mut ys: Vec<i64> = (0..m).map(|_| input.value()).collect();

    xs.push(0);
    ys.push(0);

    xs.sort_unstable();
    ys.sort_unstable();

    let x_max = *xs.last().unwrap();
    let y_max = *ys.last().unwrap();
    if x_max != y_max {
        writeln!(output, "-1").unwrap();
        return;
    }

    let mut xs = xs.into_iter().rev().peekable();
    let mut ys = ys.into_iter().rev().peekable();

    let mut h_prev = x_max;
    let mut x_count = 0;
    let mut y_count = 0;
    let mut min = 0;
    let mut max = 0;

    let mut min_area = 0;
    let mut max_area = 0;
    let mut x_count_prev = 0;
    let mut x_count_prev2 = 0;
    let mut y_count_prev = 0;
    let mut y_count_prev2 = 0;
    loop {
        let h = match (xs.peek(), ys.peek()) {
            (Some(&x), Some(&y)) if x >= y => {
                x_count += 1;
                xs.next().unwrap()
            }
            (Some(&_), Some(&_)) => {
                y_count += 1;
                ys.next().unwrap()
            }
            (Some(&_), None) => {
                x_count += 1;
                xs.next().unwrap()
            }
            (None, Some(&_)) => {
                y_count += 1;
                ys.next().unwrap()
            }
            (None, None) => break,
        };
        let dh = h_prev - h;

        if dh == 0 {
            x_count_prev = x_count;
            y_count_prev = y_count;
        } else {
            min_area += (x_count_prev - x_count_prev2).max(y_count_prev - y_count_prev2);
            x_count_prev2 = x_count_prev;
            y_count_prev2 = y_count_prev;
            x_count_prev = x_count;
            y_count_prev = y_count;
        }
        min += dh * min_area;
        max += dh * max_area;

        max_area = x_count * y_count;
        h_prev = h;
    }

    writeln!(output, "{} {}", min, max).unwrap();
}
