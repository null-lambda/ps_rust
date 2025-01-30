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

pub mod cht {
    // Max-hull of lines with increasing slopes
    type V = i64;

    pub struct Line {
        pub slope: V,
        pub intercept: V,
    }

    impl Line {
        pub fn new(slope: V, intercept: V) -> Self {
            Self { slope, intercept }
        }

        pub fn eval(&self, x: &V) -> V {
            self.slope * x + self.intercept
        }

        fn should_remove(&self, lhs: &Self, rhs: &Self) -> bool {
            debug_assert!(lhs.slope < self.slope && self.slope <= rhs.slope);
            if self.slope == rhs.slope {
                self.intercept <= rhs.intercept
            } else {
                (rhs.slope - self.slope) * (self.intercept - lhs.intercept)
                    <= (self.slope - lhs.slope) * (rhs.intercept - self.intercept)
            }
        }
    }

    pub struct MonotoneStack {
        lines: Vec<Line>,
    }

    impl MonotoneStack {
        pub fn new() -> Self {
            Self { lines: vec![] }
        }

        pub fn insert(&mut self, line: Line) {
            while self.lines.len() >= 2 {
                let n = self.lines.len();
                if self.lines[n - 1].should_remove(&self.lines[n - 2], &line) {
                    self.lines.pop();
                } else {
                    break;
                }
            }
            self.lines.push(line);
        }

        pub fn eval(&self, x: &V) -> V {
            assert!(!self.lines.is_empty());
            let mut left = 0;
            let mut right = self.lines.len() - 1;
            while left < right {
                let mid = left + right >> 1;
                if self.lines[mid].eval(x) >= self.lines[mid + 1].eval(x) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }
            self.lines[left].eval(x)
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut xs = vec![0i64];
    for _ in 1..=n {
        xs.push(input.value());
    }

    let mut prefix = xs.clone();
    for i in 1..=n {
        prefix[i] += prefix[i - 1];
    }

    let mut ans = 0;

    let mut max_hull = cht::MonotoneStack::new();
    for i in (1..=n).rev() {
        max_hull.insert(cht::Line::new(-(i as i64), -prefix[i]));
        ans = ans.max(max_hull.eval(&(-xs[i])) - i as i64 * xs[i] + prefix[i]);
    }

    max_hull = cht::MonotoneStack::new();
    for i in 1..=n {
        max_hull.insert(cht::Line::new(i as i64, -prefix[i - 1]));
        ans = ans.max(max_hull.eval(&xs[i]) - i as i64 * xs[i] + prefix[i - 1]);
    }

    ans += (1..=n).map(|i| i as i64 * xs[i]).sum::<i64>();
    writeln!(output, "{}", ans).unwrap();
}
