use std::{
    io::Write,
    ops::{Add, Mul, Sub},
};

use cht::LineContainer;

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

#[derive(Debug, Clone, Copy)]
struct OrderedF64(f64);

impl From<f64> for OrderedF64 {
    fn from(f: f64) -> Self {
        Self(f)
    }
}

impl PartialEq for OrderedF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Add for OrderedF64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub for OrderedF64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Mul for OrderedF64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

mod cht {
    // Line Container for Convex hull trick
    // adapted from KACTL
    // https://github.com/kth-competitive-programming/kactl/blob/main/content/data-structures/LineContainer.h

    use crate::OrderedF64;
    use std::{cell::Cell, cmp::Ordering, collections::BTreeSet};

    type V = i64;
    type I = OrderedF64;

    const NEG_INF: I = OrderedF64(-1e18);
    const INF: I = OrderedF64(1e18);

    // fn div_floor(x: V, y: V) -> V {
    //     let x = x.0;
    //     let y = y.0;
    //     OrderedF64(x / y - (((x < 0.0) ^ (y < 0.0)) && x % y != 0.0) as u8 as f64)
    // }

    #[derive(Clone, Debug)]
    pub struct Line {
        pub slope: V,
        pub intercept: V,
        right_end: Cell<I>,
        point_query: bool, // Bypass BTreeMap's API with some additional runtime cost
    }

    #[derive(Debug)]
    pub struct LineContainer {
        lines: BTreeSet<Line>,
        stack: Vec<Line>,
    }

    impl Line {
        fn new(slope: V, intercept: V) -> Self {
            Self {
                slope,
                intercept,
                right_end: Cell::new(INF),
                point_query: false,
            }
        }

        fn point_query(x: I) -> Self {
            Self {
                slope: 0,
                intercept: 0,
                right_end: Cell::new(x),
                point_query: true,
            }
        }

        fn inter(&self, other: &Line) -> I {
            if self.slope != other.slope {
                OrderedF64(
                    (self.intercept - other.intercept) as f64 / (other.slope - self.slope) as f64,
                )
                // div_floor(self.intercept - other.intercept, other.slope - self.slope)
            } else if self.intercept > other.intercept {
                INF
            } else {
                NEG_INF
            }
        }
    }

    impl PartialEq for Line {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == Ordering::Equal
        }
    }

    impl Eq for Line {}

    impl PartialOrd for Line {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Line {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            if !self.point_query && !other.point_query {
                self.slope.cmp(&other.slope)
            } else {
                self.right_end.get().cmp(&other.right_end.get())
            }
        }
    }

    impl LineContainer {
        pub fn new() -> Self {
            Self {
                lines: Default::default(),
                stack: Default::default(),
            }
        }

        pub fn insert(&mut self, slope: V, intercept: V) {
            let y = Line::new(slope, intercept);

            let to_remove = &mut self.stack;
            for z in self.lines.range(&y..) {
                y.right_end.set(y.inter(z));
                if y.right_end < z.right_end {
                    break;
                }
                to_remove.push(z.clone());
            }

            let mut r = self.lines.range(..&y).rev();
            if let Some(x) = r.next() {
                let new_x_right_end = x.inter(&y);
                if !(new_x_right_end < y.right_end.get()) {
                    return;
                }
                x.right_end.set(new_x_right_end);

                let mut x_prev = x;
                for x in r {
                    if x.right_end < x_prev.right_end {
                        break;
                    }
                    x.right_end.set(x.inter(&y));
                    to_remove.push(x_prev.clone());

                    x_prev = x;
                }
            }

            for x in to_remove.drain(..) {
                self.lines.remove(&x);
            }
            self.lines.insert(y);
        }

        pub fn query(&self, x: I) -> Option<Line> {
            self.lines.range(Line::point_query(x)..).next().cloned()
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let ps: Vec<(i64, i64)> = (0..n).map(|_| (input.value(), input.value())).collect();

    let mut hull = LineContainer::new();
    let (r, s) = ps[0];
    hull.insert(r, s);

    let mut ans = 0;
    for &(r, s) in &ps[1..] {
        let l = hull.query((r as f64 / s as f64).into()).unwrap();
        ans = ans.max(l.slope * r + l.intercept * s);
        hull.insert(r, s);
    }
    writeln!(output, "{}", ans).unwrap();
}
