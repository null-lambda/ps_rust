use std::io::Write;

use geometry::Point;

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

#[macro_use]
mod geometry {
    use std::ops::{Add, Index, IndexMut, Mul, Sub};

    pub trait Scalar:
        Copy
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + PartialOrd
        + PartialEq
        + Default
    {
        fn zero() -> Self {
            Self::default()
        }
    }

    impl Scalar for f64 {}
    impl Scalar for i64 {}

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Point<T>(pub [T; 2]);

    impl<T: Scalar> From<[T; 2]> for Point<T> {
        fn from(p: [T; 2]) -> Self {
            Point(p)
        }
    }

    impl<T: Scalar> Index<usize> for Point<T> {
        type Output = T;
        fn index(&self, i: usize) -> &Self::Output {
            &self.0[i]
        }
    }

    impl<T: Scalar> IndexMut<usize> for Point<T> {
        fn index_mut(&mut self, i: usize) -> &mut Self::Output {
            &mut self.0[i]
        }
    }

    macro_rules! impl_binop {
        ($trait:ident, $fn:ident) => {
            impl<T: Scalar> $trait for Point<T> {
                type Output = Self;
                fn $fn(self, other: Self) -> Self::Output {
                    Point([self[0].$fn(other[0]), self[1].$fn(other[1])])
                }
            }
        };
    }

    impl_binop!(Add, add);
    impl_binop!(Sub, sub);
    impl_binop!(Mul, mul);

    impl<T: Scalar> Mul<T> for Point<T> {
        type Output = Self;
        fn mul(self, k: T) -> Self::Output {
            Point([self[0].mul(k), self[1].mul(k)])
        }
    }

    pub fn signed_area<T: Scalar>(p: Point<T>, q: Point<T>, r: Point<T>) -> T {
        (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    }

    pub fn lower_convex_hull<T: Scalar>(points: &mut [Point<T>]) -> Vec<Point<T>> {
        // monotone chain algorithm
        let n = points.len();
        if n <= 1 {
            return points.to_vec();
        }
        assert!(n >= 2);

        points.sort_unstable_by(|&p, &q| p.partial_cmp(&q).unwrap());

        let mut lower = Vec::new();
        for &p in points.iter() {
            while matches!(lower.as_slice(), &[.., l1, l2] if signed_area(p, l1, l2) <= T::zero()) {
                lower.pop();
            }
            lower.push(p);
        }
        lower
    }
}

fn partition_point<P>(mut left: u32, mut right: u32, mut pred: P) -> u32
where
    P: FnMut(u32) -> bool,
{
    while left < right {
        let mid = left + (right - left) / 2;
        if pred(mid) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();

    let mut ts_acc: Vec<i64> = vec![0];
    let mut acc = 0;
    for _ in 0..n {
        acc += input.value::<i64>();
        ts_acc.push(acc);
    }

    let fs: Vec<i64> = (0..m).map(|_| input.value()).collect();

    // Convex hull trick
    // Flip sign to compute upper convex hull
    let mut ps: Vec<Point<i64>> = ts_acc.windows(2).map(|w| Point([-w[0], -w[1]])).collect();
    ps = geometry::lower_convex_hull(&mut ps);

    let mut ans: i64 = 0;
    for f in fs.windows(2) {
        let i = partition_point(0, ps.len() as u32 - 1, |i| {
            geometry::signed_area(
                Point([0, 0]),
                ps[i as usize + 1] - ps[i as usize],
                Point([f[0], f[1]]),
            ) >= 0
        }) as usize;
        ans -= ps[i][1] * f[0] - ps[i][0] * f[1];
    }
    ans += fs[m - 1] * ts_acc[n];
    writeln!(output, "{}", ans).unwrap();
}
