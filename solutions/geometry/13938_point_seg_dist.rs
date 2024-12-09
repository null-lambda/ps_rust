use std::{collections::HashMap, io::Write};

use geometry::*;

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
    use std::{
        cmp::Ordering,
        ops::{Add, Index, IndexMut, Mul, Neg, Sub},
    };

    pub trait Scalar:
        Copy
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Neg<Output = Self>
        + PartialOrd
        + PartialEq
        + Default
    {
        fn zero() -> Self {
            Self::default()
        }

        fn abs(self) -> Self {
            if self < Self::zero() {
                -self
            } else {
                self
            }
        }
    }

    impl Scalar for f64 {}
    impl Scalar for i64 {}

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Point<T>([T; 2]);

    impl<T: Scalar> Point<T> {
        pub fn zero() -> Self {
            Point([T::zero(), T::zero()])
        }

        pub fn new(x: T, y: T) -> Self {
            Point([x, y])
        }

        pub fn dot(self, other: Self) -> T {
            self[0] * other[0] + self[1] * other[1]
        }

        pub fn norm_sq(self) -> T {
            self.dot(self)
        }

        pub fn cross(self, other: Self) -> T {
            self[0] * other[1] - self[1] * other[0]
        }

        pub fn rot(self) -> Self {
            Point([-self[1], self[0]])
        }
    }

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

    #[derive(Debug, Clone, Copy)]
    pub struct Angle<T>(pub Point<T>);

    impl<T: Scalar> Angle<T> {
        pub fn on_lower_half(self) -> bool {
            (self.0[1], self.0[0]) < (T::zero(), T::zero())
        }

        pub fn circular_cmp(&self, other: &Self) -> Ordering {
            T::zero().partial_cmp(&self.0.cross(other.0)).unwrap()
        }
    }

    impl<T: Scalar> PartialEq for Angle<T> {
        fn eq(&self, other: &Self) -> bool {
            self.on_lower_half() == other.on_lower_half() && self.0.cross(other.0) == T::zero()
        }
    }

    impl<T: Scalar> Eq for Angle<T> {}

    impl<T: Scalar> PartialOrd for Angle<T> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(
                (self.on_lower_half().cmp(&other.on_lower_half()))
                    .then_with(|| self.circular_cmp(other)),
            )
        }
    }

    impl<T: Scalar> Ord for Angle<T> {
        fn cmp(&self, other: &Self) -> Ordering {
            self.partial_cmp(other).unwrap()
        }
    }

    pub fn signed_area<T: Scalar>(p: Point<T>, q: Point<T>, r: Point<T>) -> T {
        (q - p).cross(r - p)
    }

    pub fn convex_hull<T: Scalar>(points: &mut [Point<T>]) -> Vec<Point<T>> {
        // monotone chain algorithm
        let n = points.len();
        if n <= 1 {
            return points.to_vec();
        }
        assert!(n >= 2);

        points.sort_unstable_by(|&p, &q| p.partial_cmp(&q).unwrap());

        let mut lower = Vec::new();
        let mut upper = Vec::new();
        for &p in points.iter() {
            while matches!(lower.as_slice(), &[.., l1, l2] if signed_area(p, l1, l2) <= T::zero()) {
                lower.pop();
            }
            lower.push(p);
        }
        for &p in points.iter().rev() {
            while matches!(upper.as_slice(), &[.., l1, l2] if signed_area(p, l1, l2) <= T::zero()) {
                upper.pop();
            }
            upper.push(p);
        }
        lower.pop();
        upper.pop();

        lower.extend(upper);
        lower
    }

    pub fn convex_hull_area<I>(points: I) -> f64
    where
        I: IntoIterator<Item = [f64; 2]>,
        I::IntoIter: Clone,
    {
        let mut area: f64 = 0.0;
        let points = points.into_iter();
        let points_shifted = points.clone().skip(1).chain(points.clone().next());
        for ([x1, y1], [x2, y2]) in points.zip(points_shifted) {
            area += x1 * y2 - x2 * y1;
        }
        area = (area / 2.0).abs();
        area
    }

    pub fn seg_point_dist(l0: Point<f64>, l1: Point<f64>, p: Point<f64>) -> f64 {
        let mut u = l1 - l0;
        u = u * u.norm_sq().powf(-0.5);
        let x = u.dot(p - l0);
        let y = u.dot(p - l1);
        if x * y < 0.0 {
            return u.cross(p - l0).abs();
        }

        let d0 = (l0 - p).norm_sq().sqrt();
        let d1 = (l1 - p).norm_sq().sqrt();
        d0.min(d1)
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    assert!(n >= 3);
    let ps: Vec<Point<f64>> = (0..n)
        .map(|_| Point::new(input.value(), input.value()))
        .collect();

    let r = 3999.5;
    let r_sq = r * r;

    let mut centers = vec![];
    'outer: for i in 0..n {
        let a = ps[(i + n - 1) % n];
        let p = ps[i];
        let b = ps[(i + 1) % n];

        let mut da = a - p;
        let mut db = b - p;
        da = da * da.norm_sq().powf(-0.5);
        db = db * db.norm_sq().powf(-0.5);

        if !(signed_area(p, a, b) > 0.0) {
            continue;
        }

        let mut mid = da + db;
        mid = mid * mid.norm_sq().powf(-0.5);

        let sin = da.cross(mid).abs();
        let center = p + mid * (r / sin);

        for j in 0..n {
            let q = ps[j];
            let s = ps[(j + 1) % n];
            if seg_point_dist(q, s, center) <= r - 1e-1 {
                continue 'outer;
            }
        }

        centers.push(center);
    }

    let mut ans = None;
    'outer: for i in 0..centers.len() {
        for j in i + 1..centers.len() {
            if (centers[i] - centers[j]).norm_sq().sqrt() >= 2.0 * r {
                ans = Some((centers[i], centers[j]));
                break 'outer;
            }
        }
    }

    if let Some((p, q)) = ans {
        writeln!(output, "{} {}", p[0], p[1]).unwrap();
        writeln!(output, "{} {}", q[0], q[1]).unwrap();
    } else {
        writeln!(output, "impossible").unwrap();
    }
}
