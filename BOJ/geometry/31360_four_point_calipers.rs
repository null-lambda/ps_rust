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
        + PartialEq
        + Eq
        + PartialOrd
        + Ord
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

    impl Scalar for i64 {}

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Point<T>([T; 2]);

    impl<T: Scalar> Point<T> {
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

        pub fn map(self, f: impl FnMut(T) -> T) -> Self {
            Point(self.0.map(f))
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

    impl<T: Scalar> Neg for Point<T> {
        type Output = Self;
        fn neg(self) -> Self::Output {
            Point(self.0.map(T::neg))
        }
    }

    macro_rules! impl_binop {
        ($trait:ident, $fn:ident) => {
            impl<T: Scalar> $trait for Point<T> {
                type Output = Self;
                fn $fn(self, other: Self) -> Self::Output {
                    Point(std::array::from_fn(|i| self[i].$fn(other[i])))
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

    pub mod cvhull {
        use super::*;

        pub fn dim2<T: Scalar>(points: &mut [Point<T>]) -> Vec<Point<T>> {
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
                while matches!(lower.as_slice(), &[.., l1, l2] if signed_area(p, l1, l2) <= T::zero())
                {
                    lower.pop();
                }
                lower.push(p);
            }
            for &p in points.iter().rev() {
                while matches!(upper.as_slice(), &[.., l1, l2] if signed_area(p, l1, l2) <= T::zero())
                {
                    upper.pop();
                }
                upper.push(p);
            }
            lower.pop();
            upper.pop();

            lower.extend(upper);
            lower
        }

        pub fn area<I>(points: I) -> f64
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

        pub fn rotating_calipers<T: Scalar>(
            hull: &[Point<T>],
            mut yield_antipodals: impl FnMut(usize, usize, Ordering),
        ) {
            use std::cmp::Ordering::*;
            let n_verts = hull.len();
            let inc = |i| (i + 1) % n_verts;
            let signed_area = |i, j, k| signed_area::<T>(hull[i], hull[j], hull[k]);
            let compare_segments =
                |i, j| signed_area(i, inc(i), inc(j)).cmp(&signed_area(i, inc(i), j));

            let mut i = 0;
            let mut j = (1..n_verts)
                .find(|&j| compare_segments(i, j).is_le())
                .unwrap();
            let i_last = j;
            let j_last = i;
            while (i, j) != (i_last, j_last) {
                let angle_relation = compare_segments(i, j);
                yield_antipodals(i, j, angle_relation);
                match angle_relation {
                    Less => i = inc(i),
                    Greater => j = inc(j),
                    Equal => {
                        yield_antipodals(i, inc(j), Less);
                        yield_antipodals(inc(i), j, Greater);
                        i = inc(i);
                        j = inc(j);
                    }
                }
            }
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut ps: Vec<Point<i64>> = (0..n)
        .map(|_| [input.value(), input.value()].into())
        .collect();
    let hull = geometry::cvhull::dim2(&mut ps);

    let mut min_area = f64::INFINITY;
    let mut rect = [[0.0; 2]; 4];
    let m = hull.len();
    let mut right = 0;
    let mut j = 0;
    let mut left = 0;
    for i in 0..m {
        let inc = |k| (k + 1) % m;
        let dec = |k| (k + m - 1) % m;

        let h = hull[inc(i)] - hull[i];
        let r = h.rot();

        let x = |k: usize| hull[k].dot(h);
        let y = |k: usize| hull[k].dot(r);
        right = (0..m)
            .map(|k| (k + right) % m)
            .find(|&k| x(k) >= x(dec(k)) && x(k) > x(inc(k)))
            .unwrap();
        j = (0..m)
            .map(|k| (k + j) % m)
            .find(|&k| y(k) >= y(dec(k)) && y(k) > y(inc(k)))
            .unwrap();
        left = (0..m)
            .map(|k| (k + left) % m)
            .find(|&k| x(k) <= x(dec(k)) && x(k) < x(inc(k)))
            .unwrap();

        let width = (x(right) - x(left)) as f64 / (h.norm_sq() as f64).sqrt();
        let height = (y(j) as f64 - y(i) as f64) / (r.norm_sq() as f64).sqrt();
        let area = width * height;
        if area < min_area {
            min_area = area;
            let ax = [
                (h * x(left), h.norm_sq() as f64),
                (r * y(i), r.norm_sq() as f64),
                (h * x(right), h.norm_sq() as f64),
                (r * y(j), r.norm_sq() as f64),
            ]
            .map(|(p, d)| [p[0] as f64 / d, p[1] as f64 / d]);
            rect = std::array::from_fn(|i| {
                let p = ax[i];
                let q = ax[(i + 1) % 4];
                [p[0] + q[0], p[1] + q[1]]
            });
        }
    }
    for p in rect {
        writeln!(output, "{} {}", p[0], p[1]).unwrap();
    }
}
