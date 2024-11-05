use std::collections::{BTreeMap, BTreeSet};
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

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf: buf, iter }
    }
}

#[macro_use]
mod geometry {
    use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

    pub trait Scalar:
        Copy
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Neg<Output = Self>
        + PartialOrd
        + PartialEq
        + Default
        + std::fmt::Debug
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
    impl Scalar for i32 {}

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct PointNd<const N: usize, T>(pub [T; N]);

    pub type Point<T> = PointNd<2, T>;
    pub type Point3<T> = PointNd<3, T>;

    impl<T> Point<T> {
        pub fn new(x: T, y: T) -> Self {
            Self([x, y])
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Angle<T>(pub Point<T>);

    impl<T: Scalar> Angle<T> {
        pub fn circular_cmp(&self, other: &Self) -> std::cmp::Ordering {
            T::zero().partial_cmp(&cross(self.0, other.0)).unwrap()
        }
    }

    impl<T: Scalar> PartialEq for Angle<T> {
        fn eq(&self, other: &Self) -> bool {
            debug_assert!(self.0 != PointNd([T::zero(), T::zero()]));
            debug_assert!(other.0 != PointNd([T::zero(), T::zero()]));
            cross(self.0, other.0) == T::zero()
        }
    }

    impl<T: Scalar> Eq for Angle<T> {}

    impl<const N: usize, T: Scalar> PointNd<N, T> {
        pub fn dot(self, other: Self) -> T {
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| a * b)
                .fold(T::zero(), |acc, x| acc + x)
        }

        pub fn max_norm(self) -> T {
            self.0
                .into_iter()
                .map(|a| a.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        }
    }

    impl<const N: usize, T: Scalar> From<[T; N]> for PointNd<N, T> {
        fn from(p: [T; N]) -> Self {
            Self(p)
        }
    }

    impl<const N: usize, T: Scalar> Index<usize> for PointNd<N, T> {
        type Output = T;
        fn index(&self, i: usize) -> &Self::Output {
            &self.0[i]
        }
    }

    impl<const N: usize, T: Scalar> IndexMut<usize> for PointNd<N, T> {
        fn index_mut(&mut self, i: usize) -> &mut Self::Output {
            &mut self.0[i]
        }
    }

    macro_rules! impl_binop_dims {
        ($N:expr, $($idx:expr )+, $trait:ident, $fn:ident) => {
            impl<T: Scalar> $trait for PointNd<$N, T> {
                type Output = Self;
                fn $fn(self, other: Self) -> Self::Output {
                    PointNd([$(self[$idx].$fn(other[$idx])),+])
                }
            }
        };
    }

    macro_rules! impl_binop {
        ($trait:ident, $fn:ident) => {
            impl_binop_dims!(2, 0 1, $trait, $fn);
            impl_binop_dims!(3, 0 1 2, $trait, $fn);
        };
    }

    impl_binop!(Add, add);
    impl_binop!(Sub, sub);
    impl_binop!(Mul, mul);

    impl<const N: usize, T: Scalar> Default for PointNd<N, T> {
        fn default() -> Self {
            PointNd([T::zero(); N])
        }
    }

    impl<const N: usize, T: Scalar> Neg for PointNd<N, T> {
        type Output = Self;
        fn neg(self) -> Self::Output {
            PointNd(self.0.map(|x| -x))
        }
    }

    impl<const N: usize, T: Scalar> Mul<T> for PointNd<N, T> {
        type Output = Self;
        fn mul(self, k: T) -> Self::Output {
            PointNd(self.0.map(|x| x * k))
        }
    }

    pub fn cross<T: Scalar>(p: Point<T>, q: Point<T>) -> T {
        p[0] * q[1] - p[1] * q[0]
    }

    pub fn signed_area<T: Scalar>(p: Point<T>, q: Point<T>, r: Point<T>) -> T {
        cross(q - p, r - p)
    }
}

use geometry::*;

#[derive(PartialEq, Eq, PartialOrd, Debug, Clone)]
struct Segment {
    points: [Point<i64>; 2],
}

impl Ord for Segment {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let [a, b] = self.points;
        let [c, d] = other.points;
        let result = if signed_area(Point::default(), a, c) > 0 {
            signed_area(a, b, c).cmp(&0)
        } else {
            signed_area(c, d, a).cmp(&0).reverse()
        };
        debug_assert!(self.points == other.points || result != std::cmp::Ordering::Equal);
        result.then(self.points.cmp(&other.points))
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());

    let n: usize = input.value();
    let mut events: Vec<(Segment, usize, bool)> = Vec::with_capacity(n);
    for i in 0..n {
        let m: usize = input.value();
        let points: Vec<Point<i64>> = (0..m)
            .map(|_| Point::new(input.value(), input.value()))
            .collect();

        let min_angle = points
            .iter()
            .cloned()
            .min_by(|&p, &q| Angle(p).circular_cmp(&Angle(q)))
            .unwrap();
        let max_angle = points
            .iter()
            .cloned()
            .max_by(|&p, &q| Angle(p).circular_cmp(&Angle(q)))
            .unwrap();

        let seg = Segment {
            points: [min_angle, max_angle],
        };
        events.push((seg.clone(), i, false));
        events.push((seg, i, true));
    }
    fn endpoint(e: &(Segment, usize, bool)) -> Point<i64> {
        if e.2 {
            e.0.points[1]
        } else {
            e.0.points[0]
        }
    }
    events.sort_unstable_by(|e, f| Angle(endpoint(e)).circular_cmp(&Angle(endpoint(f))));

    // println!("{:?}", events);

    let mut active: BTreeSet<(Segment, usize)> = BTreeSet::new();
    let mut result = vec![];
    for (seg, i, p_end) in events {
        if !p_end {
            active.insert((seg, i));
        } else {
            active.remove(&(seg, i));
        }
        if let Some(&(_, i)) = active.iter().next() {
            result.push(i);
        }
        // println!("{:?}", active);
    }
    result.sort_unstable();
    result.dedup();
    let n_invisible = n - result.len();
    writeln!(output, "{}", n_invisible).unwrap();
}
