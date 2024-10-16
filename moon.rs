use std::{io::Write, iter::once};

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

#[allow(dead_code)]
mod cheap_rand {
    // Written in 2015 by Sebastiano Vigna (vigna@acm.org)
    // https://xoshiro.di.unimi.it/splitmix64.c
    use std::ops::Range;
    pub struct Rng(u64);

    impl Rng {
        pub fn new(seed: u64) -> Self {
            assert_ne!(seed, 0);
            Self(seed)
        }

        pub fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = self.0;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        }

        pub fn range_u64(&mut self, range: Range<u64>) -> u64 {
            let Range { start, end } = range;
            assert!(start < end);

            let width = end - start;
            let test = (u64::MAX - width) % width;
            loop {
                let value = self.next_u64();
                if value >= test {
                    return start + value % width;
                }
            }
        }

        pub fn shuffle<T>(&mut self, xs: &mut [T]) {
            let n = xs.len();
            for i in 0..n - 1 {
                let j = self.range_u64(i as u64..n as u64) as usize;
                xs.swap(i, j);
            }
        }
    }
}

#[allow(dead_code)]
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

    pub fn ccw<T: Scalar>(p: Point<T>, q: Point<T>, r: Point<T>) -> bool {
        T::zero() < signed_area(p, q, r)
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
            while matches!(lower.as_slice(), [.., l1, l2] if !ccw(*l1, *l2, p)) {
                lower.pop();
            }
            lower.push(p);
        }
        for &p in points.iter().rev() {
            while matches!(upper.as_slice(), [.., l1, l2] if !ccw(*l1, *l2, p)) {
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
        I: IntoIterator<Item = Point<f64>>,
        I::IntoIter: Clone,
    {
        let points = points.into_iter();
        let points_shifted = points.clone().skip(1).chain(points.clone().next());
        points
            .zip(points_shifted)
            .map(|(p, q)| cross(p, q))
            .sum::<f64>()
            .abs()
            / 2.0
    }

    pub mod dim3 {
        use super::*;

        pub fn cross<T: Scalar>(p: Point3<T>, q: Point3<T>) -> Point3<T> {
            PointNd([
                p[1] * q[2] - p[2] * q[1],
                p[2] * q[0] - p[0] * q[2],
                p[0] * q[1] - p[1] * q[0],
            ])
        }

        pub fn signed_vol<T: Scalar>(a: Point3<T>, b: Point3<T>, c: Point3<T>, d: Point3<T>) -> T {
            (b - a).dot(cross(c - a, d - a))
        }
        // points should be sorted randomly
        // incremental convex hull
        pub fn convex_hull<T: Scalar>(points: &[Point3<T>]) -> Vec<[usize; 3]> {
            macro_rules! unwrap_or_return {
                ($e:expr, $r: expr) => {
                    match $e {
                        Some(x) => x,
                        None => return $r,
                    }
                };
            }
            use std::collections::HashSet;
            use std::iter::once;

            if points.len() <= 2 {
                return vec![];
            }

            let [i, j] = [0, 1];
            let mut k = 2 + unwrap_or_return!(
                points[2..].iter().position(|&pk| {
                    cross(points[j] - points[i], pk - points[i]) != Point3::default()
                }),
                vec![]
            );

            let mut l = k + unwrap_or_return!(
                (k + 1 < points.len())
                    .then(|| points[k..].iter().position(|&pl| signed_vol(
                        points[i], points[j], points[k], pl
                    ) != T::zero()))
                    .flatten(),
                (2..points.len()).map(|k| [0, 1, k]).collect()
            );

            if signed_vol(points[i], points[j], points[k], points[l]) > T::zero() {
                std::mem::swap(&mut k, &mut l)
            }
            let mut faces = vec![[i, j, k], [i, k, l], [i, l, j], [j, l, k]];

            for (p_idx, &p) in points.iter().enumerate() {
                let (visible_faces, invisible_faces) = faces.into_iter().partition(|&[i, j, k]| {
                    signed_vol(p, points[i], points[j], points[k]) > T::zero()
                });
                faces = visible_faces;

                // point is inside of convex hull
                if invisible_faces.is_empty() {
                    continue;
                }

                let iter_boundary =
                    |[i, j, k]: [usize; 3]| once([i, j]).chain(once([j, k])).chain(once([k, i]));
                let invisible_half_edges: HashSet<[usize; 2]> = invisible_faces
                    .iter()
                    .flat_map(|&face| iter_boundary(face))
                    .collect();
                let boundary = invisible_half_edges
                    .iter()
                    .copied()
                    .filter(|&[i, j]| !invisible_half_edges.contains(&[j, i]));
                faces.extend(boundary.map(|[i, j]| [i, j, p_idx]));
            }
            faces
        }
    }
}

use dim3::signed_vol;
use geometry::*;

#[test]
fn gen_test_cases() {
    use std::time::{SystemTime, UNIX_EPOCH};
    let mut rng = cheap_rand::Rng::new(loop {
        let seed = (SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis()
            % u64::MAX as u128) as u64;
        if seed != 0 {
            break seed;
        }
    });

    let mut output_buf = Vec::<u8>::new();

    let n = 10000;
    let n_queries = 10;
    let mut pick = || rng.range_u64(0..4001) as i64 - 2000;
    writeln!(output_buf, "{} {}", n, n_queries).unwrap();
    for _ in 0..n {
        let [x, y, z] = loop {
            let [x, y, z] = [pick(), pick(), pick()];
            let [dx, dy, dz] = [x, y, z];
            if dx * dx + dy * dy + dz * dz <= 1000 * 1000 {
                break [x, y, z];
            }
        };
        writeln!(output_buf, "{} {} {}", x, y, z * 0 + 300).unwrap();
    }
    for _ in 0..n_queries {
        // writeln!(output_buf, "{} {} {} {}", pick(), pick(), pick(), pick()).unwrap();
        let [a, b, c] = [pick(), pick(), pick()];
        let d = 0;
        writeln!(output_buf, "{} {} {} {}", a, b, c, d).unwrap();
    }
    std::fs::write("input.txt", &output_buf).unwrap();
}

fn main() {
    let mut input = simple_io::stdin_at_once();

    let n_points: usize = input.value();
    let mut points: Vec<Point3<i64>> = (0..n_points)
        .map(|_| {
            {
                let p: Point3<f64> = [input.value(), input.value(), input.value()].into();
                let p = p * (1e6 / p.dot(p).sqrt());
                let p: Point3<i64> = [p[0] as i64, p[1] as i64, p[2] as i64].into();
                p
            }
            .into()
        })
        .collect();

    let mut rng = cheap_rand::Rng::new(10905525723936348110);
    rng.shuffle(&mut points);

    let faces = dim3::convex_hull(&points);
    let normalization_factor = 2.0 / (4.0 * std::f64::consts::PI);
    let spherical_vol = faces
        .iter()
        .filter_map(|&[i, j, k]| {
            let [a, b, c] = [points[i], points[j], points[k]];
            let numer = signed_vol(Point3::default(), a, b, c);
            // dbg!(numer);
            if numer <= 0 {
                return None;
            }

            let denom = 1 + a.dot(b) + b.dot(c) + c.dot(a);
            let area_half = (numer as f64 / denom as f64).atan();
            Some(area_half)
        })
        .sum::<f64>()
        * normalization_factor;
    let result = 1. - spherical_vol;
    println!("{}", result);
}
