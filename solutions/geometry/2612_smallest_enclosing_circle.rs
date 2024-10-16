use std::{collections::HashMap, io::Write, iter::once};

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
mod geometry {
    use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

    pub trait Scalar:
        Copy
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
        + Neg<Output = Self>
        + PartialOrd
        + PartialEq
        + Default
        + std::fmt::Debug
    {
        fn zero() -> Self {
            Self::default()
        }

        fn one() -> Self;

        fn recip(self) -> Self {
            Self::one() / self
        }

        fn abs(self) -> Self {
            if self < Self::zero() {
                -self
            } else {
                self
            }
        }
    }

    macro_rules! impl_scalar {
        ($t:ty, $one:expr) => {
            impl Scalar for $t {
                fn one() -> Self {
                    $one
                }
            }
        };
    }

    impl_scalar!(i32, 1);
    impl_scalar!(i64, 1);
    impl_scalar!(i128, 1);
    impl_scalar!(f64, 1.0);

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

        pub fn norm_sq(self) -> T {
            self.0
                .into_iter()
                .map(|a| a * a)
                .fold(T::zero(), |acc, x| acc + x.into())
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

    pub fn convex_hull<T: Scalar>(points: &[Point<T>], tol: T) -> Vec<usize> {
        // monotone chain algorithm
        let n = points.len();
        if n <= 1 {
            return (0..n).collect();
        }
        assert!(n >= 2);

        let mut points: Vec<(usize, Point<T>)> = points.into_iter().copied().enumerate().collect();
        points.sort_unstable_by(|&(_, p), &(_, q)| p.partial_cmp(&q).unwrap());

        let mut lower: Vec<usize> = vec![];
        let mut upper: Vec<usize> = vec![];
        for p in 0..points.len() {
            while matches!(lower.as_slice(), &[.., l1, l2] if tol >= signed_area(points[l1].1, points[l2].1, points[p].1))
            {
                lower.pop();
            }
            lower.push(p);
        }
        for p in (0..points.len()).rev() {
            while matches!(upper.as_slice(), &[.., l1, l2] if tol >= signed_area(points[l1].1, points[l2].1, points[p].1))
            {
                upper.pop();
            }
            upper.push(p);
        }
        lower.pop();
        upper.pop();

        lower
            .into_iter()
            .chain(upper)
            .map(|i| points[i].0)
            .collect()
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
        use std::collections::HashMap;

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

        #[derive(Debug)]
        pub enum Mesh {
            Empty,
            Dim0,
            Dim1([usize; 2]),
            Dim2(Vec<usize>),
            Dim3(Vec<[usize; 3]>),
        }

        // points should be sorted randomly
        // incremental convex hull, with conflict graph
        // https://cw.fel.cvut.cz/b221/_media/courses/cg/lectures/05-convexhull-3d.pdf
        pub fn convex_hull_fast<T: Scalar>(points: &[Point3<T>], tol: T) -> Mesh {
            // assumption that all points are distinct
            use std::collections::HashSet;
            use std::iter::once;

            match points.len() {
                0 => return Mesh::Empty,
                1 => return Mesh::Dim0,
                2 => return Mesh::Dim1([0, 1]),
                _ => {}
            }

            let [i, j] = [0, 1];
            let mut k = 2 + match points[2..]
                .iter()
                .position(|&pk| cross(points[j] - points[i], pk - points[i]).max_norm() > tol)
            {
                Some(x) => x,
                None => {
                    let i = points
                        .iter()
                        .enumerate()
                        .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
                        .unwrap();
                    let j = points
                        .iter()
                        .enumerate()
                        .min_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
                        .unwrap();
                    return Mesh::Dim1([i.0, j.0]);
                }
            };

            let mut l = k + match points[k..]
                .iter()
                .position(|&pl| signed_vol(points[i], points[j], points[k], pl).abs() > tol)
            {
                Some(x) => x,
                None => {
                    let normal = cross(points[j] - points[i], points[k] - points[i]);
                    let normal_max = normal.max_norm();
                    let mut projection: Vec<Point<T>> = if normal[0].abs() == normal_max {
                        points.iter().map(|&p| [p[1], p[2]].into()).collect()
                    } else if normal[1].abs() == normal_max {
                        points.iter().map(|&p| [p[0], p[2]].into()).collect()
                    } else {
                        points.iter().map(|&p| [p[0], p[1]].into()).collect()
                    };
                    let cvhull_2d = super::convex_hull(&mut projection, tol);
                    return Mesh::Dim2(cvhull_2d);
                }
            };

            if !(signed_vol(points[i], points[j], points[k], points[l]) > tol) {
                std::mem::swap(&mut k, &mut l)
            }

            let mut faces: HashSet<_> = [[i, k, j], [i, l, k], [i, j, l], [j, k, l]].into();
            let mut face_left_to_edge: HashMap<[usize; 2], [usize; 3]> = HashMap::new();
            for &f in &faces {
                let [i, j, k] = f;
                face_left_to_edge.insert([i, j], f);
                face_left_to_edge.insert([j, k], f);
                face_left_to_edge.insert([k, i], f);
            }

            // initialize conflict graph
            let mut visible_faces: Vec<Vec<[usize; 3]>> = vec![vec![]; points.len()];
            let mut visible_points: HashMap<[usize; 3], HashSet<usize>> =
                faces.iter().map(|&f| (f, HashSet::new())).collect();

            for p_idx in 0..points.len() {
                if [i, j, k, l].contains(&p_idx) {
                    continue;
                }

                for (_, &f) in faces.iter().enumerate() {
                    let [i, j, k] = f;
                    if signed_vol(points[p_idx], points[i], points[j], points[k]) < -tol {
                        visible_faces[p_idx].push(f);
                        visible_points.entry(f).or_default().insert(p_idx);
                    }
                }
            }

            for p_idx in 0..points.len() {
                if [i, j, k, l].contains(&p_idx) {
                    continue;
                }
                visible_faces[p_idx].retain(|&f| faces.contains(&f));
                if visible_faces[p_idx].is_empty() {
                    continue;
                }

                for &f in &visible_faces[p_idx] {
                    faces.remove(&f);
                }

                let iter_boundary =
                    |[i, j, k]: [usize; 3]| once([i, j]).chain(once([j, k])).chain(once([k, i]));
                let visible_half_edges: HashSet<[usize; 2]> = visible_faces[p_idx]
                    .iter()
                    .flat_map(|&face| iter_boundary(face))
                    .collect();
                let boundary = || {
                    visible_half_edges
                        .iter()
                        .copied()
                        .filter(|&[i, j]| !visible_half_edges.contains(&[j, i]))
                };
                for [i, j] in boundary() {
                    let f_new = [i, j, p_idx];
                    faces.insert(f_new);
                    visible_points.insert(f_new, Default::default());

                    let mut p_next: HashSet<usize> = Default::default();
                    if face_left_to_edge.contains_key(&[i, j]) {
                        p_next.extend(visible_points[&face_left_to_edge[&[i, j]]].iter().cloned());
                        p_next.extend(visible_points[&face_left_to_edge[&[j, i]]].iter().cloned());
                    }

                    for &q in &p_next {
                        if signed_vol(points[q], points[i], points[j], points[p_idx]) < -tol {
                            visible_faces[q].push(f_new);
                            visible_points.get_mut(&f_new).unwrap().insert(q);
                        }
                    }

                    *face_left_to_edge.get_mut(&[i, j]).unwrap() = f_new;
                    face_left_to_edge.insert([j, p_idx], f_new);
                    face_left_to_edge.insert([p_idx, i], f_new);
                }
                for face in visible_faces[p_idx].clone() {
                    visible_points.remove(&face);
                }
            }

            Mesh::Dim3(faces.into_iter().collect())
        }
    }
}

use geometry::*;

#[derive(Debug, Clone, Copy)]
pub struct Circle<T> {
    center: Point<T>,
    r_sq: T,
}

// welzl's algorithm
pub fn smallest_enclosing_circ<T: Scalar>(
    rng: &mut cheap_rand::Rng,
    points: &mut [Point<T>],
    found: &mut Vec<Point<T>>,
) -> Option<Circle<T>> {
    debug_assert!(found.len() <= 3);
    let n = points.len();
    if n == 0 || found.len() == 3 {
        return match found.len() {
            0 => None,
            1 => Some(Circle {
                center: found[0],
                r_sq: T::zero(),
            }),
            2 => {
                let half = (T::one() + T::one()).recip();
                Some(Circle {
                    center: (found[0] + found[1]) * half,
                    r_sq: (found[0] - found[1]).norm_sq() * (half * half),
                })
            }
            3 => {
                let PointNd([x1, y1]) = found[0];
                let PointNd([x2, y2]) = found[1];
                let PointNd([x3, y3]) = found[2];

                let double = |x: T| x + x;
                let sq = |x: T| x * x;

                let (x12, y12) = (x1 - x2, y1 - y2);
                let (x32, y32) = (x3 - x2, y3 - y2);
                let a = sq(x1) - sq(x2) + sq(y1) - sq(y2);
                let b = sq(x3) - sq(x2) + sq(y3) - sq(y2);

                let denom = double(x32 * y12 - x12 * y32);
                let x = -(a * y32 - b * y12) / denom;
                let y = (a * x32 - b * x12) / denom;

                let center = PointNd([x, y]);
                let r_sq = (center - found[0]).norm_sq();
                Some(Circle { center, r_sq })
            }
            _ => unreachable!(),
        };
    }

    points.swap(rng.range_u64(0..n as u64) as usize, n - 1);
    let p = points[n - 1];
    if let Some(c) = smallest_enclosing_circ(rng, &mut points[..n - 1], &mut found.clone()) {
        if (c.center - p).norm_sq() <= c.r_sq {
            return Some(c);
        }
    }

    found.push(points[n - 1]);
    smallest_enclosing_circ(rng, &mut points[..n - 1], found)
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = std::io::BufWriter::new(std::io::stdout());

    let n: usize = input.value();
    let mut ps: Vec<Point<f64>> = (0..n)
        .map(|_| PointNd([input.value(), input.value()]))
        .collect();

    let circ =
        smallest_enclosing_circ(&mut cheap_rand::Rng::new(1234), &mut ps, &mut vec![]).unwrap();

    let round = |mut x: f64| {
        if x.abs() < 1e-5 {
            x = 0.;
        }
        (x as f64 * 1e3).round() / 1e3
    };
    writeln!(
        output,
        "{:.3} {:.3}",
        round(circ.center[0]),
        round(circ.center[1])
    )
    .unwrap();
    writeln!(output, "{:.3}", round((circ.r_sq as f64).sqrt())).unwrap();
}
