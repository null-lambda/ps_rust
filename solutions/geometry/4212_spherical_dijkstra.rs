use core::f64;
use geometry::{Ordered, Point3};
use std::{cmp::Reverse, collections::BinaryHeap, f64::consts::PI, io::Write};

mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> Option<T>
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().ok()
        }
    }

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

pub mod cmp {
    use std::cmp::Ordering;

    // x <= y iff x = y
    #[derive(Debug, Copy, Clone, Default)]
    pub struct Trivial<T>(pub T);

    impl<T> PartialEq for Trivial<T> {
        #[inline]
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    impl<T> Eq for Trivial<T> {}

    impl<T> PartialOrd for Trivial<T> {
        #[inline]
        fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
            Some(Ordering::Equal)
        }
    }

    impl<T> Ord for Trivial<T> {
        #[inline]
        fn cmp(&self, _other: &Self) -> Ordering {
            Ordering::Equal
        }
    }
}

#[macro_use]
pub mod geometry {
    use core::f64;
    use std::{
        f64::consts::PI,
        ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub},
    };

    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
    pub struct Ordered<T>(T);

    impl From<f64> for Ordered<f64> {
        fn from(x: f64) -> Self {
            debug_assert!(!x.is_nan());
            Self(x)
        }
    }

    impl Into<f64> for Ordered<f64> {
        fn into(self) -> f64 {
            self.0
        }
    }

    impl From<f32> for Ordered<f64> {
        fn from(x: f32) -> Self {
            debug_assert!(x.is_finite());
            Self(x as f64)
        }
    }

    impl<T: PartialEq> Eq for Ordered<T> {}
    impl<T: PartialOrd> Ord for Ordered<T> {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.partial_cmp(other).unwrap()
        }
    }

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

        fn two() -> Self {
            Self::one() + Self::one()
        }

        fn min(self, other: Self) -> Self {
            if self < other {
                self
            } else {
                other
            }
        }

        fn max(self, other: Self) -> Self {
            if self < other {
                other
            } else {
                self
            }
        }

        fn abs(self) -> Self {
            if self < Self::zero() {
                -self
            } else {
                self
            }
        }

        fn sq(self) -> Self {
            self * self
        }
    }

    impl Scalar for f64 {
        fn one() -> Self {
            1.0
        }
    }

    impl Scalar for i64 {
        fn one() -> Self {
            1
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct PointNd<const N: usize, T>(pub [T; N]);

    impl<const N: usize, T: Scalar> PointNd<N, T> {
        fn map<F, S>(self, mut f: F) -> PointNd<N, S>
        where
            F: FnMut(T) -> S,
        {
            PointNd(self.0.map(|x| f(x)))
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
    impl_binop!(Div, div);

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

    impl<const N: usize, T: Scalar> PointNd<N, T> {
        pub fn zero() -> Self {
            Self::default()
        }

        pub fn dot(self, other: Self) -> T {
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| a * b)
                .reduce(|acc, x| acc + x)
                .unwrap()
        }

        pub fn norm_sq(self) -> T {
            self.dot(self)
        }

        pub fn max_norm(self) -> T {
            self.0
                .into_iter()
                .map(|a| a.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        }
    }

    pub type Point3<T> = PointNd<3, T>;

    impl<T> Point3<T> {
        pub fn new(x: T, y: T, z: T) -> Self {
            PointNd([x, y, z])
        }
    }

    impl<T: Scalar> Point3<T> {
        pub fn cross(self, other: Self) -> Self {
            Point3::new(
                self[1] * other[2] - self[2] * other[1],
                self[2] * other[0] - self[0] * other[2],
                self[0] * other[1] - self[1] * other[0],
            )
        }
    }

    impl Point3<f64> {
        pub fn normalized(self) -> Self {
            self * self.norm_sq().powf(-0.5)
        }
        pub fn sph(phi: f64, theta: f64) -> Self {
            Point3::new(
                phi.cos() * theta.cos(),
                phi.sin() * theta.cos(),
                theta.sin(),
            )
            .normalized()
        }
    }

    pub fn solve_quad(a: f64, b: f64, c: f64) -> Option<(f64, f64)> {
        debug_assert!((1.0 / (2.0 * a)).is_finite());
        let d = b * b - 4.0 * a * c;
        if d < 0.0 {
            return None;
        }
        let sqrt_d = d.sqrt();
        let x1 = (-b - sqrt_d) / (2.0 * a);
        let x2 = (-b + sqrt_d) / (2.0 * a);
        Some((x1, x2))
    }

    pub fn airport_inter(p1: Point3<f64>, p2: Point3<f64>, rcos: f64) -> Option<[Point3<f64>; 2]> {
        debug_assert!(rcos > 0.0);
        debug_assert!((p1.norm_sq() - 1.0).abs() < 1e-9);
        debug_assert!((p2.norm_sq() - 1.0).abs() < 1e-9);

        let m = (p1 + p2).normalized();

        let h = m.cross(p2 - p1).normalized();
        let t = ((m.dot(p1) / rcos).sq() - 1.0).sqrt();
        if !t.is_finite() {
            return None;
        }

        let mut l1 = m + h * t;
        let mut l2 = m - h * t;

        let l_inv = l1.norm_sq().powf(-0.5);
        if !l_inv.is_finite() {
            return None;
        }
        l1 = l1 * l_inv;
        l2 = l2 * l_inv;

        debug_assert!((rcos - l1.dot(p1)).abs() < 1e-9);
        debug_assert!((rcos - l2.dot(p1)).abs() < 1e-9);
        debug_assert!((rcos - l1.dot(p2)).abs() < 1e-9);
        debug_assert!((rcos - l2.dot(p2)).abs() < 1e-9);

        Some([l1, l2])
    }

    pub fn path_inter(
        p1: Point3<f64>,
        p2: Point3<f64>,
        q: Point3<f64>,
        rcos: f64,
    ) -> Option<(f64, f64)> {
        // if (p1 - p2).norm_sq() < 1e-9 {
        //     // Zero length path
        //     return (p1.dot(q) >= rcos - 1e-9).then(|| (0.0, 1.0));
        // }
        // if (p1 + p2).norm_sq() < 0.0 {
        //     // We may ignore any path consting of antipodal points,
        //     // since they are always decomposable into sub-paths.
        //     return None;
        // }

        // let gamma = p1.cross(p2).norm_sq().sqrt().atan2(p1.dot(p2));
        let gamma = p1.dot(p2).acos();

        let n = p1.cross(p2);
        let u = p1;
        let v = n.cross(u).normalized();

        let a = u.dot(q);
        let b = v.dot(q);
        let l = f64::hypot(a, b);
        let alpha = f64::atan2(b, a);
        let beta = (rcos / l).acos();
        if !beta.is_finite() {
            return None;
        }
        debug_assert!((0.0..=PI / 2.0).contains(&beta));

        let theta0 = (alpha - beta).rem_euclid(2.0 * PI);
        let theta1 = (alpha + beta).rem_euclid(2.0 * PI);

        debug_assert!(
            !theta0.is_finite()
                || ((u * theta0.cos() + v * theta0.sin()).dot(q) - rcos).abs() < 1e-9
        );
        debug_assert!(
            !theta1.is_finite()
                || ((u * theta1.cos() + v * theta1.sin()).dot(q) - rcos).abs() < 1e-9
        );

        // clip to [0, gamma], mod 2PI (It is garanteed that there are at most one interval)
        let (theta0, theta1) = if theta0 <= theta1 {
            (theta0.min(gamma), theta1.min(gamma))
        } else {
            debug_assert!(theta0 >= gamma - 1e-9);
            (0.0, theta1.min(gamma))
        };
        let (theta0, theta1) = (theta0 / gamma, theta1 / gamma);

        if !theta0.is_finite() || !theta1.is_finite() {
            return None;
        }

        Some((theta0, theta1))
    }

    pub fn sph_seg_len(p1: Point3<f64>, p2: Point3<f64>) -> f64 {
        debug_assert!((p1.norm_sq() - 1.0).abs() < 1e-9);
        debug_assert!((p2.norm_sq() - 1.0).abs() < 1e-9);

        p1.dot(p2).min(1.0).max(-1.0).acos()
    }
}

fn debug_to_obj(
    n: usize,
    points: &[Point3<f64>],
    neighbors: &[Vec<(usize, f64)>],
    r: f64,
    i_tc: usize,
) {
    // Graphical debugging
    // Use .obj viewer such as: https://3dviewer.net/
    if !cfg!(debug_assertions) {
        return;
    }

    let mut obj = Vec::new();

    for i in 0..points.len() {
        writeln!(obj, "v {} {} {}", points[i][0], points[i][1], points[i][2]).unwrap();
    }
    for u in 0..points.len() {
        for &(v, _) in &neighbors[u] {
            if u < v {
                writeln!(obj, "l {} {}", u + 1, v + 1).unwrap();
            }
        }
    }

    let mut idx0 = points.len();
    for i in 0..n {
        // draw spherical circle (lines) with radius r

        let mut draw_circle = |p: Point3<f64>, r: f64| {
            let mut ax1 = Point3::new(1.0, 0.0, 0.0).cross(p).normalized();
            if !((ax1.norm_sq() - 1.0).abs() < 1e-9) {
                ax1 = Point3::new(0.0, 1.0, 0.0).cross(p).normalized();
            }
            let mut ax2 = p.cross(ax1).normalized();

            let center = p * r.cos();
            ax1 = ax1 * r.sin();
            ax2 = ax2 * r.sin();

            let n_poly = 100;
            for j in 0..n_poly {
                let theta = 2.0 * PI * j as f64 / n_poly as f64;
                let p0 = center + ax1 * theta.cos() + ax2 * theta.sin();
                writeln!(obj, "v {} {} {}", p0[0], p0[1], p0[2]).unwrap();
            }
            for j0 in 0..n_poly {
                let j1 = (j0 + 1) % n_poly;
                writeln!(obj, "l {} {}", idx0 + j0 + 1, idx0 + j1 + 1).unwrap();
            }
            idx0 += n_poly;
        };

        draw_circle(points[i], r);
        draw_circle(points[i], r * 0.1);
    }

    std::fs::write(format!("debug_{}.obj", i_tc), obj).unwrap();
}

fn dijkstra2(neighbors: &[Vec<(usize, f64)>], start: usize) -> Vec<f64> {
    let mut dist = vec![f64::INFINITY; neighbors.len()];
    dist[start] = 0.0;
    let mut visited = vec![false; neighbors.len()];

    for _ in 0..neighbors.len() {
        let u = (0..neighbors.len())
            .filter(|&u| !visited[u])
            .min_by_key(|&u| Ordered::from(dist[u]))
            .unwrap();
        visited[u] = true;
        for &(v, w) in &neighbors[u] {
            dist[v] = dist[v].min(dist[u] + w);
        }
    }
    dist
}

fn dijkstra3(n: usize, dist_mat: &[Vec<f64>], fuel: f64, start: usize, end: usize) -> f64 {
    let mut dist = vec![f64::INFINITY; n];
    dist[start] = 0.0;
    let mut visited = vec![false; n];
    for _ in 0..n {
        let u = (0..n)
            .filter(|&u| !visited[u])
            .min_by_key(|&u| Ordered::from(dist[u]))
            .unwrap();
        visited[u] = true;
        if u == end {
            break;
        }
        for v in 0..n {
            let w = dist_mat[u][v];
            if w >= fuel + 1e-9 {
                continue;
            }
            dist[v] = dist[v].min(dist[u] + w);
        }
    }
    dist[end]
}

fn main() {
    use std::f64::consts::PI;
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    for i_tc in 1.. {
        let Some(n): Option<usize> = input.value() else {
            break;
        };
        let r0: f64 = 6370.0;
        let r: f64 = input.value::<f64>().unwrap() / r0;
        let rcos: f64 = r.cos();

        let points: Vec<Point3<f64>> = (0..n)
            .map(|_| {
                let phi = input.value::<f64>().unwrap() * PI / 180.0;
                let theta = input.value::<f64>().unwrap() * PI / 180.0;
                Point3::sph(phi, theta)
            })
            .collect();
        let mut points = points.clone();
        let mut neighbors: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
        for i in 0..n {
            for j in i + 1..n {
                if let Some([l1, l2]) = geometry::airport_inter(points[i], points[j], rcos) {
                    // let l1_idx = points.len();
                    // let l2_idx = points.len() + 1;
                    points.push(l1);
                    points.push(l2);
                    neighbors.push(vec![]);
                    neighbors.push(vec![]);
                }
            }
        }
        debug_assert_eq!(points.len(), neighbors.len());

        let mut check_path = |u: usize, v: usize| {
            let mut intervals: Vec<(f64, f64)> = (0..n)
                .flat_map(|q| {
                    let (x0, x1) = geometry::path_inter(points[u], points[v], points[q], rcos)?;
                    Some(((x0 - 1e-9).max(0.0), (x1 + 1e-9).min(1.0)))
                })
                .collect();
            intervals.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // test whether (merged interval) = [0.0, 1.0]
            let mut merged: Option<(f64, f64)> = None;
            for &(x0, x1) in intervals.iter() {
                if merged.is_none() {
                    merged = Some((x0, x1));
                } else if merged.unwrap().1 < x0 {
                    merged = None;
                    break;
                } else {
                    let (_, end) = merged.as_mut().unwrap();
                    *end = end.max(x1);
                }
            }
            if merged == Some((0.0, 1.0)) {
                let d = geometry::sph_seg_len(points[u], points[v]);
                neighbors[u].push((v, d));
                neighbors[v].push((u, d));
            }
        };
        for u in 0..points.len() {
            for v in u + 1..points.len() {
                check_path(u, v);
            }
        }

        let mut new_dist: Vec<Vec<f64>> = vec![];
        for u in 0..n {
            let mut dist_u = dijkstra2(&neighbors, u);
            dist_u.resize(n, f64::INFINITY);
            new_dist.push(dist_u);
        }

        let q: usize = input.value().unwrap();
        writeln!(output, "Case {}:", i_tc).unwrap();
        for _ in 0..q {
            let u: usize = input.value::<usize>().unwrap() - 1;
            let v: usize = input.value::<usize>().unwrap() - 1;
            let mut fuel: f64 = input.value().unwrap();
            fuel /= r0;

            let dist_uv = dijkstra3(n, &new_dist, fuel, u, v);

            if dist_uv < f64::INFINITY {
                let round = |x: f64| (x * 1000.0).round() / 1000.0;
                writeln!(output, "{:.3}", round(dist_uv * r0)).unwrap();
            } else {
                writeln!(output, "impossible").unwrap();
            }
        }
        debug_to_obj(n, &points, &neighbors, r, i_tc);
    }
}
