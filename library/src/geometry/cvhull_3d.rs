mod rand {
    // Written in 2015 by Sebastiano Vigna (vigna@acm.org)
    // https://xoshiro.di.unimi.it/splitmix64.c
    use std::ops::Range;

    pub struct SplitMix64(u64);

    impl SplitMix64 {
        pub fn new(seed: u64) -> Self {
            assert_ne!(seed, 0);
            Self(seed)
        }

        // Available on x86-64 and target feature rdrand only.
        #[cfg(target_arch = "x86_64")]
        pub fn from_entropy() -> Self {
            let mut seed = 0;
            unsafe {
                if std::arch::x86_64::_rdrand64_step(&mut seed) == 1 {
                    Self(seed)
                } else {
                    panic!("Failed to get entropy");
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        pub fn from_entropy() -> Self {
            use std::time::{SystemTime, UNIX_EPOCH};
            let seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            Self(seed as u64)
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
            debug_assert!(start < end);

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
    // Todo: factor out Point and Angle from cvhull

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
        use std::collections::HashMap;
        use std::collections::HashSet;

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

        /// # Incremental 3D Convex Hull, with a conflict graph.
        /// Points should be randomly sorted beforehand. Time complexity: O(n log n) on average.
        /// ## Reference:
        /// https://cw.fel.cvut.cz/b221/_media/courses/cg/lectures/05-convexhull-3d.pdf
        pub fn convex_hull_fast<T: Scalar>(points: &mut [Point3<T>]) -> Vec<[u32; 3]> {
            if points.len() <= 2 {
                return vec![];
            }

            let [i, j] = [0, 1];
            let Some(k) = points[2..]
                .iter()
                .position(|&pk| cross(points[j] - points[i], pk - points[i]) != Point3::default())
                .map(|k| k + 2)
            else {
                return vec![];
            };
            let Some(l) = (k + 1 < points.len())
                .then(|| {
                    points[k..].iter().position(|&pl| {
                        signed_vol(points[i], points[j], points[k], pl) != T::zero()
                    })
                })
                .flatten()
                .map(|l| l + k)
            else {
                return (2..points.len()).map(|k| [0, 1, k as u32]).collect();
            };

            points.swap(2, k);
            points.swap(3, l);
            if !(signed_vol(points[0], points[1], points[2], points[3]) > T::zero()) {
                points.swap(2, 3)
            }

            let mut faces: HashSet<_> = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]].into();
            let mut face_left_to_edge: HashMap<[u32; 2], [u32; 3]> = HashMap::new();
            for &f in &faces {
                let [i, j, k] = f;
                face_left_to_edge.insert([i, j], f);
                face_left_to_edge.insert([j, k], f);
                face_left_to_edge.insert([k, i], f);
            }

            // Initialize conflict graph
            let mut visible_faces: Vec<HashSet<[u32; 3]>> = vec![HashSet::new(); points.len()];
            let mut visible_points: HashMap<[u32; 3], Vec<u32>> =
                faces.iter().map(|&f| (f, vec![])).collect();

            for p_idx in 4..points.len() as u32 {
                for (_, &f) in faces.iter().enumerate() {
                    let [i, j, k] = f;
                    if signed_vol(
                        points[p_idx as usize],
                        points[i as usize],
                        points[j as usize],
                        points[k as usize],
                    ) < T::zero()
                    {
                        visible_faces[p_idx as usize].insert(f);
                        visible_points.entry(f).or_default().push(p_idx);
                    }
                }
            }

            for p_idx in 4..points.len() as u32 {
                visible_faces[p_idx as usize].retain(|&f| faces.contains(&f));

                for &f in &visible_faces[p_idx as usize] {
                    faces.remove(&f);
                }

                let triangle_boundary = |[i, j, k]: [u32; 3]| [[i, j], [j, k], [k, i]];
                let visible_half_edges: HashSet<[u32; 2]> = visible_faces[p_idx as usize]
                    .iter()
                    .flat_map(|&face| triangle_boundary(face))
                    .collect();
                let boundary = visible_half_edges
                    .iter()
                    .copied()
                    .filter(|&[i, j]| !visible_half_edges.contains(&[j, i]));

                for [i, j] in boundary {
                    let f_new = [i, j, p_idx];
                    faces.insert(f_new);
                    visible_points.insert(f_new, Default::default());

                    let mut p_next: HashSet<u32> = Default::default();
                    p_next.extend(visible_points[&face_left_to_edge[&[i, j]]].iter().cloned());
                    p_next.extend(visible_points[&face_left_to_edge[&[j, i]]].iter().cloned());

                    for &q in &p_next {
                        if signed_vol(
                            points[q as usize],
                            points[i as usize],
                            points[j as usize],
                            points[p_idx as usize],
                        ) < T::zero()
                        {
                            visible_faces[q as usize].insert(f_new);
                            visible_points.get_mut(&f_new).unwrap().push(q);
                        }
                    }

                    *face_left_to_edge.get_mut(&[i, j]).unwrap() = f_new;
                    face_left_to_edge.insert([j, p_idx], f_new);
                    face_left_to_edge.insert([p_idx, i], f_new);
                }
                for face in &visible_faces[p_idx as usize] {
                    visible_points.remove(face);
                }
            }

            faces.into_iter().collect()
        }
    }
}
