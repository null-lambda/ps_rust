use std::io::Write;

use geometry::{half_plane::HalfPlane, Point};

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

        fn one() -> Self;

        fn abs(self) -> Self {
            if self < Self::zero() {
                -self
            } else {
                self
            }
        }

        fn signum(self) -> Self {
            match (&self).cmp(&Self::zero()) {
                Ordering::Less => -Self::one(),
                Ordering::Equal => Self::zero(),
                Ordering::Greater => Self::one(),
            }
        }
    }

    impl Scalar for i64 {
        fn one() -> Self {
            1
        }
    }

    impl Scalar for i128 {
        fn one() -> Self {
            1
        }
    }

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

    pub mod half_plane {
        use std::collections::VecDeque;

        use super::*;

        // A half plane, defined as { x : n dot x < s }
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct HalfPlane<T> {
            pub normal_outward: Point<T>,
            pub shift: T,
        }

        impl<T: Scalar> HalfPlane<T> {
            pub fn left_side(s: Point<T>, e: Point<T>) -> Self {
                debug_assert!(s != e);
                let normal_outward = (s - e).rot();
                let shift = normal_outward.dot(s);
                HalfPlane {
                    normal_outward,
                    shift,
                }
            }

            // Avoid division at all costs
            pub fn inter_frac(&self, other: &Self) -> Option<(Point<T>, T)> {
                let det = self.normal_outward.cross(other.normal_outward);
                let x_mul_det =
                    self.shift * other.normal_outward[1] - other.shift * self.normal_outward[1];
                let y_mul_det =
                    self.normal_outward[0] * other.shift - other.normal_outward[0] * self.shift;
                (det != T::zero()).then(|| ([x_mul_det, y_mul_det].into(), det))
            }

            pub fn antiparallel(&self, other: &Self) -> bool {
                self.normal_outward.cross(other.normal_outward) == T::zero()
                    && self.normal_outward.dot(other.normal_outward) < T::zero()
            }

            pub fn contains_frac(&self, (numer, denom): (Point<T>, T)) -> bool {
                self.normal_outward.dot(numer) * denom.signum() < self.shift * denom.abs()
            }
        }

        pub fn bbox<T: Scalar>(
            bottom_left: Point<T>,
            top_right: Point<T>,
        ) -> impl Iterator<Item = HalfPlane<T>> {
            let bottom_right = Point::new(top_right[0], bottom_left[1]);
            let top_left = Point::new(bottom_left[0], top_right[1]);
            [
                HalfPlane::left_side(bottom_left, bottom_right),
                HalfPlane::left_side(bottom_right, top_right),
                HalfPlane::left_side(top_right, top_left),
                HalfPlane::left_side(top_left, bottom_left),
            ]
            .into_iter()
        }

        // pub fn intersection<T: Scalar>(
        pub fn intersection<T: Scalar + std::fmt::Debug>(
            // WIP
            half_planes: impl IntoIterator<Item = HalfPlane<T>>,
            bottom_left: Point<T>,
            top_right: Point<T>,
        ) -> VecDeque<HalfPlane<T>> {
            let mut half_planes: Vec<_> = half_planes
                .into_iter()
                .chain(bbox(bottom_left, top_right)) // Handling caseworks without a bbox is a huge pain.
                .collect();
            half_planes.sort_unstable_by_key(|h| (Angle(h.normal_outward), h.shift));
            half_planes.dedup_by_key(|h| Angle(h.normal_outward)); // Dedup parallel half planes

            let mut half_planes = half_planes.into_iter();

            let mut inter = VecDeque::new();
            inter.extend(half_planes.next());
            inter.extend(half_planes.next());

            for h in half_planes {
                while inter.len() >= 2 {
                    let [l, m] = [&inter[inter.len() - 2], &inter[inter.len() - 1]];

                    if l.inter_frac(m).map_or(true, |p| h.contains_frac(p)) {
                        break;
                    }
                    inter.pop_back();
                }

                while inter.len() >= 2 {
                    let [l, m] = [&inter[0], &inter[1]];
                    if l.inter_frac(m).map_or(true, |p| h.contains_frac(p)) {
                        break;
                    }
                    inter.pop_front();
                }

                let l = &inter[inter.len() - 1];
                if h.antiparallel(l) {
                    let det = h.shift * (l.normal_outward[0].abs() + l.normal_outward[1].abs())
                        - l.shift
                            * (h.normal_outward[0] * l.normal_outward[0].signum()
                                + h.normal_outward[1] * l.normal_outward[1].signum());
                    if det <= T::zero() {
                        // Exclude boundary
                        return Default::default();
                    }
                    //if det < 0 {
                    //    //Include boundary
                    //    return Default::default();
                    //}
                }

                inter.push_back(h);
            }

            while inter.len() >= 3 {
                let [l, m, h] = [&inter[inter.len() - 2], &inter[inter.len() - 1], &inter[0]];
                if l.inter_frac(m).map_or(true, |p| h.contains_frac(p)) {
                    break;
                }
                inter.pop_back();
            }
            while inter.len() >= 3 {
                let [l, m, h] = [&inter[inter.len() - 1], &inter[0], &inter[1]];
                if l.inter_frac(m).map_or(true, |p| h.contains_frac(p)) {
                    break;
                }
                inter.pop_front();
            }

            if inter.len() < 3 {
                return Default::default();
            }

            inter
        }
    }
}

pub mod ordered {
    use std::{
        cmp::Ordering,
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    };

    #[derive(Clone, Copy, Default)]
    pub struct F64(pub f64);

    impl F64 {
        pub fn new(x: f64) -> Self {
            Self(x)
        }

        pub fn map_inner(self, f: impl FnOnce(f64) -> f64) -> Self {
            Self(f(self.0))
        }
    }

    impl PartialEq for F64 {
        fn eq(&self, other: &Self) -> bool {
            self.0.total_cmp(&other.0).is_eq()
        }
    }

    impl Eq for F64 {}

    impl PartialOrd for F64 {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.0.total_cmp(&other.0))
        }
    }

    impl Ord for F64 {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.total_cmp(&other.0)
        }
    }

    impl std::hash::Hash for F64 {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.0.to_bits().hash(state);
        }
    }

    impl std::fmt::Debug for F64 {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    macro_rules! impl_binop {
        ($trait:ident, $fn:ident, $trait_assign:ident, $fn_assign:ident) => {
            impl $trait for F64 {
                type Output = Self;
                fn $fn(self, other: Self) -> Self::Output {
                    Self(self.0.$fn(other.0))
                }
            }

            impl $trait_assign for F64 {
                fn $fn_assign(&mut self, other: Self) {
                    self.0.$fn_assign(other.0);
                }
            }
        };
    }

    impl Neg for F64 {
        type Output = Self;
        fn neg(self) -> Self::Output {
            Self(-self.0)
        }
    }

    impl_binop!(Add, add, AddAssign, add_assign);
    impl_binop!(Sub, sub, SubAssign, sub_assign);
    impl_binop!(Mul, mul, MulAssign, mul_assign);
    impl_binop!(Div, div, DivAssign, div_assign);

    impl super::geometry::Scalar for F64 {
        fn one() -> Self {
            Self(1.0)
        }
    }
}

fn partition_point_f64<P>(
    mut left: f64,
    mut right: f64,
    eps: f64,
    mut max_iter: u32,
    mut pred: P,
) -> f64
where
    P: FnMut(f64) -> bool,
{
    while right - left > eps && max_iter > 0 {
        let mid = left + (right - left) / 2.0;
        if pred(mid) {
            left = mid;
        } else {
            right = mid;
        }
        max_iter -= 1;
    }
    left
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    loop {
        let n: usize = input.value();
        if n == 0 {
            break;
        }

        let ps = (0..n)
            .map(|_| Point::new(ordered::F64(input.value()), ordered::F64(input.value())))
            .collect::<Vec<_>>();

        const X_MAX: f64 = 10010.0;
        let coverable = |r: f64| {
            let hs = ps.iter().zip(ps.iter().cycle().skip(1)).map(|(&p, &q)| {
                let mut h = HalfPlane::left_side(p, q);
                let n = h.normal_outward.norm_sq().0.sqrt();
                h.shift -= ordered::F64(n * r);
                h
            });
            let inter = geometry::half_plane::intersection(
                hs,
                Point::new(ordered::F64(-X_MAX), ordered::F64(-X_MAX)),
                Point::new(ordered::F64(X_MAX), ordered::F64(X_MAX)),
            );
            !inter.is_empty()
        };
        let r = partition_point_f64(0.0, X_MAX, 1e-9, 300, coverable);
        writeln!(output, "{}", r).unwrap();
    }
}
