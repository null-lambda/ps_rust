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
}
