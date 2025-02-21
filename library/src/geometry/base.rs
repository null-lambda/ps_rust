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
    }

    impl Scalar for i64 {
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
}
