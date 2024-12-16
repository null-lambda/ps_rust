use std::collections::HashSet;
use std::io::Write;

use geometry::dim3::Point3;

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
pub mod geometry {
    pub mod dim3 {
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

        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct Point3<T>([T; 3]);

        impl<T: Scalar> Point3<T> {
            pub fn new(x: T, y: T, z: T) -> Self {
                Point3([x, y, z])
            }

            pub fn zero() -> Self {
                Point3([T::zero(), T::zero(), T::zero()])
            }

            pub fn dot(self, other: Self) -> T {
                self[0] * other[0] + self[1] * other[1] + self[2] * other[2]
            }

            pub fn norm_sq(self) -> T {
                self.dot(self)
            }

            pub fn cross(self, other: Self) -> Point3<T> {
                Point3([
                    self[1] * other[2] - self[2] * other[1],
                    self[2] * other[0] - self[0] * other[2],
                    self[0] * other[1] - self[1] * other[0],
                ])
            }
        }

        impl<T: Scalar> From<[T; 3]> for Point3<T> {
            fn from(p: [T; 3]) -> Self {
                Point3(p)
            }
        }

        impl<T: Scalar> Index<usize> for Point3<T> {
            type Output = T;
            fn index(&self, i: usize) -> &Self::Output {
                &self.0[i]
            }
        }

        impl<T: Scalar> IndexMut<usize> for Point3<T> {
            fn index_mut(&mut self, i: usize) -> &mut Self::Output {
                &mut self.0[i]
            }
        }

        impl<T: Scalar> Point3<T> {
            pub fn map<S: Scalar>(self, f: impl FnMut(T) -> S) -> Point3<S> {
                Point3(self.0.map(f))
            }

            pub fn try_map<S: Scalar>(
                self,
                mut f: impl FnMut(T) -> Option<S>,
            ) -> Option<Point3<S>> {
                let mut res = [S::zero(); 3];
                for i in 0..3 {
                    res[i] = f(self[i])?;
                }
                Some(Point3(res))
            }
        }

        macro_rules! impl_binop {
            ($trait:ident, $fn:ident) => {
                impl<T: Scalar> $trait for Point3<T> {
                    type Output = Self;
                    fn $fn(self, other: Self) -> Self::Output {
                        Point3([
                            self[0].$fn(other[0]),
                            self[1].$fn(other[1]),
                            self[2].$fn(other[2]),
                        ])
                    }
                }
            };
        }

        impl_binop!(Add, add);
        impl_binop!(Sub, sub);
        impl_binop!(Mul, mul);

        impl<T: Scalar> Mul<T> for Point3<T> {
            type Output = Self;
            fn mul(self, k: T) -> Self::Output {
                Point3([self[0].mul(k), self[1].mul(k), self[2].mul(k)])
            }
        }

        pub fn is_colinear<T: Scalar>(a: Point3<T>, b: Point3<T>, c: Point3<T>) -> bool {
            (b - a).cross(c - a) == Point3::zero()
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut ps: Vec<Point3<i32>> = (0..n)
        .map(|_| [input.value(), input.value(), input.value()].into())
        .collect();
    ps.sort_unstable();
    let ps_set: HashSet<Point3<i32>> = ps.iter().cloned().collect();
    let contained_in_ps = |p: Point3<i64>| {
        let Some(p) = p.try_map(|x| i32::try_from(x).ok()) else {
            return false;
        };
        ps_set.contains(&p)
    };

    let mut m_max = 0u32;

    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                let pi = ps[i].map(i64::from);
                let pj = ps[j].map(i64::from);
                let pk = ps[k].map(i64::from);
                if geometry::dim3::is_colinear(pi, pj, pk) {
                    continue;
                }
                m_max = m_max.max(3);

                let q = pj + pk - pi;
                if contained_in_ps(q) {
                    m_max = m_max.max(4);
                }

                let d1 = pj - pi;
                let d2 = pk - pi;
                let dq = q - pi;

                let r0 = pi + dq * 2;
                let r1 = r0 - d1;
                let r2 = r0 - d2;

                if contained_in_ps(r0) && contained_in_ps(r1) && contained_in_ps(r2) {
                    m_max = m_max.max(6);
                }
            }
        }
    }

    let ans = n as u32 * m_max;
    writeln!(output, "{}", ans).unwrap();
}
