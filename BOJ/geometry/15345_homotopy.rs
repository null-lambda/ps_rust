use std::io::Write;

use geometry::*;

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

        pub fn zero() -> Self {
            Point([T::zero(); 2])
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

fn ordered_pair<T: Ord>(a: T, b: T) -> (T, T) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Intersection {
    Point((i64, i64)),
    Disjoint,
    ColinearOrDisjoint,
}

fn test_seg_inter(
    p1: Point<i64>,
    p2: Point<i64>,
    q1_open: Point<i64>,
    q2: Point<i64>,
) -> Intersection {
    // intersection = p1 + t * (p2 - p1) = q1 + s * (q2 - q1), 0 <= t <= 1, 0 < s <= 1
    // => t (p2 - p1) - s (q2 - q1) + (p1 - q1) = 0
    // => t (p2 - p1) - s (q2 - q1) = q1 - p1
    let pd = p2 - p1;
    let qd = q2 - q1_open;
    let r = q1_open - p1;

    // solve linear equation
    let det = -pd[0] * qd[1] + pd[1] * qd[0];
    let mul_det_t = -qd[1] * r[0] + qd[0] * r[1];
    let mul_det_s = -pd[1] * r[0] + pd[0] * r[1];

    if det != 0 {
        let param_range = ordered_pair(0, det);
        let param_range = param_range.0..=param_range.1;

        if mul_det_s != 0 && param_range.contains(&mul_det_t) && param_range.contains(&mul_det_s) {
            let sign = det.signum() as i64;
            Intersection::Point((sign * mul_det_t, sign * det))
        } else {
            Intersection::Disjoint
        }
    } else {
        Intersection::ColinearOrDisjoint
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    // Problem Statement: Given a space X of a plane minus a finite set of points,
    // determine whether the loop C is contractible into a point (null-homotopic).
    //
    // Approach: Contract X into a 1-skeleton, a graph embedded in the plane.
    // Implementation details:
    //     1. Connect all N points into a single piecewise linear path. Augment a ray at the first endpoint
    //        (in other words, connect the first endpoint with the point at infinity).
    //        Ensure that no self-intersections occur.
    //     2. The dual graph D of G is the desired 1-skeleton, which has 1 vertex and N self-edges.
    //        Convert the given loop to a cycle in G by detecting every segment intersection
    //        with G. Time complexity: O(N^2 log N).
    //     3. Label each directed edge with some alphabets { e(1), e(1)^-1, e(2), e(2)^-1, ..., e(n+1), e(n+1)^-1 },
    //        which constitute the free group of rank N+1, equivalent to the homotopy group of G or X.
    //        Convert C into a word in the free group, e.g., w = e(1) e(2) e(1)^-1 e(2)^-1.
    //     4. Determine whether the word is equal to the identity, modulo {e(i) e(i)^-1}.

    loop {
        let n: usize = input.value();
        let m: usize = input.value();
        let k: usize = input.value();
        if (n, m, k) == (0, 0, 0) {
            break;
        }

        let mut read_point = || Point::<i64>::new(input.value(), input.value());
        let s = read_point();
        let t = read_point();
        let mut holes: Vec<_> = (0..n).map(|_| read_point()).collect();
        let path1: Vec<_> = (0..m).map(|_| read_point()).collect();
        let path2: Vec<_> = (0..k).map(|_| read_point()).collect();
        let cycle: Vec<_> = std::iter::once(s)
            .chain(path1)
            .chain(std::iter::once(t))
            .chain(path2.into_iter().rev())
            .chain(std::iter::once(s))
            .collect();

        // println!("{:?}", cycle);

        // Step 1.
        let point_at_pseudo_infinity = Point::new(-100_000_000, 0);
        holes.push(point_at_pseudo_infinity);
        holes.sort_unstable_by_key(|p| (p[0], p[1]));
        holes.dedup();

        // Step 2, 3.
        let mut cycle_word = vec![];
        let label = |u| u as i32 + 1;
        let label_inv = |u| -label(u);
        for w in cycle.windows(2) {
            let [c0, c1] = [w[0], w[1]];
            let mut subword = vec![];
            for (u, separator) in holes.windows(2).enumerate() {
                let [s1_open, s2] = [separator[0], separator[1]];

                if let Intersection::Point(time) = test_seg_inter(c0, c1, s1_open, s2) {
                    let side0 = geometry::signed_area(s1_open, s2, c0) > 0;
                    let side1 = geometry::signed_area(s1_open, s2, c1) > 0;
                    match (side0, side1) {
                        (true, false) => subword.push((time, label(u))),
                        (false, true) => subword.push((time, label_inv(u))),
                        _ => {}
                    }
                }
            }
            subword.sort_unstable_by(|(t1, _), (t2, _)| {
                (t1.0 as i128 * t2.1 as i128).cmp(&(t2.0 as i128 * t1.1 as i128))
            });

            // println!("{:?}", subword);
            cycle_word.extend(subword.into_iter().map(|(_, v)| v));
        }

        // Step 4.
        let mut reduced = vec![];
        for &u in &cycle_word {
            if let Some(&v) = reduced.last() {
                if v == -u {
                    reduced.pop();
                } else {
                    reduced.push(u);
                }
            } else {
                reduced.push(u);
            }
        }

        // println!("{:?}", n_verts);
        // println!("{:?}", cycle_word);
        // println!("{:?}", reduced);
        // println!();

        let ans = reduced.is_empty();
        writeln!(output, "{}", if ans { "Yes" } else { "No" }).unwrap();
    }
}
