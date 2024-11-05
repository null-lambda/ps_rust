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
mod geometry {
    use std::cmp::{max, min, Ordering};
    use std::ops::{Add, Mul, Neg, Sub};

    pub trait Zero: Sized + Add<Self, Output = Self> {
        fn zero() -> Self;
    }

    pub trait One: Sized + Add<Self, Output = Self> {
        fn one() -> Self;
    }

    macro_rules! trait_const_impl {
    ($trait_name:ident, $const_name:ident, $v:expr, $($t:ty)*) => {$(
        impl $trait_name for $t {
            #[inline]
            fn $const_name() -> $t {
                $v
            }
        }
    )*}
}

    trait_const_impl! {Zero, zero, 0, isize i8 i16 i32 i64 i128}
    trait_const_impl! {Zero, zero, 0, usize u8 u16 u32 u64 u128}
    trait_const_impl! {Zero, zero, 0.0, f32 f64}

    trait_const_impl! {One, one, 1, isize i8 i16 i32 i64 i128}
    trait_const_impl! {One, one, 1, usize u8 u16 u32 u64 u128}
    trait_const_impl! {One, one, 1.0, f32 f64}

    macro_rules! trait_alias {
    ($name:ident = $($value:tt)+) => {
        pub trait $name: $($value)+ {}
        impl<T> $name for T where T: $($value)+ {}
    };
}

    trait_alias! { Scalar = Copy + Clone + Add<Output = Self> + Sub<Output = Self> + Neg<Output = Self> + Mul<Output = Self> + Zero + One + PartialOrd + PartialEq }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Point<T: Scalar> {
        pub x: T,
        pub y: T,
    }

    impl<T: Scalar> From<(T, T)> for Point<T> {
        fn from(p: (T, T)) -> Self {
            Self { x: p.0, y: p.1 }
        }
    }

    impl<T: Scalar> From<Point<T>> for (T, T) {
        fn from(p: Point<T>) -> Self {
            (p.x, p.y)
        }
    }

    impl<T: Scalar> Add for Point<T> {
        type Output = Self;
        fn add(self, other: Self) -> Self {
            Self {
                x: self.x + other.x,
                y: self.y + other.y,
            }
        }
    }

    impl<T: Scalar> Sub for Point<T> {
        type Output = Self;
        fn sub(self, other: Self) -> Self {
            Self {
                x: self.x - other.x,
                y: self.y - other.y,
            }
        }
    }

    impl<T: Scalar> Neg for Point<T> {
        type Output = Self;
        fn neg(self) -> Self {
            Self {
                x: -self.x,
                y: -self.y,
            }
        }
    }

    impl<T: Scalar> Mul<T> for Point<T> {
        type Output = Self;
        fn mul(self, other: T) -> Self {
            Self {
                x: self.x * other,
                y: self.y * other,
            }
        }
    }

    impl<T: Scalar> Zero for Point<T> {
        fn zero() -> Self {
            Self {
                x: T::zero(),
                y: T::zero(),
            }
        }
    }

    //  check t, s in closed interval [0, 1]
    fn reorder<T: Ord>(a: T, b: T) -> (T, T) {
        if a < b {
            (a, b)
        } else {
            (b, a)
        }
    }

    // Returns signed area of a triangle; positive if ccw
    fn signed_area<T: Scalar>(p: &Point<T>, q: &Point<T>, r: &Point<T>) -> T {
        (q.x - p.x) * (r.y - p.y) + (r.x - p.x) * (p.y - q.y)
    }

    pub enum Intersection<T: Scalar> {
        Disjoint,
        Point(Point<T>),
        Segment,
    }

    pub fn segment_intersection(
        p1: Point<i64>,
        p2: Point<i64>,
        q1: Point<i64>,
        q2: Point<i64>,
    ) -> Intersection<f64> {
        // intersection = p1 + t * (p2 - p1) = q1 + s * (q2 - q1),
        // => t (p2 - p1) - s (q2 - q1) + (p1 - q1) = 0
        // => t (p2 - p1) - s (q2 - q1) = q1 - p1
        let pd = p2 - p1;
        let qd = q2 - q1;
        let r = q1 - p1;

        // solve linear equation
        let det = -pd.x * qd.y + pd.y * qd.x;
        let mul_det_t = -qd.y * r.x + qd.x * r.y;
        let mul_det_s = -pd.y * r.x + pd.x * r.y;

        if i64::zero() != det {
            let param_range = reorder(i64::zero(), det);
            let param_range = param_range.0..=param_range.1;
            if param_range.contains(&mul_det_t) && param_range.contains(&mul_det_s) {
                let i1 = p1 * det + pd * mul_det_t;
                let i1: Point<_> = (i1.x as f64, i1.y as f64).into();
                Intersection::Point(i1 * (f64::one() / (det as f64)))
            } else {
                Intersection::Disjoint
            }
        } else {
            if signed_area(&Point::zero(), &pd, &r) == i64::zero() {
                let ((a1, a2), (b1, b2)) = if p1.x != p2.x {
                    (reorder(p1.x, p2.x), reorder(q1.x, q2.x))
                } else {
                    (reorder(p1.y, p2.y), reorder(q1.y, q2.y))
                };
                match &max(a1, b1).cmp(&min(a2, b2)) {
                    Ordering::Less => Intersection::Segment,
                    Ordering::Equal => Intersection::Point({
                        if p1 == q1 || p1 == q2 {
                            (p1.x as f64, p1.y as f64).into()
                        } else {
                            (p2.x as f64, p2.y as f64).into()
                        }
                    }),
                    Ordering::Greater => Intersection::Disjoint,
                }
            } else {
                Intersection::Disjoint
            }
        }
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
            while matches!(lower.as_slice(), &[.., l1, l2] if signed_area(&p, &l1, &l2) <= T::zero())
            {
                lower.pop();
            }
            lower.push(p);
        }
        for &p in points.iter().rev() {
            while matches!(upper.as_slice(), &[.., l1, l2] if signed_area(&p, &l1, &l2) <= T::zero())
            {
                upper.pop();
            }
            upper.push(p);
        }
        lower.pop();
        upper.pop();

        lower.extend(upper);
        lower
    }

    pub fn convex_hull_contains<T: Scalar>(cvhull: &[Point<T>], p: Point<T>, tol: T) -> bool {
        if cvhull.len() == 1 {
            return cvhull[0] == p;
        }

        let edges = || {
            cvhull
                .iter()
                .zip(cvhull.iter().skip(1).chain(cvhull.iter().take(1)))
        };
        edges().all(|(&a, &b)| signed_area(&a, &b, &p) <= tol)
            || edges().all(|(&a, &b)| signed_area(&a, &b, &p) >= -tol)
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

use geometry::*;

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut ps: Vec<Point<i64>> = (0..n)
        .map(|_| Point::from((input.value(), input.value())))
        .collect();
    let mut qs: Vec<Point<i64>> = (0..m)
        .map(|_| Point::from((input.value(), input.value())))
        .collect();

    let ps_hull = convex_hull(&mut ps);
    let qs_hull = convex_hull(&mut qs);

    let edges = |ps: &[Point<i64>]| {
        ps.iter()
            .copied()
            .zip(ps.iter().skip(1).chain(ps.iter().take(1)).copied())
            .collect::<Vec<_>>()
    };

    let ps_edges = edges(&ps_hull);
    let qs_edges = edges(&qs_hull);
    let inter: Vec<Point<f64>> = ps_edges
        .iter()
        .flat_map(|&(a, b)| qs_edges.iter().map(move |&(c, d)| (a, b, c, d)))
        .flat_map(|(a, b, c, d)| match segment_intersection(a, b, c, d) {
            Intersection::Point(point) => Some(point),
            _ => None,
        })
        .collect();

    let ps_hull_f64: Vec<Point<f64>> = ps_hull
        .iter()
        .map(|&p| (p.x as f64, p.y as f64).into())
        .collect();
    let qs_hull_f64: Vec<Point<f64>> = qs_hull
        .iter()
        .map(|&p| (p.x as f64, p.y as f64).into())
        .collect();

    let mut inner: Vec<_> = ps_hull
        .iter()
        .chain(&qs_hull)
        .map(|&p| Point::<f64>::from((p.x as f64, p.y as f64)))
        .chain(inter)
        .filter(|&p| {
            convex_hull_contains(&ps_hull_f64, p, 1e-9)
                && convex_hull_contains(&qs_hull_f64, p, 1e-9)
        })
        .collect();

    let innher_hull = convex_hull(&mut inner);

    let ans = convex_hull_area(innher_hull.iter().map(|p| [p.x, p.y]));
    writeln!(output, "{}", ans).unwrap();
}
