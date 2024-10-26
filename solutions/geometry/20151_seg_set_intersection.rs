use std::io::Write;

#[allow(dead_code)]
mod simple_io {
    pub struct InputAtOnce(std::str::SplitAsciiWhitespace<'static>);

    impl InputAtOnce {
        pub fn token(&mut self) -> &str {
            self.0.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(buf.into_boxed_str());
        InputAtOnce(buf.split_ascii_whitespace())
    }

    pub fn stdout_buf() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

use std::cmp::Ordering;
use std::ops::{Add, Mul, Neg, Sub};

trait Zero: Sized + Add<Self, Output = Self> {
    fn zero() -> Self;
}

macro_rules! zero_impl {
    ($v:expr, $($t:ty)*) => {$(
        impl Zero for $t {
            #[inline]
            fn zero() -> $t {
                $v
            }
        }
    )*}
}

zero_impl! {0, isize i8 i16 i32 i64 i128}
zero_impl! {0, usize u8 u16 u32 u64 u128}
zero_impl! {0.0, f32 f64}

macro_rules! trait_alias {
    ($name:ident = $($value:tt)+) => {
        trait $name: $($value)+ {}
        impl<T> $name for T where T: $($value)+ {}
    };
}

trait_alias! { Scalar = Copy + Clone + Add<Output = Self> + Sub<Output = Self> + Neg<Output = Self> + Mul<Output = Self> + Zero + PartialOrd + Ord + PartialEq + Eq }

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Point<T: Scalar> {
    x: T,
    y: T,
}

impl<T: Scalar> From<(T, T)> for Point<T> {
    fn from(p: (T, T)) -> Self {
        Self { x: p.0, y: p.1 }
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

// Returns signed area of a triangle; positive if ccw
fn signed_area<T: Scalar>(p: &Point<T>, q: &Point<T>, r: &Point<T>) -> T {
    (q.x - p.x) * (r.y - p.y) + (r.x - p.x) * (p.y - q.y)
}

fn cross<T: Scalar>(p: Point<T>, q: Point<T>) -> T {
    signed_area(&Point::zero(), &p, &q)
}

#[derive(Debug, Clone)]
struct Angle<T: Scalar>(Point<T>);

impl<T: Scalar> Angle<T> {
    fn rev(&self) -> Self {
        Angle(Point {
            x: self.0.x,
            y: -self.0.y,
        })
    }
}

impl<T: Scalar> Angle<T> {
    pub fn circular_cmp(&self, other: &Self) -> std::cmp::Ordering {
        T::zero().partial_cmp(&cross(self.0, other.0)).unwrap()
    }
}

impl<T: Scalar> PartialOrd for Angle<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.circular_cmp(other))
        // Some(
        //     ((self.0.y, self.0.x) < (T::zero(), T::zero()))
        //         .cmp(&((other.0.y, other.0.x) < (T::zero(), T::zero())))
        //         .then_with(|| (self.circular_cmp(other))),
        // )
    }
}

impl<T: Scalar> Ord for Angle<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<T: Scalar> PartialEq for Angle<T> {
    fn eq(&self, other: &Self) -> bool {
        debug_assert!(self.0 != Point::zero());
        debug_assert!(other.0 != Point::zero());
        ((self.0.y, self.0.x) < (T::zero(), T::zero()))
            == ((other.0.y, other.0.x) < (T::zero(), T::zero()))
            && cross(self.0, other.0) == T::zero()
    }
}

impl<T: Scalar> Eq for Angle<T> {}

fn check_seg_inter<T: Scalar + Ord + std::fmt::Debug + std::fmt::Display>(
    p1: Point<T>,
    p2: Point<T>,
    q1: Point<T>,
    q2: Point<T>,
) -> bool {
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

    // let is_endpoint = (T::zero() == mul_det_t || mul_det_t == det)
    //     && (T::zero() == mul_det_s || mul_det_s == det);
    let is_endpoint = p1 == q1 || p1 == q2 || p2 == q1 || p2 == q2;
    if is_endpoint {
        return false;
    }
    //  check t, s in closed interval [0, 1]
    match &det.cmp(&T::zero()) {
        Ordering::Greater => {
            (T::zero()..=det).contains(&mul_det_t) && (T::zero()..=det).contains(&mul_det_s)
        }
        Ordering::Less => {
            (det..=T::zero()).contains(&mul_det_t) && (det..=T::zero()).contains(&mul_det_s)
        }
        // two segments are either parallel or colinear
        Ordering::Equal => {
            fn reorder<T: Ord>(a: T, b: T) -> (T, T) {
                if a < b {
                    (a, b)
                } else {
                    (b, a)
                }
            }

            if signed_area(&Point::zero(), &pd, &r) == T::zero() {
                let ((a1, a2), (b1, b2)) = if p1.x != p2.x {
                    (reorder(p1.x, p2.x), reorder(q1.x, q2.x))
                } else {
                    (reorder(p1.y, p2.y), reorder(q1.y, q2.y))
                };
                return a1 <= b2 && b1 <= a2;
            }
            false
        }
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout_buf();

    use std::collections::BTreeSet;
    let n: usize = input.value();
    let mut read_point = || -> Point<i64> { (input.value(), input.value()).into() };

    #[derive(Debug, Copy, Clone)]
    enum EventType {
        Add = 1,
        Remove = 0,
    }
    let mut events = Vec::with_capacity(2 * n);

    let segments: Vec<_> = (0..n)
        .map(|i| {
            let mut p1 = read_point();
            let mut p2 = read_point();

            if (p1.x, p1.y) > (p2.x, p2.y) {
                std::mem::swap(&mut p1, &mut p2);
            }
            events.push((EventType::Add, p1, p2));
            events.push((EventType::Remove, p1, p2));
            [p1, p2]
        })
        .collect();
    events.sort_by_key(|&(event_type, p1, p2)| {
        (match event_type {
            EventType::Remove => (p2.x, p2.y, 0, Angle(p2 - p1).rev()),
            EventType::Add => (p1.x, p1.y, 1, Angle(p2 - p1)),
        },)
    });

    let result = (|| {
        let mut active_segs = BTreeSet::new();
        for (event_type, p1, p2) in events {
            let key = (p1.y, p1.x, Angle(p2 - p1), p2.y, p2.x);
            match event_type {
                EventType::Add => {
                    let next = active_segs.range(&key..).next();
                    let prev = active_segs.range(..&key).rev().next();
                    for &(q1y, q1x, _, q2y, q2x) in prev.into_iter().chain(next) {
                        let (q1, q2) = (Point { x: q1x, y: q1y }, Point { x: q2x, y: q2y });
                        if check_seg_inter(p1, p2, q1, q2) {
                            return true;
                        }
                    }
                    active_segs.insert(key);
                }
                EventType::Remove => {
                    active_segs.remove(&key);
                    let next = active_segs.range(&key..).next();
                    let prev = active_segs.range(..&key).rev().next();
                    if let (Some(&(q1y, q1x, _, q2y, q2x)), Some(&(r1y, r1x, _, r2y, r2x))) =
                        (prev, next)
                    {
                        let (q1, q2) = (Point { x: q1x, y: q1y }, Point { x: q2x, y: q2y });
                        let (r1, r2) = (Point { x: r1x, y: r1y }, Point { x: r2x, y: r2y });
                        if check_seg_inter(q1, q2, r1, r2) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    })();

    writeln!(output, "{}", result as i64).unwrap();
}
