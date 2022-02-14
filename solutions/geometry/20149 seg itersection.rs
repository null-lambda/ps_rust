mod io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }

        #[inline]
        fn value<T>(&mut self) -> T
        where
            T: FromStr,
            T::Err: Debug,
        {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }

    #[inline]
    fn is_whitespace(c: u8) -> bool {
        c <= b' '
    }

    fn trim_newline(s: &[u8]) -> &[u8] {
        let mut s = s;
        while s
            .last()
            .map(|&c| match c {
                b'\n' | b'\r' | 0 => true,
                _ => false,
            })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| !is_whitespace(c))
                .expect("no available tokens left");
            *self = &self[idx..];
            let idx = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(idx);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|idx| idx + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(idx);
            *self = buf_new;
            trim_newline(line)
        }
    }
}

use std::io::{BufReader, Read, Write};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

use std::cmp::{max, min, Ordering};
use std::ops::{Add, Div, Mul, Neg, Sub};

trait Zero: Sized + Add<Self, Output = Self> {
    fn zero() -> Self;
}

trait One: Sized + Add<Self, Output = Self> {
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
        trait $name: $($value)+ {}
        impl<T> $name for T where T: $($value)+ {}
    };
}

trait_alias! { Scalar = Copy + Clone + Add<Output = Self> + Sub<Output = Self> + Neg<Output = Self> + Mul<Output = Self> + Zero + One}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Point<T: Scalar> {
    x: T,
    y: T,
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

type T = i64;

enum Intersection<T: Scalar> {
    Disjoint,
    Point(Point<T>),
    Segment,
}

fn segment_intersection(
    p1: Point<T>,
    p2: Point<T>,
    q1: Point<T>,
    q2: Point<T>,
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

    if T::zero() != det {
        let param_range = reorder(T::zero(), det);
        let param_range = param_range.0..=param_range.1;
        if param_range.contains(&mul_det_t) && param_range.contains(&mul_det_s) {
            let i1 = p1 * det + pd * mul_det_t;
            let i1: Point<_> = (i1.x as f64, i1.y as f64).into();
            Intersection::Point(i1 * (f64::one() / (det as f64)))
        } else {
            Intersection::Disjoint
        }
    } else {
        if signed_area(&Point::zero(), &pd, &r) == T::zero() {
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

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let mut read_point = || (input.value(), input.value()).into();
    let (p1, p2): (Point<i64>, Point<i64>) = (read_point(), read_point());
    let (q1, q2): (Point<i64>, Point<i64>) = (read_point(), read_point());

    let it = segment_intersection(p1, p2, q1, q2);
    match it {
        Intersection::Point(Point { x, y }) => {
            writeln!(output_buf, "1\n{} {}", x, y).unwrap();
        }
        Intersection::Segment => {
            writeln!(output_buf, "1").unwrap();
        }
        Intersection::Disjoint => {
            writeln!(output_buf, "0").unwrap();
        }
    }

    std::io::stdout().write(&output_buf[..]).unwrap();
}
