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

fn interval_overlaps((a1, a2): (i32, i32), (b1, b2): (i32, i32)) -> bool {
    let (a1, a2) = reorder(a1, a2);
    let (b1, b2) = reorder(b1, b2);
    max(a1, b1) <= min(a2, b2)
}

fn segment_intersects(p1: Point<i32>, p2: Point<i32>, q1: Point<i32>, q2: Point<i32>) -> bool {
    // aabb test for fast rejection
    if !interval_overlaps((p1.x, p2.x), (q1.x, q2.x))
        || !interval_overlaps((p1.y, p2.y), (q1.y, q2.y))
    {
        return false;
    }

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

    match &det.cmp(&0) {
        Ordering::Greater => (0..=det).contains(&mul_det_t) && (0..=det).contains(&mul_det_s),
        Ordering::Less => (det..=0).contains(&mul_det_t) && (det..=0).contains(&mul_det_s),
        Ordering::Equal => signed_area(&(0, 0).into(), &pd, &r) == 0,
    }
}

struct DisjointSet {
    parent: Vec<usize>,
    size: Vec<usize>,
    count: usize,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: std::iter::repeat(1).take(n).collect(),
            count: n,
        }
    }

    fn find_root(&mut self, u: usize) -> usize {
        if u == self.parent[u] {
            u
        } else {
            self.parent[u] = self.find_root(self.parent[u]);
            self.parent[u]
        }
    }

    fn merge(&mut self, mut u: usize, mut v: usize) {
        u = self.find_root(u);
        v = self.find_root(v);
        if u != v {
            if self.size[u] > self.size[v] {
                std::mem::swap(&mut u, &mut v);
            }
            self.parent[v] = u;
            self.size[u] += self.size[v];
            self.count -= 1;
        }
    }
}

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let n = input.value();
    let mut read_point = || (input.value(), input.value()).into();
    let ps: Vec<(Point<i32>, Point<i32>)> = (0..n).map(|_| (read_point(), read_point())).collect();

    let mut dset = DisjointSet::new(n);
    for (i, &(p1, p2)) in ps.iter().enumerate() {
        for (j, &(q1, q2)) in ps.iter().enumerate() {
            if segment_intersects(p1, p2, q1, q2) {
                dset.merge(i, j);
            }
        }
    }

    writeln!(output_buf, "{}", dset.count);
    writeln!(output_buf, "{}", *dset.size.iter().max().unwrap());

    std::io::stdout().write(&output_buf[..]).unwrap();
}
