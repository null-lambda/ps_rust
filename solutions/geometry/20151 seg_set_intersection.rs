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
            .map(|&c| matches!(c, b'\n' | b'\r' | 0))
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let i = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
            *self = &self[i..];
            let i = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(i);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let i = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|i| i + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(i);
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

trait_alias! { Scalar = Copy + Clone + Add<Output = Self> + Sub<Output = Self> + Neg<Output = Self> + Mul<Output = Self> + Zero }

#[derive(Debug, Copy, Clone)]
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

fn segment_intersects<T: Scalar + Ord + std::fmt::Debug + std::fmt::Display>(
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

    /*
    let i1 = p1 * det + pd * mul_det_t;
    let mut i1: Point<f64> = (i1.x as f64, i1.y as f64).into();
    i1 = i1 * (1.0 / (det as f64));
    i1

    let i2 = q1 * det + qd * mul_det_s;
    let mut i2: Point<f64> = (i2.x as f64, i2.y as f64).into();
    i2 = i2 * (1.0 / (det as f64));
    println!("{} {} {} , {:?}={:?}", det, mul_det_t, mul_det_s, i1, i2);
    */

    let is_endpoint = (T::zero() == mul_det_t || mul_det_t == det) 
        &&(T::zero() == mul_det_s || mul_det_s == det);
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
                return max(a1, b1) < min(a2, b2);
            }
            false
        }
    }
}

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    use std::collections::BTreeSet;
    let n: usize = input.value();
    let mut read_point = || -> Point<i64> { (input.value(), input.value()).into() };

    #[derive(Debug, Copy, Clone)]
    enum EventType {
        Add,
        Remove,
    }
    let mut events = Vec::with_capacity(2 * n);

    let segments: Vec<_> = (0..n)
        .map(|i| {
            let mut p1 = read_point();
            let mut p2 = read_point();

            if p1.x > p2.x {
                std::mem::swap(&mut p1, &mut p2);
            }
            events.push((EventType::Add, p1.x, i));
            events.push((EventType::Remove, p2.x, i));
            [p1, p2]
        })
        .collect();
    events.sort_by_key(|&(event_type, x, _)| (x, matches!(event_type, EventType::Remove)));

    let result = (|| {
        let mut current_segments = BTreeSet::new();
        for (event_type, _, idx) in events {
            let [p1, p2] = segments[idx];
            // println!("{:?} {:?}", [p1, p2], event_type);

            match event_type {
                EventType::Add => {
                    let next = current_segments.range((p1.y, idx)..).next();
                    let prev = current_segments.range(..(p1.y, idx)).next_back();
                    // println!("{:?}", (prev, next));
                    for &(_, idx) in prev.into_iter().chain(next) {
                        let [q1, q2] = segments[idx];
                        // println!("{:?} {:?}", [p1, p2], [q1, q2]);
                        if segment_intersects(p1, p2, q1, q2) {
                            return true;
                        }
                    }
                    current_segments.insert((p1.y, idx));
                }
                EventType::Remove => {
                    current_segments.remove(&(p1.y, idx));

                    let next = current_segments.range((p1.y, idx)..).next();
                    let prev = current_segments.range(..(p1.y, idx)).next_back();
                    if let (Some(&(_, idx1)), Some(&(_, idx2))) = (prev, next) {
                        let [p1, p2] = segments[idx1];
                        let [q1, q2] = segments[idx2];
                        // println!("{:?} {:?}", [p1, p2], [q1, q2]);
                        if segment_intersects(p1, p2, q1, q2) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    })();

    writeln!(output_buf, "{}", result as i64).unwrap();

    std::io::stdout().write(&output_buf[..]).unwrap();
}
