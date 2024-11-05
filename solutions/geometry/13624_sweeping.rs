use std::collections::BTreeMap;
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
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf: buf, iter }
    }
}

mod cheap_rand {
    // Written in 2015 by Sebastiano Vigna (vigna@acm.org)
    // https://xoshiro.di.unimi.it/splitmix64.c
    use std::ops::Range;
    pub struct Rng(u64);

    impl Rng {
        pub fn new(seed: u64) -> Self {
            assert_ne!(seed, 0);
            Self(seed)
        }

        pub fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = self.0;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        }

        pub fn range_u64(&mut self, range: Range<u64>) -> u64 {
            let Range { start, end } = range;
            assert!(start < end);

            let width = end - start;
            let test = (u64::MAX - width) % width;
            loop {
                let value = self.next_u64();
                if value >= test {
                    return start + value % width;
                }
            }
        }

        pub fn shuffle<T>(&mut self, xs: &mut [T]) {
            let n = xs.len();
            for i in 0..n - 1 {
                let j = self.range_u64(i as u64..n as u64) as usize;
                xs.swap(i, j);
            }
        }
    }
}

#[macro_use]
mod geometry {
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
        + std::fmt::Debug
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

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct PointNd<const N: usize, T>(pub [T; N]);

    pub type Point<T> = PointNd<2, T>;
    pub type Point3<T> = PointNd<3, T>;

    impl<T> Point<T> {
        pub fn new(x: T, y: T) -> Self {
            Self([x, y])
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Angle<T>(pub Point<T>);

    impl<T: Scalar> Angle<T> {
        pub fn circular_cmp(&self, other: &Self) -> std::cmp::Ordering {
            T::zero().partial_cmp(&cross(self.0, other.0)).unwrap()
        }
    }

    impl<T: Scalar> PartialEq for Angle<T> {
        fn eq(&self, other: &Self) -> bool {
            debug_assert!(self.0 != PointNd([T::zero(), T::zero()]));
            debug_assert!(other.0 != PointNd([T::zero(), T::zero()]));
            cross(self.0, other.0) == T::zero()
        }
    }

    impl<T: Scalar> Eq for Angle<T> {}

    impl<const N: usize, T: Scalar> PointNd<N, T> {
        pub fn dot(self, other: Self) -> T {
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| a * b)
                .fold(T::zero(), |acc, x| acc + x)
        }

        pub fn max_norm(self) -> T {
            self.0
                .into_iter()
                .map(|a| a.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        }
    }

    impl<const N: usize, T: Scalar> From<[T; N]> for PointNd<N, T> {
        fn from(p: [T; N]) -> Self {
            Self(p)
        }
    }

    impl<const N: usize, T: Scalar> Index<usize> for PointNd<N, T> {
        type Output = T;
        fn index(&self, i: usize) -> &Self::Output {
            &self.0[i]
        }
    }

    impl<const N: usize, T: Scalar> IndexMut<usize> for PointNd<N, T> {
        fn index_mut(&mut self, i: usize) -> &mut Self::Output {
            &mut self.0[i]
        }
    }

    macro_rules! impl_binop_dims {
        ($N:expr, $($idx:expr )+, $trait:ident, $fn:ident) => {
            impl<T: Scalar> $trait for PointNd<$N, T> {
                type Output = Self;
                fn $fn(self, other: Self) -> Self::Output {
                    PointNd([$(self[$idx].$fn(other[$idx])),+])
                }
            }
        };
    }

    macro_rules! impl_binop {
        ($trait:ident, $fn:ident) => {
            impl_binop_dims!(2, 0 1, $trait, $fn);
            impl_binop_dims!(3, 0 1 2, $trait, $fn);
        };
    }

    impl_binop!(Add, add);
    impl_binop!(Sub, sub);
    impl_binop!(Mul, mul);

    impl<const N: usize, T: Scalar> Default for PointNd<N, T> {
        fn default() -> Self {
            PointNd([T::zero(); N])
        }
    }

    impl<const N: usize, T: Scalar> Neg for PointNd<N, T> {
        type Output = Self;
        fn neg(self) -> Self::Output {
            PointNd(self.0.map(|x| -x))
        }
    }

    impl<const N: usize, T: Scalar> Mul<T> for PointNd<N, T> {
        type Output = Self;
        fn mul(self, k: T) -> Self::Output {
            PointNd(self.0.map(|x| x * k))
        }
    }

    pub fn cross<T: Scalar>(p: Point<T>, q: Point<T>) -> T {
        p[0] * q[1] - p[1] * q[0]
    }

    pub fn signed_area<T: Scalar>(p: Point<T>, q: Point<T>, r: Point<T>) -> T {
        cross(q - p, r - p)
    }

    pub fn ccw<T: Scalar>(p: Point<T>, q: Point<T>, r: Point<T>) -> bool {
        T::zero() < signed_area(p, q, r)
    }
}

use geometry::*;

mod collections {
    use std::cell::Cell;

    #[derive(Debug)]
    pub struct DisjointSet {
        parent: Vec<Cell<usize>>,
        size: Vec<u32>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self {
                parent: (0..n).map(|i| Cell::new(i)).collect(),
                size: vec![1; n],
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            if u == self.parent[u].get() {
                u
            } else {
                self.parent[u].set(self.find_root(self.parent[u].get()));
                self.parent[u].get()
            }
        }

        pub fn get_size(&self, u: usize) -> u32 {
            self.size[self.find_root(u)]
        }

        // returns whether two set were different
        pub fn merge(&mut self, mut u: usize, mut v: usize) -> bool {
            u = self.find_root(u);
            v = self.find_root(v);
            if u == v {
                return false;
            }
            if self.size[u] > self.size[v] {
                std::mem::swap(&mut u, &mut v);
            }
            self.parent[v].set(u);
            self.size[u] += self.size[v];
            true
        }
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());

    let n: usize = input.value();
    let c: usize = input.value();
    let mut segments: Vec<[Point<i64>; 2]> = Vec::with_capacity(n);
    for _ in 0..n {
        let x1 = input.value();
        let y1 = input.value();
        let x2 = input.value();
        let y2 = input.value();

        let p1 = Point::new(x1, y1);
        let p2 = Point::new(x2, y2);
        segments.push(if x1 < x2 { [p1, p2] } else { [p2, p1] });
    }
    let below = |s: [Point<i64>; 2], t: [Point<i64>; 2]| -> std::cmp::Ordering {
        let [a, b] = s;
        let [c, d] = t;
        let top = if a[1] > b[1] { a } else { b };
        if !(c[0]..=d[0]).contains(&top[0]) {
            return std::cmp::Ordering::Equal;
        }
        signed_area(c, d, top).cmp(&0)
    };
    segments.sort_unstable_by(|&p, &q| {
        below(p, q)
            .then(below(q, p).reverse())
            .then((p[0][1].max(p[1][1])).cmp(&(q[0][1].max(q[1][1]))))
            .then(p.cmp(&q))
    });

    let mut queries: Vec<i64> = Vec::with_capacity(c);
    for _ in 0..c {
        queries.push(input.value());
    }

    let mut dset = collections::DisjointSet::new(c);
    let mut pos_to_root: BTreeMap<i64, usize> = Default::default();
    for i in 0..c {
        pos_to_root
            .entry(queries[i])
            .and_modify(|root| {
                dset.merge(*root, i);
                *root = dset.find_root(*root)
            })
            .or_insert(i);
    }

    let mut blocked = vec![];
    for [a, b] in segments {
        let top = if a[1] > b[1] { a } else { b };
        let flat = a[1] == b[1];

        let mut keys = vec![];

        if !flat {
            let range = if a[1] > b[1] {
                a[0] + 1..b[0] + 1
            } else {
                a[0]..b[0]
            };

            let merged_root = pos_to_root
                .range(range)
                .map(|(&x, &idx)| {
                    keys.push(x);
                    idx
                })
                .reduce(|i, j| {
                    dset.merge(i, j);
                    i
                })
                .map(|i| dset.find_root(i));

            if let Some(root) = merged_root {
                for &x in &keys {
                    pos_to_root.remove(&x);
                }

                pos_to_root
                    .entry(top[0])
                    .and_modify(|r| {
                        dset.merge(*r, root);
                        *r = dset.find_root(*r);
                    })
                    .or_insert(root);
            }
        } else {
            pos_to_root.range(a[0]..=b[0]).for_each(|(&x, &idx)| {
                keys.push(x);
                blocked.push((x, a[1], idx));
            });

            for &x in &keys {
                pos_to_root.remove(&x);
            }
        }
        // println!(" {:?} {:?} {} ", a, b, flat);
        // println!("{:?}", pos_to_root);
        // println!("{:?}", blocked);
        // println!("{:?}", dset);
    }

    let mut children = vec![vec![]; c];
    for i in 0..c {
        children[dset.find_root(i)].push(i);
    }

    let mut result = vec![(0, None); c];
    for (&x, &root) in pos_to_root.iter() {
        // println!("{} {:?}", x, children[root]);
        for &child in &children[root] {
            result[child] = (x, None);
        }
    }
    for &(x, y, idx) in &blocked {
        for &child in &children[dset.find_root(idx)] {
            result[child] = (x, Some(y));
        }
    }

    for i in 0..c {
        let (x, y) = result[i];
        if let Some(y) = y {
            writeln!(output, "{} {}", x, y);
        } else {
            writeln!(output, "{}", x);
        }
    }
    // println!("{:?}", children);
    // println!("{:?}", pos_to_root);
    // println!("{:?}", blocked);
}
