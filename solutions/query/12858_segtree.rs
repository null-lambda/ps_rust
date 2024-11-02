use segtree::{Monoid, SegTree};
use std::{io::Write, iter};

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

pub mod segtree {
    use std::{iter, ops::Range};

    pub trait Monoid {
        type Elem;
        fn id(&self) -> Self::Elem;
        fn op(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
    }

    #[derive(Debug)]
    pub struct SegTree<M>
    where
        M: Monoid,
    {
        n: usize,
        sum: Vec<M::Elem>,
        monoid: M,
    }

    impl<M: Monoid> SegTree<M> {
        pub fn with_size(n: usize, monoid: M) -> Self {
            Self {
                n,
                sum: (0..2 * n).map(|_| monoid.id()).collect(),
                monoid,
            }
        }

        pub fn from_iter<I>(n: usize, iter: I, monoid: M) -> Self
        where
            I: Iterator<Item = M::Elem>,
        {
            let mut sum: Vec<_> = (0..n)
                .map(|_| monoid.id())
                .chain(iter)
                .chain(iter::repeat_with(|| monoid.id()))
                .take(2 * n)
                .collect();
            for i in (0..n).rev() {
                sum[i] = monoid.op(&sum[i << 1], &sum[i << 1 | 1]);
            }
            Self { n, sum, monoid }
        }

        pub fn set(&mut self, mut idx: usize, value: M::Elem) {
            debug_assert!(idx < self.n);
            idx += self.n;
            self.sum[idx] = value;
            while idx > 1 {
                idx >>= 1;
                self.sum[idx] = self.monoid.op(&self.sum[idx << 1], &self.sum[idx << 1 | 1]);
            }
        }

        pub fn get(&self, idx: usize) -> &M::Elem {
            &self.sum[idx + self.n]
        }

        pub fn query_range(&self, range: Range<usize>) -> M::Elem {
            let Range { mut start, mut end } = range;
            debug_assert!(start < self.n && end <= self.n);
            start += self.n;
            end += self.n;
            let (mut result_left, mut result_right) = (self.monoid.id(), self.monoid.id());
            while start < end {
                if start & 1 != 0 {
                    result_left = self.monoid.op(&result_left, &self.sum[start]);
                }
                if end & 1 != 0 {
                    result_right = self.monoid.op(&self.sum[end - 1], &result_right);
                }
                start = (start + 1) >> 1;
                end >>= 1;
            }

            self.monoid.op(&result_left, &result_right)
        }
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

struct Additive;
impl Monoid for Additive {
    type Elem = i64;
    fn id(&self) -> Self::Elem {
        0
    }
    fn op(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        a + b
    }
}

struct Gcd;
impl Monoid for Gcd {
    type Elem = u64;
    fn id(&self) -> Self::Elem {
        0
    }
    fn op(&self, &a: &Self::Elem, &b: &Self::Elem) -> Self::Elem {
        gcd(a, b)
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let xs: Vec<i64> = (0..n).map(|_| input.value()).collect();
    let mut dx_add = SegTree::from_iter(
        n + 1,
        iter::once(xs[0])
            .chain(xs.windows(2).map(|w| w[1] - w[0]))
            .chain(iter::once(0)),
        Additive,
    );
    let mut dx_gcd = SegTree::from_iter(
        n + 1,
        iter::once(xs[0])
            .chain(xs.windows(2).map(|w| (w[1] - w[0]).abs()))
            .chain(iter::once(0))
            .map(|x| x as u64),
        Gcd,
    );

    for _ in 0..input.value() {
        let t: i64 = input.value();
        let a: usize = input.value();
        let b: usize = input.value();

        if t == 0 {
            let left = dx_add.query_range(0..a) as u64;
            let rest = dx_gcd.query_range(a..b);
            let ans = gcd(left, rest);
            writeln!(output, "{}", ans).unwrap();
        } else {
            let old_a = *dx_add.get(a - 1);
            let old_b = *dx_add.get(b);
            dx_add.set(a - 1, old_a + t);
            dx_gcd.set(a - 1, (old_a + t).abs() as u64);
            dx_add.set(b, old_b - t);
            dx_gcd.set(b, (old_b - t).abs() as u64);
        }
    }
}
