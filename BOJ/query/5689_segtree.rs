use std::io::Write;

use segtree::*;

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

struct RollingHash {
    value: u64,
    pow_size: u64,
}

#[derive(Clone)]
struct RollingHashBuilder {
    base: u64,
    modulo: u64,
}

impl RollingHashBuilder {
    fn singleton(&self, value: u64) -> RollingHash {
        RollingHash {
            value,
            pow_size: self.base,
        }
    }
}

impl Monoid for RollingHashBuilder {
    type Elem = RollingHash;

    fn id(&self) -> Self::Elem {
        RollingHash {
            value: 0,
            pow_size: 1,
        }
    }

    fn op(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        let value = (a.value * b.pow_size + b.value) % self.modulo;
        let pow_size = a.pow_size * b.pow_size % self.modulo;
        RollingHash { value, pow_size }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    loop {
        let base: u64 = input.value();
        let modulo: u64 = input.value();
        let l: usize = input.value();
        let n: usize = input.value();
        if (base, modulo, l, n) == (0, 0, 0, 0) {
            break;
        }

        let op = RollingHashBuilder { base, modulo };

        let mut xs = SegTree::from_iter(l, (0..l).map(|_| op.singleton(0)), op.clone());
        for _ in 0..n {
            match input.token() {
                "E" => {
                    let i = input.value::<usize>() - 1;
                    let x: u64 = input.value();
                    xs.set(i, op.singleton(x));
                }
                "H" => {
                    let i = input.value::<usize>() - 1;
                    let j = input.value::<usize>() - 1;
                    let x = xs.query_range(i..j + 1).value;
                    writeln!(output, "{}", x).unwrap();
                }
                _ => panic!(),
            }
        }
        writeln!(output, "-").unwrap();
    }
}
