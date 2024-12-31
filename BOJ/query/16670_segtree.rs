use std::io::Write;

use segtree::*;

mod simple_io {
    use std::string::*;

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
            for i in (1..n).rev() {
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

struct Interval {
    sum: u64,
    end: u64,
}

impl Interval {
    fn singleton(t: u64, delay: u64) -> Self {
        Self {
            sum: delay,
            end: t + delay,
        }
    }
}

struct IntervalOp;

impl Monoid for IntervalOp {
    type Elem = Interval;
    fn id(&self) -> Self::Elem {
        Interval { sum: 0, end: 0 }
    }
    fn op(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        Interval {
            sum: a.sum + b.sum,
            end: (a.end + b.sum).max(b.end),
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let t_max = 1_000_000;
    let mut tree = SegTree::from_iter(
        t_max as usize + 1,
        (0..=t_max).map(|t| Interval::singleton(t, 0)),
        IntervalOp,
    );
    let q = input.value();
    let mut history = vec![u64::MAX; q + 1];
    for i in 1..=q {
        match input.token() {
            "+" => {
                let t: u64 = input.value();
                let d = input.value();
                tree.set(t as usize, Interval::singleton(t, d));
                history[i] = t;
            }
            "-" => {
                let i: usize = input.value();
                let t = history[i];
                tree.set(t as usize, Interval::singleton(t, 0));
            }

            "?" => {
                let t: u64 = input.value();
                assert!(t <= t_max);
                let result = tree.query_range(0..t as usize + 1);
                writeln!(output, "{}", result.end - t).unwrap();
            }
            _ => panic!(),
        }
    }
}
