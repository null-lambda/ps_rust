use std::{io::Write, iter};

use segtree::Monoid;

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
            if start >= end {
                return self.monoid.id();
            }
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

#[derive(Debug, Clone)]
struct IntervalSum {
    sum: i64,
    max: i64,
    max_left: i64,
    max_right: i64,
}

struct IntervalSumMonoid;

impl Monoid for IntervalSumMonoid {
    type Elem = Option<IntervalSum>;
    fn id(&self) -> Self::Elem {
        None
    }
    fn op(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        if a.is_none() {
            return b.clone();
        }
        if b.is_none() {
            return a.clone();
        }
        let a = a.as_ref().unwrap();
        let b = b.as_ref().unwrap();
        Some(IntervalSum {
            sum: a.sum + b.sum,
            max: a.max.max(b.max).max(a.max_right + b.max_left),
            max_left: a.max_left.max(a.sum + b.max_left),
            max_right: b.max_right.max(b.sum + a.max_right),
        })
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n = input.value();
    let tree = segtree::SegTree::from_iter(
        n,
        iter::repeat_with(|| {
            let x: i64 = input.value();
            Some(IntervalSum {
                sum: x,
                max: x,
                max_left: x,
                max_right: x,
            })
        }),
        IntervalSumMonoid,
    );

    let m = input.value();
    for _ in 0..m {
        let x1 = input.value::<usize>() - 1;
        let y1 = input.value::<usize>() - 1;
        let x2 = input.value::<usize>() - 1;
        let y2 = input.value::<usize>() - 1;
        let ans = if y1 < x2 {
            tree.query_range(x1..y1 + 1).unwrap().max_right
                + tree.query_range(y1 + 1..x2).map_or(0, |x| x.sum)
                + tree.query_range(x2..y2 + 1).unwrap().max_left
        } else {
            let left = tree.query_range(x1..x2).map_or(0, |x| x.max_right);
            let mid = tree.query_range(x2..y1 + 1).unwrap();
            let right = tree.query_range(y1 + 1..y2 + 1).map_or(0, |x| x.max_left);
            mid.max
                .max(left + mid.max_left)
                .max(mid.max_right + right)
                .max(left + mid.sum + right)
        };
        writeln!(output, "{}", ans).unwrap();
    }
}
