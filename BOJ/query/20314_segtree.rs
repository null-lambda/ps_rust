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

struct MaxOp<T> {
    lower_bound: T,
}

impl<T> MaxOp<T> {
    pub fn new(lower_bound: T) -> Self {
        Self { lower_bound }
    }
}

impl<T: Ord + Clone> Monoid for MaxOp<T> {
    type Elem = T;
    fn id(&self) -> Self::Elem {
        self.lower_bound.clone()
    }
    fn op(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        a.max(b).clone()
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let hs: Vec<u32> = (0..n).map(|_| input.value()).collect();
    let ts_acc: Vec<u64> = iter::once(0)
        .chain((0..n - 1).map(|_| input.value::<u64>()).scan(0, |acc, x| {
            *acc += x;
            Some(*acc)
        }))
        .collect();

    let mut left = vec![0; n];
    let mut j = n;
    for i in (0..n).rev() {
        while j > 0 && ts_acc[i] <= hs[j - 1] as u64 + ts_acc[j - 1] {
            j -= 1;
        }
        left[i] = j;
    }

    let mut right = vec![n - 1; n];
    let mut j = 0;
    for i in 0..n {
        while j + 1 < n && ts_acc[j + 1] <= hs[j + 1] as u64 + ts_acc[i] {
            j += 1;
        }
        right[i] = j;
    }

    let hs = segtree::SegTree::from_iter(n, hs.into_iter(), MaxOp::new(0));

    for i in 0..n {
        let ans = hs.query_range(left[i]..right[i] + 1);
        write!(output, "{} ", ans).unwrap();
    }
}
