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

// Longest Arithmetic Progression
#[derive(Debug, Clone)]
struct Asc {
    min: i32,
    max: i32,
    max_asc: i32,
}

impl Asc {
    fn new(value: i32) -> Self {
        Self {
            min: value,
            max: value,
            max_asc: 0,
        }
    }
}

struct AscMonoid;

impl Monoid for AscMonoid {
    type Elem = Option<Asc>;
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

        Some(Asc {
            min: a.min.min(b.min),
            max: a.max.max(b.max),
            max_asc: a.max_asc.max(b.max_asc).max(b.max - a.min),
        })
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n = input.value();
    let mut tree = segtree::SegTree::from_iter(
        n,
        iter::repeat_with(|| {
            let x: i32 = input.value();
            Some(Asc::new(x))
        }),
        AscMonoid,
    );

    let m = input.value();
    for _ in 0..m {
        match input.token() {
            "1" => {
                let k = input.value::<usize>() - 1;
                let x: i32 = input.value();
                tree.set(k, Some(Asc::new(x)));
            }
            "2" => {
                let i = input.value::<usize>() - 1;
                let j = input.value::<usize>() - 1;

                let ans = tree.query_range(i..j + 1).unwrap().max_asc;
                writeln!(output, "{}", ans).unwrap();
            }
            _ => panic!(),
        }
    }
}
