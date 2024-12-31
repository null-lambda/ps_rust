use std::io::Write;

use segtree::{Monoid, SegTree};
use std::cmp::Ordering::{self, *};

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

#[derive(Debug, Clone)]
struct Interval {
    empty: bool,
    full: bool,
    left_end: Ordering,
    right_end: Ordering,
    left_alt_count: u32,
    right_alt_count: u32,
    base_cost: u32,
}

impl Interval {
    fn singleton(x: Ordering) -> Self {
        let (full, alt_count, cost) = match x {
            Equal => (false, 0, 0),
            _ => (true, 1, 1),
        };
        Interval {
            empty: false,
            full,
            left_end: x,
            right_end: x,
            left_alt_count: alt_count,
            right_alt_count: alt_count,
            base_cost: cost,
        }
    }

    fn is_empty(&self) -> bool {
        self.empty
    }

    fn cost(&self) -> u32 {
        self.base_cost + self.left_alt_count / 2 + self.right_alt_count / 2
    }
}

struct IntervalOp;

impl Monoid for IntervalOp {
    type Elem = Interval;
    fn id(&self) -> Self::Elem {
        Interval {
            empty: true,
            full: false,
            left_end: Equal,
            right_end: Equal,
            left_alt_count: 0,
            right_alt_count: 0,
            base_cost: 0,
        }
    }

    fn op(&self, lhs: &Self::Elem, rhs: &Self::Elem) -> Self::Elem {
        fn is_alternating(lhs: Ordering, rhs: Ordering) -> bool {
            match (lhs, rhs) {
                (Less, Greater) | (Greater, Less) => true,
                _ => false,
            }
        }

        if lhs.is_empty() {
            return rhs.clone();
        }
        if rhs.is_empty() {
            return lhs.clone();
        }

        debug_assert!(!lhs.full || lhs.left_alt_count == lhs.right_alt_count);
        debug_assert!(!rhs.full || rhs.left_alt_count == rhs.right_alt_count);
        let join = is_alternating(lhs.right_end, rhs.left_end);
        Interval {
            empty: false,
            full: lhs.full && rhs.full && join,
            left_end: lhs.left_end,
            right_end: rhs.right_end,
            left_alt_count: if lhs.full && join {
                lhs.left_alt_count + rhs.left_alt_count
            } else {
                lhs.left_alt_count
            },
            right_alt_count: if rhs.full && join {
                rhs.right_alt_count + lhs.right_alt_count
            } else {
                rhs.right_alt_count
            },
            base_cost: lhs.base_cost
                + rhs.base_cost
                + if join {
                    if lhs.full || rhs.full {
                        0
                    } else {
                        (lhs.right_alt_count + rhs.left_alt_count) / 2
                    }
                } else {
                    (if lhs.full { 0 } else { lhs.right_alt_count / 2 })
                        + (if rhs.full { 0 } else { rhs.left_alt_count / 2 })
                },
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: u32 = input.value();

    let xs: Vec<u32> = (0..n).map(|_| input.value()).collect();
    let mut segtree = SegTree::from_iter(
        n * 2,
        (0..n * 2).map(|_| Interval::singleton(Greater)),
        IntervalOp,
    );

    let mut pos = vec![vec![]; m as usize + 1];
    for (i, &x) in xs.iter().enumerate() {
        pos[x as usize].push(i as u32);
    }

    for pivot in 1..=m {
        for &i in &pos[pivot as usize] {
            segtree.set(i as usize, Interval::singleton(Equal));
            segtree.set(i as usize + n, Interval::singleton(Equal));
        }
        for &i in &pos[pivot as usize - 1] {
            segtree.set(i as usize, Interval::singleton(Less));
            segtree.set(i as usize + n, Interval::singleton(Less));
        }

        if let Some(&i) = pos[pivot as usize].first() {
            let ans = segtree.query_range(i as usize..i as usize + n).cost();
            write!(output, "{} ", ans).unwrap();
        } else {
            write!(output, "-1 ").unwrap();
        }
    }
}
