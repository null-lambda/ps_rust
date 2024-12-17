use std::{cmp::Reverse, io::Write};

use segtree::{Monoid, SegTree};

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
            let n = n.max(1);
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

            debug_assert!(start <= end && end <= self.n);

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

struct MinOp;

impl Monoid for MinOp {
    type Elem = i64;
    fn id(&self) -> Self::Elem {
        1 << 56
    }
    fn op(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        (*a).min(*b)
    }
}

fn partition_point<P>(mut left: usize, mut right: usize, mut pred: P) -> usize
where
    P: FnMut(usize) -> bool,
{
    while left < right {
        let mid = left + (right - left) / 2;
        if pred(mid) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let l: i64 = input.value();
    let mut ps = vec![];
    for _ in 0..n {
        let a: i64 = input.value();
        let b: i64 = input.value();
        ps.push((a - b, a));
    }

    ps.sort_unstable_by_key(|&t| Reverse(t));
    let delta = |i: usize| ps[i].0;
    let a = |i: usize| ps[i].1;

    let cs: Vec<i64> = (0..n).map(|_| input.value()).collect();

    let delta_bound = ps.iter().position(|&(delta, _)| !(delta > 0)).unwrap_or(n);
    if delta_bound == 0 {
        if (0..n).map(|i| a(i)).max().unwrap() >= l {
            writeln!(output, "1").unwrap();
        } else {
            writeln!(output, "-1").unwrap();
        }
        return;
    }

    let mut delta_prefix = vec![0];
    let mut acc = 0;
    for i in 0..delta_bound {
        acc += delta(i);
        delta_prefix.push(acc);
    }

    let mut cs_prefix = vec![0];
    let mut acc = 0;
    for i in 0..delta_bound {
        acc += cs[i];
        cs_prefix.push(acc);
    }

    // delta_prefix[i] - cs_prefix[i]
    let space0 = SegTree::from_iter(
        delta_bound,
        (0..delta_bound).map(|i| delta_prefix[i + 1] - cs_prefix[i + 1]),
        MinOp,
    );

    // delta_prefix[i] - cs_prefix[i-1]
    let space1 = SegTree::from_iter(
        delta_bound.checked_sub(1).unwrap_or(0),
        (1..delta_bound).map(|i| delta_prefix[i + 1] - cs_prefix[i]),
        MinOp,
    );

    let inf = 1 << 56;
    let mut ans = inf;
    for last in 0..n {
        let t_bound = if delta_bound <= last {
            (delta_bound + 1).min(n)
        } else {
            delta_bound
        };
        let max_dist = |t: usize| {
            debug_assert!(1 <= t && t <= t_bound);
            if t - 1 <= last {
                delta_prefix[t - 1] + a(last)
            } else {
                delta_prefix[t] + a(last) - delta(last)
            }
        };
        let test = |t: usize| {
            debug_assert!(1 <= t && t <= t_bound);
            if t - 1 <= last {
                space0.query_range(0..t - 1) > 0
            } else {
                space0.query_range(0..last) > 0 && space1.query_range(last..t - 1) - delta(last) > 0
            }
        };

        let t_min = partition_point(1, t_bound + 1, |t| max_dist(t) < l);
        if t_min <= t_bound && test(t_min) {
            ans = ans.min(t_min);
        }
    }

    if ans == inf {
        writeln!(output, "-1").unwrap();
    } else {
        writeln!(output, "{}", ans).unwrap();
    }
}
