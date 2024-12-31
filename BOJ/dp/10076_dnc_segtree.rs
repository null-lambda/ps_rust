use std::{cmp::Reverse, io::Write, ops::Range};

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

// Compute row minimum C(i) = min_j A(i, j)
// where opt(i) = argmin_j A(i, j) is monotonically increasing.
//
// Arguments:
// - naive: Compute opt(i) for a given range of j, with an implicit evaluation of C(i).
fn dnc_row_min(
    naive: &mut impl FnMut(usize, Range<usize>) -> usize,
    i: Range<usize>,
    j: Range<usize>,
) {
    if i.start >= i.end {
        return;
    }
    let i_mid = i.start + i.end >> 1;
    let j_opt = naive(i_mid, j.clone());
    dnc_row_min(naive, i.start..i_mid, j.start..j_opt + 1);
    dnc_row_min(naive, i_mid + 1..i.end, j_opt..j.end);
}

pub mod segtree {
    use std::{iter, ops::Range};

    pub trait Monoid {
        type Elem;
        fn id(&self) -> Self::Elem;
        fn combine(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
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
            let n = n.next_power_of_two();
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
            let n = n.next_power_of_two();
            let mut sum: Vec<_> = (0..n)
                .map(|_| monoid.id())
                .chain(iter)
                .chain(iter::repeat_with(|| monoid.id()))
                .take(2 * n)
                .collect();
            for i in (1..n).rev() {
                sum[i] = monoid.combine(&sum[i << 1], &sum[i << 1 | 1]);
            }
            Self { n, sum, monoid }
        }

        pub fn set(&mut self, mut idx: usize, value: M::Elem) {
            debug_assert!(idx < self.n);
            idx += self.n;
            self.sum[idx] = value;
            while idx > 1 {
                idx >>= 1;
                self.sum[idx] = self
                    .monoid
                    .combine(&self.sum[idx << 1], &self.sum[idx << 1 | 1]);
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
                    result_left = self.monoid.combine(&result_left, &self.sum[start]);
                }
                if end & 1 != 0 {
                    result_right = self.monoid.combine(&self.sum[end - 1], &result_right);
                }
                start = (start + 1) >> 1;
                end >>= 1;
            }

            self.monoid.combine(&result_left, &result_right)
        }

        pub fn partition_point(&self, mut pred: impl FnMut(&M::Elem) -> bool) -> usize {
            let mut left = 0;
            let mut right = self.n;
            let mut node = 1;
            let mut acc = self.monoid.id();
            while left + 1 < right {
                let mid = right + left >> 1;
                node <<= 1;
                let combined = self.monoid.combine(&acc, &self.sum[node]);
                if pred(&combined) {
                    left = mid;
                    acc = combined;
                    node |= 1;
                } else {
                    right = mid;
                }
            }

            let mut res = left;
            if res < self.n && pred(&acc) {
                res += 1;
            }
            res
        }
    }
}

struct Additive;

impl Monoid for Additive {
    type Elem = (u32, u64);
    fn id(&self) -> Self::Elem {
        (0, 0)
    }
    fn combine(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        (a.0 + b.0, a.1 + b.1)
    }
}

fn range_diff(lhs: Range<usize>, rhs: Range<usize>) -> impl Iterator<Item = usize> {
    let left = lhs.start..lhs.end.min(rhs.start);
    let right = rhs.end.max(lhs.start)..lhs.end;
    left.chain(right)
}

fn solve(xs: &[u64], start: usize, d: usize) -> u64 {
    let n = xs.len();

    let mut xs_sorted: Vec<(u64, u32)> = (0..n as u32).map(|i| (xs[i as usize], i)).collect();
    xs_sorted.sort_unstable_by_key(|&(x, _)| Reverse(x));

    let mut order = vec![0; n];
    for (o, &(_, i)) in xs_sorted.iter().enumerate() {
        order[i as usize] = o;
    }
    drop(xs_sorted);

    let mut sums = SegTree::with_size(n, Additive);

    let mut res = 0;
    let mut prev = 0..0;
    let mut naive = |i: usize, j: Range<usize>| {
        let curr = i..j.start;
        for p in range_diff(prev.clone(), curr.clone()) {
            sums.set(order[p], (0u32.into(), 0u64.into()));
        }
        for p in range_diff(curr.clone(), prev.clone()) {
            sums.set(order[p], (1u32.into(), xs[p].into()));
        }

        let (max, opt_j) = (j.start..j.end.min(d + 2 * i - start + 1))
            .map(|j| {
                sums.set(order[j], (1u32.into(), xs[j].into()));

                let p = j + start - 2 * i;
                let l = (d - p).min(j - i + 1);
                let k = sums.partition_point(|(count, _)| *count < l as u32) as usize;
                let (_, sum_largest) = sums.query_range(0..k as usize);

                (sum_largest, j)
            })
            .max()
            .unwrap();

        prev = i..j.end;

        res = res.max(max);
        opt_j
    };
    dnc_row_min(
        &mut naive,
        start.checked_sub(d / 2).unwrap_or(0)..start + 1,
        start..n.min(start + d),
    );
    res
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let start: usize = input.value();
    let d: usize = input.value();
    let mut xs: Vec<u64> = (0..n).map(|_| input.value()).collect();

    let mut ans = solve(&xs, start, d);

    xs.reverse();
    ans = ans.max(solve(&xs, n - start - 1, d));

    writeln!(output, "{}", ans).unwrap();
}
