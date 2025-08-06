use std::io::Write;

use segtree::*;
use std::iter::*;
use std::ops::Range;

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

fn group_indices_by<'a, T>(
    xs: &'a [T],
    mut pred: impl 'a + FnMut(&T, &T) -> bool,
) -> impl 'a + Iterator<Item = [usize; 2]> {
    let mut i = 0;
    std::iter::from_fn(move || {
        if i == xs.len() {
            return None;
        }

        let mut j = i + 1;
        while j < xs.len() && pred(&xs[j - 1], &xs[j]) {
            j += 1;
        }
        let res = [i, j];
        i = j;
        Some(res)
    })
}

fn group_by<'a, T>(
    xs: &'a [T],
    pred: impl 'a + FnMut(&T, &T) -> bool,
) -> impl 'a + Iterator<Item = &'a [T]> {
    group_indices_by(xs, pred).map(|w| &xs[w[0]..w[1]])
}

fn group_by_key<'a, T, K: PartialEq>(
    xs: &'a [T],
    mut key: impl 'a + FnMut(&T) -> K,
) -> impl 'a + Iterator<Item = &'a [T]> {
    group_by(xs, move |a, b| key(a) == key(b))
}

pub mod segtree {
    use std::ops::Range;

    pub trait Monoid {
        type X;
        const IS_COMMUTATIVE: bool = false;
        fn id(&self) -> Self::X;
        fn op(&self, a: &Self::X, b: &Self::X) -> Self::X;
    }

    #[derive(Debug, Clone)]
    pub struct SegTree<M>
    where
        M: Monoid,
    {
        n: usize,
        sum: Vec<M::X>,
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

        pub fn from_iter<I>(iter: I, monoid: M) -> Self
        where
            I: IntoIterator<Item = M::X>,
            I::IntoIter: ExactSizeIterator<Item = M::X>,
        {
            let iter = iter.into_iter();
            let n = iter.len();
            let mut sum: Vec<_> = (0..n).map(|_| monoid.id()).chain(iter).collect();
            for i in (0..n).rev() {
                sum[i] = monoid.op(&sum[i << 1], &sum[i << 1 | 1]);
            }
            Self { n, sum, monoid }
        }

        pub fn modify(&mut self, mut idx: usize, f: impl FnOnce(&mut M::X)) {
            debug_assert!(idx < self.n);
            idx += self.n;
            f(&mut self.sum[idx]);
            while idx > 1 {
                idx >>= 1;
                self.sum[idx] = self.monoid.op(&self.sum[idx << 1], &self.sum[idx << 1 | 1]);
            }
        }

        pub fn get(&self, idx: usize) -> &M::X {
            &self.sum[idx + self.n]
        }

        pub fn mapped_sum_range<N: Monoid>(
            &self,
            range: Range<usize>,
            codomain: &N,
            morphism: impl Fn(&M::X) -> N::X,
        ) -> N::X {
            let Range { mut start, mut end } = range;
            if start >= end {
                return codomain.id();
            }
            debug_assert!(start < self.n && end <= self.n);
            start += self.n;
            end += self.n;

            if N::IS_COMMUTATIVE {
                let mut result = codomain.id();
                while start < end {
                    if start & 1 != 0 {
                        result = codomain.op(&result, &morphism(&self.sum[start]));
                    }
                    if end & 1 != 0 {
                        result = codomain.op(&morphism(&self.sum[end - 1]), &result);
                    }
                    start = (start + 1) >> 1;
                    end >>= 1;
                }
                result
            } else {
                let (mut result_left, mut result_right) = (codomain.id(), codomain.id());
                while start < end {
                    if start & 1 != 0 {
                        result_left = codomain.op(&result_left, &morphism(&self.sum[start]));
                    }
                    if end & 1 != 0 {
                        result_right = codomain.op(&morphism(&self.sum[end - 1]), &result_right);
                    }
                    start = (start + 1) >> 1;
                    end >>= 1;
                }
                codomain.op(&result_left, &result_right)
            }
        }

        pub fn sum_all(&self) -> &M::X {
            assert!(self.n.is_power_of_two());
            &self.sum[1]
        }
    }

    impl<M: Monoid> SegTree<M>
    where
        M::X: Clone,
    {
        pub fn sum_range(&self, range: Range<usize>) -> M::X {
            self.mapped_sum_range(range, &self.monoid, |x| x.clone())
        }
    }
}

const BOUND: i32 = 200_005;

#[derive(Clone)]
struct MaxOp;

impl Monoid for MaxOp {
    type X = u32;

    fn id(&self) -> Self::X {
        0
    }

    fn op(&self, a: &Self::X, b: &Self::X) -> Self::X {
        *a.max(b)
    }
}

#[derive(Clone)]
struct Column {
    ys: Vec<i32>,
    counter: SegTree<MaxOp>,
}

struct MergeSortTree {
    n: usize,
    ps: Vec<(i32, i32)>,
    nodes: Vec<Column>,
}

pub fn merge_unique<T: Ord + Copy + std::fmt::Debug>(xs: &[T], ys: &[T]) -> Vec<T> {
    let mut res = Vec::with_capacity(xs.len() + ys.len());
    let (mut i, mut j) = (0, 0);
    while i < xs.len() || j < ys.len() {
        if j == ys.len() {
            res.push(xs[i]);
            i += 1;
        } else if i == xs.len() {
            res.push(ys[j]);
            j += 1;
        } else if xs[i] < ys[j] {
            res.push(xs[i]);
            i += 1;
        } else if ys[j] < xs[i] {
            res.push(ys[j]);
            j += 1;
        } else {
            res.push(xs[i]);
            i += 1;
            j += 1;
        }
    }
    res
}

impl MergeSortTree {
    fn from_iter(ps: impl Iterator<Item = (i32, i32)>) -> Self {
        let mut ps: Vec<_> = ps.collect();
        ps.sort_unstable();
        ps.dedup();

        let mut n = ps.len();
        n = n.next_power_of_two();
        ps.resize(n, (BOUND * 4, BOUND * 4));

        let mut nodes: Vec<_> = (0..2 * n)
            .map(|_| Column {
                ys: vec![],
                counter: SegTree::with_size(0, MaxOp),
            })
            .collect();

        for i in (0..n).rev() {
            nodes[i + n].ys.push(ps[i].1);
        }

        for i in (1..n).rev() {
            nodes[i].ys = merge_unique(&nodes[i * 2].ys, &nodes[i * 2 + 1].ys);
        }

        for i in 1..2 * n {
            nodes[i].counter = SegTree::with_size(nodes[i].ys.len(), MaxOp);
        }

        Self { n, ps, nodes }
    }

    fn modify(&mut self, p: (i32, i32), mut update_with: impl FnMut(&mut u32)) {
        let ip = self.ps.partition_point(|&q| q < p);
        assert_eq!(p, self.ps[ip]);

        let mut u = ip + self.n;
        while u > 0 {
            let iy = self.nodes[u].ys.partition_point(|&y| y < p.1);
            assert_eq!(self.nodes[u].ys[iy], p.1);
            self.nodes[u].counter.modify(iy, |x| update_with(x));
            u >>= 1;
        }
    }

    fn sum_rect(&self, x_range: Range<i32>, y_range: Range<i32>) -> u32 {
        let mut start = self.ps.partition_point(|&p| p.0 < x_range.start);
        let mut end = self.ps.partition_point(|&p| p.0 < x_range.end);

        start += self.n;
        end += self.n;

        let mut ans = MaxOp.id();

        let mut visit_column = |i: usize| {
            let l = self.nodes[i].ys.partition_point(|&y| y < y_range.start);
            let r = self.nodes[i].ys.partition_point(|&y| y < y_range.end);
            ans = MaxOp.op(&ans, &self.nodes[i].counter.sum_range(l..r));
        };

        while start < end {
            if start % 2 == 1 {
                visit_column(start);
            }
            if end % 2 == 1 {
                visit_column(end - 1);
            }
            start = (start + 1) >> 1;
            end >>= 1;
        }
        ans
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut updates = vec![];
    for _ in 0..n {
        let x0: i32 = input.value();
        let y0: i32 = input.value();
        let w: u32 = input.value();
        let l: i32 = input.value();

        let x = x0 + y0;
        let y = x0 - y0;
        updates.push((w, x, y, l));
    }
    updates.sort_unstable();

    let mut mst = MergeSortTree::from_iter(updates.iter().map(|&(_, x, y, _)| (x, y)));
    let mut ans = 0;
    for gs in group_by_key(&updates, |&(w, ..)| w) {
        let mut row = vec![];
        for &(_, x, y, l) in gs {
            let mut old = 0;
            old = old.max(mst.sum_rect(x - l..x + l + 1, y - l..y + l + 1));

            let new = old + 1;
            ans = ans.max(new);
            row.push(((x, y), new));
        }

        for (p, new) in row {
            mst.modify(p, |v| *v = (*v).max(new));
        }
    }

    writeln!(output, "{}", ans).unwrap();
}
