use std::{io::Write, ops::Range};

use segtree::{Monoid, SegTree};

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
    use std::ops::Range;

    pub trait Monoid {
        type X;
        const IS_COMMUTATIVE: bool = false;
        fn id(&self) -> Self::X;
        fn op(&self, a: &Self::X, b: &Self::X) -> Self::X;
    }

    #[derive(Debug)]
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

const INF: u32 = u32::MAX;
const NEG_INF: u32 = u32::MIN;

struct MinMax;

impl MinMax {
    fn singleton(x: u32) -> (u32, u32) {
        (x, x)
    }
}

impl Monoid for MinMax {
    type X = (u32, u32);

    fn id(&self) -> Self::X {
        (INF, NEG_INF)
    }

    fn op(&self, a: &Self::X, b: &Self::X) -> Self::X {
        (a.0.min(b.0), a.1.max(b.1))
    }
}

fn partition_in_place<T>(xs: &mut [T], mut pred: impl FnMut(&T) -> bool) -> (&mut [T], &mut [T]) {
    let n = xs.len();
    let mut i = 0;
    for j in 0..n {
        if pred(&xs[j]) {
            xs.swap(i, j);
            i += 1;
        }
    }
    xs.split_at_mut(i)
}

type Query = ((u32, u32), u32);

struct Cx {
    perm: SegTree<MinMax>,
    inv: SegTree<MinMax>,
}

impl Cx {
    fn expand_once(&self, query_range: &mut (u32, u32)) -> bool {
        let (ref mut l, ref mut r) = query_range;
        let x_bound = self.perm.sum_range(*l as usize..*r as usize + 1);
        let idx_bound = self
            .inv
            .sum_range(x_bound.0 as usize..x_bound.1 as usize + 1);
        let (l_new, r_new) = (idx_bound.0, idx_bound.1);

        debug_assert!(l_new <= *l && *r <= r_new);
        if *l == l_new && r_new == *r {
            return false;
        }

        *l = l_new;
        *r = r_new;
        true
    }

    fn expand_batch_dnc(&self, queries: &mut [Query], bound: Range<u32>) {
        if bound.start + 1 == bound.end {
            return;
        }

        let mid = (bound.start + bound.end) / 2;
        let (left, rest) = partition_in_place(queries, |&((_l, r), _)| r < mid);
        let (right, _) = partition_in_place(rest, |&((l, _r), _)| l >= mid);

        self.expand_batch_dnc(left, bound.start..mid);
        self.expand_batch_dnc(right, mid..bound.end);

        debug_assert!(mid >= 1);
        let (unprocessed, _) = partition_in_place(queries, |&((l, r), _)| l <= mid - 1 && mid <= r);
        {
            unprocessed.sort_unstable_by_key(|&((l, _r), _)| l);
            let mut unprocessed = unprocessed.iter_mut().rev().peekable();
            let mut curr = (mid - 1, mid);
            loop {
                while self.expand_once(&mut curr) && bound.start <= curr.0 {}
                while let Some(((l, r), _)) = unprocessed.next_if(|&&mut ((l, _r), _)| curr.0 <= l)
                {
                    *l = curr.0;
                    *r = (*r).max(curr.1);
                }
                if curr.0 <= bound.start {
                    break;
                }
                curr.0 -= 1;
            }
        }

        {
            unprocessed.sort_unstable_by_key(|&((_l, r), _)| r);
            let mut unprocessed = unprocessed.iter_mut().peekable();
            let mut curr = (mid - 1, mid);
            loop {
                while self.expand_once(&mut curr) && curr.1 <= bound.end - 1 {}
                while let Some(((l, r), _)) = unprocessed.next_if(|&&mut ((_l, r), _)| r <= curr.1)
                {
                    *r = curr.1;
                    *l = (*l).min(curr.0);
                }
                if curr.1 >= bound.end - 1 {
                    break;
                }
                curr.1 += 1;
            }
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let perm: Vec<_> = (0..n).map(|_| input.value::<u32>() - 1).collect();
    let mut inv = vec![0; n];
    for (i, &x) in perm.iter().enumerate() {
        inv[x as usize] = i as u32;
    }
    let perm = SegTree::from_iter(perm.into_iter().map(MinMax::singleton), MinMax);
    let inv = SegTree::from_iter(inv.into_iter().map(MinMax::singleton), MinMax);
    let cx = Cx { perm, inv };

    let q: usize = input.value();
    let mut queries: Vec<_> = (0..q)
        .map(|i| {
            (
                (input.value::<u32>() - 1, input.value::<u32>() - 1),
                i as u32,
            )
        })
        .collect();

    cx.expand_batch_dnc(&mut queries, 0..n as u32);
    queries.sort_unstable_by_key(|&(_, i)| i);
    for ((l, r), _) in queries {
        writeln!(output, "{} {}", l + 1, r + 1).unwrap();
    }
}
