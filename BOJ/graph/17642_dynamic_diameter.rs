use std::io::Write;

use segtree::Monoid;

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

// max { -sum[i..k] + sum[k..j] : i <= k <= j} for all i..=j
#[derive(Clone)]
struct IntervalAgg {
    sum: i64,
    left_max: i64,
    right_min: i64,
    wedge_full: i64,
    wedge_left: i64,
    wedge_right: i64,
    wedge_inner: i64,
}

impl IntervalAgg {
    fn singleton(value: i64) -> Self {
        Self {
            sum: value,
            left_max: value,
            right_min: value,
            wedge_full: value.abs(),
            wedge_left: value.abs(),
            wedge_right: value.abs(),
            wedge_inner: value.abs(),
        }
    }
}

struct IntervalMonoid;

const INF: i64 = 1 << 56;
const NEG_INF: i64 = -INF;

impl Monoid for IntervalMonoid {
    type X = IntervalAgg;

    fn id(&self) -> Self::X {
        IntervalAgg {
            sum: 0,
            left_max: INF,
            right_min: NEG_INF,
            wedge_full: NEG_INF,
            wedge_left: NEG_INF,
            wedge_right: NEG_INF,
            wedge_inner: NEG_INF,
        }
    }

    fn op(&self, a: &Self::X, b: &Self::X) -> Self::X {
        IntervalAgg {
            sum: a.sum + b.sum,
            left_max: a.left_max.max(a.sum + b.left_max),
            right_min: b.right_min.min(a.right_min + b.sum),
            wedge_full: (a.wedge_full + b.sum).max(-a.sum + b.wedge_full),
            wedge_left: a
                .wedge_left
                .max(-a.sum + b.wedge_left)
                .max(a.wedge_full + b.left_max),
            wedge_right: b
                .wedge_right
                .max(a.wedge_right + b.sum)
                .max(-a.right_min + b.wedge_full),
            wedge_inner: a
                .wedge_inner
                .max(b.wedge_inner)
                .max(a.wedge_right + b.left_max)
                .max(-a.right_min + b.wedge_left),
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();
    let w_bound: i64 = input.value();
    let mut degree = vec![0u32; n];
    let mut xor_neighbors = vec![0u32; n];

    let root = 0;
    degree[root] = 2;
    let mut edges = vec![];
    for _ in 0..n - 1 {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        let w: i64 = input.value();
        degree[u as usize] += 1;
        degree[v as usize] += 1;
        edges.push((u, v, w));
        xor_neighbors[u as usize] ^= v;
        xor_neighbors[v as usize] ^= u;
    }

    let mut topological_order = vec![];
    let mut size = vec![1; n];
    for mut u in 0..n as u32 {
        while degree[u as usize] == 1 {
            let p = xor_neighbors[u as usize];
            degree[u as usize] -= 1;
            degree[p as usize] -= 1;
            xor_neighbors[p as usize] ^= u;
            topological_order.push((u, p));

            size[p as usize] += size[u as usize];

            u = p;
        }
    }

    let mut euler_in: Vec<_> = (0..n).map(|u| size[u as usize] * 2 - 1).collect(); // 1-based
    for (u, p) in topological_order.into_iter().rev() {
        euler_in[p as usize] -= euler_in[u as usize] + 1;
        euler_in[u as usize] += euler_in[p as usize];
    }
    let euler_in = |u: usize| euler_in[u] as usize - 1; // 0-based
    let euler_out = |u: usize| (euler_in(u) + size[u] * 2) as usize;

    let mut delta_depth = vec![0; 2 * n.next_power_of_two()];
    let mut edge_bot = vec![];
    for (u, v, w) in edges {
        let b = if euler_in(u as usize) < euler_in(v as usize) {
            v
        } else {
            u
        };
        edge_bot.push(b);
        delta_depth[euler_in(b as usize)] = w;
        delta_depth[euler_out(b as usize) - 1] = -w;
    }

    let mut segtree = segtree::SegTree::from_iter(
        delta_depth.iter().map(|&d| IntervalAgg::singleton(d)),
        IntervalMonoid,
    );

    let mut ans = 0;
    for _ in 0..q {
        let mut i_edge: i64 = input.value();
        let mut w_new: i64 = input.value();
        i_edge = (i_edge + ans) % (n as i64 - 1);
        w_new = (w_new + ans) % w_bound;

        let b = edge_bot[i_edge as usize];
        segtree.modify(euler_in(b as usize), |agg| {
            *agg = IntervalAgg::singleton(w_new)
        });
        segtree.modify(euler_out(b as usize) - 1, |agg| {
            *agg = IntervalAgg::singleton(-w_new)
        });

        ans = segtree.sum_all().wedge_inner;
        writeln!(output, "{}", ans).unwrap();
    }
}
