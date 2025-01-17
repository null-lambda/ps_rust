use std::io::Write;

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

pub mod dset {
    pub mod potential {
        pub trait Group: Clone {
            fn id() -> Self;
            fn add_assign(&mut self, b: &Self);
            fn sub_assign(&mut self, b: &Self);
        }

        #[derive(Clone, Copy)]
        struct Link(i32); // Represents parent if >= 0, size if < 0

        impl Link {
            fn node(p: u32) -> Self {
                Self(p as i32)
            }

            fn size(s: u32) -> Self {
                Self(-(s as i32))
            }

            fn get(&self) -> Result<u32, u32> {
                if self.0 >= 0 {
                    Ok(self.0 as u32)
                } else {
                    Err((-self.0) as u32)
                }
            }
        }

        pub struct DisjointSet<E> {
            links: Vec<(Link, E)>,
        }

        impl<E: Group + Eq> DisjointSet<E> {
            pub fn with_size(n: usize) -> Self {
                Self {
                    links: (0..n).map(|_| (Link::size(1), E::id())).collect(),
                }
            }

            pub fn find_root_with_size(&mut self, u: usize) -> (usize, E, u32) {
                let (l, w) = &self.links[u];
                match l.get() {
                    Ok(p) => {
                        let mut w_acc = w.clone();
                        let (root, w_to_root, size) = self.find_root_with_size(p as usize);
                        w_acc.add_assign(&w_to_root);
                        self.links[u] = (Link::node(root as u32), w_acc.clone());
                        (root, w_acc, size)
                    }
                    Err(size) => (u, w.clone(), size),
                }
            }

            pub fn find_root(&mut self, u: usize) -> usize {
                self.find_root_with_size(u).0
            }

            pub fn get_size(&mut self, u: usize) -> u32 {
                self.find_root_with_size(u).2
            }

            // Returns true if two sets were previously disjoint
            pub fn merge(&mut self, u: usize, v: usize, mut weight_uv: E) -> Result<bool, ()> {
                let (mut u, mut weight_u, mut size_u) = self.find_root_with_size(u);
                let (mut v, mut weight_v, mut size_v) = self.find_root_with_size(v);
                if u == v {
                    let mut weight_u_expected = weight_uv;
                    weight_u_expected.add_assign(&weight_v);

                    if weight_u == weight_u_expected {
                        return Ok(false);
                    } else {
                        return Err(());
                    }
                }

                if size_u < size_v {
                    std::mem::swap(&mut u, &mut v);
                    std::mem::swap(&mut weight_u, &mut weight_v);
                    std::mem::swap(&mut size_u, &mut size_v);

                    let mut neg = E::id();
                    neg.sub_assign(&weight_uv);
                    weight_uv = neg;
                }

                weight_u.add_assign(&weight_uv);
                weight_v.sub_assign(&weight_u);
                self.links[v] = (Link::node(u as u32), weight_v);
                self.links[u] = (Link::size(size_u + size_v), E::id());
                Ok(true)
            }

            pub fn delta_potential(&mut self, u: usize, v: usize) -> Option<E> {
                let (u, weight_u, _) = self.find_root_with_size(u);
                let (v, weight_v, _) = self.find_root_with_size(v);
                (u == v).then(|| {
                    let mut delta = weight_u.clone();
                    delta.sub_assign(&weight_v);
                    delta
                })
            }
        }
    }
}

impl dset::potential::Group for bool {
    fn id() -> Self {
        false
    }
    fn add_assign(&mut self, b: &Self) {
        *self ^= b;
    }
    fn sub_assign(&mut self, b: &Self) {
        *self ^= b;
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

struct MinOp<T> {
    inf: T,
}

impl<T: Ord + Clone> Monoid for MinOp<T> {
    type X = T;
    const IS_COMMUTATIVE: bool = true;

    fn id(&self) -> Self::X {
        self.inf.clone()
    }

    fn op(&self, a: &Self::X, b: &Self::X) -> Self::X {
        a.min(b).clone()
    }
}

const P: u64 = 1_000_000_007;

fn mod_pow(mut base: u64, mut exp: u64) -> u64 {
    let mut res = 1;
    while exp > 0 {
        if exp & 1 == 1 {
            res = res * base % P;
        }
        base = base * base % P;
        exp >>= 1;
    }
    res
}

const INF: u32 = 1 << 30;
const UNSET: u32 = 1 << 30;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut intervals: Vec<(u32, u32)> = (0..n).map(|_| (input.value(), input.value())).collect();
    intervals.sort_unstable_by_key(|&(_, e)| e);
    let mut dsu = dset::potential::DisjointSet::<bool>::with_size(n);

    let inf_entry = (INF, INF, UNSET);
    let mut active = SegTree::with_size(2 * n + 1, MinOp { inf: inf_entry });

    for u in 0..n {
        let (su, eu) = intervals[u];

        let mut acc_dual = (INF, 0, UNSET);
        loop {
            // Find all active intervals containing su (There are at most two, one for each parity,
            // since each same-colored set of intervals is disjoint.)
            let (sv, ev, v) = active.sum_range(su as usize..2 * n + 1);
            if sv > su {
                break;
            }
            match dsu.merge(u, v as usize, true) {
                Ok(_) => {
                    active.modify(ev as usize, |x| *x = inf_entry);
                    acc_dual = (acc_dual.0.min(sv), acc_dual.1.max(ev), v);
                }
                Err(()) => {
                    writeln!(output, "0").unwrap();
                    return;
                }
            }
        }

        active.modify(eu as usize, |x| *x = (su, eu, u as u32));
        if acc_dual.2 != UNSET {
            active.modify(acc_dual.1 as usize, |x| *x = acc_dual);
        }
    }

    let n_components = (0..n).filter(|&u| dsu.find_root(u) == u).count() as u64;
    writeln!(output, "{}", mod_pow(2, n_components)).unwrap();
}
