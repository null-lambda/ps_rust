use std::io::Write;

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
            I: IntoIterator<Item = M::Elem>,
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

pub mod hld {
    const UNSET: u32 = u32::MAX;

    // Heavy-Light Decomposition
    #[derive(Debug)]
    pub struct HLD {
        pub size: Vec<u32>,
        pub depth: Vec<u32>,
        pub parent: Vec<u32>,
        pub heavy_child: Vec<u32>,
        pub chain_top: Vec<u32>,
        pub euler_idx: Vec<u32>,
        pub topological_order: Vec<u32>,
    }

    impl HLD {
        pub fn len(&self) -> usize {
            self.parent.len()
        }

        pub fn from_edges<'a>(
            n: usize,
            edges: impl IntoIterator<Item = (u32, u32)>,
            root: usize,
        ) -> Self {
            // Fast tree reconstruction with XOR-linked tree traversal
            // https://codeforces.com/blog/entry/135239
            let mut degree = vec![0u32; n];
            let mut xor_neighbors: Vec<u32> = vec![0u32; n];
            for (u, v) in edges.into_iter().flat_map(|(u, v)| [(u, v), (v, u)]) {
                debug_assert!(u != v);
                degree[u as usize] += 1;
                xor_neighbors[u as usize] ^= v;
            }

            let mut size = vec![1; n];
            let mut heavy_child = vec![UNSET; n];
            degree[root] += 2;
            let mut topological_order = vec![];
            for mut u in 0..n {
                while degree[u] == 1 {
                    // Toposort
                    let p = xor_neighbors[u];
                    topological_order.push(u as u32);
                    degree[u] = 0;
                    degree[p as usize] -= 1;
                    xor_neighbors[p as usize] ^= u as u32;

                    // Upward propagation
                    size[p as usize] += size[u as usize];
                    let h = &mut heavy_child[p as usize];
                    if *h == UNSET || size[*h as usize] < size[u as usize] {
                        *h = u as u32;
                    }

                    u = p as usize;
                }
            }
            topological_order.push(root as u32);
            topological_order.reverse();
            assert!(topological_order.len() == n, "Invalid tree structure");

            let mut parent = xor_neighbors;
            parent[root] = UNSET;

            // Downward propagation
            let mut depth = vec![0; n];
            let mut chain_top = vec![root as u32; n];
            for &u in &topological_order[1..] {
                let p = parent[u as usize];
                depth[u as usize] = depth[p as usize] + 1;

                let h = heavy_child[p as usize];
                chain_top[u as usize] = if u == h { chain_top[p as usize] } else { u };
            }

            let mut euler_idx = vec![UNSET; n];
            let mut timer = 0;
            for u in &topological_order {
                let mut u = *u;
                while u != UNSET && euler_idx[u as usize] == UNSET {
                    euler_idx[u as usize] = timer;
                    timer += 1;
                    u = heavy_child[u as usize];
                }
            }

            Self {
                size,
                depth,
                parent,
                heavy_child,
                chain_top,
                euler_idx,
                topological_order,
            }
        }

        pub fn for_each_path<F>(&self, mut u: usize, mut v: usize, mut visitor: F)
        where
            F: FnMut(usize, usize, bool),
        {
            debug_assert!(u < self.len() && v < self.len());

            while self.chain_top[u] != self.chain_top[v] {
                if self.depth[self.chain_top[u] as usize] < self.depth[self.chain_top[v] as usize] {
                    std::mem::swap(&mut u, &mut v);
                }
                visitor(self.chain_top[u] as usize, u, false);
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.euler_idx[u] > self.euler_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            visitor(u, v, true);
        }

        pub fn for_each_path_splitted<F>(&self, mut u: usize, mut v: usize, mut visit: F)
        where
            F: FnMut(usize, usize, bool, bool),
        {
            debug_assert!(u < self.len() && v < self.len());
            if self.euler_idx[u] > self.euler_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            while self.chain_top[u] != self.chain_top[v] {
                if self.depth[self.chain_top[u] as usize] > self.depth[self.chain_top[v] as usize] {
                    visit(self.chain_top[u] as usize, u, true, false);
                    u = self.parent[self.chain_top[u] as usize] as usize;
                } else {
                    visit(self.chain_top[v] as usize, v, false, false);
                    v = self.parent[self.chain_top[v] as usize] as usize;
                }
            }
            if self.depth[u] > self.depth[v] {
                visit(v, u, true, true);
            } else {
                visit(u, v, false, true);
            }
        }

        pub fn lca(&self, mut u: usize, mut v: usize) -> usize {
            debug_assert!(u < self.len() && v < self.len());
            while self.chain_top[u] != self.chain_top[v] {
                if self.depth[self.chain_top[u] as usize] < self.depth[self.chain_top[v] as usize] {
                    std::mem::swap(&mut u, &mut v);
                }
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.euler_idx[u] > self.euler_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            u
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct MaxOp;

impl Monoid for MaxOp {
    type Elem = u32;

    fn id(&self) -> Self::Elem {
        0
    }

    fn op(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        (*a).max(*b)
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let edges: Vec<_> = (0..n - 1)
        .map(|_| {
            (
                input.value::<u32>() - 1,
                input.value::<u32>() - 1,
                input.value(),
            )
        })
        .collect();

    let hld = hld::HLD::from_edges(n as usize, edges.iter().map(|&(u, v, _)| (u, v)), 0);

    let mut edge_bottom = vec![0; n];
    let mut weights = vec![0; n];
    for (i, &(u, v, w)) in edges.iter().enumerate() {
        if hld.depth[u as usize] > hld.depth[v as usize] {
            edge_bottom[i] = u;
            weights[u as usize] = w;
        } else {
            edge_bottom[i] = v;
            weights[v as usize] = w;
        }
    }

    let mut weights_euler = vec![0; n];
    for u in 0..n {
        weights_euler[hld.euler_idx[u] as usize] = weights[u];
    }
    let mut segtree = segtree::SegTree::from_iter(n, weights_euler, MaxOp);

    let n_queries = input.value();
    for _ in 0..n_queries {
        match input.token() {
            "1" => {
                let i: usize = input.value();
                let c = input.value();
                segtree.set(hld.euler_idx[edge_bottom[i - 1] as usize] as usize, c);
            }
            "2" => {
                let u = input.value::<usize>() - 1;
                let v = input.value::<usize>() - 1;
                let mut ans = MaxOp.id();
                hld.for_each_path(u, v, |u, v, is_lca| {
                    let eu = hld.euler_idx[u] as usize;
                    let ev = hld.euler_idx[v] as usize;
                    if is_lca {
                        ans = ans.max(segtree.query_range(eu + 1..ev + 1));
                    } else {
                        ans = ans.max(segtree.query_range(eu..ev + 1));
                    }
                });
                writeln!(output, "{}", ans).unwrap();
            }
            _ => panic!(),
        }
    }
}
