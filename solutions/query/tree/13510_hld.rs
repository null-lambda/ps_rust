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
        pub monoid: M,
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

pub mod hld {
    // Heavy-Light Decomposition
    #[derive(Debug)]
    pub struct HLD {
        pub size: Vec<u32>,
        pub depth: Vec<u32>,
        pub parent: Vec<u32>,
        pub heavy_child: Vec<u32>,
        pub chain_top: Vec<u32>,
        pub euler_idx: Vec<u32>,
    }

    impl HLD {
        pub fn len(&self) -> usize {
            self.parent.len()
        }

        fn dfs_size(&mut self, children: &[Vec<u32>], u: usize) {
            self.size[u] = 1;
            if children.is_empty() {
                return;
            }
            for &v in &children[u] {
                self.depth[v as usize] = self.depth[u] + 1;
                self.parent[v as usize] = u as u32;
                self.dfs_size(children, v as usize);
                self.size[u] += self.size[v as usize];
            }
            self.heavy_child[u] = children[u]
                .iter()
                .copied()
                .max_by_key(|&v| self.size[v as usize])
                .unwrap_or(0);
        }

        fn dfs_decompose(&mut self, children: &[Vec<u32>], u: usize, order: &mut u32) {
            self.euler_idx[u] = *order;
            *order += 1;
            if children[u].is_empty() {
                return;
            }
            let h = self.heavy_child[u];
            self.chain_top[h as usize] = self.chain_top[u];
            self.dfs_decompose(children, h as usize, order);
            for &v in children[u].iter().filter(|&&v| v != h) {
                self.chain_top[v as usize] = v;
                self.dfs_decompose(children, v as usize, order);
            }
        }

        pub fn from_graph(children: &[Vec<u32>]) -> Self {
            let n = children.len();
            let mut hld = Self {
                size: vec![0; n],
                depth: vec![0; n],
                parent: vec![0; n],
                heavy_child: vec![u32::MAX; n],
                chain_top: vec![0; n],
                euler_idx: vec![0; n],
            };
            hld.dfs_size(children, 0);
            hld.dfs_decompose(children, 0, &mut 0);
            hld
        }

        pub fn for_each_path<F>(&self, mut u: usize, mut v: usize, mut visitor: F)
        where
            F: FnMut(usize, usize),
        {
            debug_assert!(u < self.len() && v < self.len());

            while self.chain_top[u] != self.chain_top[v] {
                if self.depth[self.chain_top[u] as usize] < self.depth[self.chain_top[v] as usize] {
                    std::mem::swap(&mut u, &mut v);
                }
                visitor(
                    self.euler_idx[self.chain_top[u] as usize] as usize,
                    self.euler_idx[u] as usize,
                );
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.euler_idx[u] > self.euler_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            visitor(self.euler_idx[u] as usize + 1, self.euler_idx[v] as usize);
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Max;

impl Monoid for Max {
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
    let mut neighbors: Vec<Vec<(u32, u32, u32)>> = (0..n).map(|_| Vec::new()).collect();
    for i_edge in 0..n as u32 - 1 {
        let u: u32 = input.value();
        let v: u32 = input.value();
        let weight: u32 = input.value();
        neighbors[u as usize - 1].push((i_edge, v - 1, weight));
        neighbors[v as usize - 1].push((i_edge, u - 1, weight));
    }

    let mut children: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();
    let mut weights = vec![0; n];
    let mut edge_bottom = vec![0; n];
    {
        let mut visited = vec![false; n];
        visited[0] = true;
        fn dfs(
            neighbors: &[Vec<(u32, u32, u32)>],
            children: &mut [Vec<u32>],
            weights: &mut [u32],
            edge_bottom: &mut [u32],
            visited: &mut [bool],
            u: u32,
        ) {
            for &(i_edge, v, weight) in &neighbors[u as usize] {
                if !visited[v as usize] {
                    visited[v as usize] = true;
                    children[u as usize].push(v);
                    weights[v as usize] = weight;
                    edge_bottom[i_edge as usize] = v;
                    dfs(neighbors, children, weights, edge_bottom, visited, v);
                }
            }
        }
        dfs(
            &neighbors,
            &mut children,
            &mut weights,
            &mut edge_bottom,
            &mut visited,
            0,
        );
    }

    let hld = hld::HLD::from_graph(&children);
    let mut segtree = segtree::SegTree::with_size(n, Max);
    for u in 0..n {
        segtree.set(hld.euler_idx[u] as usize, weights[u]);
    }

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
                let mut res = Max.id();
                hld.for_each_path(u, v, |l, r| {
                    res = Max.op(&res, &segtree.query_range(l..r + 1));
                });
                writeln!(output, "{}", res).unwrap();
            }
            _ => panic!(),
        }
    }
}
