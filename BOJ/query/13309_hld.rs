use std::io::Write;

use collections::Jagged;
use hld::HLD;
use segtree::SegTree;

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

pub mod collections {
    use std::fmt::Debug;
    use std::ops::Index;

    // compressed sparse row format for jagged array
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Jagged<T> {
        data: Vec<T>,
        head: Vec<u32>,
    }

    impl<T> Debug for Jagged<T>
    where
        T: Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let v: Vec<Vec<&T>> = (0..self.len()).map(|i| self[i].iter().collect()).collect();
            v.fmt(f)
        }
    }

    impl<T, I> FromIterator<I> for Jagged<T>
    where
        I: IntoIterator<Item = T>,
    {
        fn from_iter<J>(iter: J) -> Self
        where
            J: IntoIterator<Item = I>,
        {
            let mut data = vec![];
            let mut head = vec![];
            head.push(0);

            let mut cnt = 0;
            for row in iter {
                data.extend(row.into_iter().inspect(|_| cnt += 1));
                head.push(cnt);
            }
            Jagged { data, head }
        }
    }

    impl<T> Jagged<T> {
        pub fn len(&self) -> usize {
            self.head.len() - 1
        }
    }

    impl<T> Index<usize> for Jagged<T> {
        type Output = [T];
        fn index(&self, index: usize) -> &[T] {
            let start = self.head[index] as usize;
            let end = self.head[index + 1] as usize;
            &self.data[start..end]
        }
    }

    impl<T> Jagged<T> {
        pub fn iter(&self) -> Iter<T> {
            Iter { src: self, pos: 0 }
        }
    }

    impl<'a, T> IntoIterator for &'a Jagged<T> {
        type Item = &'a [T];
        type IntoIter = Iter<'a, T>;
        fn into_iter(self) -> Self::IntoIter {
            self.iter()
        }
    }

    pub struct Iter<'a, T> {
        src: &'a Jagged<T>,
        pos: usize,
    }

    impl<'a, T> Iterator for Iter<'a, T> {
        type Item = &'a [T];
        fn next(&mut self) -> Option<Self::Item> {
            if self.pos < self.src.len() {
                let item = &self.src[self.pos];
                self.pos += 1;
                Some(item)
            } else {
                None
            }
        }
    }
}

pub mod hld {
    use crate::collections::Jagged;

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

        fn dfs_size(&mut self, neighbors: &Jagged<u32>, u: usize) {
            self.size[u] = 1;
            for &v in &neighbors[u] {
                if v == self.parent[u] {
                    continue;
                }
                self.depth[v as usize] = self.depth[u] + 1;
                self.parent[v as usize] = u as u32;
                self.dfs_size(neighbors, v as usize);
                self.size[u] += self.size[v as usize];
            }
            if let Some(h) = neighbors[u]
                .iter()
                .copied()
                .filter(|&v| v != self.parent[u])
                .max_by_key(|&v| self.size[v as usize])
            {
                self.heavy_child[u] = h;
            }
        }

        fn dfs_decompose(&mut self, neighbors: &Jagged<u32>, u: usize, order: &mut u32) {
            self.euler_idx[u] = *order;
            *order += 1;
            if self.heavy_child[u] == u32::MAX {
                return;
            }
            let h = self.heavy_child[u];
            self.chain_top[h as usize] = self.chain_top[u];
            self.dfs_decompose(neighbors, h as usize, order);
            for &v in neighbors[u].iter().filter(|&&v| v != h) {
                if v == self.parent[u] {
                    continue;
                }
                self.chain_top[v as usize] = v;
                self.dfs_decompose(neighbors, v as usize, order);
            }
        }

        pub fn from_graph(neighbors: &Jagged<u32>) -> Self {
            let n = neighbors.len();
            let mut hld = Self {
                size: vec![0; n],
                depth: vec![0; n],
                parent: vec![u32::MAX; n],
                heavy_child: vec![u32::MAX; n],
                chain_top: vec![0; n],
                euler_idx: vec![0; n],
            };
            hld.dfs_size(neighbors, 0);
            hld.dfs_decompose(neighbors, 0, &mut 0);
            hld
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

struct Disjunction;

impl segtree::Monoid for Disjunction {
    type Elem = bool;

    fn id(&self) -> Self::Elem {
        false
    }

    fn op(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        *a || *b
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();

    let mut neighbors = vec![vec![]; n];
    for u in 1..n {
        let v = input.value::<u32>() - 1;
        neighbors[v as usize].push(u as u32);
    }

    let hld = HLD::from_graph(&neighbors.into_iter().collect::<Jagged<_>>());
    let mut segtree = SegTree::with_size(n, Disjunction);

    for _ in 0..q {
        let b = input.value::<u32>() - 1;
        let c = input.value::<u32>() - 1;
        let cmd = input.token();

        let mut connected = true;
        hld.for_each_path(b as usize, c as usize, |u, v, is_lca| {
            let mut eu = hld.euler_idx[u] as usize;
            if is_lca {
                eu += 1;
            }
            let ev = hld.euler_idx[v] as usize;
            connected &= !segtree.query_range(eu..ev + 1);
        });
        writeln!(output, "{}", if connected { "YES" } else { "NO" }).unwrap();

        if cmd == "1" {
            if connected {
                segtree.set(hld.euler_idx[b as usize] as usize, true);
            } else {
                segtree.set(hld.euler_idx[c as usize] as usize, true);
            }
        }
    }
}
