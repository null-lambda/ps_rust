use std::io::Write;

use collections::DisjointSet;

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
                visitor(self.chain_top[u] as usize, u);
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.euler_idx[u] > self.euler_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            visitor(u, v);
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

pub mod fenwick_tree {
    pub trait Group {
        type Elem: Clone;
        fn id(&self) -> Self::Elem;
        fn add_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem);
        fn sub_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem);
    }

    pub struct FenwickTree<G: Group> {
        n: usize,
        n_ceil: usize,
        group: G,
        data: Vec<G::Elem>,
    }

    impl<G: Group> FenwickTree<G> {
        pub fn new(n: usize, group: G) -> Self {
            let n_ceil = n.next_power_of_two();
            let data = (0..n_ceil).map(|_| group.id()).collect();
            Self {
                n,
                n_ceil,
                group,
                data,
            }
        }

        pub fn add(&mut self, mut idx: usize, value: G::Elem) {
            while idx < self.n {
                self.group.add_assign(&mut self.data[idx], value.clone());
                idx |= idx + 1;
            }
        }
        pub fn get(&self, idx: usize) -> G::Elem {
            self.sum_range(idx..idx + 1)
        }

        pub fn sum_range(&self, range: std::ops::Range<usize>) -> G::Elem {
            let mut res = self.group.id();
            let mut r = range.end;
            while r > 0 {
                self.group.add_assign(&mut res, self.data[r - 1].clone());
                r &= r - 1;
            }

            let mut l = range.start;
            while l > 0 {
                self.group.sub_assign(&mut res, self.data[l - 1].clone());
                l &= l - 1;
            }

            res
        }
    }
}

use fenwick_tree::*;

struct Additive<T>(std::marker::PhantomData<T>);

impl<T> Additive<T> {
    fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<T: Default + std::ops::AddAssign + std::ops::SubAssign + Clone> Group for Additive<T> {
    type Elem = T;
    fn id(&self) -> Self::Elem {
        T::default()
    }
    fn add_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem) {
        *lhs += rhs;
    }
    fn sub_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem) {
        *lhs -= rhs;
    }
}

mod collections {
    use std::cell::Cell;

    pub struct DisjointSet {
        parent: Vec<Cell<usize>>,
        size: Vec<u32>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self {
                parent: (0..n).map(|i| Cell::new(i)).collect(),
                size: vec![1; n],
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            if u == self.parent[u].get() {
                u
            } else {
                self.parent[u].set(self.find_root(self.parent[u].get()));
                self.parent[u].get()
            }
        }

        pub fn get_size(&self, u: usize) -> u32 {
            self.size[self.find_root(u)]
        }

        // returns whether two set were different
        pub fn merge(&mut self, mut u: usize, mut v: usize) -> bool {
            u = self.find_root(u);
            v = self.find_root(v);
            if u == v {
                return false;
            }
            if self.size[u] > self.size[v] {
                std::mem::swap(&mut u, &mut v);
            }
            self.parent[v].set(u);
            self.size[u] += self.size[v];
            true
        }
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let weights_orig: Vec<i64> = (0..n).map(|_| input.value()).collect();

    let q: usize = input.value();
    let queries: Vec<_> = (0..q)
        .map(|_| {
            let cmd = input.token().as_bytes()[0];
            let a: u32 = input.value();
            let b: u32 = input.value();
            (cmd, a, b)
        })
        .collect();

    let mut neighbors = vec![vec![]; n];
    let mut dset = DisjointSet::new(n);
    for &(cmd, a, b) in queries.iter() {
        if cmd == b'b' && dset.merge(a as usize - 1, b as usize - 1) {
            neighbors[a as usize - 1].push(b - 1);
            neighbors[b as usize - 1].push(a - 1);
        }
    }

    for u in 0..n {
        if dset.merge(0, u) {
            neighbors[0].push(u as u32);
            neighbors[u].push(0);
        }
    }

    let mut children = vec![vec![]; n];
    let mut visited = vec![false; n];
    visited[0] = true;
    let mut stack = vec![0];
    while let Some(u) = stack.pop() {
        for &v in &neighbors[u] {
            if visited[v as usize] {
                continue;
            }
            visited[v as usize] = true;
            children[u].push(v);
            stack.push(v as usize);
        }
    }

    let hld = hld::HLD::from_graph(&children);
    let mut tree = FenwickTree::new(n, Additive::<i64>::new());

    for u in 0..n {
        tree.add(hld.euler_idx[u] as usize, weights_orig[u]);
    }

    let mut dset = DisjointSet::new(n);
    for (cmd, a, b) in queries {
        match cmd {
            b'b' => {
                if dset.merge(a as usize - 1, b as usize - 1) {
                    writeln!(output, "yes").unwrap();
                } else {
                    writeln!(output, "no").unwrap();
                }
            }
            b'p' => {
                let old = tree.get(hld.euler_idx[a as usize - 1] as usize);
                tree.add(hld.euler_idx[a as usize - 1] as usize, b as i64 - old);
            }
            b'e' => {
                if dset.find_root(a as usize - 1) != dset.find_root(b as usize - 1) {
                    writeln!(output, "impossible").unwrap();
                } else {
                    let mut ans = 0;
                    hld.for_each_path(a as usize - 1, b as usize - 1, |u, v| {
                        ans += tree
                            .sum_range(hld.euler_idx[u] as usize..hld.euler_idx[v] as usize + 1);
                    });
                    writeln!(output, "{}", ans).unwrap();
                }
            }
            _ => panic!(),
        }
    }
}
