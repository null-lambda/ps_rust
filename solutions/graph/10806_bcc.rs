use std::{collections::HashSet, io::Write};

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

fn cut_edges(n: usize, neighbors: &[Vec<u32>]) -> Vec<(u32, u32)> {
    let mut dfs_order = vec![0; n];
    let mut cut_edges = vec![];
    let mut order = 1;
    fn dfs(
        u: u32,
        parent: u32,
        neighbors: &[Vec<u32>],
        dfs_order: &mut Vec<u32>,
        cut_edges: &mut Vec<(u32, u32)>,
        order: &mut u32,
    ) -> u32 {
        dfs_order[u as usize] = *order;
        *order += 1;
        let mut low_u = *order;
        for &v in &neighbors[u as usize] {
            if parent == v {
                continue;
            }
            if dfs_order[v as usize] != 0 {
                low_u = low_u.min(dfs_order[v as usize]);
            } else {
                let low_v = dfs(v, u, neighbors, dfs_order, cut_edges, order);
                if low_v > dfs_order[u as usize] {
                    cut_edges.push((u.min(v), u.max(v)));
                }
                low_u = low_u.min(low_v);
            }
        }
        low_u
    }

    const UNDEFINED: u32 = i32::MAX as u32;
    dfs(
        0,
        UNDEFINED,
        neighbors,
        &mut dfs_order,
        &mut cut_edges,
        &mut order,
    );
    cut_edges
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
    let m: usize = input.value();

    let mut neighbors = vec![vec![]; n];
    let mut edges = HashSet::new();
    let mut double_edges = HashSet::new();

    for _ in 0..m {
        let u: u32 = input.value::<u32>() - 1;
        let v: u32 = input.value::<u32>() - 1;
        neighbors[u as usize].push(v);
        neighbors[v as usize].push(u);

        let e = (u.min(v), u.max(v));
        if edges.contains(&e) {
            double_edges.insert(e);
        } else {
            edges.insert(e);
        }
    }

    let mut cut_edges: HashSet<_> = cut_edges(n, &neighbors).into_iter().collect();
    cut_edges.retain(|e| !double_edges.contains(e));

    let mut dset = DisjointSet::new(n);
    for &(u, v) in &edges {
        if !cut_edges.contains(&(u, v)) {
            dset.merge(u as usize, v as usize);
        }
    }

    let mut cut_neighbors = vec![vec![]; n];
    for &(u, v) in &cut_edges {
        let u = dset.find_root(u as usize);
        let v = dset.find_root(v as usize);
        cut_neighbors[u].push(v);
        cut_neighbors[v].push(u);
    }

    let mut visited = vec![false; n];
    let mut leaves = vec![];
    for u in 0..n {
        let u = dset.find_root(u);
        if visited[u] {
            continue;
        }
        visited[u] = true;
        let mut stack = vec![u];

        while let Some(u) = stack.pop() {
            debug_assert_eq!(u, dset.find_root(u));
            if cut_neighbors[u].len() == 1 {
                leaves.push(u);
            }
            for &v in &cut_neighbors[u] {
                if visited[v] {
                    continue;
                }
                visited[v] = true;
                stack.push(v);
            }
        }
    }

    let l = leaves.len();
    writeln!(output, "{}", l.div_ceil(2)).unwrap();
    for i in 0..l / 2 {
        writeln!(output, "{} {}", leaves[i] + 1, leaves[i + l / 2] + 1).unwrap();
    }
    if l % 2 != 0 {
        writeln!(output, "{} {}", leaves[0] + 1, leaves[l - 1] + 1).unwrap();
    }
}
