use std::{collections::HashSet, io::Write};

use dset::DisjointSet;

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

mod dset {
    use std::{cell::Cell, mem};

    pub struct DisjointSet {
        // Represents parent if >= 0, size if < 0
        parent_or_size: Vec<Cell<i32>>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self {
                parent_or_size: vec![Cell::new(-1); n],
            }
        }

        fn get_parent_or_size(&self, u: usize) -> Result<usize, u32> {
            let x = self.parent_or_size[u].get();
            if x >= 0 {
                Ok(x as usize)
            } else {
                Err((-x) as u32)
            }
        }

        fn set_parent(&self, u: usize, p: usize) {
            self.parent_or_size[u].set(p as i32);
        }

        fn set_size(&self, u: usize, s: u32) {
            self.parent_or_size[u].set(-(s as i32));
        }

        pub fn find_root_with_size(&self, u: usize) -> (usize, u32) {
            match self.get_parent_or_size(u) {
                Ok(p) => {
                    let (root, size) = self.find_root_with_size(p);
                    self.set_parent(u, root);
                    (root, size)
                }
                Err(size) => (u, size),
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            self.find_root_with_size(u).0
        }

        pub fn get_size(&self, u: usize) -> u32 {
            self.find_root_with_size(u).1
        }

        // Returns true if two sets were previously disjoint
        pub fn merge(&mut self, u: usize, v: usize) -> bool {
            let (mut u, size_u) = self.find_root_with_size(u);
            let (mut v, size_v) = self.find_root_with_size(v);
            if u == v {
                return false;
            }

            if size_u < size_v {
                mem::swap(&mut u, &mut v);
            }
            self.set_parent(v, u);
            self.set_size(u, size_u + size_v);
            true
        }
    }
}

fn dfs_msf(neighbors: &[Vec<usize>], parents: &mut [usize], depth: &mut [u32], u: usize, p: usize) {
    parents[u] = p;
    for &v in &neighbors[u] {
        if v != p {
            depth[v] = depth[u] + 1;
            dfs_msf(neighbors, parents, depth, v, u);
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let z: usize = input.value();

    let mut edges_residual: HashSet<(u32, u32)> = (0..m)
        .map(|_| {
            let mut u = input.value::<u32>() - 1;
            let mut v = input.value::<u32>() - 1;
            if u > v {
                std::mem::swap(&mut u, &mut v);
            }
            (u, v)
        })
        .collect();

    let mut queries = vec![];
    for _ in 0..z {
        let cmd = input.token();
        let mut u = input.value::<u32>() - 1;
        let mut v = input.value::<u32>() - 1;
        if u > v {
            std::mem::swap(&mut u, &mut v);
        }

        queries.push((cmd, u, v));
        match cmd {
            "Z" => {
                edges_residual.remove(&(u, v));
            }
            _ => {}
        }
    }

    let mut msf_edges = vec![]; // Minimum spanning forest
    {
        let mut ccs = DisjointSet::new(n); // Connected components
        for (u, v) in (edges_residual.iter().copied()).chain(
            queries
                .iter()
                .rev()
                .filter(|(cmd, ..)| cmd == &"Z")
                .map(|&(_, u, v)| (u, v)),
        ) {
            if ccs.merge(u as usize, v as usize) {
                msf_edges.push((u, v));
            }
        }
    }

    let mut msf_neighbors = vec![vec![]; n];
    for (u, v) in msf_edges {
        msf_neighbors[u as usize].push(v as usize);
        msf_neighbors[v as usize].push(u as usize);
    }

    const UNSET: usize = std::usize::MAX;
    let mut msf_parent = vec![UNSET; n];
    let mut msf_depth = vec![0; n];
    for u in 0..n {
        if msf_parent[u] != UNSET {
            continue;
        }
        dfs_msf(&msf_neighbors, &mut msf_parent, &mut msf_depth, u, u);
    }

    let mut ccs = DisjointSet::new(n); // Connected components
    let mut two_ccs = DisjointSet::new(n); // 2-edge-connected components
    let add_edge =
        |mut u: usize, mut v: usize, ccs: &mut DisjointSet, two_ccs: &mut DisjointSet| {
            if ccs.merge(u, v) {
                return;
            }

            loop {
                if u == v {
                    return;
                }

                if msf_depth[u] < msf_depth[v] {
                    std::mem::swap(&mut u, &mut v);
                }

                let p = msf_parent[u];
                two_ccs.merge(p, u);
                u = p;
            }
        };

    for (u, v) in edges_residual {
        add_edge(u as usize, v as usize, &mut ccs, &mut two_ccs);
    }

    let mut ans_rev = vec![];
    for (cmd, u, v) in queries.into_iter().rev() {
        match cmd {
            "Z" => {
                add_edge(u as usize, v as usize, &mut ccs, &mut two_ccs);
            }
            "P" => {
                let res = two_ccs.find_root(u as usize) == two_ccs.find_root(v as usize);
                ans_rev.push(res);
            }
            _ => panic!(),
        }
    }

    for ans in ans_rev.into_iter().rev() {
        writeln!(output, "{}", if ans { "TAK" } else { "NIE" }).unwrap();
    }
}
