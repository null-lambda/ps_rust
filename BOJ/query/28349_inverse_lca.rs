use std::io::Write;

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

pub mod hld {
    // Heavy-Light Decomposition
    #[inline(always)]
    pub unsafe fn assert_unchecked(b: bool) {
        if !b {
            std::hint::unreachable_unchecked();
        }
    }

    #[inline(always)]
    pub fn likely(b: bool) -> bool {
        #[cold]
        #[inline(always)]
        pub fn cold() {}

        if !b {
            cold();
        }
        b
    }

    const UNSET: u32 = u32::MAX;

    #[derive(Debug, Clone)]
    pub struct HLD {
        pub size: Vec<u32>,
        pub parent: Vec<u32>,
        pub heavy_child: Vec<u32>,
        pub chain_top: Vec<u32>,
        pub chain_bot: Vec<u32>,
        pub segmented_idx: Vec<u32>,
        pub topological_order: Vec<u32>,
    }

    impl HLD {
        pub fn len(&self) -> usize {
            self.parent.len()
        }

        pub fn from_edges<'a>(
            n: usize,
            edges: impl IntoIterator<Item = [u32; 2]>,
            root: usize,
            use_dfs_ordering: bool,
        ) -> Self {
            // Fast tree reconstruction with XOR-linked tree traversal
            // https://codeforces.com/blog/entry/135239
            let mut degree = vec![0u32; n];
            let mut xor_neighbors: Vec<u32> = vec![0u32; n];
            for [u, v] in edges {
                debug_assert!(u != v);
                degree[u as usize] += 1;
                degree[v as usize] += 1;
                xor_neighbors[u as usize] ^= v;
                xor_neighbors[v as usize] ^= u;
            }

            let mut size = vec![1; n];
            let mut heavy_child = vec![UNSET; n];
            let mut chain_bot = vec![UNSET; n];
            degree[root] += 2;
            let mut topological_order = Vec::with_capacity(n);
            for mut u in 0..n {
                while degree[u] == 1 {
                    // Topological sort
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

                    let h = heavy_child[u as usize];
                    chain_bot[u] = if h == UNSET {
                        u as u32
                    } else {
                        chain_bot[h as usize]
                    };

                    assert!(u != p as usize);
                    u = p as usize;
                }
            }
            topological_order.push(root as u32);
            assert!(topological_order.len() == n, "Invalid tree structure");

            let h = heavy_child[root];
            chain_bot[root] = if h == UNSET {
                root as u32
            } else {
                chain_bot[h as usize]
            };

            let mut parent = xor_neighbors;
            parent[root] = UNSET;

            // Downward propagation
            let mut chain_top = vec![root as u32; n];
            let mut segmented_idx = vec![UNSET; n];
            if !use_dfs_ordering {
                // A rearranged topological index continuous in a chain, for path queries
                let mut timer = 0;
                for mut u in topological_order.iter().copied().rev() {
                    if segmented_idx[u as usize] != UNSET {
                        continue;
                    }
                    let u0 = u;
                    loop {
                        chain_top[u as usize] = u0;
                        segmented_idx[u as usize] = timer;
                        timer += 1;
                        u = heavy_child[u as usize];
                        if u == UNSET {
                            break;
                        }
                    }
                }
            } else {
                // DFS ordering for path & subtree queries
                let mut offset = vec![0; n];
                for mut u in topological_order.iter().copied().rev() {
                    if segmented_idx[u as usize] != UNSET {
                        continue;
                    }

                    let mut p = parent[u as usize];
                    let mut timer = 0;
                    if likely(p != UNSET) {
                        timer = offset[p as usize] + 1;
                        offset[p as usize] += size[u as usize] as u32;
                    }

                    let u0 = u;
                    loop {
                        chain_top[u as usize] = u0;
                        offset[u as usize] = timer;
                        segmented_idx[u as usize] = timer;
                        timer += 1;

                        p = u as u32;
                        u = heavy_child[p as usize];
                        unsafe { assert_unchecked(u != p) };
                        if u == UNSET {
                            break;
                        }
                        offset[p as usize] += size[u as usize] as u32;
                    }
                }
            }

            Self {
                size,
                parent,
                heavy_child,
                chain_top,
                chain_bot,
                segmented_idx,
                topological_order,
            }
        }

        pub fn for_each_path<F>(&self, mut u: usize, mut v: usize, mut visitor: F)
        where
            F: FnMut(usize, usize, bool),
        {
            debug_assert!(u < self.len() && v < self.len());

            while self.chain_top[u] != self.chain_top[v] {
                if self.segmented_idx[self.chain_top[u] as usize]
                    < self.segmented_idx[self.chain_top[v] as usize]
                {
                    std::mem::swap(&mut u, &mut v);
                }
                visitor(self.chain_top[u] as usize, u, false);
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.segmented_idx[u] > self.segmented_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            visitor(u, v, true);
        }

        pub fn for_each_path_splitted<F>(&self, mut u: usize, mut v: usize, mut visit: F)
        where
            F: FnMut(usize, usize, bool, bool),
        {
            debug_assert!(u < self.len() && v < self.len());
            while self.chain_top[u] != self.chain_top[v] {
                if self.segmented_idx[self.chain_top[u] as usize]
                    > self.segmented_idx[self.chain_top[v] as usize]
                {
                    visit(self.chain_top[u] as usize, u, true, false);
                    u = self.parent[self.chain_top[u] as usize] as usize;
                } else {
                    visit(self.chain_top[v] as usize, v, false, false);
                    v = self.parent[self.chain_top[v] as usize] as usize;
                }
            }
            if self.segmented_idx[u] > self.segmented_idx[v] {
                visit(v, u, true, true);
            } else {
                visit(u, v, false, true);
            }
        }

        pub fn lca(&self, mut u: usize, mut v: usize) -> usize {
            debug_assert!(u < self.len() && v < self.len());
            while self.chain_top[u] != self.chain_top[v] {
                if self.segmented_idx[self.chain_top[u] as usize]
                    < self.segmented_idx[self.chain_top[v] as usize]
                {
                    std::mem::swap(&mut u, &mut v);
                }
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.segmented_idx[u] > self.segmented_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            u
        }
    }
}

pub mod jagged {
    use std::fmt::Debug;
    use std::mem::MaybeUninit;
    use std::ops::{Index, IndexMut};

    // Trait for painless switch between different representations of a jagged array
    pub trait Jagged<T>: IndexMut<usize, Output = [T]> {
        fn len(&self) -> usize;
    }

    impl<T, C> Jagged<T> for C
    where
        C: AsRef<[Vec<T>]> + IndexMut<usize, Output = [T]>,
    {
        fn len(&self) -> usize {
            <Self as AsRef<[Vec<T>]>>::as_ref(self).len()
        }
    }

    // Compressed sparse row format for jagged array
    // Provides good locality for graph traversal, but works only for static ones.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CSR<T> {
        data: Vec<T>,
        head: Vec<u32>,
    }

    impl<T> Debug for CSR<T>
    where
        T: Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let v: Vec<Vec<&T>> = (0..self.len()).map(|i| self[i].iter().collect()).collect();
            v.fmt(f)
        }
    }

    impl<T, I> FromIterator<I> for CSR<T>
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
            CSR { data, head }
        }
    }

    impl<T: Clone> CSR<T> {
        pub fn from_pairs(n: usize, pairs: impl Iterator<Item = (u32, T)> + Clone) -> Self {
            let mut head = vec![0u32; n + 1];

            for (u, _) in pairs.clone() {
                debug_assert!(u < n as u32);
                head[u as usize] += 1;
            }
            for i in 0..n {
                head[i + 1] += head[i];
            }
            let mut data: Vec<_> = (0..head[n]).map(|_| MaybeUninit::uninit()).collect();

            for (u, v) in pairs {
                head[u as usize] -= 1;
                data[head[u as usize] as usize] = MaybeUninit::new(v.clone());
            }

            // Rustc is likely to perform in‑place iteration without new allocation.
            // [https://doc.rust-lang.org/stable/std/iter/trait.FromIterator.html#impl-FromIterator%3CT%3E-for-Vec%3CT%3E]
            let data = data
                .into_iter()
                .map(|x| unsafe { x.assume_init() })
                .collect();

            CSR { data, head }
        }
    }

    impl<T> Index<usize> for CSR<T> {
        type Output = [T];

        fn index(&self, index: usize) -> &Self::Output {
            &self.data[self.head[index] as usize..self.head[index + 1] as usize]
        }
    }

    impl<T> IndexMut<usize> for CSR<T> {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            &mut self.data[self.head[index] as usize..self.head[index + 1] as usize]
        }
    }

    impl<T> Jagged<T> for CSR<T> {
        fn len(&self) -> usize {
            self.head.len() - 1
        }
    }
}

pub mod bcc {
    /// Biconnected components & 2-edge-connected components
    /// Verified with [Yosupo library checker](https://judge.yosupo.jp/problem/biconnected_components)
    use super::jagged;

    pub const UNSET: u32 = !0;

    pub struct BlockCutForest<'a, E, J> {
        // DFS tree structure
        pub neighbors: &'a J,
        pub parent: Vec<u32>,
        pub euler_in: Vec<u32>,
        pub low: Vec<u32>, // Lowest euler index on a subtree's back edge

        /// Block-cut tree structure,  
        /// represented as a rooted bipartite tree between  
        /// vertex nodes (indices in 0..n) and virtual BCC nodes (indices in n..).  
        /// A vertex node is a cut vertex iff its degree is >= 2,
        /// and the neighbors of a virtual BCC node represents all its belonging vertices.
        pub bct_parent: Vec<u32>,
        pub bct_degree: Vec<u32>,

        /// BCC structure
        pub bcc_edges: Vec<Vec<(u32, u32, E)>>,
    }

    impl<'a, E: 'a + Copy, J: jagged::Jagged<(u32, E)>> BlockCutForest<'a, E, J> {
        pub fn from_assoc_list(neighbors: &'a J) -> Self {
            let n = neighbors.len();

            let mut parent = vec![UNSET; n];
            let mut low = vec![0; n];
            let mut euler_in = vec![0; n];
            let mut timer = 1u32;

            let mut bct_parent = vec![UNSET; n];
            let mut bct_degree = vec![1u32; n];

            let mut bcc_edges = vec![];

            bct_parent.reserve_exact(n * 2);

            let mut current_edge = vec![0u32; n];
            let mut stack = vec![];
            let mut edges_stack: Vec<(u32, u32, E)> = vec![];
            for root in 0..n {
                if euler_in[root] != 0 {
                    continue;
                }

                bct_degree[root] -= 1;
                parent[root] = UNSET;
                let mut u = root as u32;
                loop {
                    let p = parent[u as usize];
                    let iv = &mut current_edge[u as usize];
                    if *iv == 0 {
                        // On enter
                        euler_in[u as usize] = timer;
                        low[u as usize] = timer + 1;
                        timer += 1;
                        stack.push(u);
                    }
                    if (*iv as usize) == neighbors[u as usize].len() {
                        // On exit
                        if p == UNSET {
                            break;
                        }

                        low[p as usize] = low[p as usize].min(low[u as usize]);
                        if low[u as usize] >= euler_in[p as usize] {
                            // Found a BCC
                            let bcc_node = bct_parent.len() as u32;
                            bct_degree[p as usize] += 1;

                            bct_parent.push(p);
                            bct_degree.push(1);

                            while let Some(c) = stack.pop() {
                                bct_parent[c as usize] = bcc_node;
                                bct_degree[bcc_node as usize] += 1;

                                if c == u {
                                    break;
                                }
                            }

                            let mut es = vec![];
                            while let Some(e) = edges_stack.pop() {
                                es.push(e);
                                if (e.0, e.1) == (p, u) {
                                    break;
                                }
                            }
                            bcc_edges.push(es);
                        }

                        u = p;
                        continue;
                    }

                    let (v, w) = neighbors[u as usize][*iv as usize];
                    *iv += 1;
                    if v == p {
                        continue;
                    }

                    if euler_in[v as usize] < euler_in[u as usize] {
                        // Unvisited edge
                        edges_stack.push((u, v, w));
                    }
                    if euler_in[v as usize] != 0 {
                        // Back edge
                        low[u as usize] = low[u as usize].min(euler_in[v as usize]);
                        continue;
                    }

                    // Forward edge (a part of DFS spanning tree)
                    parent[v as usize] = u;
                    u = v;
                }

                // For an isolated vertex, manually add a virtual BCC node.
                if neighbors[root].is_empty() {
                    bct_degree[root] += 1;

                    bct_parent.push(root as u32);
                    bct_degree.push(1);

                    bcc_edges.push(vec![]);
                }
            }

            Self {
                neighbors,
                parent,
                low,
                euler_in,

                bct_parent,
                bct_degree,

                bcc_edges,
            }
        }

        pub fn is_cut_vert(&self, u: usize) -> bool {
            debug_assert!(u < self.neighbors.len());
            self.bct_degree[u] >= 2
        }

        pub fn is_bridge(&self, u: usize, v: usize) -> bool {
            debug_assert!(u < self.neighbors.len() && v < self.neighbors.len() && u != v);
            self.euler_in[v] < self.low[u] || self.euler_in[u] < self.low[v]
        }

        pub fn bcc_node_range(&self) -> std::ops::Range<usize> {
            self.neighbors.len()..self.bct_parent.len()
        }

        pub fn get_bccs(&self) -> Vec<Vec<u32>> {
            let mut bccs = vec![vec![]; self.bcc_node_range().len()];
            let n = self.neighbors.len();
            for u in 0..n {
                let b = self.bct_parent[u];
                if b != UNSET {
                    bccs[b as usize - n].push(u as u32);
                }
            }
            for b in self.bcc_node_range() {
                bccs[b - n].push(self.bct_parent[b]);
            }
            bccs
        }

        pub fn get_2ccs(&self) -> Vec<Vec<u32>> {
            unimplemented!()
        }
    }
}

mod dset {
    use std::{cell::Cell, mem};

    #[derive(Clone)]
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

// Decremental connectivity
pub mod dec_conn {

    fn sorted2<T: PartialOrd>(mut xs: [T; 2]) -> [T; 2] {
        if xs[0] > xs[1] {
            xs.swap(0, 1)
        }
        xs
    }

    // A prototype that takes O(N) for each query
    pub mod prototype {
        use super::*;
        use crate::dset::*;
        use std::collections::HashSet;

        // ## Requirements:
        //   O(polylog n) cut, is_connected
        //   O(polylog n) set_label, get_label
        //
        // - copy-paste HDLT or something simpler
        #[derive(Clone)]
        pub struct Labeled<T> {
            n: usize,
            edges: HashSet<[u32; 2]>,
            label: Vec<T>,
        }

        impl<T: Clone> Labeled<T> {
            pub fn new(
                n: usize,
                edges: impl IntoIterator<Item = [u32; 2]>,
                weights: Vec<T>,
            ) -> Self {
                Self {
                    n,
                    edges: HashSet::from_iter(edges.into_iter().map(sorted2)),
                    label: weights,
                }
            }

            pub fn is_connected(&self, u: u32, v: u32) -> bool {
                let mut conn = DisjointSet::new(self.n);
                for &[u, v] in &self.edges {
                    conn.merge(u as usize, v as usize);
                }
                conn.find_root(u as usize) == conn.find_root(v as usize)
            }

            pub fn cut(&mut self, u: u32, v: u32) -> bool {
                self.edges.remove(&sorted2([u, v]))
            }

            pub fn set_label(&mut self, u: u32, x: T) {
                let roots = self.find_roots();
                for &v in &self.find_components()[roots[u as usize] as usize] {
                    self.label[v as usize] = x.clone();
                }
            }

            pub fn get_label(&mut self, u: u32) -> T {
                self.label[u as usize].clone()
            }

            fn find_roots(&self) -> Vec<u32> {
                let mut conn = DisjointSet::new(self.n);
                for &[u, v] in &self.edges {
                    conn.merge(u as usize, v as usize);
                }

                let mut roots = vec![!0u32; self.n];
                for u in 0..self.n {
                    roots[u] = conn.find_root(u as usize) as u32;
                }
                roots
            }

            fn find_components(&self) -> Vec<Vec<u32>> {
                let roots = self.find_roots();
                let mut components = vec![vec![]; self.n];
                for u in 0..self.n as u32 {
                    components[roots[u as usize] as usize].push(u);
                }
                components
            }
        }

        // Notify once if the path u ~ v is disconnected in the graph minus {w}
        //     <=> the path (u, v) is disconnected in the block-cut tree minus {w}
        //   (Constraints: u, v, and w are pairwise distinct)
        //
        // ## Requirements:
        //   O(polylog n) cut
        //
        // ## Notes:
        // - We may maintain a link-cut tree augmented with distance aggregates.
        //     Q. Is the total number of edge additions/removals on the block-cut tree reasonable?
        //     A. Likely yes, if we apply smaller-to-larger amortization in reverse.
        //     Q. Would O(n log n) link/cut operations be fast enough?
        //
        // [https://arxiv.org/abs/2503.21733]?
        pub struct CutNotify {
            n: usize,
            edges: HashSet<[u32; 2]>,
            queries: Vec<([u32; 2], u32)>,
        }

        impl CutNotify {
            pub fn new(
                n: usize,
                edges: impl IntoIterator<Item = [u32; 2]>,
                queries: impl IntoIterator<Item = ([u32; 2], u32)>,
            ) -> Self {
                Self {
                    n,
                    edges: HashSet::from_iter(edges.into_iter().map(sorted2)),
                    queries: Vec::from_iter(queries),
                }
            }

            pub fn cut(&mut self, u: u32, v: u32, mut notify: impl FnMut(&u32)) -> bool {
                if !self.edges.remove(&sorted2([u, v])) {
                    return false;
                }

                let neighbors = crate::jagged::CSR::from_pairs(
                    self.n,
                    self.edges
                        .iter()
                        .flat_map(|&[u, v]| [(u, (v, ())), (v, (u, ()))]),
                );
                let bct = crate::bcc::BlockCutForest::from_assoc_list(&neighbors);

                let s = bct.bct_parent.len();
                let mut bct_conn = DisjointSet::new(s);
                for u in 0..s {
                    let p = bct.bct_parent[u];
                    if p == crate::bcc::UNSET {
                        continue;
                    }
                    bct_conn.merge(u, p as usize);
                }

                let mut inv_map = vec![0u32; s];
                let mut size = vec![0u32; s];
                let mut edges = vec![vec![]; s];
                for u in 0..s {
                    let r = bct_conn.find_root(u) as usize;

                    inv_map[u] = size[r];
                    size[r] += 1;

                    let p = bct.bct_parent[u] as usize;
                    if p != crate::bcc::UNSET as usize {
                        edges[r].push([u as u32, p as u32]);
                    }
                }

                let mut bct_dist = vec![None; s];
                for r in 0..s {
                    let es = &edges[r];
                    if es.is_empty() {
                        continue;
                    }
                    let es = es
                        .iter()
                        .map(|&[u, v]| [inv_map[u as usize], inv_map[v as usize]]);

                    let t = edges[r].len() + 1;
                    let hld = crate::hld::HLD::from_edges(t, es, 0, false);
                    let mut depth = vec![0u32; t];
                    for &u in hld.topological_order.iter().rev().skip(1) {
                        depth[u as usize] = depth[hld.parent[u as usize] as usize] + 1;
                    }

                    bct_dist[r] = Some(move |u: usize, v: usize| {
                        depth[u] + depth[v] - depth[hld.lca(u, v)] * 2
                    });
                }

                self.queries.retain(|([u, v], w)| {
                    let ru = bct_conn.find_root(*u as usize);
                    let rv = bct_conn.find_root(*v as usize);
                    if ru != rv {
                        notify(w);
                        return false;
                    }

                    let iu = inv_map[*u as usize] as usize;
                    let iv = inv_map[*v as usize] as usize;
                    let iw = inv_map[*w as usize] as usize;

                    let dist = bct_dist[ru].as_ref().unwrap();
                    if dist(iu, iv) == dist(iu, iw) + dist(iw, iv) {
                        notify(w);
                        return false;
                    }

                    true
                });

                true
            }
        }
    }
}

fn solve_cubic(n: usize, conds: &[[u32; 3]]) -> Option<Vec<u32>> {
    let mut conds = conds.to_vec();
    let mut stack = vec![((0..n as u32).collect::<Vec<_>>(), 0)];
    let mut parent = vec![0u32; n];
    'outer: while let Some((cs, p)) = stack.pop() {
        'check_root: for &r in &cs {
            let mut conn = dset::DisjointSet::new(n);
            for &[u, v, w] in &conds {
                if (u == r || v == r) && w != r {
                    continue 'check_root;
                } else if w != r {
                    conn.merge(w as usize, u as usize);
                    conn.merge(w as usize, v as usize);
                }
            }

            for &[u, v, w] in &conds {
                if u != r
                    && v != r
                    && w == r
                    && conn.find_root(u as usize) == conn.find_root(v as usize)
                {
                    continue 'check_root;
                }
            }

            let mut components = vec![vec![]; n];
            for &u in &cs {
                if u == r {
                    continue;
                }
                components[conn.find_root(u as usize)].push(u);
            }
            stack.extend(
                components
                    .into_iter()
                    .filter(|cs| !cs.is_empty())
                    .map(|cs| (cs, r + 1)),
            );
            conds.retain(|&[u, v, w]| u != r && v != r && w != r);

            parent[r as usize] = p;
            continue 'outer;
        }
        return None;
    }

    Some(parent)
}

fn solve_quad_polylog(n: usize, conds: &[[u32; 3]]) -> Option<Vec<u32>> {
    let mut desc = vec![vec![]; n];

    let edges = || {
        conds
            .iter()
            .flat_map(|&[u, v, w]| [[u, w], [v, w]])
            .filter(|&[u, v]| u != v)
    };

    // A lock for two types of constraints - desc and sep
    let mut indegree = vec![0i32; n];
    let mut sep_edges = vec![];

    for [u, p] in edges() {
        desc[p as usize].push(u);
        indegree[u as usize] += 1;
    }

    for &[u, v, w] in conds {
        if u == w {
            continue;
        }

        // TODO: replace with a static block-cut tree and tree distance queries
        let mut conn_minus_w = dset::DisjointSet::new(n);
        for [u, v] in edges() {
            if u != w && v != w {
                conn_minus_w.merge(u as usize, v as usize);
            }
        }
        if conn_minus_w.find_root(u as usize) == conn_minus_w.find_root(v as usize) {
            indegree[w as usize] += 1;
            sep_edges.push(([u, v], w));
        }
    }

    let mut parent = vec![0u32; n];
    let mut conn_labeled = dec_conn::prototype::Labeled::new(n, edges(), vec![0u32; n]);
    let mut notifier = dec_conn::prototype::CutNotify::new(n, edges(), sep_edges);

    let mut toposort: Vec<_> = (0..n as u32)
        .filter(|&u| indegree[u as usize] == 0)
        .collect();
    let mut timer = 0;

    while let Some(&r) = toposort.get(timer) {
        timer += 1;
        if indegree[r as usize] != 0 {
            continue;
        }

        parent[r as usize] = conn_labeled.get_label(r);
        conn_labeled.set_label(r, r + 1);

        // Delete the vertex r from the connectivity graph
        let mut mark = |u: u32| {
            indegree[u as usize] -= 1;
            if indegree[u as usize] == 0 {
                toposort.push(u);
            }
        };
        for &u in &desc[r as usize] {
            conn_labeled.cut(r, u);
            notifier.cut(r, u, |&w| mark(w));
            mark(u);
        }
    }

    if toposort.len() < n {
        return None;
    }

    // Merge the forest into a single tree
    let roots: Vec<_> = (0..n as u32).filter(|&u| parent[u as usize] == 0).collect();
    let (&r0, rest) = roots.split_first().unwrap();
    for &r in rest {
        parent[r as usize] = r0 + 1;
    }

    Some(parent)
}

fn solve_linear_polylog(n: usize, conds: &[[u32; 3]]) -> Option<Vec<u32>> {
    // dynamic/decremental (bi)connectivity? online? yuck...
    todo!()
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for _ in 0..input.value() {
        let n: usize = input.value();
        let m: usize = input.value();

        let mut conds = vec![];
        for _ in 0..m {
            let mut u = input.value::<u32>() - 1;
            let mut v = input.value::<u32>() - 1;
            let j = input.value::<u32>() - 1;
            if v == j {
                std::mem::swap(&mut u, &mut v);
            }
            conds.push([u, v, j]);
        }

        // let ans = solve_cubic(n, &conds);
        let ans = solve_quad_polylog(n, &conds);
        // let ans = solve_linear_polylog(n, &conds);

        if let Some(parent) = ans {
            for p in parent {
                write!(output, "{} ", p).unwrap();
            }
            writeln!(output).unwrap();
        } else {
            writeln!(output, "NIE").unwrap();
        }
    }
}
