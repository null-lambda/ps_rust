use std::{hint::unreachable_unchecked, io::Write};

use jagged::CSR;

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    extern "C" {
        fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64)
            -> *mut u8;
        fn fstat(fd: i32, stat: *mut usize) -> i32;
    }

    pub struct InputAtOnce {
        buf: &'static [u8],
    }

    impl InputAtOnce {
        fn skip(&mut self) {
            loop {
                match self.buf {
                    &[..=b' ', ..] => self.buf = &self.buf[1..],
                    _ => break,
                }
            }
        }

        fn u32_noskip(&mut self) -> u32 {
            let mut acc = 0;
            loop {
                match self.buf {
                    &[b'0'..=b'9', ..] => acc = acc * 10 + (self.buf[0] - b'0') as u32,
                    _ => break,
                }
                self.buf = &self.buf[1..];
            }
            acc
        }

        pub fn token(&mut self) -> &'static str {
            self.skip();
            let start = self.buf.as_ptr();
            loop {
                match self.buf {
                    &[..=b' ', ..] => break,
                    _ => self.buf = &self.buf[1..],
                }
            }
            let end = self.buf.as_ptr();
            unsafe {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    start,
                    end.offset_from(start) as usize,
                ))
            }
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }

        pub fn u32(&mut self) -> u32 {
            self.skip();
            self.u32_noskip()
        }

        pub fn i32(&mut self) -> i32 {
            self.skip();
            match self.buf {
                &[b'-', ..] => {
                    self.buf = &self.buf[1..];
                    -(self.u32_noskip() as i32)
                }
                _ => self.u32_noskip() as i32,
            }
        }
    }

    pub fn stdin() -> InputAtOnce {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let buf = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        let buf =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(buf, stat[6])) };
        InputAtOnce {
            buf: buf.as_bytes(),
        }
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::with_capacity(1 << 16, stdout)
    }
}

pub mod debug {
    pub fn with(#[allow(unused_variables)] f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
    }
}

pub mod jagged {
    use std::fmt::Debug;
    use std::iter;
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

    #[derive(Debug)]
    pub struct Unweighted<J: Jagged<(u32, ())>>(pub J);

    impl<J: Jagged<(u32, ())>> Index<usize> for Unweighted<J> {
        type Output = [u32];
        fn index(&self, index: usize) -> &Self::Output {
            let xs = &self.0[index];
            unsafe { std::slice::from_raw_parts(xs.as_ptr().cast::<u32>(), xs.len()) }
        }
    }

    impl<J: Jagged<(u32, ())>> IndexMut<usize> for Unweighted<J> {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            let xs = &mut self.0[index];
            unsafe { std::slice::from_raw_parts_mut(xs.as_mut_ptr().cast::<u32>(), xs.len()) }
        }
    }

    impl<J: Jagged<(u32, ())>> Jagged<u32> for Unweighted<J> {
        fn len(&self) -> usize {
            self.0.len()
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
        pub fn from_edges(n: usize, pairs: &[(u32, T)]) -> Self {
            let mut head = vec![0u32; n + 1];

            for &(u, _) in pairs {
                debug_assert!(u < n as u32);
                head[u as usize + 1] += 1;
            }
            for i in 2..n + 1 {
                head[i] += head[i - 1];
            }
            let mut data: Vec<_> = iter::repeat_with(|| MaybeUninit::uninit())
                .take(head[n] as usize)
                .collect();
            let mut pos = head.clone();

            for (u, v) in pairs {
                data[pos[*u as usize] as usize] = MaybeUninit::new(v.clone());
                pos[*u as usize] += 1;
            }

            let data = std::mem::ManuallyDrop::new(data);
            let data = unsafe {
                Vec::from_raw_parts(data.as_ptr() as *mut T, data.len(), data.capacity())
            };

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

    #[derive(Debug)]
    pub struct BlockCutTree<'a, J> {
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
        pub bct_children: Vec<Vec<u32>>,
    }

    impl<'a, J: jagged::Jagged<u32>> BlockCutTree<'a, J> {
        pub fn from_neighbors(neighbors: &'a J, root: usize) -> Self {
            let n = neighbors.len();

            let mut parent = vec![UNSET; n];
            let mut low = vec![0; n];
            let mut euler_in = vec![0; n];
            let mut timer = 1u32;

            let mut bct_parent = vec![UNSET; n];
            let mut bct_degree = vec![1u32; n];
            let mut bct_children = vec![vec![]; n];

            bct_parent.reserve_exact(n * 2);

            let mut current_edge = vec![0u32; n];
            let mut stack = vec![];
            let mut edges_stack: Vec<(u32, u32)> = vec![];

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
                        bct_children[p as usize].push(bcc_node);
                        bct_children.push(vec![]);

                        while let Some(c) = stack.pop() {
                            bct_parent[c as usize] = bcc_node;
                            bct_degree[bcc_node as usize] += 1;
                            bct_children[bcc_node as usize].push(c);

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
                    }

                    u = p;
                    continue;
                }

                let v = neighbors[u as usize][*iv as usize];
                *iv += 1;
                if v == p {
                    continue;
                }

                if euler_in[v as usize] < euler_in[u as usize] {
                    // Unvisited edge
                    edges_stack.push((u, v));
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
                bct_children.push(vec![]);
                bct_children[root].push(bct_parent.len() as u32 - 1);
            }

            Self {
                neighbors,
                parent,
                low,
                euler_in,

                bct_parent,
                bct_degree,
                bct_children,
            }
        }

        pub fn bcc_node_range(&self) -> std::ops::Range<usize> {
            self.neighbors.len()..self.bct_parent.len()
        }
    }
}

pub mod static_top_tree {
    // https://github.com/null-lambda/ps_rust/tree/main/library/src/tree
    pub mod rooted {

        use std::{hint::unreachable_unchecked, num::NonZeroU32};

        use crate::{bcc, jagged::Jagged};

        pub const UNSET: u32 = !0;

        #[derive(Debug)]
        pub enum Cluster<C: ClusterCx> {
            Compress(C::C),
            Rake(C::R),
        }

        pub trait ClusterCx: Sized {
            /// Vertex weight / weight of an upward edge (u -> parent(u)).
            type V: Default + Clone;

            type C: Clone; // Path cluster (aggregate on a subchain)
            type R: Clone; // Point cluster (Aggregate of light edges)

            /// Compress monoid.
            /// Left side is always the top side.
            fn id_compress() -> Self::C;
            fn compress(&self, lhs: &Self::C, rhs: &Self::C) -> Self::C;

            /// Rake monoid, commutative.
            fn id_rake() -> Self::R;
            fn rake(&self, lhs: &Self::R, rhs: &Self::R) -> Self::R;

            /// A projection.
            fn collapse_compressed(&self, c: &Self::C) -> Self::R;
            // A leaf cluster, possibly attached to some rake clusters.
            fn make_leaf(
                &self,
                rake_left: Option<&Self::R>,
                weight: &Self::V,
                rake_right: Option<&Self::R>,
            ) -> Self::C;

            /// This is how everything is summed up.
            fn pull_up(
                &self,
                node: &mut Cluster<Self>,
                children: [Option<&mut Cluster<Self>>; 2],
                weight: &Self::V,
            ) {
                use Cluster::*;
                match (node, children) {
                    (Compress(c), [Some(Compress(lhs)), Some(Compress(rhs))]) => {
                        *c = self.compress(lhs, rhs)
                    }
                    (Compress(c), [rl, rr]) => {
                        let rl = rl.map(|r| unsafe { r.get_rake().unwrap_unchecked() });
                        let rr = rr.map(|r| unsafe { r.get_rake().unwrap_unchecked() });
                        *c = self.make_leaf(rl, weight, rr)
                    }
                    (Rake(r), [Some(Rake(lhs)), Some(Rake(rhs))]) => *r = self.rake(lhs, rhs),
                    (Rake(r), [Some(Compress(top)), None]) => *r = self.collapse_compressed(top),
                    _ => unsafe { unreachable_unchecked() },
                }
            }
        }

        impl<Cx: ClusterCx> Clone for Cluster<Cx> {
            fn clone(&self) -> Self {
                match self {
                    Cluster::Compress(c) => Cluster::Compress(c.clone()),
                    Cluster::Rake(r) => Cluster::Rake(r.clone()),
                }
            }
        }

        impl<Cx: ClusterCx> Cluster<Cx> {
            pub fn into_result(self) -> Result<Cx::C, Cx::R> {
                match self {
                    Cluster::Compress(c) => Ok(c),
                    Cluster::Rake(r) => Err(r),
                }
            }

            pub fn into_compress(self) -> Option<Cx::C> {
                match self {
                    Cluster::Compress(c) => Some(c),
                    _ => None,
                }
            }

            pub fn get_compress(&self) -> Option<&Cx::C> {
                match self {
                    Cluster::Compress(c) => Some(c),
                    _ => None,
                }
            }

            pub fn get_compress_mut(&mut self) -> Option<&mut Cx::C> {
                match self {
                    Cluster::Compress(c) => Some(c),
                    _ => None,
                }
            }

            pub fn into_rake(self) -> Option<Cx::R> {
                match self {
                    Cluster::Rake(r) => Some(r),
                    _ => None,
                }
            }

            pub fn get_rake(&self) -> Option<&Cx::R> {
                match self {
                    Cluster::Rake(r) => Some(r),
                    _ => None,
                }
            }

            pub fn get_rake_mut(&mut self) -> Option<&mut Cx::R> {
                match self {
                    Cluster::Rake(r) => Some(r),
                    _ => None,
                }
            }
        }

        /// Heavy-Light Decomposition, prior to top tree construction.
        #[derive(Debug, Default)]
        pub struct HLD {
            // Rooted tree structure
            pub size: Vec<u32>,
            pub topological_order: Vec<u32>,

            // Chain structure
            pub heavy_child: Vec<u32>,
            pub chain_top: Vec<u32>,

            // Rake tree, circularly ordered.
            pub left_light_children: Vec<Vec<u32>>,
            pub right_light_children: Vec<Vec<u32>>,
        }

        impl HLD {
            pub fn len(&self) -> usize {
                self.size.len()
            }

            pub fn from_bct<'a, J: Jagged<u32>>(
                n_verts: usize,
                bct: &bcc::BlockCutTree<'a, J>,
                root: usize,
            ) -> Self {
                assert!(n_verts >= 1);

                let mut bfs = vec![root as u32];
                let mut timer = 0;
                while let Some(&u) = bfs.get(timer) {
                    timer += 1;
                    for &v in &bct.bct_children[u as usize] {
                        if v == bct.bct_parent[u as usize] {
                            continue;
                        }
                        bfs.push(v);
                    }
                }
                assert!(bfs.len() == n_verts, "Invalid tree structure");

                // Upward propagation
                let mut size = vec![1; n_verts];
                let mut heavy_child = vec![UNSET; n_verts];
                for &u in bfs[1..].iter().rev() {
                    let p = bct.bct_parent[u as usize];
                    size[p as usize] += size[u as usize];
                    let h = &mut heavy_child[p as usize];
                    if *h == UNSET || size[*h as usize] < size[u as usize] {
                        *h = u as u32;
                    }
                }

                // Downward propagation
                let mut chain_top = vec![UNSET; n_verts];
                for &u in &bfs {
                    if chain_top[u as usize] != UNSET {
                        continue;
                    }
                    let mut h = u;
                    loop {
                        chain_top[h as usize] = u;
                        h = heavy_child[h as usize];
                        if h == UNSET {
                            break;
                        }
                    }
                }

                let mut left_light_children = vec![vec![]; n_verts];
                let mut right_light_children = vec![vec![]; n_verts];
                for u in 0..n_verts {
                    if bct.bct_children[u].is_empty() {
                        continue;
                    }

                    let mut iv = 0;
                    while iv < bct.bct_children[u].len() {
                        let v = bct.bct_children[u][iv];
                        if v == heavy_child[u] {
                            break;
                        }
                        left_light_children[u].push(v);
                        iv += 1;
                    }
                    iv += 1;

                    while iv < bct.bct_children[u].len() {
                        let v = bct.bct_children[u][iv];
                        right_light_children[u].push(v);
                        iv += 1;
                    }
                }

                // Downward propagation

                bfs.reverse();
                Self {
                    size,
                    topological_order: bfs,

                    heavy_child,
                    chain_top,

                    left_light_children,
                    right_light_children,
                }
            }
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct NodeRef(NonZeroU32);

        impl NodeRef {
            fn new(idx: u32) -> Self {
                Self(NonZeroU32::new(idx).unwrap())
            }

            pub fn usize(&self) -> usize {
                self.0.get() as usize
            }

            fn get_with_children_in<'a, T>(
                &self,
                children: &'a [[Option<NodeRef>; 2]],
                xs: &'a mut [T],
            ) -> (&'a mut T, [Option<&'a mut T>; 2]) {
                let children = &children[self.usize()];
                let ptr = xs.as_mut_ptr();

                unsafe {
                    (
                        &mut *ptr.add(self.usize()),
                        children.map(|c| c.map(|c| &mut *ptr.add(c.usize()))),
                    )
                }
            }
        }

        pub struct StaticTopTree<Cx: ClusterCx> {
            // Represented tree structure
            pub hld: HLD,
            n_verts: usize,

            // Internal tree struture
            pub n_nodes: usize,
            root_node: NodeRef,
            size: Vec<u32>,
            pub children: Vec<[Option<NodeRef>; 2]>,
            parent: Vec<Option<NodeRef>>,

            pub compress_leaf: Vec<NodeRef>, // Leaf node in compress tree (true leaf, or a collapsed rake tree)
            compress_root: Vec<NodeRef>,     // Root node in compress tree

            // Weights and aggregates
            pub cx: Cx,
            pub weights: Vec<Cx::V>,
            pub clusters: Vec<Cluster<Cx>>,
        }

        impl<Cx: ClusterCx> StaticTopTree<Cx> {
            pub fn from_bct<'a, J: Jagged<u32>>(
                bct: &bcc::BlockCutTree<'a, J>,
                root: usize,
                cx: Cx,
            ) -> Self
            where
                J: std::fmt::Debug,
            {
                let n_verts = bct.bcc_node_range().end;
                let hld = HLD::from_bct(n_verts, bct, root);
                crate::debug::with(|| println!("bct: {bct:?}"));
                crate::debug::with(|| println!("hld: {hld:?}"));
                let dangling = NodeRef::new(!0);
                let nodes_cap = n_verts * 4 + 1;
                let mut this = Self {
                    hld: Default::default(),
                    n_verts,

                    n_nodes: 1,
                    root_node: dangling,
                    size: vec![1; nodes_cap],
                    children: vec![[None; 2]; nodes_cap],
                    parent: vec![None; nodes_cap],

                    compress_leaf: vec![dangling; nodes_cap],
                    compress_root: vec![dangling; nodes_cap],

                    cx,
                    weights: vec![Default::default(); nodes_cap],
                    clusters: vec![Cluster::Compress(Cx::id_compress()); nodes_cap],
                };

                this.build_topology(&hld);
                this.hld = hld;

                this
            }

            // Build the internal tree

            fn alloc(&mut self, children: [Option<NodeRef>; 2]) -> NodeRef {
                let u = NodeRef::new(self.n_nodes as u32);
                self.children[u.usize()] = children;
                for &child in children.iter().flatten() {
                    self.parent[child.usize()] = Some(u);
                    self.size[u.usize()] += self.size[child.usize()];
                }
                self.n_nodes += 1;
                u
            }

            fn build_topology(&mut self, hld: &HLD) {
                for &u in &hld.topological_order {
                    // Build a rake tree
                    let rake_trees: [_; 2] = [
                        &hld.left_light_children[u as usize],
                        &hld.right_light_children[u as usize],
                    ]
                    .map(|cs| {
                        let light_edges: Vec<_> = cs
                            .iter()
                            .map(|&r| self.alloc([Some(self.compress_root[r as usize]), None]))
                            .collect();
                        (!light_edges.is_empty()).then(|| {
                            self.fold_balanced_rec(&light_edges, || Cluster::Rake(Cx::id_rake()))
                        })
                    });
                    self.compress_leaf[u as usize] = self.alloc(rake_trees);

                    if hld.chain_top[u as usize] == u {
                        // Build a compress tree
                        let mut h = u as usize;
                        let mut chain = vec![];
                        loop {
                            chain.push(self.compress_leaf[h]);
                            h = hld.heavy_child[h] as usize;
                            if h == UNSET as usize {
                                break;
                            }
                        }
                        self.compress_root[u as usize] =
                            self.fold_balanced_rec(&chain, || Cluster::Compress(Cx::id_compress()));
                    }
                }
                self.root_node =
                    self.compress_root[*hld.topological_order.last().unwrap() as usize];
            }

            fn fold_balanced_rec(
                &mut self,
                nodes: &[NodeRef],
                id_cluster: impl Fn() -> Cluster<Cx> + Copy,
            ) -> NodeRef {
                debug_assert!(!nodes.is_empty());
                if nodes.len() == 1 {
                    self.clusters[nodes[0].usize()] = id_cluster();
                    return nodes[0];
                }

                // Make the tree balanced in the global sense, by split at the middle size.
                // TODO: If the split point is not exact, make the tree left-skewed.
                let mut total_size = nodes.iter().map(|u| self.size[u.usize()]).sum::<u32>() as i32;
                let i = nodes
                    .iter()
                    .rposition(|u| {
                        total_size -= self.size[u.usize()] as i32 * 2;
                        total_size <= 0
                    })
                    .unwrap()
                    .max(1);

                let (lhs, rhs) = nodes.split_at(i);
                let lhs = self.fold_balanced_rec(lhs, id_cluster);
                let rhs = self.fold_balanced_rec(rhs, id_cluster);
                let node = self.alloc([Some(lhs), Some(rhs)]);
                self.clusters[node.usize()] = id_cluster();
                node
            }

            pub fn init_weights(&mut self, weights: impl IntoIterator<Item = (usize, Cx::V)>) {
                for (u, w) in weights {
                    debug_assert!(u < self.n_verts);
                    self.weights[self.compress_leaf[u].usize()] = w;
                }

                for u in (1..self.n_nodes as u32).map(NodeRef::new) {
                    self.pull_up(u);
                }
            }

            fn pull_up(&mut self, u: NodeRef) {
                let (node, children) = u.get_with_children_in(&self.children, &mut self.clusters);
                self.cx.pull_up(node, children, &self.weights[u.usize()]);
            }

            fn pull_up_to_root(&mut self, mut u: NodeRef) {
                self.pull_up(u);
                while let Some(p) = self.parent[u.usize()] {
                    u = p;
                    self.pull_up(u);
                }
            }

            // Point update
            pub fn modify<T>(&mut self, u: usize, update_with: impl FnOnce(&mut Cx::V) -> T) -> T {
                let u = self.compress_leaf[u];
                let res = update_with(&mut self.weights[u.usize()]);
                self.pull_up_to_root(u);
                res
            }

            pub fn sum_all(&mut self) -> &Cx::C {
                unsafe {
                    self.clusters[self.root_node.usize()]
                        .get_compress()
                        .unwrap_unchecked()
                }
            }
        }
    }
}

type X = u32;
const INF: X = 1e9 as X + 10;

#[derive(Debug, Clone, Copy)]
pub struct MinPlus3x3([[X; 3]; 3]);

impl MinPlus3x3 {
    #[inline(always)]
    fn from_fn(f: impl Fn(usize, usize) -> X) -> Self {
        Self(std::array::from_fn(|i| std::array::from_fn(|j| f(i, j))))
    }

    #[inline(always)]
    fn get(&self, i: usize, j: usize) -> X {
        self.0[i][j]
    }

    fn transpose(&self) -> Self {
        Self::from_fn(|i, j| self.get(j, i))
    }

    const fn diag(p: [X; 3]) -> Self {
        Self([[p[0], INF, INF], [INF, p[1], INF], [INF, INF, p[2]]])
    }

    const fn id() -> Self {
        Self::diag([0, 0, 0])
    }

    const fn sep() -> Self {
        Self([[INF, 0, 0], [0, INF, 0], [0, 0, INF]])
    }

    fn mul_vec(lhs: [X; 3], rhs: [X; 3]) -> [X; 3] {
        std::array::from_fn(|i| (lhs[i] + rhs[i]).min(INF))
    }

    fn mul_parallel(&self, other: &Self) -> Self {
        Self::from_fn(|i, j| (self.get(i, j) + other.get(i, j)).min(INF))
    }

    fn enclose(&self) -> X {
        *self
            .0
            .map(|row| *row.iter().min().unwrap())
            .iter()
            .min()
            .unwrap()
    }

    fn enclose_right(&self) -> [X; 3] {
        std::array::from_fn(|i| (0..3).map(|j| self.get(i, j)).min().unwrap())
    }

    fn flip_ends(&self) -> MinPlus3x3 {
        Self::sep() * *self * Self::sep()
    }
}

impl std::ops::Add for MinPlus3x3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(std::array::from_fn(|i| self.0[i].min(rhs.0[i])))
    }
}

impl std::ops::Mul for MinPlus3x3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_fn(|i, j| {
            (0..3)
                .map(|k| self.get(i, k) + rhs.get(k, j))
                .min()
                .unwrap()
                .min(INF)
        })
    }
}

struct CactusSum;

#[derive(Clone, Debug)]
struct Compress {
    is_left_bcc: bool,
    value: MinPlus3x3,
}

#[derive(Clone, Debug)]
enum Rake {
    BCC([X; 3]),
    Vertex(MinPlus3x3),
}

impl Rake {
    unsafe fn bcc_unchecked(&self) -> &[X; 3] {
        match self {
            Rake::BCC(x) => x,
            Rake::Vertex(_) => unreachable_unchecked(),
        }
    }

    unsafe fn vertex_unchecked(&self) -> &MinPlus3x3 {
        match self {
            Rake::BCC(_) => unreachable_unchecked(),
            Rake::Vertex(m) => m,
        }
    }
}

impl static_top_tree::rooted::ClusterCx for CactusSum {
    type V = [X; 3];

    type C = Compress;
    type R = Rake;

    fn id_compress() -> Self::C {
        // Dummy value
        Compress {
            is_left_bcc: false,
            value: MinPlus3x3::id(),
        }
    }

    fn compress(&self, lhs: &Self::C, rhs: &Self::C) -> Self::C {
        Compress {
            is_left_bcc: lhs.is_left_bcc,
            value: lhs.value * rhs.value,
        }
    }

    fn id_rake() -> Self::R {
        // Dummy value
        Rake::BCC([0; 3])
    }

    fn rake(&self, lhs: &Self::R, rhs: &Self::R) -> Self::R {
        match (lhs, rhs) {
            (Rake::BCC(lhs), Rake::BCC(rhs)) => Rake::BCC(MinPlus3x3::mul_vec(*lhs, *rhs)),
            (Rake::Vertex(lhs), Rake::Vertex(rhs)) => Rake::Vertex(*lhs * MinPlus3x3::sep() * *rhs),
            _ => unsafe { unreachable_unchecked() },
        }
    }

    fn collapse_compressed(&self, c: &Self::C) -> Self::R {
        let v = c.value.enclose_right();
        if c.is_left_bcc {
            Rake::BCC(v)
        } else {
            Rake::Vertex(MinPlus3x3::diag(v))
        }
    }

    fn make_leaf(&self, rl: Option<&Self::R>, v: &Self::V, rr: Option<&Self::R>) -> Self::C {
        if v[0] == INF {
            let rl = rl
                .map(|r| unsafe { r.vertex_unchecked().flip_ends() })
                .unwrap_or(MinPlus3x3::sep());
            let rr = rr
                .map(|r| unsafe { r.vertex_unchecked().transpose().flip_ends() })
                .unwrap_or(MinPlus3x3::sep());
            let m = rl.mul_parallel(&rr);

            Compress {
                is_left_bcc: true,
                value: m,
            }
        } else {
            let mut v = *v;
            rl.map(|r| v = unsafe { MinPlus3x3::mul_vec(v, *r.bcc_unchecked()) });
            rr.map(|r| v = unsafe { MinPlus3x3::mul_vec(v, *r.bcc_unchecked()) });
            Compress {
                is_left_bcc: false,
                value: MinPlus3x3::diag(v),
            }
        }
    }
}

pub fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let q: usize = input.value();

    let weights: Vec<_> = (0..n)
        .map(|_| [input.u32(), input.u32(), input.u32()])
        .collect();

    let mut edges = vec![];
    for _ in 0..m {
        let u = input.u32() - 1;
        let v = input.u32() - 1;
        edges.push((u, (v, ())));
        edges.push((v, (u, ())));
    }
    let neighbors = jagged::Unweighted(CSR::from_edges(n, &edges));

    let root = 0;
    let bct = bcc::BlockCutTree::from_neighbors(&neighbors, root);
    let n_verts = bct.bcc_node_range().end;
    let mut stt = static_top_tree::rooted::StaticTopTree::from_bct(&bct, root, CactusSum);

    let weights_bct = weights
        .into_iter()
        .chain(std::iter::repeat([INF; 3]))
        .take(n_verts);
    stt.init_weights(weights_bct.enumerate());

    for _ in 0..q {
        let a = input.u32() as usize - 1;
        let b = input.u32() as usize - 1;
        let c = input.u32() as X;
        stt.modify(a, |v| v[b] = c);

        let ans = stt.sum_all().value.enclose();
        writeln!(output, "{ans}").unwrap();
    }
}
