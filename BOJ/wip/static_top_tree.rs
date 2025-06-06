use std::io::Write;

use static_top_tree::rooted::{Cluster, ClusterCx, MonoidAction, StaticTopTree};

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

pub mod debug {
    pub fn with(f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
    }
}

pub mod static_top_tree {
    pub mod rooted {
        /// # Static Top Tree
        /// Extend segment tree to rooted trees in O(N log N).
        /// Compared to usual edge-based top trees, this one is vertex-based and not path-reversible;
        /// Each compress cluster is represents a left-open, right-closed path.
        ///
        /// Since we cannot expose the path cluster directly, implementing path and subtree queries
        /// are a bit tricky.
        ///
        /// ## Reference:
        /// - [[Tutorial] Theorically Faster HLD and Centroid Decomposition](https://codeforces.com/blog/entry/104997/)
        /// - [ABC 351G Editorial](https://atcoder.jp/contests/abc351/editorial/9899)
        /// - [Self-adjusting top tree](https://renatowerneck.wordpress.com/wp-content/uploads/2016/06/tw05-self-adjusting-top-tree.pdf)
        ///
        /// See also:
        /// - [maomao90's static top tree visualisation](https://maomao9-0.github.io/static-top-tree-visualisation/)
        ///
        /// ## TODO
        /// - path query
        /// - Lazy propagation + path query + subtree query
        ///   (Probably, implmenting a dynamic top tree would be easier)
        /// - Persistence!
        use std::num::NonZeroU32;

        pub const UNSET: u32 = !0;

        // Heavy-Light Decomposition, prior to top tree construction.
        // Golfable within ~20 lines with two recursive DFS traversals.
        #[derive(Debug, Default)]
        pub struct HLD {
            // Rooted tree structure
            pub size: Vec<u32>,
            pub parent: Vec<u32>,
            pub topological_order: Vec<u32>,

            // Chain structure
            pub heavy_child: Vec<u32>,
            pub chain_top: Vec<u32>,

            // Light edges, in linked list
            pub first_light_child: Vec<u32>,
            pub xor_light_siblings: Vec<u32>,
        }

        impl HLD {
            pub fn len(&self) -> usize {
                self.parent.len()
            }

            pub fn from_edges<'a>(
                n_verts: usize,
                edges: impl IntoIterator<Item = (u32, u32)>,
                root: usize,
            ) -> Self {
                assert!(n_verts >= 1);
                let mut degree = vec![0u32; n_verts];
                let mut xor_neighbors: Vec<u32> = vec![0u32; n_verts];
                for (u, v) in edges {
                    debug_assert!(u != v);
                    degree[u as usize] += 1;
                    degree[v as usize] += 1;
                    xor_neighbors[u as usize] ^= v;
                    xor_neighbors[v as usize] ^= u;
                }

                // Upward propagation
                let mut size = vec![1; n_verts];
                let mut heavy_child = vec![UNSET; n_verts];
                degree[root] += 2;
                let mut topological_order = Vec::with_capacity(n_verts);
                for mut u in 0..n_verts {
                    while degree[u] == 1 {
                        let p = xor_neighbors[u];
                        topological_order.push(u as u32);
                        degree[u] = 0;
                        degree[p as usize] -= 1;
                        xor_neighbors[p as usize] ^= u as u32;

                        size[p as usize] += size[u as usize];
                        let h = &mut heavy_child[p as usize];
                        if *h == UNSET || size[*h as usize] < size[u as usize] {
                            *h = u as u32;
                        }

                        u = p as usize;
                    }
                }
                topological_order.push(root as u32);
                assert!(topological_order.len() == n_verts, "Invalid tree structure");
                let mut parent = xor_neighbors;
                parent[root] = UNSET;

                let mut first_light_child = vec![UNSET; n_verts];
                let mut xor_light_siblings = vec![UNSET; n_verts];
                for &u in &topological_order[..n_verts - 1] {
                    let p = parent[u as usize];
                    if u == heavy_child[p as usize] {
                        continue;
                    }

                    let c = first_light_child[p as usize];
                    xor_light_siblings[u as usize] = c ^ UNSET;
                    if c != UNSET {
                        xor_light_siblings[c as usize] ^= u as u32 ^ UNSET;
                    }
                    first_light_child[p as usize] = u;
                }

                // Downward propagation
                let mut chain_top = vec![UNSET; n_verts];
                for u in topological_order.iter().copied().rev() {
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

                Self {
                    size,
                    parent,
                    topological_order,

                    heavy_child,
                    chain_top,

                    first_light_child,
                    xor_light_siblings,
                }
            }
        }

        pub trait ClusterCx: Sized {
            // Vertex weight / weight of an upward edge (u -> parent(u)).
            type V: Default + Clone;

            type Compress: Clone; // Path cluster (aggregate on a subchain)
            type Rake: Clone; // Point cluster (Aggregate of light edges)

            // Compress monoid. Left side is always the top side.
            fn id_compress() -> Self::Compress;
            fn compress(&self, lhs: &Self::Compress, rhs: &Self::Compress) -> Self::Compress;

            // Rake monoid, commutative.
            fn id_rake() -> Self::Rake;
            fn rake(&self, lhs: &Self::Rake, rhs: &Self::Rake) -> Self::Rake;

            // A projection.
            fn collapse_compressed(&self, path: &Self::Compress) -> Self::Rake;
            // Attach a rake cluster to a leaf compress cluster.
            fn collapse_raked(&self, point: &Self::Rake, top_weight: &Self::V) -> Self::Compress;
            // Make a leaf compress cluster without any rake edge.
            fn make_leaf(&self, weight: &Self::V) -> Self::Compress; // In case of no associated rake edge

            // This is how everything is summed up.
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
                    (Compress(c), [Some(Rake(top)), None]) => *c = self.collapse_raked(top, weight),
                    (Compress(c), [None, None]) => *c = self.make_leaf(weight),
                    (Rake(r), [Some(Rake(lhs)), Some(Rake(rhs))]) => *r = self.rake(lhs, rhs),
                    (Rake(r), [Some(Compress(top)), None]) => *r = self.collapse_compressed(top),
                    _ => unsafe { std::hint::unreachable_unchecked() },
                }
            }

            // Lazy propagation (Implement it yourself)
            const LAZY: bool;
            fn push_down(
                &self,
                node: &mut Cluster<Self>,
                children: [Option<&mut Cluster<Self>>; 2],
                weight: &mut Self::V,
            ) {
                assert!(Self::LAZY, "Implement push_down for lazy propagation");
                use Cluster::*;

                #[allow(unused_variables)]
                match (node, children) {
                    (Compress(c), [Some(Compress(lhs)), Some(Compress(rhs))]) => todo!(),
                    (Compress(c), [Some(Rake(top)), None]) => todo!(),
                    (Compress(c), [None, None]) => todo!(),
                    (Rake(r), [Some(Rake(lhs)), Some(Rake(rhs))]) => todo!(),
                    (Rake(r), [Some(Compress(top)), None]) => todo!(),
                    _ => unsafe { std::hint::unreachable_unchecked() },
                }
            }
        }

        // Lazy propagation (Implement it yourself, Part II)
        pub trait MonoidAction<Cx: ClusterCx> {
            fn apply(&self, cluster: &mut Cluster<Cx>);
            fn apply_to_weight(&self, weight: &mut Cx::V);
        }

        #[derive(Debug)]
        pub enum Cluster<C: ClusterCx> {
            Compress(C::Compress),
            Rake(C::Rake),
        }

        #[derive(Debug, Clone, Copy)]
        pub enum ClusterType {
            Compress,
            Rake,
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
            pub fn into_result(self) -> Result<Cx::Compress, Cx::Rake> {
                match self {
                    Cluster::Compress(c) => Ok(c),
                    Cluster::Rake(r) => Err(r),
                }
            }

            pub fn get_compress(&self) -> Option<&Cx::Compress> {
                match self {
                    Cluster::Compress(c) => Some(c),
                    _ => None,
                }
            }

            pub fn get_rake(&self) -> Option<&Cx::Rake> {
                match self {
                    Cluster::Rake(r) => Some(r),
                    _ => None,
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
            pub cx: Cx,
            pub hld: HLD,
            n_verts: usize,

            root_node: NodeRef,
            size: Vec<u32>,
            children: Vec<[Option<NodeRef>; 2]>,
            parent: Vec<Option<NodeRef>>,
            n_nodes: usize,

            compress_leaf: Vec<NodeRef>, // Leaf node in compress tree (true leaf, or a collapsed rake tree)
            compress_root: Vec<NodeRef>, // Root node in compress tree

            clusters: Vec<Cluster<Cx>>,
            weights: Vec<Cx::V>,

            // Maps node indices to their positions in the binary completion of the top tree.
            // This is required for fast node locations, path queries and lazy propagations.
            index_in_binary_completion: Vec<u64>,
        }

        impl<Cx: ClusterCx> StaticTopTree<Cx> {
            pub fn from_edges(
                n_verts: usize,
                edges: impl IntoIterator<Item = (u32, u32)>,
                root: usize,
                cx: Cx,
            ) -> Self {
                let hld = HLD::from_edges(n_verts, edges, root);
                let dummy = NodeRef::new(!0);
                let nodes_cap = n_verts * 4 + 1;
                let mut this = Self {
                    hld: Default::default(),
                    n_verts,

                    root_node: dummy,
                    size: vec![1; nodes_cap],
                    children: vec![[None; 2]; nodes_cap],
                    parent: vec![None; nodes_cap],
                    n_nodes: 1,

                    compress_leaf: vec![dummy; nodes_cap],
                    compress_root: vec![dummy; nodes_cap],

                    clusters: vec![Cluster::Compress(Cx::id_compress()); nodes_cap],
                    weights: vec![Default::default(); nodes_cap],

                    cx,

                    index_in_binary_completion: vec![0; nodes_cap],
                };

                this.build_topology(&hld);
                this.build_locators();
                this.hld = hld;

                this
            }

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
                    let mut light_edges = vec![];
                    let mut l = hld.first_light_child[u as usize];
                    let mut prev = UNSET;
                    while l != UNSET {
                        // Collapse a compress tree
                        light_edges.push(self.alloc([Some(self.compress_root[l as usize]), None]));

                        let next = hld.xor_light_siblings[l as usize] ^ prev;
                        prev = l;
                        l = next;
                    }

                    self.compress_leaf[u as usize] = if light_edges.is_empty() {
                        // Make a leaf cluster
                        self.alloc([None, None])
                    } else {
                        // Collapse a rake tree
                        let rake_root =
                            self.fold_balanced_rec(&light_edges, || Cluster::Rake(Cx::id_rake()));
                        self.alloc([Some(rake_root), None])
                    };

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

            // Make the tree balanced in the global sense.
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

                // Split at the middle. If the split point is not exact, make the tree left-skewed.
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

            fn build_locators(&mut self) {
                self.index_in_binary_completion[self.root_node.usize()] = 1;
                for u in (1..self.n_nodes as u32).rev().map(NodeRef::new) {
                    let i = self.index_in_binary_completion[u.usize()];
                    for branch in 0..2 {
                        if let Some(c) = self.children[u.usize()][branch as usize] {
                            self.index_in_binary_completion[c.usize()] = i << 1 | branch;
                        }
                    }
                }
            }

            fn depth(&self, u: NodeRef) -> u32 {
                let path = self.index_in_binary_completion[u.usize()];
                u64::BITS - 1 - u64::leading_zeros(path)
            }

            pub fn init_weights(&mut self, weights: &[Cx::V]) {
                assert_eq!(weights.len(), self.n_verts);
                for u in 0..weights.len() {
                    self.weights[self.compress_leaf[u].usize()] = weights[u].clone();
                }

                for u in (1..self.n_nodes as u32).map(NodeRef::new) {
                    self.pull_up(u);
                }
            }

            // A bunch of propagation helpers

            fn push_down(&mut self, u: NodeRef) {
                if !Cx::LAZY {
                    return;
                }

                let (node, children) = u.get_with_children_in(&self.children, &mut self.clusters);
                self.cx
                    .push_down(node, children, &mut self.weights[u.usize()]);
            }

            fn pull_up(&mut self, u: NodeRef) {
                let (node, children) = u.get_with_children_in(&self.children, &mut self.clusters);
                self.cx.pull_up(node, children, &self.weights[u.usize()]);
            }

            fn push_down_from_root(&mut self, u: NodeRef) {
                if !Cx::LAZY {
                    return;
                }

                let mut v = self.root_node;
                self.push_down(v);
                let path = self.index_in_binary_completion[u.usize()];
                for d in (0..self.depth(u)).rev() {
                    let branch = (path >> d) & 1;
                    v = unsafe { self.children[v.usize()][branch as usize].unwrap_unchecked() };
                    self.push_down(v);
                }

                assert_eq!(v, u); // TODO
            }

            fn pull_up_to_root(&mut self, mut u: NodeRef) {
                self.pull_up(u);
                while let Some(p) = self.parent[u.usize()] {
                    u = p;
                    self.pull_up(u);
                }
            }

            // Sum queries

            pub fn get(&mut self, u: usize) -> &Cx::V {
                self.push_down_from_root(self.compress_leaf[u]);
                &self.weights[self.compress_leaf[u].usize()]
            }

            pub fn sum_all(&mut self) -> &Cx::Compress {
                unsafe {
                    self.clusters[self.root_node.usize()]
                        .get_compress()
                        .unwrap_unchecked()
                }
            }

            // Sum on the proper subtree
            pub fn sum_subtree(&mut self, u: usize) -> (Cx::Rake, &Cx::V) {
                self.push_down_from_root(self.compress_leaf[u]);

                let (u, top) = (
                    self.compress_leaf[u],
                    self.compress_root[self.hld.chain_top[u] as usize],
                );
                let mut v = u;
                let mut suffix = Cx::id_compress();
                while v != top {
                    let p = self.parent[v.usize()].unwrap();
                    let branch = (self.children[p.usize()][1] == Some(v)) as usize;
                    if branch == 0 {
                        let rhs = unsafe { self.children[p.usize()][1].unwrap_unchecked() };
                        let rhs =
                            unsafe { self.clusters[rhs.usize()].get_compress().unwrap_unchecked() };
                        suffix = self.cx.compress(&suffix, rhs);
                    }
                    v = p;
                }

                let mut sum_as_rake = self.cx.collapse_compressed(&suffix);
                if let Some(lhs) = self.children[u.usize()][0]
                    .and_then(|lhs| self.clusters[lhs.usize()].get_rake())
                {
                    sum_as_rake = self.cx.rake(lhs, &sum_as_rake);
                }
                (sum_as_rake, &self.weights[u.usize()])
            }

            // Sum on the complement of the proper subtree (WIP)
            pub fn sum_subtree_complement(&mut self, u: usize) -> Cx::Compress {
                eprintln!("Warning: sum_subtree_complement not tested yet");
                self.push_down_from_root(self.compress_leaf[u]);

                let mut u = self.compress_leaf[u];
                let mut sum = Cluster::Compress(self.cx.make_leaf(&self.weights[u.usize()]));
                while let Some(p) = self.parent[u.usize()] {
                    let mut prev = std::mem::replace(&mut sum, self.clusters[p.usize()].clone());

                    let (_, mut modified_children) =
                        p.get_with_children_in(&self.children, &mut self.clusters);
                    let branch = (self.children[p.usize()][1] == Some(u)) as usize;
                    modified_children[branch] = Some(&mut prev);
                    self.cx
                        .pull_up(&mut sum, modified_children, &self.weights[p.usize()]);
                    u = p;
                }
                unsafe { sum.into_result().unwrap_unchecked() }
            }

            pub fn sum_to_root(&mut self, u: usize) -> Cx::Compress {
                self.push_down_from_root(self.compress_leaf[u]);

                unimplemented!()
            }

            pub fn sum_path(
                &mut self,
                mut u: usize,
                mut v: usize,
            ) -> (Cx::Compress, Cx::V, Cx::Compress) {
                // TODO: Push down from the LCA (for performance)
                self.push_down_from_root(self.compress_leaf[u]);
                self.push_down_from_root(self.compress_leaf[v]);

                // We cannot expose the path cluster directly... Should we collect the path to lca?
                // Also, we should determine which is on the left side, and which is on the right side.
                unimplemented!()
            }

            // Modification query

            pub fn modify(&mut self, u: usize, update_with: impl FnOnce(&mut Cx::V)) {
                assert!(!Cx::LAZY, "Do not mix point updates with lazy propagation");
                let u = self.compress_leaf[u];
                update_with(&mut self.weights[u.usize()]);
                self.pull_up_to_root(u);
            }

            pub fn apply_all(&mut self, action: impl MonoidAction<Cx>) {
                assert!(Cx::LAZY, "Lazy propagation is not enabled");
                unimplemented!()
            }

            pub fn apply_point(&mut self, u: usize, action: impl MonoidAction<Cx>) {
                self.push_down_from_root(u);

                self.pull_up_to_root(u);
            }

            pub fn apply_path(&mut self, u: usize, v: usize, action: impl MonoidAction<Cx>) {
                assert!(Cx::LAZY, "Lazy propagation is not enabled");
                unimplemented!()
            }

            pub fn apply_proper_subtree(&mut self, u: usize, action: impl MonoidAction<Cx>) {
                assert!(Cx::LAZY, "Lazy propagation is not enabled");

                let (u, top) = (
                    self.compress_leaf[u],
                    self.compress_root[self.hld.chain_top[u] as usize],
                );
                let mut v = u;
                let mut suffix = Cx::id_compress();
                // while v != top {
                //     let p = self.parent[v.usize()].unwrap();
                //     let branch = (self.children[p.usize()][1] == Some(v)) as usize;
                //     if branch == 0 {
                //         let rhs = unsafe { self.children[p.usize()][1].unwrap_unchecked() };
                //         let rhs =
                //             unsafe { self.clusters[rhs.usize()].get_compress().unwrap_unchecked() };
                //         suffix = self.cx.compress(&suffix, rhs);
                //     }
                //     v = p;
                // }

                // let mut sum_as_rake = self.cx.collapse_compressed(&suffix);
                // if let Some(lhs) = self.children[u.usize()][0]
                //     .and_then(|lhs| self.clusters[lhs.usize()].get_rake())
                // {
                //     sum_as_rake = self.cx.rake(lhs, &sum_as_rake);
                // }

                unimplemented!()
            }

            pub fn apply_subtree_complement(&mut self, u: usize, action: impl MonoidAction<Cx>) {
                assert!(Cx::LAZY, "Lazy propagation is not enabled");
                unimplemented!()
            }

            pub fn debug_chains(&self, mut visitor: impl FnMut(&Cx::Compress, bool)) {
                let mut visited = vec![false; self.n_verts];
                for mut u in self.hld.topological_order.iter().rev().copied() {
                    if visited[u as usize] {
                        continue;
                    }
                    unsafe {
                        visitor(
                            self.clusters[self.compress_root[u as usize].usize()]
                                .get_compress()
                                .unwrap_unchecked(),
                            true,
                        );

                        loop {
                            visited[u as usize] = true;
                            visitor(
                                self.clusters[self.compress_leaf[u as usize].usize()]
                                    .get_compress()
                                    .unwrap_unchecked(),
                                false,
                            );

                            u = self.hld.heavy_child[u as usize];
                            if u == UNSET {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

type X = u32;

#[derive(Copy, Clone, Default)]
struct LazyNode {
    sum: X,
    lazy_add_subtree: X,
}

#[derive(Debug)]
struct Additive;

impl ClusterCx for Additive {
    type V = X;
    type Compress = LazyNode;
    type Rake = LazyNode;

    const LAZY: bool = true;

    fn id_compress() -> Self::Compress {
        LazyNode::default()
    }

    fn make_leaf(&self, &weight: &Self::V) -> Self::Compress {
        LazyNode {
            sum: weight,
            lazy_add_subtree: 0,
        }
    }

    fn compress(&self, lhs: &Self::Compress, rhs: &Self::Compress) -> Self::Compress {
        LazyNode {
            sum: lhs.sum + rhs.sum,
            lazy_add_subtree: 0,
        }
    }

    fn collapse_compressed(&self, &path: &Self::Compress) -> Self::Rake {
        path
    }

    fn id_rake() -> Self::Rake {
        LazyNode::default()
    }

    fn rake(&self, lhs: &Self::Rake, rhs: &Self::Rake) -> Self::Rake {
        self.compress(lhs, rhs)
    }

    fn collapse_raked(&self, point: &Self::Rake, &weight: &Self::V) -> Self::Compress {
        LazyNode {
            sum: point.sum + weight,
            lazy_add_subtree: point.lazy_add_subtree,
        }
    }

    fn push_down(
        &self,
        node: &mut Cluster<Self>,
        children: [Option<&mut Cluster<Self>>; 2],
        weight: &mut Self::V,
    ) {
        use Cluster::*;
        let c = match node {
            Compress(c) | Rake(c) => c,
        };
        let action = std::mem::take(&mut c.lazy_add_subtree);
        for child in children.into_iter().flatten() {
            action.apply(child);
            action.apply_to_weight(weight);
        }
    }
}

impl MonoidAction<Additive> for X {
    fn apply(&self, cluster: &mut Cluster<Additive>) {
        match cluster {
            Cluster::Compress(c) | Cluster::Rake(c) => {
                c.sum += *self;
                c.lazy_add_subtree += *self;
            }
        }
    }

    fn apply_to_weight(&self, weight: &mut <Additive as ClusterCx>::V) {
        *weight += self;
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();

    let mut parent = vec![0u32; n];
    for u in 0..n {
        let p = input.value::<i32>() - 1;
        if u == 0 {
            continue;
        }
        parent[u] = p as u32;
    }

    let edges = (1..n).map(|u| (u as u32, parent[u]));
    let mut counter = StaticTopTree::from_edges(n, edges, 0, Additive);

    for _ in 0..m {
        match input.token() {
            "1" => {
                let u = input.value::<usize>() - 1;
                let w: X = input.value();
                counter.apply_proper_subtree(u, w);
            }
            "2" => {
                let u = input.value::<usize>() - 1;
                let ans = counter.get(u);
                writeln!(output, "{}", ans).unwrap();
            }
            _ => panic!(),
        }
    }
}
