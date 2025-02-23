pub mod static_top_tree {
    pub mod rooted {
        /// # Static Top Tree
        /// Extend segment tree to rooted trees in O(N log N).
        /// Compared to usual edge-based top trees, this one is vertex-based and not path-reversible;
        /// Each compress cluster is represented by a left-open, right-closed path.
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
        /// - Lazy propagation + path query + subtree query
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
            pub segmented_idx: Vec<u32>,

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
                let mut segmented_idx = vec![UNSET; n_verts];
                let mut timer = 0;
                for u in topological_order.iter().copied().rev() {
                    if segmented_idx[u as usize] != UNSET {
                        continue;
                    }
                    let mut h = u;
                    loop {
                        chain_top[h as usize] = u;
                        segmented_idx[h as usize] = timer;
                        timer += 1;
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
                    segmented_idx,

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
            fn make_edge(&self, weight: &Self::V) -> Self::Compress; // In case of no associated rake edge

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
                    (Compress(c), [None, None]) => *c = self.make_edge(weight),
                    (Rake(r), [Some(Rake(lhs)), Some(Rake(rhs))]) => *r = self.rake(lhs, rhs),
                    (Rake(r), [Some(Compress(top)), None]) => *r = self.collapse_compressed(top),
                    _ => unsafe { std::hint::unreachable_unchecked() },
                }
            }

            // Lazy propagation (Implement it yourself)
            const LAZY: bool;
            fn push_down(
                &self,
                _node: &mut Cluster<Self>,
                _children: [Option<&mut Cluster<Self>>; 2],
                _weight: &Self::V,
            ) {
                assert!(!Self::LAZY, "Implement push_down for lazy propagation");
            }
        }

        // Lazy propagation (Implement it yourself, Part II)
        pub trait MonoidAction<Cx: ClusterCx> {
            fn try_apply_to_compressed(&self, chain: &mut Cx::Compress) -> bool;
            fn try_apply_to_raked(&self, point: &mut Cx::Rake) -> bool;
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

        impl<C: ClusterCx> Clone for Cluster<C> {
            fn clone(&self) -> Self {
                match self {
                    Cluster::Compress(c) => Cluster::Compress(c.clone()),
                    Cluster::Rake(r) => Cluster::Rake(r.clone()),
                }
            }
        }

        impl<C: ClusterCx> Cluster<C> {
            pub fn get_compress(&self) -> Option<&C::Compress> {
                match self {
                    Cluster::Compress(c) => Some(c),
                    _ => None,
                }
            }

            pub fn get_rake(&self) -> &C::Rake {
                match self {
                    Cluster::Rake(r) => r,
                    _ => unsafe { std::hint::unreachable_unchecked() },
                }
            }
        }
        #[derive(Debug, Clone, Copy)]
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

        pub struct StaticTopTree<C: ClusterCx> {
            pub cx: C,
            pub hld: HLD,
            n_verts: usize,

            root_node: NodeRef,
            size: Vec<u32>,
            children: Vec<[Option<NodeRef>; 2]>,
            parent: Vec<Option<NodeRef>>,
            n_nodes: usize,

            compress_leaf: Vec<NodeRef>, // Leaf node in compress tree (true leaf, or a collapsed rake tree)
            compress_root: Vec<NodeRef>, // Root node in compress tree

            clusters: Vec<Cluster<C>>,
            pub weights: Vec<C::V>,
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
                };

                this.build_topology(&hld);
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

            fn push_down(&mut self, u: NodeRef) {
                if !Cx::LAZY {
                    return;
                }
                unimplemented!();
            }

            fn pull_up(&mut self, u: NodeRef) {
                let (node, children) = u.get_with_children_in(&self.children, &mut self.clusters);
                self.cx.pull_up(node, children, &self.weights[u.usize()]);
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

            pub fn sum_all(&mut self) -> &Cx::Compress {
                if Cx::LAZY {
                    self.push_down(self.root_node);
                }

                unsafe {
                    self.clusters[self.root_node.usize()]
                        .get_compress()
                        .unwrap_unchecked()
                }
            }

            pub fn sum_path(&mut self, u: usize, v: usize) -> (Cx::Compress, Cx::V, Cx::Compress) {
                unimplemented!()
            }

            pub fn sum_subtree(&mut self, u: usize) -> (Cx::Rake, Cx::V) {
                unimplemented!()
            }

            pub fn modify(&mut self, u: usize, update_with: impl FnOnce(&mut Cx::V)) {
                assert!(!Cx::LAZY, "Do not mix point updates with lazy propagation");

                let mut h = self.compress_leaf[u];
                update_with(&mut self.weights[h.usize()]);
                loop {
                    self.pull_up(h);
                    let Some(p) = self.parent[h.usize()] else {
                        break;
                    };
                    h = p;
                }
            }

            pub fn apply_all(&mut self, action: impl MonoidAction<Cx>) {
                assert!(Cx::LAZY, "Lazy propagation is not enabled");
                unimplemented!()
            }

            pub fn apply_path(&mut self, u: usize, v: usize, action: impl MonoidAction<Cx>) {
                assert!(Cx::LAZY, "Lazy propagation is not enabled");
                unimplemented!()
            }

            pub fn apply_subtree(&mut self, u: usize, action: impl MonoidAction<Cx>) {
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

type E = i64;

#[derive(Debug)]
struct DiameterOp;

#[derive(Debug, Clone)]
struct Compress {
    diameter: E,
    left: E,
    right: E,
    len: E,
}

#[derive(Debug, Clone)]
struct Rake {
    diameter: E,
    left: E,
}

impl ClusterCx for DiameterOp {
    type V = E;
    type Compress = Compress;
    type Rake = Rake;

    const LAZY: bool = false;

    fn id_compress() -> Self::Compress {
        Compress {
            diameter: 0,
            left: 0,
            right: 0,
            len: 0,
        }
    }

    fn compress(&self, lhs: &Self::Compress, rhs: &Self::Compress) -> Self::Compress {
        Compress {
            diameter: lhs.diameter.max(rhs.diameter).max(lhs.right + rhs.left),
            left: lhs.left.max(lhs.len + rhs.left),
            right: rhs.right.max(rhs.len + lhs.right),
            len: lhs.len + rhs.len,
        }
    }

    fn id_rake() -> Self::Rake {
        Rake {
            diameter: 0,
            left: 0,
        }
    }

    fn rake(&self, lhs: &Self::Rake, rhs: &Self::Rake) -> Self::Rake {
        Rake {
            diameter: (lhs.diameter).max(rhs.diameter).max(lhs.left + rhs.left),
            left: lhs.left.max(rhs.left),
        }
    }

    fn collapse_compressed(&self, path: &Self::Compress) -> Self::Rake {
        Rake {
            diameter: path.diameter,
            left: path.left,
        }
    }

    fn collapse_raked(&self, point: &Self::Rake, &weight: &Self::V) -> Self::Compress {
        Compress {
            diameter: point.diameter.max(weight + point.left),
            left: weight + point.left,
            right: weight.max(point.left),
            len: weight,
        }
    }

    fn make_edge(&self, &weight: &Self::V) -> Self::Compress {
        Compress {
            diameter: weight,
            left: weight,
            right: weight,
            len: weight,
        }
    }
}
