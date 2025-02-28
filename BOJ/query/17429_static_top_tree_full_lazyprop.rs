use std::{hint::unreachable_unchecked, io::Write};

use static_top_tree::rooted::{Action, ActionRange, Cluster, ClusterCx, SplitProj, StaticTopTree};

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

pub mod static_top_tree {
    /// The full version is available at:
    /// [https://github.com/null-lambda/ps_rust/tree/main/library/src/tree/top_tree/]
    pub mod rooted {
        /// # Static Top Tree
        /// Extend dynamic Divide & Conquer (segment tree) to rooted trees with rebalanced HLD, in O(N log N).
        ///
        /// - Supports subtree and path queries with lazy propagation.
        ///   Pseudo-rerooting is also supported for path-reversible clusters.
        ///
        /// - All tree DP problems with the monoid property can be solved, even dynamically.  
        ///
        /// - Heavily optimized for performance, with high level of abstraction.
        ///   (Don't get scared of the large amount of code-It's just mostly boilerplates.
        ///   Portability & code golfing is not our concerns.)
        ///
        /// - Unlike traditional edge-based top trees, this one is vertex-based. Each compress
        ///   cluster represents a left-open, right-closed path.
        ///   Reversibility of both vertex and edge weights is not provided by default.
        ///   (To achieve complete reversibility, store endpoint weights in the compressed clusters.)
        ///
        /// ## Reference:
        /// - [[Tutorial] Theorically Faster HLD and Centroid Decomposition](https://codeforces.com/blog/entry/104997/)
        /// - [ABC 351G Editorial](https://atcoder.jp/contests/abc351/editorial/9899)
        /// - [Self-adjusting top trees](https://renatowerneck.wordpress.com/wp-content/uploads/2016/06/tw05-self-adjusting-top-tree.pdf)
        /// - [[Tutorial] Fully Dynamic Trees Supporting Path/Subtree Aggregates and Lazy Path/Subtree Updates](https://codeforces.com/blog/entry/103726)
        ///
        /// See also:
        /// - [maomao90's static top tree visualisation](https://maomao9-0.github.io/static-top-tree-visualisation/)
        ///
        /// ## TODO
        /// - Persistence!
        ///   (This should be straightforward: Convert SoA-based nodes to SoA,
        ///   and then replace every instance of `&mut` access with `Rc::make_mut`.)
        /// - Rerooting queries - complement of a subtree.
        /// - Refactor path queries.
        /// - Add more examples on  subtree & path queries and reducers,
        /// - Implement a **REAL, DYNAMIC** ones.
        use std::{hint::unreachable_unchecked, num::NonZeroU32};

        pub const UNSET: u32 = !0;

        #[derive(Debug)]
        pub enum Cluster<C: ClusterCx> {
            Compress(C::Compress),
            Rake(C::Rake),
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum WeightType {
            Vertex,
            UpwardEdge,
        }

        pub trait ClusterCx: Sized {
            /// Vertex weight / weight of an upward edge (u -> parent(u)).
            type V: Default + Clone;

            type Compress: Clone; // Path cluster (aggregate on a subchain)
            type Rake: Clone; // Point cluster (Aggregate of light edges)

            /// Compress monoid.
            /// Left side is always the top side.
            fn id_compress() -> Self::Compress;
            fn compress(&self, lhs: &Self::Compress, rhs: &Self::Compress) -> Self::Compress;

            /// Rake monoid, commutative.
            fn id_rake() -> Self::Rake;
            fn rake(&self, lhs: &Self::Rake, rhs: &Self::Rake) -> Self::Rake;

            /// A projection.
            fn collapse_compressed(&self, c: &Self::Compress) -> Self::Rake;
            /// Attach a rake cluster to a leaf compress cluster.
            fn collapse_raked(&self, r: &Self::Rake, weight: &Self::V) -> Self::Compress;
            /// Make a leaf compress cluster without any rake edge.
            fn make_leaf(&self, weight: &Self::V) -> Self::Compress; // In case of no associated rake edge

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
                    (Compress(c), [Some(Rake(top)), None]) => *c = self.collapse_raked(top, weight),
                    (Compress(c), [None, None]) => *c = self.make_leaf(weight),
                    (Rake(r), [Some(Rake(lhs)), Some(Rake(rhs))]) => *r = self.rake(lhs, rhs),
                    (Rake(r), [Some(Compress(top)), None]) => *r = self.collapse_compressed(top),
                    _ => unsafe { unreachable_unchecked() },
                }
            }

            /// Lazy propagation (Implement it yourself).
            /// Store lazy tags in your own rake/compress clusters.
            /// To support both subtree updates and path updates, we need multiple aggregates/lazy tags:
            /// one for the path and one for the proper subtree.
            fn push_down(
                &self,
                _node: &mut Cluster<Self>,
                _children: [Option<&mut Cluster<Self>>; 2],
                _weight: &mut Self::V,
            ) {
            }

            /// An optimization flag to indicate no-op lazy propagation.  
            /// (Theoretically, the compiler should be able to inline `push_down`, but Rustc often fails to do so.)  
            const NO_OP_LAZY: bool = false;

            /// Path-reverse ops.
            /// Required for rerooting operations.
            const REVERSE_TYPE: Option<WeightType> = None;
            fn reverse(&self, _c: &Self::Compress) -> Self::Compress {
                panic!("Implement reverse for rerooting operations");
            }
        }

        #[derive(Debug, Copy, Clone)]
        pub enum ActionRange {
            Subtree,
            Path,
        }

        /// Lazy propagation (Implement it yourself, Part II)
        pub trait Action<Cx: ClusterCx> {
            fn apply_to_compress(&mut self, compress: &mut Cx::Compress, range: ActionRange);
            fn apply_to_rake(&mut self, rake: &mut Cx::Rake);
            fn apply_to_weight(&mut self, weight: &mut Cx::V);

            fn apply(&mut self, cluster: &mut Cluster<Cx>, range: ActionRange) {
                match (cluster, range) {
                    (Cluster::Compress(c), _) => self.apply_to_compress(c, range),
                    (Cluster::Rake(r), ActionRange::Subtree) => self.apply_to_rake(r),
                    _ => {}
                }
            }
        }

        /// A morphism of clusters, with common weight type V. There are two main use cases:
        /// - Supporting both path and subtree sum queries.
        ///   Path sums do not propagate from rake trees, whereas subtree sums do.
        ///   Therefore, we need to store both path and subtree aggregates in your clusters,
        ///   and the projection helps reduce computation time efficiently for each sum query.
        /// - Nesting another data structure within nodes (e.g., sets, segment trees, ropes, ... ).
        ///   The user has control over querying a specific node before performing the summation.
        /// Some set of combinators are provided: identity, tuple and path-sum.
        ///
        /// TODO: modify most of the sum_ functions to accept an additional reducer.
        pub trait Reducer<Dom: ClusterCx> {
            type Co: ClusterCx<V = Dom::V>;
            fn co(&self) -> &Self::Co;
            fn map_compress(
                &self,
                c: &<Dom as ClusterCx>::Compress,
            ) -> <Self::Co as ClusterCx>::Compress;
            fn map_rake(&self, r: &<Dom as ClusterCx>::Rake) -> <Self::Co as ClusterCx>::Rake;
        }

        /// An identity.
        ///
        /// # Examples
        ///
        /// ```
        /// let cx = || { ... }
        /// let mut stt = StaticTopTree::from_edges(n, edges, root, cx());
        /// ...
        /// let total_sum = stt.sum_all(Id(cx()));
        /// ```
        pub struct Id<Cx>(pub Cx);
        impl<Cx: ClusterCx> Reducer<Cx> for Id<Cx> {
            type Co = Cx;
            fn co(&self) -> &Self::Co {
                &self.0
            }
            fn map_compress(&self, c: &<Cx as ClusterCx>::Compress) -> <Cx as ClusterCx>::Compress {
                c.clone()
            }
            fn map_rake(&self, r: &<Cx as ClusterCx>::Rake) -> <Cx as ClusterCx>::Rake {
                r.clone()
            }
        }

        /// A pure mapping.
        impl<Dom, Co, FC, FR> Reducer<Dom> for (Co, FC, FR)
        where
            Dom: ClusterCx,
            Co: ClusterCx<V = Dom::V>,
            FC: Fn(&Dom::Compress) -> Co::Compress,
            FR: Fn(&Dom::Rake) -> Co::Rake,
        {
            type Co = Co;
            fn co(&self) -> &Self::Co {
                &self.0
            }
            fn map_compress(&self, c: &Dom::Compress) -> Co::Compress {
                (self.1)(c)
            }
            fn map_rake(&self, r: &Dom::Rake) -> Co::Rake {
                (self.2)(r)
            }
        }

        /// Disable propagation along rake trees, for path queries.
        pub struct Path<Cx>(pub Cx);

        impl<Cx: ClusterCx> ClusterCx for Path<Cx> {
            type V = Cx::V;
            type Compress = Cx::Compress;
            type Rake = ();
            fn id_compress() -> Self::Compress {
                Cx::id_compress()
            }
            fn compress(&self, lhs: &Self::Compress, rhs: &Self::Compress) -> Self::Compress {
                self.0.compress(lhs, rhs)
            }
            fn id_rake() -> Self::Rake {}
            fn rake(&self, _: &Self::Rake, _: &Self::Rake) -> Self::Rake {
                ()
            }
            fn collapse_compressed(&self, _: &Self::Compress) -> Self::Rake {}
            fn collapse_raked(&self, _: &Self::Rake, weight: &Self::V) -> Self::Compress {
                self.0.make_leaf(weight)
            }
            fn make_leaf(&self, weight: &Self::V) -> Self::Compress {
                self.0.make_leaf(weight)
            }

            const REVERSE_TYPE: Option<WeightType> = Cx::REVERSE_TYPE;
            fn reverse(&self, c: &Self::Compress) -> Self::Compress {
                self.0.reverse(c)
            }
        }

        impl<Cx: ClusterCx> Reducer<Cx> for Path<Cx> {
            type Co = Path<Cx>;
            fn co(&self) -> &Self::Co {
                &self
            }
            fn map_compress(&self, c: &Cx::Compress) -> Cx::Compress {
                c.clone()
            }
            fn map_rake(&self, _r: &Cx::Rake) -> () {}
        }

        /// A pure projeciton, but with an induced cluster operations from its right inverse.
        /// (The compiler should be able to inline this, but it's not garanteed.)
        pub trait SplitProj {
            type Dom: ClusterCx;
            type CoCompress: Clone;
            type CoRake: Clone;
            fn dom(&self) -> &Self::Dom;
            fn proj_compress(c: &<Self::Dom as ClusterCx>::Compress) -> Self::CoCompress;
            fn proj_rake(r: &<Self::Dom as ClusterCx>::Rake) -> Self::CoRake;
            fn embed_compress(c: &Self::CoCompress) -> <Self::Dom as ClusterCx>::Compress;
            fn embed_rake(c: &Self::CoRake) -> <Self::Dom as ClusterCx>::Rake;
        }

        fn proj<P: SplitProj>(cl: &Cluster<P::Dom>) -> Cluster<P> {
            match cl {
                Cluster::Compress(c) => Cluster::Compress(P::proj_compress(c)),
                Cluster::Rake(r) => Cluster::Rake(P::proj_rake(r)),
            }
        }

        fn embed<P: SplitProj>(cl: &Cluster<P>) -> Cluster<P::Dom> {
            match cl {
                Cluster::Compress(c) => Cluster::Compress(P::embed_compress(c)),
                Cluster::Rake(r) => Cluster::Rake(P::embed_rake(r)),
            }
        }

        impl<P: SplitProj> ClusterCx for P {
            type V = <P::Dom as ClusterCx>::V;
            type Compress = P::CoCompress;
            type Rake = P::CoRake;

            fn id_compress() -> Self::Compress {
                Self::proj_compress(&P::Dom::id_compress())
            }
            fn compress(&self, lhs: &Self::Compress, rhs: &Self::Compress) -> Self::Compress {
                Self::proj_compress(
                    &self
                        .dom()
                        .compress(&Self::embed_compress(lhs), &Self::embed_compress(rhs)),
                )
            }

            fn id_rake() -> Self::Rake {
                Self::proj_rake(&P::Dom::id_rake())
            }
            fn rake(&self, lhs: &Self::Rake, rhs: &Self::Rake) -> Self::Rake {
                Self::proj_rake(
                    &self
                        .dom()
                        .rake(&Self::embed_rake(lhs), &Self::embed_rake(rhs)),
                )
            }

            fn collapse_compressed(&self, c: &Self::Compress) -> Self::Rake {
                Self::proj_rake(&self.dom().collapse_compressed(&Self::embed_compress(c)))
            }
            fn collapse_raked(&self, r: &Self::Rake, weight: &Self::V) -> Self::Compress {
                Self::proj_compress(&self.dom().collapse_raked(&Self::embed_rake(r), weight))
            }
            fn make_leaf(&self, weight: &Self::V) -> Self::Compress {
                Self::proj_compress(&self.dom().make_leaf(weight))
            }

            fn push_down(
                &self,
                node: &mut Cluster<Self>,
                children: [Option<&mut Cluster<Self>>; 2],
                weight: &mut Self::V,
            ) {
                let mut lifted_node = embed(node);
                let mut lifted_children: [_; 2] =
                    std::array::from_fn(|i| children[i].as_ref().map(|c| embed(c)));
                let lifted_children_mut = lifted_children.split_at_mut(1);
                let lifted_children_mut = [
                    (&mut lifted_children_mut.0[0]).as_mut(),
                    (&mut lifted_children_mut.1[0]).as_mut(),
                ];

                self.dom()
                    .push_down(&mut lifted_node, lifted_children_mut, weight);

                *node = proj(&lifted_node);
                for (c, l) in children.into_iter().zip(lifted_children) {
                    c.map(|c| *c = proj(&l.unwrap()));
                }

                todo!(" Test push_down");
            }

            fn pull_up(
                &self,
                node: &mut Cluster<Self>,
                children: [Option<&mut Cluster<Self>>; 2],
                weight: &Self::V,
            ) {
                let mut lifted_node = embed(node);
                let mut lifted_children: [_; 2] =
                    std::array::from_fn(|i| children[i].as_ref().map(|c| embed(c)));
                let lifted_children_mut = lifted_children.split_at_mut(1);
                let lifted_children_mut = [
                    (&mut lifted_children_mut.0[0]).as_mut(),
                    (&mut lifted_children_mut.1[0]).as_mut(),
                ];

                self.dom()
                    .pull_up(&mut lifted_node, lifted_children_mut, weight);

                *node = proj(&lifted_node);
                for (c, l) in children.into_iter().zip(lifted_children) {
                    c.map(|c| *c = proj(&l.unwrap()));
                }

                todo!(" Test pull_up");
            }

            const REVERSE_TYPE: Option<WeightType> = <P::Dom as ClusterCx>::REVERSE_TYPE;
            fn reverse(&self, c: &Self::Compress) -> Self::Compress {
                Self::proj_compress(&self.dom().reverse(&Self::embed_compress(c)))
            }
        }

        impl<P: SplitProj> Reducer<P::Dom> for P {
            type Co = P;
            fn co(&self) -> &Self::Co {
                self
            }
            fn map_compress(&self, c: &<P::Dom as ClusterCx>::Compress) -> P::CoCompress {
                Self::proj_compress(c)
            }
            fn map_rake(&self, r: &<P::Dom as ClusterCx>::Rake) -> <Self::Co as ClusterCx>::Rake {
                Self::proj_rake(r)
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
            pub fn into_result(self) -> Result<Cx::Compress, Cx::Rake> {
                match self {
                    Cluster::Compress(c) => Ok(c),
                    Cluster::Rake(r) => Err(r),
                }
            }

            pub fn into_compress(self) -> Option<Cx::Compress> {
                match self {
                    Cluster::Compress(c) => Some(c),
                    _ => None,
                }
            }

            pub fn get_compress(&self) -> Option<&Cx::Compress> {
                match self {
                    Cluster::Compress(c) => Some(c),
                    _ => None,
                }
            }

            pub fn get_compress_mut(&mut self) -> Option<&mut Cx::Compress> {
                match self {
                    Cluster::Compress(c) => Some(c),
                    _ => None,
                }
            }

            pub fn into_rake(self) -> Option<Cx::Rake> {
                match self {
                    Cluster::Rake(r) => Some(r),
                    _ => None,
                }
            }

            pub fn get_rake(&self) -> Option<&Cx::Rake> {
                match self {
                    Cluster::Rake(r) => Some(r),
                    _ => None,
                }
            }

            pub fn get_rake_mut(&mut self) -> Option<&mut Cx::Rake> {
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

            // Stores sequence of branches from the root to the node, bit-packed. Padded an extra 1 to indicate the end.
            // (e.g. root -> right child -> right child -> left child is represented by `0b1011`.)
            // Required for fast node locations, path queries and lazy propagations.
            path: Vec<u64>,

            // Weights and aggregates
            pub cx: Cx,
            pub weights: Vec<Cx::V>,
            pub clusters: Vec<Cluster<Cx>>,
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

                    n_nodes: 1,
                    root_node: dummy,
                    size: vec![1; nodes_cap],
                    children: vec![[None; 2]; nodes_cap],
                    parent: vec![None; nodes_cap],

                    compress_leaf: vec![dummy; nodes_cap],
                    compress_root: vec![dummy; nodes_cap],

                    path: vec![0; nodes_cap],

                    cx,
                    weights: vec![Default::default(); nodes_cap],
                    clusters: vec![Cluster::Compress(Cx::id_compress()); nodes_cap],
                };

                this.build_topology(&hld);
                this.build_locators();
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
                let i = (0..nodes.len() - 1)
                    .min_by_key(|&i| {
                        total_size -= self.size[nodes[i].usize()] as i32 * 2;
                        total_size.abs()
                    })
                    .map_or(nodes.len() - 1, |i| i + 1);

                let (lhs, rhs) = nodes.split_at(i);
                let lhs = self.fold_balanced_rec(lhs, id_cluster);
                let rhs = self.fold_balanced_rec(rhs, id_cluster);
                let node = self.alloc([Some(lhs), Some(rhs)]);
                self.clusters[node.usize()] = id_cluster();
                node
            }

            fn build_locators(&mut self) {
                self.path[self.root_node.usize()] = 1;
                for u in (1..self.n_nodes as u32).rev().map(NodeRef::new) {
                    let path_u = self.path[u.usize()];
                    let depth = u64::BITS - 1 - u64::leading_zeros(path_u);
                    for branch in 0..2 {
                        if let Some(c) = self.children[u.usize()][branch as usize] {
                            self.path[c.usize()] = ((0b11 ^ branch) << depth) ^ path_u;
                        }
                    }
                }
            }

            fn depth(&self, u: NodeRef) -> u32 {
                let path_u = self.path[u.usize()];
                u64::BITS - 1 - u64::leading_zeros(path_u)
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

            // Pin-point traversal & binary search

            #[inline]
            pub unsafe fn walk_down_internal(
                &mut self,
                mut u: NodeRef,
                mut locator: impl FnMut(&mut Self, NodeRef) -> Option<usize>,
            ) -> NodeRef {
                loop {
                    match locator(self, u) {
                        Some(branch) => u = self.children[u.usize()][branch].unwrap_unchecked(),
                        None => return u,
                    }
                }
            }

            #[inline]
            pub unsafe fn walk_down_path_internal(
                &mut self,
                u: NodeRef,
                mut path: u64,
                mut visitor: impl FnMut(&mut Self, NodeRef),
            ) -> NodeRef {
                self.walk_down_internal(u, |this, u| {
                    (path != 0b1).then(|| {
                        visitor(this, u);
                        let branch = (path & 1) as usize;
                        path >>= 1;
                        branch
                    })
                })
            }

            // A bunch of propagation helpers.

            fn push_down(&mut self, u: NodeRef) {
                if Cx::NO_OP_LAZY {
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
                if Cx::NO_OP_LAZY {
                    return;
                }

                unsafe {
                    let u = self.walk_down_path_internal(
                        self.root_node,
                        self.path[u.usize()],
                        Self::push_down,
                    );
                    self.push_down(u);
                }
            }

            fn pull_up_to_root(&mut self, mut u: NodeRef) {
                self.pull_up(u);
                while let Some(p) = self.parent[u.usize()] {
                    u = p;
                    self.pull_up(u);
                }
            }

            /// Sum on the proper subtree
            pub fn sum_subtree<Co: ClusterCx<V = Cx::V>, P: Reducer<Cx, Co = Co>>(
                &mut self,
                u: usize,
                reducer: P,
            ) -> (Co::Rake, &Cx::V) {
                self.push_down_from_root(self.compress_leaf[u]);
                self.push_down(self.compress_leaf[u]);
                if !Cx::NO_OP_LAZY {
                    self.pull_up_to_root(self.compress_leaf[u]);
                }

                let top = self.compress_root[self.hld.chain_top[u] as usize];
                let u = self.compress_leaf[u];
                let mut v = u;
                let mut suffix = Co::id_compress();
                while v != top {
                    let p = self.parent[v.usize()].unwrap();
                    let branch = (self.children[p.usize()][1] == Some(v)) as usize;
                    if branch == 0 {
                        let rhs = unsafe { self.children[p.usize()][1].unwrap_unchecked() };
                        let rhs = unsafe {
                            reducer.map_compress(
                                (&self.clusters[rhs.usize()])
                                    .get_compress()
                                    .unwrap_unchecked(),
                            )
                        };
                        suffix = reducer.co().compress(&suffix, &rhs);
                    }
                    v = p;
                }

                let mut sum_as_rake = reducer.co().collapse_compressed(&suffix);
                if let Some(lhs) = self.children[u.usize()][0] {
                    let lhs = unsafe { self.clusters[lhs.usize()].get_rake().unwrap_unchecked() };
                    sum_as_rake = reducer.co().rake(&reducer.map_rake(&lhs), &sum_as_rake);
                }

                (sum_as_rake, &self.weights[u.usize()])
            }

            pub fn apply_subtree<A: Action<Cx>>(
                &mut self,
                u: usize,
                apply_to_root: bool,
                mut action: A,
            ) {
                self.push_down_from_root(self.compress_leaf[u]);

                // Apply to the rake tree + suffix of the compress tree
                let top = self.compress_root[self.hld.chain_top[u] as usize];
                let u = self.compress_leaf[u];

                if let Some(lhs) = self.children[u.usize()][0] {
                    let lhs =
                        unsafe { self.clusters[lhs.usize()].get_rake_mut().unwrap_unchecked() };
                    action.apply_to_rake(lhs);
                }

                if apply_to_root {
                    action.apply_to_weight(&mut self.weights[u.usize()]);
                }

                let mut v = u;
                while v != top {
                    let p = self.parent[v.usize()].unwrap();
                    let branch = (self.children[p.usize()][1] == Some(v)) as usize;
                    if branch == 0 {
                        let rhs = unsafe { self.children[p.usize()][1].unwrap_unchecked() };
                        action.apply(&mut self.clusters[rhs.usize()], ActionRange::Subtree);
                    }
                    v = p;
                }

                self.pull_up_to_root(u);
            }

            pub fn sum_path<Co: ClusterCx<V = Cx::V>, P: Reducer<Cx, Co = Co>>(
                &mut self,
                u: usize,
                v: usize,
                reducer: P,
            ) -> (Co::Compress, &Cx::V, Co::Compress) {
                // Since we cannot expose the path cluster directly as in ST-(Link-Cut-)top trees,
                // implementing path queries becomes a bit tricky:
                // We perform a range sum for the topmost chain and a prefix sum for the rest.
                // Both the LCA in the represented tree and the internal tree must be identified.

                let mut u = self.compress_leaf[u];
                let mut v = self.compress_leaf[v];
                let mut path_u = self.path[u.usize()];
                let mut path_v = self.path[v.usize()];
                let mut depth_u = self.depth(u);
                let mut depth_v = self.depth(u);

                // Locate the internal LCA, and propagate downwards.
                let depth_lca_internal =
                    (path_u ^ path_v).trailing_zeros().min(depth_u).min(depth_v);
                let path_lca_internal =
                    (path_u & path_v & ((1 << depth_lca_internal) - 1)) | (1 << depth_lca_internal);
                let lca_internal = unsafe {
                    self.walk_down_path_internal(self.root_node, path_lca_internal, Self::push_down)
                };

                path_u >>= depth_lca_internal;
                path_v >>= depth_lca_internal;
                if !Cx::NO_OP_LAZY {
                    unsafe {
                        self.walk_down_path_internal(lca_internal, path_u, Self::push_down);
                        self.walk_down_path_internal(lca_internal, path_v, Self::push_down);
                    }
                    self.push_down(u);
                    self.push_down(v);
                }

                // Make the topmost chain belong to the path_u.
                let swap_uv = if depth_u < depth_lca_internal {
                    path_u & 1 == 0
                } else {
                    path_v & 1 == 1
                };
                if swap_uv {
                    std::mem::swap(&mut u, &mut v);
                    std::mem::swap(&mut path_u, &mut path_v);
                    std::mem::swap(&mut depth_u, &mut depth_v);
                }

                let mut qu = lca_internal;
                let mut qv = lca_internal;

                let mut sum_u = Co::id_compress();
                let mut sum_v = Co::id_compress();

                // Find the LCA in the represented tree.
                let mut lca_represented = lca_internal;
                let lca_represented_on_top_chain;
                {
                    use Cluster::*;
                    match &self.clusters[lca_internal.usize()] {
                        Rake(_) => {
                            lca_represented_on_top_chain = false;

                            // Ascend to the first compress node.
                            loop {
                                lca_represented = unsafe {
                                    self.parent[lca_represented.usize()].unwrap_unchecked()
                                };
                                match &self.clusters[lca_represented.usize()] {
                                    Compress(_) => break,
                                    _ => {}
                                }
                            }
                        }
                        Compress(_) => {
                            lca_represented_on_top_chain = true;

                            // Descend to the last compress node.
                            let mut path = path_v;
                            while path != 0b1 {
                                match lca_represented
                                    .get_with_children_in(&self.children, &mut self.clusters)
                                {
                                    (Compress(_), [Some(Compress(_)), Some(Compress(_))]) => {}
                                    _ => break,
                                }

                                let branch = (path & 1) as usize;
                                path >>= 1;
                                lca_represented = unsafe {
                                    self.children[lca_represented.usize()][branch]
                                        .unwrap_unchecked()
                                };
                            }
                        }
                    }
                }

                // Descend.
                let c = |clusters: &[Cluster<_>], u: NodeRef| unsafe {
                    reducer.map_compress(clusters[u.usize()].get_compress().unwrap_unchecked())
                };
                let step_down = |clusters: &mut [_], dir: usize, q: &mut NodeRef, path: &mut _| {
                    let branch = (*path & 1) as usize;
                    let p = *q;
                    *path >>= 1;
                    *q = unsafe { self.children[q.usize()][branch].unwrap_unchecked() };

                    use Cluster::*;
                    match p.get_with_children_in(&self.children, clusters) {
                        (Compress(_), [Some(Compress(_)), Some(Compress(_))])
                            if branch == (dir ^ 1) =>
                        {
                            let r = unsafe { self.children[p.usize()][dir].unwrap_unchecked() };
                            Some(c(clusters, r))
                        }
                        (Compress(_), [_rake_root, None]) => Some(c(clusters, p)),
                        _ => None,
                    }
                };

                if path_u != 0b1 {
                    step_down(&mut self.clusters, 0, &mut qu, &mut path_u);
                    while path_u != 0b1 {
                        step_down(&mut self.clusters, 0, &mut qu, &mut path_u)
                            .map(|x| sum_u = reducer.co().compress(&sum_u, &x));
                    }
                }

                if path_v != 0b1 {
                    if lca_represented_on_top_chain {
                        let branch = (path_v & 1) as usize;
                        let prev = qv;
                        step_down(&mut self.clusters, 1, &mut qv, &mut path_v);
                        if prev != lca_represented && branch == 0 {
                            while path_v != 0b1 {
                                if qv == lca_represented {
                                    step_down(&mut self.clusters, 1, &mut qv, &mut path_v);
                                    break;
                                }
                                step_down(&mut self.clusters, 1, &mut qv, &mut path_v)
                                    .map(|x| sum_u = reducer.co().compress(&x, &sum_u));
                            }
                        }
                    }

                    while path_v != 0b1 {
                        step_down(&mut self.clusters, 0, &mut qv, &mut path_v)
                            .map(|x| sum_v = reducer.co().compress(&sum_v, &x));
                    }
                }

                if u != lca_represented {
                    let x = c(&mut self.clusters, u);
                    sum_u = reducer.co().compress(&sum_u, &x);
                }
                if v != lca_represented {
                    let x = c(&mut self.clusters, v);
                    sum_v = reducer.co().compress(&sum_v, &x);
                }

                if swap_uv {
                    std::mem::swap(&mut sum_u, &mut sum_v);
                }

                let top_weight = &self.weights[lca_represented.usize()];

                (sum_u, top_weight, sum_v)
            }

            pub fn apply_path(
                &mut self,
                u: usize,
                v: usize,
                apply_to_lca: bool,
                mut action: impl Action<Cx>,
            ) {
                // Implementation details:
                // `apply_to_weight` is used instead of `apply_to_compress` on terminal compressed clusters.

                let mut u = self.compress_leaf[u];
                let mut v = self.compress_leaf[v];
                let mut path_u = self.path[u.usize()];
                let mut path_v = self.path[v.usize()];
                let mut depth_u = self.depth(u);
                let mut depth_v = self.depth(u);

                // Locate the internal LCA, and propagate downwards.
                let depth_lca_internal =
                    (path_u ^ path_v).trailing_zeros().min(depth_u).min(depth_v);
                let path_lca_internal =
                    (path_u & path_v & ((1 << depth_lca_internal) - 1)) | (1 << depth_lca_internal);
                let lca_internal = unsafe {
                    self.walk_down_path_internal(self.root_node, path_lca_internal, Self::push_down)
                };

                path_u >>= depth_lca_internal;
                path_v >>= depth_lca_internal;
                if !Cx::NO_OP_LAZY {
                    unsafe {
                        self.walk_down_path_internal(lca_internal, path_u, Self::push_down);
                        self.walk_down_path_internal(lca_internal, path_v, Self::push_down);
                    }
                    self.push_down(u);
                    self.push_down(v);
                }

                // Make the topmost chain belong to the path_u.
                let swap_uv = if depth_u < depth_lca_internal {
                    path_u & 1 == 0
                } else {
                    path_v & 1 == 1
                };
                if swap_uv {
                    std::mem::swap(&mut u, &mut v);
                    std::mem::swap(&mut path_u, &mut path_v);
                    std::mem::swap(&mut depth_u, &mut depth_v);
                }

                let mut qu = lca_internal;
                let mut qv = lca_internal;

                // Find the LCA in the represented tree.
                let mut lca_represented = lca_internal;
                let lca_represented_on_top_chain;
                {
                    use Cluster::*;
                    match &self.clusters[lca_internal.usize()] {
                        Rake(_) => {
                            lca_represented_on_top_chain = false;

                            // Ascend to the first compress node.
                            loop {
                                lca_represented = unsafe {
                                    self.parent[lca_represented.usize()].unwrap_unchecked()
                                };
                                match &self.clusters[lca_represented.usize()] {
                                    Compress(_) => break,
                                    _ => {}
                                }
                            }
                        }
                        Compress(_) => {
                            lca_represented_on_top_chain = true;

                            // Descend to the last compress node.
                            let mut path = path_v;
                            while path != 0b1 {
                                match lca_represented
                                    .get_with_children_in(&self.children, &mut self.clusters)
                                {
                                    (Compress(_), [Some(Compress(_)), Some(Compress(_))]) => {}
                                    _ => break,
                                }

                                let branch = (path & 1) as usize;
                                path >>= 1;
                                lca_represented = unsafe {
                                    self.children[lca_represented.usize()][branch]
                                        .unwrap_unchecked()
                                };
                            }
                        }
                    }
                }

                // Descend.
                fn c<Cx: ClusterCx>(clusters: &mut [Cluster<Cx>], u: NodeRef) -> &mut Cx::Compress {
                    unsafe { clusters[u.usize()].get_compress_mut().unwrap_unchecked() }
                }
                let mut step_down = |clusters: &mut [Cluster<Cx>],
                                     dir: usize,
                                     q: &mut NodeRef,
                                     path: &mut _,
                                     apply: bool| {
                    let branch = (*path & 1) as usize;
                    let p = *q;
                    *path >>= 1;
                    *q = unsafe { self.children[q.usize()][branch].unwrap_unchecked() };

                    use Cluster::*;

                    if apply {
                        match p.get_with_children_in(&self.children, clusters) {
                            (Compress(_), [Some(Compress(_)), Some(Compress(_))])
                                if branch == (dir ^ 1) =>
                            {
                                let r = unsafe { self.children[p.usize()][dir].unwrap_unchecked() };
                                action.apply_to_compress(c(clusters, r), ActionRange::Path);
                            }
                            (Compress(_), [_rake_root, None]) => {
                                action.apply_to_weight(&mut self.weights[p.usize()]);
                            }
                            _ => {}
                        }
                    }
                };

                if path_u != 0b1 {
                    step_down(&mut self.clusters, 0, &mut qu, &mut path_u, false);
                    while path_u != 0b1 {
                        step_down(&mut self.clusters, 0, &mut qu, &mut path_u, true)
                    }
                }

                if path_v != 0b1 {
                    if lca_represented_on_top_chain {
                        let branch = (path_v & 1) as usize;
                        let prev = qv;
                        step_down(&mut self.clusters, 1, &mut qv, &mut path_v, false);
                        if prev != lca_represented && branch == 0 {
                            while path_v != 0b1 {
                                if qv == lca_represented {
                                    step_down(&mut self.clusters, 1, &mut qv, &mut path_v, false);
                                    break;
                                }
                                step_down(&mut self.clusters, 1, &mut qv, &mut path_v, true)
                            }
                        }
                    }

                    while path_v != 0b1 {
                        step_down(&mut self.clusters, 0, &mut qv, &mut path_v, true)
                    }
                }

                if u != lca_represented {
                    action.apply_to_weight(&mut self.weights[u.usize()]);
                }
                if v != lca_represented {
                    action.apply_to_weight(&mut self.weights[v.usize()]);
                }
                if apply_to_lca {
                    action.apply_to_weight(&mut self.weights[lca_represented.usize()]);
                }

                self.pull_up_to_root(u);
                self.pull_up_to_root(v);
            }
        }
    }
}

type X = u32;
type F = (u32, u32);

struct AffineSpace;

impl AffineSpace {
    fn id(&self) -> X {
        0
    }

    fn combine(&self, lhs: &X, rhs: &X) -> X {
        lhs.wrapping_add(*rhs)
    }

    fn id_action(&self) -> F {
        (0, 1)
    }

    fn combine_action(&self, lhs: &F, rhs: &F) -> F {
        (
            lhs.1.wrapping_mul(rhs.0).wrapping_add(lhs.0),
            lhs.1.wrapping_mul(rhs.1),
        )
    }

    fn apply_to_sum(&self, f: &F, x_count: u32, x_sum: &X) -> X {
        f.0.wrapping_mul(x_count)
            .wrapping_add(f.1.wrapping_mul(*x_sum))
    }
}

#[derive(Clone, Debug)]
pub struct LazyNode {
    count_path: u32,
    count_proper_subtree: u32,

    sum_path: X,
    sum_proper_subtree: X,

    lazy_path: F,
    lazy_proper_subtree: F,
}

impl Default for LazyNode {
    fn default() -> Self {
        Self {
            count_path: 0,
            count_proper_subtree: 0,

            sum_path: AffineSpace.id(),
            sum_proper_subtree: AffineSpace.id(),

            lazy_path: AffineSpace.id_action(),
            lazy_proper_subtree: AffineSpace.id_action(),
        }
    }
}

#[derive(Debug)]
pub struct LazyPropCx;

impl Action<LazyPropCx> for F {
    fn apply_to_compress(
        &mut self,
        cl: &mut <LazyPropCx as ClusterCx>::Compress,
        range: ActionRange,
    ) {
        cl.sum_path = AffineSpace.apply_to_sum(self, cl.count_path, &cl.sum_path);
        cl.lazy_path = AffineSpace.combine_action(self, &cl.lazy_path);
        match range {
            ActionRange::Subtree => {
                cl.sum_proper_subtree =
                    AffineSpace.apply_to_sum(self, cl.count_proper_subtree, &cl.sum_proper_subtree);
                cl.lazy_proper_subtree = AffineSpace.combine_action(self, &cl.lazy_proper_subtree);
            }
            ActionRange::Path => {}
        }
    }

    fn apply_to_rake(&mut self, cl: &mut <LazyPropCx as ClusterCx>::Rake) {
        cl.sum_proper_subtree =
            AffineSpace.apply_to_sum(self, cl.count_proper_subtree, &cl.sum_proper_subtree);
        cl.lazy_proper_subtree = AffineSpace.combine_action(self, &cl.lazy_proper_subtree);
    }

    fn apply_to_weight(&mut self, weight: &mut <LazyPropCx as ClusterCx>::V) {
        *weight = AffineSpace.apply_to_sum(self, 1, weight);
    }
}

impl ClusterCx for LazyPropCx {
    type V = X;
    type Compress = LazyNode;
    type Rake = LazyNode;

    fn id_compress() -> Self::Compress {
        Default::default()
    }
    fn compress(&self, lhs: &Self::Compress, rhs: &Self::Compress) -> Self::Compress {
        LazyNode {
            sum_path: AffineSpace.combine(&lhs.sum_path, &rhs.sum_path),
            sum_proper_subtree: AffineSpace
                .combine(&lhs.sum_proper_subtree, &rhs.sum_proper_subtree),
            count_path: lhs.count_path + rhs.count_path,
            count_proper_subtree: lhs.count_proper_subtree + rhs.count_proper_subtree,

            ..Default::default()
        }
    }

    fn id_rake() -> Self::Rake {
        Default::default()
    }
    fn rake(&self, lhs: &Self::Rake, rhs: &Self::Rake) -> Self::Rake {
        LazyNode {
            count_proper_subtree: lhs.count_proper_subtree + rhs.count_proper_subtree,
            sum_proper_subtree: AffineSpace
                .combine(&lhs.sum_proper_subtree, &rhs.sum_proper_subtree),

            ..Default::default()
        }
    }

    fn collapse_compressed(&self, c: &Self::Compress) -> Self::Rake {
        LazyNode {
            count_proper_subtree: c.count_proper_subtree + c.count_path,
            sum_proper_subtree: c.sum_proper_subtree + c.sum_path,
            ..Default::default()
        }
    }
    fn collapse_raked(&self, r: &Self::Rake, weight: &Self::V) -> Self::Compress {
        LazyNode {
            count_path: 1,
            count_proper_subtree: r.count_proper_subtree,

            sum_path: *weight,
            sum_proper_subtree: r.sum_proper_subtree,

            ..Default::default()
        }
    }

    fn make_leaf(&self, weight: &Self::V) -> Self::Compress {
        LazyNode {
            count_path: 1,
            count_proper_subtree: 0,

            sum_path: *weight,
            sum_proper_subtree: 0,

            ..Default::default()
        }
    }

    fn push_down(
        &self,
        node: &mut Cluster<Self>,
        children: [Option<&mut Cluster<Self>>; 2],
        weight: &mut Self::V,
    ) {
        use Cluster::*;
        match (node, children) {
            (Compress(c), [Some(Compress(lhs)), Some(Compress(rhs))]) => {
                for child in [lhs, rhs] {
                    child.sum_path =
                        AffineSpace.apply_to_sum(&c.lazy_path, child.count_path, &child.sum_path);
                    child.lazy_path = AffineSpace.combine_action(&c.lazy_path, &child.lazy_path);

                    child.sum_proper_subtree = AffineSpace.apply_to_sum(
                        &c.lazy_proper_subtree,
                        child.count_proper_subtree,
                        &child.sum_proper_subtree,
                    );
                    child.lazy_proper_subtree = AffineSpace
                        .combine_action(&c.lazy_proper_subtree, &child.lazy_proper_subtree);
                }

                c.lazy_path = AffineSpace.id_action();
                c.lazy_proper_subtree = AffineSpace.id_action();
            }
            (Compress(c), [Some(Rake(top)), None]) => {
                c.lazy_path.apply_to_weight(weight);
                c.lazy_proper_subtree.apply_to_rake(top);

                c.lazy_path = AffineSpace.id_action();
                c.lazy_proper_subtree = AffineSpace.id_action();
            }
            (Compress(c), [None, None]) => {
                c.lazy_path.apply_to_weight(weight);

                c.lazy_path = AffineSpace.id_action();
            }
            (Rake(r), [Some(Rake(lhs)), Some(Rake(rhs))]) => {
                for child in [lhs, rhs] {
                    r.lazy_proper_subtree.apply_to_rake(child);
                }

                r.lazy_proper_subtree = AffineSpace.id_action();
            }
            (Rake(r), [Some(Compress(top)), None]) => {
                r.lazy_proper_subtree
                    .apply_to_compress(top, ActionRange::Subtree);

                r.lazy_proper_subtree = AffineSpace.id_action();
            }
            _ => unsafe { unreachable_unchecked() },
        }
    }
}

struct Additive<const SUBTREE: bool>;

impl<const SUBTREE: bool> SplitProj for Additive<SUBTREE> {
    type Dom = LazyPropCx;
    type CoCompress = u32;
    type CoRake = u32;

    fn dom(&self) -> &Self::Dom {
        &LazyPropCx
    }

    fn proj_compress(c: &<Self::Dom as ClusterCx>::Compress) -> Self::CoCompress {
        if SUBTREE {
            c.sum_path + c.sum_proper_subtree
        } else {
            c.sum_path
        }
    }

    fn proj_rake(r: &<Self::Dom as ClusterCx>::Rake) -> Self::CoRake {
        if SUBTREE {
            r.sum_proper_subtree
        } else {
            0
        }
    }

    fn embed_compress(c: &Self::CoCompress) -> <Self::Dom as ClusterCx>::Compress {
        LazyNode {
            sum_path: *c,
            ..Default::default()
        }
    }

    fn embed_rake(c: &Self::CoRake) -> <Self::Dom as ClusterCx>::Rake {
        if SUBTREE {
            LazyNode {
                sum_proper_subtree: *c,
                ..Default::default()
            }
        } else {
            Default::default()
        }
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();
    let edges = (0..n - 1).map(|_| (input.u32() - 1, input.u32() - 1));
    let mut stt = StaticTopTree::from_edges(n, edges, 0, LazyPropCx);
    let weights = (0..n).map(|u| 0);
    stt.init_weights(weights.enumerate());

    for _ in 0..q {
        let cmd = input.token();
        let action = |value| match cmd {
            "1" | "2" => (value, 1),
            "3" | "4" => (0, value),
            _ => unreachable!(),
        };
        match cmd {
            "1" | "3" => {
                let u = input.u32() as usize - 1;
                let x = input.u32();
                stt.apply_subtree(u, true, action(x));
            }
            "2" | "4" => {
                let u = input.u32() as usize - 1;
                let v = input.u32() as usize - 1;
                let x = input.u32();
                stt.apply_path(u, v, true, action(x));
            }
            "5" => {
                let u = input.u32() as usize - 1;
                let (proper_subtree, top) = stt.sum_subtree(u, Additive::<true>);
                let ans = AffineSpace.combine(&proper_subtree, &top);
                writeln!(output, "{}", ans).unwrap();
            }
            "6" => {
                let u = input.u32() as usize - 1;
                let v = input.u32() as usize - 1;
                let (path_u, lca, path_v) = stt.sum_path(u, v, Additive::<false>);
                let ans = AffineSpace.combine(&path_u, &AffineSpace.combine(&lca, &path_v));
                writeln!(output, "{}", ans).unwrap();
            }
            _ => panic!(),
        }
    }
}
