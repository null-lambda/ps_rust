use std::io::Write;

use static_top_tree::StaticTopTree;

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

pub mod static_top_tree {
    /// # Static Top Tree
    /// Extend segment tree to rooted trees in O(N log N).
    /// Compared to usual edge-based top trees, this one is vertex-based.
    ///
    /// ## Reference:
    /// - ([Tutorial] Theorically Faster HLD and Centroid Decomposition)[https://codeforces.com/blog/entry/104997/
    /// - (ABC 351G Editorial)[https://atcoder.jp/contests/abc351/editorial/9899]
    ///
    /// ## Note
    /// If you need a query on a forest, separate each connected component and reindex vertices.
    ///
    /// ## TODO
    //  - Persistence
    //  - Lazy propagation + path query + subtree query
    //  - (Optimization) Reduce pointer chasing by reindexing the tree into a complete binary tree.
    //    Pad to power of two and fill holes with identity clusters.
    //        => children(u) = [u << 1, u << 1 | 1], parent(u) = u >> 1, root = 1
    use std::num::NonZeroU32;

    pub const UNSET: u32 = !0;

    pub trait ClusterCx {
        // Vertex weight / weight of an upward edge (u, parent).
        type V: Default + Clone;

        type Compress: Clone; // Path cluster (aggregate on a subchain)
        type Rake: Clone; // Point cluster (Aggregate of light edges)

        // const USE_LAZY: bool = false;
        // type F;

        // const USE_SUBTREE_QUERY: bool = true;

        // Compress monoid
        fn id_chain(&self) -> Self::Compress;
        fn compress(&self, lhs: &Self::Compress, rhs: &Self::Compress) -> Self::Compress;

        // Rake monoid, commutative
        fn id_light_edge(&self) -> Self::Rake;
        fn rake(&self, lhs: &Self::Rake, rhs: &Self::Rake) -> Self::Rake;

        fn collapse_compressed(&self, path: &Self::Compress) -> Self::Rake;
        fn collapse_raked(&self, point: &Self::Rake, top_weight: &Self::V) -> Self::Compress;
        fn make_leaf(&self, weight: &Self::V) -> Self::Compress; // In case of no associated rake edge

        // fn try_apply_to_compressed(&self, chain: &mut Self::Compress, f: &Self::F) -> bool;
        // fn try_apply_to_raked(&self, point: &mut Self::Rake, f: &Self::F) -> bool;
    }

    fn for_each_in_list(xor_links: &[u32], entry: u32, mut visitor: impl FnMut(u32)) -> u32 {
        let mut u = entry;
        let mut prev = UNSET;
        loop {
            visitor(u);
            let next = xor_links[u as usize] ^ prev;
            if next == UNSET {
                return u;
            }
            prev = u;
            u = next;
        }
    }

    // Heavy-Light Decomposition, prior to top tree construction.
    // Golfable within ~20 lines with two recursive DFS traversals.
    #[derive(Debug)]
    pub struct HLD {
        // Rooted tree structure
        pub size: Vec<u32>,
        pub parent: Vec<u32>,
        pub topological_order: Vec<u32>,

        // Chain structure
        pub heavy_child: Vec<u32>,
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
            n: usize,
            edges: impl IntoIterator<Item = (u32, u32)>,
            root: usize,
        ) -> Self {
            assert!(n >= 1);
            let mut degree = vec![0u32; n];
            let mut xor_neighbors: Vec<u32> = vec![0u32; n];
            for (u, v) in edges {
                debug_assert!(u != v);
                degree[u as usize] += 1;
                degree[v as usize] += 1;
                xor_neighbors[u as usize] ^= v;
                xor_neighbors[v as usize] ^= u;
            }

            // Upward propagation
            let mut size = vec![1; n];
            let mut heavy_child = vec![UNSET; n];
            degree[root] += 2;
            let mut topological_order = Vec::with_capacity(n);
            for mut u in 0..n {
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
            assert!(topological_order.len() == n, "Invalid tree structure");
            let mut parent = xor_neighbors;
            parent[root] = UNSET;

            let mut first_light_child = vec![UNSET; n];
            let mut xor_light_siblings = vec![UNSET; n];
            for &u in &topological_order[..n - 1] {
                let p = parent[u as usize];

                let c = first_light_child[p as usize];
                xor_light_siblings[u as usize] = c ^ UNSET;
                if c != UNSET {
                    xor_light_siblings[c as usize] ^= u as u32 ^ UNSET;
                }
                first_light_child[p as usize] = u;
            }

            // Downward propagation
            let mut segmented_idx = vec![UNSET; n];
            let mut timer = 0;
            for mut u in topological_order.iter().copied().rev() {
                if segmented_idx[u as usize] != UNSET {
                    continue;
                }
                loop {
                    segmented_idx[u as usize] = timer;
                    timer += 1;
                    u = heavy_child[u as usize];
                    if u == UNSET {
                        break;
                    }
                }
            }

            Self {
                size,
                parent,
                topological_order,

                heavy_child,
                segmented_idx,

                first_light_child,
                xor_light_siblings,
            }
        }
    }

    pub enum Cluster<C: ClusterCx> {
        Compress(C::Compress),
        Rake(C::Rake),
    }

    impl<C: ClusterCx> Clone for Cluster<C> {
        fn clone(&self) -> Self {
            match self {
                Cluster::Compress(c) => Cluster::Compress(c.clone()),
                Cluster::Rake(r) => Cluster::Rake(r.clone()),
            }
        }
    }

    // pub enum NodeType {
    //     Compress,
    //     CollapseRaked,
    //     Rake,
    //     CollapseCompressed,
    //     Leaf,
    // }

    #[derive(Debug, Clone, Copy)]
    struct NodeRef(NonZeroU32);

    impl NodeRef {
        const fn new(idx: u32) -> Self {
            Self(NonZeroU32::new(idx).unwrap())
        }

        fn usize(&self) -> usize {
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

    pub struct Topology {
        root: NodeRef,
        size: Vec<u32>,
        children: Vec<[Option<NodeRef>; 2]>,
        parent: Vec<Option<NodeRef>>,
        n_nodes: u32,
        vert_to_node: Vec<NodeRef>,
    }

    pub struct StaticTopTree<C: ClusterCx> {
        pub cx: C,
        pub hld: HLD, // Unbalanced HLD

        root: NodeRef,
        size: Vec<u32>,
        children: Vec<[Option<NodeRef>; 2]>,
        parent: Vec<Option<NodeRef>>,
        n_nodes: u32,
        vert_to_node: Vec<NodeRef>,

        pub sum: Vec<Cluster<C>>,
        pub weights: Vec<C::V>,
    }

    impl<C: ClusterCx> StaticTopTree<C> {
        pub fn from_edges(
            n: usize,
            edges: impl IntoIterator<Item = (u32, u32)>,
            root: usize,
            cx: C,
        ) -> Self {
            let hld = HLD::from_edges(n, edges, root);
            let dummy = NodeRef::new(!0);
            let nodes_cap = n * 4;
            let mut this = Self {
                hld,

                root: dummy,
                size: vec![1; nodes_cap],
                children: vec![[None; 2]; nodes_cap],
                parent: vec![None; nodes_cap],
                n_nodes: 0,
                vert_to_node: vec![dummy; nodes_cap],

                sum: vec![Cluster::Compress(cx.id_chain())],
                weights: vec![Default::default()],

                cx,
            };
            this.root = this.build_compress_tree(root);
            this
        }

        fn push_down(&mut self, u: NodeRef) {
            unimplemented!("Lazy propagation is not implemented yet");
        }

        fn pull_up(&mut self, u: NodeRef) {
            use Cluster::*;
            let (node, children) = u.get_with_children_in(&self.children, &mut self.sum);
            match (node, children) {
                (Compress(c), [Some(Compress(lhs)), Some(Compress(rhs))]) => {
                    *c = self.cx.compress(lhs, rhs)
                }
                (Compress(c), [Some(Rake(top)), None]) => {
                    *c = self.cx.collapse_raked(top, &self.weights[u.usize()])
                }
                (Compress(c), [None, None]) => {
                    *c = self.cx.make_leaf(&self.weights[u.usize()]);
                }
                (Rake(r), [Some(Rake(lhs)), Some(Rake(rhs))]) => {
                    *r = self.cx.rake(lhs, rhs);
                }
                (Rake(r), [Some(Compress(top)), None]) => {
                    *r = self.cx.collapse_compressed(top);
                }
                _ => unsafe { std::hint::unreachable_unchecked() },
            }
        }

        fn alloc(&mut self, children: [Option<NodeRef>; 2]) -> NodeRef {
            let u = NodeRef::new(self.n_nodes);
            self.n_nodes += 1;
            self.children[u.usize()] = children;
            for &child in children.iter().flatten() {
                self.parent[child.usize()] = Some(u);
                self.size[u.usize()] += self.size[child.usize()];
            }
            u
        }

        fn build_compress_tree(&mut self, u0: usize) -> NodeRef {
            let mut u = u0;
            let mut chain = vec![self.collapse_rake_tree(u)];
            loop {
                u = self.hld.heavy_child[u] as usize;
                if u == UNSET as usize {
                    break;
                }
                chain.push(self.collapse_rake_tree(u));
            }
            let id = Cluster::Compress(self.cx.id_chain());
            self.fold_balanced_rec(&chain, &|| id.clone())
        }

        fn build_rake_tree(&mut self, u: usize) -> Option<NodeRef> {
            let mut light_edges = vec![];
            for_each_in_list(
                &self.hld.xor_light_siblings,
                self.hld.first_light_child[u],
                |v| light_edges.push(self.collapse_compress_tree(v as usize)),
            );

            if light_edges.is_empty() {
                return None;
            }

            let id = Cluster::Rake(self.cx.id_light_edge());
            Some(self.fold_balanced_rec(&light_edges, &|| id.clone()))
        }

        fn collapse_rake_tree(&mut self, u: usize) -> NodeRef {
            let c = self.build_rake_tree(u);
            self.alloc([c, None])
        }

        fn collapse_compress_tree(&mut self, u: usize) -> NodeRef {
            let c = self.build_compress_tree(u);
            self.alloc([Some(c), None])
        }

        fn fold_balanced_rec(
            &mut self,
            nodes: &[NodeRef],
            id_cluster: &impl Fn() -> Cluster<C>,
        ) -> NodeRef {
            debug_assert!(!nodes.is_empty());
            if nodes.len() == 1 {
                return nodes[0];
            }
            let mut total_size = nodes.iter().map(|u| self.size[u.usize()]).sum::<u32>() as i32;
            let i = nodes
                .iter()
                .position(|u| {
                    total_size -= self.size[u.usize()] as i32 * 2;
                    total_size < 0
                })
                .unwrap_or(nodes.len());
            let (lhs, rhs) = nodes.split_at(i);
            let lhs = self.fold_balanced_rec(lhs, id_cluster);
            let rhs = self.fold_balanced_rec(rhs, id_cluster);
            let node = self.alloc([Some(lhs), Some(rhs)]);
            self.sum[node.usize()] = id_cluster();
            node
        }

        pub fn init_weights(&mut self, weights: &[C::V]) {
            todo!()
        }
        pub fn query_all(&self) -> &C::Compress {
            match &self.sum[self.root.usize()] {
                Cluster::Compress(c) => c,
                _ => unsafe { std::hint::unreachable_unchecked() },
            }
        }

        pub fn query_path(&self, u: usize, v: usize) -> C::Compress {
            todo!()
        }

        pub fn query_subtree(&self, u: usize) -> C::Compress {
            todo!()
        }
    }
}

type E = u32;

struct DiameterOp;

#[derive(Clone)]
struct Rake {
    diameter: E,
    depth: E,
}

#[derive(Clone)]
struct Compress {
    diameter: E,
    depth: E,
    len: E,
}

impl static_top_tree::ClusterCx for DiameterOp {
    type V = E;
    type Compress = Compress;
    type Rake = Rake;

    fn id_chain(&self) -> Self::Compress {
        Compress {
            diameter: 0,
            depth: 0,
            len: 0,
        }
    }

    fn compress(&self, lhs: &Self::Compress, rhs: &Self::Compress) -> Self::Compress {
        Compress {
            diameter: lhs.diameter.max(rhs.diameter),
            depth: lhs.depth.max(lhs.len + rhs.depth),
            len: lhs.len + rhs.len,
        }
    }

    fn id_light_edge(&self) -> Self::Rake {
        Rake {
            diameter: 0,
            depth: 0,
        }
    }

    fn rake(&self, lhs: &Self::Rake, rhs: &Self::Rake) -> Self::Rake {
        Rake {
            diameter: lhs.diameter.max(rhs.diameter),
            depth: lhs.depth.max(rhs.depth),
        }
    }

    fn collapse_compressed(&self, path: &Self::Compress) -> Self::Rake {
        Rake {
            diameter: path.diameter,
            depth: path.depth,
        }
    }

    fn collapse_raked(&self, point: &Self::Rake, weight: &Self::V) -> Self::Compress {
        Compress {
            diameter: point.diameter.max(point.depth + weight),
            depth: point.depth + weight,
            len: *weight,
        }
    }

    fn make_leaf(&self, weight: Self::V) -> Self::Compress {
        Compress {
            diameter: weight,
            depth: weight,
            len: weight,
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut edges = vec![];
    for _ in 0..n - 1 {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        let w: u32 = input.value();
        edges.push((u, v, w));
    }

    let mut stt =
        StaticTopTree::from_edges(n, edges.iter().map(|&(u, v, _)| (u, v)), 0, DiameterOp);

    let mut weights = vec![0; n];

    for &(u, v, w) in &edges {
        let bot = if stt.hld.parent[v as usize] == u {
            v
        } else {
            u
        };
        weights[bot as usize] = w;
    }
    stt.init_weights(&weights);

    let ans = stt.query_all().diameter;

    writeln!(output, "{}", ans).unwrap();
}
