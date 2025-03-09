pub mod top_tree {
    /// # Splay Top Tree
    /// Manage dynamic tree dp with link-cut operations in O(N log N).
    ///
    /// Most implementation are derived from the paper of Tarjan & Werneck, with some minor modifications:
    /// - Circular order between rake edges does not preserved, to simplify node structure.
    /// - Each compress node represents an **open** interval (without boundary points),
    ///   and rake node represents a **left-open, right-closed** interval.
    ///   Boundary vertices are processed as late as possible (or not at all) to minimize over-computation.
    /// - We use a separate cluster and node type for compress trees and rake trees.
    ///
    /// ## Reference:
    /// - [Self-adjusting top trees](https://renatowerneck.wordpress.com/wp-content/uploads/2016/06/tw05-self-adjusting-top-tree.pdf)
    /// - [[Tutorial] Fully Dynamic Trees Supporting Path/Subtree Aggregates and Lazy Path/Subtree Updates](https://codeforces.com/blog/entry/103726)
    ///
    /// ## TODO
    ///
    /// ## Features
    /// - Implement link/Cut operations.
    /// - Implement `get_parent` and `lca` (Lowest Common Ancestor) queries with a specified root.
    /// - Implement subtree and rerooting queries.
    /// - Implement binary walking with the signature `fn(impl FnMut(&Cx::R, &Cx::R) -> bool) -> usize`.
    ///
    /// ### Optimization
    /// - Replace unnecessary `push_down`, `pull_up` with a new fn `TopTree::update_handles`.
    /// - Replace the `node::Compress.rev_lazy` tag with endpoint checks (and perform benchmarking).
    /// - Re-implement `modify_edge` without using link/cut operations.
    ///
    /// ## Implementation Checklist
    /// - Always verify the internal tree topology:
    /// `Parent(children(u)[0]) = Parent(children(u)[1]) = u`.
    /// - Do not forget lazy propagation: always call `push_down` before `pull_up`.
    /// - Ensure handles are updated immediately.
    use std::{
        hint::unreachable_unchecked,
        marker::PhantomData,
        num::NonZeroU32,
        ops::{Index, IndexMut},
    };

    use node::BinaryNode;

    pub const UNSET: u32 = !0;

    pub trait ClusterCx: Sized {
        // Vertex weight.
        type V: Default;

        /// Path cluster (aggregation on a subchain), represented as an **open** interval.
        type C;

        /// Point cluster (aggregation of light edges), represented as a **left-open, right-closed** interval.
        type R;

        /// Compress monoid.
        /// Left side is always the top side.
        fn id_compress() -> Self::C;

        fn compress(&self, children: [&Self::C; 2], v: &Self::V, rake: Option<&Self::R>)
            -> Self::C;

        /// Rake monoid, commutative.
        fn id_rake() -> Self::R;

        fn rake(&self, lhs: &Self::R, rhs: &Self::R) -> Self::R;

        /// Enclose the right end of a path cluster with a vertex.
        fn collapse_path(&self, c: &Self::C, vr: &Self::V) -> Self::R;

        /// Lazy propagation (implement it yourself)
        #[allow(unused_variables)]
        fn push_down_compress(
            &self,
            node: &mut Self::C,
            children: [&mut Self::C; 2],
            v: &mut Self::V,
            rake: Option<&mut Self::R>,
        ) {
        }

        #[allow(unused_variables)]
        fn push_down_rake(&self, node: &mut Self::R, children: [&mut Self::R; 2]) {}

        #[allow(unused_variables)]
        fn push_down_collapsed(&self, node: &mut Self::R, c: &mut Self::C, vr: &mut Self::V) {}

        fn reverse(&self, c: &Self::C) -> Self::C;
    }

    #[derive(Debug, Copy, Clone)]
    pub enum ActionRange {
        Subtree,
        Path,
    }

    /// Lazy propagation (Implement it yourself, Part II)
    pub trait Action<Cx: ClusterCx> {
        fn apply_to_compress(&mut self, compress: &mut Cx::C, range: ActionRange);
        fn apply_to_rake(&mut self, rake: &mut Cx::R);
        fn apply_to_weight(&mut self, weight: &mut Cx::V);
    }

    pub struct NodeRef<T> {
        idx: NonZeroU32,
        _phantom: std::marker::PhantomData<*mut T>,
    }

    #[derive(Debug, Clone)]
    pub struct Pool<T> {
        pub nodes: Vec<T>,
        pub free: Vec<NodeRef<T>>,
    }

    impl<T> Index<NodeRef<T>> for Pool<T> {
        type Output = T;
        fn index(&self, index: NodeRef<T>) -> &Self::Output {
            &self.nodes[index.idx.get() as usize]
        }
    }

    impl<T> IndexMut<NodeRef<T>> for Pool<T> {
        fn index_mut(&mut self, index: NodeRef<T>) -> &mut Self::Output {
            &mut self.nodes[index.idx.get() as usize]
        }
    }

    impl<T> Pool<T> {
        pub unsafe fn many_mut<'a, const N: usize>(
            &'a mut self,
            indices: [NodeRef<T>; N],
        ) -> [&'a mut T; N] {
            let ptr = self.nodes.as_mut_ptr();
            indices.map(|i| &mut *ptr.add(i.idx.get() as usize))
        }
    }

    impl<T> NodeRef<T> {
        pub fn new(idx: u32) -> Self {
            Self {
                idx: NonZeroU32::new(idx).unwrap(),
                _phantom: Default::default(),
            }
        }

        pub unsafe fn dangling() -> Self {
            Self {
                idx: NonZeroU32::new(UNSET).unwrap(),
                _phantom: PhantomData,
            }
        }
    }

    impl<T> Clone for NodeRef<T> {
        fn clone(&self) -> Self {
            Self {
                idx: self.idx,
                _phantom: Default::default(),
            }
        }
    }

    impl<T> Copy for NodeRef<T> {}

    impl<T> PartialEq for NodeRef<T> {
        fn eq(&self, other: &Self) -> bool {
            self.idx == other.idx
        }
    }

    impl<T> Eq for NodeRef<T> {}

    impl<T> std::fmt::Debug for NodeRef<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.idx.get())
        }
    }

    pub mod node {
        use std::fmt::Debug;

        use crate::debug;

        use super::*;

        pub enum Parent<Cx: ClusterCx> {
            Compress(NodeRef<Compress<Cx>>),
            Rake(NodeRef<Rake<Cx>>),
        }

        impl<Cx: ClusterCx> Clone for Parent<Cx> {
            fn clone(&self) -> Self {
                match self {
                    Parent::Compress(c) => Parent::Compress(*c),
                    Parent::Rake(r) => Parent::Rake(*r),
                }
            }
        }

        impl<Cx: ClusterCx> Copy for Parent<Cx> {}

        impl<Cx: ClusterCx> PartialEq for Parent<Cx> {
            fn eq(&self, other: &Self) -> bool {
                match (self, other) {
                    (Parent::Compress(c1), Parent::Compress(c2)) if c1 == c2 => true,
                    (Parent::Rake(r1), Parent::Rake(r2)) if r1 == r2 => true,
                    _ => false,
                }
            }
        }

        impl<Cx: ClusterCx> Eq for Parent<Cx> {}

        #[derive(Debug)]
        pub struct CompressPivot<Cx: ClusterCx> {
            pub children: [NodeRef<Compress<Cx>>; 2],
            pub rake_tree: Option<NodeRef<Rake<Cx>>>,

            pub rev_lazy: bool,
        }

        #[derive(Debug)]
        pub struct Compress<Cx: ClusterCx> {
            pub ends: [u32; 2],

            pub parent: Option<Parent<Cx>>,
            pub pivot: Option<CompressPivot<Cx>>,

            pub sum: Cx::C,

            pub debug_inorder: debug::Label<debug::tree::Pretty>,
        }

        #[derive(Debug)]
        pub struct Rake<Cx: ClusterCx> {
            pub parent: Parent<Cx>,
            pub children: Result<[NodeRef<Rake<Cx>>; 2], NodeRef<Compress<Cx>>>,

            pub sum: Cx::R,

            pub debug_inorder: debug::Label<debug::tree::Pretty>,
        }

        impl<Cx: ClusterCx> CompressPivot<Cx> {
            pub unsafe fn uninit() -> Self {
                Self {
                    children: [NodeRef::dangling(); 2],
                    rake_tree: None,
                    rev_lazy: false,
                }
            }
        }

        impl<Cx: ClusterCx> Debug for Parent<Cx> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    Parent::Compress(c) => write!(f, "{c:?}"),
                    Parent::Rake(r) => write!(f, "{r:?}"),
                }
            }
        }

        pub trait BinaryNode: Sized {
            type Parent: Copy;

            unsafe fn uninit() -> Self;

            fn internal_parent(&self) -> Option<NodeRef<Self>>;
            fn parent_mut(&mut self) -> &mut Self::Parent;
            fn is_internal_root(&self) -> bool {
                self.internal_parent().is_none()
            }

            fn children(&self) -> Option<&[NodeRef<Self>; 2]>;
            fn children_mut(&mut self) -> Option<&mut [NodeRef<Self>; 2]>;

            fn debug_inorder(&mut self) -> &mut debug::Label<debug::tree::Pretty>;
        }

        impl<Cx: ClusterCx> BinaryNode for Compress<Cx> {
            type Parent = Option<Parent<Cx>>;

            unsafe fn uninit() -> Self {
                Self {
                    ends: [UNSET; 2],

                    parent: None,
                    pivot: None,

                    sum: Cx::id_compress(),

                    debug_inorder: debug::Label::new_with(|| {
                        debug::tree::Pretty("uninit".into(), vec![])
                    }),
                }
            }

            fn internal_parent(&self) -> Option<NodeRef<Self>> {
                match self.parent {
                    Some(Parent::Compress(c)) => Some(c),
                    _ => None,
                }
            }

            fn parent_mut(&mut self) -> &mut Self::Parent {
                &mut self.parent
            }

            fn children(&self) -> Option<&[NodeRef<Self>; 2]> {
                Some(&self.pivot.as_ref()?.children)
            }

            fn children_mut(&mut self) -> Option<&mut [NodeRef<Self>; 2]> {
                Some(&mut self.pivot.as_mut()?.children)
            }

            fn debug_inorder(&mut self) -> &mut debug::Label<debug::tree::Pretty> {
                &mut self.debug_inorder
            }
        }

        impl<Cx: ClusterCx> Compress<Cx> {
            pub fn reverse(&mut self, cx: &Cx) {
                cx.reverse(&mut self.sum);
                self.ends.swap(0, 1);
                self.children_mut().map(|cs| cs.swap(0, 1));
                self.pivot.as_mut().map(|pivot| pivot.rev_lazy ^= true);
            }
        }

        impl<Cx: ClusterCx> BinaryNode for Rake<Cx> {
            type Parent = Parent<Cx>;

            unsafe fn uninit() -> Self {
                Self {
                    parent: Parent::Compress(NodeRef::dangling()),
                    children: Err(NodeRef::dangling()),
                    sum: Cx::id_rake(),

                    debug_inorder: debug::Label::new_with(|| {
                        debug::tree::Pretty("uninit".into(), vec![])
                    }),
                }
            }

            fn internal_parent(&self) -> Option<NodeRef<Self>> {
                match self.parent {
                    Parent::Rake(r) => Some(r),
                    _ => None,
                }
            }

            fn parent_mut(&mut self) -> &mut Self::Parent {
                &mut self.parent
            }

            fn children(&self) -> Option<&[NodeRef<Self>; 2]> {
                self.children.as_ref().ok()
            }

            fn children_mut(&mut self) -> Option<&mut [NodeRef<Self>; 2]> {
                self.children.as_mut().ok()
            }

            fn debug_inorder(&mut self) -> &mut debug::Label<debug::tree::Pretty> {
                &mut self.debug_inorder
            }
        }
    }

    impl<T: BinaryNode> Default for Pool<T> {
        fn default() -> Self {
            Self {
                nodes: vec![unsafe { node::BinaryNode::uninit() }],
                free: vec![],
            }
        }
    }

    impl<T: BinaryNode> Pool<T> {
        fn alloc(&mut self, node: T) -> NodeRef<T> {
            let u = if let Some(u) = self.free.pop() {
                self[u] = node;
                u
            } else {
                let idx = self.nodes.len() as u32;
                self.nodes.push(node);
                NodeRef::new(idx)
            };

            self[u].debug_inorder().with(|p| p.0 = format!("{u:?}"));

            u
        }

        unsafe fn free(&mut self, u: NodeRef<T>) -> T {
            self.free.push(u);
            todo!()
        }
    }

    pub struct TopTree<Cx: ClusterCx> {
        pub cx: Cx,

        /// Compress tree
        pub cs: Pool<node::Compress<Cx>>,
        /// Rake tree
        pub rs: Pool<node::Rake<Cx>>,

        /// Vertex info
        pub n_verts: usize,
        pub weights: Vec<Cx::V>,

        /// `handle(v)` is the only compress-node that requires vertex information. There are three cases:
        /// 1. If `degree(v)` = 0 (isolated vertex), then `handle(v)` = null.
        /// 2. If `degree(v)` = 1 (boundary vertex), then `handle(v)` = [topmost root compress-node].
        /// 3. If `degree(v)` ≥ 2 (interior vertex), then `handle(v)` = [compress-node with v as the pivot].

        /// Inversely, each compress-node interacts with at most three vertices:
        /// - Non-root leaf node: 0 vertices.
        /// - Non-root branch node: 1 vertex (compression pivot).
        /// - Collapsed root node: 1 or 2 vertices (right end with an optional compression pivot).
        /// - True root node: 2 or 3 vertices (both ends with an optional compression pivot).
        pub handle: Vec<Option<NodeRef<node::Compress<Cx>>>>,
    }

    /// Splay tree structure for compress and rake trees.
    /// Handles operations where the node topology of compress and rake trees does not swizzle.
    pub trait InternalSplay<T: BinaryNode> {
        fn pool(&mut self) -> &mut Pool<T>;

        fn push_down(&mut self, u: NodeRef<T>);
        fn pull_up(&mut self, u: NodeRef<T>);

        fn branch(&mut self, u: NodeRef<T>) -> Option<(NodeRef<T>, usize)> {
            let pool = self.pool();
            let p = pool[u].internal_parent()?;
            let branch = unsafe { pool[p].children().unwrap_unchecked() }[1] == u;
            Some((p, branch as usize))
        }

        /// Converts `p ->(left child) u ->(right child) c` to `u ->(right child) p ->(left child) c`.
        /// (If `p ->(right child) u`, flip (left child) <-> (right child).)
        ///
        /// ## Constraints
        /// 1. u must be a non-root branch.
        /// 2. push_down and pull_up for (g?), p, and u must be called beforehand.
        ///
        /// ## Diagram
        /// ┌────────┐     ┌───────┐
        /// │      g?│     │    g? │
        /// │     /  │     │   /   │
        /// │    p   │     │  u    │
        /// │   / \  │ ==> │ / \   │
        /// │  u   4 │     │0   p  │
        /// │ / \    │     │   / \ │
        /// │0   c   │     │  c   4│
        /// └────────┘     └───────┘
        unsafe fn rotate(&mut self, u: NodeRef<T>) {
            let (p, bp) = self.branch(u).unwrap_unchecked();
            let c = std::mem::replace(
                &mut self.pool()[u].children_mut().unwrap_unchecked()[bp ^ 1],
                p,
            );
            self.pool()[p].children_mut().unwrap_unchecked()[bp] = c;

            if let Some((g, bg)) = self.branch(p) {
                self.pool()[g].children_mut().unwrap_unchecked()[bg as usize] = u;
            }
            let pp = *self.pool()[p].parent_mut();
            *self.pool()[p].parent_mut() = *self.pool()[c].parent_mut();
            *self.pool()[c].parent_mut() = *self.pool()[u].parent_mut();
            *self.pool()[u].parent_mut() = pp;
        }

        /// Drag `u` up under the guard node. If `u` is a leaf, drag `parent(u)` if it exists.
        /// (An internal tree should be a full binary tree, so the leaves cannot be splayed.)"
        ///
        /// ## Diagram
        /// Step zig-zig:
        /// ┌─────────┐     ┌─────────┐
        /// │      g  │     │  u      │
        /// │     / \ │     │ / \     │
        /// │    p   6│     │0   p    │
        /// │   / \   │ ==> │   / \   │
        /// │  u   4  │     │  2   g  │
        /// │ / \     │     │     / \ │
        /// │0   2    │     │    4   6│
        /// └─────────┘     └─────────┘
        ///
        /// Step zig-zag:
        /// ┌───────┐
        /// │    g  │     ┌───────────┐
        /// │   / \ │     │     u     │
        /// │  p   6│     │   /   \   │
        /// │ / \   │ ==> │  p     g  │
        /// │0   u  │     │ / \   / \ │
        /// │   / \ │     │0   2 4   6│
        /// │   2  4│     └───────────┘
        /// └───────┘
        ///
        /// Step zig:
        /// ┌────────┐     ┌───────┐
        /// │    p   │     │  u    │
        /// │   / \  │     │ / \   │
        /// │  u   4 │ ==> │0   p  │
        /// │ / \    │     │   / \ │
        /// │0   c   │     │  c   4│
        /// └────────┘     └───────┘
        unsafe fn guarded_splay(&mut self, mut u: NodeRef<T>, guard: Option<NodeRef<T>>) {
            if self.pool()[u].children().is_none() {
                if let Some(p) = self.pool()[u].internal_parent() {
                    u = p;
                } else {
                    return;
                }
            }

            unsafe {
                while let Some(p) = self.pool()[u]
                    .internal_parent()
                    .filter(|&p| Some(p) != guard)
                {
                    if let Some(g) = self.pool()[p]
                        .internal_parent()
                        .filter(|&g| Some(g) != guard)
                    {
                        self.push_down(g);
                        self.push_down(p);
                        self.push_down(u);

                        let (_, bp) = self.branch(u).unwrap_unchecked();
                        let (_, bg) = self.branch(p).unwrap_unchecked();
                        if bp == bg {
                            self.rotate(p); // zig-zig
                        } else {
                            self.rotate(u); // zig-zag
                        }
                        self.rotate(u);

                        self.pull_up(g);
                        self.pull_up(p);
                        self.pull_up(u);
                    } else {
                        self.push_down(p);
                        self.push_down(u);

                        self.rotate(u); // zig

                        self.pull_up(p);
                        self.pull_up(u);
                    }
                }
            }
        }

        /// Drag `u` up to the root. If `u` is a leaf, drag `parent(u)` if it exists.
        ///
        /// Splaying `handle(u)` should always make it the root in the compressed tree.
        /// However, splaying the rake-node doesn't guarantee this.
        fn splay(&mut self, u: NodeRef<T>) {
            unsafe { self.guarded_splay(u, None) };
        }

        unsafe fn splay_prev(&mut self, root: NodeRef<T>) -> Option<NodeRef<T>> {
            debug_assert!(self.pool()[root].internal_parent().is_none());

            let mut u = root;
            u = self.pool()[u].children()?[0];
            while let Some(cs) = self.pool()[u].children() {
                u = cs[0];
            }

            self.splay(u);
            Some(u)
        }

        unsafe fn splay_last(&mut self, root: NodeRef<T>) -> NodeRef<T> {
            debug_assert!(self.pool()[root].internal_parent().is_none());

            let mut u = root;
            while let Some(cs) = self.pool()[u].children() {
                u = cs[1];
            }

            self.splay(u);
            u
        }

        fn inorder(&mut self, root: NodeRef<T>, visitor: &mut impl FnMut(&mut Self, NodeRef<T>)) {
            if let Some(cs) = self.pool()[root].children().copied() {
                self.inorder(cs[0], visitor);
                visitor(self, root);
                self.inorder(cs[1], visitor);
            } else {
                visitor(self, root);
            }
        }
    }

    impl<Cx: ClusterCx> InternalSplay<node::Compress<Cx>> for TopTree<Cx> {
        fn pool(&mut self) -> &mut Pool<node::Compress<Cx>> {
            &mut self.cs
        }

        fn push_down(&mut self, u: NodeRef<node::Compress<Cx>>) {
            // crate::debug::with(|| println!("call push_down({u:?})"));
            unsafe {
                if let Some(pivot) = &mut self.cs[u].pivot {
                    let rev_lazy = std::mem::take(&mut pivot.rev_lazy);
                    let rake_tree = pivot.rake_tree.map(|r| &mut self.rs[r].sum);
                    let [l, r] = pivot.children;

                    let [u, l, r] = self.cs.many_mut([u, l, r]);

                    if rev_lazy {
                        l.reverse(&self.cx);
                        r.reverse(&self.cx);
                    }

                    debug_assert_eq!(l.ends[1], r.ends[0]);
                    let vert = l.ends[1];

                    self.cx.push_down_compress(
                        &mut u.sum,
                        [&mut l.sum, &mut r.sum],
                        &mut self.weights[vert as usize],
                        rake_tree,
                    );
                }
            }
        }

        fn pull_up(&mut self, u: NodeRef<node::Compress<Cx>>) {
            // crate::debug::with(|| println!("call pull_up({u:?})"));
            unsafe {
                {
                    let ends = self.cs[u].ends;
                    self.cs[u]
                        .debug_inorder()
                        .with(|p| p.0 = format!("{u:?}  ends:{:?}", ends));
                }

                if let Some(pivot) = &self.cs[u].pivot {
                    let rake_tree = pivot.rake_tree;
                    let rake_tree_sum = rake_tree.map(|r| &self.rs[r].sum);

                    let [l, r] = pivot.children;
                    let [u_mut, l, r] = self.cs.many_mut([u, l, r]);

                    debug_assert_eq!(l.ends[1], r.ends[0], "u={u:?} {:?} {:?}", l.ends, r.ends);
                    let vert = l.ends[1];
                    u_mut.ends = [l.ends[0], r.ends[1]];
                    self.handle[vert as usize] = Some(u);

                    u_mut.sum = self.cx.compress(
                        [&l.sum, &r.sum],
                        &mut self.weights[vert as usize],
                        rake_tree_sum,
                    );

                    {
                        let ends = u_mut.ends;
                        u_mut
                            .debug_inorder()
                            .get_mut()
                            .zip(l.debug_inorder().get_mut())
                            .zip(r.debug_inorder().get_mut())
                            .map(|((p, l), r)| {
                                p.0 = format!("{u:?}- ends:{:?}", ends);
                                if let Some(rake_tree) = rake_tree {
                                    p.0 = format!("{u:?}-(r{rake_tree:?})  ends:{ends:?}");
                                }
                                p.1 = vec![l.clone(), r.clone()];
                            });
                    }
                }

                match self.cs[u].parent {
                    Some(node::Parent::Compress(_)) => {}
                    Some(node::Parent::Rake(_)) => {
                        self.handle[self.cs[u].ends[1] as usize] = Some(u);
                    }
                    None => {
                        self.handle[self.cs[u].ends[0] as usize] = Some(u);
                        self.handle[self.cs[u].ends[1] as usize] = Some(u);
                    }
                }
            }
        }
    }

    impl<Cx: ClusterCx> InternalSplay<node::Rake<Cx>> for TopTree<Cx> {
        fn pool(&mut self) -> &mut Pool<node::Rake<Cx>> {
            &mut self.rs
        }

        fn push_down(&mut self, u: NodeRef<node::Rake<Cx>>) {
            unsafe {
                match self.rs[u].children {
                    Ok([l, r]) => {
                        let [u, l, r] = self.rs.many_mut([u, l, r]);
                        self.cx.push_down_rake(&mut u.sum, [&mut l.sum, &mut r.sum]);
                    }
                    Err(compress_tree) => {
                        let compress_tree = &mut self.cs[compress_tree];
                        self.cx.push_down_collapsed(
                            &mut self.rs[u].sum,
                            &mut compress_tree.sum,
                            &mut self.weights[compress_tree.ends[1] as usize],
                        );
                    }
                }
            }
        }

        fn pull_up(&mut self, u: NodeRef<node::Rake<Cx>>) {
            unsafe {
                match self.rs[u].children {
                    Ok([l, r]) => {
                        let [u_mut, l, r] = self.rs.many_mut([u, l, r]);
                        u_mut.sum = self.cx.rake(&u_mut.sum, &self.cx.rake(&l.sum, &r.sum));

                        u_mut
                            .debug_inorder()
                            .get_mut()
                            .zip(l.debug_inorder().get_mut())
                            .zip(r.debug_inorder().get_mut())
                            .map(|((p, l), r)| {
                                p.0 = format!("{u:?}");
                                p.1 = vec![l.clone(), r.clone()];
                            });
                    }
                    Err(compress_tree) => {
                        self.rs[u].sum = self.cx.collapse_path(
                            &self.cs[compress_tree].sum,
                            &self.weights[self.cs[compress_tree].ends[1] as usize],
                        );

                        self.rs[u].debug_inorder().with(|p| {
                            p.0 = format!("{u:?}-(c{compress_tree:?})");
                            p.1 = vec![];
                        });
                    }
                }
            }
        }
    }

    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub enum SoftExposeType {
        Vertex,
        NonEmpty,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct NoEdgeError;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DisconnError;

    impl<Cx: ClusterCx> TopTree<Cx> {
        pub fn new(weights: impl IntoIterator<Item = Cx::V>, cx: Cx) -> Self {
            let weights: Vec<_> = weights.into_iter().collect();
            let n_verts = weights.len();
            Self {
                cx,

                cs: Pool::default(),
                rs: Pool::default(),

                n_verts,
                weights,
                handle: vec![None; n_verts],
            }
        }

        /// Change the partition of the paths.
        ///
        /// ## Diagram
        ///
        /// ### Represented tree
        /// Swap the right-side chain `(c1)` with the raked chain `(cu)`, along the
        /// pivot vertex `v`.
        /// ┌─────────────────┐     ┌─────────────────┐
        /// │v0-(c0)-v-(c1)-v1│     │v0-(c0)-v-(cu)-vu│
        /// │        .        │     │        .        │
        /// │        .        │     │        .        │
        /// │       (cu)      │ ==> │       (c1)      │
        /// │        |        │     │        |        │
        /// │        vu       │     │        v1       │
        /// └─────────────────┘     └─────────────────┘
        ///
        /// ### Internal tree
        /// Splay the collapsed rake-node(`ru`), and then swap the rhs of
        /// the given compress-node(`c1`) with a collapsed rake-node(`cu`).
        /// Since the splay operation handles only branch nodes, we need a multiple
        /// splay pass to bring a leaf closest to the root, with [distance to the rake root] = 0, 1, or 2.
        ///
        /// - Case dist = 1. `parent(ru)` is the root of the rake tree.
        /// ┌───────┐     ┌────────┐
        /// │  cv   │     │  cv    │
        /// │ /| \  │     │ /| \   │
        /// │? rp c1│     │? rp cu │
        /// │ /  \  │     │ /  \   │
        /// │?    ru│ ==> │?    ru │
        /// │     | │     │     |  │
        /// │     cu│     │     c1 │
        /// └───────┘     └────────┘
        ///
        /// ## Constraints
        /// `[cu, ..r_path, cv]` must form an unward path.
        /// Here are the explicit verifications:
        /// ```
        /// debug_assert!(self.cs[cu].parent == Some(node::Parent::Rake(r_path[0])));
        /// debug_assert!(
        ///     (1..N - 1).all(|i| self.rs[r_path[i - 1]].parent == node::Parent::Rake(r_path[i])),
        /// );
        /// debug_assert!(self.rs[r_path[N - 1]].parent == node::Parent::Compress(cv));
        /// ```
        unsafe fn splice<const N: usize>(
            &mut self,
            guard: Option<NodeRef<node::Compress<Cx>>>,
            cu: &mut NodeRef<node::Compress<Cx>>,
            r_path: [NodeRef<node::Rake<Cx>>; N],
            cv: NodeRef<node::Compress<Cx>>,
        ) {
            assert!(1 <= N && N <= 3);

            self.guarded_splay(cv, guard);

            // Flip guard if necessary, ensuring it's not spliced off from the root path.
            if let Some(g) = guard {
                self.push_down(g);
                if self.branch(cv) == Some((g, 0)) {
                    debug_assert!(self.cs[g].is_internal_root());

                    self.cs[g].reverse(&self.cx);
                    self.push_down(g);
                    debug_assert!(self.branch(cv) == Some((g, 1)));
                }
            }

            self.push_down(cv);
            for &r in r_path.iter().rev() {
                self.push_down(r);
            }

            // Swap path
            let ru = r_path[0];
            let c1 = self.cs[cv].pivot.as_ref().unwrap_unchecked().children[1];

            self.rs[ru].children = Err(c1);
            self.cs[cv].pivot.as_mut().unwrap_unchecked().children[1] = *cu;
            self.cs[*cu].parent = Some(node::Parent::Compress(cv));
            self.cs[c1].parent = Some(node::Parent::Rake(ru));

            // Unnecessary (only for subsequent pull_up, should be optimized)
            self.push_down(c1);
            // Update handle[c1.ends[1]] to cu
            self.pull_up(c1);

            for &r in &r_path {
                self.pull_up(r);
            }
            self.pull_up(cv);

            *cu = cv;
        }

        /// Connect all chains between the root and u, then splay handle(u).
        /// This is equivalent to the access operation in a link-cut tree.
        /// If there is a guard (which should be a root internal node), ensure that it is not
        /// spliced off by appropriately flipping the path.
        pub unsafe fn guarded_access(
            &mut self,
            u: usize,
            guard: Option<NodeRef<node::Compress<Cx>>>,
        ) {
            unsafe {
                let Some(mut cu) = self.handle[u] else {
                    // If u is an isolated vertex, do nothing.
                    return;
                };

                self.guarded_splay(cu, guard);

                while let Some(node::Parent::Rake(ru)) = self.cs[cu].parent {
                    let rp_old = self.rs[ru].internal_parent();
                    self.splay(ru);
                    if let Some(rp) = self.rs[ru].internal_parent() {
                        self.guarded_splay(rp, rp_old);
                    }

                    match self.rs[ru].parent {
                        node::Parent::Compress(cv) => self.splice(guard, &mut cu, [ru], cv),
                        node::Parent::Rake(rp1) => match self.rs[rp1].parent {
                            node::Parent::Compress(cv) => {
                                self.splice(guard, &mut cu, [ru, rp1], cv)
                            }
                            node::Parent::Rake(rp2) => match self.rs[rp2].parent {
                                node::Parent::Compress(cv) => {
                                    self.splice(guard, &mut cu, [ru, rp1, rp2], cv)
                                }
                                node::Parent::Rake(_) => unreachable_unchecked(),
                            },
                        },
                    }

                    self.guarded_splay(cu, guard);
                }

                self.guarded_splay(self.handle[u].unwrap_unchecked(), guard);
            }
        }

        pub fn access(&mut self, u: usize) {
            unsafe { self.guarded_access(u, None) };
        }

        /// Set `handle(u)` as the internal root,
        /// ensuring that the root path includes both `u` and `v` (if they are connected).
        /// Flip the root path if necessary to position `u` on the left side of `v`.
        /// - If either `u` or `v` is a boundary vertex, set `handle(u) = handle(v) = [internal root]`.
        /// - If both `u` and `v` are interior vertices, restructure the tree as
        ///   `[internal root] = handle(u) ->(right child) handle(v) ->(left child) path(u ~ v)`.
        pub fn soft_expose(&mut self, u: usize, v: usize) -> Result<SoftExposeType, DisconnError> {
            unsafe {
                crate::debug::with(|| println!("soft_expose {u} {v}"));
                self.access(u);
                if u == v {
                    return Ok(SoftExposeType::Vertex);
                }

                if self.handle[u].is_none() || self.handle[v].is_none() {
                    self.access(v);
                    return Err(DisconnError);
                };

                let hu = self.handle[u].unwrap_unchecked();
                if u as u32 == self.cs[hu].ends[1] || v as u32 == self.cs[hu].ends[0] {
                    self.cs[hu].reverse(&self.cx);
                }

                if u as u32 == self.cs[hu].ends[0] {
                    self.access(v);

                    let hv = self.handle[v].unwrap_unchecked();
                    if u as u32 != self.cs[hv].ends[0] {
                        return Err(DisconnError);
                    }
                } else {
                    self.guarded_access(v, self.handle[u]);
                    self.push_down(self.handle[u].unwrap_unchecked());
                    self.pull_up(self.handle[u].unwrap_unchecked());

                    let hu = self.handle[u].unwrap_unchecked();
                    let hv = self.handle[v].unwrap_unchecked();
                    if hu == hv {
                        return Ok(SoftExposeType::NonEmpty);
                    }

                    match self.branch(hv) {
                        Some((p, 0)) if p == hu => self.cs[hu].reverse(&self.cx),
                        Some((p, 1)) if p == hu => {}
                        _ => return Err(DisconnError),
                    }
                }
                Ok(SoftExposeType::NonEmpty)
            }
        }

        pub fn is_connected(&mut self, u: usize, v: usize) -> bool {
            self.soft_expose(u, v).is_ok()
        }

        unsafe fn link_left_end(&mut self, u: usize, ce: &mut NodeRef<node::Compress<Cx>>) {
            debug_assert!(u == self.cs[*ce].ends[0] as usize);

            if let Some(cu) = self.handle[u] {
                debug_assert!(self.cs[cu].parent == None);

                if u as u32 == self.cs[cu].ends[0] {
                    self.cs[cu].reverse(&self.cx);
                }

                if u as u32 == self.cs[cu].ends[1] {
                    // Case 1. `u` is a boundary vertex.
                    // Insert the new edge to the right end of the path.
                    let cp = self.cs.alloc(node::Compress {
                        pivot: Some(node::CompressPivot {
                            children: [cu, *ce],
                            ..unsafe { node::CompressPivot::uninit() }
                        }),
                        ..unsafe { node::Compress::uninit() }
                    });

                    self.cs[cu].parent = Some(node::Parent::Compress(cp));
                    self.cs[*ce].parent = Some(node::Parent::Compress(cp));
                    self.pull_up(cp);

                    *ce = cp;
                } else {
                    // Case 2. `u` is an interior vertex.
                    // Take out the right path `c1` replaced with the new edge `ce`, then push it into the
                    // rake tree.
                    self.push_down(cu);

                    fn lifetime_hint<A, B, F: for<'a> Fn(&'a mut A) -> &'a mut B>(f: F) -> F {
                        f
                    }
                    let pivot = lifetime_hint(|this: &mut Self| {
                        this.cs[cu].pivot.as_mut().unwrap_unchecked()
                    });

                    let c1 = std::mem::replace(&mut pivot(self).children[1], *ce);
                    self.cs[*ce].parent = Some(node::Parent::Compress(cu));

                    let r1 = self.rs.alloc(node::Rake {
                        children: Err(c1),
                        ..unsafe { node::Rake::uninit() }
                    });
                    self.cs[c1].parent = Some(node::Parent::Rake(r1));
                    self.push_down(c1);
                    self.pull_up(c1); // Update handle[c1.ends[1]]
                    self.pull_up(r1);

                    if let Some(r0) = pivot(self).rake_tree {
                        let rp = self.rs.alloc(node::Rake {
                            children: Ok([r0, r1]),
                            ..unsafe { node::Rake::uninit() }
                        });
                        self.rs[r0].parent = node::Parent::Rake(rp);
                        self.rs[r1].parent = node::Parent::Rake(rp);
                        self.pull_up(rp);

                        pivot(self).rake_tree = Some(rp);
                        self.rs[rp].parent = node::Parent::Compress(cu);
                        self.pull_up(cu);
                    } else {
                        pivot(self).rake_tree = Some(r1);
                        self.rs[r1].parent = node::Parent::Compress(cu);
                        self.pull_up(cu);
                    }

                    *ce = cu;
                }
            } else {
                // Case 0. `u` is an isolated vertex.
                self.handle[u] = Some(*ce);
            }
        }

        pub fn link(&mut self, u: usize, v: usize, e: Cx::C) -> bool {
            if self.soft_expose(u, v).is_ok() {
                return false;
            }

            crate::debug::with(|| println!("link {u} {v}"));

            let mut ce = self.cs.alloc(node::Compress {
                ends: [u as u32, v as u32],
                sum: e,
                ..unsafe { node::Compress::uninit() }
            });

            unsafe {
                let hu_old = self.handle[u];
                self.cs[ce].reverse(&self.cx);
                self.link_left_end(v, &mut ce);
                self.cs[ce].reverse(&self.cx);
                self.handle[u] = hu_old;

                self.link_left_end(u, &mut ce);
            }

            true
        }

        pub fn cut(&mut self, u: usize, v: usize) -> Result<Cx::C, NoEdgeError> {
            if self.soft_expose(u, v) != Ok(SoftExposeType::NonEmpty) {
                return Err(NoEdgeError);
            }

            let hu = self.handle[u].unwrap();
            let hv = self.handle[v].unwrap();
            let ends = self.cs[hu].ends;
            self.push_down(hu);
            self.push_down(hv);

            // free?

            match (ends[0] as usize == u, ends[1] as usize == v) {
                (true, true) => {
                    let h_inner = hu;
                    if self.cs[h_inner].children().is_some() {
                        return Err(NoEdgeError);
                    }

                    let c = std::mem::replace(&mut self.cs[h_inner].sum, Cx::id_compress());
                    self.handle[u] = None;
                    self.handle[v] = None;
                    unsafe { self.pool().free(h_inner) };
                    Ok(c)
                }
                (true, false) => {
                    let h_inner = self.cs[hu].pivot.as_ref().unwrap().children[0];
                    if self.cs[h_inner].children().is_some() {
                        return Err(NoEdgeError);
                    }
                    todo!()
                }
                (false, true) => {
                    let h_inner = self.cs[hu].pivot.as_ref().unwrap().children[1];
                    if self.cs[h_inner].children().is_some() {
                        return Err(NoEdgeError);
                    }
                    todo!()
                }
                (false, false) => {
                    self.push_down(self.handle[v].unwrap());
                    let h_inner = self.cs[hv].pivot.as_ref().unwrap().children[0];
                    if self.cs[h_inner].children().is_some() {
                        return Err(NoEdgeError);
                    }
                    todo!()
                }
            }
        }

        pub fn reroot(&mut self, u: usize) {
            todo!()
        }

        pub fn modify_vertex(&mut self, u: usize, update_with: impl FnOnce(&mut Cx::V)) {
            self.access(u);
            if let Some(hu) = self.handle[u] {
                self.push_down(hu);
                update_with(&mut self.weights[u]);
                self.pull_up(hu);
            } else {
                update_with(&mut self.weights[u])
            }
        }

        pub fn modify_edge(
            &mut self,
            u: usize,
            v: usize,
            update_with: impl FnOnce(&mut Cx::C),
        ) -> Result<(), NoEdgeError> {
            let mut w = self.cut(u, v)?;
            update_with(&mut w);
            self.link(u, v, w);
            Ok(())
        }

        pub fn sum_path(
            &mut self,
            u: usize,
            v: usize,
        ) -> Result<(&Cx::V, Option<(&Cx::C, &Cx::V)>), DisconnError> {
            let pseudo_rake = match self.soft_expose(u, v)? {
                SoftExposeType::Vertex => None,
                SoftExposeType::NonEmpty => {
                    let hu = self.handle[u].unwrap();
                    let hv = self.handle[v].unwrap();
                    let ends = self.cs[hu].ends;

                    let h_inner = match (ends[0] as usize == u, ends[1] as usize == v) {
                        (true, true) => hu,
                        (true, false) => {
                            self.push_down(hu);
                            self.cs[hu].pivot.as_ref().unwrap().children[0]
                        }
                        (false, true) => {
                            self.push_down(hu);
                            self.cs[hu].pivot.as_ref().unwrap().children[1]
                        }
                        (false, false) => {
                            self.push_down(hu);
                            self.push_down(hv);
                            self.cs[hv].pivot.as_ref().unwrap().children[0]
                        }
                    };
                    Some((&self.cs[h_inner].sum, &self.weights[v]))
                }
            };
            Ok((&self.weights[u], pseudo_rake))
        }

        pub fn apply_path(
            &mut self,
            u: usize,
            v: usize,
            mut action: impl Action<Cx>,
        ) -> Result<(), DisconnError> {
            match self.soft_expose(u, v)? {
                SoftExposeType::Vertex => {
                    action.apply_to_weight(&mut self.weights[u]);
                }
                SoftExposeType::NonEmpty => {
                    action.apply_to_weight(&mut self.weights[u]);
                    action.apply_to_weight(&mut self.weights[v]);

                    let hu = self.handle[u].unwrap();
                    let hv = self.handle[v].unwrap();
                    let ends = self.cs[hu].ends;

                    let mut update_with = |this: &mut Self, h_inner| {
                        action.apply_to_compress(&mut this.cs[h_inner].sum, ActionRange::Path);
                    };
                    match (ends[0] as usize == u, ends[1] as usize == v) {
                        (true, true) => update_with(self, hu),
                        (true, false) => {
                            self.push_down(hu);
                            update_with(self, self.cs[hu].pivot.as_ref().unwrap().children[0]);
                            self.pull_up(hu);
                        }
                        (false, true) => {
                            self.push_down(hu);
                            update_with(self, self.cs[hu].pivot.as_ref().unwrap().children[1]);
                            self.pull_up(hu);
                        }
                        (false, false) => {
                            self.push_down(hu);
                            self.push_down(hv);
                            update_with(self, self.cs[hv].pivot.as_ref().unwrap().children[0]);
                            self.pull_up(hv);
                            self.pull_up(hu);
                        }
                    };
                }
            };
            Ok(())
        }

        // pub fn sum_subtree(&mut self, u: usize, v: usize) -> (&Cx::V, &Cx::R) {
        //     todo!()
        // }

        // pub fn apply_subtree(&mut self, u: usize, p: usize, action: impl Action<Cx>) -> bool {
        //     todo!()
        // }

        pub fn debug_cons_chain(
            &mut self,
            nodes: impl IntoIterator<Item = Cx::C>,
        ) -> Option<NodeRef<node::Compress<Cx>>> {
            let mut nodes = nodes.into_iter().enumerate();

            let (u, c) = nodes.next()?;
            let mut lhs = self.cs.alloc(node::Compress {
                ends: [u as u32, u as u32 + 1],
                sum: c,
                ..unsafe { node::Compress::uninit() }
            });

            for (u, c) in nodes {
                let rhs = self.cs.alloc(node::Compress {
                    ends: [u as u32, u as u32 + 1],
                    sum: c,
                    ..unsafe { node::Compress::uninit() }
                });

                let c = self.cs.alloc(node::Compress {
                    pivot: Some(node::CompressPivot {
                        children: [lhs, rhs],
                        rake_tree: None,
                        rev_lazy: false,
                    }),
                    ..unsafe { node::Compress::uninit() }
                });

                self.cs[lhs].parent = Some(node::Parent::Compress(c));
                self.cs[rhs].parent = Some(node::Parent::Compress(c));
                self.pull_up(c);
                lhs = c;
            }

            Some(lhs)
        }

        pub fn debug_pretty_compress(
            &self,
            u: NodeRef<node::Compress<Cx>>,
        ) -> crate::debug::tree::Pretty {
            let mut children = vec![];
            if let Some(pivot) = &self.cs[u].pivot {
                let [c0, c1] = pivot.children;
                children.push(self.debug_pretty_compress(c0));
                if let Some(r) = pivot.rake_tree {
                    children.push(self.debug_pretty_rake(r));
                }
                children.push(self.debug_pretty_compress(c1));
            }
            crate::debug::tree::Pretty(format!("c{u:?}  ends: {:?}", self.cs[u].ends), children)
        }

        pub fn debug_pretty_rake(&self, u: NodeRef<node::Rake<Cx>>) -> crate::debug::tree::Pretty {
            let mut children = vec![];
            match self.rs[u].children {
                Ok(rs) => {
                    let [r0, r1] = rs;
                    children.push(self.debug_pretty_rake(r0));
                    children.push(self.debug_pretty_rake(r1));
                }
                Err(c) => children.push(self.debug_pretty_compress(c)),
            };
            crate::debug::tree::Pretty(format!("r{u:?}"), children)
        }
    }

    ///// A morphism of clusters, with common weight type V. There are two main use cases:
    ///// - Supporting both path and subtree sum queries.
    /////   Path sums do not propagate from rake trees, whereas subtree sums do.
    /////   Therefore, we need to store both path and subtree aggregates in your clusters,
    /////   and the projection helps reduce computation time efficiently for each sum query.
    ///// - Nesting another data structure within nodes (e.g., sets, segment trees, ropes, ... ).
    /////   The user has control over querying a specific node before performing the summation.
    ///// Some set of combinators are provided: identity, tuple and path-sum.
    /////
    ///// TODO: modify most of the sum_ functions to accept an additional reducer.
    //pub trait Reducer<Dom: ClusterCx> {
    //    type Co: ClusterCx<V = Dom::V>;
    //    fn co(&self) -> &Self::Co;
    //    fn map_compress(
    //        &self,
    //        c: &<Dom as ClusterCx>::Compress,
    //    ) -> <Self::Co as ClusterCx>::Compress;
    //    fn map_rake(&self, r: &<Dom as ClusterCx>::Rake) -> <Self::Co as ClusterCx>::Rake;
    //}

    ///// An identity.
    /////
    ///// # Examples
    /////
    ///// ```
    ///// let cx = || { ... }
    ///// let mut stt = StaticTopTree::from_edges(n, edges, root, cx());
    ///// ...
    ///// let total_sum = stt.sum_all(Id(cx()));
    ///// ```
    //pub struct Id<Cx>(pub Cx);
    //impl<Cx: ClusterCx> Reducer<Cx> for Id<Cx> {
    //    type Co = Cx;
    //    fn co(&self) -> &Self::Co {
    //        &self.0
    //    }
    //    fn map_compress(&self, c: &<Cx as ClusterCx>::Compress) -> <Cx as ClusterCx>::Compress {
    //        c.clone()
    //    }
    //    fn map_rake(&self, r: &<Cx as ClusterCx>::Rake) -> <Cx as ClusterCx>::Rake {
    //        r.clone()
    //    }
    //}
}
