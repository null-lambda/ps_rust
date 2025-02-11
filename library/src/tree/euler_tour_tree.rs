pub mod splay {
    // Reversible rope, based on a splay tree.
    use std::{
        fmt::{self, Debug},
        mem::MaybeUninit,
        num::NonZeroU32,
    };

    // Adjoin an identity element to a binary operation.
    fn lift_binary<A>(
        combine: impl FnOnce(A, A) -> A,
    ) -> impl FnOnce(Option<A>, Option<A>) -> Option<A> {
        |lhs, rhs| match (lhs, rhs) {
            (Some(lhs), Some(rhs)) => Some(combine(lhs, rhs)),
            (None, rhs) => rhs,
            (lhs, None) => lhs,
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Branch {
        Left = 0,
        Right = 1,
    }

    impl Branch {
        pub fn usize(self) -> usize {
            self as usize
        }

        pub fn inv(&self) -> Self {
            match self {
                Branch::Left => Branch::Right,
                Branch::Right => Branch::Left,
            }
        }
    }

    // Intrusive node link, invertible.
    #[derive(Default, Debug)]
    pub struct Link {
        children: [Option<NodeRef>; 2],
        parent: Option<NodeRef>,
    }

    pub trait IntrusiveNode {
        fn link(&self) -> &Link;
        fn link_mut(&mut self) -> &mut Link;
    }

    pub trait NodeSpec: IntrusiveNode {
        fn push_down(&mut self, _children: [Option<&mut Self>; 2]) {}
        fn pull_up(&mut self, _children: [Option<&mut Self>; 2]) {}
        fn on_reverse(&mut self) {}

        // type Cx;
        // fn push_down(&mut self, cx: &mut Self::Cx, children: [Option<&mut Self>; 2]);
        // fn pull_up(&mut self, cx: &mut Self::Cx, children: [Option<&mut Self>; 2]);
    }

    pub trait SizedNode: NodeSpec {
        fn size(&self) -> usize;
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NodeRef {
        pub idx: NonZeroU32,
    }

    impl NodeRef {
        fn get(&self) -> usize {
            self.idx.get() as usize
        }
    }

    impl Debug for NodeRef {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.idx.get())
        }
    }

    #[derive(Debug)]
    pub struct SplayForest<V> {
        pub pool: Vec<MaybeUninit<V>>,
    }

    impl<V: NodeSpec> SplayForest<V> {
        pub fn new() -> Self {
            let dummy = MaybeUninit::uninit();
            Self { pool: vec![dummy] }
        }

        pub fn add_root(&mut self, node: V) -> NodeRef {
            let idx = self.pool.len();
            self.pool.push(MaybeUninit::new(node));
            NodeRef {
                idx: unsafe { NonZeroU32::new(idx as u32).unwrap_unchecked() },
            }
        }

        pub fn get<'a>(&'a self, u: NodeRef) -> &'a V {
            unsafe { &self.pool[u.get()].assume_init_ref() }
        }

        pub unsafe fn get_mut<'a>(&'a mut self, u: NodeRef) -> &'a mut V {
            self.pool[u.get()].assume_init_mut()
        }

        pub unsafe fn get_with_children<'a>(
            &'a mut self,
            u: NodeRef,
        ) -> (&'a mut V, [Option<&'a mut V>; 2]) {
            unsafe {
                let pool_ptr = self.pool.as_mut_ptr();
                let node = (&mut *pool_ptr.add(u.get())).assume_init_mut();
                let children = node.link().children.map(|child| {
                    child.map(|child| (&mut *pool_ptr.add(child.get())).assume_init_mut())
                });
                (node, children)
            }
        }

        pub fn with<T>(&mut self, u: NodeRef, f: impl FnOnce(&mut V) -> T) -> T {
            f(unsafe { self.get_mut(u) })
        }

        fn push_down(&mut self, u: NodeRef) {
            unsafe {
                let (node, children) = self.get_with_children(u);
                node.push_down(children);
            }
        }

        pub fn pull_up(&mut self, node: NodeRef) {
            unsafe {
                let (node, children) = self.get_with_children(node);
                node.pull_up(children);
            }
        }

        pub fn get_parent(&self, u: NodeRef) -> Option<(NodeRef, Branch)> {
            let p = self.get(u).link().parent?;
            if self.get(p).link().children[Branch::Left.usize()] == Some(u) {
                Some((p, Branch::Left))
            } else if self.get(p).link().children[Branch::Right.usize()] == Some(u) {
                Some((p, Branch::Right))
            } else {
                None
            }
        }

        pub fn is_root(&self, u: NodeRef) -> bool {
            self.get_parent(u).is_none()
        }

        fn attach(&mut self, u: NodeRef, child: NodeRef, branch: Branch) {
            debug_assert_ne!(u, child);
            unsafe {
                self.get_mut(u).link_mut().children[branch as usize] = Some(child);
                self.get_mut(child).link_mut().parent = Some(u);
            }
        }

        fn detach(&mut self, u: NodeRef, branch: Branch) -> Option<NodeRef> {
            unsafe {
                let child = self.get_mut(u).link_mut().children[branch as usize].take()?;
                self.get_mut(child).link_mut().parent = None;
                Some(child)
            }
        }

        fn rotate(&mut self, u: NodeRef) {
            let (parent, branch) = self.get_parent(u).expect("Root shouldn't be rotated");
            let child = self.detach(u, branch.inv());
            if let Some(child) = child {
                self.attach(parent, child, branch);
            } else {
                self.detach(parent, branch);
            }

            match self.get_parent(parent) {
                Some((grandparent, grandbranch)) => {
                    self.attach(grandparent, u, grandbranch);
                }
                None => unsafe {
                    self.get_mut(u).link_mut().parent = None;
                },
            }
            self.attach(u, parent, branch.inv());

            self.pull_up(parent);
            self.pull_up(u);
        }

        pub fn splay(&mut self, u: NodeRef) {
            while let Some((parent, branch)) = self.get_parent(u) {
                if let Some((grandparent, grandbranch)) = self.get_parent(parent) {
                    self.push_down(grandparent);
                    self.push_down(parent);
                    self.push_down(u);
                    if branch != grandbranch {
                        self.rotate(u);
                    } else {
                        self.rotate(parent);
                    }
                } else {
                    self.push_down(parent);
                    self.push_down(u);
                }
                self.rotate(u);
            }
            self.push_down(u);
        }

        // Caution: breaks amortized time complexity if not splayed afterwards.
        pub unsafe fn find_by(
            &mut self,
            mut u: NodeRef,
            mut next: impl FnMut(&Self, NodeRef) -> Option<Branch>,
        ) -> NodeRef {
            loop {
                self.push_down(u);
                if let Some(next) =
                    next(self, u).and_then(|branch| self.get(u).link().children[branch.usize()])
                {
                    u = next;
                } else {
                    break;
                }
            }
            u
        }

        // Caution: if u is not a root, then only the subtree nodes can be accessed.
        // Call splay(u) beforehand to walk on the full tree.
        pub fn splay_by(
            &mut self,
            u: &mut NodeRef,
            mut next: impl FnMut(&Self, NodeRef) -> Option<Branch>,
        ) {
            *u = unsafe { self.find_by(*u, &mut next) };
            self.splay(*u);
        }

        pub fn splay_first(&mut self, u: &mut NodeRef) {
            self.splay_by(u, |_, _| Some(Branch::Left))
        }

        pub fn splay_last(&mut self, u: &mut NodeRef) {
            self.splay_by(u, |_, _| Some(Branch::Right))
        }

        pub fn inorder(&mut self, u: NodeRef, visitor: &mut impl FnMut(&mut Self, NodeRef)) {
            self.push_down(u);
            if let Some(left) = self.get(u).link().children[Branch::Left.usize()] {
                self.inorder(left, visitor);
            }
            visitor(self, u);
            if let Some(right) = self.get(u).link().children[Branch::Right.usize()] {
                self.inorder(right, visitor);
            }
        }

        pub fn split_left(&mut self, u: NodeRef) -> Option<NodeRef> {
            self.splay(u);
            let left = self.detach(u, Branch::Left)?;
            self.pull_up(u);
            Some(left)
        }

        pub fn split_right(&mut self, u: NodeRef) -> Option<NodeRef> {
            self.splay(u);
            let right = self.detach(u, Branch::Right)?;
            self.pull_up(u);
            Some(right)
        }

        pub fn merge_nonnull(&mut self, mut lhs: NodeRef, mut rhs: NodeRef) -> NodeRef {
            self.splay(lhs);
            self.splay_last(&mut lhs);
            self.splay(rhs);
            self.splay_first(&mut rhs);
            debug_assert!(self.is_root(lhs) && self.is_root(rhs) && lhs != rhs);
            self.attach(rhs, lhs, Branch::Left);
            self.pull_up(rhs);
            rhs
        }

        pub fn merge(&mut self, lhs: Option<NodeRef>, rhs: Option<NodeRef>) -> Option<NodeRef> {
            lift_binary(|lhs, rhs| self.merge_nonnull(lhs, rhs))(lhs, rhs)
        }
    }

    impl<V> Drop for SplayForest<V> {
        fn drop(&mut self) {
            for node in self.pool.iter_mut().skip(1) {
                unsafe {
                    node.assume_init_drop();
                }
            }
        }
    }
}

pub mod euler_tour_tree {
    use std::collections::HashMap;

    use super::splay;
    // use super::wbtree;

    fn rotate_to_front<S: splay::NodeSpec>(forest: &mut splay::SplayForest<S>, u: splay::NodeRef) {
        forest.splay(u);
        let left = forest.split_left(u);
        forest.merge(Some(u), left);
    }

    pub struct DynamicEulerTour<S: splay::NodeSpec> {
        pub forest: splay::SplayForest<S>,
        pub freed: Vec<splay::NodeRef>,

        pub verts: Vec<splay::NodeRef>,
        pub edges: HashMap<(u32, u32), splay::NodeRef>,
        pub n_verts: usize,
    }

    impl<S: splay::NodeSpec> DynamicEulerTour<S> {
        pub fn new(vert_nodes: impl IntoIterator<Item = S>) -> Self {
            let mut this = Self {
                forest: splay::SplayForest::new(),
                freed: vec![], // Reused deleted edge nodes

                verts: vec![],
                edges: HashMap::new(),
                n_verts: 0,
            };
            for node in vert_nodes {
                let u = this.forest.add_root(node);
                this.verts.push(u);
                this.n_verts += 1;
            }
            this
        }

        pub fn add_root(&mut self, node: S) -> splay::NodeRef {
            if let Some(u) = self.freed.pop() {
                self.forest.with(u, |u| {
                    *u = node;
                });
                u
            } else {
                self.forest.add_root(node)
            }
        }

        pub fn reroot(&mut self, u: usize) {
            rotate_to_front(&mut self.forest, self.edges[&(u as u32, u as u32)]);
        }

        pub fn find_root(&mut self, u: usize) -> splay::NodeRef {
            let mut u = self.verts[u];
            self.forest.splay(u);
            self.forest.splay_first(&mut u);
            u
        }

        pub fn is_connected(&mut self, u: usize, v: usize) -> bool {
            self.find_root(u) == self.find_root(v)
        }

        pub fn link(&mut self, u: usize, v: usize, edge_uv: S, edge_vu: S) -> bool {
            if self.is_connected(u, v) {
                return false;
            }
            let vert_u = self.verts[u];
            let vert_v = self.verts[v];
            let edge_uv = self.add_root(edge_uv);
            let edge_vu = self.add_root(edge_vu);
            self.edges.insert((u as u32, v as u32), edge_uv);
            self.edges.insert((v as u32, u as u32), edge_vu);

            rotate_to_front(&mut self.forest, vert_u);
            rotate_to_front(&mut self.forest, vert_v);
            let lhs = self.forest.merge_nonnull(vert_u, edge_uv);
            let rhs = self.forest.merge_nonnull(vert_v, edge_vu);
            self.forest.merge_nonnull(lhs, rhs);
            true
        }

        pub fn cut(&mut self, u: usize, v: usize) -> bool {
            let (Some(edge_uv), Some(edge_vu)) = (
                self.edges.remove(&(u as u32, v as u32)),
                self.edges.remove(&(v as u32, u as u32)),
            ) else {
                return false;
            };

            rotate_to_front(&mut self.forest, edge_uv);
            self.forest.split_right(edge_uv);
            self.forest.split_left(edge_vu);
            self.forest.split_right(edge_vu);
            self.freed.push(edge_uv);
            self.freed.push(edge_vu);
            true
        }
    }
}
