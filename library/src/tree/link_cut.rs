pub mod link_cut {
    use std::{
        fmt::{self, Debug},
        num::NonZeroU32,
    };

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

    // Intrusive node link
    #[derive(Default, Debug)]
    pub struct Link {
        pub inv: bool,
        pub children: [Option<NodeRef>; 2],
        pub parent: Option<NodeRef>,
    }

    pub trait IntrusiveNode {
        fn link(&self) -> &Link;
        fn link_mut(&mut self) -> &mut Link;
    }

    pub trait NodeSpec: IntrusiveNode {
        fn push_down(&mut self, _children: [Option<&mut Self>; 2]) {}
        fn pull_up(&mut self, _children: [Option<&mut Self>; 2]) {}

        // type Cx;
        // fn push_down(&mut self, cx: &mut Self::Cx, children: [Option<&mut Self>; 2]);
        // fn pull_up(&mut self, cx: &mut Self::Cx, children: [Option<&mut Self>; 2]);
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
    pub struct LinkCutForest<S> {
        pub pool: Vec<S>,
    }

    impl<S: NodeSpec> LinkCutForest<S> {
        pub fn new() -> Self {
            let dummy = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
            Self { pool: vec![dummy] }
        }

        pub fn add_root(&mut self, node: S) -> NodeRef {
            let idx = self.pool.len();
            self.pool.push(node);
            NodeRef {
                idx: unsafe { NonZeroU32::new(idx as u32).unwrap_unchecked() },
            }
        }

        pub fn get_node<'a>(&'a self, u: NodeRef) -> &'a S {
            &self.pool[u.get()]
        }

        pub unsafe fn get_node_mut<'a>(&'a mut self, u: NodeRef) -> &'a mut S {
            &mut self.pool[u.get()]
        }

        pub unsafe fn get_node_with_children<'a>(
            &'a mut self,
            u: NodeRef,
        ) -> (&'a mut S, [Option<&'a mut S>; 2]) {
            unsafe {
                let pool_ptr = self.pool.as_mut_ptr();
                let node = &mut *pool_ptr.add(u.get());
                let children = node
                    .link()
                    .children
                    .map(|child| child.map(|child| &mut *pool_ptr.add(child.get())));
                (node, children)
            }
        }

        fn push_down(&mut self, u: NodeRef) {
            unsafe {
                let node = self.get_node_mut(u);
                let link = node.link_mut();
                if link.inv {
                    link.inv = false;
                    link.children.swap(0, 1);
                    for child in link.children.into_iter().flatten() {
                        self.get_node_mut(child).link_mut().inv ^= true;
                    }
                }

                let (node, children) = self.get_node_with_children(u);
                node.push_down(children);
            }
        }

        fn pull_up(&mut self, node: NodeRef) {
            unsafe {
                let (node, children) = self.get_node_with_children(node);
                node.pull_up(children);
            }
        }

        pub fn get_internal_parent(
            &self,
            u: NodeRef,
        ) -> Result<(NodeRef, Branch), Option<NodeRef>> {
            match self.get_node(u).link().parent {
                Some(p) => {
                    if self.get_node(p).link().children[Branch::Left.usize()] == Some(u) {
                        Ok((p, Branch::Left)) // parent on a chain
                    } else if self.get_node(p).link().children[Branch::Right.usize()] == Some(u) {
                        Ok((p, Branch::Right)) // parent on a chain
                    } else {
                        Err(Some(p)) // path-parent
                    }
                }
                None => Err(None), // true root
            }
        }

        pub fn is_root(&self, u: NodeRef) -> bool {
            self.get_internal_parent(u).is_err()
        }

        fn attach(&mut self, u: NodeRef, child: NodeRef, branch: Branch) {
            debug_assert_ne!(u, child);
            unsafe {
                self.get_node_mut(u).link_mut().children[branch as usize] = Some(child);
                self.get_node_mut(child).link_mut().parent = Some(u);
            }
        }

        fn detach(&mut self, u: NodeRef, branch: Branch) -> Option<NodeRef> {
            unsafe {
                let child = self.get_node_mut(u).link_mut().children[branch as usize].take()?;
                self.get_node_mut(child).link_mut().parent = None;
                Some(child)
            }
        }

        fn rotate(&mut self, u: NodeRef) {
            let (parent, branch) = self
                .get_internal_parent(u)
                .expect("Root shouldn't be rotated");
            let child = self.detach(u, branch.inv());
            if let Some(child) = child {
                self.attach(parent, child, branch);
            } else {
                self.detach(parent, branch);
            }

            match self.get_internal_parent(parent) {
                Ok((grandparent, grandbranch)) => {
                    self.attach(grandparent, u, grandbranch);
                }
                Err(path_parent) => unsafe {
                    self.get_node_mut(u).link_mut().parent = path_parent;
                },
            }
            self.attach(u, parent, branch.inv());

            self.pull_up(parent);
            self.pull_up(u);
        }

        pub fn splay(&mut self, u: NodeRef) {
            while let Ok((parent, branch)) = self.get_internal_parent(u) {
                if let Ok((grandparent, grandbranch)) = self.get_internal_parent(parent) {
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

        pub fn access(&mut self, u: NodeRef) {
            unsafe {
                self.splay(u);
                self.get_node_mut(u).link_mut().children[Branch::Right.usize()] = None;
                while let Some(path_parent) = self.get_internal_parent(u).unwrap_err_unchecked() {
                    self.splay(path_parent);
                    self.get_node_mut(path_parent).link_mut().children[Branch::Right.usize()] =
                        Some(u);
                    self.splay(u);
                }
            }
        }

        pub fn reroot(&mut self, u: NodeRef) {
            self.access(u);
            unsafe { self.get_node_mut(u) }.link_mut().inv ^= true;
        }

        pub fn link(&mut self, parent: NodeRef, child: NodeRef) {
            self.reroot(child);
            self.link_root(parent, child);
        }

        pub fn link_root(&mut self, parent: NodeRef, child: NodeRef) {
            self.access(child);
            self.access(parent);
            self.attach(child, parent, Branch::Left);
            self.pull_up(child);
        }

        pub fn cut(&mut self, child: NodeRef) {
            self.access(child);
            if self.get_node(child).link().children[Branch::Left.usize()].is_some() {
                self.detach(child, Branch::Left);
                self.pull_up(child);
            }
        }

        pub fn find_root(&mut self, mut u: NodeRef) -> NodeRef {
            self.access(u);
            while let Some(left) = self.get_node(u).link().children[Branch::Left.usize()] {
                u = left;
                self.push_down(u);
            }
            self.splay(u);
            u
        }

        pub fn get_parent(&mut self, u: NodeRef) -> Option<NodeRef> {
            self.access(u);
            let mut left = self.get_node(u).link().children[Branch::Left.usize()]?;
            self.push_down(left);
            while let Some(right) = self.get_node(left).link().children[Branch::Right.usize()] {
                left = right;
                self.push_down(left);
            }
            self.splay(left);
            Some(left)
        }

        pub fn get_lca(&mut self, lhs: NodeRef, rhs: NodeRef) -> NodeRef {
            self.access(lhs);
            self.access(rhs);
            self.splay(lhs);
            self.get_node(lhs).link().parent.unwrap_or(lhs)
        }

        pub fn access_vertex_path(&mut self, path_top: NodeRef, path_bot: NodeRef) {
            self.reroot(path_top);
            self.access(path_bot);
            self.splay(path_top);
        }
    }
}
