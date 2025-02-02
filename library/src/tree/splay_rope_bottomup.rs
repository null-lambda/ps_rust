pub mod splay {
    // Reversible rope, based on a splay tree.
    use std::{
        cmp::Ordering,
        fmt::{self, Debug},
        mem::MaybeUninit,
        num::NonZeroU32,
        ops::Range,
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
        lazy_inv: bool,
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

        pub fn reverse(&mut self, u: NodeRef) {
            unsafe {
                let node = self.get_mut(u);
                node.on_reverse();
                node.link_mut().lazy_inv ^= true;
            }
        }

        fn push_down(&mut self, u: NodeRef) {
            unsafe {
                let link = self.get_mut(u).link_mut();
                if link.lazy_inv {
                    link.lazy_inv = false;
                    link.children.swap(0, 1);
                    for child in link.children.into_iter().flatten() {
                        self.reverse(child);
                    }
                }

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

        pub fn attach(&mut self, u: NodeRef, child: NodeRef, branch: Branch) {
            debug_assert_ne!(u, child);
            unsafe {
                self.get_mut(u).link_mut().children[branch as usize] = Some(child);
                self.get_mut(child).link_mut().parent = Some(u);
            }
        }

        pub fn detach(&mut self, u: NodeRef, branch: Branch) -> Option<NodeRef> {
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

        pub fn collect_from(&mut self, iter: impl IntoIterator<Item = V>) -> Option<NodeRef> {
            let mut iter = iter.into_iter();
            let mut root = self.add_root(iter.next()?);
            for node in iter {
                let u = self.add_root(node);
                self.attach(u, root, Branch::Left);
                self.pull_up(u);
                root = u;
            }
            Some(root)
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

        pub fn push(&mut self, root: &mut NodeRef, node: V) {
            self.splay_last(root);
            let new = self.add_root(node);
            self.attach(*root, new, Branch::Right);
            self.pull_up(*root);
        }

        pub fn predecessor(&mut self, mut u: NodeRef) -> Option<NodeRef> {
            self.splay(u);
            u = self.get(u).link().children[Branch::Left.usize()]?;
            self.splay_last(&mut u);
            Some(u)
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

        pub fn remove(&mut self, u: NodeRef) -> Option<NodeRef> {
            self.splay(u);
            let left = self.detach(u, Branch::Left);
            let right = self.detach(u, Branch::Right);
            self.pull_up(u);

            self.merge(left, right)
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

    impl<V: SizedNode> SplayForest<V> {
        pub fn splay_nth(&mut self, u: &mut NodeRef, mut idx: usize) {
            debug_assert!(idx < self.get(*u).size());
            self.splay_by(u, |forest, u| {
                let left_size = forest.get(u).link().children[Branch::Left.usize()]
                    .map_or(0, |left| forest.get(left).size());
                match idx.cmp(&left_size) {
                    Ordering::Equal => None,
                    Ordering::Less => Some(Branch::Left),
                    Ordering::Greater => {
                        idx -= left_size as usize + 1;
                        Some(Branch::Right)
                    }
                }
            });
        }

        pub fn position(&mut self, u: NodeRef) -> usize {
            self.splay(u);
            self.get(u).link().children[Branch::Left.usize()]
                .map_or(0, |left| self.get(left).size())
        }

        pub fn split_at(
            &mut self,
            mut u: NodeRef,
            idx: usize,
        ) -> (Option<NodeRef>, Option<NodeRef>) {
            debug_assert!(idx <= self.get(u).size());
            if idx == self.get(u).size() {
                return (Some(u), None);
            } else {
                self.splay_nth(&mut u, idx);
                let left = self.split_left(u);
                (left, Some(u))
            }
        }

        pub fn insert_at(&mut self, root: &mut NodeRef, idx: usize, u: NodeRef) {
            debug_assert!(idx <= self.get(*root).size());
            let (lhs, rhs) = self.split_at(*root, idx);
            let mid = self.merge(lhs, Some(u));
            *root = self.merge(mid, rhs).unwrap()
        }

        pub fn with_range(
            &mut self,
            root: &mut NodeRef,
            range: Range<usize>,
            f: impl FnOnce(&mut Self, NodeRef),
        ) {
            assert!(range.start < range.end && range.end <= self.get(*root).size());
            let (rest, rhs) = self.split_at(*root, range.end);
            let (lhs, mid) = self.split_at(rest.unwrap(), range.start);
            self.splay(unsafe { mid.unwrap_unchecked() });
            f(self, unsafe { mid.unwrap_unchecked() });
            self.merge(lhs, mid);
            *root = self.merge(rest, rhs).unwrap();
        }
    }
}
