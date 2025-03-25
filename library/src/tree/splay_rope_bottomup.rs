pub mod splay {
    // Reversible rope, based on a splay tree.
    use std::{
        cmp::Ordering,
        fmt::{self, Debug},
        marker::PhantomData,
        mem::MaybeUninit,
        num::NonZeroU32,
        ops::{Index, IndexMut, Range},
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

    // Intrusive node link, invertible.
    #[derive(Default, Debug)]
    pub struct Link<V> {
        lazy_inv: bool,
        children: [Option<NodeRef<V>>; 2],
        parent: Option<NodeRef<V>>,
    }

    pub trait IntrusiveNode: Sized {
        fn link(&self) -> &Link<Self>;
        fn link_mut(&mut self) -> &mut Link<Self>;
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

    pub struct NodeRef<V> {
        pub idx: NonZeroU32,
        _phantom: PhantomData<*mut V>,
    }

    impl<V> Clone for NodeRef<V> {
        fn clone(&self) -> Self {
            Self {
                idx: self.idx,
                _phantom: PhantomData,
            }
        }
    }

    impl<V> Copy for NodeRef<V> {}

    impl<V> PartialEq for NodeRef<V> {
        fn eq(&self, other: &Self) -> bool {
            self.idx == other.idx
        }
    }

    impl<V> Eq for NodeRef<V> {}

    impl<V> NodeRef<V> {
        fn usize(&self) -> usize {
            self.idx.get() as usize
        }
    }

    impl<V> Debug for NodeRef<V> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.idx.get())
        }
    }

    #[derive(Debug)]
    pub struct RopePool<V> {
        pub nodes: Vec<MaybeUninit<V>>,
    }

    impl<V> Index<NodeRef<V>> for RopePool<V> {
        type Output = V;
        fn index(&self, index: NodeRef<V>) -> &Self::Output {
            unsafe { self.nodes[index.usize()].assume_init_ref() }
        }
    }

    impl<V> IndexMut<NodeRef<V>> for RopePool<V> {
        fn index_mut(&mut self, index: NodeRef<V>) -> &mut Self::Output {
            unsafe { self.nodes[index.usize()].assume_init_mut() }
        }
    }

    impl<V: NodeSpec> RopePool<V> {
        pub fn new() -> Self {
            let dummy = MaybeUninit::uninit();
            Self { nodes: vec![dummy] }
        }

        pub fn add_root(&mut self, node: V) -> NodeRef<V> {
            let idx = self.nodes.len();
            self.nodes.push(MaybeUninit::new(node));
            NodeRef {
                idx: unsafe { NonZeroU32::new(idx as u32).unwrap_unchecked() },
                _phantom: PhantomData,
            }
        }

        pub fn get_with_children<'a>(
            &'a mut self,
            u: NodeRef<V>,
        ) -> (&'a mut V, [Option<&'a mut V>; 2]) {
            unsafe {
                let pool_ptr = self.nodes.as_mut_ptr();
                let node = (&mut *pool_ptr.add(u.usize())).assume_init_mut();
                let children = node.link().children.map(|child| {
                    child.map(|child| (&mut *pool_ptr.add(child.usize())).assume_init_mut())
                });
                (node, children)
            }
        }

        pub fn reverse(&mut self, u: NodeRef<V>) {
            self[u].on_reverse();
            self[u].link_mut().lazy_inv ^= true;
        }

        fn push_down(&mut self, u: NodeRef<V>) {
            let link = self[u].link_mut();
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

        pub fn pull_up(&mut self, node: NodeRef<V>) {
            let (node, children) = self.get_with_children(node);
            node.pull_up(children);
        }

        pub fn branch(&self, u: NodeRef<V>) -> Option<(NodeRef<V>, Branch)> {
            let p = self[u].link().parent?;
            if self[p].link().children[Branch::Left.usize()] == Some(u) {
                Some((p, Branch::Left))
            } else if self[p].link().children[Branch::Right.usize()] == Some(u) {
                Some((p, Branch::Right))
            } else {
                None
            }
        }

        pub fn is_root(&self, u: NodeRef<V>) -> bool {
            self.branch(u).is_none()
        }

        pub fn attach(&mut self, u: NodeRef<V>, child: NodeRef<V>, branch: Branch) {
            debug_assert_ne!(u, child);
            self[u].link_mut().children[branch.usize()] = Some(child);
            self[child].link_mut().parent = Some(u);
        }

        pub fn detach(&mut self, u: NodeRef<V>, branch: Branch) -> Option<NodeRef<V>> {
            let child = self[u].link_mut().children[branch.usize()].take()?;
            self[child].link_mut().parent = None;
            Some(child)
        }

        fn rotate(&mut self, u: NodeRef<V>) {
            let (p, bp) = self.branch(u).expect("Root shouldn't be rotated");
            let c = self[u].link_mut().children[bp.inv().usize()].replace(p);
            self[p].link_mut().children[bp.usize()] = c;
            if let Some(c) = c {
                self[c].link_mut().parent = Some(p);
            }

            if let Some((g, bg)) = self.branch(p) {
                self[g].link_mut().children[bg.usize()] = Some(u);
            }

            self[u].link_mut().parent = self[p].link().parent;
            self[p].link_mut().parent = Some(u);
        }

        pub fn splay(&mut self, u: NodeRef<V>) {
            while let Some((p, bp)) = self.branch(u) {
                if let Some((g, bg)) = self.branch(p) {
                    self.push_down(g);
                    self.push_down(p);
                    self.push_down(u);

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
            self.push_down(u);
        }

        pub fn collect_from(&mut self, iter: impl IntoIterator<Item = V>) -> Option<NodeRef<V>> {
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

        fn walk_down_internal(
            &mut self,
            mut u: NodeRef<V>,
            mut next: impl FnMut(&Self, NodeRef<V>) -> Option<Branch>,
        ) -> NodeRef<V> {
            loop {
                self.push_down(u);
                if let Some(next) =
                    next(self, u).and_then(|branch| self[u].link().children[branch.usize()])
                {
                    u = next;
                } else {
                    break;
                }
            }
            u
        }

        /// # Caution
        /// if u is not a root, then only the subtree nodes can be accessed.
        /// Call splay(u) beforehand to walk on the full tree.
        pub fn splay_by(
            &mut self,
            u: &mut NodeRef<V>,
            mut next: impl FnMut(&Self, NodeRef<V>) -> Option<Branch>,
        ) {
            *u = self.walk_down_internal(*u, &mut next);
            self.splay(*u);
        }

        pub fn splay_first(&mut self, u: &mut NodeRef<V>) {
            self.splay_by(u, |_, _| Some(Branch::Left))
        }

        pub fn splay_last(&mut self, u: &mut NodeRef<V>) {
            self.splay_by(u, |_, _| Some(Branch::Right))
        }

        pub fn push(&mut self, root: &mut NodeRef<V>, node: V) {
            self.splay_last(root);
            let new = self.add_root(node);
            self.attach(*root, new, Branch::Right);
            self.pull_up(*root);
        }

        pub fn prev(&mut self, mut u: NodeRef<V>) -> Option<NodeRef<V>> {
            self.splay(u);
            u = self[u].link().children[Branch::Left.usize()]?;
            self.splay_last(&mut u);
            Some(u)
        }

        pub fn inorder(&mut self, u: NodeRef<V>, visitor: &mut impl FnMut(&mut Self, NodeRef<V>)) {
            self.push_down(u);
            if let Some(left) = self[u].link().children[Branch::Left.usize()] {
                self.inorder(left, visitor);
            }
            visitor(self, u);
            if let Some(right) = self[u].link().children[Branch::Right.usize()] {
                self.inorder(right, visitor);
            }
        }

        pub fn split_left(&mut self, u: NodeRef<V>) -> Option<NodeRef<V>> {
            self.splay(u);
            let left = self.detach(u, Branch::Left)?;
            self.pull_up(u);
            Some(left)
        }

        pub fn split_right(&mut self, u: NodeRef<V>) -> Option<NodeRef<V>> {
            self.splay(u);
            let right = self.detach(u, Branch::Right)?;
            self.pull_up(u);
            Some(right)
        }

        pub fn merge_nonnull(&mut self, mut lhs: NodeRef<V>, mut rhs: NodeRef<V>) -> NodeRef<V> {
            self.splay(lhs);
            self.splay_last(&mut lhs);
            self.splay(rhs);
            self.splay_first(&mut rhs);
            debug_assert!(self.is_root(lhs) && self.is_root(rhs) && lhs != rhs);
            self.attach(rhs, lhs, Branch::Left);
            self.pull_up(rhs);
            rhs
        }

        pub fn merge(
            &mut self,
            lhs: Option<NodeRef<V>>,
            rhs: Option<NodeRef<V>>,
        ) -> Option<NodeRef<V>> {
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => Some((|lhs, rhs| self.merge_nonnull(lhs, rhs))(lhs, rhs)),
                (None, rhs) => rhs,
                (lhs, None) => lhs,
            }
        }

        pub fn remove(&mut self, u: NodeRef<V>) -> Option<NodeRef<V>> {
            self.splay(u);
            let left = self.detach(u, Branch::Left);
            let right = self.detach(u, Branch::Right);
            self.pull_up(u);

            self.merge(left, right)
        }
    }

    impl<V> Drop for RopePool<V> {
        fn drop(&mut self) {
            for node in self.nodes.iter_mut().skip(1) {
                unsafe {
                    node.assume_init_drop();
                }
            }
        }
    }

    impl<V: SizedNode> RopePool<V> {
        pub fn splay_nth(&mut self, u: &mut NodeRef<V>, mut idx: usize) {
            debug_assert!(idx < self[*u].size());
            self.splay_by(u, |forest, u| {
                let left_size = forest[u].link().children[Branch::Left.usize()]
                    .map_or(0, |left| forest[left].size());
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

        pub fn position(&mut self, u: NodeRef<V>) -> usize {
            self.splay(u);
            self[u].link().children[Branch::Left.usize()].map_or(0, |left| self[left].size())
        }

        pub fn split_at(
            &mut self,
            mut u: NodeRef<V>,
            idx: usize,
        ) -> (Option<NodeRef<V>>, Option<NodeRef<V>>) {
            debug_assert!(idx <= self[u].size());
            if idx == self[u].size() {
                return (Some(u), None);
            } else {
                self.splay_nth(&mut u, idx);
                let left = self.split_left(u);
                (left, Some(u))
            }
        }

        pub fn insert_at(&mut self, root: &mut NodeRef<V>, idx: usize, u: NodeRef<V>) {
            debug_assert!(idx <= self[*root].size());
            let (lhs, rhs) = self.split_at(*root, idx);
            let mid = self.merge(lhs, Some(u));
            *root = self.merge(mid, rhs).unwrap()
        }

        pub fn with_range(
            &mut self,
            root: &mut NodeRef<V>,
            range: Range<usize>,
            f: impl FnOnce(&mut Self, NodeRef<V>),
        ) {
            assert!(range.start < range.end && range.end <= self[*root].size());
            let (rest, rhs) = self.split_at(*root, range.end);
            let (lhs, mid) = self.split_at(rest.unwrap(), range.start);
            self.splay(unsafe { mid.unwrap_unchecked() });
            f(self, unsafe { mid.unwrap_unchecked() });
            self.merge(lhs, mid);
            *root = self.merge(rest, rhs).unwrap();
        }
    }
}
