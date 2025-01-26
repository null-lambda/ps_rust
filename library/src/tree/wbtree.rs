pub mod wbtree {
    // Weight-balanced tree
    // https://koosaga.com/342
    // https://yoichihirai.com/bst.pdf

    use std::{cmp::Ordering, mem::MaybeUninit, num::NonZeroU32, ops::Range};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct NodeRef(NonZeroU32);

    impl NodeRef {
        pub fn get(self) -> usize {
            self.0.get() as usize
        }
    }

    #[derive(Default, Debug)]
    pub struct SizedNode<V> {
        pub size: u32,
        pub children: Option<[NodeRef; 2]>,
        pub inner: V,
    }

    pub trait NodeSpec: Sized + Default {
        fn push_down(forest: &mut WBForest<Self>, u: NodeRef);
        fn pull_up(forest: &mut WBForest<Self>, u: NodeRef);
    }

    pub trait SizedNodeSpec: NodeSpec {
        fn size(&self) -> u32;
        fn set_size(&mut self, size: u32);
    }

    pub trait MonoidReducer<V: NodeSpec> {
        type X: Clone;
        fn proj(forest: &WBForest<V>, u: NodeRef) -> Self::X;
        fn id() -> Self::X;
        fn combine(x: &Self::X, y: &Self::X) -> Self::X;
    }

    pub struct WBForest<V: NodeSpec> {
        pool: Vec<MaybeUninit<SizedNode<V>>>,
        free: Vec<NodeRef>,
    }

    impl<V: NodeSpec> WBForest<V> {
        pub fn new() -> Self {
            let dummy = MaybeUninit::uninit();
            Self {
                pool: vec![dummy],
                free: vec![],
            }
        }

        fn alloc(&mut self, node: SizedNode<V>) -> NodeRef {
            if let Some(u) = self.free.pop() {
                self.pool[u.get()] = MaybeUninit::new(node);
                return u;
            }
            let idx = self.pool.len() as u32;
            self.pool.push(MaybeUninit::new(node));
            unsafe { NodeRef(NonZeroU32::new_unchecked(idx)) }
        }

        unsafe fn free_unchecked(&mut self, u: NodeRef) {
            self.free.push(u);
        }

        pub fn new_leaf(&mut self, value: V) -> NodeRef {
            self.alloc(SizedNode {
                size: 1,
                children: None,
                inner: value,
            })
        }

        pub fn new_branch(&mut self, children: [NodeRef; 2]) -> NodeRef {
            let node = self.alloc(SizedNode {
                size: 0,             // Uninit
                inner: V::default(), // Uninit
                children: Some(children),
            });
            V::pull_up(self, node);
            node
        }

        pub fn get(&self, u: NodeRef) -> &SizedNode<V> {
            unsafe { self.pool[u.get()].assume_init_ref() }
        }

        // Mutable references are inherently unsafe, since NodeRef may alias easily.
        pub unsafe fn get_mut(&mut self, u: NodeRef) -> &mut SizedNode<V> {
            unsafe { self.pool[u.get()].assume_init_mut() }
        }

        pub unsafe fn get_many<const N: usize>(
            &mut self,
            us: [NodeRef; N],
        ) -> Option<[&mut SizedNode<V>; N]> {
            if cfg!(debug_assertions) {
                // Check for multiple aliases
                for i in 0..N {
                    for j in i + 1..N {
                        if us[i] == us[j] {
                            return None;
                        }
                    }
                }
            }

            let ptr = self.pool.as_mut_ptr();
            Some(us.map(|u| unsafe { (&mut *ptr.add(u.get())).assume_init_mut() }))
        }

        pub fn size(&self, u: NodeRef) -> u32 {
            self.get(u).size
        }
    }

    impl<S: NodeSpec> Drop for WBForest<S> {
        fn drop(&mut self) {
            for u in self.free.drain(..) {
                self.pool[u.get()] = MaybeUninit::new(Default::default());
            }
            for i in (1..self.pool.len()).rev() {
                unsafe {
                    self.pool[i].assume_init_drop();
                }
            }
        }
    }

    fn should_rotate(size_left: u32, size_right: u32) -> bool {
        (size_left + 1) * 3 < (size_right + 1)
    }

    fn should_rotate_once(size_left: u32, size_right: u32) -> bool {
        (size_left + 1) < (size_right + 1) * 2
    }

    fn is_balanced(size_left: u32, size_right: u32) -> bool {
        !should_rotate(size_left, size_right) && !should_rotate(size_right, size_left)
    }

    impl<V: NodeSpec> WBForest<V> {
        pub fn check_balance(&mut self, u: NodeRef) -> bool {
            if let Some([left, right]) = self.get(u).children {
                is_balanced(self.size(left), self.size(right))
                    && self.check_balance(left)
                    && self.check_balance(right)
            } else {
                true
            }
        }

        pub fn merge(&mut self, mut lhs: NodeRef, mut rhs: NodeRef) -> NodeRef {
            // self.new_branch([lhs, rhs]) // TODO

            unsafe {
                let take_children = |forest: &mut Self, u: NodeRef| {
                    V::push_down(forest, u);
                    let res = forest.get(u).children;
                    res
                };
                let update_children = |forest: &mut Self, u: NodeRef, children| {
                    forest.get_mut(u).children = children;
                    V::pull_up(forest, u);
                };

                if should_rotate(self.size(lhs), self.size(rhs)) {
                    let [mid, rhs_right] = take_children(self, rhs).unwrap_unchecked();
                    lhs = self.merge(lhs, mid);
                    if is_balanced(self.size(lhs), self.size(rhs_right)) {
                        update_children(self, rhs, Some([lhs, rhs_right]));
                        return rhs;
                    }
                    let [lhs_left, mid] = take_children(self, lhs).unwrap_unchecked();
                    if should_rotate_once(self.size(mid), self.size(lhs_left)) {
                        update_children(self, rhs, Some([mid, rhs_right]));
                        update_children(self, lhs, Some([lhs_left, rhs]));
                        return lhs;
                    }
                    let [mid_left, mid_right] = take_children(self, mid).unwrap_unchecked();
                    update_children(self, lhs, Some([lhs_left, mid_left]));
                    update_children(self, rhs, Some([mid_right, rhs_right]));
                    update_children(self, mid, Some([lhs, rhs]));
                    return mid;
                } else if should_rotate(self.size(rhs), self.size(lhs)) {
                    let [lhs_left, mid] = take_children(self, lhs).unwrap_unchecked();
                    rhs = self.merge(mid, rhs);
                    if is_balanced(self.size(lhs_left), self.size(rhs)) {
                        update_children(self, lhs, Some([lhs_left, rhs]));
                        return lhs;
                    }
                    let [mid, rhs_right] = take_children(self, rhs).unwrap_unchecked();
                    if should_rotate_once(self.size(mid), self.size(rhs_right)) {
                        update_children(self, lhs, Some([lhs_left, mid]));
                        update_children(self, rhs, Some([lhs, rhs_right]));
                        return rhs;
                    }
                    let [mid_left, mid_right] = take_children(self, mid).unwrap_unchecked();
                    update_children(self, lhs, Some([lhs_left, mid_left]));
                    update_children(self, rhs, Some([mid_right, rhs_right]));
                    update_children(self, mid, Some([lhs, rhs]));
                    return mid;
                } else {
                    self.new_branch([lhs, rhs])
                }
            }
        }

        fn split_inner(&mut self, u: NodeRef, pos: usize) -> (NodeRef, NodeRef) {
            debug_assert!(0 < pos && pos < self.size(u) as usize);
            unsafe {
                V::push_down(self, u);

                let [left, right] = self.get(u).children.unwrap_unchecked(); // size >= 2, so it must be a branch
                self.free_unchecked(u);

                let left_size = self.size(left) as usize;
                match pos.cmp(&left_size) {
                    Ordering::Equal => (left, right),
                    Ordering::Less => {
                        let (left, mid) = self.split_inner(left, pos);
                        let right = self.merge(mid, right);
                        (left, right)
                    }
                    Ordering::Greater => {
                        let (mid, right) = self.split_inner(right, pos - left_size);
                        let left = self.merge(left, mid);
                        (left, right)
                    }
                }
            }
        }

        pub fn split(&mut self, u: NodeRef, pos: usize) -> (Option<NodeRef>, Option<NodeRef>) {
            let n = self.size(u);
            debug_assert!(pos <= n as usize);
            if pos == 0 {
                (None, Some(u))
            } else if pos == n as usize {
                (Some(u), None)
            } else {
                let (left, right) = self.split_inner(u, pos);
                (Some(left), Some(right))
            }
        }

        pub fn query_range<R: MonoidReducer<V>>(
            &mut self,
            u: NodeRef,
            range: Range<usize>,
        ) -> R::X {
            self.query_range_rec::<R>(&range, 0..self.size(u) as usize, u)
        }

        fn query_range_rec<R: MonoidReducer<V>>(
            &mut self,
            query: &Range<usize>,
            view: Range<usize>,
            u: NodeRef,
        ) -> R::X {
            unsafe {
                if query.end <= view.start || view.end <= query.start {
                    R::id()
                } else if query.start <= view.start && view.end <= query.end {
                    V::push_down(self, u);
                    R::proj(self, u)
                } else {
                    V::push_down(self, u);
                    let [left, right] = self.get(u).children.unwrap_unchecked();
                    let mid = view.start + self.size(left) as usize;
                    R::combine(
                        &self.query_range_rec::<R>(query, view.start..mid, left),
                        &self.query_range_rec::<R>(query, mid..view.end, right),
                    )
                }
            }
        }

        pub fn collect_from(&mut self, xs: impl ExactSizeIterator<Item = V>) -> Option<NodeRef> {
            let n = xs.len();
            (n > 0).then(|| self.collect_from_rec(&mut xs.into_iter(), 0..n as u32))
        }

        fn collect_from_rec(
            &mut self,
            xs: &mut impl Iterator<Item = V>,
            range: Range<u32>,
        ) -> NodeRef {
            let Range { start, end } = range;
            debug_assert!(start != end);
            if start + 1 == end {
                self.new_leaf(xs.next().unwrap())
            } else {
                let mid = start + end >> 1;
                let left = self.collect_from_rec(xs, start..mid);
                let right = self.collect_from_rec(xs, mid..end);
                self.new_branch([left, right])
            }
        }
    }
}
