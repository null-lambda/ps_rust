pub mod persistent_leftist_heap {
    // Persistent meldable heap

    pub type Rc<T> = std::rc::Rc<T>;
    // pub type Rc<T> = crate::rc_acyclic::Rc<T>;

    type Link<K> = Option<Rc<Node<K>>>;

    #[derive(Clone)]
    struct Node<K> {
        value: K,
        rank: u32, // Distance to the shallowest leaf
        children: [Link<K>; 2],
    }

    #[derive(Clone)]
    pub struct LeftistHeap<K> {
        root: Link<K>,
    }

    impl<K> Default for LeftistHeap<K> {
        fn default() -> Self {
            Self { root: None }
        }
    }

    impl<K: Ord + Clone> LeftistHeap<K> {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn singleton(value: K) -> Self {
            Self {
                root: Some(Rc::new(Node {
                    value,
                    rank: 1,
                    children: Default::default(),
                })),
            }
        }

        fn merge_nonnull(mut l: Rc<Node<K>>, mut r: Rc<Node<K>>) -> Rc<Node<K>> {
            if l.value < r.value {
                std::mem::swap(&mut l, &mut r);
            }

            let l_inner = Rc::make_mut(&mut l);
            let [x, y] = &mut l_inner.children;
            *y = Self::merge_inner(y.take(), Some(r));

            let rank_x = x.as_ref().map_or(0, |u| u.rank);
            let rank_y = y.as_ref().map_or(0, |u| u.rank);
            l_inner.rank = rank_x.min(rank_y) + 1;
            if rank_x < rank_y {
                std::mem::swap(x, y);
            }

            l
        }

        fn merge_inner(l: Link<K>, r: Link<K>) -> Link<K> {
            match (l, r) {
                (None, node) | (node, None) => node,
                (Some(l), Some(r)) => Some(Self::merge_nonnull(l, r)),
            }
        }

        pub fn merge(self, other: Self) -> Self {
            Self {
                root: Self::merge_inner(self.root, other.root),
            }
        }

        pub fn merge_with(&mut self, other: Self) {
            self.root = Self::merge_inner(self.root.take(), other.root);
        }

        pub fn push(&mut self, value: K) {
            self.merge_with(Self::singleton(value));
        }

        pub fn peek(&self) -> Option<&K> {
            Some(&self.root.as_ref()?.value)
        }

        pub fn pop(&mut self) -> Option<K> {
            let root = Rc::unwrap_or_clone(self.root.take()?);
            let [c0, c1] = root.children;
            self.root = Self::merge_inner(c0, c1);
            Some(root.value)
        }
    }

    impl<K: Ord + Clone> FromIterator<K> for LeftistHeap<K> {
        fn from_iter<T: IntoIterator<Item = K>>(iter: T) -> Self {
            let mut acc = Self::default();
            for x in iter {
                acc.push(x);
            }
            acc
        }
    }

    impl<K: Ord + Clone + std::fmt::Debug> std::fmt::Debug for LeftistHeap<K> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let mut this = self.clone();
            f.debug_list()
                .entries(std::iter::from_fn(|| this.pop()))
                .finish()
        }
    }
}
