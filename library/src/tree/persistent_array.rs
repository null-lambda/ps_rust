pub mod persistent {
    use std::ops::{Index, Range};

    pub type Rc<T> = std::rc::Rc<T>;

    #[derive(Clone)]
    enum Node<T> {
        Leaf(T),
        Branch([Rc<Node<T>>; 2]),
    }

    #[derive(Clone)]
    pub struct Array<T> {
        n: usize,
        root: Option<Rc<Node<T>>>,
    }

    impl<T: Clone> Index<usize> for Array<T> {
        type Output = T;
        fn index(&self, idx: usize) -> &T {
            debug_assert!(idx < self.n);
            let mut node = self.root.as_ref().unwrap();
            let [mut start, mut end] = [0, self.n];
            loop {
                match node.as_ref() {
                    Node::Leaf(value) => return value,
                    Node::Branch([lhs, rhs]) => {
                        let mid = (start + end) >> 1;
                        if idx < mid {
                            node = lhs;
                            end = mid;
                        } else {
                            node = rhs;
                            start = mid;
                        }
                    }
                }
            }
        }
    }

    impl<T: Clone> Array<T> {
        pub fn from_iter(mut iter: impl ExactSizeIterator<Item = T>) -> Self {
            let n = iter.len();
            Self {
                n,
                root: (n > 0).then(|| Self::build_rec(&mut iter, 0..n)),
            }
        }

        fn build_rec(iter: &mut impl Iterator<Item = T>, view: Range<usize>) -> Rc<Node<T>> {
            debug_assert!(view.start < view.end);
            if view.start + 1 == view.end {
                return Rc::new(Node::Leaf(iter.next().unwrap()));
            }
            let mid = view.start + view.end >> 1;
            let lhs = Self::build_rec(iter, view.start..mid);
            let rhs = Self::build_rec(iter, mid..view.end);
            Rc::new(Node::Branch([lhs, rhs]))
        }

        pub fn modify<S>(&mut self, idx: usize, update_with: impl FnOnce(&mut T) -> S) -> S {
            debug_assert!(idx < self.n);

            let root = self.root.as_mut().unwrap();
            Self::modify_rec(Rc::make_mut(root), idx, update_with, 0..self.n)
        }

        fn modify_rec<S>(
            node: &mut Node<T>,
            idx: usize,
            update_with: impl FnOnce(&mut T) -> S,
            view: Range<usize>,
        ) -> S {
            match node {
                Node::Leaf(value) => update_with(value),
                Node::Branch([lhs, rhs]) => {
                    let mid = view.start + view.end >> 1;
                    let res = if idx < mid {
                        Self::modify_rec(Rc::make_mut(lhs), idx, update_with, view.start..mid)
                    } else {
                        Self::modify_rec(Rc::make_mut(rhs), idx, update_with, mid..view.end)
                    };

                    res
                }
            }
        }

        pub fn set(&mut self, idx: usize, value: T) {
            self.modify(idx, |x| *x = value);
        }
    }
}
