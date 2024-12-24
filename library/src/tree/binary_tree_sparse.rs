mod binary_tree {
    #[derive(Debug)]
    struct Node<T> {
        value: T,
        left: Option<Box<Node<T>>>,
        right: Option<Box<Node<T>>>,
    }

    pub struct BinaryTree<T>(Option<Box<Node<T>>>);

    impl<T> BinaryTree<T> {
        pub fn traverse<F1, F2, F3>(&mut self, mut f_pre: F1, mut f_in: F2, mut f_post: F3)
        where
            F1: FnMut(&mut T),
            F2: FnMut(&mut T),
            F3: FnMut(&mut T),
        {
            fn inner<T, F1, F2, F3>(
                node: &mut Option<Box<Node<T>>>,
                f_pre: &mut F1,
                f_in: &mut F2,
                f_post: &mut F3,
            ) where
                F1: FnMut(&mut T),
                F2: FnMut(&mut T),
                F3: FnMut(&mut T),
            {
                if let Some(node) = node {
                    f_pre(&mut node.value);
                    inner(&mut node.left, f_pre, f_in, f_post);
                    f_in(&mut node.value);
                    inner(&mut node.right, f_pre, f_in, f_post);
                    f_post(&mut node.value);
                }
            }

            inner(&mut self.0, &mut f_pre, &mut f_in, &mut f_post);
        }
    }
}
