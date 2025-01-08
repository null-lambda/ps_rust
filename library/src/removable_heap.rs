pub mod heap {
    use std::collections::BinaryHeap;

    #[derive(Clone)]
    pub struct RemovableHeap<T> {
        items: BinaryHeap<T>,
        to_remove: BinaryHeap<T>,
    }

    impl<T: Ord> RemovableHeap<T> {
        pub fn new() -> Self {
            Self {
                items: BinaryHeap::new().into(),
                to_remove: BinaryHeap::new().into(),
            }
        }

        pub fn push(&mut self, item: T) {
            self.items.push(item);
        }

        pub fn remove(&mut self, item: T) {
            self.to_remove.push(item);
        }

        fn clean_top(&mut self) {
            while let Some((r, x)) = self.to_remove.peek().zip(self.items.peek()) {
                if r != x {
                    break;
                }
                self.to_remove.pop();
                self.items.pop();
            }
        }

        pub fn peek(&mut self) -> Option<&T> {
            self.clean_top();
            self.items.peek()
        }

        pub fn pop(&mut self) -> Option<T> {
            self.clean_top();
            self.items.pop()
        }
    }
}
