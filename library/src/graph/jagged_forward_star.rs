pub mod jagged {
    pub type EdgeId = u32;
    const UNSET: EdgeId = u32::MAX;

    // Forward-star representation (linked lists) for incremental jagged array
    #[derive(Clone, PartialEq, Eq)]
    pub struct FS<T> {
        pub head: Vec<EdgeId>,
        pub links: Vec<(EdgeId, T)>,
    }

    pub struct RowIter<'a, T> {
        owner: &'a FS<T>,
        e: u32,
    }

    impl<'a, T> Iterator for RowIter<'a, T> {
        type Item = &'a T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.e == UNSET {
                return None;
            }
            let (e_next, value) = &self.owner.links[self.e as usize];
            self.e = *e_next;
            Some(value)
        }
    }

    impl<T> FS<T> {
        pub fn with_size(n: usize) -> Self {
            Self {
                head: vec![UNSET; n],
                links: vec![],
            }
        }

        pub fn insert(&mut self, u: usize, v: T) -> EdgeId {
            let e = self.links.len() as u32;
            self.links.push((self.head[u], v));
            self.head[u] = e;
            e
        }

        pub fn iter_row<'a>(&'a self, u: usize) -> RowIter<'a, T> {
            RowIter {
                owner: &self,
                e: self.head[u],
            }
        }
    }
}
