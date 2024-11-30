mod collections {
    use std::cell::Cell;
    use std::mem::{self, MaybeUninit};

    pub struct DisjointMap<T> {
        // Represents parent if >= 0, size if < 0
        parent_or_size: Vec<Cell<i32>>,
        values: Vec<MaybeUninit<T>>,
    }

    impl<T> DisjointMap<T> {
        pub fn new(values: impl IntoIterator<Item = T>) -> Self {
            let node_weights: Vec<_> = values.into_iter().map(|c| MaybeUninit::new(c)).collect();
            let n = node_weights.len();
            Self {
                parent_or_size: vec![Cell::new(-1); n],
                values: node_weights,
            }
        }

        fn get_parent_or_size(&self, u: usize) -> Result<usize, u32> {
            let x = self.parent_or_size[u].get();
            if x >= 0 {
                Ok(x as usize)
            } else {
                Err((-x) as u32)
            }
        }

        fn set_parent(&self, u: usize, p: usize) {
            self.parent_or_size[u].set(p as i32);
        }

        fn set_size(&self, u: usize, s: u32) {
            self.parent_or_size[u].set(-(s as i32));
        }

        pub fn find_root_with_size(&self, u: usize) -> (usize, u32) {
            match self.get_parent_or_size(u) {
                Ok(p) => {
                    let (root, size) = self.find_root_with_size(p);
                    self.set_parent(u, root);
                    (root, size)
                }
                Err(size) => (u, size),
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            self.find_root_with_size(u).0
        }

        pub fn get_size(&self, u: usize) -> u32 {
            self.find_root_with_size(u).1
        }

        pub fn get_mut(&mut self, u: usize) -> &mut T {
            let r = self.find_root(u);
            unsafe { self.values[r].assume_init_mut() }
        }

        // Returns true if two sets were previously disjoint
        pub fn merge(
            &mut self,
            u: usize,
            v: usize,
            mut combine_values: impl FnMut(T, T) -> T,
        ) -> bool {
            let (mut u, size_u) = self.find_root_with_size(u);
            let (mut v, size_v) = self.find_root_with_size(v);
            if u == v {
                return false;
            }

            let value = unsafe {
                MaybeUninit::new(combine_values(
                    self.values[u].assume_init_read(),
                    self.values[v].assume_init_read(),
                ))
            };
            if size_u < size_v {
                mem::swap(&mut u, &mut v);
            }
            self.set_parent(v, u);
            self.set_size(u, size_u + size_v);
            self.values[u] = value;
            true
        }
    }

    impl<T> Drop for DisjointMap<T> {
        fn drop(&mut self) {
            for u in 0..self.parent_or_size.len() {
                if self.get_parent_or_size(u).is_err() {
                    unsafe {
                        self.values[u].assume_init_drop();
                    }
                }
            }
        }
    }
}
