mod dset {
    use std::{cell::Cell, mem};

    #[derive(Clone)]
    pub struct DisjointSet {
        // Represents parent if >= 0, size if < 0
        link: Vec<Cell<i32>>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self {
                link: vec![Cell::new(-1); n],
            }
        }

        pub fn find_root_with_size(&self, u: usize) -> (usize, u32) {
            let p = self.link[u].get();
            if p >= 0 {
                let (root, size) = self.find_root_with_size(p as usize);
                self.link[u].set(root as i32);
                (root, size)
            } else {
                (u, (-p) as u32)
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            self.find_root_with_size(u).0
        }

        // Returns true iif two sets were previously disjoint
        pub fn merge(&mut self, u: usize, v: usize) -> bool {
            let (mut u, size_u) = self.find_root_with_size(u);
            let (mut v, size_v) = self.find_root_with_size(v);
            if u == v {
                return false;
            }

            if size_u < size_v {
                mem::swap(&mut u, &mut v);
            }
            self.link[v].set(u as i32);
            self.link[u].set(-((size_u + size_v) as i32));
            true
        }
    }
}
