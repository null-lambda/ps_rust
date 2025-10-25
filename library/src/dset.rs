mod dset {
    #[derive(Clone)]
    pub struct DisjointSet {
        // Represents parent if >= 0, size if < 0
        link: Vec<i32>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self { link: vec![-1; n] }
        }

        pub fn root_with_size(&mut self, u: u32) -> (u32, u32) {
            let p = self.link[u as usize];
            if p >= 0 {
                let (root, size) = self.root_with_size(p as u32);
                self.link[u as usize] = root as i32;
                (root, size)
            } else {
                (u, (-p) as u32)
            }
        }

        pub fn root(&mut self, u: u32) -> u32 {
            self.root_with_size(u).0
        }

        pub fn merge(&mut self, u: u32, v: u32) -> bool {
            let (mut u, size_u) = self.root_with_size(u);
            let (mut v, size_v) = self.root_with_size(v);
            if u == v {
                return false;
            }

            if size_u < size_v {
                std::mem::swap(&mut u, &mut v);
            }
            self.link[v as usize] = u as i32;
            self.link[u as usize] = -((size_u + size_v) as i32);
            true
        }
    }
}
