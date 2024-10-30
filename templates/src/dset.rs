mod collections {
    use std::cell::Cell;

    pub struct DisjointSet {
        parent: Vec<Cell<usize>>,
        size: Vec<u32>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self {
                parent: (0..n).map(|i| Cell::new(i)).collect(),
                size: vec![1; n],
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            if u == self.parent[u].get() {
                u
            } else {
                self.parent[u].set(self.find_root(self.parent[u].get()));
                self.parent[u].get()
            }
        }

        pub fn get_size(&self, u: usize) -> u32 {
            self.size[self.find_root(u)]
        }

        // returns whether two set were different
        pub fn merge(&mut self, mut u: usize, mut v: usize) -> bool {
            u = self.find_root(u);
            v = self.find_root(v);
            if u == v {
                return false;
            }
            if self.size[u] > self.size[v] {
                std::mem::swap(&mut u, &mut v);
            }
            self.parent[v].set(u);
            self.size[u] += self.size[v];
            true
        }
    }
}
