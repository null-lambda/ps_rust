pub mod hld {
    use crate::collections::Jagged;

    // Heavy-Light Decomposition
    #[derive(Debug)]
    pub struct HLD {
        pub size: Vec<u32>,
        pub depth: Vec<u32>,
        pub parent: Vec<u32>,
        pub heavy_child: Vec<u32>,
        pub chain_top: Vec<u32>,
        pub euler_idx: Vec<u32>,
    }

    impl HLD {
        pub fn len(&self) -> usize {
            self.parent.len()
        }

        fn dfs_size(&mut self, neighbors: &Jagged<u32>, u: usize) {
            self.size[u] = 1;
            for &v in &neighbors[u] {
                if v == self.parent[u] {
                    continue;
                }
                self.depth[v as usize] = self.depth[u] + 1;
                self.parent[v as usize] = u as u32;
                self.dfs_size(neighbors, v as usize);
                self.size[u] += self.size[v as usize];
            }
            if let Some(h) = neighbors[u]
                .iter()
                .copied()
                .filter(|&v| v != self.parent[u])
                .max_by_key(|&v| self.size[v as usize])
            {
                self.heavy_child[u] = h;
            }
        }

        fn dfs_decompose(&mut self, neighbors: &Jagged<u32>, u: usize, order: &mut u32) {
            self.euler_idx[u] = *order;
            *order += 1;
            if self.heavy_child[u] == u32::MAX {
                return;
            }
            let h = self.heavy_child[u];
            self.chain_top[h as usize] = self.chain_top[u];
            self.dfs_decompose(neighbors, h as usize, order);
            for &v in neighbors[u].iter().filter(|&&v| v != h) {
                if v == self.parent[u] {
                    continue;
                }
                self.chain_top[v as usize] = v;
                self.dfs_decompose(neighbors, v as usize, order);
            }
        }

        pub fn from_graph(neighbors: &Jagged<u32>) -> Self {
            let n = neighbors.len();
            let mut hld = Self {
                size: vec![0; n],
                depth: vec![0; n],
                parent: vec![u32::MAX; n],
                heavy_child: vec![u32::MAX; n],
                chain_top: vec![0; n],
                euler_idx: vec![0; n],
            };
            hld.dfs_size(neighbors, 0);
            hld.dfs_decompose(neighbors, 0, &mut 0);
            hld
        }

        pub fn for_each_path<F>(&self, mut u: usize, mut v: usize, mut visitor: F)
        where
            F: FnMut(usize, usize, bool),
        {
            debug_assert!(u < self.len() && v < self.len());

            while self.chain_top[u] != self.chain_top[v] {
                if self.depth[self.chain_top[u] as usize] < self.depth[self.chain_top[v] as usize] {
                    std::mem::swap(&mut u, &mut v);
                }
                visitor(self.chain_top[u] as usize, u, false);
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.euler_idx[u] > self.euler_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            visitor(u, v, true);
        }

        pub fn lca(&self, mut u: usize, mut v: usize) -> usize {
            debug_assert!(u < self.len() && v < self.len());
            while self.chain_top[u] != self.chain_top[v] {
                if self.depth[self.chain_top[u] as usize] < self.depth[self.chain_top[v] as usize] {
                    std::mem::swap(&mut u, &mut v);
                }
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.euler_idx[u] > self.euler_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            u
        }
    }
}
