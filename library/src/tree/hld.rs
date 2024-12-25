pub mod hld {
    const UNSET: u32 = u32::MAX;

    // Heavy-Light Decomposition
    #[derive(Debug)]
    pub struct HLD {
        pub size: Vec<u32>,
        pub depth: Vec<u32>,
        pub parent: Vec<u32>,
        pub heavy_child: Vec<u32>,
        pub chain_top: Vec<u32>,
        // pub segmented_idx: Vec<u32>,
        pub euler_in: Vec<u32>,
        pub topological_order: Vec<u32>,
    }

    impl HLD {
        pub fn len(&self) -> usize {
            self.parent.len()
        }

        pub fn from_edges<'a>(
            n: usize,
            edges: impl IntoIterator<Item = (u32, u32)>,
            root: usize,
        ) -> Self {
            // Fast tree reconstruction with XOR-linked tree traversal
            // https://codeforces.com/blog/entry/135239
            let mut degree = vec![0u32; n];
            let mut xor_neighbors: Vec<u32> = vec![0u32; n];
            for (u, v) in edges.into_iter().flat_map(|(u, v)| [(u, v), (v, u)]) {
                debug_assert!(u != v);
                degree[u as usize] += 1;
                xor_neighbors[u as usize] ^= v;
            }

            let mut size = vec![1; n];
            let mut heavy_child = vec![UNSET; n];
            degree[root] += 2;
            let mut topological_order = vec![];
            for mut u in 0..n {
                while degree[u] == 1 {
                    // Topological sort
                    let p = xor_neighbors[u];
                    topological_order.push(u as u32);
                    degree[u] = 0;
                    degree[p as usize] -= 1;
                    xor_neighbors[p as usize] ^= u as u32;

                    // Upward propagation
                    size[p as usize] += size[u as usize];
                    let h = &mut heavy_child[p as usize];
                    if *h == UNSET || size[*h as usize] < size[u as usize] {
                        *h = u as u32;
                    }

                    u = p as usize;
                }
            }
            topological_order.push(root as u32);
            topological_order.reverse();
            assert!(topological_order.len() == n, "Invalid tree structure");

            let mut parent = xor_neighbors;
            parent[root] = UNSET;

            // Downward propagation
            let mut depth = vec![0; n];
            let mut chain_top = vec![root as u32; n];
            for &u in &topological_order[1..] {
                let p = parent[u as usize];
                depth[u as usize] = depth[p as usize] + 1;
            }

            // // Rearranged topological index continuous in a chain, for path queries
            // let mut segmented_idx = vec![UNSET; n];
            // let mut timer = 0;
            // for u in &topological_order {
            //     let mut u = *u;
            //     while u != UNSET && segmented_idx[u as usize] == UNSET {
            //         segmented_idx[u as usize] = timer;
            //         timer += 1;
            //         u = heavy_child[u as usize];
            //     }
            // }

            // Dfs ordering for path & subtree queries
            let mut euler_in = vec![UNSET; n];
            let mut offset = vec![0; n];
            for u in &topological_order {
                let mut u = *u;
                if euler_in[u as usize] != UNSET {
                    continue;
                }

                let mut p = parent[u as usize];
                let mut timer = 0;
                if p != UNSET {
                    timer = offset[p as usize] + 1;
                    offset[p as usize] += size[u as usize] as u32;
                }
                euler_in[u as usize] = timer;
                offset[u as usize] = timer;
                chain_top[u as usize] = u;

                timer += 1;

                loop {
                    p = u;
                    u = heavy_child[u as usize];
                    if u == UNSET {
                        break;
                    }

                    chain_top[u as usize] = chain_top[p as usize];
                    offset[p as usize] += size[u as usize] as u32;
                    offset[u as usize] = timer;
                    euler_in[u as usize] = timer;
                    timer += 1;
                }
            }

            Self {
                size,
                depth,
                parent,
                heavy_child,
                chain_top,
                // segmented_idx,
                euler_in,
                topological_order,
            }
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
            if self.euler_in[u] > self.euler_in[v] {
                std::mem::swap(&mut u, &mut v);
            }
            visitor(u, v, true);
        }

        pub fn for_each_path_splitted<F>(&self, mut u: usize, mut v: usize, mut visit: F)
        where
            F: FnMut(usize, usize, bool, bool),
        {
            debug_assert!(u < self.len() && v < self.len());
            if self.euler_in[u] > self.euler_in[v] {
                std::mem::swap(&mut u, &mut v);
            }
            while self.chain_top[u] != self.chain_top[v] {
                if self.depth[self.chain_top[u] as usize] > self.depth[self.chain_top[v] as usize] {
                    visit(self.chain_top[u] as usize, u, true, false);
                    u = self.parent[self.chain_top[u] as usize] as usize;
                } else {
                    visit(self.chain_top[v] as usize, v, false, false);
                    v = self.parent[self.chain_top[v] as usize] as usize;
                }
            }
            if self.depth[u] > self.depth[v] {
                visit(v, u, true, true);
            } else {
                visit(u, v, false, true);
            }
        }

        pub fn lca(&self, mut u: usize, mut v: usize) -> usize {
            debug_assert!(u < self.len() && v < self.len());
            while self.chain_top[u] != self.chain_top[v] {
                if self.depth[self.chain_top[u] as usize] < self.depth[self.chain_top[v] as usize] {
                    std::mem::swap(&mut u, &mut v);
                }
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.euler_in[u] > self.euler_in[v] {
                std::mem::swap(&mut u, &mut v);
            }
            u
        }
    }
}
