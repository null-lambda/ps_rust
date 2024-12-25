pub mod lca {
    // O(1) LCA with O(n) preprocessing
    // Farach-Colton and Bender algorithm
    // https://cp-algorithms.com/graph/lca_farachcoltonbender.html
    const UNSET: u32 = u32::MAX;
    const INF: u32 = u32::MAX;

    fn log2(x: u32) -> u32 {
        assert!(x > 0);
        u32::BITS - 1 - x.leading_zeros()
    }

    #[derive(Clone, Copy)]
    struct CmpBy<K, V>(K, V);

    impl<K: PartialEq, V> PartialEq for CmpBy<K, V> {
        fn eq(&self, other: &Self) -> bool {
            self.0.eq(&other.0)
        }
    }

    impl<K: Eq, V> Eq for CmpBy<K, V> {}

    impl<K: PartialOrd, V> PartialOrd for CmpBy<K, V> {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.0.partial_cmp(&other.0)
        }
    }

    impl<K: Ord, V> Ord for CmpBy<K, V> {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.cmp(&other.0)
        }
    }

    pub struct LCA {
        n: usize,

        n_euler: usize,
        euler_tour: Vec<u32>,
        euler_in: Vec<u32>,

        block_size: usize,
        n_blocks: usize,
        height: Vec<u32>,
        min_sparse: Vec<Vec<CmpBy<u32, u32>>>,

        block_mask: Vec<u32>,
        min_idx_in_block: Vec<Vec<Vec<u32>>>,
    }

    impl LCA {
        pub fn from_graph(
            n: usize,
            edges: impl IntoIterator<Item = (u32, u32)>,
            root: usize,
        ) -> Self {
            let mut neighbors = vec![vec![]; n];
            for (u, v) in edges {
                neighbors[u as usize].push(v);
                neighbors[v as usize].push(u);
            }

            let n_euler = 2 * n - 1;
            let block_size = (log2(n_euler as u32) as usize / 2).max(1);
            let n_blocks = n_euler.div_ceil(block_size);

            let mut this = LCA {
                n,

                n_euler,
                euler_in: vec![UNSET; n],
                euler_tour: vec![],

                block_size,
                n_blocks,
                height: vec![0; n],
                min_sparse: vec![
                    vec![CmpBy(INF, UNSET); n_blocks];
                    log2(n_blocks as u32) as usize + 1
                ],

                block_mask: vec![0; n_blocks],
                min_idx_in_block: vec![vec![]; 1 << block_size - 1],
            };

            this.build_euler(&neighbors, root as u32, root as u32);
            assert_eq!(this.euler_tour.len(), n_euler);

            this.build_sparse();
            this
        }

        fn build_euler(&mut self, neighbors: &[Vec<u32>], u: u32, p: u32) {
            self.euler_in[u as usize] = self.euler_tour.len() as u32;
            self.euler_tour.push(u);

            for &v in &neighbors[u as usize] {
                if v == p {
                    continue;
                }
                self.height[v as usize] = self.height[u as usize] + 1;
                self.build_euler(neighbors, v, u);
                self.euler_tour.push(u);
            }
        }

        fn key(&self, tour_idx: usize) -> CmpBy<u32, u32> {
            let u = self.euler_tour[tour_idx] as usize;
            CmpBy(self.height[u], u as u32)
        }

        fn build_sparse(&mut self) {
            for i in 0..self.n_euler {
                let b = i / self.block_size;
                self.min_sparse[0][b] = self.min_sparse[0][b].min(self.key(i));
            }
            for exp in 1..self.min_sparse.len() {
                for i in 0..self.n_blocks {
                    let j = i + (1 << exp - 1);
                    self.min_sparse[exp][i] = self.min_sparse[exp - 1][i];
                    if j < self.n_blocks {
                        self.min_sparse[exp][i] =
                            self.min_sparse[exp][i].min(self.min_sparse[exp - 1][j]);
                    }
                }
            }

            for i in 0..self.n_euler {
                let (b, s) = (i / self.block_size, i % self.block_size);
                if s > 0 && self.key(i - 1) < self.key(i) {
                    self.block_mask[b] |= 1 << s - 1;
                }
            }

            for b in 0..self.n_blocks {
                let mask = self.block_mask[b] as usize;
                if !self.min_idx_in_block[mask].is_empty() {
                    continue;
                }
                self.min_idx_in_block[mask] = vec![vec![UNSET; self.block_size]; self.block_size];
                for l in 0..self.block_size {
                    self.min_idx_in_block[mask][l][l] = l as u32;
                    for r in l + 1..self.block_size {
                        self.min_idx_in_block[mask][l][r] = self.min_idx_in_block[mask][l][r - 1];
                        if b * self.block_size + r < self.n_euler
                            && self.key(
                                b * self.block_size + self.min_idx_in_block[mask][l][r] as usize,
                            ) > self.key(b * self.block_size + r)
                        {
                            self.min_idx_in_block[mask][l][r] = r as u32;
                        }
                    }
                }
            }
        }

        fn min_in_block(&self, b: usize, l: usize, r: usize) -> CmpBy<u32, u32> {
            let mask = self.block_mask[b] as usize;
            let shift = self.min_idx_in_block[mask][l][r];
            self.key(b * self.block_size + shift as usize)
        }

        pub fn get(&self, u: usize, v: usize) -> usize {
            debug_assert!(u < self.n);
            debug_assert!(v < self.n);

            let l = self.euler_in[u].min(self.euler_in[v]) as usize;
            let r = self.euler_in[u].max(self.euler_in[v]) as usize;

            let (bl, sl) = (l / self.block_size, l % self.block_size);
            let (br, sr) = (r / self.block_size, r % self.block_size);
            if bl == br {
                return self.min_in_block(bl, sl, sr).1 as usize;
            }

            let prefix = self.min_in_block(bl, sl, self.block_size - 1);
            let suffix = self.min_in_block(br, 0, sr);
            let mut res = prefix.min(suffix);
            if bl + 1 < br {
                let exp = log2((br - bl - 1) as u32) as usize;
                res = res.min(self.min_sparse[exp][bl + 1]);
                res = res.min(self.min_sparse[exp][br - (1 << exp)]);
            }

            res.1 as usize
        }
    }

    // Build a max cartesian tree from inorder traversal
    fn max_cartesian_tree<T>(
        n: usize,
        iter: impl IntoIterator<Item = (usize, T)>,
    ) -> (Vec<u32>, usize)
    where
        T: Ord,
    {
        let mut parent = vec![UNSET; n];

        // Monotone stack
        let mut stack = vec![];
        for (u, h) in iter {
            let u = u as u32;

            let mut c = None;
            while let Some((prev, _)) = stack.last() {
                if prev > &h {
                    break;
                }
                c = stack.pop();
            }
            if let Some(&(_, p)) = stack.last() {
                parent[u as usize] = p;
            }
            if let Some((_, c)) = c {
                parent[c as usize] = u;
            }
            stack.push((h, u));
        }
        let root = (0..n).find(|&i| parent[i] == UNSET).unwrap();
        (parent, root)
    }

    pub struct StaticRangeMax<T> {
        xs: Vec<T>,
        cartesian_tree: LCA,
    }

    impl<T: Ord> StaticRangeMax<T> {
        pub fn from_iter(xs: impl Iterator<Item = T>) -> Self {
            let xs: Vec<_> = xs.into_iter().collect();
            let n = xs.len();
            assert!(n >= 1);

            let (parent, root) = max_cartesian_tree(n, xs.iter().enumerate());
            Self {
                xs,
                cartesian_tree: LCA::from_graph(
                    n,
                    (0..n)
                        .map(|i| (i as u32, parent[i]))
                        .filter(|&(_, v)| v != UNSET),
                    root,
                ),
            }
        }

        pub fn argmax_range(&self, range: std::ops::Range<usize>) -> usize {
            debug_assert!(range.start < range.end && range.end <= self.xs.len());
            self.cartesian_tree.get(range.start, range.end - 1)
        }

        pub fn max_range(&self, range: std::ops::Range<usize>) -> &T {
            &self.xs[self.argmax_range(range)]
        }
    }
}
