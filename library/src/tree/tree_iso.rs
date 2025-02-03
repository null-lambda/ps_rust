pub mod tree_iso {
    // AHU algorithm for classifying (rooted) trees up to isomorphism
    // Time complexity: O(N log N)    (O(N) with radix sort))
    pub type Code<E> = Box<[(u32, E)]>;

    #[derive(Debug, Clone)]
    pub struct AHULabelCompression<E> {
        // Extracted classification codes
        pub forest_roots: (Vec<usize>, E),
        pub levels: Vec<Vec<Code<E>>>,

        // Auxiliary information for isomorphism construction
        pub ordered_children: Vec<Vec<(u32, E)>>,
        pub depth: Vec<u32>,
        pub code_in_parent: Vec<u32>,
        pub bfs_tour: Vec<u32>,
    }

    impl<E: Clone + Ord + Default> AHULabelCompression<E> {
        pub fn from_edges(
            n_verts: usize,
            edges: impl IntoIterator<Item = (u32, u32, E)>,
            root: Option<usize>, // None for unrooted tree isomorphism
        ) -> Self {
            assert!(n_verts > 0);

            // Build the tree structure
            let base_root = root.unwrap_or(0) as usize;
            let mut degree = vec![0u32; n_verts];
            let mut xor_neighbors = vec![0u32; n_verts];
            let mut neighbors = vec![vec![]; n_verts];
            for (u, v, w) in edges {
                debug_assert!(u < n_verts as u32);
                debug_assert!(v < n_verts as u32);
                degree[u as usize] += 1;
                degree[v as usize] += 1;
                xor_neighbors[u as usize] ^= v;
                xor_neighbors[v as usize] ^= u;
                neighbors[u as usize].push((v, w.clone()));
                neighbors[v as usize].push((u, w));
            }
            degree[base_root] += 2;

            // Upward propagation
            let mut size = vec![1u32; n_verts];
            let mut topological_order = vec![];
            for mut u in 0..n_verts as u32 {
                while degree[u as usize] == 1 {
                    let p = xor_neighbors[u as usize];
                    xor_neighbors[p as usize] ^= u;
                    degree[u as usize] -= 1;
                    degree[p as usize] -= 1;
                    topological_order.push(u);

                    size[p as usize] += size[u as usize];

                    u = p;
                }
            }
            assert!(
                topological_order.len() == n_verts - 1,
                "Invalid tree structure"
            );
            let parent = xor_neighbors;

            let mut forest_roots = if let Some(root) = root {
                (vec![root], E::default())
            } else {
                // For unrooted tree classification, assign some distinguished roots.

                // Reroot down to the lowest centroid
                let mut root = base_root;
                let threshold = (n_verts as u32 + 1) / 2;
                for u in topological_order.into_iter().rev() {
                    let p = parent[u as usize] as usize;
                    if p == root && size[u as usize] >= threshold {
                        size[p as usize] -= size[u as usize];
                        size[u as usize] += size[p as usize];
                        root = u as usize;
                    }
                }

                // Check the direct parent for another centroid
                let mut aux_root = None;
                let p = parent[root];
                if p != root as u32 && size[p as usize] >= threshold {
                    aux_root = Some(p as usize);
                }

                if let Some(aux_root) = aux_root {
                    // Split the double-centroid tree into a two-component forest
                    let i = neighbors[root]
                        .iter()
                        .position(|&(v, _)| v as usize == aux_root)
                        .unwrap();
                    let (_, w) = neighbors[root].swap_remove(i);
                    neighbors[root].retain(|&(v, _)| v != aux_root as u32);
                    neighbors[aux_root].retain(|&(v, _)| v != root as u32);
                    size[root] -= size[aux_root];

                    (vec![root, aux_root], w)
                } else {
                    (vec![root], E::default())
                }
            };

            // Downward propagation
            let mut depth = vec![0u32; n_verts];
            let mut bfs_tour: Vec<_> = forest_roots.0.iter().map(|&u| u as u32).collect();
            let mut timer = 0;
            while let Some(&u) = bfs_tour.get(timer) {
                timer += 1;
                neighbors[u as usize]
                    .iter()
                    .position(|&(v, _)| size[v as usize] > size[u as usize])
                    .map(|ip| neighbors[u as usize].swap_remove(ip));
                for &(v, _) in &neighbors[u as usize] {
                    depth[v as usize] = depth[u as usize] + 1;
                    bfs_tour.push(v);
                }
            }

            // Encode the forest starting from the deepest level
            let mut code_in_parent: Vec<u32> = vec![Default::default(); n_verts];
            let mut levels: Vec<_> = vec![];
            let (mut start, mut end) = (n_verts, n_verts);
            for d in (0..n_verts as u32).rev() {
                while start > 0 && depth[bfs_tour[start - 1] as usize] >= d {
                    start -= 1;
                }
                let level_nodes = &bfs_tour[start..end];
                end = start;

                let mut codes_d: Vec<(u32, Code<E>)> = level_nodes
                    .iter()
                    .map(|&u| {
                        neighbors[u as usize].sort_unstable_by(|(v1, w1), (v2, w2)| {
                            code_in_parent[*v1 as usize]
                                .cmp(&code_in_parent[*v2 as usize])
                                .then_with(|| w1.cmp(w2))
                        });
                        let code: Code<E> = neighbors[u as usize]
                            .iter()
                            .map(|(v, w)| (code_in_parent[*v as usize], w.clone()))
                            .collect();
                        (u, code)
                    })
                    .collect();
                codes_d.sort_unstable_by(|(_, code_a), (_, code_b)| code_a.cmp(code_b));

                let mut c = 0u32;
                let mut prev_code = None;
                for (u, code) in &codes_d {
                    if prev_code.is_some() && prev_code != Some(code) {
                        c += 1;
                    }
                    code_in_parent[*u as usize] = c;
                    prev_code = Some(code);
                }
                levels.push(codes_d.into_iter().map(|(_, code)| code).collect());
            }
            let ordered_children = neighbors;
            forest_roots.0.sort_unstable_by_key(|&u| code_in_parent[u]);

            Self {
                forest_roots,
                levels,

                ordered_children,
                depth,
                code_in_parent,
                bfs_tour,
            }
        }

        pub fn is_iso_to(&self, other: &Self) -> bool {
            self.forest_roots.0.len() == other.forest_roots.0.len()
                && self.forest_roots.1 == other.forest_roots.1
                && self.levels == other.levels
        }

        pub fn cmp_by_labels(&self, other: &Self) -> std::cmp::Ordering {
            (self.forest_roots.0.len().cmp(&other.forest_roots.0.len()))
                .then_with(|| self.forest_roots.1.cmp(&other.forest_roots.1))
                .then_with(|| self.levels.cmp(&other.levels))
        }

        pub fn get_mapping(&self, other: &Self) -> Option<Vec<u32>> {
            if !self.is_iso_to(other) {
                return None;
            }

            const UNSET: u32 = !0;
            let mut mapping = vec![UNSET; self.ordered_children.len()];

            let mut queue = vec![];
            for (&u1, &u2) in self.forest_roots.0.iter().zip(&other.forest_roots.0) {
                mapping[u1] = u2 as u32;
                queue.push(u1 as u32);
            }

            let mut timer = 0;
            while let Some(&u1) = queue.get(timer) {
                timer += 1;
                let u2 = mapping[u1 as usize];

                for (&(v1, _), &(v2, _)) in self.ordered_children[u1 as usize]
                    .iter()
                    .zip(&other.ordered_children[u2 as usize])
                {
                    mapping[v1 as usize] = v2;
                    queue.push(v1);
                }
            }
            debug_assert!(mapping.iter().all(|&v| v != UNSET));

            Some(mapping)
        }
    }
}
