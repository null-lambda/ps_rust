use std::io::Write;

mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

pub mod tree_iso {
    // AHU algorithm for classifying (rooted) trees up to isomorphism
    // Time complexity: O(N log N)
    pub type Code = Box<[u32]>;

    #[derive(Debug)]
    pub struct AHULabelCompression {
        // Extracted classification codes
        forest_roots: Vec<usize>,
        levels: Vec<Vec<Code>>,

        // Auxiliary information for isomorphism construction
        ordered_children: Vec<Vec<u32>>,
    }

    impl AHULabelCompression {
        pub fn from_edges(
            n_verts: usize,
            edges: impl IntoIterator<Item = (u32, u32)>,
            root: Option<usize>, // None for unrooted tree isomorphism
        ) -> Self {
            assert!(n_verts > 0);

            // Build the tree structure
            let base_root = root.unwrap_or(0) as usize;
            let mut degree = vec![0u32; n_verts];
            let mut xor_neighbors = vec![0u32; n_verts];
            let mut neighbors = vec![vec![]; n_verts];
            for (u, v) in edges {
                debug_assert!(u < n_verts as u32);
                debug_assert!(v < n_verts as u32);
                degree[u as usize] += 1;
                degree[v as usize] += 1;
                xor_neighbors[u as usize] ^= v;
                xor_neighbors[v as usize] ^= u;
                neighbors[u as usize].push(v);
                neighbors[v as usize].push(u);
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

            let mut forest_roots = vec![];
            if let Some(root) = root {
                forest_roots.push(root);
            } else if root.is_none() {
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
                    neighbors[root].retain(|&v| v != aux_root as u32);
                    neighbors[aux_root].retain(|&v| v != root as u32);
                    size[root] -= size[aux_root];
                }

                forest_roots.push(root);
                forest_roots.extend(aux_root);
            }

            // Downward propagation
            let mut depth = vec![0u32; n_verts];
            let mut bfs_tour: Vec<_> = forest_roots.iter().map(|&u| u as u32).collect();
            let mut timer = 0;
            while let Some(&u) = bfs_tour.get(timer) {
                timer += 1;
                neighbors[u as usize]
                    .iter()
                    .position(|&v| size[v as usize] > size[u as usize])
                    .map(|ip| neighbors[u as usize].swap_remove(ip));
                for &v in &neighbors[u as usize] {
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

                let mut codes_d: Vec<(u32, Code)> = level_nodes
                    .iter()
                    .map(|&u| {
                        neighbors[u as usize].sort_unstable_by_key(|&v| code_in_parent[v as usize]);
                        let code = neighbors[u as usize]
                            .iter()
                            .map(|&v| code_in_parent[v as usize])
                            .collect();
                        (u, code)
                    })
                    .collect();
                codes_d.sort_unstable_by(|(_, ca), (_, cb)| ca.cmp(cb));

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
            forest_roots.sort_unstable_by_key(|&u| code_in_parent[u]);

            Self {
                forest_roots,
                levels,
                ordered_children,
            }
        }

        pub fn is_iso_to(&self, other: &Self) -> bool {
            self.forest_roots.len() == other.forest_roots.len() && self.levels == other.levels
        }

        pub fn get_mapping(&self, other: &Self) -> Option<Vec<u32>> {
            if !self.is_iso_to(other) {
                return None;
            }

            const UNSET: u32 = !0;
            let mut mapping = vec![UNSET; self.ordered_children.len()];

            let mut queue = vec![];
            for (&u1, &u2) in self.forest_roots.iter().zip(&other.forest_roots) {
                mapping[u1] = u2 as u32;
                queue.push(u1 as u32);
            }

            let mut timer = 0;
            while let Some(&u1) = queue.get(timer) {
                timer += 1;
                let u2 = mapping[u1 as usize];

                for (&v1, &v2) in self.ordered_children[u1 as usize]
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

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut trees = vec![];
    for _ in 0..2 {
        let edges = (0..n - 1).map(|_| (input.value::<u32>(), input.value::<u32>()));
        let comp = tree_iso::AHULabelCompression::from_edges(n, edges, None);
        // println!("{:?}", comp);
        trees.push(comp);
    }

    if let Some(mapping) = trees[0].get_mapping(&trees[1]) {
        writeln!(output, "JAH").unwrap();
        for to in mapping {
            writeln!(output, "{}", to).unwrap();
        }
    } else {
        writeln!(output, "EI").unwrap();
    }
}
