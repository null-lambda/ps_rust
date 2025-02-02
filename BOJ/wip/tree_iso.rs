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
    // AHU algorithm for classying (rooted) trees up to isomorphism
    struct Level {}

    pub fn ahu_label_compression(
        n_verts: usize,
        edges: impl IntoIterator<Item = (u32, u32)>,
        root: Option<u32>, // None for unrooted tree isomorphism
    ) {
        assert!(n_verts > 0);

        let base_root = root.unwrap_or(0) as usize;
        let mut degree = vec![0u32; n_verts + 1];
        let mut xor_neighbors = vec![0u32; n_verts + 1];
        let mut neighbors = vec![vec![]; n_verts + 1];
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
        let mut size = vec![1u32; n_verts + 1];
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
        let parent = xor_neighbors;

        if root.is_none() {
            // Reroot down to the lowest centroid
            let mut root = base_root;
            let threshold = (n_verts as u32 + 1) / 2;
            for u in topological_order.into_iter().rev() {
                let p = parent[u as usize] as usize;
                if p as usize != base_root {
                    continue;
                }
                if size[u as usize] >= threshold {
                    size[p as usize] -= size[u as usize];
                    size[u as usize] += size[p as usize];
                    root = u as usize;
                }
            }
            let mut aux_root = None;

            // Check the direct parent for another root
            let p = parent[root];
            if p != root as u32 && size[p as usize] >= threshold {
                aux_root = Some(p as usize);
            }

            // Split the double-root tree into two-compoment forest
            // // Add a virtual root to unify multiple roots.
            // // Ensure that single-root tree is distinguished with double roots.
            let virtual_root = n_verts;
            if let Some(aux_root) = aux_root {
                // Toggle edges in a triangle
                neighbors[root].retain(|&v| v != aux_root as u32);
                neighbors[aux_root].retain(|&v| v != root as u32);
                // neighbors[root].push(virtual_root as u32);
                // neighbors[aux_root].push(virtual_root as u32);
                // neighbors[virtual_root].push(root as u32);
                // neighbors[virtual_root as usize].push(aux_root as u32);
                size[root] -= size[aux_root];
            } else {
                // neighbors[root].push(virtual_root as u32);
                // neighbors[virtual_root as usize].push(centroid as u32);
            }
            size[virtual_root as usize] = n_verts as u32 + 1;
        }

        // Bucket sort by size (the original algorithm uses depth; we reuse the size array from
        // the centroid calculation)
        let mut level_nodes = vec![vec![]; n_verts + 1];
        for u in 0..n_verts {
            level_nodes[size[u] as usize].push(u as u32);
        }

        let mut codes: Vec<(u32, Box<[u32]>)> = vec![Default::default(); n_verts];
        let mut levels = vec![];
        for s in 1..=n_verts as u32 {
            let mut codes_s: Vec<_> = level_nodes[s as usize]
                .iter()
                .map(|&u| {
                    neighbors[u as usize]
                        .iter()
                        .position(|&v| size[v as usize] > s)
                        .map(|ip| neighbors[u as usize].swap_remove(ip));

                    let mut code: Box<[u32]> = neighbors[u as usize]
                        .iter()
                        .map(|&v| codes[v as usize].0)
                        .collect();
                    code.sort_unstable();
                    (u, code)
                })
                .collect();
            codes_s.sort_unstable_by(|(_, ca), (_, cb)| ca.cmp(cb));

            let mut compressed_code = 0u32;
            let mut prev_code = None;
            for (u, code) in codes_s.iter() {
                if prev_code.is_some() && prev_code != Some(code) {
                    compressed_code += 1;
                }
                codes[*u as usize] = compressed_code;

                prev_code = Some(code);
            }
            levels.push(codes_s);
        }
        let children = neighbors;

        println!("{:?}", children);
        println!("{:?}", codes);
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    for _ in 0..2 {
        let edges = (0..n - 1).map(|_| (input.value::<u32>(), input.value::<u32>()));
        tree_iso::ahu_label_compression(n, edges, None);
    }
}
