pub mod centroid {
    /// Centroid Decomposition
    use crate::jagged::Jagged;

    pub fn init_size<'a, E: 'a>(
        neighbors: &'a impl Jagged<'a, (u32, E)>,
        size: &mut [u32],
        u: usize,
        p: usize,
    ) {
        size[u] = 1;
        for &(v, _) in neighbors.get(u) {
            if v as usize == p {
                continue;
            }
            init_size(neighbors, size, v as usize, u);
            size[u] += size[v as usize];
        }
    }

    fn reroot_to_centroid<'a, _E: 'a>(
        neighbors: &'a impl Jagged<'a, (u32, _E)>,
        size: &mut [u32],
        visited: &[bool],
        mut u: usize,
    ) -> usize {
        let threshold = (size[u] + 1) / 2;
        let mut p = u;
        'outer: loop {
            for &(v, _) in neighbors.get(u) {
                if v as usize == p || visited[v as usize] {
                    continue;
                }
                if size[v as usize] >= threshold {
                    size[u] -= size[v as usize];
                    size[v as usize] += size[u];

                    p = u;
                    u = v as usize;
                    continue 'outer;
                }
            }
            return u;
        }
    }

    pub fn dnc<'a, E: 'a + Clone>(
        neighbors: &'a impl Jagged<'a, (u32, E)>,
        size: &mut [u32],
        visited: &mut [bool],
        yield_rooted_tree: &mut impl FnMut(&[u32], &[bool], usize),
        init: usize,
    ) {
        let root = reroot_to_centroid(neighbors, size, visited, init);
        visited[root] = true;
        yield_rooted_tree(size, visited, root);
        for &(v, _) in neighbors.get(root) {
            if visited[v as usize] {
                continue;
            }
            dnc(neighbors, size, visited, yield_rooted_tree, v as usize)
        }
    }

    pub fn build_centroid_tree<'a, _E: 'a + Clone>(
        neighbors: &'a impl Jagged<'a, (u32, _E)>,
        size: &mut [u32],
        visited: &mut [bool],
        parent_centroid: &mut [u32],
        init: usize,
    ) -> usize {
        let root = reroot_to_centroid(neighbors, size, visited, init);
        visited[root] = true;

        for &(v, _) in neighbors.get(root) {
            if visited[v as usize] {
                continue;
            }
            let sub_root =
                build_centroid_tree(neighbors, size, visited, parent_centroid, v as usize);
            parent_centroid[sub_root] = root as u32;
        }
        root
    }
}
