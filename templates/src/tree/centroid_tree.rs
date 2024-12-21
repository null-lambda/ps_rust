pub mod centroid {
    // Centroid Decomposition

    fn reroot_on_edge(size: &mut [u32], u: usize, p: usize) {
        size[p] -= size[u];
        size[u] += size[p];
    }

    fn find_centroid<E>(
        neighbors: &[Vec<(u32, E)>],
        size: &mut [u32],
        visited: &[bool],
        n_half: u32,
        path: &mut Vec<u32>,
        u: usize,
        p: usize,
    ) -> usize {
        path.push(u as u32);
        for &(v, _) in &neighbors[u] {
            if v as usize == p || visited[v as usize] {
                continue;
            }
            if size[v as usize] > n_half {
                reroot_on_edge(size, v as usize, u);
                return find_centroid(neighbors, size, visited, n_half, path, v as usize, u);
            }
        }
        u
    }

    fn update_size<E>(
        neighbors: &[Vec<(u32, E)>],
        size: &mut [u32],
        visited: &[bool],
        u: usize,
        p: usize,
    ) {
        size[u] = 1;
        for &(v, _) in &neighbors[u] {
            if v as usize == p || visited[v as usize] {
                continue;
            }
            update_size(neighbors, size, visited, v as usize, u);
            size[u] += size[v as usize];
        }
    }

    pub fn init_size<E>(
        neighbors: &[Vec<(u32, E)>],
        size: &mut [u32],
        visited: &mut [bool],
        init: usize,
    ) {
        update_size(neighbors, size, visited, init, init); // TODO
    }

    pub fn dnc<E, F>(
        neighbors: &[Vec<(u32, E)>],
        size: &mut [u32],
        visited: &mut [bool],
        rooted_solver: &mut F,
        init: usize,
    ) where
        F: FnMut(&[Vec<(u32, E)>], &[u32], &[bool], usize),
    {
        println!("init: {:?}", size);

        update_size(neighbors, size, visited, init, init);
        let mut path = vec![];
        let root = find_centroid(
            neighbors,
            size,
            visited,
            size[init] / 2,
            &mut path,
            init,
            init,
        );

        visited[root] = true;
        rooted_solver(neighbors, size, visited, root);

        for &(v, _) in &neighbors[root] {
            if visited[v as usize] {
                continue;
            }
            dnc(neighbors, size, visited, rooted_solver, v as usize);
        }

        loop {
            match &path[..] {
                [.., p, u] => reroot_on_edge(size, *p as usize, *u as usize),
                _ => break,
            }
            path.pop();
        }
        path.clear();
    }
}
