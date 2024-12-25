pub mod tree {
    pub fn euler_tour<'a>(
        n: usize,
        edges: impl IntoIterator<Item = (u32, u32)>,
        root: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        // XOR-linked tree traversal
        let mut degree = vec![0u32; n];
        let mut xor_neighbors: Vec<u32> = vec![0u32; n];
        for (u, v) in edges.into_iter().flat_map(|(u, v)| [(u, v), (v, u)]) {
            debug_assert!(u != v);
            degree[u as usize] += 1;
            xor_neighbors[u as usize] ^= v;
        }

        let mut size = vec![1; n];
        degree[root] += 2;
        let mut topological_order = Vec::with_capacity(n);
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
                u = p as usize;
            }
        }
        assert!(topological_order.len() == n - 1, "Invalid tree structure");

        let parent = xor_neighbors;

        // Downward propagation
        let mut euler_in = size.clone();
        for u in topological_order.into_iter().rev() {
            let p = parent[u as usize];
            let final_index = euler_in[p as usize];
            euler_in[p as usize] -= euler_in[u as usize];
            euler_in[u as usize] = final_index;
        }

        let mut euler_out = size;
        for u in 0..n {
            euler_in[u] -= 1;
            euler_out[u] += euler_in[u];
        }

        (euler_in, euler_out)
    }
}
