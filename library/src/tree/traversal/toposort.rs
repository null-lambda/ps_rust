fn xor_traversal(
    mut degree: Vec<u32>,
    mut xor_neighbors: Vec<u32>,
    root: u32,
) -> (Vec<u32>, Vec<u32>) {
    let n = degree.len();
    degree[root as usize] += 2;

    let mut toposort = vec![];

    for mut u in 0..n as u32 {
        while degree[u as usize] == 1 {
            let p = xor_neighbors[u as usize];
            xor_neighbors[p as usize] ^= u;
            degree[u as usize] -= 1;
            degree[p as usize] -= 1;

            toposort.push(u);

            u = p;
        }
    }
    toposort.push(root);

    let mut parent = xor_neighbors;
    parent[root as usize] = root;
    (toposort, parent)
}

fn toposort(
    n_verts: usize,
    edges: impl IntoIterator<Item = [u32; 2]>,
    root: u32,
) -> (Vec<u32>, Vec<u32>) {
    let mut degree = vec![0u32; n_verts];
    let mut xor_neighbors = vec![0u32; n_verts];
    for [u, v] in edges {
        degree[u as usize] += 1;
        degree[v as usize] += 1;
        xor_neighbors[u as usize] ^= v;
        xor_neighbors[v as usize] ^= u;
    }
    xor_traversal(degree, xor_neighbors, root)
}
