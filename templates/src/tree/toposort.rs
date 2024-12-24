pub mod tree {
    // Fast tree reconstruction with XOR-linked tree traversal
    // https://codeforces.com/blog/entry/135239
    pub fn toposort<'a, const N: usize, E: Clone + Default + 'a>(
        n_verts: usize,
        edges: impl IntoIterator<Item = (u32, u32, E)>,
        root: usize,
    ) -> (Vec<u32>, Vec<(u32, E)>)
    where
        E: AsBytes<N>,
    {
        let mut degree = vec![0u32; n_verts];
        let mut xor: Vec<(u32, [u8; N])> = vec![(0 as u32, [0u8; N]); n_verts];

        fn xor_assign_bytes<const N: usize>(xs: &mut [u8; N], ys: [u8; N]) {
            for (x, y) in xs.iter_mut().zip(&ys) {
                *x ^= *y;
            }
        }

        for (u, v, w) in edges
            .into_iter()
            .flat_map(|(u, v, w)| [(u, v, w.clone()), (v, u, w)])
        {
            debug_assert!(u != v);
            degree[u as usize] += 1;
            xor[u as usize].0 ^= v;
            xor_assign_bytes(&mut xor[u as usize].1, unsafe {
                AsBytes::as_bytes(w.clone())
            });
        }

        degree[root] += 2;
        let mut order = vec![];
        for mut u in 0..n_verts {
            while degree[u] == 1 {
                let (v, w_encoded) = xor[u];
                order.push(u as u32);
                degree[u] = 0;
                degree[v as usize] -= 1;
                xor[v as usize].0 ^= u as u32;
                xor_assign_bytes(&mut xor[v as usize].1, w_encoded);
                u = v as usize;
            }
        }
        order.push(root as u32);
        order.reverse();

        // Note: Copying entire vec (from xor to parent) is necessary, since
        // transmuting from (u32, [u8; N]) to (u32, E)
        // or *const (u32, [u8; N]) to *const (u32, E) is UB. (tuples are
        // #[align(rust)] struct and the field ordering is not guaranteed)
        // We may try messing up with custom #[repr(C)] structs, or just trust the compiler.
        xor[root].0 = root as u32;
        let parent: Vec<(u32, E)> = xor
            .into_iter()
            .map(|(v, w)| (v, unsafe { AsBytes::decode(w) }))
            .collect();
        (order, parent)
    }
}
