pub mod reroot {
    pub mod invertible {
        // O(n) rerooting dp for trees, with invertible pulling operation.

        pub trait RootData<E> {
            // Constraints: (x, inv) |-> (p |-> pull_from(p, x, inv)) must form a commutative group action. i.e.
            //   { p.pull_from(c1, inv1); p.pull_from(c2, inv2); }
            //   is equivalent to:
            //   { p.pull_from(c2, inv2); p.pull_from(c1, inv1); }
            fn pull_from(&mut self, child: &Self, weight: &E, inv: bool);

            fn finalize(&mut self);
        }

        fn get_two<T>(xs: &mut [T], i: usize, j: usize) -> Option<(&mut T, &mut T)> {
            debug_assert!(i < xs.len() && j < xs.len());
            if i == j {
                return None;
            }
            let ptr = xs.as_mut_ptr();
            Some(unsafe { (&mut *ptr.add(i), &mut *ptr.add(j)) })
        }

        fn dfs_init<E, R: RootData<E>>(
            neighbors: &[Vec<(usize, E)>],
            data: &mut [R],
            u: usize,
            p: usize,
        ) {
            for (v, w) in &neighbors[u] {
                if *v == p {
                    continue;
                }
                dfs_init(neighbors, data, *v, u);
                let (data_u, data_v) = unsafe { get_two(data, u, *v).unwrap_unchecked() };
                data_u.pull_from(&data_v, &w, false);
            }

            data[u].finalize();
        }

        fn reroot_on_edge<E, R: RootData<E>>(data: &mut [R], u: usize, w: &E, p: usize) {
            let (data_u, data_p) = unsafe { get_two(data, u, p).unwrap_unchecked() };
            data_p.pull_from(data_u, &w, true);
            data_p.finalize();

            data_u.pull_from(data_p, &w, false);
            data_u.finalize();
        }

        fn dfs_reroot<E, R: RootData<E> + Clone>(
            neighbors: &[Vec<(usize, E)>],
            data: &mut [R],
            yield_root_data: &mut impl FnMut(usize, &R),
            u: usize,
            p: usize,
        ) {
            yield_root_data(u, &data[u]);
            for (v, w) in &neighbors[u] {
                if *v == p {
                    continue;
                }
                let data_u_old = data[u].clone();
                let data_v_old = data[*v].clone();
                reroot_on_edge(data, *v, w, u);
                dfs_reroot(neighbors, data, yield_root_data, *v, u);
                data[*v] = data_v_old;
                data[u] = data_u_old;
            }
        }

        pub fn run<E, R: RootData<E> + Clone>(
            neighbors: &[Vec<(usize, E)>],
            data: &mut [R],
            root_init: usize,
            yield_node_dp: &mut impl FnMut(usize, &R),
        ) {
            dfs_init(neighbors, data, root_init, root_init);
            dfs_reroot(neighbors, data, yield_node_dp, root_init, root_init);
        }
    }
}
