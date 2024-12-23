pub mod reroot {
    pub mod invertible {
        // O(n) rerooting dp for trees, with invertible pulling operation.

        pub trait RootData<E> {
            // Constraints: (x, inv) |-> (p |-> pull_from(p, x, inv)) must form a commutative group action. i.e.
            //   { p.pull_from(c1, inv1); p.pull_from(c2, inv2); }
            //   is equivalent to:
            //   { p.pull_from(c2, inv2); p.pull_from(c1, inv1); }
            fn pull_from(&mut self, child: &Self, weight: &E, inv: bool);

            fn finalize(&mut self) {}
        }

        fn get_two<T>(xs: &mut [T], i: usize, j: usize) -> Option<(&mut T, &mut T)> {
            debug_assert!(i < xs.len() && j < xs.len());
            if i == j {
                return None;
            }
            let ptr = xs.as_mut_ptr();
            Some(unsafe { (&mut *ptr.add(i), &mut *ptr.add(j)) })
        }

        fn reroot_on_edge<E, R: RootData<E>>(data: &mut [R], u: usize, w: &E, p: usize) {
            let (data_u, data_p) = unsafe { get_two(data, u, p).unwrap_unchecked() };
            data_p.pull_from(data_u, &w, true);
            data_p.finalize();

            data_u.pull_from(data_p, &w, false);
            data_u.finalize();
        }

        fn dfs_reroot<E, R: RootData<E> + Clone>(
            neighbors: &[Vec<(u32, E)>],
            data: &mut [R],
            yield_root_data: &mut impl FnMut(usize, &R),
            u: usize,
            p: usize,
        ) {
            yield_root_data(u, &data[u]);
            for (v, w) in &neighbors[u] {
                if *v as usize == p {
                    continue;
                }
                reroot_on_edge(data, *v as usize, w, u);
                dfs_reroot(neighbors, data, yield_root_data, *v as usize, u);
                reroot_on_edge(data, u, w, *v as usize);
            }
        }

        pub fn run<E: Clone + Default, R: RootData<E> + Clone>(
            neighbors: &[Vec<(u32, E)>],
            data: &mut [R],
            root_init: usize,
            yield_node_dp: &mut impl FnMut(usize, &R),
        ) {
            let mut preorder = vec![]; // Reversed postorder
            let mut parent = vec![(root_init, E::default()); neighbors.len()];
            let mut stack = vec![(root_init, root_init)];
            while let Some((u, p)) = stack.pop() {
                preorder.push(u);
                for (v, w) in &neighbors[u] {
                    if *v as usize == p {
                        continue;
                    }
                    parent[*v as usize] = (u, w.clone());
                    stack.push((*v as usize, u));
                }
            }

            // Init tree DP
            for &u in preorder.iter().rev() {
                data[u].finalize();

                let (p, w) = &parent[u];
                if u != root_init {
                    let (data_u, data_p) = unsafe { get_two(data, u, *p).unwrap_unchecked() };
                    data_p.pull_from(data_u, &w, false);
                }
            }

            // Reroot
            dfs_reroot(neighbors, data, yield_node_dp, root_init, root_init);
        }
    }
}
