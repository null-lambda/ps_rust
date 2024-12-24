pub mod reroot {
    pub mod invertible {
        // O(n) rerooting dp for trees, with invertible pulling operation. (group action)
        use crate::jagged::Jagged;

        pub trait AsBytes<const N: usize> {
            unsafe fn as_bytes(self) -> [u8; N];
            unsafe fn decode(bytes: [u8; N]) -> Self;
        }

        #[macro_export]
        macro_rules! impl_as_bytes {
            ($T:ty, $N: ident) => {
                const $N: usize = std::mem::size_of::<$T>();

                impl crate::reroot::invertible::AsBytes<$N> for $T {
                    unsafe fn as_bytes(self) -> [u8; $N] {
                        std::mem::transmute::<$T, [u8; $N]>(self)
                    }

                    unsafe fn decode(bytes: [u8; $N]) -> $T {
                        std::mem::transmute::<[u8; $N], $T>(bytes)
                    }
                }
            };
        }
        pub use impl_as_bytes;

        impl_as_bytes!((), __N_UNIT);
        impl_as_bytes!(u32, __N_U32);
        impl_as_bytes!(u64, __N_U64);
        impl_as_bytes!(i32, __N_I32);
        impl_as_bytes!(i64, __N_I64);

        pub trait RootData<E> {
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

        fn rec_reroot<'a, E: 'a, R: RootData<E> + Clone>(
            neighbors: &'a impl Jagged<'a, (u32, E)>,
            data: &mut [R],
            yield_root_data: &mut impl FnMut(usize, &R),
            u: usize,
            p: usize,
        ) {
            yield_root_data(u, &data[u]);
            for (v, w) in neighbors.get(u) {
                if *v as usize == p {
                    continue;
                }
                reroot_on_edge(data, *v as usize, w, u);
                rec_reroot(neighbors, data, yield_root_data, *v as usize, u);
                reroot_on_edge(data, u, w, *v as usize);
            }
        }

        fn toposort<'a, const N: usize, E: Clone + Default + 'a>(
            neighbors: &'a impl Jagged<'a, (u32, E)>,
        ) -> (Vec<u32>, Vec<(u32, E)>)
        where
            E: AsBytes<N>,
        {
            // Fast tree reconstruction with XOR-linked tree traversal.
            // Restriction: The graph should be undirected i.e. (u, v, weight) in E <=> (v, u, weight) in E.
            // https://codeforces.com/blog/entry/135239

            let n = neighbors.len();
            if n == 1 {
                return (vec![0], vec![(0, E::default())]);
            }

            let mut degree = vec![0; n];
            let mut xor: Vec<(u32, [u8; N])> = vec![(0u32, [0u8; N]); n];

            fn xor_assign_bytes<const N: usize>(xs: &mut [u8; N], ys: [u8; N]) {
                for (x, y) in xs.iter_mut().zip(&ys) {
                    *x ^= *y;
                }
            }

            for u in 0..n {
                for (v, w) in neighbors.get(u) {
                    degree[u] += 1;
                    xor[u].0 ^= v;
                    xor_assign_bytes(&mut xor[u].1, unsafe { AsBytes::as_bytes(w.clone()) });
                }
            }
            degree[0] += 2;

            let mut toposort = vec![];
            for mut u in 0..n {
                while degree[u] == 1 {
                    let (v, w_encoded) = xor[u];
                    toposort.push(u as u32);
                    degree[u] = 0;
                    degree[v as usize] -= 1;
                    xor[v as usize].0 ^= u as u32;
                    xor_assign_bytes(&mut xor[v as usize].1, w_encoded);
                    u = v as usize;
                }
            }
            toposort.push(0);
            toposort.reverse();

            // Note: Copying entire vec (from xor to parent) is necessary, since
            // transmuting from (u32, [u8; N]) to (u32, E)
            // or *const (u32, [u8; N]) to *const (u32, E) is UB. (tuples are
            // #[align(rust)] struct and the field ordering is not guaranteed)
            // We may try messing up with custom #[repr(C)] structs, or just trust the compiler.
            let parent: Vec<(u32, E)> = xor
                .into_iter()
                .map(|(v, w)| (v, unsafe { AsBytes::decode(w) }))
                .collect();
            (toposort, parent)
        }

        pub fn run<'a, const N: usize, E: Clone + Default + 'a, R: RootData<E> + Clone>(
            neighbors: &'a impl Jagged<'a, (u32, E)>,
            data: &mut [R],
            yield_node_dp: &mut impl FnMut(usize, &R),
        ) where
            E: AsBytes<N>,
        {
            let (order, parent) = toposort(neighbors);
            let root = order[0] as usize;

            // Init tree DP
            for &u in order.iter().rev() {
                data[u as usize].finalize();

                let (p, w) = &parent[u as usize];
                if u as usize != root {
                    let (data_u, data_p) =
                        unsafe { get_two(data, u as usize, *p as usize).unwrap_unchecked() };
                    data_p.pull_from(data_u, &w, false);
                }
            }

            // Reroot
            rec_reroot(neighbors, data, yield_node_dp, root, root);
        }
    }
}
