pub mod reroot {
    pub mod lazy {
        // O(n) rerooting dp for trees, with combinable pulling operation. (Monoid action)
        // https://codeforces.com/blog/entry/124286
        // https://github.com/koosaga/olympiad/blob/master/Library/codes/data_structures/all_direction_tree_dp.cpp
        use crate::jagged::Jagged;

        pub trait AsBytes<const N: usize> {
            unsafe fn as_bytes(self) -> [u8; N];
            unsafe fn decode(bytes: [u8; N]) -> Self;
        }

        #[macro_export]
        macro_rules! impl_as_bytes {
            ($T:ty, $N: ident) => {
                const $N: usize = std::mem::size_of::<$T>();

                impl crate::reroot::lazy::AsBytes<$N> for $T {
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

        pub trait RootData {
            type E: Clone; // edge weight
            type F: Clone; // pulling operation (edge dp)
            fn lift_to_action(&self, weight: &Self::E) -> Self::F;
            fn id_action() -> Self::F;
            fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F;
            fn apply(&mut self, action: &Self::F);
            fn finalize(&self) -> Self;
        }

        fn get_two<T>(xs: &mut [T], i: usize, j: usize) -> Option<(&mut T, &mut T)> {
            debug_assert!(i < xs.len() && j < xs.len());
            if i == j {
                return None;
            }
            let ptr = xs.as_mut_ptr();
            Some(unsafe { (&mut *ptr.add(i), &mut *ptr.add(j)) })
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

        pub fn run<
            'a,
            const N: usize,
            E: Clone + Default + AsBytes<N> + 'a,
            R: RootData<E = E> + Clone,
        >(
            neighbors: &'a impl Jagged<'a, (u32, E)>,
            data: &[R],
            yield_node_dp: &mut impl FnMut(usize, R),
            yield_edge_dp: &mut impl FnMut(&R::F, &R::F, &E),
        ) {
            let n = neighbors.len();
            let (order, parent) = toposort(neighbors);
            let root = order[0] as usize;

            // Init tree DP
            let mut sum_upward = data.to_owned();
            let mut action_upward = vec![R::id_action(); n];
            for &u in order[1..].iter().rev() {
                let (p, w) = &parent[u as usize];
                let (data_u, data_p) =
                    unsafe { get_two(&mut sum_upward, u as usize, *p as usize).unwrap_unchecked() };
                action_upward[u as usize] = data_u.finalize().lift_to_action(w);
                data_p.apply(&action_upward[u as usize]);
            }

            // Reroot
            let mut action_from_parent = vec![R::id_action(); n];
            for &u in &order {
                let (p, w) = &parent[u as usize];

                let mut sum_u = sum_upward[u as usize].clone();
                sum_u.apply(&action_from_parent[u as usize]);
                yield_node_dp(u as usize, sum_u.finalize());
                if u as usize != root {
                    yield_edge_dp(
                        &action_upward[u as usize],
                        &action_from_parent[u as usize],
                        w,
                    );
                }

                let n_child = neighbors.get(u as usize).len() - (u as usize != root) as usize;
                match n_child {
                    0 => {}
                    1 => {
                        for (v, w) in neighbors.get(u as usize) {
                            if *v == *p {
                                continue;
                            }
                            let exclusive = action_from_parent[u as usize].clone();
                            let mut sum_exclusive = data[u as usize].clone();
                            sum_exclusive.apply(&exclusive);
                            action_from_parent[*v as usize] =
                                sum_exclusive.finalize().lift_to_action(&w);
                        }
                    }
                    _ => {
                        let mut prefix: Vec<R::F> = neighbors
                            .get(u as usize)
                            .map(|(v, _)| {
                                if *v == *p {
                                    action_from_parent[u as usize].clone()
                                } else {
                                    action_upward[*v as usize].clone()
                                }
                            })
                            .collect();
                        let mut postfix = prefix.clone();
                        for i in (1..neighbors.get(u as usize).len()).rev() {
                            postfix[i - 1] =
                                data[u as usize].combine_action(&postfix[i - 1], &postfix[i]);
                        }
                        for i in 1..neighbors.get(u as usize).len() {
                            prefix[i] = data[u as usize].combine_action(&prefix[i - 1], &prefix[i]);
                        }

                        for (i, (v, w)) in neighbors.get(u as usize).enumerate() {
                            if *v == *p {
                                continue;
                            }
                            let exclusive = if i == 0 {
                                postfix[1].clone()
                            } else if i == neighbors.get(u as usize).len() - 1 {
                                prefix[neighbors.get(u as usize).len() - 2].clone()
                            } else {
                                data[u as usize].combine_action(&prefix[i - 1], &postfix[i + 1])
                            };

                            let mut sum_exclusive = data[u as usize].clone();
                            sum_exclusive.apply(&exclusive);
                            action_from_parent[*v as usize] =
                                sum_exclusive.finalize().lift_to_action(&w);
                        }
                    }
                }
            }
        }
    }
}
