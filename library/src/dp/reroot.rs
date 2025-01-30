pub mod reroot {
    // O(n) rerooting dp for trees with combinable, non-invertible pulling operation. (Monoid action)
    // https://codeforces.com/blog/entry/124286
    // https://github.com/koosaga/olympiad/blob/master/Library/codes/data_structures/all_direction_tree_dp.cpp
    pub trait AsBytes<const N: usize> {
        unsafe fn as_bytes(self) -> [u8; N];
        unsafe fn decode(bytes: [u8; N]) -> Self;
    }

    #[macro_export]
    macro_rules! impl_as_bytes {
        ($T:ty, $N: ident) => {
            const $N: usize = std::mem::size_of::<$T>();

            impl crate::reroot::AsBytes<$N> for $T {
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

    pub trait DpSpec {
        type E: Clone; // edge weight
        type V: Clone; // Subtree aggregate on a node
        type F: Clone; // pulling operation (edge dp)
        fn lift_to_action(&self, node: &Self::V, weight: &Self::E) -> Self::F;
        fn id_action(&self) -> Self::F;
        fn rake_action(&self, node: &Self::V, lhs: &mut Self::F, rhs: &Self::F);
        fn apply(&self, node: &mut Self::V, action: &Self::F);
        fn finalize(&self, node: &mut Self::V);
    }

    fn xor_assign_bytes<const N: usize>(xs: &mut [u8; N], ys: [u8; N]) {
        for (x, y) in xs.iter_mut().zip(&ys) {
            *x ^= *y;
        }
    }

    const UNSET: u32 = !0;

    fn for_each_in_list(xor_links: &[u32], start: u32, mut visitor: impl FnMut(u32)) -> u32 {
        let mut u = start;
        let mut prev = UNSET;
        loop {
            visitor(u);
            let next = xor_links[u as usize] ^ prev;
            if next == UNSET {
                return u;
            }
            prev = u;
            u = next;
        }
    }

    pub fn run<'a, const N: usize, R>(
        cx: &R,
        n_verts: usize,
        edges: impl IntoIterator<Item = (u32, u32, R::E)>,
        data: &mut [R::V],
        yield_edge_dp: &mut impl FnMut(usize, &R::F, &R::F, &R::E),
    ) where
        R: DpSpec,
        R::E: Default + AsBytes<N>,
    {
        // Fast tree reconstruction with XOR-linked traversal
        // https://codeforces.com/blog/entry/135239
        let root = 0;
        let mut degree = vec![0; n_verts];
        let mut xor_neighbors: Vec<(u32, [u8; N])> = vec![(0u32, [0u8; N]); n_verts];
        for (u, v, w) in edges
            .into_iter()
            .flat_map(|(u, v, w)| [(u, v, w.clone()), (v, u, w)])
        {
            degree[u as usize] += 1;
            xor_neighbors[u as usize].0 ^= v;
            xor_assign_bytes(&mut xor_neighbors[u as usize].1, unsafe {
                AsBytes::as_bytes(w)
            });
        }

        // Upward propagation
        let data_orig = data.to_owned();
        let mut action_upward = vec![cx.id_action(); n_verts];
        let mut topological_order = vec![];
        let mut first_child = vec![UNSET; n_verts];
        let mut xor_siblings = vec![UNSET; n_verts];
        degree[root] += 2;
        for mut u in 0..n_verts {
            while degree[u as usize] == 1 {
                let (p, w_encoded) = xor_neighbors[u as usize];
                degree[u as usize] = 0;
                degree[p as usize] -= 1;
                xor_neighbors[p as usize].0 ^= u as u32;
                xor_assign_bytes(&mut xor_neighbors[p as usize].1, w_encoded);
                let w = unsafe { AsBytes::decode(w_encoded) };

                let c = first_child[p as usize];
                xor_siblings[u as usize] = c ^ UNSET;
                if c != UNSET {
                    xor_siblings[c as usize] ^= u as u32 ^ UNSET;
                }
                first_child[p as usize] = u as u32;

                let mut sum_u = data[u as usize].clone();
                cx.finalize(&mut sum_u);
                action_upward[u as usize] = cx.lift_to_action(&sum_u, &w);
                cx.apply(&mut data[p as usize], &action_upward[u as usize]);

                topological_order.push((u as u32, p, w));
                u = p as usize;
            }
        }
        topological_order.push((root as u32, UNSET, R::E::default()));
        cx.finalize(&mut data[root]);

        // Downward propagation
        let mut action_exclusive = vec![cx.id_action(); n_verts];
        for (u, p, w) in topological_order.into_iter().rev() {
            let action_from_parent;
            if p != UNSET {
                let mut sum_exclusive = data_orig[p as usize].clone();
                cx.apply(&mut sum_exclusive, &action_exclusive[u as usize]);
                cx.finalize(&mut sum_exclusive);
                action_from_parent = cx.lift_to_action(&sum_exclusive, &w);

                let sum_u = &mut data[u as usize];
                cx.apply(sum_u, &action_from_parent);
                cx.finalize(sum_u);
                yield_edge_dp(
                    u as usize,
                    &action_upward[u as usize],
                    &action_from_parent,
                    &w,
                );
            } else {
                action_from_parent = cx.id_action();
            }

            if first_child[u as usize] != UNSET {
                let mut prefix = action_from_parent.clone();
                let last = for_each_in_list(&xor_siblings, first_child[u as usize], |v| {
                    action_exclusive[v as usize] = prefix.clone();
                    cx.rake_action(
                        &data_orig[u as usize],
                        &mut prefix,
                        &action_upward[v as usize],
                    );
                });

                let mut postfix = cx.id_action();
                for_each_in_list(&xor_siblings, last, |v| {
                    cx.rake_action(
                        &data_orig[u as usize],
                        &mut action_exclusive[v as usize],
                        &postfix,
                    );
                    cx.rake_action(
                        &data_orig[u as usize],
                        &mut postfix,
                        &action_upward[v as usize],
                    );
                });
            }
        }
    }
}
