pub mod reroot {
    pub mod lazy {
        // O(n) rerooting dp for trees, with combinable pulling operation. (Monoid action)

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

        pub fn run<E: Clone + Default, R: RootData<E = E> + Clone>(
            neighbors: &[Vec<(u32, E)>],
            data: &[R],
            root: usize,
            yield_node_dp: &mut impl FnMut(usize, R),
        ) {
            let n = neighbors.len();
            let mut preorder = vec![];
            let mut parent = vec![(root, E::default()); n];
            let mut stack = vec![(root, root)];
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
            let mut sum_upward = data.to_owned();
            let mut action_upward = vec![R::id_action(); n];
            for &u in preorder[1..].iter().rev() {
                let (p, w) = &parent[u];
                let (data_u, data_p) =
                    unsafe { get_two(&mut sum_upward, u, *p).unwrap_unchecked() };
                action_upward[u] = data_u.finalize().lift_to_action(w);
                data_p.apply(&action_upward[u]);
            }

            // Reroot
            let mut action_from_parent = vec![R::id_action(); n];
            for &u in &preorder {
                let mut sum_u = sum_upward[u].clone();
                sum_u.apply(&action_from_parent[u]);
                yield_node_dp(u, sum_u.finalize());

                let &(p, _) = &parent[u];
                let n_child = neighbors[u].len() - (u != root) as usize;
                match n_child {
                    0 => {}
                    1 => {
                        for (v, w) in &neighbors[u] {
                            if *v as usize == p {
                                continue;
                            }
                            let exclusive = action_from_parent[u].clone();
                            let mut sum_exclusive = data[u].clone();
                            sum_exclusive.apply(&exclusive);
                            action_from_parent[*v as usize] =
                                sum_exclusive.finalize().lift_to_action(&w);
                        }
                    }
                    _ => {
                        let mut prefix: Vec<R::F> = neighbors[u]
                            .iter()
                            .map(|(v, _)| {
                                if *v as usize == p {
                                    action_from_parent[u].clone()
                                } else {
                                    action_upward[*v as usize].clone()
                                }
                            })
                            .collect();
                        let mut postfix = prefix.clone();
                        for i in (2..neighbors[u].len()).rev() {
                            postfix[i - 1] = data[u].combine_action(&postfix[i - 1], &postfix[i]);
                        }
                        for i in 1..neighbors[u].len() - 1 {
                            prefix[i] = data[u].combine_action(&prefix[i - 1], &prefix[i]);
                        }

                        for (i, (v, w)) in neighbors[u].iter().enumerate() {
                            if *v as usize == p {
                                continue;
                            }
                            let exclusive = if i == 0 {
                                postfix[1].clone()
                            } else if i == neighbors[u].len() - 1 {
                                prefix[neighbors[u].len() - 2].clone()
                            } else {
                                data[u].combine_action(&prefix[i - 1], &postfix[i + 1])
                            };

                            let mut sum_exclusive = data[u].clone();
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
