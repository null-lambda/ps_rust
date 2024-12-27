use std::io::Write;

mod simple_io {
    use std::string::*;

    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

pub mod reroot {
    pub mod lazy {
        // O(n) rerooting dp for trees, with combinable pulling operation. (Monoid action)
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
            fn combine_action(&self, lhs: &mut Self::F, rhs: &Self::F);
            fn apply(&mut self, action: &Self::F);
            fn finalize(&mut self);
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

        pub fn run<
            'a,
            const N: usize,
            E: Clone + Default + AsBytes<N> + 'a,
            R: RootData<E = E> + Clone,
        >(
            n_verts: usize,
            edges: impl IntoIterator<Item = (u32, u32, E)>,
            data: &mut [R],
            yield_edge_dp: &mut impl FnMut(usize, &R::F, &R::F, &E),
        ) -> Vec<(u32, E)> {
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
            let mut action_upward = vec![R::id_action(); n_verts];
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
                    sum_u.finalize();
                    action_upward[u as usize] = sum_u.lift_to_action(&w);
                    data[p as usize].apply(&action_upward[u as usize]);

                    topological_order.push((u as u32, p, w));
                    u = p as usize;
                }
            }
            let mut parent = xor_neighbors;
            parent[root as usize].0 = UNSET;
            topological_order.push((root as u32, UNSET, E::default()));
            data[root].finalize();

            // Downward propagation
            let mut action_exclusive = vec![R::id_action(); n_verts];
            for (u, p, w) in topological_order.into_iter().rev() {
                let action_from_parent;
                if p != UNSET {
                    let mut sum_exclusive = data_orig[p as usize].clone();
                    sum_exclusive.apply(&action_exclusive[u as usize]);
                    sum_exclusive.finalize();
                    action_from_parent = sum_exclusive.lift_to_action(&w);

                    let sum_u = &mut data[u as usize];
                    sum_u.apply(&action_from_parent);
                    sum_u.finalize();
                    yield_edge_dp(
                        u as usize,
                        &action_upward[u as usize],
                        &action_from_parent,
                        &w,
                    );
                } else {
                    action_from_parent = R::id_action();
                }

                if first_child[u as usize] != UNSET {
                    let mut prefix = action_from_parent.clone();
                    let last = for_each_in_list(&xor_siblings, first_child[u as usize], |v| {
                        action_exclusive[v as usize] = prefix.clone();
                        data_orig[u as usize]
                            .combine_action(&mut prefix, &action_upward[v as usize]);
                    });

                    let mut postfix = R::id_action();
                    for_each_in_list(&xor_siblings, last, |v| {
                        data_orig[u as usize]
                            .combine_action(&mut action_exclusive[v as usize], &postfix);
                        data_orig[u as usize]
                            .combine_action(&mut postfix, &action_upward[v as usize]);
                    });
                }
            }

            let parent = parent
                .into_iter()
                .map(|(u, w)| (u, unsafe { AsBytes::decode(w) }))
                .collect();
            parent
        }
    }
}

#[derive(Clone, Default, Debug)]
struct NodeDp {
    diam: u32,
    depth: u32,
}

impl reroot::lazy::RootData for NodeDp {
    type E = ();
    type F = NodeDp;

    fn lift_to_action(&self, (): &()) -> NodeDp {
        NodeDp {
            diam: self.diam,
            depth: self.depth + 1,
        }
    }

    fn id_action() -> NodeDp {
        NodeDp { diam: 0, depth: 0 }
    }

    fn combine_action(&self, lhs: &mut NodeDp, rhs: &NodeDp) {
        lhs.apply(rhs);
    }

    fn apply(&mut self, action: &NodeDp) {
        self.diam = self.diam.max(action.diam).max(self.depth + action.depth);
        self.depth = self.depth.max(action.depth);
    }

    fn finalize(&mut self) {}
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();
    let edges = (0..n - 1).map(|_| (input.value::<u32>() - 1, input.value::<u32>() - 1, ()));

    let mut ans = vec![0; n];
    let parent = reroot::lazy::run(
        n,
        edges,
        &mut vec![NodeDp::default(); n],
        &mut |u, e1, e2, ()| {
            ans[u] = e1.diam + e2.diam + 1;
        },
    );

    for _ in 0..q {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        let x = if parent[u as usize].0 == v { u } else { v };
        writeln!(output, "{}", ans[x as usize]).unwrap();
    }
}
