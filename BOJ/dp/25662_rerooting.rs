use std::io::Write;

mod simple_io {
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

// Max-plus convolution, bounded up to k
#[derive(Clone)]
struct NodeData {
    size: u32,
    base: u32,
    score: Vec<u32>,
}

impl NodeData {
    fn singleton(weight: u32) -> Self {
        Self {
            size: 1,
            base: weight,
            score: vec![0],
        }
    }
}

struct Cx {
    n: usize,
    k: usize,
}

impl Cx {
    fn collapse(&self, node: &NodeData) -> u32 {
        node.score[self.k]
    }
}

impl reroot::DpSpec for Cx {
    type V = NodeData;
    type E = ();
    type F = Option<NodeData>;

    fn lift_to_action(&self, node: &Self::V, _weight: &Self::E) -> Self::F {
        Some(node.clone())
    }

    fn id_action(&self) -> Self::F {
        None
    }

    fn rake_action(&self, _node: &Self::V, lhs: &mut Self::F, rhs: &Self::F) {
        match lhs.as_mut() {
            None => *lhs = rhs.clone(),
            Some(lhs) => self.apply(lhs, rhs),
        }
    }

    fn finalize(&self, node: &mut Self::V) {
        node.score = std::iter::once(0)
            .chain(node.score.iter().map(|&x| x + node.base))
            .take(self.k + 1)
            .collect();
    }

    fn apply(&self, node: &mut Self::V, action: &Self::F) {
        let lhs = node;
        let Some(rhs) = action else { return };

        lhs.size += rhs.size;

        // Max-plus convolution up to k
        let size = lhs.size as usize;
        let residual_size = self.n - size; // Reduce redundant computation

        let l_max = (self.k - 1).min(lhs.score.len() + rhs.score.len() - 2);
        let l_min = l_max.saturating_sub(residual_size);
        let mut conv = vec![0; l_max - l_min + 1];
        for l in l_min..=l_max {
            for i in l.saturating_sub(rhs.score.len() - 1)..lhs.score.len().min(l + 1) {
                let j = l - i;
                conv[l - l_min] = conv[l - l_min].max(lhs.score[i] + rhs.score[j]);
            }
        }
        lhs.score.resize(l_max + 1, 0);
        lhs.score[l_min..=l_max].copy_from_slice(&conv);
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let k: usize = input.value();
    let xs = (0..n).map(|_| input.value::<u32>());

    let cx = Cx { n, k };
    let mut dp: Vec<_> = xs.map(NodeData::singleton).collect();
    let edges = (0..n - 1).map(|_| (input.value::<u32>() - 1, input.value::<u32>() - 1, ()));
    reroot::run(&cx, n, edges, &mut dp, &mut |_, _, _, _| {});

    for u in 0..n {
        write!(output, "{} ", cx.collapse(&dp[u])).unwrap();
    }
    writeln!(output).unwrap();
}
