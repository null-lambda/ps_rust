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
    pub mod invertible {
        // O(n) rerooting dp for trees, with invertible pulling operation. (group action)

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

        pub trait RootData<E>: Clone {
            fn pull_from(&mut self, child: &Self, weight: &E, inv: bool);
            fn finalize(&mut self) {}

            // Override this method for further optimization.
            fn reroot_on_edge(&mut self, old_root: &Self, weight: &E) {
                let mut old_root = old_root.clone();
                old_root.pull_from(self, weight, true);
                old_root.finalize();

                self.pull_from(&old_root, weight, false);
                self.finalize();
            }
        }

        fn get_two<T>(xs: &mut [T], i: usize, j: usize) -> Option<(&mut T, &mut T)> {
            debug_assert!(i < xs.len() && j < xs.len());
            if i == j {
                return None;
            }
            let ptr = xs.as_mut_ptr();
            Some(unsafe { (&mut *ptr.add(i), &mut *ptr.add(j)) })
        }

        fn xor_assign_bytes<const N: usize>(xs: &mut [u8; N], ys: [u8; N]) {
            for (x, y) in xs.iter_mut().zip(&ys) {
                *x ^= *y;
            }
        }

        pub fn run<'a, const N: usize, E: Clone + AsBytes<N> + 'a, R: RootData<E>>(
            n: usize,
            edges: impl IntoIterator<Item = (u32, u32, E)>,
            data: &mut [R],
        ) {
            // Fast tree reconstruction with XOR-linked tree traversal
            // https://codeforces.com/blog/entry/135239
            let root = 0;
            let mut degree = vec![0; n];
            let mut xor_neighbors: Vec<(u32, [u8; N])> = vec![(0u32, [0u8; N]); n];
            for (u, v, w) in edges
                .into_iter()
                .flat_map(|(u, v, w)| [(u, v, w.clone()), (v, u, w)])
            {
                degree[u as usize] += 1;
                xor_neighbors[u as usize].0 ^= v;
                xor_assign_bytes(&mut xor_neighbors[u as usize].1, unsafe {
                    AsBytes::as_bytes(w.clone())
                });
            }

            // Upward propagation
            let mut topological_order = vec![];
            degree[root] += 2;
            for mut u in 0..n {
                while degree[u] == 1 {
                    let (p, w_encoded) = xor_neighbors[u];
                    degree[u] = 0;
                    degree[p as usize] -= 1;
                    xor_neighbors[p as usize].0 ^= u as u32;
                    xor_assign_bytes(&mut xor_neighbors[p as usize].1, w_encoded);
                    let w = unsafe { AsBytes::decode(w_encoded) };

                    data[u as usize].finalize();
                    let (data_u, data_p) =
                        unsafe { get_two(data, u as usize, p as usize).unwrap_unchecked() };
                    data_p.pull_from(data_u, &w, false);
                    topological_order.push((u, (p, w)));

                    u = p as usize;
                }
            }
            data[root].finalize();

            // Downward propagation
            for (u, (p, w)) in topological_order.into_iter().rev() {
                let (data_u, data_p) =
                    unsafe { get_two(data, u as usize, p as usize).unwrap_unchecked() };
                data_u.reroot_on_edge(&data_p, &w);
            }
        }
    }
}

#[derive(Clone)]
struct NodeData {
    count: [u32; 2],
    sum_sq: u64,
}

impl NodeData {
    fn new(color: u8) -> Self {
        let count = match color {
            0 => [1, 0],
            1 => [0, 1],
            _ => panic!(),
        };
        Self { count, sum_sq: 0 }
    }
}

impl reroot::invertible::RootData<()> for NodeData {
    fn pull_from(&mut self, child: &Self, (): &(), inv: bool) {
        if !inv {
            for i in 0..2 {
                self.count[i] += child.count[i];
            }
            self.sum_sq += child.count[0] as u64 * child.count[1] as u64;
        } else {
            unsafe { std::hint::unreachable_unchecked() };
        }
    }

    fn reroot_on_edge(&mut self, old_root: &Self, weight: &()) {
        let mut old_root = old_root.clone();
        for i in 0..2 {
            old_root.count[i] -= self.count[i];
        }

        self.pull_from(&old_root, weight, false);
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();

    let weights = (0..n).map(|_| input.value::<u8>());
    let weights: Vec<NodeData> = weights.map(NodeData::new).collect();
    let mut dp = weights.clone();

    let edges: Vec<_> = (0..n - 1)
        .map(|_| (input.value::<u32>() - 1, input.value::<u32>() - 1, ()))
        .collect();

    reroot::invertible::run(n, edges.iter().cloned(), &mut dp);
    let mut ans = vec![0u64; n];
    for i in 0..n {
        let s0 = dp[i].count[0] - weights[i].count[0];
        let s1 = dp[i].count[1] - weights[i].count[1];
        ans[i] = s0 as u64 * s1 as u64 - dp[i].sum_sq;
    }

    let q: usize = input.value();
    for _ in 0..q {
        let u = input.value::<usize>() - 1;
        writeln!(output, "{}", ans[u]).unwrap();
    }
}
