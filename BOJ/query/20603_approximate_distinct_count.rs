// https://en.wikipedia.org/wiki/HyperLogLog
// # HyperLogLog count-distinct estimator:
// use (roughly) (sum_i 2^floor(log_2 min_s hash(c_si)))^-1 as an relative estimator,
// where s in [1,m] denotes sample index and i denotes item index.

use std::io::Write;

use segtree::Monoid;

mod simple_io {
    pub struct InputAtOnce {
        iter: std::str::SplitAsciiWhitespace<'static>,
    }

    impl InputAtOnce {
        pub fn token(&mut self) -> &'static str {
            self.iter.next().unwrap_or_default()
        }

        pub fn try_value<T: std::str::FromStr>(&mut self) -> Option<T> {
            self.token().parse().ok()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.try_value().unwrap()
        }
    }

    pub fn stdin() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(Box::new(buf));
        let iter = buf.split_ascii_whitespace();
        InputAtOnce { iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

mod rand {
    // Written in 2015 by Sebastiano Vigna (vigna@acm.org)
    // https://xoshiro.di.unimi.it/splitmix64.c
    use std::ops::Range;

    pub struct SplitMix64(u64);

    impl SplitMix64 {
        pub fn new(seed: u64) -> Self {
            assert_ne!(seed, 0);
            Self(seed)
        }

        // Available on x86-64 and target feature rdrand only.
        #[cfg(target_arch = "x86_64")]
        pub fn from_entropy() -> Option<Self> {
            let mut seed = 0;
            unsafe { (std::arch::x86_64::_rdrand64_step(&mut seed) == 1).then(|| Self(seed)) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        pub fn from_entropy() -> Self {
            use std::time::{SystemTime, UNIX_EPOCH};
            let seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            Self(seed as u64)
        }

        pub fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = self.0;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        }

        pub fn range_u64(&mut self, range: Range<u64>) -> u64 {
            let Range { start, end } = range;
            debug_assert!(start < end);

            let width = end - start;
            let test = (u64::MAX - width) % width;
            loop {
                let value = self.next_u64();
                if value >= test {
                    return start + value % width;
                }
            }
        }

        pub fn shuffle<T>(&mut self, xs: &mut [T]) {
            let n = xs.len();
            if n == 0 {
                return;
            }

            for i in 0..n - 1 {
                let j = self.range_u64(i as u64..n as u64) as usize;
                xs.swap(i, j);
            }
        }
    }
}

pub mod hld {
    // Heavy-Light Decomposition
    pub const UNSET: u32 = u32::MAX;

    fn inv_perm(perm: &[u32]) -> Vec<u32> {
        let mut res = vec![UNSET; perm.len()];
        for u in 0..perm.len() as u32 {
            res[perm[u as usize] as usize] = u;
        }
        res
    }

    #[derive(Debug)]
    pub struct HLD {
        pub parent: Vec<u32>,
        pub size: Vec<u32>,
        pub t_in: Vec<u32>,
        pub tour: Vec<u32>,

        pub heavy_child: Vec<u32>,
        pub chain_top: Vec<u32>,
        pub chain_bot: Vec<u32>,
    }

    impl HLD {
        pub fn len(&self) -> usize {
            self.parent.len()
        }

        pub fn from_edges<'a>(
            n: usize,
            edges: impl IntoIterator<Item = [u32; 2]>,
            root: usize,
        ) -> Self {
            // Fast tree reconstruction with XOR-linked tree traversal
            // https://codeforces.com/blog/entry/135239
            let mut degree = vec![0u32; n];
            let mut xor_neighbors: Vec<u32> = vec![0u32; n];
            for [u, v] in edges {
                debug_assert!(u != v);
                degree[u as usize] += 1;
                degree[v as usize] += 1;
                xor_neighbors[u as usize] ^= v;
                xor_neighbors[v as usize] ^= u;
            }

            let mut size = vec![1; n];
            let mut heavy_child = vec![UNSET; n];
            let mut chain_bot = vec![UNSET; n];
            degree[root] += 2;
            let mut toposort = Vec::with_capacity(n);
            for mut u in 0..n {
                while degree[u] == 1 {
                    // Topological sort
                    let p = xor_neighbors[u];
                    toposort.push(u as u32);
                    degree[u] = 0;
                    degree[p as usize] -= 1;
                    xor_neighbors[p as usize] ^= u as u32;

                    // Upward propagation
                    size[p as usize] += size[u as usize];
                    let h = &mut heavy_child[p as usize];
                    if *h == UNSET || size[*h as usize] < size[u as usize] {
                        *h = u as u32;
                    }

                    let h = heavy_child[u as usize];
                    chain_bot[u] = if h == UNSET {
                        u as u32
                    } else {
                        chain_bot[h as usize]
                    };

                    debug_assert!(u != p as usize);
                    u = p as usize;
                }
            }
            toposort.push(root as u32);
            assert!(toposort.len() == n, "Invalid tree structure");

            let h = heavy_child[root];
            chain_bot[root] = if h == UNSET {
                root as u32
            } else {
                chain_bot[h as usize]
            };

            let mut parent = xor_neighbors;
            parent[root] = UNSET;

            // Preorder index, continuous in any chain
            let mut t_in = vec![UNSET; n];
            let mut chain_top = vec![root as u32; n];
            let mut offset = vec![0; n];

            // Downward propagation
            for mut u in toposort.into_iter().rev() {
                if t_in[u as usize] != UNSET {
                    continue;
                }

                let mut p = parent[u as usize];
                let mut timer = 0;
                if p != UNSET {
                    timer = offset[p as usize] + 1;
                    offset[p as usize] += size[u as usize] as u32;
                }

                let u0 = u;
                loop {
                    chain_top[u as usize] = u0;
                    offset[u as usize] = timer;
                    t_in[u as usize] = timer;
                    timer += 1;

                    p = u as u32;
                    u = heavy_child[p as usize];
                    if u == UNSET {
                        break;
                    }
                    offset[p as usize] += size[u as usize] as u32;
                }
            }

            let tour = inv_perm(&t_in);
            Self {
                size,
                parent,
                heavy_child,
                chain_top,
                chain_bot,
                t_in,
                tour,
            }
        }

        pub fn lca(&self, mut u: usize, mut v: usize) -> usize {
            debug_assert!(u < self.len() && v < self.len());
            while self.chain_top[u] != self.chain_top[v] {
                if self.t_in[self.chain_top[u] as usize] < self.t_in[self.chain_top[v] as usize] {
                    std::mem::swap(&mut u, &mut v);
                }
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.t_in[u] < self.t_in[v] {
                std::mem::swap(&mut u, &mut v);
            }
            v
        }

        pub fn chains_in_path(
            &self,
            mut u: usize,
            mut v: usize,
            mut visit_subchain: impl FnMut(usize, usize, bool, bool), /* (top, bot, is_top_lca, on_left) */
        ) {
            debug_assert!(u < self.len() && v < self.len());
            let mut on_left = true;
            while self.chain_top[u] != self.chain_top[v] {
                if self.t_in[self.chain_top[u] as usize] < self.t_in[self.chain_top[v] as usize] {
                    std::mem::swap(&mut u, &mut v);
                    on_left ^= true;
                }
                visit_subchain(self.chain_top[u] as usize, u, false, on_left);
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.t_in[u] < self.t_in[v] {
                std::mem::swap(&mut u, &mut v);
                on_left ^= true;
            }
            visit_subchain(v, u, true, on_left);
        }

        pub fn nth_parent(&self, mut u: usize, mut k: u64) -> Result<usize, u64> {
            loop {
                let top = self.chain_top[u as usize] as usize;
                let d = (self.t_in[u] - self.t_in[top]) as u64;
                if k <= d {
                    return Ok(self.tour[self.t_in[u] as usize - k as usize] as usize);
                }
                u = self.parent[top] as usize;
                k -= d + 1;

                if u == UNSET as usize {
                    return Err(k + 1);
                }
            }
        }
    }
}

pub mod segtree {
    use std::ops::Range;

    pub trait Monoid {
        type X;
        const IS_COMMUTATIVE: bool = false;
        fn id(&self) -> Self::X;
        fn op(&self, a: &Self::X, b: &Self::X) -> Self::X;
    }

    #[derive(Debug)]
    pub struct SegTree<M>
    where
        M: Monoid,
    {
        n: usize,
        sum: Vec<M::X>,
        monoid: M,
    }

    impl<M: Monoid> SegTree<M> {
        pub fn with_size(n: usize, monoid: M) -> Self {
            Self {
                n,
                sum: (0..2 * n).map(|_| monoid.id()).collect(),
                monoid,
            }
        }

        pub fn from_iter<I>(iter: I, monoid: M) -> Self
        where
            I: IntoIterator<Item = M::X>,
            I::IntoIter: ExactSizeIterator<Item = M::X>,
        {
            let iter = iter.into_iter();
            let n = iter.len();
            let mut sum: Vec<_> = (0..n).map(|_| monoid.id()).chain(iter).collect();
            for i in (0..n).rev() {
                sum[i] = monoid.op(&sum[i << 1], &sum[i << 1 | 1]);
            }
            Self { n, sum, monoid }
        }

        pub fn modify(&mut self, mut idx: usize, f: impl FnOnce(&mut M::X)) {
            debug_assert!(idx < self.n);
            idx += self.n;
            f(&mut self.sum[idx]);
            while idx > 1 {
                idx >>= 1;
                self.sum[idx] = self.monoid.op(&self.sum[idx << 1], &self.sum[idx << 1 | 1]);
            }
        }

        pub fn get(&self, idx: usize) -> &M::X {
            &self.sum[idx + self.n]
        }

        pub fn mapped_sum_range<N: Monoid>(
            &self,
            range: Range<usize>,
            codomain: &N,
            morphism: impl Fn(&M::X) -> N::X,
        ) -> N::X {
            let Range { mut start, mut end } = range;
            if start >= end {
                return codomain.id();
            }
            debug_assert!(start < self.n && end <= self.n);
            start += self.n;
            end += self.n;

            if N::IS_COMMUTATIVE {
                let mut result = codomain.id();
                while start < end {
                    if start & 1 != 0 {
                        result = codomain.op(&result, &morphism(&self.sum[start]));
                    }
                    if end & 1 != 0 {
                        result = codomain.op(&morphism(&self.sum[end - 1]), &result);
                    }
                    start = (start + 1) >> 1;
                    end >>= 1;
                }
                result
            } else {
                let (mut result_left, mut result_right) = (codomain.id(), codomain.id());
                while start < end {
                    if start & 1 != 0 {
                        result_left = codomain.op(&result_left, &morphism(&self.sum[start]));
                    }
                    if end & 1 != 0 {
                        result_right = codomain.op(&morphism(&self.sum[end - 1]), &result_right);
                    }
                    start = (start + 1) >> 1;
                    end >>= 1;
                }
                codomain.op(&result_left, &result_right)
            }
        }

        pub fn sum_all(&self) -> &M::X {
            assert!(self.n.is_power_of_two());
            &self.sum[1]
        }
    }

    impl<M: Monoid> SegTree<M>
    where
        M::X: Clone,
    {
        pub fn sum_range(&self, range: Range<usize>) -> M::X {
            self.mapped_sum_range(range, &self.monoid, |x| x.clone())
        }
    }
}

#[allow(non_camel_case_types)]
#[repr(align(32))]
#[derive(Clone, Debug)]
struct u8a<const W: usize>([u8; W]);

impl<const W: usize> u8a<W> {
    fn vmax(&self, other: &Self) -> Self {
        #[target_feature(enable = "avx2")]
        unsafe fn inner<const W: usize>(xs: &u8a<W>, ys: &u8a<W>) -> u8a<W> {
            u8a(std::array::from_fn(|i| xs.0[i].max(ys.0[i])))
        }

        unsafe { inner(self, other) }
    }
}

struct VectorMaxOp<const W: usize>;

impl<const W: usize> segtree::Monoid for VectorMaxOp<W> {
    type X = Option<u8a<W>>;

    const IS_COMMUTATIVE: bool = true;

    fn id(&self) -> Self::X {
        None
    }

    fn op(&self, a: &Self::X, b: &Self::X) -> Self::X {
        match (a, b) {
            (None, _) => b.clone(),
            (_, None) => a.clone(),
            (Some(a), Some(b)) => Some(a.vmax(b)),
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    const H: usize = 44;
    const W: usize = 128;

    for _ in 0..input.value() {
        let n: usize = input.value();
        let q: usize = input.value();

        let color: Vec<_> = (0..n).map(|_| input.value::<u32>() - 1).collect();
        let edges = (0..n - 1).map(|_| [input.value::<u32>() - 1, input.value::<u32>() - 1]);
        let hld = hld::HLD::from_edges(n, edges, 0);

        let mut rng = rand::SplitMix64::from_entropy().unwrap();
        // let mut rng = rand::SplitMix64::new(42);
        let hash = (0..n)
            .map(|_| {
                u8a::<W>(std::array::from_fn(|_| {
                    (rng.next_u64().leading_zeros() as u8).min(H as u8 - 1)
                }))
            })
            .collect::<Vec<_>>();

        let mut st = segtree::SegTree::from_iter(
            hld.tour
                .iter()
                .map(|&u| Some(hash[color[u as usize] as usize].clone())),
            VectorMaxOp,
        );

        let mut cnt = 0;
        for _ in 0..q {
            match input.token() {
                "1" => {
                    let u = (input.value::<u32>() ^ cnt) - 1;
                    let c = (input.value::<u32>() ^ cnt) - 1;
                    st.modify(hld.t_in[u as usize] as usize, |x| {
                        *x = Some(hash[c as usize].clone())
                    });
                }
                _ => {
                    let a = (input.value::<u32>() ^ cnt) - 1;
                    let b = (input.value::<u32>() ^ cnt) - 1;
                    let c = (input.value::<u32>() ^ cnt) - 1;
                    let d = (input.value::<u32>() ^ cnt) - 1;

                    let mut det = 0;
                    for (s, x, y) in [(-1i64, a, b), (1i64, c, d)] {
                        let mut h = None;
                        hld.chains_in_path(x as usize, y as usize, |u, v, _, _| {
                            let tu = hld.t_in[u as usize];
                            let tv = hld.t_in[v as usize];
                            h = VectorMaxOp.op(&h, &st.sum_range(tu as usize..tv as usize + 1));
                        });
                        let h = h.unwrap();

                        let mut acc = 0i64;
                        for b in h.0 {
                            acc += 1 << (H as u8 - 1 - b);
                        }
                        det += s * acc;
                    }

                    let ans = det > 0;
                    cnt += ans as u32;

                    let ans = if ans { "Yes" } else { "No" };
                    writeln!(output, "{}", ans).unwrap();
                }
            }
        }
    }
}
