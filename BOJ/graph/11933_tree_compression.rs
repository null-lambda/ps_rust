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

pub mod lca {
    // O(1) LCA with O(n) preprocessing
    // Farach-Colton and Bender algorithm
    // https://cp-algorithms.com/graph/lca_farachcoltonbender.html
    const UNSET: u32 = u32::MAX;
    const INF: u32 = u32::MAX;

    fn log2(x: u32) -> u32 {
        assert!(x > 0);
        u32::BITS - 1 - x.leading_zeros()
    }

    #[derive(Clone, Copy)]
    struct CmpBy<K, V>(K, V);

    impl<K: PartialEq, V> PartialEq for CmpBy<K, V> {
        fn eq(&self, other: &Self) -> bool {
            self.0.eq(&other.0)
        }
    }

    impl<K: Eq, V> Eq for CmpBy<K, V> {}

    impl<K: PartialOrd, V> PartialOrd for CmpBy<K, V> {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.0.partial_cmp(&other.0)
        }
    }

    impl<K: Ord, V> Ord for CmpBy<K, V> {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.cmp(&other.0)
        }
    }

    pub struct LCA {
        n: usize,

        n_euler: usize,
        euler_tour: Vec<u32>,
        pub euler_in: Vec<u32>,

        block_size: usize,
        n_blocks: usize,
        height: Vec<u32>,
        min_sparse: Vec<Vec<CmpBy<u32, u32>>>,

        block_mask: Vec<u32>,
        min_idx_in_block: Vec<Vec<Vec<u32>>>,
    }

    impl LCA {
        pub fn new<E>(neighbors: &[Vec<(u32, E)>], root: usize) -> Self {
            let n = neighbors.len();
            let n_euler = 2 * n - 1;
            let block_size = (log2(n_euler as u32) as usize / 2).max(1);
            let n_blocks = n_euler.div_ceil(block_size);

            let mut this = LCA {
                n,

                n_euler,
                euler_in: vec![UNSET; n],
                euler_tour: vec![],

                block_size,
                n_blocks,
                height: vec![0; n],
                min_sparse: vec![
                    vec![CmpBy(INF, UNSET); n_blocks];
                    log2(n_blocks as u32) as usize + 1
                ],

                block_mask: vec![0; n_blocks],
                min_idx_in_block: vec![vec![]; 1 << block_size - 1],
            };

            this.build_euler(&neighbors, root as u32, root as u32);
            assert_eq!(this.euler_tour.len(), n_euler);

            this.build_sparse();
            this
        }

        fn build_euler<E>(&mut self, neighbors: &[Vec<(u32, E)>], u: u32, p: u32) {
            self.euler_in[u as usize] = self.euler_tour.len() as u32;
            self.euler_tour.push(u);

            for &(v, _) in &neighbors[u as usize] {
                if v == p {
                    continue;
                }
                self.height[v as usize] = self.height[u as usize] + 1;
                self.build_euler(neighbors, v, u);
                self.euler_tour.push(u);
            }
        }

        fn key(&self, tour_idx: usize) -> CmpBy<u32, u32> {
            let u = self.euler_tour[tour_idx] as usize;
            CmpBy(self.height[u], u as u32)
        }

        fn build_sparse(&mut self) {
            for i in 0..self.n_euler {
                let b = i / self.block_size;
                self.min_sparse[0][b] = self.min_sparse[0][b].min(self.key(i));
            }
            for exp in 1..self.min_sparse.len() {
                for i in 0..self.n_blocks {
                    let j = i + (1 << exp - 1);
                    self.min_sparse[exp][i] = self.min_sparse[exp - 1][i];
                    if j < self.n_blocks {
                        self.min_sparse[exp][i] =
                            self.min_sparse[exp][i].min(self.min_sparse[exp - 1][j]);
                    }
                }
            }

            for i in 0..self.n_euler {
                let (b, s) = (i / self.block_size, i % self.block_size);
                if s > 0 && self.key(i - 1) < self.key(i) {
                    self.block_mask[b] |= 1 << s - 1;
                }
            }

            for b in 0..self.n_blocks {
                let mask = self.block_mask[b] as usize;
                if !self.min_idx_in_block[mask].is_empty() {
                    continue;
                }
                self.min_idx_in_block[mask] = vec![vec![UNSET; self.block_size]; self.block_size];
                for l in 0..self.block_size {
                    self.min_idx_in_block[mask][l][l] = l as u32;
                    for r in l + 1..self.block_size {
                        self.min_idx_in_block[mask][l][r] = self.min_idx_in_block[mask][l][r - 1];
                        if b * self.block_size + r < self.n_euler
                            && self.key(
                                b * self.block_size + self.min_idx_in_block[mask][l][r] as usize,
                            ) > self.key(b * self.block_size + r)
                        {
                            self.min_idx_in_block[mask][l][r] = r as u32;
                        }
                    }
                }
            }
        }

        fn min_in_block(&self, b: usize, l: usize, r: usize) -> CmpBy<u32, u32> {
            let mask = self.block_mask[b] as usize;
            let shift = self.min_idx_in_block[mask][l][r];
            self.key(b * self.block_size + shift as usize)
        }

        pub fn get(&self, u: usize, v: usize) -> usize {
            debug_assert!(u < self.n);
            debug_assert!(v < self.n);

            let l = self.euler_in[u].min(self.euler_in[v]) as usize;
            let r = self.euler_in[u].max(self.euler_in[v]) as usize;

            let (bl, sl) = (l / self.block_size, l % self.block_size);
            let (br, sr) = (r / self.block_size, r % self.block_size);
            if bl == br {
                return self.min_in_block(bl, sl, sr).1 as usize;
            }

            let prefix = self.min_in_block(bl, sl, self.block_size - 1);
            let suffix = self.min_in_block(br, 0, sr);
            let mut res = prefix.min(suffix);
            if bl + 1 < br {
                let exp = log2((br - bl - 1) as u32) as usize;
                res = res.min(self.min_sparse[exp][bl + 1]);
                res = res.min(self.min_sparse[exp][br - (1 << exp)]);
            }

            res.1 as usize
        }
    }
}

const UNSET: u32 = u32::MAX;
const INF: u64 = u64::MAX / 4;

fn dfs_dist(neighbors: &[Vec<(u32, u64)>], dist: &mut [u64], u: usize, p: usize) {
    for &(v, d) in &neighbors[u] {
        if v as usize == p {
            continue;
        }
        dist[v as usize] = dist[u] + d;
        dfs_dist(neighbors, dist, v as usize, u);
    }
}

#[derive(Debug, Clone, Copy)]
struct NodeData {
    min_dist: [u64; 2],
}

impl NodeData {
    fn singleton(color: u8) -> Self {
        Self {
            min_dist: match color {
                0 => [INF, INF],
                1 => [0, INF],
                2 => [INF, 0],
                _ => panic!(),
            },
        }
    }

    fn pull_from(&mut self, other: &Self, w: u64) {
        for i in 0..2 {
            self.min_dist[i] = self.min_dist[i].min(other.min_dist[i] + w);
        }
    }

    fn finalize(&self, ans: &mut u64) {
        *ans = (*ans).min(self.min_dist[0] + self.min_dist[1]);
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();

    let mut neighbors = vec![vec![]; n];
    for _ in 0..n - 1 {
        let u: usize = input.value();
        let v: usize = input.value();
        let d: u64 = input.value();
        neighbors[u].push((v as u32, d));
        neighbors[v].push((u as u32, d));
    }

    let root = 0;
    let mut dist = vec![0; n];
    dfs_dist(&neighbors, &mut dist, root, root);

    let lca = lca::LCA::new(&neighbors, root);
    let euler_in = &lca.euler_in;

    let mut color = vec![0u8; n];
    let mut parents = vec![(UNSET, INF); n];
    let mut dp = vec![NodeData::singleton(0); n];
    for _ in 0..q {
        let s: usize = input.value();
        let t: usize = input.value();

        let mut relevant = vec![];
        for _ in 0..s {
            let x: usize = input.value();
            relevant.push(x);
            color[x] = 1u8;
        }
        for _ in 0..t {
            let x: usize = input.value();
            relevant.push(x);
            color[x] = 2u8;
        }

        relevant.sort_unstable_by_key(|&x| euler_in[x]);
        for i in 1..relevant.len() {
            relevant.push(lca.get(relevant[i - 1], relevant[i]));
        }
        relevant.sort_unstable_by_key(|&x| euler_in[x]);
        relevant.dedup();

        for i in 1..relevant.len() {
            let (u, v) = (relevant[i - 1], relevant[i]);
            let j = lca.get(u, v);
            parents[v] = (j as u32, dist[v] - dist[j]);
        }

        let mut ans = INF;
        for &u in &relevant {
            dp[u] = NodeData::singleton(color[u]);
        }
        for &u in relevant.iter().rev() {
            dp[u].finalize(&mut ans);

            let (p, w) = parents[u];
            if p != UNSET {
                let dp_u = dp[u];
                dp[p as usize].pull_from(&dp_u, w);
            }
        }
        assert!(ans < INF);
        writeln!(output, "{}", ans).unwrap();

        for &u in &relevant {
            color[u] = 0;
        }
        for &u in &relevant[..relevant.len() - 1] {
            parents[u] = (UNSET, INF);
        }
    }
}
