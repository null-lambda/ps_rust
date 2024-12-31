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

use std::iter;

pub fn preorder_edge_lazy<'a>(
    neighbors: &'a [Vec<u32>],
    node: u32,
    parent: u32,
) -> impl Iterator<Item = (u32, u32)> + 'a {
    let mut stack = vec![(node, parent, neighbors[node as usize].iter())];
    iter::from_fn(move || {
        stack.pop().map(|(node, parent, mut iter_child)| {
            let child = *iter_child.next()?;
            stack.push((node, parent, iter_child));
            if child == parent {
                return None;
            }
            stack.push((child, node, neighbors[child as usize].iter()));
            Some((child, node))
        })
    })
    .flatten()
}

fn dfs_euler(
    children: &[Vec<u32>],
    euler_tour: &mut Vec<u32>,
    euler_in: &mut [u32],
    euler_out: &mut [u32],
    order: &mut u32,
    u: usize,
) {
    euler_tour.push(u as u32);
    euler_in[u] = *order;
    *order += 1;

    for &v in &children[u] {
        dfs_euler(children, euler_tour, euler_in, euler_out, order, v as usize);
    }

    euler_tour.push(u as u32);
    euler_out[u] = *order;
    *order += 1;
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

    pub struct LCA {
        n: usize,

        n_euler: usize,
        euler_tour: Vec<u32>,
        euler_in: Vec<u32>,

        block_size: usize,
        n_blocks: usize,
        height: Vec<u32>,
        min_sparse: Vec<Vec<(u32, u32)>>,

        block_mask: Vec<u32>,
        dp_in_block: Vec<Vec<Vec<u32>>>,
    }

    impl LCA {
        pub fn new(neighbors: &[Vec<u32>], root: usize) -> Self {
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
                min_sparse: vec![vec![(INF, UNSET); n_blocks]; log2(n_blocks as u32) as usize + 1],

                block_mask: vec![0; n_blocks],
                dp_in_block: vec![vec![]; 1 << block_size - 1],
            };

            this.build_euler(&neighbors, root as u32, root as u32);
            assert_eq!(this.euler_tour.len(), n_euler);

            this.build_sparse();
            this
        }

        fn build_euler(&mut self, neighbors: &[Vec<u32>], u: u32, p: u32) {
            self.euler_in[u as usize] = self.euler_tour.len() as u32;
            self.euler_tour.push(u);

            for &v in &neighbors[u as usize] {
                if v == p {
                    continue;
                }
                self.height[v as usize] = self.height[u as usize] + 1;
                self.build_euler(neighbors, v, u);
                self.euler_tour.push(u);
            }
        }

        fn key(&self, tour_idx: usize) -> (u32, u32) {
            if tour_idx >= self.n_euler {
                return (INF, UNSET);
            }

            let u = self.euler_tour[tour_idx] as usize;
            (self.height[u], u as u32)
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
                if !self.dp_in_block[mask].is_empty() {
                    continue;
                }
                self.dp_in_block[mask] = vec![vec![UNSET; self.block_size]; self.block_size];
                for l in 0..self.block_size {
                    self.dp_in_block[mask][l][l] = l as u32;
                    for r in l + 1..self.block_size {
                        self.dp_in_block[mask][l][r] = self.dp_in_block[mask][l][r - 1];
                        if self.key(b * self.block_size + self.dp_in_block[mask][l][r] as usize)
                            > self.key(b * self.block_size + r)
                        {
                            self.dp_in_block[mask][l][r] = r as u32;
                        }
                    }
                }
            }
        }

        fn min_in_block(&self, b: usize, l: usize, r: usize) -> (u32, u32) {
            let shift = self.dp_in_block[self.block_mask[b] as usize][l][r];
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

pub mod mo {
    pub fn even_odd_order(n: usize) -> impl Fn(u32, u32) -> (u32, i32) {
        assert!(n > 0);
        let bucket_size = (n as f64).sqrt() as u32;
        move |l, r| {
            let k = l / bucket_size;
            let l = if k % 2 == 0 { r as i32 } else { -(r as i32) };
            (k, l)
        }
    }

    // Mo's algorithm with space filling curve
    // https://codeforces.com/blog/entry/61203
    // https://codeforces.com/blog/entry/115590
    // use sort_with_cached_key instead of sort_unstable for better performance
    pub fn hilbert_order(n: usize) -> impl Fn(u32, u32) -> i64 {
        assert!(n > 0);
        let log2n_ceil = usize::BITS - 1 - n.next_power_of_two().leading_zeros();

        fn inner(mut x: u32, mut y: u32, mut exp: u32) -> i64 {
            let mut res = 0;
            let mut sign = 1;
            let mut rot = 0;

            while exp > 0 {
                let w_half = 1 << exp - 1;
                let quadrant = match (x < w_half, y < w_half) {
                    (true, true) => (rot + 0) % 4,
                    (false, true) => (rot + 1) % 4,
                    (false, false) => (rot + 2) % 4,
                    (true, false) => (rot + 3) % 4,
                };
                rot = match quadrant {
                    0 => (rot + 3) % 4,
                    1 => (rot + 0) % 4,
                    2 => (rot + 0) % 4,
                    3 => (rot + 1) % 4,
                    _ => unsafe { std::hint::unreachable_unchecked() },
                };

                x &= !w_half;
                y &= !w_half;

                let square_area_half = 1 << 2 * exp - 2;
                res += sign * quadrant as i64 * square_area_half;
                if quadrant == 0 || quadrant == 3 {
                    res += sign * (square_area_half - 1);
                    sign = -sign;
                };

                exp -= 1;
            }
            res
        }

        move |l, r| {
            debug_assert!(l < n as u32);
            debug_assert!(r < n as u32);
            inner(l, r, log2n_ceil as u32)
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let ws: Vec<u32> = (0..n).map(|_| input.value()).collect();
    let mut neighbors = vec![vec![]; n];
    for _ in 0..n - 1 {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        neighbors[u as usize].push(v);
        neighbors[v as usize].push(u);
    }

    let root = 0;
    let mut children = vec![vec![]; n];
    for (u, p) in preorder_edge_lazy(&neighbors, root, root) {
        children[p as usize].push(u);
    }

    let mut euler_in = vec![0; n];
    let mut euler_out = vec![0; n];
    let mut euler_tour = vec![];
    dfs_euler(
        &children,
        &mut euler_tour,
        &mut euler_in,
        &mut euler_out,
        &mut 0,
        root as usize,
    );

    let lca = lca::LCA::new(&neighbors, 0);

    let k = euler_tour.len();

    let mut queries: Vec<_> = (0..input.value())
        .map(|i| {
            let mut u = input.value::<usize>() - 1;
            let mut v = input.value::<usize>() - 1;
            if euler_in[u] > euler_in[v] {
                std::mem::swap(&mut u, &mut v);
            }

            let j = lca.get(u, v);
            if j == u {
                (euler_in[u], euler_in[v], None, i)
            } else if j == v {
                panic!()
            } else {
                (euler_out[u], euler_in[v], Some(euler_in[j]), i)
            }
        })
        .collect();

    // let key = mo::hilbert_order(k);
    let key = mo::even_odd_order(k);
    queries.sort_by_cached_key(|&(l, r, ..)| key(l, r));

    let mut ans = vec![i32::MAX; queries.len()];
    let (mut start, mut end) = (1, 0);
    let mut unique_count = 0i32;

    let w_bound = *ws.iter().max().unwrap() as usize + 1;
    let mut freq = vec![0; w_bound];
    let mut parity = vec![false; n];
    let mut toggle_state = |freq: &mut [i32], unique_count: &mut i32, j: usize| {
        let u = euler_tour[j as usize] as usize;
        let x = ws[u] as usize;
        parity[u] ^= true;
        if parity[u] {
            freq[x] += 1;
            if freq[x] == 1 {
                *unique_count += 1;
            }
        } else {
            if freq[x] == 1 {
                *unique_count -= 1;
            }
            freq[x] -= 1;
        }
    };

    for (l, r, j, i) in queries {
        while start > l {
            start -= 1;
            toggle_state(&mut freq, &mut unique_count, start as usize);
        }
        while end < r {
            end += 1;
            toggle_state(&mut freq, &mut unique_count, end as usize);
        }
        while start < l {
            toggle_state(&mut freq, &mut unique_count, start as usize);
            start += 1;
        }
        while end > r {
            toggle_state(&mut freq, &mut unique_count, end as usize);
            end -= 1;
        }

        if let Some(j) = j {
            toggle_state(&mut freq, &mut unique_count, j as usize);
        }
        ans[i] = unique_count;
        if let Some(j) = j {
            toggle_state(&mut freq, &mut unique_count, j as usize);
        }
    }

    for a in ans {
        writeln!(output, "{}", a).unwrap();
    }
}
