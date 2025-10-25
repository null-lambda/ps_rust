use std::io::Write;

use hld::UNSET;

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

fn global_push_down(hld: &hld::HLD, edge_weight: &mut [u32]) {
    for &u in &hld.tour[1..] {
        let p = hld.parent[u as usize];
        edge_weight[u as usize] += edge_weight[p as usize];
    }
}

fn ascend_to_last(hld: &hld::HLD, mut u: usize, mut pred: impl FnMut(usize) -> bool) -> usize {
    if !pred(u) {
        return UNSET as usize;
    }

    let mut t;
    loop {
        t = hld.chain_top[u as usize] as usize;
        let p = hld.parent[t] as usize;
        if p == UNSET as usize || !pred(p) {
            break;
        }

        u = p;
    }

    let d = hld.t_in[u] - hld.t_in[t];
    let mut left = hld.t_in[t] as usize;
    let mut right = left + d as usize;
    while left < right {
        let mid = left + right >> 1;
        if !pred(hld.tour[mid] as usize) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    hld.tour[left] as usize
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for _ in 0..input.value() {
        let n: usize = input.value();
        let z: i64 = input.value();
        let xs: Vec<i64> = (0..n).map(|_| input.value()).collect();

        let mut parent = vec![UNSET; n + 1];
        let mut j = 0;
        for i in 0..n {
            j = j.max(i + 1);
            while j < n && xs[i] + z >= xs[j] {
                j += 1;
            }
            parent[i] = j as u32;
        }
        let small = hld::HLD::from_edges(n + 1, (0..n).map(|u| [u as u32, parent[u]]), n);

        let mut ds_small = vec![1u32; n + 1];
        ds_small[n] = 0;
        global_push_down(&small, &mut ds_small);

        let mut junction = vec![UNSET; n + 1];
        for i in 0..n {
            junction[i] = small.lca(i, i + 1) as u32;
        }
        let large = hld::HLD::from_edges(n + 1, (0..n).map(|u| [u as u32, junction[u]]), n);
        let mut ds_large = vec![0u32; n + 1];
        for u in 0..n {
            ds_large[u] = ds_small[u] + ds_small[u + 1] - 2 * ds_small[junction[u] as usize];
        }
        global_push_down(&large, &mut ds_large);

        for _ in 0..input.value() {
            let l = input.value::<usize>() - 1;
            let r = input.value::<usize>() - 1;

            let mut ans = 0;
            let x = ascend_to_last(&large, l, |u| u <= r);
            ans += ds_large[l] - ds_large[x];

            let p = ascend_to_last(&small, x, |u| u <= r);
            ans += ds_small[x] - ds_small[p] + 1;

            if x + 1 <= r {
                let q = ascend_to_last(&small, x + 1, |u| u <= r);
                ans += ds_small[x + 1] - ds_small[q] + 1;
            }

            writeln!(output, "{}", ans).unwrap();
        }
    }
}
