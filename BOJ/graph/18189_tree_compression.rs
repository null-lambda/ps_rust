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

pub mod debug {
    pub fn with(#[allow(unused_variables)] f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
    }
}

pub mod hld {
    // Heavy-Light Decomposition
    #[inline(always)]
    pub unsafe fn assert_unchecked(b: bool) {
        if !b {
            std::hint::unreachable_unchecked();
        }
    }

    #[inline(always)]
    pub fn likely(b: bool) -> bool {
        #[cold]
        #[inline(always)]
        pub fn cold() {}

        if !b {
            cold();
        }
        b
    }

    pub const UNSET: u32 = u32::MAX;

    #[derive(Debug)]
    pub struct HLD {
        pub size: Vec<u32>,
        pub parent: Vec<u32>,
        pub heavy_child: Vec<u32>,
        pub chain_top: Vec<u32>,
        pub chain_bot: Vec<u32>,
        pub segmented_idx: Vec<u32>,
        pub topological_order: Vec<u32>,
    }

    impl HLD {
        pub fn len(&self) -> usize {
            self.parent.len()
        }

        pub fn from_edges<'a>(
            n: usize,
            edges: impl IntoIterator<Item = (u32, u32)>,
            root: usize,
            use_dfs_ordering: bool,
        ) -> Self {
            // Fast tree reconstruction with XOR-linked tree traversal
            // https://codeforces.com/blog/entry/135239
            let mut degree = vec![0u32; n];
            let mut xor_neighbors: Vec<u32> = vec![0u32; n];
            for (u, v) in edges.into_iter().flat_map(|(u, v)| [(u, v), (v, u)]) {
                debug_assert!(u != v);
                degree[u as usize] += 1;
                xor_neighbors[u as usize] ^= v;
            }

            let mut size = vec![1; n];
            let mut heavy_child = vec![UNSET; n];
            let mut chain_bot = vec![UNSET; n];
            degree[root] += 2;
            let mut topological_order = Vec::with_capacity(n);
            for mut u in 0..n {
                while degree[u] == 1 {
                    // Topological sort
                    let p = xor_neighbors[u];
                    topological_order.push(u as u32);
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

                    assert!(u != p as usize);
                    u = p as usize;
                }
            }
            topological_order.push(root as u32);
            assert!(topological_order.len() == n, "Invalid tree structure");

            let h = heavy_child[root];
            chain_bot[root] = if h == UNSET {
                root as u32
            } else {
                chain_bot[h as usize]
            };

            let mut parent = xor_neighbors;
            parent[root] = UNSET;

            // Downward propagation
            let mut chain_top = vec![root as u32; n];
            let mut segmented_idx = vec![UNSET; n];
            if !use_dfs_ordering {
                // A rearranged topological index continuous in a chain, for path queries
                let mut timer = 0;
                for mut u in topological_order.iter().copied().rev() {
                    if segmented_idx[u as usize] != UNSET {
                        continue;
                    }
                    let u0 = u;
                    loop {
                        chain_top[u as usize] = u0;
                        segmented_idx[u as usize] = timer;
                        timer += 1;
                        u = heavy_child[u as usize];
                        if u == UNSET {
                            break;
                        }
                    }
                }
            } else {
                // DFS ordering for path & subtree queries
                let mut offset = vec![0; n];
                for mut u in topological_order.iter().copied().rev() {
                    if segmented_idx[u as usize] != UNSET {
                        continue;
                    }

                    let mut p = parent[u as usize];
                    let mut timer = 0;
                    if likely(p != UNSET) {
                        timer = offset[p as usize] + 1;
                        offset[p as usize] += size[u as usize] as u32;
                    }

                    let u0 = u;
                    loop {
                        chain_top[u as usize] = u0;
                        offset[u as usize] = timer;
                        segmented_idx[u as usize] = timer;
                        timer += 1;

                        p = u as u32;
                        u = heavy_child[p as usize];
                        unsafe { assert_unchecked(u != p) };
                        if u == UNSET {
                            break;
                        }
                        offset[p as usize] += size[u as usize] as u32;
                    }
                }
            }

            Self {
                size,
                parent,
                heavy_child,
                chain_top,
                chain_bot,
                segmented_idx,
                topological_order,
            }
        }

        pub fn lca(&self, mut u: usize, mut v: usize) -> usize {
            debug_assert!(u < self.len() && v < self.len());
            while self.chain_top[u] != self.chain_top[v] {
                if self.segmented_idx[self.chain_top[u] as usize]
                    < self.segmented_idx[self.chain_top[v] as usize]
                {
                    std::mem::swap(&mut u, &mut v);
                }
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.segmented_idx[u] > self.segmented_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            u
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let root = 0;

    let mut count = 0;
    let mut sum = 0;
    let mut sum_sq = 0;
    (|| {
        let n: usize = input.value();
        let mut groups = vec![vec![]; n + 1];
        for u in 0..n {
            let c: u32 = input.value();
            groups[c as usize].push(u as u32);
        }

        let edges = (0..n - 1).map(|_| (input.value::<u32>() - 1, input.value::<u32>() - 1));
        let hld = hld::HLD::from_edges(n, edges, root, true);

        let mut sid_inv = vec![!0; n];
        for u in 0..n {
            sid_inv[hld.segmented_idx[u] as usize] = u as u32;
        }

        let mut level = vec![0u32; n];
        for &u in hld.topological_order.iter().rev().skip(1) {
            let p = hld.parent[u as usize];
            level[u as usize] = level[p as usize] + 1;
        }

        let nth_parent = |mut u: usize, mut k: usize| -> Option<usize> {
            loop {
                let top = hld.chain_top[u as usize] as usize;
                let d = (level[u] - level[top]) as usize;
                if k <= d {
                    return Some(sid_inv[hld.segmented_idx[u] as usize - k] as usize);
                }
                u = hld.parent[top] as usize;
                k -= d + 1;

                if u == hld::UNSET as usize {
                    return None;
                }
            }
        };

        let mut erase_count = vec![0i32; n];
        let mut degree = vec![0u32; n];
        let mut parent = vec![hld::UNSET; n];
        for g in groups {
            if g.len() <= 1 {
                continue;
            }

            let mut aux = g.clone();
            aux.sort_unstable_by_key(|&u| hld.segmented_idx[u as usize]);

            for i in 1..aux.len() {
                aux.push(hld.lca(aux[i - 1] as usize, aux[i] as usize) as u32);
            }
            aux.sort_unstable_by_key(|&u| hld.segmented_idx[u as usize]);
            aux.dedup();

            parent[aux[0] as usize] = hld::UNSET;
            degree[aux[0] as usize] = 0;
            for i in 1..aux.len() {
                degree[aux[i] as usize] = 1;
            }
            for i in 1..aux.len() {
                let p = hld.lca(aux[i - 1] as usize, aux[i] as usize);
                parent[aux[i] as usize] = p as u32;
                degree[p as usize] += 1;
            }

            let mut erase_complement = false;
            for &u in &g {
                if degree[u as usize] >= 2 {
                    return;
                }

                if u == aux[0] {
                    erase_count[root] += 1;
                    erase_complement = true;
                } else {
                    erase_count[u as usize] += 1;
                }
            }

            if erase_complement {
                for &u in &aux[1..] {
                    if parent[u as usize] == aux[0] {
                        let d = (level[u as usize] - level[aux[0] as usize]) as usize;
                        let c = nth_parent(u as usize, d - 1).unwrap();
                        erase_count[c as usize] -= 1;
                    }
                }
            }
        }

        for &u in hld.topological_order.iter().rev() {
            let p = hld.parent[u as usize];
            if p != hld::UNSET {
                erase_count[u as usize] += erase_count[p as usize];
            }

            if erase_count[u as usize] == 0 {
                let c = u as u64 + 1;
                count += 1;
                sum += c;
                sum_sq += c * c;
            }
        }
    })();

    writeln!(output, "{}", count).unwrap();
    writeln!(output, "{}", sum).unwrap();
    writeln!(output, "{}", sum_sq).unwrap();
}
