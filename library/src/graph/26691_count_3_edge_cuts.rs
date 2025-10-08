use std::{collections::HashMap, io::Write};

use jagged::CSR;

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

pub mod jagged {
    use std::fmt::Debug;
    use std::mem::MaybeUninit;
    use std::ops::{Index, IndexMut};

    // Compressed sparse row format, for static jagged array
    // Provides good locality for graph traversal
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CSR<T> {
        pub links: Vec<T>,
        head: Vec<u32>,
    }

    impl<T> Default for CSR<T> {
        fn default() -> Self {
            Self {
                links: vec![],
                head: vec![0],
            }
        }
    }

    impl<T: Clone> CSR<T> {
        pub fn from_pairs(n: usize, pairs: impl Iterator<Item = (u32, T)> + Clone) -> Self {
            let mut head = vec![0u32; n + 1];

            for (u, _) in pairs.clone() {
                debug_assert!(u < n as u32);
                head[u as usize] += 1;
            }
            for i in 0..n {
                head[i + 1] += head[i];
            }
            let mut data: Vec<_> = (0..head[n]).map(|_| MaybeUninit::uninit()).collect();

            for (u, v) in pairs {
                head[u as usize] -= 1;
                data[head[u as usize] as usize] = MaybeUninit::new(v.clone());
            }

            // Rustc is likely to perform in‑place iteration without new allocation.
            // [https://doc.rust-lang.org/stable/std/iter/trait.FromIterator.html#impl-FromIterator%3CT%3E-for-Vec%3CT%3E]
            let data = data
                .into_iter()
                .map(|x| unsafe { x.assume_init() })
                .collect();

            CSR { links: data, head }
        }
    }

    impl<T, I> FromIterator<I> for CSR<T>
    where
        I: IntoIterator<Item = T>,
    {
        fn from_iter<J>(iter: J) -> Self
        where
            J: IntoIterator<Item = I>,
        {
            let mut data = vec![];
            let mut head = vec![];
            head.push(0);

            let mut cnt = 0;
            for row in iter {
                data.extend(row.into_iter().inspect(|_| cnt += 1));
                head.push(cnt);
            }
            CSR { links: data, head }
        }
    }

    impl<T> CSR<T> {
        pub fn len(&self) -> usize {
            self.head.len() - 1
        }

        pub fn edge_range(&self, index: usize) -> std::ops::Range<usize> {
            self.head[index] as usize..self.head[index as usize + 1] as usize
        }
    }

    impl<T> Index<usize> for CSR<T> {
        type Output = [T];

        fn index(&self, index: usize) -> &Self::Output {
            &self.links[self.edge_range(index)]
        }
    }

    impl<T> IndexMut<usize> for CSR<T> {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            let es = self.edge_range(index);
            &mut self.links[es]
        }
    }

    impl<T> Debug for CSR<T>
    where
        T: Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let v: Vec<Vec<&T>> = (0..self.len()).map(|i| self[i].iter().collect()).collect();
            v.fmt(f)
        }
    }
}

const UNSET: u32 = !0;

mod dset {
    use std::{cell::Cell, mem};

    #[derive(Clone)]
    pub struct DisjointSet {
        // Represents parent if >= 0, size if < 0
        parent_or_size: Vec<Cell<i32>>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self {
                parent_or_size: vec![Cell::new(-1); n],
            }
        }

        fn get_parent_or_size(&self, u: usize) -> Result<usize, u32> {
            let x = self.parent_or_size[u].get();
            if x >= 0 {
                Ok(x as usize)
            } else {
                Err((-x) as u32)
            }
        }

        fn set_parent(&self, u: usize, p: usize) {
            self.parent_or_size[u].set(p as i32);
        }

        fn set_size(&self, u: usize, s: u32) {
            self.parent_or_size[u].set(-(s as i32));
        }

        pub fn find_root_with_size(&self, u: usize) -> (usize, u32) {
            match self.get_parent_or_size(u) {
                Ok(p) => {
                    let (root, size) = self.find_root_with_size(p);
                    self.set_parent(u, root);
                    (root, size)
                }
                Err(size) => (u, size),
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            self.find_root_with_size(u).0
        }

        // Returns true if two sets were previously disjoint
        pub fn merge(&mut self, u: usize, v: usize) -> bool {
            let (mut u, size_u) = self.find_root_with_size(u);
            let (mut v, size_v) = self.find_root_with_size(v);
            if u == v {
                return false;
            }

            if size_u < size_v {
                mem::swap(&mut u, &mut v);
            }
            self.set_parent(v, u);
            self.set_size(u, size_u + size_v);
            true
        }
    }
}

fn binom2(n: u32) -> u64 {
    n as u64 * (n as u64 - 1) / 2
}

fn binom3(n: u32) -> u64 {
    n as u64 * (n as u64 - 1) * (n as u64 - 2) / 6
}

fn count_3ec_naive(n: usize, edges: &[[u32; 2]], emult: &[u32]) -> u64 {
    let m = edges.len();

    let mut res = 0u64;

    let mut base_conn = dset::DisjointSet::new(n);
    for e in 0..m {
        if emult[e] == 0 {
            continue;
        }
        base_conn.merge(edges[e][0] as usize, edges[e][1] as usize);
    }
    let base_size: Vec<_> = (0..n).map(|u| base_conn.find_root_with_size(u).1).collect();

    for x in 0..m {
        for y in x + 1..m {
            for z in y + 1..m {
                let mut conn = dset::DisjointSet::new(n);
                for e in 0..m {
                    if emult[e] == 0 || [x, y, z].contains(&e) {
                        continue;
                    }
                    conn.merge(edges[e][0] as usize, edges[e][1] as usize);
                }
                res += (0..n).any(|u| base_size[u] != conn.find_root_with_size(u).1) as u64
                    * emult[x] as u64
                    * emult[y] as u64
                    * emult[z] as u64;
            }
        }
    }
    res
}

fn gen_min_cover_edge(
    edges: &[[u32; 2]],
    parent: &[u32],
    parent_edge: &[u32],
    t_in: &[u32],
    key_bound: u32,
    mut key: impl FnMut([u32; 2]) -> u32,
) -> Vec<u32> {
    let n = parent.len();

    let mut buckets = vec![vec![]; key_bound as usize];
    for (e, &[_u, v]) in edges.iter().enumerate() {
        if parent_edge[v as usize] == e as u32 {
            continue;
        }
        buckets[key(edges[e]) as usize].push(e as u32);
    }

    let mut group_by_key = dset::DisjointSet::new(n);
    let mut local_root: Vec<_> = (0..n as u32).map(|u| (t_in[u as usize], u)).collect();
    let mut min_cover = vec![UNSET; n];
    for eb in buckets.into_iter().flatten() {
        let [mut u, v] = edges[eb as usize];
        debug_assert!(t_in[u as usize] > t_in[v as usize], "Back edge");

        u = local_root[group_by_key.find_root(u as usize)].1;
        while t_in[u as usize] > t_in[v as usize] {
            debug_assert!(min_cover[u as usize] == UNSET);
            min_cover[u as usize] = eb;

            let p = parent[u as usize];

            let ru = group_by_key.find_root(u as usize);
            let rp = group_by_key.find_root(p as usize);

            group_by_key.merge(u as usize, p as usize);

            let rm = group_by_key.find_root(u as usize);
            local_root[rm] = local_root[ru as usize].min(local_root[rp as usize]);
            u = local_root[rm].1;
        }
    }

    min_cover
}

fn cubic() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();

    let edges: Vec<_> = (0..m)
        .map(|_| [input.value::<u32>() - 1, input.value::<u32>() - 1])
        .collect();
    let emult = vec![1u32; m];

    let n_3ec = count_3ec_naive(n, &edges, &emult);
    writeln!(output, "{}", n_3ec).unwrap();
}

fn linear() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();

    let mut edges: Vec<_> = (0..m)
        .map(|_| [input.value::<u32>() - 1, input.value::<u32>() - 1])
        .collect();
    let mut emult = vec![1u32; m];

    let mut rng = rand::SplitMix64::from_entropy().unwrap();
    let zobrist: Vec<_> = (0..m)
        .map(|_| (rng.next_u64() as u128) << 64 | rng.next_u64() as u128)
        .collect();

    let mut n_1ec = 0u32; // bridge
    let mut n_3ec = 0u64; // 3-edge cuts (including non-minimal ones)

    // Step 1. Count 1ec's and 2ec's,
    // and obtain 3-edge-connected components with a weighted virtual edges for 3-edge-connectivity.
    // (Clarification: An induced subgraph from a 3-edge-connected components **may not be connected**.
    {
        let neighbors = CSR::from_pairs(
            n,
            edges
                .iter()
                .enumerate()
                .flat_map(|(e, &[u, v])| [(u, (v, e as u32)), (v, (u, e as u32))]),
        );

        let mut lowpt = vec![0u32; n];
        let mut lowe = vec![UNSET; n];
        let mut n_cover = vec![0i32; n];
        let mut xor_zobrist_cover = vec![0; n];

        let mut t_in = vec![UNSET; n];
        let mut parent: Vec<_> = (0..n as u32).collect();
        let mut parent_edge = vec![UNSET; n];
        let mut timer = 0;

        let mut groups_1t1b = vec![vec![]; m];
        let mut groups_2t = HashMap::<_, Vec<u32>>::new();

        let mut current_edge: Vec<_> = (0..n)
            .map(|u| neighbors.edge_range(u).start as u32)
            .collect();
        for root in 0..n {
            if t_in[root] != UNSET {
                continue;
            }

            let mut u = root as u32;
            loop {
                let p = parent[u as usize];
                let ie = current_edge[u as usize];
                current_edge[u as usize] += 1;
                if ie == neighbors.edge_range(u as usize).start as u32 {
                    // On enter
                    t_in[u as usize] = timer;
                    lowpt[u as usize] = timer;
                    timer += 1;
                }
                if ie == neighbors.edge_range(u as usize).end as u32 {
                    // On exit
                    if p == u {
                        break;
                    }

                    match n_cover[u as usize] {
                        0 => {
                            n_1ec += 1; // Bridge (1-edge cut)
                            emult[parent_edge[u as usize] as usize] = 0;
                        }
                        1 => {
                            // 2-edge cut with 1 tree edge, 1 back edge
                            groups_1t1b[lowe[u as usize] as usize].push(u);
                        }
                        _ => {
                            // A candidate for 2-edge cut with 2 tree edge, sharing same fundamental cycles.
                            groups_2t
                                .entry(xor_zobrist_cover[u as usize])
                                .or_default()
                                .push(u);
                        }
                    }

                    if lowpt[u as usize] < lowpt[p as usize] {
                        lowpt[p as usize] = lowpt[u as usize];
                        lowe[p as usize] = lowe[u as usize];
                    }

                    n_cover[p as usize] += n_cover[u as usize];
                    xor_zobrist_cover[p as usize] ^= xor_zobrist_cover[u as usize];

                    u = p;
                    continue;
                }

                let (v, e) = neighbors.links[ie as usize];
                if e == parent_edge[u as usize] {
                    continue;
                }

                // Reorder edge
                if t_in[v as usize] == UNSET {
                    // Tree edge
                    edges[e as usize] = [u, v];
                    parent[v as usize] = u;
                    parent_edge[v as usize] = e;

                    u = v;
                } else if t_in[v as usize] < t_in[u as usize] {
                    // Back edge
                    edges[e as usize] = [u, v];
                    if t_in[v as usize] < lowpt[u as usize] {
                        lowpt[u as usize] = t_in[v as usize];
                        lowe[u as usize] = e;
                    }

                    n_cover[u as usize] += 1;
                    n_cover[v as usize] -= 1;
                    xor_zobrist_cover[u as usize] ^= zobrist[e as usize];
                    xor_zobrist_cover[v as usize] ^= zobrist[e as usize];
                }
            }
        }

        // 3-edge cuts where the minimum cut in subset is 1
        n_3ec += binom3(m as u32) - binom3(m as u32 - n_1ec);

        for (eb, path) in groups_1t1b.into_iter().enumerate() {
            if path.is_empty() {
                continue;
            }

            let cycle_size = 1 + path.len() as u32;

            // 3-edge cuts where the minimum cut in subset is 2
            n_3ec += binom2(cycle_size) * (m as u64 - n_1ec as u64 - cycle_size as u64)
                + binom3(cycle_size);

            emult[eb as usize] = 0;

            let [mut tail, head] = edges[eb as usize];
            for &u in &path {
                emult[parent_edge[u as usize] as usize] = 0;

                if tail != u {
                    edges.push([u, tail]);
                    emult.push(cycle_size);
                }

                tail = parent[u as usize];
            }
            if tail != head {
                edges.push([tail, head]);
                emult.push(cycle_size);
            }
        }

        for path in groups_2t.into_values() {
            if path.len() <= 1 {
                continue;
            }

            let cycle_size = path.len() as u32;

            // 3-edge cuts where the minimum cut in subset is 2
            n_3ec += binom2(cycle_size) * (m as u64 - n_1ec as u64 - cycle_size as u64)
                + binom3(cycle_size);

            let mut tail = path[0];
            let head = parent[*path.last().unwrap() as usize];
            edges.push([tail, head]);
            emult.push(cycle_size);

            for &u in &path {
                emult[parent_edge[u as usize] as usize] = 0;

                if tail != u {
                    edges.push([u, tail]);
                    emult.push(cycle_size);
                }

                tail = parent[u as usize];
            }
        }
    }

    let mut n = n;
    let (mut edges, mut emult): (Vec<_>, Vec<_>) = edges
        .into_iter()
        .zip(emult)
        .filter(|&(_, m)| m != 0)
        .unzip();

    // Step 2. For each 2-edge-connected graphs, count 3ec's.
    // n_3ec += count_3ec_naive(n, &edges, &emult);
    while edges.len() >= 3 {
        // ## Reference
        // - Determining 4-edge-connected components in linear time
        //   [https://arxiv.org/abs/2105.01699] (4 May 2021)

        let neighbors = CSR::from_pairs(
            n,
            edges
                .iter()
                .enumerate()
                .flat_map(|(e, &[u, v])| [(u, (v, e as u32)), (v, (u, e as u32))]),
        );

        let mut lowpt = vec![0u32; n];
        let mut lowe = vec![UNSET; n];
        let mut n_cover = vec![0i32; n];
        let mut xor_cover = vec![0; n];
        let mut xor_zobrist_cover = vec![0; n];

        let mut t_in = vec![UNSET; n];
        let mut parent: Vec<_> = (0..n as u32).collect();
        let mut parent_edge = vec![UNSET; n];
        let mut timer = 0;

        let mut current_edge: Vec<_> = (0..n)
            .map(|u| neighbors.edge_range(u).start as u32)
            .collect();

        let mut conn_through_back_edge = dset::DisjointSet::new(n);
        let mut three_ecs = vec![];

        for root in 0..n {
            if t_in[root] != UNSET {
                continue;
            }

            let mut u = root as u32;
            loop {
                let p = parent[u as usize];
                let ie = current_edge[u as usize];
                current_edge[u as usize] += 1;
                if ie == neighbors.edge_range(u as usize).start as u32 {
                    // On enter
                    t_in[u as usize] = timer;
                    lowpt[u as usize] = timer;
                    timer += 1;
                }
                if ie == neighbors.edge_range(u as usize).end as u32 {
                    // On exit
                    if p == u {
                        break;
                    }

                    if n_cover[u as usize] == 2 {
                        // A 3-edge cut with 1 tree edge, 2 back edge
                        three_ecs.push([
                            parent_edge[u as usize],
                            lowe[u as usize],
                            lowe[u as usize] ^ xor_cover[u as usize],
                        ]);
                    }

                    if lowpt[u as usize] < lowpt[p as usize] {
                        lowpt[p as usize] = lowpt[u as usize];
                        lowe[p as usize] = lowe[u as usize];
                    }

                    n_cover[p as usize] += n_cover[u as usize];
                    xor_cover[p as usize] ^= xor_cover[u as usize];
                    xor_zobrist_cover[p as usize] ^= xor_zobrist_cover[u as usize];

                    u = p;
                    continue;
                }

                let (v, e) = neighbors.links[ie as usize];
                if e == parent_edge[u as usize] {
                    continue;
                }

                // Reorder edge
                if t_in[v as usize] == UNSET {
                    // Tree edge
                    edges[e as usize] = [u, v];

                    parent[v as usize] = u;
                    parent_edge[v as usize] = e;

                    u = v;
                } else if t_in[v as usize] < t_in[u as usize] {
                    // Back edge
                    edges[e as usize] = [u, v];

                    conn_through_back_edge.merge(u as usize, v as usize);

                    if t_in[v as usize] < lowpt[u as usize] {
                        lowpt[u as usize] = t_in[v as usize];
                        lowe[u as usize] = e;
                    }

                    n_cover[u as usize] += 1;
                    n_cover[v as usize] -= 1;
                    xor_cover[u as usize] ^= e;
                    xor_cover[v as usize] ^= e;
                    xor_zobrist_cover[u as usize] ^= zobrist[e as usize];
                    xor_zobrist_cover[v as usize] ^= zobrist[e as usize];
                }
            }
        }

        let max_up =
            gen_min_cover_edge(&edges, &parent, &parent_edge, &t_in, n as u32, |[_u, v]| {
                n as u32 - 1 - t_in[v as usize]
            });
        let max_down =
            gen_min_cover_edge(&edges, &parent, &parent_edge, &t_in, n as u32, |[u, _v]| {
                n as u32 - 1 - t_in[u as usize]
            });
        let min_down =
            gen_min_cover_edge(&edges, &parent, &parent_edge, &t_in, n as u32, |[u, _v]| {
                t_in[u as usize]
            });

        let inv_xor_zobrist_cover: HashMap<_, u32> = (0..n as u32)
            .filter(|&u| parent_edge[u as usize] != UNSET)
            .map(|u| (xor_zobrist_cover[u as usize], u))
            .collect();
        for u in 0..n {
            let f = parent_edge[u as usize];
            if f == UNSET {
                continue;
            }

            let mut check = |g| {
                if g != UNSET {
                    if let Some(&v) = inv_xor_zobrist_cover
                        .get(&(xor_zobrist_cover[u as usize] ^ zobrist[g as usize]))
                    {
                        let e = parent_edge[v as usize];
                        three_ecs.push([e, f, g]);
                    }
                }
            };

            check(max_up[u as usize]);
            check(max_down[u as usize]);
            check(min_down[u as usize]);
            // if max_down[u as usize] != min_down[u as usize] {
            //     check(min_down[u as usize]);
            // }
        }

        // TODO: remove this block
        {
            for cut in &mut three_ecs {
                cut.sort_unstable();
            }
            three_ecs.sort_unstable();
            three_ecs.dedup();
        }

        for &[e, f, g] in &three_ecs {
            n_3ec += emult[e as usize] as u64 * emult[f as usize] as u64 * emult[g as usize] as u64;
        }

        let mut n_next = 0;
        let mut edges_next = vec![];
        let mut emult_next = vec![];
        let mut trans = vec![UNSET; n];
        for u in 0..n {
            let e = parent_edge[u];
            if e == UNSET {
                continue;
            }

            let [v, w] = edges[e as usize].map(|u| {
                let r = conn_through_back_edge.find_root(u as usize);
                if trans[r] == UNSET {
                    trans[r] = n_next as u32;
                    n_next += 1;
                }
                trans[r]
            });
            if v == w {
                continue;
            }

            edges_next.push([v, w]);
            emult_next.push(emult[e as usize]);
        }

        n = n_next;
        edges = edges_next;
        emult = emult_next;
    }

    writeln!(output, "{}", n_3ec).unwrap();
}

fn main() {
    if std::env::args().any(|s| s == "naive") {
        cubic()
    } else {
        linear()
    }
}
