use std::{collections::BTreeMap, io::Write};

use jagged::Jagged;

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

pub mod jagged {
    use std::fmt::Debug;
    use std::iter;
    use std::mem::MaybeUninit;

    // Trait for painless switch between different representations of a jagged array
    pub trait Jagged<'a, T: 'a> {
        type ItemRef: ExactSizeIterator<Item = &'a T>;
        fn len(&self) -> usize;
        fn get(&'a self, u: usize) -> Self::ItemRef;
    }

    impl<'a, T, C> Jagged<'a, T> for C
    where
        C: AsRef<[Vec<T>]> + 'a,
        T: 'a,
    {
        type ItemRef = std::slice::Iter<'a, T>;
        fn len(&self) -> usize {
            <Self as AsRef<[Vec<T>]>>::as_ref(self).len()
        }
        fn get(&'a self, u: usize) -> Self::ItemRef {
            let res = <Self as AsRef<[Vec<T>]>>::as_ref(self)[u].iter();
            res
        }
    }

    // Compressed sparse row format for jagged array
    // Provides good locality for graph traversal, but works only for static ones.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CSR<T> {
        data: Vec<T>,
        head: Vec<u32>,
    }

    impl<T> Debug for CSR<T>
    where
        T: Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let v: Vec<Vec<&T>> = (0..self.len()).map(|i| self.get(i).collect()).collect();
            v.fmt(f)
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
            CSR { data, head }
        }
    }

    impl<T: Clone> CSR<T> {
        pub fn from_assoc_list(n: usize, pairs: &[(u32, T)]) -> Self {
            let mut head = vec![0u32; n + 1];

            for &(u, _) in pairs {
                debug_assert!(u < n as u32);
                head[u as usize + 1] += 1;
            }
            for i in 2..n + 1 {
                head[i] += head[i - 1];
            }
            let mut data: Vec<_> = iter::repeat_with(|| MaybeUninit::uninit())
                .take(head[n] as usize)
                .collect();
            let mut pos = head.clone();

            for (u, v) in pairs {
                data[pos[*u as usize] as usize] = MaybeUninit::new(v.clone());
                pos[*u as usize] += 1;
            }

            let data = std::mem::ManuallyDrop::new(data);
            let data = unsafe {
                Vec::from_raw_parts(data.as_ptr() as *mut T, data.len(), data.capacity())
            };

            CSR { data, head }
        }
    }

    impl<'a, T: 'a> Jagged<'a, T> for CSR<T> {
        type ItemRef = std::slice::Iter<'a, T>;

        fn len(&self) -> usize {
            self.head.len() - 1
        }

        fn get(&'a self, u: usize) -> Self::ItemRef {
            self.data[self.head[u] as usize..self.head[u + 1] as usize].iter()
        }
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
        euler_in: Vec<u32>,

        block_size: usize,
        n_blocks: usize,
        height: Vec<u32>,
        min_sparse: Vec<Vec<CmpBy<u32, u32>>>,

        block_mask: Vec<u32>,
        min_idx_in_block: Vec<Vec<Vec<u32>>>,
    }

    impl LCA {
        pub fn from_graph(
            n: usize,
            edges: impl IntoIterator<Item = (u32, u32)>,
            root: usize,
        ) -> Self {
            let mut neighbors = vec![vec![]; n];
            for (u, v) in edges {
                neighbors[u as usize].push(v);
                neighbors[v as usize].push(u);
            }

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

    // Build a max cartesian tree from inorder traversal
    fn max_cartesian_tree<T>(
        n: usize,
        iter: impl IntoIterator<Item = (usize, T)>,
    ) -> (Vec<u32>, usize)
    where
        T: Ord,
    {
        let mut parent = vec![UNSET; n];

        // Monotone stack
        let mut stack = vec![];
        for (u, h) in iter {
            let u = u as u32;

            let mut c = None;
            while let Some((prev, _)) = stack.last() {
                if prev > &h {
                    break;
                }
                c = stack.pop();
            }
            if let Some(&(_, p)) = stack.last() {
                parent[u as usize] = p;
            }
            if let Some((_, c)) = c {
                parent[c as usize] = u;
            }
            stack.push((h, u));
        }
        let root = (0..n).find(|&i| parent[i] == UNSET).unwrap();
        (parent, root)
    }

    pub struct StaticRangeMax<T> {
        xs: Vec<T>,
        cartesian_tree: LCA,
    }

    impl<T: Ord> StaticRangeMax<T> {
        pub fn from_iter(xs: impl Iterator<Item = T>) -> Self {
            let xs: Vec<_> = xs.into_iter().collect();
            let n = xs.len();
            assert!(n >= 1);

            let (parent, root) = max_cartesian_tree(n, xs.iter().enumerate());
            Self {
                xs,
                cartesian_tree: LCA::from_graph(
                    n,
                    (0..n)
                        .map(|i| (i as u32, parent[i]))
                        .filter(|&(_, v)| v != UNSET),
                    root,
                ),
            }
        }

        pub fn argmax_range(&self, range: std::ops::Range<usize>) -> usize {
            debug_assert!(range.start < range.end && range.end <= self.xs.len());
            self.cartesian_tree.get(range.start, range.end - 1)
        }

        pub fn max_range(&self, range: std::ops::Range<usize>) -> &T {
            &self.xs[self.argmax_range(range)]
        }
    }
}

pub mod centroid {
    /// Centroid Decomposition
    use crate::jagged::Jagged;

    pub fn init_size<'a, E: 'a>(
        neighbors: &'a impl Jagged<'a, (u32, E)>,
        size: &mut [u32],
        u: usize,
        p: usize,
    ) {
        size[u] = 1;
        for &(v, _) in neighbors.get(u) {
            if v as usize == p {
                continue;
            }
            init_size(neighbors, size, v as usize, u);
            size[u] += size[v as usize];
        }
    }

    fn reroot_to_centroid<'a, _E: 'a>(
        neighbors: &'a impl Jagged<'a, (u32, _E)>,
        size: &mut [u32],
        visited: &[bool],
        mut u: usize,
    ) -> usize {
        let threshold = (size[u] + 1) / 2;
        let mut p = u;
        'outer: loop {
            for &(v, _) in neighbors.get(u) {
                if v as usize == p || visited[v as usize] {
                    continue;
                }
                if size[v as usize] >= threshold {
                    size[u] -= size[v as usize];
                    size[v as usize] += size[u];

                    p = u;
                    u = v as usize;
                    continue 'outer;
                }
            }
            return u;
        }
    }

    pub fn build_centroid_tree<'a, _E: 'a + Clone>(
        neighbors: &'a impl Jagged<'a, (u32, _E)>,
        size: &mut [u32],
        visited: &mut [bool],
        parent_centroid: &mut [u32],
        init: usize,
    ) -> usize {
        let root = reroot_to_centroid(neighbors, size, visited, init);
        visited[root] = true;

        for &(v, _) in neighbors.get(root) {
            if visited[v as usize] {
                continue;
            }
            let sub_root =
                build_centroid_tree(neighbors, size, visited, parent_centroid, v as usize);
            parent_centroid[sub_root] = root as u32;
        }
        root
    }
}

const INF: u32 = u32::MAX / 3;
const UNSET: u32 = u32::MAX;

fn build_depth<'a>(
    neighbors: &'a impl Jagged<'a, (u32, ())>,
    depth: &mut [u32],
    u: usize,
    p: usize,
) {
    for &(v, ()) in neighbors.get(u) {
        if v as usize == p {
            continue;
        }
        depth[v as usize] = depth[u] + 1;
        build_depth(neighbors, depth, v as usize, u);
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut edges = vec![];
    for _ in 0..n - 1 {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        edges.push((u, (v, ())));
        edges.push((v, (u, ())));
    }
    let neighbors = jagged::CSR::from_assoc_list(n, &edges);

    let base_root = 0;
    let lca = lca::LCA::from_graph(
        n,
        edges
            .iter()
            .map(|&(u, (v, _))| (u, v))
            .filter(|&(u, v)| u < v),
        base_root,
    );
    let mut depth = vec![0; n];
    build_depth(&neighbors, &mut depth, base_root, n);
    let dist = |u: usize, v: usize| depth[u] + depth[v] - 2 * depth[lca.get(u, v) as usize];

    let mut size = vec![0; n];
    let mut parent_centroid = vec![UNSET; n];
    centroid::init_size(&neighbors, &mut size, base_root, n);
    centroid::build_centroid_tree(
        &neighbors,
        &mut size,
        &mut vec![false; n],
        &mut parent_centroid,
        base_root,
    );

    let mut is_white = vec![false; n];
    let mut white_dist_freq = vec![BTreeMap::<u32, u32>::new(); n];

    for _ in 0..input.value() {
        match input.token() {
            "1" => {
                let u0 = input.value::<usize>() - 1;
                is_white[u0] ^= true;
                let mut u = u0;
                if is_white[u0] {
                    loop {
                        white_dist_freq[u]
                            .entry(dist(u0, u))
                            .and_modify(|x| *x += 1)
                            .or_insert(1);
                        u = parent_centroid[u] as usize;
                        if u == UNSET as usize {
                            break;
                        }
                    }
                } else {
                    loop {
                        let d = dist(u0, u);
                        let freq = white_dist_freq[u].get_mut(&d).unwrap();
                        *freq -= 1;
                        if *freq == 0 {
                            white_dist_freq[u].remove(&d);
                        }
                        u = parent_centroid[u] as usize;
                        if u == UNSET as usize {
                            break;
                        }
                    }
                }
            }
            "2" => {
                let u0 = input.value::<usize>() - 1;
                let mut u = u0;
                let mut ans = INF;
                loop {
                    if let Some((&d, _)) = white_dist_freq[u].iter().next() {
                        ans = ans.min(d + dist(u0, u));
                    }
                    u = parent_centroid[u] as usize;
                    if u == UNSET as usize {
                        break;
                    }
                }

                if ans == INF {
                    writeln!(output, "-1").unwrap();
                } else {
                    writeln!(output, "{}", ans).unwrap();
                }
            }
            _ => panic!(),
        }
    }
}
