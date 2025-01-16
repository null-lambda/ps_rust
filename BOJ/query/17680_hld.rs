use std::cmp::Ordering;
use std::io::Write;
use std::{collections::HashMap, hash::Hash};

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

    const UNSET: u32 = u32::MAX;

    #[derive(Debug)]
    pub struct HLD {
        pub size: Vec<u32>,
        pub parent: Vec<u32>,
        pub heavy_child: Vec<u32>,
        pub chain_top: Vec<u32>,
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

                    assert!(u != p as usize);
                    u = p as usize;
                }
            }
            topological_order.push(root as u32);
            assert!(topological_order.len() == n, "Invalid tree structure");

            let mut parent = xor_neighbors;
            parent[root] = UNSET;

            // Downward propagation
            let mut chain_top = vec![root as u32; n];
            let mut segmented_idx = vec![UNSET; n];
            if !use_dfs_ordering {
                // A rearranged topological index continuous in a chain, for path queries
                let mut timer = 0;
                for mut u in topological_order.iter().rev().copied() {
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
                for mut u in topological_order.iter().rev().copied() {
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
                segmented_idx,
                topological_order,
            }
        }

        pub fn for_each_path<F>(&self, mut u: usize, mut v: usize, mut visitor: F)
        where
            F: FnMut(usize, usize, bool),
        {
            debug_assert!(u < self.len() && v < self.len());

            while self.chain_top[u] != self.chain_top[v] {
                if self.segmented_idx[self.chain_top[u] as usize]
                    < self.segmented_idx[self.chain_top[v] as usize]
                {
                    std::mem::swap(&mut u, &mut v);
                }
                visitor(self.chain_top[u] as usize, u, false);
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.segmented_idx[u] > self.segmented_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            visitor(u, v, true);
        }

        pub fn for_each_path_splitted<F>(&self, mut u: usize, mut v: usize, mut visit: F)
        where
            F: FnMut(usize, usize, bool, bool),
        {
            debug_assert!(u < self.len() && v < self.len());
            if self.segmented_idx[u] > self.segmented_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            while self.chain_top[u] != self.chain_top[v] {
                if self.segmented_idx[self.chain_top[u] as usize]
                    > self.segmented_idx[self.chain_top[v] as usize]
                {
                    visit(self.chain_top[u] as usize, u, true, false);
                    u = self.parent[self.chain_top[u] as usize] as usize;
                } else {
                    visit(self.chain_top[v] as usize, v, false, false);
                    v = self.parent[self.chain_top[v] as usize] as usize;
                }
            }
            if self.segmented_idx[u] > self.segmented_idx[v] {
                visit(v, u, true, true);
            } else {
                visit(u, v, false, true);
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

pub mod segtree_wide {
    // Cache-friendly segment tree, based on a B-ary tree.
    // https://en.algorithmica.org/hpc/data-structures/segment-trees/#wide-segment-trees

    // const CACHE_LINE_SIZE: usize = 64;

    // const fn adaptive_block_size<T>() -> usize {
    //     assert!(
    //         std::mem::size_of::<T>() > 0,
    //         "Zero-sized types are not supported"
    //     );
    //     let mut res = CACHE_LINE_SIZE / std::mem::size_of::<T>();
    //     if res < 2 {
    //         res = 2;
    //     }
    //     res
    // }

    use std::iter;

    const fn height<const B: usize>(mut node: usize) -> u32 {
        debug_assert!(node > 0);
        let mut res = 1;
        while node > B {
            res += 1;
            node = node.div_ceil(B);
        }
        res
    }

    // yields (h, offset)
    fn offsets<const B: usize>(size: usize) -> impl Iterator<Item = usize> {
        let mut offset = 0;
        let mut n = size;
        iter::once(0).chain((1..).map(move |_| {
            n = n.div_ceil(B);
            offset += n * B;
            offset
        }))
    }

    fn offset<const B: usize>(size: usize, h: u32) -> usize {
        offsets::<B>(size).nth(h as usize).unwrap()
    }

    fn log<const B: usize>() -> u32 {
        usize::BITS - B.leading_zeros() - 1
    }

    fn round<const B: usize>(x: usize) -> usize {
        x & !(B - 1)
    }

    const fn compute_mask<const B: usize>() -> [[X; B]; B] {
        let mut res = [[0; B]; B];
        let mut i = 0;
        while i < B {
            let mut j = 0;
            while j < B {
                res[i][j] = if i < j { !0 } else { 0 };
                j += 1;
            }
            i += 1;
        }
        res
    }

    type X = i32;

    #[derive(Debug, Clone)]
    pub struct SegTree<const B: usize> {
        n: usize,
        sum: Vec<X>,
        mask: [[X; B]; B],
        offsets: Vec<usize>,
    }

    impl<const B: usize> SegTree<B> {
        pub fn with_size(n: usize) -> Self {
            assert!(B >= 2 && B.is_power_of_two());
            let max_height = height::<B>(n);
            Self {
                n,
                sum: vec![0; offset::<B>(n, max_height)],
                mask: compute_mask::<B>(),
                offsets: offsets::<B>(n).take(max_height as usize).collect(),
            }
        }

        #[target_feature(enable = "avx2")] // Required. __mm256 has significant performance benefits over __m128.
        unsafe fn add_avx2(&mut self, mut idx: usize, value: X) {
            debug_assert!(idx < self.n);
            for (_, offset) in self.offsets.iter().enumerate() {
                let block = &mut self.sum[offset + round::<B>(idx)..];
                for (b, m) in block.iter_mut().zip(&self.mask[idx % B]) {
                    *b += value & m;
                }
                idx >>= log::<B>();
            }
        }

        pub fn add(&mut self, idx: usize, value: X) {
            unsafe {
                self.add_avx2(idx, value);
            }
        }

        pub fn sum_prefix(&mut self, idx: usize) -> X {
            debug_assert!(idx <= self.n);
            let mut res = 0;
            for (h, offset) in self.offsets.iter().enumerate() {
                res += self.sum[offset + (idx >> h as u32 * log::<B>())];
            }
            res
        }

        pub fn sum_range(&mut self, range: std::ops::Range<usize>) -> X {
            debug_assert!(range.start <= range.end && range.end <= self.n);
            let r = self.sum_prefix(range.end);
            let l = self.sum_prefix(range.start);
            r - l
        }
    }
}

fn compress_coord<T: Ord + Clone + Hash>(
    xs: impl IntoIterator<Item = T>,
) -> (Vec<T>, HashMap<T, u32>) {
    let mut x_map: Vec<T> = xs.into_iter().collect();
    x_map.sort_unstable();
    x_map.dedup();

    let x_map_inv = x_map
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, x)| (x, i as u32))
        .collect();

    (x_map, x_map_inv)
}

#[derive(Default, Clone, Debug)]
struct ChainAgg {
    prefix_stack: Vec<(u32, u32)>,
}

impl ChainAgg {
    fn push_front(&mut self, color: u32, len: u32) {
        self.prefix_stack.push((color, len));
    }

    fn overwrite_prefix(
        &mut self,
        color: u32,
        end_inclusive: u32,
        mut yield_erased: impl FnMut(u32, u32),
    ) {
        while let Some((color_old, len_old)) = self.prefix_stack.last_mut() {
            match end_inclusive.cmp(len_old) {
                Ordering::Less => {
                    yield_erased(*color_old, end_inclusive);
                    self.prefix_stack.push((color, end_inclusive));
                    return;
                }
                Ordering::Equal => {
                    yield_erased(*color_old, *len_old);
                    *color_old = color;
                    return;
                }
                Ordering::Greater => {
                    yield_erased(*color_old, *len_old);
                    self.prefix_stack.pop();
                }
            }
        }
        self.prefix_stack.push((color, end_inclusive));
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let weights: Vec<u32> = (0..n).map(|_| input.value()).collect();
    let edges: Vec<(u32, u32)> = (0..n - 1)
        .map(|_| (input.value::<u32>() - 1, input.value::<u32>() - 1))
        .collect();

    let root = 0;
    let hld = hld::HLD::from_edges(n, edges.iter().copied(), root, false);
    let sid = |u: usize| hld.segmented_idx[u] as usize;

    let mut chain_agg = vec![ChainAgg::default(); n];
    for &u in &hld.topological_order {
        let top = hld.chain_top[u as usize] as usize;
        chain_agg[top].push_front(weights[u as usize], (sid(u as usize) - sid(top)) as u32);
    }

    for (a, b) in edges {
        let mut seq = vec![];
        let c_new = weights[b as usize];
        let mut group = vec![];
        hld.for_each_path(a as usize, root as usize, |top, end, _| {
            chain_agg[top as usize].overwrite_prefix(
                c_new,
                sid(end) as u32 - sid(top) as u32,
                |c, len| group.push((c, len)),
            );

            if !group.is_empty() {
                for i in (1..group.len()).rev() {
                    group[i].1 -= group[i - 1].1;
                }
                group[0].1 += 1;

                seq.extend(group.drain(..).rev());
            } else {
            }
        });

        let (_, x_inv) = compress_coord(seq.iter().map(|&(c, _)| c));
        let x_bound = x_inv.len();

        let mut counter = segtree_wide::SegTree::<16>::with_size(x_bound);
        let mut inversion_count = 0u64;
        for (c, l) in seq {
            inversion_count += counter.sum_prefix(x_inv[&c] as usize) as u64 * l as u64;
            counter.add(x_inv[&c] as usize, l as i32);
        }
        writeln!(output, "{}", inversion_count).unwrap();
    }
}
