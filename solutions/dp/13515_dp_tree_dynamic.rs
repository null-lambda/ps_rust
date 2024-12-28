use std::{io::Write, ops::Range};

use fenwick_tree::{FenwickTree, Group};

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    pub struct InputAtOnce {
        _buf: &'static str,
        iter: std::str::SplitAsciiWhitespace<'static>,
    }

    impl InputAtOnce {
        pub fn token(&mut self) -> &'static str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    extern "C" {
        fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64)
            -> *mut u8;
        fn fstat(fd: i32, stat: *mut usize) -> i32;
    }

    pub fn stdin() -> InputAtOnce {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let _buf = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        let _buf =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(_buf, stat[6])) };
        let iter = _buf.split_ascii_whitespace();
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::with_capacity(1 << 16, stdout)
    }

    pub struct IntScanner {
        buf: &'static [u8],
    }

    impl IntScanner {
        fn skip(&mut self) {
            loop {
                match self.buf {
                    &[..=b' ', ..] => self.buf = &self.buf[1..],
                    _ => break,
                }
            }
        }

        fn u32_noskip(&mut self) -> u32 {
            let mut acc = 0;
            loop {
                match self.buf {
                    &[b'0'..=b'9', ..] => acc = acc * 10 + (self.buf[0] - b'0') as u32,
                    _ => break,
                }
                self.buf = &self.buf[1..];
            }
            acc
        }

        pub fn u32(&mut self) -> u32 {
            self.skip();
            self.u32_noskip()
        }

        pub fn i32(&mut self) -> i32 {
            self.skip();
            match self.buf {
                &[b'-', ..] => {
                    self.buf = &self.buf[1..];
                    -(self.u32_noskip() as i32)
                }
                _ => self.u32_noskip() as i32,
            }
        }
    }

    pub fn stdin_int() -> IntScanner {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let buf = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        let buf =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(buf, stat[6])) };
        IntScanner {
            buf: buf.as_bytes(),
        }
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
        pub chain_bottom: Vec<u32>,
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
            let mut chain_bottom = vec![UNSET; n];
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
                    chain_bottom[u] = if h == UNSET {
                        u as u32
                    } else {
                        chain_bottom[h as usize]
                    };

                    assert!(u != p as usize);
                    u = p as usize;
                }
            }
            topological_order.push(root as u32);
            assert!(topological_order.len() == n, "Invalid tree structure");

            let h = heavy_child[root];
            chain_bottom[root] = if h == UNSET {
                root as u32
            } else {
                chain_bottom[h as usize]
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
                chain_bottom,
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

pub mod fenwick_tree {
    pub trait Group {
        type X: Clone;
        fn id(&self) -> Self::X;
        fn add_assign(&self, lhs: &mut Self::X, rhs: Self::X);
        fn sub_assign(&self, lhs: &mut Self::X, rhs: Self::X);
    }

    #[derive(Clone)]
    pub struct FenwickTree<G: Group> {
        n: usize,
        group: G,
        sum: Vec<G::X>,
    }

    impl<G: Group> FenwickTree<G> {
        pub fn new(n: usize, group: G) -> Self {
            let n = n.next_power_of_two(); // Required for binary search
            let sum = (0..n).map(|_| group.id()).collect();
            Self { n, group, sum }
        }

        pub fn from_iter(iter: impl IntoIterator<Item = G::X>, group: G) -> Self {
            let mut sum: Vec<_> = iter.into_iter().collect();
            let n = sum.len();

            let n = n.next_power_of_two(); // Required for binary search
            sum.resize_with(n, || group.id());

            for i in 1..n {
                let prev = sum[i - 1].clone();
                group.add_assign(&mut sum[i], prev);
            }
            for i in (1..n).rev() {
                let j = i & (i + 1);
                if j >= 1 {
                    let prev = sum[j - 1].clone();
                    group.sub_assign(&mut sum[i], prev);
                }
            }

            Self { n, group, sum }
        }

        pub fn add(&mut self, mut idx: usize, value: G::X) {
            debug_assert!(idx < self.n);
            while idx < self.n {
                self.group.add_assign(&mut self.sum[idx], value.clone());
                idx |= idx + 1;
            }
        }

        // Exclusive prefix sum (0..idx)
        pub fn sum_prefix(&self, idx: usize) -> G::X {
            debug_assert!(idx <= self.n);
            let mut res = self.group.id();
            let mut r = idx;
            while r > 0 {
                self.group.add_assign(&mut res, self.sum[r - 1].clone());
                r &= r - 1;
            }
            res
        }

        pub fn sum_range(&self, range: std::ops::Range<usize>) -> G::X {
            debug_assert!(range.start <= range.end && range.end <= self.n);
            let mut res = self.sum_prefix(range.end);
            self.group
                .sub_assign(&mut res, self.sum_prefix(range.start));
            res
        }

        pub fn get(&self, idx: usize) -> G::X {
            self.sum_range(idx..idx + 1)
        }

        // find the first i, such that equiv pred(sum_range(0..=i)) == false
        pub fn partition_point_prefix(&self, mut pred: impl FnMut(&G::X) -> bool) -> usize {
            let p1_log2 = usize::BITS - self.n.leading_zeros();
            let mut idx = 0;
            let mut sum = self.group.id();
            for i in (0..p1_log2).rev() {
                let idx_next = idx | (1 << i);
                if idx_next > self.n {
                    continue;
                }
                let mut sum_next = sum.clone();
                self.group
                    .add_assign(&mut sum_next, self.sum[idx_next - 1].clone());
                if pred(&sum_next) {
                    sum = sum_next;
                    idx = idx_next;
                }
            }
            idx
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Additive;

impl Group for Additive {
    type X = i32;

    fn id(&self) -> Self::X {
        0
    }

    fn add_assign(&self, lhs: &mut Self::X, rhs: Self::X) {
        *lhs += rhs;
    }

    fn sub_assign(&self, lhs: &mut Self::X, rhs: Self::X) {
        *lhs -= rhs;
    }
}

// A compressed chain, as an action acting on a chain top's parent node.
#[derive(Clone, Debug)]
struct LightEdgeData {
    prefix_count: i32,
}

impl LightEdgeData {
    fn id() -> Self {
        Self { prefix_count: 0 }
    }

    // Invertible rake operation
    fn combine(&mut self, other: &Self, inv: bool) {
        if !inv {
            self.prefix_count += other.prefix_count;
        } else {
            self.prefix_count -= other.prefix_count;
        }
    }

    fn lifted(count_heavy: &[FenwickTree<Additive>], color: usize, chain: Range<usize>) -> Self {
        let Range { start, end } = chain;

        let dual = 1 - color;
        let base = count_heavy[dual].sum_prefix(start);
        let prefix = count_heavy[dual]
            .partition_point_prefix(|&sum| sum <= base)
            .min(end)
            .max(start);
        let prefix_count = count_heavy[color].sum_range(start..prefix);

        Self { prefix_count }
    }

    fn apply(
        &self,
        count_in_chain: &FenwickTree<Additive>,
        count_heavy: &mut FenwickTree<Additive>,
        u: usize,
    ) {
        let old = count_heavy.get(u);
        let new = if count_in_chain.get(u) != 0 {
            1 + self.prefix_count
        } else {
            0
        };
        count_heavy.add(u, new - old);
    }
}

fn main() {
    let mut input = fast_io::stdin_int();
    let mut output = fast_io::stdout();

    let n = input.u32() as usize;
    let edges = (0..n - 1).map(|_| (input.u32() - 1, input.u32() - 1));
    let root = 0;
    let hld = hld::HLD::from_edges(n, edges, root, false);
    let sid = |u: usize| hld.segmented_idx[u] as usize;

    let mut count_in_chain = vec![
        FenwickTree::from_iter((0..n).map(|_| 1), Additive),
        FenwickTree::new(n, Additive),
    ]; // Track connectivity of a path
    let mut count_heavy = vec![FenwickTree::new(n, Additive); 2];
    let mut action_upward = vec![vec![LightEdgeData::id(); n]; 2];
    let mut action_agg = vec![vec![LightEdgeData::id(); n]; 2];
    for &u in &hld.topological_order[..n - 1] {
        if u != hld.chain_top[u as usize] {
            continue;
        }
        let p = hld.parent[u as usize];
        action_upward[0][u as usize] = {
            LightEdgeData {
                prefix_count: hld.size[u as usize] as i32,
            }
        };
        action_agg[0][p as usize].combine(&action_upward[0][u as usize], false);
    }
    for u in 0..n {
        action_agg[0][u].apply(&count_in_chain[0], &mut count_heavy[0], sid(u));
    }

    for _ in 0..input.u32() {
        match input.u32() {
            1 => {
                let mut u = input.u32() as usize - 1;

                // Update vertex data
                for color in 0..2 {
                    let old = count_in_chain[color].get(sid(u));
                    let new = 1 - old;
                    count_in_chain[color].add(sid(u), new - old);
                    action_agg[color][u].apply(
                        &count_in_chain[color],
                        &mut count_heavy[color],
                        sid(u),
                    );
                }

                // Ascend to the root, and update light edge data
                loop {
                    let top = hld.chain_top[u] as usize;
                    if top == root {
                        break;
                    }
                    let p = hld.parent[top] as usize;
                    let bottom = hld.chain_bottom[u] as usize;

                    for color in 0..2 {
                        action_agg[color][p].combine(&action_upward[color][top], true);
                        action_upward[color][top] =
                            LightEdgeData::lifted(&count_heavy, color, sid(top)..sid(bottom) + 1);
                        action_agg[color][p].combine(&action_upward[color][top], false);
                        action_agg[color][p].apply(
                            &count_in_chain[color],
                            &mut count_heavy[color],
                            sid(p),
                        );
                    }
                    u = p;
                }
            }
            2 => {
                let mut u = input.u32() as usize - 1;
                let color = (count_in_chain[0].sum_range(sid(u)..sid(u) + 1) == 0) as usize;
                let dual = 1 - color;

                // Ascend through the connected connected component of u, to find a node in the topmost chain
                loop {
                    let top = hld.chain_top[u] as usize;
                    if top == root {
                        break;
                    }
                    let p = hld.parent[top] as usize;

                    let prefix_len = sid(u) - sid(top) + 2;
                    let prefix_count = count_in_chain[color].sum_range(sid(top)..sid(u) + 1)
                        + count_in_chain[color].sum_range(sid(p)..sid(p) + 1);
                    if prefix_len != prefix_count as usize {
                        break;
                    }
                    u = hld.parent[top] as usize;
                }

                let su = sid(u);
                let s_top = sid(hld.chain_top[u] as usize);
                let s_bottom = sid(hld.chain_bottom[u] as usize);

                debug_assert!(count_in_chain[color].sum_range(su..su + 1) != 0);
                debug_assert!(count_in_chain[dual].sum_range(su..su + 1) == 0);

                let base = count_in_chain[dual].sum_prefix(su);
                let left = if base == 0 {
                    0
                } else {
                    (count_in_chain[dual].partition_point_prefix(|&sum| sum < base) + 1)
                        .max(s_top)
                        .min(su)
                };

                let base = count_in_chain[dual].sum_prefix(su + 1);
                let right = count_in_chain[dual]
                    .partition_point_prefix(|&sum| sum <= base)
                    .max(su + 1)
                    .min(s_bottom + 1);

                let ans = count_heavy[color].sum_range(left..right);
                writeln!(output, "{}", ans).unwrap();
            }
            _ => panic!(),
        }
    }
}
