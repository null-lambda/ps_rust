use std::{io::Write, mem::MaybeUninit};

use segtree::{Monoid, SegTree};

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
                for mut u in topological_order.into_iter().rev() {
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
                for mut u in topological_order.into_iter().rev() {
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

pub mod segtree {
    use std::{mem::MaybeUninit, ops::Range};

    pub trait Monoid {
        type X;
        fn id(&self) -> Self::X;
        fn combine(&self, a: &Self::X, b: &Self::X) -> Self::X;
    }

    #[derive(Debug)]
    pub struct SegTree<'a, M>
    where
        M: Monoid,
    {
        n: usize,
        sum: &'a mut [M::X],
        monoid: M,
    }

    impl<'a, M: Monoid> SegTree<'a, M> {
        pub fn new_in(
            n: usize,
            monoid: M,
            buffer: &'a mut [MaybeUninit<M::X>],
        ) -> Option<(Self, &'a mut [MaybeUninit<M::X>])> {
            let n = n.next_power_of_two();
            let n_sum = 2 * n;
            if buffer.len() < n_sum {
                return None;
            }
            let (sum, rest) = buffer.split_at_mut(n_sum);
            for i in 0..n_sum {
                sum[i].write(monoid.id());
            }
            let sum = unsafe { std::mem::transmute::<&mut [MaybeUninit<M::X>], &mut [M::X]>(sum) };
            Some((Self { n, sum, monoid }, rest))
        }

        pub fn modify(&mut self, mut idx: usize, f: impl FnOnce(&mut M::X)) {
            debug_assert!(idx < self.n);
            idx += self.n;
            f(&mut self.sum[idx]);
            while idx > 1 {
                idx >>= 1;
                self.sum[idx] = self
                    .monoid
                    .combine(&self.sum[idx << 1], &self.sum[idx << 1 | 1]);
            }
        }

        pub fn get(&self, idx: usize) -> &M::X {
            &self.sum[idx + self.n]
        }

        pub fn query_range(&self, range: Range<usize>) -> M::X {
            let Range { mut start, mut end } = range;
            debug_assert!(start < self.n && end <= self.n);
            start += self.n;
            end += self.n;
            let (mut result_left, mut result_right) = (self.monoid.id(), self.monoid.id());
            while start < end {
                if start & 1 != 0 {
                    result_left = self.monoid.combine(&result_left, &self.sum[start]);
                }
                if end & 1 != 0 {
                    result_right = self.monoid.combine(&self.sum[end - 1], &result_right);
                }
                start = (start + 1) >> 1;
                end >>= 1;
            }

            self.monoid.combine(&result_left, &result_right)
        }

        pub fn query_all(&self) -> &M::X {
            // Warning: works only if n is power of two (otherwise we have a forest, not tree)
            &self.sum[1]
        }
    }
}

const INF: i32 = 1_000_000;

#[derive(Clone, Copy)]
struct ChainAgg([[i32; 2]; 2]);

impl ChainAgg {
    const fn singleton(color: u8) -> Self {
        Self(match color {
            1 => [[0, INF], [INF, INF]],
            2 => [[INF, INF], [INF, 0]],
            3 => [[0, INF], [INF, 0]],
            _ => panic!(),
        })
    }

    fn min_cost(&self) -> i32 {
        *self.0.iter().flatten().min().unwrap()
    }
}

struct ChainOp;

impl Monoid for ChainOp {
    type X = ChainAgg;

    fn id(&self) -> Self::X {
        ChainAgg::singleton(3)
    }

    fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X {
        ChainAgg(std::array::from_fn(|s| {
            std::array::from_fn(|e| {
                INF.min(lhs.0[s][0] + rhs.0[0][e])
                    .min(lhs.0[s][0] + rhs.0[1][e] + 1)
                    .min(lhs.0[s][1] + rhs.0[0][e] + 1)
                    .min(lhs.0[s][1] + rhs.0[1][e])
            })
        }))
    }
}

#[derive(Clone, Copy)]
struct LightTreeAgg([i32; 2]);

impl LightTreeAgg {
    fn zero() -> Self {
        Self([0; 2])
    }

    fn pull_from(&mut self, light: &ChainAgg, inv: bool) {
        let collapsed: [_; 2] = std::array::from_fn(|i| light.0[i][0].min(light.0[i][1]));
        let delta = [
            collapsed[0].min(collapsed[1] + 1),
            collapsed[1].min(collapsed[0] + 1),
        ];
        if !inv {
            self.0 = [self.0[0] + delta[0], self.0[1] + delta[1]];
        } else {
            self.0 = [self.0[0] - delta[0], self.0[1] - delta[1]];
        }
    }

    fn push_up(&self, heavy: &mut ChainAgg) {
        if heavy.0[0][0] != INF {
            heavy.0[0][0] = self.0[0];
        }
        if heavy.0[1][1] != INF {
            heavy.0[1][1] = self.0[1];
        }
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
    let idx_in_chain = |u: usize| sid(u) - sid(hld.chain_top[u] as usize);
    let chain_len = |u: usize| {
        let top = hld.chain_top[u] as usize;
        let bottom = hld.chain_bottom[u] as usize;
        sid(bottom) - sid(top) + 1
    };

    let mut buffer = vec![MaybeUninit::uninit(); 4 * n];
    let mut view = buffer.as_mut_slice();
    let mut dp_heavy: Vec<_> = (0..n).map(|_| MaybeUninit::uninit()).collect();
    for u in 0..n {
        if u != hld.chain_top[u] as usize {
            continue;
        }
        let (tree, view_rest) = SegTree::new_in(chain_len(u), ChainOp, view).unwrap();
        view = view_rest;
        dp_heavy[u] = MaybeUninit::new(tree);
    }

    let mut dp_light = vec![LightTreeAgg::zero(); n];

    for _ in 0..input.u32() {
        let mut color = Some(input.u32() as u8);
        let mut u = input.u32() as usize - 1;

        let update_heavy = |heavy: &mut ChainAgg, light: LightTreeAgg, color: &mut Option<u8>| {
            if let Some(color) = color.take() {
                *heavy = ChainAgg::singleton(color)
            }
            light.push_up(heavy);
        };

        loop {
            let top = hld.chain_top[u] as usize;
            if top == root {
                break;
            }
            let p = hld.parent[top] as usize;

            let chain = unsafe { dp_heavy[top].assume_init_mut() };
            dp_light[p].pull_from(chain.query_all(), true);
            chain.modify(idx_in_chain(u), |h| {
                update_heavy(h, dp_light[u], &mut color)
            });
            dp_light[p].pull_from(chain.query_all(), false);

            u = p;
        }
        let top = root;
        let chain = unsafe { dp_heavy[top].assume_init_mut() };
        chain.modify(idx_in_chain(u), |h| {
            update_heavy(h, dp_light[u], &mut color)
        });
        let ans = chain.query_all().min_cost();
        writeln!(output, "{}", ans).unwrap();
    }
}
