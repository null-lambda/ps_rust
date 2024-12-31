use std::io::Write;

use segtree::{Monoid, SegTree};

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

pub mod segtree {
    use std::{iter, ops::Range};

    pub trait Monoid {
        type X;
        fn id(&self) -> Self::X;
        fn combine(&self, a: &Self::X, b: &Self::X) -> Self::X;
    }

    #[derive(Debug, Clone)]
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
            I::IntoIter: ExactSizeIterator,
        {
            let iter = iter.into_iter();
            let n = iter.len();
            let mut sum: Vec<_> = (0..n)
                .map(|_| monoid.id())
                .chain(iter)
                .chain(iter::repeat_with(|| monoid.id()))
                .take(2 * n)
                .collect();
            for i in (1..n).rev() {
                sum[i] = monoid.combine(&sum[i << 1], &sum[i << 1 | 1]);
            }
            Self { n, sum, monoid }
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

        pub fn sum_range(&self, range: Range<usize>) -> M::X {
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

        pub fn sum_all(&self) -> &M::X {
            debug_assert!(self.n.is_power_of_two());
            &self.sum[1]
        }
    }
}

#[derive(Clone, Debug)]
struct Unary(u8);

impl Unary {
    fn new(f: impl Fn(bool) -> bool) -> Self {
        Self(f(false) as u8 | (f(true) as u8) << 1)
    }

    fn apply(&self, x: bool) -> bool {
        (self.0 >> x as u32) & 1 != 0
    }

    fn as_constant(&self) -> Option<bool> {
        let f = [self.apply(false), self.apply(true)];
        (f[0] == f[1]).then(|| f[0])
    }
}

#[derive(Clone, Debug)]
struct AutZ2;

impl Monoid for AutZ2 {
    type X = Unary;

    fn id(&self) -> Unary {
        Unary::new(|x| x)
    }

    fn combine(&self, a: &Unary, b: &Unary) -> Unary {
        Unary::new(|x| a.apply(b.apply(x)))
    }
}

#[derive(Clone, Debug)]
struct CommBinary(u8);

impl CommBinary {
    fn new(f: impl Fn(bool, bool) -> bool) -> Self {
        let f = [
            [f(false, false), f(false, true)],
            [f(true, false), f(true, true)],
        ];
        debug_assert!(f[1][0] == f[0][1]);
        Self(f[0][0] as u8 | (f[0][1] as u8) << 1 | (f[1][1] as u8) << 2)
    }

    fn from_byte(b: u8) -> Self {
        match b {
            b'0' => Self::new(|_, _| false),
            b'1' => Self::new(|_, _| true),
            b'^' => Self::new(|x, y| x != y),
            b'&' => Self::new(|x, y| x && y),
            b'|' => Self::new(|x, y| x || y),
            _ => panic!(),
        }
    }

    fn apply(&self, x: bool, y: bool) -> bool {
        (self.0 >> (x as u32 + y as u32)) & 1 != 0
    }

    fn partial_apply(&self, x: bool) -> Unary {
        Unary::new(|y| self.apply(x, y))
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let s = input.token();

    const UNSET: u32 = u32::MAX;
    let dummy = CommBinary::from_byte(b'0');
    let mut symbol = vec![];
    let mut parent = vec![];
    let mut ancestors = vec![];
    let mut pos_to_node = vec![UNSET; s.len()];
    for (i, b) in s.bytes().enumerate() {
        match b {
            b'(' => {
                parent.push(ancestors.last().copied().unwrap_or(UNSET));
                ancestors.push(symbol.len() as u32);
                symbol.push(dummy.clone());
            }
            b')' => {
                ancestors.pop();
            }
            b'0' | b'1' => {
                pos_to_node[i] = symbol.len() as u32;
                symbol.push(CommBinary::from_byte(b));
                parent.push(ancestors.last().copied().unwrap_or(UNSET));
            }
            b'&' | b'|' | b'^' => {
                let p = *ancestors.last().unwrap();
                pos_to_node[i] = p;
                symbol[p as usize] = CommBinary::from_byte(b);
            }
            _ => panic!(),
        }
    }

    let n = symbol.len();
    assert!(symbol.len() == n);

    let root = 0;
    let edges = (1..n).map(|i| (parent[i], i as u32));
    let hld = hld::HLD::from_edges(n, edges, root, false);
    let sid = |u: usize| hld.segmented_idx[u as usize] as usize;
    let chain = |u: usize| {
        let top = hld.chain_top[u as usize];
        let bottom = hld.chain_bottom[u as usize];
        sid(top as usize)..sid(bottom as usize) + 1
    };
    let idx_in_chain = |u: usize| sid(u) - sid(hld.chain_top[u as usize] as usize);

    let mut dp_chain = vec![SegTree::with_size(0, AutZ2); n];
    let mut light_child = vec![UNSET; n];

    for &u in &hld.topological_order {
        let top = hld.chain_top[u as usize];
        if u == top {
            dp_chain[top as usize] =
                SegTree::with_size(chain(top as usize).len().next_power_of_two(), AutZ2);
        }
    }
    for &u in &hld.topological_order {
        let p = hld.parent[u as usize];
        let top = hld.chain_top[u as usize];

        let c = light_child[u as usize];
        let f = if c != UNSET {
            dp_chain[c as usize].sum_all().as_constant().unwrap()
        } else {
            false
        };
        dp_chain[top as usize].modify(idx_in_chain(u as usize), |x| {
            *x = symbol[u as usize].partial_apply(f)
        });

        if p != UNSET && u == top {
            light_child[p as usize] = u;
        }
    }

    let ans = dp_chain[root].sum_all().as_constant().unwrap();
    write!(output, "{}", ans as u8).unwrap();

    for _ in 0..input.value() {
        let i = input.value::<usize>() - 1;
        let b = input.token().as_bytes()[0];

        let mut u = pos_to_node[i] as usize;
        symbol[u] = CommBinary::from_byte(b);
        let c = light_child[u] as usize;
        let f = if c != UNSET as usize {
            dp_chain[c].sum_all().as_constant().unwrap()
        } else {
            false
        };
        let top = hld.chain_top[u] as usize;
        dp_chain[top].modify(idx_in_chain(u), |x| *x = symbol[u].partial_apply(f));

        loop {
            let top = hld.chain_top[u] as usize;
            if top == root {
                break;
            }
            let p = hld.parent[top] as usize;
            let f = dp_chain[top].sum_all().as_constant().unwrap();
            let p_top = hld.chain_top[p] as usize;
            dp_chain[p_top].modify(idx_in_chain(p), |x| *x = symbol[p].partial_apply(f));

            u = p;
        }
        let ans = dp_chain[root].sum_all().as_constant().unwrap();
        write!(output, "{}", ans as u8).unwrap();
    }

    writeln!(output).unwrap();
}
