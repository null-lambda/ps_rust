use std::io::Write;

use segtree_lazy::MonoidAction;

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
        pub fn u32(&mut self) -> u32 {
            loop {
                match self.buf {
                    &[] => panic!(),
                    &[b'0'..=b'9', ..] => break,
                    _ => self.buf = &self.buf[1..],
                }
            }

            let mut acc = 0;
            loop {
                match self.buf {
                    &[] => panic!(),
                    &[b'0'..=b'9', ..] => acc = acc * 10 + (self.buf[0] - b'0') as u32,
                    _ => break,
                }
                self.buf = &self.buf[1..];
            }
            acc
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
    const UNSET: u32 = u32::MAX;

    // Heavy-Light Decomposition
    #[derive(Debug)]
    pub struct HLD {
        pub size: Vec<u32>,
        pub depth: Vec<u32>,
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
            let mut topological_order = vec![];
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

                    u = p as usize;
                }
            }
            topological_order.push(root as u32);
            topological_order.reverse();
            assert!(topological_order.len() == n, "Invalid tree structure");

            let mut parent = xor_neighbors;
            parent[root] = UNSET;

            // Downward propagation
            let mut depth = vec![0; n];
            let mut chain_top = vec![root as u32; n];
            for &u in &topological_order[1..] {
                let p = parent[u as usize];
                depth[u as usize] = depth[p as usize] + 1;
            }

            let mut segmented_idx = vec![UNSET; n];
            if !use_dfs_ordering {
                // A rearranged topological index continuous in a chain, for path queries
                let mut timer = 0;
                for u in &topological_order {
                    let mut u = *u;
                    if segmented_idx[u as usize] != UNSET {
                        continue;
                    }

                    chain_top[u as usize] = u;
                    loop {
                        segmented_idx[u as usize] = timer;
                        timer += 1;
                        let h = heavy_child[u as usize];
                        if h == UNSET {
                            break;
                        }
                        chain_top[h as usize] = chain_top[u as usize];
                        u = h;
                    }
                }
            } else {
                // DFS ordering for path & subtree queries
                let mut offset = vec![0; n];
                for u in &topological_order {
                    let mut u = *u;
                    if segmented_idx[u as usize] != UNSET {
                        continue;
                    }

                    let mut p = parent[u as usize];
                    let mut timer = 0;
                    if p != UNSET {
                        timer = offset[p as usize] + 1;
                        offset[p as usize] += size[u as usize] as u32;
                    }
                    segmented_idx[u as usize] = timer;
                    offset[u as usize] = timer;
                    chain_top[u as usize] = u;

                    timer += 1;

                    loop {
                        p = u;
                        u = heavy_child[u as usize];
                        if u == UNSET {
                            break;
                        }

                        chain_top[u as usize] = chain_top[p as usize];
                        offset[p as usize] += size[u as usize] as u32;
                        offset[u as usize] = timer;
                        segmented_idx[u as usize] = timer;
                        timer += 1;
                    }
                }
            }

            Self {
                size,
                depth,
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
                if self.depth[self.chain_top[u] as usize] < self.depth[self.chain_top[v] as usize] {
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
                if self.depth[self.chain_top[u] as usize] > self.depth[self.chain_top[v] as usize] {
                    visit(self.chain_top[u] as usize, u, true, false);
                    u = self.parent[self.chain_top[u] as usize] as usize;
                } else {
                    visit(self.chain_top[v] as usize, v, false, false);
                    v = self.parent[self.chain_top[v] as usize] as usize;
                }
            }
            if self.depth[u] > self.depth[v] {
                visit(v, u, true, true);
            } else {
                visit(u, v, false, true);
            }
        }

        pub fn lca(&self, mut u: usize, mut v: usize) -> usize {
            debug_assert!(u < self.len() && v < self.len());
            while self.chain_top[u] != self.chain_top[v] {
                if self.depth[self.chain_top[u] as usize] < self.depth[self.chain_top[v] as usize] {
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

pub mod segtree_lazy {
    use std::{iter, ops::Range};

    pub trait MonoidAction {
        type X;
        type F;
        fn id(&self) -> Self::X;
        fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X;
        fn id_action(&self) -> Self::F;
        fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F;
        fn apply_to_sum(&self, f: &Self::F, x_count: u32, x_sum: &Self::X) -> Self::X;
    }

    pub struct SegTree<M: MonoidAction> {
        n: usize,
        max_height: u32,
        pub sum: Vec<M::X>,
        pub lazy: Vec<M::F>,
        pub ma: M,
    }

    impl<M: MonoidAction> SegTree<M> {
        pub fn with_size(n: usize, ma: M) -> Self {
            Self {
                n,
                max_height: usize::BITS - n.leading_zeros(),
                sum: iter::repeat_with(|| ma.id()).take(2 * n).collect(),
                lazy: iter::repeat_with(|| ma.id_action()).take(n).collect(),
                ma,
            }
        }

        pub fn from_iter<I>(n: usize, iter: I, ma: M) -> Self
        where
            I: IntoIterator<Item = M::X>,
        {
            let mut sum: Vec<_> = (iter::repeat_with(|| ma.id()).take(n))
                .chain(
                    iter.into_iter()
                        .chain(iter::repeat_with(|| ma.id()))
                        .take(n),
                )
                .collect();
            for i in (1..n).rev() {
                sum[i] = ma.combine(&sum[i << 1], &sum[i << 1 | 1]);
            }
            Self {
                n,
                max_height: usize::BITS - n.leading_zeros(),
                sum,
                lazy: iter::repeat_with(|| ma.id_action()).take(n).collect(),
                ma,
            }
        }

        fn apply(&mut self, idx: usize, width: u32, value: &M::F) {
            self.sum[idx] = self.ma.apply_to_sum(&value, width, &self.sum[idx]);
            if idx < self.n {
                self.lazy[idx] = self.ma.combine_action(&value, &self.lazy[idx]);
            }
        }

        fn push_down(&mut self, width: u32, node: usize) {
            let value = unsafe { &*(&self.lazy[node] as *const _) };
            self.apply(node << 1, width, value);
            self.apply(node << 1 | 1, width, value);
            self.lazy[node] = self.ma.id_action();
        }

        fn push_range(&mut self, range: Range<usize>) {
            let Range { mut start, mut end } = range;
            start += self.n;
            end += self.n;

            let start_height = 1 + start.trailing_zeros();
            let end_height = 1 + end.trailing_zeros();
            for height in (start_height..=self.max_height).rev() {
                let width = 1 << height - 1;
                self.push_down(width, start >> height);
            }
            for height in (end_height..=self.max_height).rev().skip_while(|&height| {
                height >= start_height && end - 1 >> height == start >> height
            }) {
                let width = 1 << height - 1;
                self.push_down(width, end - 1 >> height);
            }
        }

        fn pull_up(&mut self, node: usize) {
            self.sum[node] = (self.ma).combine(&self.sum[node << 1], &self.sum[node << 1 | 1]);
        }

        pub fn apply_range(&mut self, range: Range<usize>, value: M::F) {
            let Range { mut start, mut end } = range;
            debug_assert!(start <= end && end <= self.n);
            if start == end {
                return;
            }

            self.push_range(range);
            start += self.n;
            end += self.n;
            let mut width: u32 = 1;
            let (mut pull_start, mut pull_end) = (false, false);
            while start < end {
                if pull_start {
                    self.pull_up(start - 1);
                }
                if pull_end {
                    self.pull_up(end);
                }
                if start & 1 != 0 {
                    self.apply(start, width, &value);
                    start += 1;
                    pull_start = true;
                }
                if end & 1 != 0 {
                    self.apply(end - 1, width, &value);
                    pull_end = true;
                }
                start >>= 1;
                end >>= 1;
                width <<= 1;
            }
            start -= 1;
            while end > 0 {
                if pull_start {
                    self.pull_up(start);
                }
                if pull_end && !(pull_start && start == end) {
                    self.pull_up(end);
                }
                start >>= 1;
                end >>= 1;
                width <<= 1;
            }
        }

        pub fn query_range(&mut self, range: Range<usize>) -> M::X {
            let Range { mut start, mut end } = range;

            self.push_range(range);
            start += self.n;
            end += self.n;
            let (mut result_left, mut result_right) = (self.ma.id(), self.ma.id());
            while start < end {
                if start & 1 != 0 {
                    result_left = self.ma.combine(&result_left, &self.sum[start]);
                }
                if end & 1 != 0 {
                    result_right = self.ma.combine(&self.sum[end - 1], &result_right);
                }
                start = (start + 1) >> 1;
                end >>= 1;
            }

            self.ma.combine(&result_left, &result_right)
        }

        pub fn partition_point(&mut self, mut pred: impl FnMut(&M::X, u32) -> bool) -> usize {
            let mut i = 1;
            let mut width = self.n as u32;
            while i < self.n {
                width >>= 1;
                let value = unsafe { &*(&self.lazy[i] as *const _) };
                self.apply(i << 1, width, value);
                self.apply(i << 1 | 1, width, value);
                self.lazy[i] = self.ma.id_action();
                i <<= 1;
                if pred(&self.sum[i], width) {
                    i |= 1;
                }
            }
            i - self.n
        }
    }
}

#[derive(Clone, Debug)]
struct IntervalSum {
    sum: i32,
    max: i32,
    max_left: i32,
    max_right: i32,
}

impl IntervalSum {
    fn new(value: i32) -> Self {
        let max = value.max(0);
        Self {
            sum: value,
            max,
            max_left: max,
            max_right: max,
        }
    }

    fn rev(&self) -> Self {
        Self {
            sum: self.sum,
            max: self.max,
            max_left: self.max_right,
            max_right: self.max_left,
        }
    }
}

struct IntervalSumOp;

impl segtree_lazy::MonoidAction for IntervalSumOp {
    type X = IntervalSum;
    type F = Option<i32>;
    fn id(&self) -> Self::X {
        IntervalSum::new(0)
    }
    fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X {
        IntervalSum {
            sum: lhs.sum + rhs.sum,
            max: lhs.max.max(rhs.max).max(lhs.max_right + rhs.max_left),
            max_left: lhs.max_left.max(lhs.sum + rhs.max_left),
            max_right: rhs.max_right.max(rhs.sum + lhs.max_right),
        }
    }
    fn id_action(&self) -> Self::F {
        None
    }
    fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F {
        lhs.or_else(|| *rhs)
    }
    fn apply_to_sum(&self, f: &Self::F, x_count: u32, x_sum: &Self::X) -> Self::X {
        let res = match f {
            None => x_sum.clone(),
            Some(value) => {
                let sum = x_count as i32 * value;
                let max = if *value > 0 {
                    x_count as i32 * *value
                } else {
                    *value
                };
                IntervalSum {
                    sum,
                    max,
                    max_left: max,
                    max_right: max,
                }
            }
        };
        res
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let weights: Vec<i32> = (0..n).map(|_| input.value()).collect();

    let edges = (0..n - 1).map(|_| (input.value::<u32>() - 1, input.value::<u32>() - 1));
    let hld = hld::HLD::from_edges(n, edges, 0, false);
    let mut weights_reindexed = vec![0; n];
    for i in 0..n {
        weights_reindexed[hld.segmented_idx[i] as usize] = weights[i];
    }
    let mut weights = segtree_lazy::SegTree::from_iter(
        n,
        weights_reindexed.into_iter().map(IntervalSum::new),
        IntervalSumOp,
    );

    let q: usize = input.value();
    for _ in 0..q {
        let cmd = input.token();
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        match cmd {
            "1" => {
                let mut res_left = weights.ma.id();
                let mut res_right = weights.ma.id();
                hld.for_each_path_splitted(u, v, |u, v, is_left_path, _has_lca| {
                    let (u, v) = (hld.segmented_idx[u] as usize, hld.segmented_idx[v] as usize);
                    let x = weights.query_range(u..v + 1);
                    if is_left_path {
                        res_left = weights.ma.combine(&x, &res_left);
                    } else {
                        res_right = weights.ma.combine(&x, &res_right);
                    }
                });
                let res = weights.ma.combine(&res_left.rev(), &res_right);
                writeln!(output, "{}", res.max).unwrap();
            }
            "2" => {
                let w = input.value();
                hld.for_each_path(u, v, |u, v, _is_lca| {
                    let (u, v) = (hld.segmented_idx[u] as usize, hld.segmented_idx[v] as usize);
                    weights.apply_range(u..v + 1, Some(w));
                });
            }

            _ => panic!(),
        }
    }
}
