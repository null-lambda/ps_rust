use std::{cmp::Ordering, io::Write};

use heap::RemovableHeap;
use jagged::Jagged;
use segtree::{Monoid, SegTree};

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

pub mod segtree {
    use std::ops::Range;

    pub trait Monoid {
        type X;
        const IS_COMMUTATIVE: bool = false;
        fn id(&self) -> Self::X;
        fn op(&self, a: &Self::X, b: &Self::X) -> Self::X;
    }

    #[derive(Debug)]
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
            I::IntoIter: ExactSizeIterator<Item = M::X>,
        {
            let iter = iter.into_iter();
            let n = iter.len();
            let mut sum: Vec<_> = (0..n).map(|_| monoid.id()).chain(iter).collect();
            for i in (0..n).rev() {
                sum[i] = monoid.op(&sum[i << 1], &sum[i << 1 | 1]);
            }
            Self { n, sum, monoid }
        }

        pub fn modify(&mut self, mut idx: usize, f: impl FnOnce(&mut M::X)) {
            debug_assert!(idx < self.n);
            idx += self.n;
            f(&mut self.sum[idx]);
            while idx > 1 {
                idx >>= 1;
                self.sum[idx] = self.monoid.op(&self.sum[idx << 1], &self.sum[idx << 1 | 1]);
            }
        }

        pub fn get(&self, idx: usize) -> &M::X {
            &self.sum[idx + self.n]
        }

        pub fn mapped_sum_range<N: Monoid>(
            &self,
            range: Range<usize>,
            codomain: &N,
            morphism: impl Fn(&M::X) -> N::X,
        ) -> N::X {
            let Range { mut start, mut end } = range;
            if start >= end {
                return codomain.id();
            }
            debug_assert!(start < self.n && end <= self.n);
            start += self.n;
            end += self.n;

            if N::IS_COMMUTATIVE {
                let mut result = codomain.id();
                while start < end {
                    if start & 1 != 0 {
                        result = codomain.op(&result, &morphism(&self.sum[start]));
                    }
                    if end & 1 != 0 {
                        result = codomain.op(&morphism(&self.sum[end - 1]), &result);
                    }
                    start = (start + 1) >> 1;
                    end >>= 1;
                }
                result
            } else {
                let (mut result_left, mut result_right) = (codomain.id(), codomain.id());
                while start < end {
                    if start & 1 != 0 {
                        result_left = codomain.op(&result_left, &morphism(&self.sum[start]));
                    }
                    if end & 1 != 0 {
                        result_right = codomain.op(&morphism(&self.sum[end - 1]), &result_right);
                    }
                    start = (start + 1) >> 1;
                    end >>= 1;
                }
                codomain.op(&result_left, &result_right)
            }
        }

        pub fn sum_all(&self) -> &M::X {
            assert!(self.n.is_power_of_two());
            &self.sum[1]
        }
    }

    impl<M: Monoid> SegTree<M>
    where
        M::X: Clone,
    {
        pub fn sum_range(&self, range: Range<usize>) -> M::X {
            self.mapped_sum_range(range, &self.monoid, |x| x.clone())
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

struct MaxOp;

impl Monoid for MaxOp {
    type X = i32;
    const IS_COMMUTATIVE: bool = true;
    fn id(&self) -> Self::X {
        -1_000_000_100
    }
    fn op(&self, a: &Self::X, b: &Self::X) -> Self::X {
        (*a).max(*b)
    }
}

struct DoubleMaxOp;

impl DoubleMaxOp {
    fn singleton(&self, value: i32) -> <Self as Monoid>::X {
        (value, MaxOp.id())
    }
}

impl Monoid for DoubleMaxOp {
    type X = (i32, i32);
    const IS_COMMUTATIVE: bool = true;
    fn id(&self) -> Self::X {
        (MaxOp.id(), MaxOp.id())
    }
    fn op(&self, a: &Self::X, b: &Self::X) -> Self::X {
        match a.0.cmp(&b.0) {
            Ordering::Less => (b.0, a.0.max(b.1)),
            Ordering::Equal => (a.0, a.0),
            Ordering::Greater => (a.0, a.1.max(b.0)),
        }
    }
}

pub mod heap {
    use std::collections::BinaryHeap;

    #[derive(Clone)]
    pub struct RemovableHeap<T> {
        items: BinaryHeap<T>,
        to_remove: BinaryHeap<T>,
    }

    impl<T: Ord> RemovableHeap<T> {
        pub fn new() -> Self {
            Self {
                items: BinaryHeap::new().into(),
                to_remove: BinaryHeap::new().into(),
            }
        }

        pub fn push(&mut self, item: T) {
            self.items.push(item);
        }

        pub fn remove(&mut self, item: T) {
            self.to_remove.push(item);
        }

        fn clean_top(&mut self) {
            while let Some((r, x)) = self.to_remove.peek().zip(self.items.peek()) {
                if r != x {
                    break;
                }
                self.to_remove.pop();
                self.items.pop();
            }
        }

        pub fn peek(&mut self) -> Option<&T> {
            self.clean_top();
            self.items.peek()
        }

        pub fn pop(&mut self) -> Option<T> {
            self.clean_top();
            self.items.pop()
        }
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
        let w: i32 = input.value();
        edges.push((u, (v, w)));
        edges.push((v, (u, w)));
    }
    let neighbors = jagged::CSR::from_assoc_list(n, &edges);

    let base_root = 0;
    let dist = {
        let mut timer = 0;
        let mut bfs_order = vec![(base_root as u32, base_root as u32)];
        let mut depth = vec![0; n];
        while let Some(&(u, p)) = bfs_order.get(timer) {
            for &(v, w) in neighbors.get(u as usize) {
                if v == p {
                    continue;
                }
                depth[v as usize] = depth[u as usize] + w;
                bfs_order.push((v, u));
            }

            timer += 1;
        }

        let edges_undirected = edges
            .iter()
            .map(|&(u, (v, _))| (u, v))
            .filter(|&(u, v)| u < v);
        let hld = hld::HLD::from_edges(n, edges_undirected, 0, false);
        move |u: usize, v: usize| depth[u] + depth[v] - 2 * depth[hld.lca(u, v)]
    };

    const UNSET: u32 = u32::MAX;
    let mut size = vec![0; n];
    let mut visited = vec![false; n];
    let mut parent_centroid = vec![UNSET; n];
    centroid::init_size(&neighbors, &mut size, base_root, base_root);
    centroid::build_centroid_tree(
        &neighbors,
        &mut size,
        &mut visited,
        &mut parent_centroid,
        base_root,
    );

    let mut n_child_centroid = vec![0; n];
    let mut index_in_parent_centroid = vec![0; n];
    for u in 0..n {
        let p = parent_centroid[u] as usize;
        if p == UNSET as usize {
            continue;
        }

        index_in_parent_centroid[u] = n_child_centroid[p];
        n_child_centroid[p] += 1;
    }

    let n_pad = n.next_power_of_two();
    let mut is_white = vec![false; n];
    let mut subtree_dist_freq: Vec<Vec<RemovableHeap<i32>>> = n_child_centroid
        .iter()
        .map(|&s| vec![RemovableHeap::new(); s])
        .collect();
    let mut subtree_agg: Vec<SegTree<_>> = n_child_centroid
        .iter()
        .map(|&s| SegTree::with_size((s + 1).next_power_of_two(), DoubleMaxOp))
        .collect();
    let mut diameter_through_subcentroid = SegTree::with_size(n_pad, MaxOp);

    let q: usize = input.value();
    let queries = (0..n).map(|u| ("1", u)).chain((0..q).map(|_| {
        let cmd = input.token();
        match cmd {
            "1" => {
                let u = input.value::<usize>() - 1;
                (cmd, u)
            }
            "2" => (cmd, 0),
            _ => panic!(),
        }
    }));

    for (cmd, u0) in queries {
        match cmd {
            "1" => {
                is_white[u0] ^= true;
                subtree_agg[u0].modify(n_child_centroid[u0], |x| {
                    *x = if is_white[u0] {
                        (0, 0)
                    } else {
                        DoubleMaxOp.id()
                    }
                });
                let mut u = u0;
                loop {
                    let (d1, d2) = *subtree_agg[u].sum_all();
                    diameter_through_subcentroid.modify(u, |x| *x = d1 + d2);

                    let p = parent_centroid[u] as usize;
                    if p == UNSET as usize {
                        break;
                    }

                    let iu = index_in_parent_centroid[u];
                    if is_white[u0] {
                        subtree_dist_freq[p][iu].push(dist(p, u0));
                    } else {
                        subtree_dist_freq[p][iu].remove(dist(p, u0));
                    }
                    subtree_agg[p].modify(iu, |x| {
                        *x = subtree_dist_freq[p][iu]
                            .peek()
                            .map_or(DoubleMaxOp.id(), |&d| DoubleMaxOp.singleton(d))
                    });

                    u = p;
                }
            }
            "2" => {
                let ans = (*diameter_through_subcentroid.sum_all()).max(-1);
                writeln!(output, "{}", ans).unwrap();
            }
            _ => panic!(),
        }
    }
}
