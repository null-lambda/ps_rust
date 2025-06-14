use std::{cmp::Ordering, collections::HashSet, io::Write};

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    extern "C" {
        fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64)
            -> *mut u8;
        fn fstat(fd: i32, stat: *mut usize) -> i32;
    }

    pub struct InputAtOnce {
        buf: &'static [u8],
    }

    impl InputAtOnce {
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

        pub fn token(&mut self) -> &'static str {
            self.skip();
            let start = self.buf.as_ptr();
            loop {
                match self.buf {
                    &[..=b' ', ..] => break,
                    _ => self.buf = &self.buf[1..],
                }
            }
            let end = self.buf.as_ptr();
            unsafe {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    start,
                    end.offset_from(start) as usize,
                ))
            }
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
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

    pub fn stdin() -> InputAtOnce {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let buf = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        let buf =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(buf, stat[6])) };
        InputAtOnce {
            buf: buf.as_bytes(),
        }
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::with_capacity(1 << 16, stdout)
    }
}

pub mod map {
    use std::{hash::Hash, mem::MaybeUninit};

    pub enum AdaptiveHashSet<K, const STACK_CAP: usize> {
        Small([MaybeUninit<K>; STACK_CAP], usize),
        Large(std::collections::HashSet<K>),
    }

    impl<K, const STACK_CAP: usize> Drop for AdaptiveHashSet<K, STACK_CAP> {
        fn drop(&mut self) {
            match self {
                Self::Small(arr, size) => {
                    for i in 0..*size {
                        unsafe { arr[i].assume_init_drop() }
                    }
                }
                _ => {}
            }
        }
    }

    impl<K: Clone, const STACK_CAP: usize> Clone for AdaptiveHashSet<K, STACK_CAP> {
        fn clone(&self) -> Self {
            match self {
                Self::Small(arr, size) => {
                    let mut cloned = std::array::from_fn(|_| MaybeUninit::uninit());
                    for i in 0..*size {
                        cloned[i] = MaybeUninit::new(unsafe { arr[i].assume_init_ref().clone() });
                    }
                    Self::Small(cloned, *size)
                }
                Self::Large(set) => Self::Large(set.clone()),
            }
        }
    }

    impl<K, const STACK_CAP: usize> Default for AdaptiveHashSet<K, STACK_CAP> {
        fn default() -> Self {
            Self::Small(std::array::from_fn(|_| MaybeUninit::uninit()), 0)
        }
    }

    impl<K: Eq + Hash, const STACK_CAP: usize> AdaptiveHashSet<K, STACK_CAP> {
        pub fn len(&self) -> usize {
            match self {
                Self::Small(_, size) => *size,
                Self::Large(set) => set.len(),
            }
        }

        pub fn contains(&self, key: &K) -> bool {
            match self {
                Self::Small(arr, size) => arr[..*size]
                    .iter()
                    .find(|&x| unsafe { x.assume_init_ref() } == key)
                    .is_some(),
                Self::Large(set) => set.contains(key),
            }
        }

        pub fn insert(&mut self, key: K) -> bool {
            if self.contains(&key) {
                return false;
            }
            match self {
                Self::Small(arr, size) => {
                    if arr[..*size]
                        .iter()
                        .find(|&x| unsafe { x.assume_init_ref() } == &key)
                        .is_some()
                    {
                        return false;
                    }

                    if *size < STACK_CAP {
                        arr[*size] = MaybeUninit::new(key);
                        *size += 1;
                    } else {
                        let arr =
                            std::mem::replace(arr, std::array::from_fn(|_| MaybeUninit::uninit()));
                        *size = 0; // Prevent `drop` call on arr elements
                        *self = Self::Large(
                            arr.into_iter()
                                .map(|x| unsafe { x.assume_init() })
                                .chain(Some(key))
                                .collect(),
                        );
                    }
                    true
                }
                Self::Large(set) => set.insert(key),
            }
        }

        pub fn remove(&mut self, key: &K) -> bool {
            match self {
                Self::Small(_, 0) => false,
                Self::Small(arr, size) => {
                    for i in 0..*size {
                        unsafe {
                            if arr[i].assume_init_ref() == key {
                                *size -= 1;
                                arr[i].assume_init_drop();
                                arr[i] = std::mem::replace(&mut arr[*size], MaybeUninit::uninit());
                                return true;
                            }
                        }
                    }

                    false
                }
                Self::Large(set) => set.remove(key),
            }
        }

        pub fn for_each(&mut self, mut visitor: impl FnMut(&K)) {
            match self {
                Self::Small(arr, size) => {
                    arr[..*size]
                        .iter()
                        .for_each(|x| visitor(unsafe { x.assume_init_ref() }));
                }
                Self::Large(set) => set.iter().for_each(visitor),
            }
        }
    }
}

pub mod tree_decomp {
    use std::collections::VecDeque;

    // pub type HashSet<T> = std::collections::HashSet<T>;
    pub type HashSet<T> = crate::map::AdaptiveHashSet<T, 5>;

    pub const UNSET: u32 = u32::MAX;

    // Tree decomposition of treewidth 2.
    #[derive(Clone)]
    pub struct TW2 {
        // Perfect elimination ordering in the chordal completion
        pub topological_order: Vec<u32>,
        pub t_in: Vec<u32>,
        pub parents: Vec<[u32; 2]>,
    }

    impl TW2 {
        pub fn from_edges(
            n_verts: usize,
            edges: impl IntoIterator<Item = [u32; 2]>,
        ) -> Option<Self> {
            let mut neighbors = vec![HashSet::default(); n_verts];
            for [u, v] in edges {
                neighbors[u as usize].insert(v);
                neighbors[v as usize].insert(u);
            }

            let mut visited = vec![false; n_verts];
            let mut parents = vec![[UNSET; 2]; n_verts];

            let mut topological_order = vec![];
            let mut t_in = vec![UNSET; n_verts];
            let mut root = None;

            let mut queue: [_; 3] = std::array::from_fn(|_| VecDeque::new());
            for u in 0..n_verts {
                let d = neighbors[u].len();
                if d <= 2 {
                    visited[u] = true;
                    queue[d].push_back(u as u32);
                }
            }

            while let Some(u) = (0..=2).flat_map(|i| queue[i].pop_front()).next() {
                t_in[u as usize] = topological_order.len() as u32;
                topological_order.push(u);

                match neighbors[u as usize].len() {
                    0 => {
                        if let Some(old_root) = root {
                            parents[old_root as usize][0] = u;
                        }
                        root = Some(u);
                    }
                    1 => {
                        let mut p = UNSET;
                        std::mem::take(&mut neighbors[u as usize]).for_each(|&v| p = v);
                        neighbors[p as usize].remove(&u);

                        parents[u as usize][0] = p;

                        if !visited[p as usize] && neighbors[p as usize].len() <= 2 {
                            visited[p as usize] = true;
                            queue[neighbors[p as usize].len()].push_back(p);
                        }
                    }
                    2 => {
                        let mut ps = [UNSET; 2];
                        let mut i = 0;
                        std::mem::take(&mut neighbors[u as usize]).for_each(|&v| {
                            ps[i] = v;
                            i += 1;
                        });
                        let [p, q] = ps;

                        neighbors[p as usize].remove(&u);
                        neighbors[q as usize].remove(&u);

                        neighbors[p as usize].insert(q);
                        neighbors[q as usize].insert(p);

                        parents[u as usize] = [p, q];

                        for w in [p, q] {
                            if !visited[w as usize] && neighbors[w as usize].len() <= 2 {
                                visited[w as usize] = true;
                                queue[neighbors[w as usize].len()].push_back(w);
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }

            if topological_order.len() != n_verts {
                return None;
            }
            assert_eq!(root, topological_order.iter().last().copied());

            for u in 0..n_verts {
                let ps = &mut parents[u];
                if ps[1] != UNSET && t_in[ps[0] as usize] > t_in[ps[1] as usize] {
                    ps.swap(0, 1);
                }
            }

            Some(Self {
                parents,
                topological_order,
                t_in,
            })
        }
    }
}

pub mod reroot {
    // O(n) rerooting dp for trees with combinable, non-invertible pulling operation. (Monoid action)
    // Technically, its a static, offline variant of the top tree.
    // https://codeforces.com/blog/entry/124286
    // https://github.com/koosaga/olympiad/blob/master/Library/codes/data_structures/all_direction_tree_dp.cpp
    pub trait AsBytes<const N: usize> {
        unsafe fn as_bytes(self) -> [u8; N];
        unsafe fn decode(bytes: [u8; N]) -> Self;
    }

    #[macro_export]
    macro_rules! impl_as_bytes {
        ($T:ty, $N: ident) => {
            const $N: usize = std::mem::size_of::<$T>();

            impl crate::reroot::AsBytes<$N> for $T {
                unsafe fn as_bytes(self) -> [u8; $N] {
                    std::mem::transmute::<$T, [u8; $N]>(self)
                }

                unsafe fn decode(bytes: [u8; $N]) -> $T {
                    std::mem::transmute::<[u8; $N], $T>(bytes)
                }
            }
        };
    }
    pub use impl_as_bytes;

    impl_as_bytes!((), __N_UNIT);
    impl_as_bytes!(u32, __N_U32);
    impl_as_bytes!(u64, __N_U64);
    impl_as_bytes!(i32, __N_I32);
    impl_as_bytes!(i64, __N_I64);

    pub trait DpSpec {
        type E: Clone; // edge weight
        type V: Clone; // Subtree aggregate on a node
        type F: Clone; // pulling operation (edge dp)
        fn lift_to_action(&self, node: &Self::V, weight: &Self::E) -> Self::F;
        fn id_action(&self) -> Self::F;
        fn rake_action(&self, node: &Self::V, lhs: &mut Self::F, rhs: &Self::F);
        fn apply(&self, node: &mut Self::V, action: &Self::F);
        fn finalize(&self, node: &mut Self::V);
    }

    fn xor_assign_bytes<const N: usize>(xs: &mut [u8; N], ys: [u8; N]) {
        for (x, y) in xs.iter_mut().zip(&ys) {
            *x ^= *y;
        }
    }

    const UNSET: u32 = !0;

    fn for_each_in_list(xor_links: &[u32], start: u32, mut visitor: impl FnMut(u32)) -> u32 {
        let mut u = start;
        let mut prev = UNSET;
        loop {
            visitor(u);
            let next = xor_links[u as usize] ^ prev;
            if next == UNSET {
                return u;
            }
            prev = u;
            u = next;
        }
    }

    pub fn run<'a, const N: usize, R>(
        cx: &R,
        n_verts: usize,
        edges: impl IntoIterator<Item = (u32, u32, R::E)>,
        data: &mut [R::V],
        mut yield_edge_dp: impl FnMut(usize, &R::F, &R::F, &R::E),
    ) where
        R: DpSpec,
        R::E: Default + AsBytes<N>,
    {
        // Fast tree reconstruction with XOR-linked traversal
        // https://codeforces.com/blog/entry/135239
        let root = 0;
        let mut degree = vec![0; n_verts];
        let mut xor_neighbors: Vec<(u32, [u8; N])> = vec![(0u32, [0u8; N]); n_verts];
        for (u, v, w) in edges
            .into_iter()
            .flat_map(|(u, v, w)| [(u, v, w.clone()), (v, u, w)])
        {
            degree[u as usize] += 1;
            xor_neighbors[u as usize].0 ^= v;
            xor_assign_bytes(&mut xor_neighbors[u as usize].1, unsafe {
                AsBytes::as_bytes(w)
            });
        }

        // Upward propagation
        let data_orig = data.to_owned();
        let mut action_upward = vec![cx.id_action(); n_verts];
        let mut topological_order = vec![];
        let mut first_child = vec![UNSET; n_verts];
        let mut xor_siblings = vec![UNSET; n_verts];
        degree[root] += 2;
        for mut u in 0..n_verts {
            while degree[u as usize] == 1 {
                let (p, w_encoded) = xor_neighbors[u as usize];
                degree[u as usize] = 0;
                degree[p as usize] -= 1;
                xor_neighbors[p as usize].0 ^= u as u32;
                xor_assign_bytes(&mut xor_neighbors[p as usize].1, w_encoded);
                let w = unsafe { AsBytes::decode(w_encoded) };

                let c = first_child[p as usize];
                xor_siblings[u as usize] = c ^ UNSET;
                if c != UNSET {
                    xor_siblings[c as usize] ^= u as u32 ^ UNSET;
                }
                first_child[p as usize] = u as u32;

                let mut sum_u = data[u as usize].clone();
                cx.finalize(&mut sum_u);
                action_upward[u as usize] = cx.lift_to_action(&sum_u, &w);
                cx.apply(&mut data[p as usize], &action_upward[u as usize]);

                topological_order.push((u as u32, p, w));
                u = p as usize;
            }
        }
        topological_order.push((root as u32, UNSET, R::E::default()));
        cx.finalize(&mut data[root]);

        // Downward propagation
        let mut action_exclusive = vec![cx.id_action(); n_verts];
        for (u, p, w) in topological_order.into_iter().rev() {
            let action_from_parent;
            if p != UNSET {
                let mut sum_exclusive = data_orig[p as usize].clone();
                cx.apply(&mut sum_exclusive, &action_exclusive[u as usize]);
                cx.finalize(&mut sum_exclusive);
                action_from_parent = cx.lift_to_action(&sum_exclusive, &w);

                let sum_u = &mut data[u as usize];
                cx.apply(sum_u, &action_from_parent);
                cx.finalize(sum_u);
                yield_edge_dp(
                    u as usize,
                    &action_upward[u as usize],
                    &action_from_parent,
                    &w,
                );
            } else {
                action_from_parent = cx.id_action();
            }

            if first_child[u as usize] != UNSET {
                let mut prefix = action_from_parent.clone();
                let last = for_each_in_list(&xor_siblings, first_child[u as usize], |v| {
                    action_exclusive[v as usize] = prefix.clone();
                    cx.rake_action(
                        &data_orig[u as usize],
                        &mut prefix,
                        &action_upward[v as usize],
                    );
                });

                let mut postfix = cx.id_action();
                for_each_in_list(&xor_siblings, last, |v| {
                    cx.rake_action(
                        &data_orig[u as usize],
                        &mut action_exclusive[v as usize],
                        &postfix,
                    );
                    cx.rake_action(
                        &data_orig[u as usize],
                        &mut postfix,
                        &action_upward[v as usize],
                    );
                });
            }
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
            edges: impl IntoIterator<Item = [u32; 2]>,
            root: usize,
            use_dfs_ordering: bool,
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

use segtree::Monoid;
use tree_decomp::UNSET;
const TW: usize = 2;
const K: usize = TW + 1;
type Bag = [u32; K];
type Mat = [[u32; K]; K];

fn floyd_warshall(mat: &mut Mat) {
    let m = bag_len(&mat[0]);
    for k in 0..m {
        for i in 0..m {
            for j in 0..m {
                mat[i][j] = mat[i][j].min(mat[i][k] + mat[k][j]);
            }
        }
    }
}

fn sorted2<T: PartialOrd>(mut xs: [T; 2]) -> [T; 2] {
    if xs[0] > xs[1] {
        xs.swap(0, 1);
    }
    xs
}

const INF: u32 = 1 << 29;

fn bag_len(bag: &Bag) -> usize {
    (0..K).find(|&i| bag[i] == UNSET).unwrap_or(K)
}

enum MergeType<T> {
    Left(T),
    Right(T),
    Equal(T, T),
}

fn iter_merge_bag(lhs: &Bag, rhs: &Bag, mut f: impl FnMut(MergeType<usize>, usize)) -> usize {
    let mut i = 0;
    let mut j = 0;
    let mut k = 0;
    while i < K && j < K {
        match lhs[i].cmp(&rhs[j]) {
            Ordering::Less => {
                f(MergeType::Left(i), k);
                i += 1;
            }
            Ordering::Greater => {
                f(MergeType::Right(j), k);
                j += 1;
            }
            Ordering::Equal => {
                f(MergeType::Equal(i, j), k);
                i += 1;
                j += 1;
                k += 1;
            }
        }
    }
    k
}

fn iter_inter_bag(lhs: &Bag, rhs: &Bag, mut f: impl FnMut(usize, usize, usize)) -> usize {
    iter_merge_bag(lhs, rhs, |ty, k| {
        if let MergeType::Equal(i, j) = ty {
            f(i, j, k);
        }
    })
}

fn inter_bag(lhs: &Bag, rhs: &Bag) -> Bag {
    let mut res = [UNSET; K];
    iter_inter_bag(lhs, rhs, |i, _, k| res[k] = lhs[i]);
    res
}

struct InnerDist;

reroot::impl_as_bytes!([Bag; 2], __N_BAG_ARRAY_2);

impl reroot::DpSpec for InnerDist {
    type E = [Bag; 2];
    type V = (Bag, Mat);
    type F = Option<Mat>;

    fn lift_to_action(&self, (bag, mat): &Self::V, inter: &Self::E) -> Self::F {
        let target_bag = if bag == &inter[0] { inter[1] } else { inter[0] };
        let mut pairs = [[UNSET; 2]; K];
        let m = iter_inter_bag(&bag, &target_bag, |i, j, k| {
            pairs[k] = [i as u32, j as u32];
        });

        let mut inter_mat = [[INF; K]; K];
        for &[i0, i1] in &pairs[..m] {
            for &[j0, j1] in &pairs[..m] {
                inter_mat[i1 as usize][j1 as usize] = mat[i0 as usize][j0 as usize];
            }
        }
        Some(inter_mat)
    }

    fn id_action(&self) -> Self::F {
        None
    }

    fn rake_action(&self, (bag, _): &Self::V, lhs: &mut Self::F, rhs: &Self::F) {
        match (lhs.as_mut(), rhs.as_ref()) {
            (Some(lhs), Some(rhs)) => {
                let m = bag_len(bag);
                for i in 0..m {
                    for j in 0..m {
                        lhs[i][j] = lhs[i][j].min(rhs[i][j]);
                    }
                }
            }
            (None, Some(_)) => *lhs = *rhs,
            _ => {}
        }
    }

    fn apply(&self, (bag, mat): &mut Self::V, action: &Self::F) {
        if let Some(sub_mat) = action {
            let m = bag_len(bag);
            for i in 0..m {
                for j in 0..m {
                    mat[i][j] = mat[i][j].min(sub_mat[i][j]);
                }
            }
        }
    }

    fn finalize(&self, (_bag, mat): &mut Self::V) {
        floyd_warshall(mat);
    }
}

struct BagDist;

impl BagDist {
    fn rev(&self, path: &<Self as Monoid>::X) -> <Self as Monoid>::X {
        path.map(|(mut ends, mat)| {
            let mut transposed = [[INF; K]; K];
            for i in 0..K {
                for j in 0..K {
                    transposed[i][j] = mat[j][i];
                }
            }
            ends.swap(0, 1);
            (ends, transposed)
        })
    }
}

impl segtree::Monoid for BagDist {
    type X = Option<([Bag; 2], Mat)>;

    fn id(&self) -> Self::X {
        None
    }

    fn op(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X {
        lhs.as_ref()
            .zip(rhs.as_ref())
            .map(|((lhs_ends, lhs_mat), (rhs_ends, rhs_mat))| {
                let mut pairs = [[UNSET; 2]; K];
                let m_inter = iter_inter_bag(&lhs_ends[1], &rhs_ends[0], |i, j, k| {
                    pairs[k] = [i as u32, j as u32];
                });

                let mut mat = [[INF; 3]; 3];
                let m0 = bag_len(&lhs_ends[0]);
                let m1 = bag_len(&rhs_ends[1]);
                for i in 0..m0 {
                    for j in 0..m1 {
                        for &[k, l] in &pairs[..m_inter] {
                            mat[i][j] =
                                mat[i][j].min(lhs_mat[i][k as usize] + rhs_mat[l as usize][j]);
                        }
                    }
                }
                ([lhs_ends[0], rhs_ends[1]], mat)
            })
            .or_else(|| *lhs)
            .or_else(|| *rhs)
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let edges = ((0..n as u32).map(|i| [i, (i + 1) % n as u32]))
        .chain((0..n - 3).map(|_| [input.u32() - 1, input.u32() - 1]))
        .map(sorted2)
        .collect::<HashSet<_>>();

    let td = tree_decomp::TW2::from_edges(n, edges.iter().cloned()).unwrap();
    let mut dp_inter = vec![];
    for u in 0..n {
        let mut bag = [UNSET; K];
        bag[0] = u as u32;
        bag[1] = td.parents[u][0];
        bag[2] = td.parents[u][1];
        bag.sort_unstable();
        let m = bag_len(&bag);

        let mut mat = [[INF; K]; K];
        for i in 0..m {
            for j in 0..m {
                mat[i][j] = if i == j {
                    0
                } else if edges.contains(&sorted2([bag[i], bag[j]])) {
                    1
                } else {
                    INF
                };
            }
        }
        dp_inter.push((bag, mat));
    }
    let bag_edges: Vec<_> = (0..n as u32)
        .filter_map(|u| {
            let p = td.parents[u as usize][0];
            (p != UNSET).then(|| (p, u, [dp_inter[p as usize].0, dp_inter[u as usize].0]))
        })
        .collect();

    reroot::run(
        &InnerDist,
        n,
        bag_edges.iter().copied(),
        &mut dp_inter,
        |_, _, _, _| {},
    );

    let bag_edges_unweighted = bag_edges.iter().map(|&(u, v, _)| [u, v]);
    let hld = hld::HLD::from_edges(n, bag_edges_unweighted, 0, false);
    let sid = |u| hld.segmented_idx[u] as usize;

    let mut weights = vec![None; n];
    for u in 0..n {
        let (bag, mat) = dp_inter[u];
        weights[sid(u)] = Some(([bag; 2], mat));
    }

    let st = segtree::SegTree::from_iter(weights, BagDist);

    for _ in 0..input.value() {
        let u = input.u32() as usize - 1;
        let v = input.u32() as usize - 1;

        let mut lhs = None;
        let mut rhs = None;
        hld.for_each_path_splitted(u, v, |u, v, on_left, _| {
            let chain_sum = st.sum_range(sid(u)..sid(v) + 1);
            if on_left {
                lhs = BagDist.op(&chain_sum, &lhs);
            } else {
                rhs = BagDist.op(&chain_sum, &rhs);
            }
        });

        let ([bag_u, bag_v], mat) = BagDist.op(&BagDist.rev(&lhs), &rhs).unwrap();

        let i = bag_u.iter().position(|&x| x == u as u32).unwrap();
        let j = bag_v.iter().position(|&x| x == v as u32).unwrap();
        let ans = mat[i][j];
        writeln!(output, "{}", ans).ok();
    }
}
