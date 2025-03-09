use std::io::Write;

use jagged::Jagged;

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

pub mod debug {
    pub fn with(#[allow(unused_variables)] f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
    }

    #[cfg(debug_assertions)]
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
    pub struct Label<T>(T);

    #[cfg(not(debug_assertions))]
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
    pub struct Label<T>(std::marker::PhantomData<T>);

    impl<T> Label<T> {
        #[inline]
        pub fn new_with(value: impl FnOnce() -> T) -> Self {
            #[cfg(debug_assertions)]
            {
                Self(value())
            }
            #[cfg(not(debug_assertions))]
            {
                Self(Default::default())
            }
        }

        pub fn with(&mut self, #[allow(unused_variables)] f: impl FnOnce(&mut T)) {
            #[cfg(debug_assertions)]
            f(&mut self.0)
        }
    }

    impl<T: std::fmt::Debug> std::fmt::Debug for Label<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            #[cfg(debug_assertions)]
            {
                write!(f, "{:?}", self.0)
            }
            #[cfg(not(debug_assertions))]
            {
                write!(f, "()")
            }
        }
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
        fn get(&'a self, u: usize) -> &'a [T];
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
        fn get(&'a self, u: usize) -> &'a [T] {
            &self.as_ref()[u]
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
            let v: Vec<Vec<&T>> = (0..self.len())
                .map(|i| self.get(i).iter().collect())
                .collect();
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

        fn get(&'a self, u: usize) -> &'a [T] {
            &self.data[self.head[u] as usize..self.head[u + 1] as usize]
        }
    }
}

fn expand_vec_with<T>(xs: &mut Vec<T>, n: usize, f: impl FnMut() -> T) {
    if xs.len() < n {
        xs.resize_with(n, f);
    }
}

fn accumulate<T: std::ops::AddAssign + Copy>(xs: &mut [T]) {
    for i in 1..xs.len() {
        xs[i] += xs[i - 1];
    }
}

fn get_or_last<T: Default + Copy>(xs: &[T], i: usize) -> T {
    xs.get(i)
        .copied()
        .unwrap_or_else(|| xs.last().copied().unwrap_or(T::default()))
}

const UNSET: u32 = hld::UNSET;

struct CentroidTree<'a, J: Jagged<'a, (u32, ())>> {
    n_cutoff: usize,

    neighbors: &'a J,

    parent: Vec<u32>,
    level: Vec<u32>,
    size: Vec<u32>,

    n_branch: usize,

    depth_prefix: Vec<Vec<u32>>,
    depth_prefix_in_branch: Vec<Vec<u32>>,

    branch_in_layer: Vec<Vec<u32>>,
    depth_in_layer: Vec<Vec<u32>>,
}

impl<'a, J: Jagged<'a, (u32, ())>> CentroidTree<'a, J> {
    fn n_verts(&self) -> usize {
        self.size.len()
    }

    fn new(neighbors: &'a J, n_cutoff: usize) -> Self {
        let n = neighbors.len();

        let mut this = Self {
            n_cutoff,

            neighbors,

            parent: vec![UNSET; n],
            level: vec![UNSET; n],
            size: vec![1; n],

            n_branch: 0,

            depth_prefix: vec![vec![0]; n],
            depth_prefix_in_branch: vec![],

            branch_in_layer: vec![],
            depth_in_layer: vec![],
        };

        let init = 0;
        this.init_size(init);
        this.decompose_dnc(init, UNSET as usize);
        this
    }

    fn expand_level(&mut self, l_max: usize) {
        let n = self.n_verts();
        expand_vec_with(&mut self.branch_in_layer, l_max + 1, || vec![UNSET; n]);
        expand_vec_with(&mut self.depth_in_layer, l_max + 1, || vec![UNSET; n]);
    }

    fn init_size(&mut self, u: usize) {
        let mut bfs = vec![(u as u32, UNSET)];
        let mut timer = 0;
        while let Some(&(u, p)) = bfs.get(timer) {
            timer += 1;
            for &(v, ()) in self.neighbors.get(u as usize) {
                if v == p {
                    continue;
                }
                bfs.push((v, u));
            }
        }

        for (u, p) in bfs.into_iter().skip(1).rev() {
            self.size[p as usize] += self.size[u as usize];
        }
    }

    fn reroot_to_centroid(&mut self, u: &mut usize) {
        let threshold = (self.size[*u] + 1) / 2;

        let mut p = UNSET as usize;
        'outer: loop {
            for &(v, ()) in self.neighbors.get(*u) {
                let v = v as usize;
                if v == p || self.level[v as usize] != UNSET {
                    continue;
                }
                if self.size[v] >= threshold {
                    self.size[*u] -= self.size[v];
                    self.size[v] += self.size[*u];

                    p = *u;
                    *u = v;
                    continue 'outer;
                }
            }
            return;
        }
    }

    fn decompose_dnc(&mut self, mut u: usize, p: usize) {
        self.reroot_to_centroid(&mut u);

        self.parent[u] = p as u32;
        self.level[u] = if p != UNSET as usize {
            self.level[p] + 1
        } else {
            0
        };

        let l = self.level[u] as usize;
        self.expand_level(l);
        self.depth_in_layer[l][u] = 0;
        if u < self.n_cutoff {
            self.depth_prefix[u][0] += 1;
        }

        for iv in 0..self.neighbors.get(u).len() {
            let (v, ()) = self.neighbors.get(u)[iv];
            if self.level[v as usize] != UNSET {
                continue;
            }

            let color = self.n_branch;
            self.n_branch += 1;
            self.depth_prefix_in_branch.push(vec![]);

            let mut bfs = vec![(v as u32, u as u32)];
            let mut timer = 0;
            while let Some(&(b, a)) = bfs.get(timer) {
                timer += 1;

                self.branch_in_layer[l][b as usize] = color as u32;
                self.depth_in_layer[l][b as usize] = self.depth_in_layer[l][a as usize] + 1;
                let d = self.depth_in_layer[l][b as usize] as usize;

                if (b as usize) < self.n_cutoff {
                    expand_vec_with(&mut self.depth_prefix[u], d + 1, || 0);
                    expand_vec_with(&mut self.depth_prefix_in_branch[color], d + 1, || 0);
                    self.depth_prefix[u][d] += 1;
                    self.depth_prefix_in_branch[color][d] += 1;
                }

                for &(c, ()) in self.neighbors.get(b as usize) {
                    if c == a || self.level[c as usize] != UNSET {
                        continue;
                    }
                    bfs.push((c, b));
                }
            }
            accumulate(&mut self.depth_prefix_in_branch[color]);

            self.decompose_dnc(v as usize, u);
        }
        accumulate(&mut self.depth_prefix[u]);
    }

    fn count_in_disc(&self, (u, r): (u32, u32)) -> u32 {
        let (u, r) = (u as usize, r as usize);
        let mut l = self.level[u] as usize;
        let mut res = 0;
        let mut c = u;
        loop {
            let d = self.depth_in_layer[l][u] as usize;
            let color = self.branch_in_layer[l][u] as usize;

            if d <= r {
                let d_twin = r - d;
                res += get_or_last(&self.depth_prefix[c], d_twin);
                if color != UNSET as usize {
                    res -= get_or_last(&self.depth_prefix_in_branch[color], d_twin);
                }
            }

            if l == 0 as usize {
                return res;
            }
            l -= 1;
            c = self.parent[c] as usize;
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

struct DiscOp {
    n_cutoff: usize,

    hld: hld::HLD,
    level: Vec<u32>,
    sid_inv: Vec<u32>,
}

impl DiscOp {
    fn n_verts(&self) -> usize {
        self.level.len()
    }

    fn from_edges(
        n_verts: usize,
        edges: impl IntoIterator<Item = (u32, u32)>,
        n_cutoff: usize,
    ) -> Self {
        let hld = hld::HLD::from_edges(n_verts, edges, 0, true);

        let mut sid_inv = vec![UNSET; n_verts];
        for u in 0..n_verts {
            sid_inv[hld.segmented_idx[u] as usize] = u as u32;
        }

        let mut depth = vec![0u32; n_verts];
        for &u in hld.topological_order.iter().rev().skip(1) {
            let p = hld.parent[u as usize];
            depth[u as usize] = depth[p as usize] + 1;
        }

        Self {
            n_cutoff,
            hld,
            sid_inv,
            level: depth,
        }
    }

    fn nth_parent(&self, mut u: usize, mut n: u32) -> Option<usize> {
        loop {
            let top = self.hld.chain_top[u as usize] as usize;
            let d = self.level[u] - self.level[top];
            if n <= d {
                return Some(self.sid_inv[(self.hld.segmented_idx[u] - n) as usize] as usize);
            }
            u = self.hld.parent[top] as usize;
            n -= d + 1;

            if u == hld::UNSET as usize {
                return None;
            }
        }
    }

    fn id(&self) -> (u32, u32) {
        (0, 1 << 29)
    }

    fn inter_nonnull(&self, (u, ru): (u32, u32), (v, rv): (u32, u32)) -> Option<(u32, u32)> {
        let (u, v) = (u as usize, v as usize);
        let j = self.hld.lca(u, v);

        let d_uj = self.level[u] - self.level[j];
        let d_vj = self.level[v] - self.level[j];
        let d = d_uj + d_vj;

        if d + rv <= ru {
            return Some((v as u32, rv));
        } else if d + ru <= rv {
            return Some((u as u32, ru));
        } else if ru + rv < d {
            return None;
        }

        let diam = ru + rv - d;

        let pu = u < self.n_cutoff;
        let pv = v < self.n_cutoff;

        let nth_in_path = |k: u32| -> usize {
            debug_assert!(k <= d);
            if k <= d_uj {
                unsafe { self.nth_parent(u, k).unwrap_unchecked() }
            } else {
                unsafe { self.nth_parent(v, d - k).unwrap_unchecked() }
            }
        };

        let center = if diam % 2 == 0 {
            nth_in_path(ru - diam / 2)
        } else {
            let pl = pu ^ ((d - rv) % 2 == 1);
            let pr = pv ^ ((d - ru) % 2 == 1);
            debug_assert!(pl != pr);

            if pl {
                nth_in_path(ru - diam / 2 - 1)
            } else {
                nth_in_path(ru - diam / 2)
            }
        } as u32;
        Some((center, diam / 2))
    }

    fn inter(&self, d1: Option<(u32, u32)>, d2: Option<(u32, u32)>) -> Option<(u32, u32)> {
        self.inter_nonnull(d1?, d2?)
    }

    fn dist(&self, u: usize, v: usize) -> u32 {
        let j = self.hld.lca(u, v);
        self.level[u] + self.level[v] - self.level[j] * 2
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let n_verts = 2 * n - 1;
    let mut edges = vec![];
    for i in 0..n - 1 {
        let e = (i + n) as u32;
        let u = input.u32() - 1;
        let v = input.u32() - 1;
        edges.push((u, (e, ())));
        edges.push((e, (u, ())));
        edges.push((v, (e, ())));
        edges.push((e, (v, ())));
    }

    let neighbors = jagged::CSR::from_assoc_list(n_verts, &edges);
    let ct = CentroidTree::new(&neighbors, n);

    let edges_undirected = edges
        .into_iter()
        .filter_map(|(u, (v, ()))| (u < v).then(|| (u, v)));
    let cx = DiscOp::from_edges(n_verts, edges_undirected, n);

    let q: usize = input.value();
    for _ in 0..q {
        let k: usize = input.u32() as usize;
        let discs: Vec<_> = (0..k).map(|_| (input.u32() - 1, input.u32() * 2)).collect();

        let mut prefix = vec![Some(cx.id())];
        for i in 0..k {
            prefix.push(cx.inter(prefix[i], Some(discs[i])));
        }

        let mut ans = 0;
        let mut postfix = Some(cx.id());
        let mut exs = debug::Label::new_with(|| vec![]);
        for i in (0..k).rev() {
            let exclusive = cx.inter(prefix[i], postfix);
            postfix = cx.inter(postfix, Some(discs[i]));
            ans += exclusive.map(|d| ct.count_in_disc(d)).unwrap_or(0) as u64;
            exs.with(|ds| ds.push(exclusive));
        }
        let common = prefix[k];
        ans -= common.map(|d| ct.count_in_disc(d)).unwrap_or(0) as u64 * (k - 1) as u64;

        writeln!(output, "{}", ans).unwrap();
    }
}
