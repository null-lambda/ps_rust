use std::{borrow::Borrow, collections::HashMap, io::Write};

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

pub mod bcc {
    /// Biconnected components & 2-edge-connected components
    /// Verified with [Yosupo library checker](https://judge.yosupo.jp/problem/biconnected_components)
    use super::jagged;

    pub const UNSET: u32 = !0;

    pub struct BlockCutForest<'a, E, J> {
        // DFS tree structure
        pub neighbors: &'a J,
        pub parent: Vec<u32>,
        pub euler_in: Vec<u32>,
        pub low: Vec<u32>, // Lowest euler index on a subtree's back edge

        /// Block-cut tree structure,  
        /// represented as a rooted bipartite tree between  
        /// vertex nodes (indices in 0..n) and virtual BCC nodes (indices in n..).  
        /// A vertex node is a cut vertex iff its degree is >= 2,
        /// and the neighbors of a virtual BCC node represents all its belonging vertices.
        pub bct_parent: Vec<u32>,
        pub bct_degree: Vec<u32>,
        pub bct_children: Vec<Vec<u32>>,

        /// BCC structure
        pub bcc_edges: Vec<Vec<(u32, u32, E)>>,
    }

    impl<'a, E: 'a + Copy, J: jagged::Jagged<'a, (u32, E)>> BlockCutForest<'a, E, J> {
        pub fn from_assoc_list(neighbors: &'a J) -> Self {
            let n = neighbors.len();

            let mut parent = vec![UNSET; n];
            let mut low = vec![0; n];
            let mut euler_in = vec![0; n];
            let mut timer = 1u32;

            let mut bct_parent = vec![UNSET; n];
            let mut bct_degree = vec![1u32; n];
            let mut bct_children = vec![vec![]; n];

            let mut bcc_edges = vec![];

            bct_parent.reserve_exact(n * 2);

            let mut current_edge = vec![0u32; n];
            let mut stack = vec![];
            let mut edges_stack: Vec<(u32, u32, E)> = vec![];
            for root in 0..n {
                if euler_in[root] != 0 {
                    continue;
                }

                bct_degree[root] -= 1;
                parent[root] = UNSET;
                let mut u = root as u32;
                loop {
                    let p = parent[u as usize];
                    let iv = &mut current_edge[u as usize];
                    if *iv == 0 {
                        // On enter
                        euler_in[u as usize] = timer;
                        low[u as usize] = timer + 1;
                        timer += 1;
                        stack.push(u);
                    }
                    if (*iv as usize) == neighbors.get(u as usize).len() {
                        // On exit
                        if p == UNSET {
                            break;
                        }

                        low[p as usize] = low[p as usize].min(low[u as usize]);
                        if low[u as usize] >= euler_in[p as usize] {
                            // Found a BCC
                            let bcc_node = bct_parent.len() as u32;
                            bct_degree[p as usize] += 1;

                            bct_parent.push(p);
                            bct_degree.push(1);
                            bct_children[p as usize].push(bcc_node);
                            bct_children.push(vec![]);

                            while let Some(c) = stack.pop() {
                                bct_parent[c as usize] = bcc_node;
                                bct_degree[bcc_node as usize] += 1;
                                bct_children[bcc_node as usize].push(c);

                                if c == u {
                                    break;
                                }
                            }

                            let mut es = vec![];
                            while let Some(e) = edges_stack.pop() {
                                es.push(e);
                                if (e.0, e.1) == (p, u) {
                                    break;
                                }
                            }
                            bcc_edges.push(es);
                        }

                        u = p;
                        continue;
                    }

                    let (v, w) = neighbors.get(u as usize)[*iv as usize];
                    *iv += 1;
                    if v == p {
                        continue;
                    }

                    if euler_in[v as usize] < euler_in[u as usize] {
                        // Unvisited edge
                        edges_stack.push((u, v, w));
                    }
                    if euler_in[v as usize] != 0 {
                        // Back edge
                        low[u as usize] = low[u as usize].min(euler_in[v as usize]);
                        continue;
                    }

                    // Forward edge (a part of DFS spanning tree)
                    parent[v as usize] = u;
                    u = v;
                }

                // For an isolated vertex, manually add a virtual BCC node.
                if neighbors.get(root).is_empty() {
                    bct_degree[root] += 1;

                    bct_parent.push(root as u32);
                    bct_degree.push(1);
                    bct_children.push(vec![]);
                    bct_children[root].push(bct_parent.len() as u32 - 1);

                    bcc_edges.push(vec![]);
                }
            }

            Self {
                neighbors,
                parent,
                low,
                euler_in,

                bct_parent,
                bct_degree,
                bct_children,

                bcc_edges,
            }
        }

        pub fn is_cut_vert(&self, u: usize) -> bool {
            debug_assert!(u < self.neighbors.len());
            self.bct_degree[u] >= 2
        }

        pub fn is_bridge(&self, u: usize, v: usize) -> bool {
            debug_assert!(u < self.neighbors.len() && v < self.neighbors.len() && u != v);
            self.euler_in[v] < self.low[u] || self.euler_in[u] < self.low[v]
        }

        pub fn bcc_node_range(&self) -> std::ops::Range<usize> {
            self.neighbors.len()..self.bct_parent.len()
        }

        pub fn get_bccs(&self) -> Vec<Vec<u32>> {
            let mut bccs = vec![vec![]; self.bcc_node_range().len()];
            let n = self.neighbors.len();
            for u in 0..n {
                let b = self.bct_parent[u];
                if b != UNSET {
                    bccs[b as usize - n].push(u as u32);
                }
            }
            for b in self.bcc_node_range() {
                bccs[b - n].push(self.bct_parent[b]);
            }
            bccs
        }

        pub fn get_2ccs(&self) -> Vec<Vec<u32>> {
            unimplemented!()
        }
    }
}

pub mod tree_iso {

    // AHU algorithm for classifying (rooted) trees up to isomorphism
    // Time complexity: O(N log N)    (O(N) with radix sort))
    pub type Code<E> = Box<[(u32, E)]>;

    #[derive(Debug, Clone)]
    pub struct AHULabelCompression<E> {
        // Extracted classification codes
        pub forest_roots: (Vec<usize>, E),
        pub levels: Vec<Vec<Code<E>>>,

        // Auxiliary information for isomorphism construction
        pub ordered_children: Vec<Vec<(u32, E)>>,
        pub depth: Vec<u32>,
        pub code_in_parent: Vec<u32>,
        pub bfs_tour: Vec<u32>,
    }

    impl<'a, E: 'a + Clone + Ord + Default> AHULabelCompression<E> {
        pub fn from_neighbors(
            n_verts: usize,
            mut neighbors: Vec<Vec<(u32, E)>>,
            is_cyclic_node: impl Fn(&usize) -> bool,
            root: Option<usize>, // None for unrooted tree isomorphism
        ) -> Self {
            assert!(n_verts > 0);

            // Build the tree structure
            let base_root = root.unwrap_or(0) as usize;
            let mut degree = vec![0u32; n_verts];
            let mut xor_neighbors = vec![0u32; n_verts];
            for u in 0..n_verts as u32 {
                for &(v, _) in &neighbors[u as usize] {
                    debug_assert!(v < n_verts as u32);
                    degree[u as usize] += 1;
                    xor_neighbors[u as usize] ^= v;
                }
            }
            degree[base_root] += 2;

            // Upward propagation
            let mut size = vec![1u32; n_verts];
            let mut topological_order = vec![];
            for mut u in 0..n_verts as u32 {
                while degree[u as usize] == 1 {
                    let p = xor_neighbors[u as usize];
                    xor_neighbors[p as usize] ^= u;
                    degree[u as usize] -= 1;
                    degree[p as usize] -= 1;
                    topological_order.push(u);

                    size[p as usize] += size[u as usize];

                    u = p;
                }
            }
            assert!(
                topological_order.len() == n_verts - 1,
                "Invalid tree structure"
            );
            let parent = xor_neighbors;

            let mut forest_roots = if let Some(root) = root {
                (vec![root], E::default())
            } else {
                // For unrooted tree classification, assign some distinguished roots.

                // Reroot down to the lowest centroid
                let mut root = base_root;
                let threshold = (n_verts as u32 + 1) / 2;
                for u in topological_order.into_iter().rev() {
                    let p = parent[u as usize] as usize;
                    if p == root && size[u as usize] >= threshold {
                        size[p as usize] -= size[u as usize];
                        size[u as usize] += size[p as usize];
                        root = u as usize;
                    }
                }

                // Modifications (for Block-cut tree): Select only one root
                (vec![root], E::default())

                // // Check the direct parent for another centroid
                // let mut aux_root = None;
                // let p = parent[root];
                // if p != root as u32 && size[p as usize] >= threshold {
                //     aux_root = Some(p as usize);
                // }

                //                 if let Some(aux_root) = aux_root {
                //                     // Split the double-centroid tree into a two-component forest
                //                     let i = neighbors[root]
                //                         .iter()
                //                         .position(|&(v, _)| v as usize == aux_root)
                //                         .unwrap();
                //                     let (_, w) = neighbors[root].swap_remove(i);
                //                     neighbors[root].retain(|&(v, _)| v != aux_root as u32);
                //                     neighbors[aux_root].retain(|&(v, _)| v != root as u32);
                //                     size[root] -= size[aux_root];

                //                     (vec![root, aux_root], w)
                //                 } else {
                //                     (vec![root], E::default())
                //                 }
            };

            // Downward propagation
            let mut depth = vec![0u32; n_verts];
            let mut bfs_tour: Vec<_> = forest_roots.0.iter().map(|&u| u as u32).collect();
            let mut timer = 0;
            while let Some(&u) = bfs_tour.get(timer) {
                timer += 1;
                neighbors[u as usize]
                    .iter()
                    .position(|&(v, _)| size[v as usize] > size[u as usize])
                    .map(|ip| {
                        neighbors[u as usize].rotate_left(ip + 1);
                        neighbors[u as usize].pop();
                    });
                for &(v, _) in &neighbors[u as usize] {
                    depth[v as usize] = depth[u as usize] + 1;
                    bfs_tour.push(v);
                }
            }

            // Encode the forest starting from the deepest level
            let mut code_in_parent: Vec<u32> = vec![Default::default(); n_verts];
            let mut levels: Vec<_> = vec![];
            let (mut start, mut end) = (n_verts, n_verts);
            for d in (0..n_verts as u32).rev() {
                while start > 0 && depth[bfs_tour[start - 1] as usize] >= d {
                    start -= 1;
                }
                let level_nodes = &bfs_tour[start..end];
                end = start;

                let mut codes_d: Vec<(u32, Code<E>)> = level_nodes
                    .iter()
                    .map(|&u| {
                        if is_cyclic_node(&(u as usize)) {
                            if (neighbors[u as usize]
                                .iter()
                                .rev()
                                .map(|(v, w)| (code_in_parent[*v as usize], w.clone())))
                            .lt(neighbors[u as usize]
                                .iter()
                                .map(|(v, w)| (code_in_parent[*v as usize], w.clone())))
                            {
                                neighbors[u as usize].reverse();
                            }
                        } else {
                            neighbors[u as usize].sort_unstable_by_key(|(v, w)| {
                                (code_in_parent[*v as usize], w.clone())
                            });
                        }
                        let code: Code<E> = neighbors[u as usize]
                            .iter()
                            .map(|(v, w)| (code_in_parent[*v as usize], w.clone()))
                            .collect();
                        (u, code)
                    })
                    .collect();
                codes_d.sort_unstable_by(|(_, code_a), (_, code_b)| code_a.cmp(code_b));

                let mut c = 0u32;
                let mut prev_code = None;
                for (u, code) in &codes_d {
                    if prev_code.is_some() && prev_code != Some(code) {
                        c += 1;
                    }
                    code_in_parent[*u as usize] = c;
                    prev_code = Some(code);
                }
                levels.push(codes_d.into_iter().map(|(_, code)| code).collect());
            }
            let ordered_children = neighbors;
            forest_roots.0.sort_unstable_by_key(|&u| code_in_parent[u]);

            Self {
                forest_roots,
                levels,

                ordered_children,
                depth,
                code_in_parent,
                bfs_tour,
            }
        }

        pub fn is_iso_to(&self, other: &Self) -> bool {
            self.forest_roots.0.len() == other.forest_roots.0.len()
                && self.forest_roots.1 == other.forest_roots.1
                && self.levels == other.levels
        }

        pub fn cmp_by_labels(&self, other: &Self) -> std::cmp::Ordering {
            (self.forest_roots.0.len().cmp(&other.forest_roots.0.len()))
                .then_with(|| self.forest_roots.1.cmp(&other.forest_roots.1))
                .then_with(|| self.levels.cmp(&other.levels))
        }

        pub fn get_mapping(&self, other: &Self) -> Option<Vec<u32>> {
            if !self.is_iso_to(other) {
                return None;
            }

            const UNSET: u32 = !0;
            let mut mapping = vec![UNSET; self.ordered_children.len()];

            let mut queue = vec![];
            for (&u1, &u2) in self.forest_roots.0.iter().zip(&other.forest_roots.0) {
                mapping[u1] = u2 as u32;
                queue.push(u1 as u32);
            }

            let mut timer = 0;
            while let Some(&u1) = queue.get(timer) {
                timer += 1;
                let u2 = mapping[u1 as usize];

                for (&(v1, _), &(v2, _)) in self.ordered_children[u1 as usize]
                    .iter()
                    .zip(&other.ordered_children[u2 as usize])
                {
                    mapping[v1 as usize] = v2;
                    queue.push(v1);
                }
            }
            debug_assert!(mapping.iter().all(|&v| v != UNSET));

            Some(mapping)
        }
    }
}

// Example
const P: u64 = 1_000_000_003;

// chunk_by in std >= 1.77
fn group_by<T, P, F>(xs: &[T], mut pred: P, mut f: F)
where
    P: FnMut(&T, &T) -> bool,
    F: FnMut(&[T]),
{
    let mut i = 0;
    while i < xs.len() {
        let mut j = i + 1;
        while j < xs.len() && pred(&xs[j - 1], &xs[j]) {
            j += 1;
        }
        f(&xs[i..j]);
        i = j;
    }
}

fn kmp<'a: 'c, 'b: 'c, 'c, T: PartialEq, R: Borrow<T>>(
    s: impl IntoIterator<Item = R> + 'a,
    pattern: &'b [T],
) -> impl Iterator<Item = usize> + 'c {
    // Build a jump table
    let mut jump_table = vec![0];
    let mut i_prev = 0;
    for i in 1..pattern.len() {
        while i_prev > 0 && pattern[i] != pattern[i_prev] {
            i_prev = jump_table[i_prev - 1];
        }
        if pattern[i] == pattern[i_prev] {
            i_prev += 1;
        }
        jump_table.push(i_prev);
    }

    // Search patterns
    let mut j = 0;
    s.into_iter().enumerate().filter_map(move |(i, c)| {
        while j == pattern.len() || j > 0 && &pattern[j] != c.borrow() {
            j = jump_table[j - 1];
        }
        if &pattern[j] == c.borrow() {
            j += 1;
        }
        (j == pattern.len()).then(|| i + 1 - pattern.len())
    })
}

fn count_cylic_eq<T: Eq + std::hash::Hash>(xs: &[T], ys: &[T]) -> u32 {
    assert!(xs.len() == ys.len());
    if xs.len() == 0 {
        return 1;
    }

    let xs_double = xs.iter().chain(&xs[..xs.len() - 1]);
    kmp(xs_double.map(|t| t), ys).count() as u32
}

fn linear_sieve(n_max: u32) -> (Vec<u32>, Vec<u32>) {
    let mut min_prime_factor = vec![0; n_max as usize + 1];
    let mut primes = Vec::new();

    for i in 2..=n_max {
        if min_prime_factor[i as usize] == 0 {
            primes.push(i);
        }
        for &p in primes.iter() {
            if i * p > n_max {
                break;
            }
            min_prime_factor[(i * p) as usize] = p;
            if i % p == 0 {
                break;
            }
        }
    }

    (min_prime_factor, primes)
}

fn on_dbg(f: impl FnOnce()) {
    #[cfg(debug_assertions)]
    f();
}

#[derive(Default)]
struct LazyProduct {
    factors: HashMap<u32, u32>,
    factorials: HashMap<u32, u32>,
}

impl LazyProduct {
    fn one() -> Self {
        Self {
            factors: HashMap::new(),
            factorials: HashMap::new(),
        }
    }

    fn mul_assign_scalar(&mut self, scalar: u32) {
        *self.factors.entry(scalar).or_default() += 1;
    }

    fn mul_assign_pow(&mut self, other: &Self, exp: u32) {
        if exp == 0 {
            return;
        }
        for (&f, &e) in &other.factors {
            *self.factors.entry(f).or_default() += e * exp as u32;
        }
        for (&f, &e) in &other.factorials {
            *self.factorials.entry(f).or_default() += e * exp;
        }
    }

    fn mul_assign_factorial(&mut self, f: u32) {
        *self.factorials.entry(f).or_default() += 1;
    }

    fn factorize(mut self) -> Vec<(u32, u32)> {
        let max_factor = self
            .factorials
            .keys()
            .chain(self.factors.keys())
            .copied()
            .max()
            .unwrap_or(0);
        let mut fs = vec![0; max_factor as usize + 1];
        for (&f, &e) in &self.factors {
            fs[f as usize] += e;
        }
        for f in (2..=max_factor).rev() {
            if let Some(e) = self.factorials.remove(&f) {
                fs[f as usize] += e;
                self.factorials
                    .entry(f - 1)
                    .and_modify(|e_sub| *e_sub += e)
                    .or_insert(e);
            }
        }

        let (min_prime_factor, _) = linear_sieve(max_factor);
        for f in (2..=max_factor).rev() {
            let p = min_prime_factor[f as usize];
            if p != 0 {
                fs[p as usize] += fs[f as usize];
                fs[(f / p) as usize] += fs[f as usize];
                fs[f as usize] = 0;
            }
        }

        (2..=max_factor)
            .filter(|&f| fs[f as usize] > 0)
            .map(|f| (f, fs[f as usize]))
            .collect()
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut edges = vec![];
    for _ in 0..m {
        let k: usize = input.value();
        let mut prev = input.u32() - 1;
        for _ in 1..k {
            let u = input.u32() - 1;
            edges.push((prev, (u, ())));
            edges.push((u, (prev, ())));
            prev = u;
        }
    }

    let neighbors = jagged::CSR::from_assoc_list(n, &edges);

    if n == 1 {
        writeln!(output, "0").unwrap();
        return;
    }

    let bct = bcc::BlockCutForest::from_assoc_list(&neighbors);
    let n_bct_nodes = bct.bct_parent.len();
    let is_bcc_node = |u| bct.bcc_node_range().contains(&u);
    let mut ordered_bct_neighbors = vec![vec![]; n_bct_nodes];
    for u in 0..n_bct_nodes {
        let p = bct.bct_parent[u];
        if p != bcc::UNSET {
            ordered_bct_neighbors[u].push((p as u32, ()));
        }
        for &c in &bct.bct_children[u] {
            ordered_bct_neighbors[u].push((c as u32, ()));
        }
    }

    let mut comp = tree_iso::AHULabelCompression::from_neighbors(
        n_bct_nodes,
        ordered_bct_neighbors.clone(),
        |&u| is_bcc_node(u),
        None,
    );
    let root = comp.forest_roots.0[0];

    on_dbg(|| {
        println!("bct parent: {:?}", bct.bct_parent);
        println!("forest roots {:?}", comp.forest_roots);
        println!("root: {}", root);
        println!("aut children {:?}", comp.ordered_children);
        println!("bfs tour: {:?}", comp.bfs_tour);
    });

    comp.bfs_tour.reverse();
    let mut dp = HashMap::<u32, LazyProduct>::new();

    group_by(
        &comp.bfs_tour,
        |&u1, &u2| comp.depth[u1 as usize] == comp.depth[u2 as usize],
        |row| {
            let dp_prev = std::mem::take(&mut dp);
            group_by(
                &row,
                |&u1, &u2| comp.code_in_parent[u1 as usize] == comp.code_in_parent[u2 as usize],
                |group| {
                    let u = group[0] as usize;
                    let cu = comp.code_in_parent[u];

                    let mut factor = LazyProduct::one();
                    let mut rle = vec![];
                    group_by(
                        &comp.ordered_children[u],
                        |&(v1, _), &(v2, _)| {
                            comp.code_in_parent[v1 as usize] == comp.code_in_parent[v2 as usize]
                        },
                        |group| {
                            let l = group.len() as u32;
                            let cv = comp.code_in_parent[group[0].0 as usize];
                            rle.push((l, cv));
                            factor.mul_assign_pow(&dp_prev[&cv], l);
                        },
                    );

                    if is_bcc_node(u) {
                        let seq_len = rle.iter().map(|(l, _)| l).sum::<u32>();
                        if u != root {
                            if seq_len >= 2 {
                                if rle.iter().eq(rle.iter().rev()) {
                                    factor.mul_assign_scalar(2);
                                }
                            }
                        } else {
                            let seq: Vec<_> = rle
                                .into_iter()
                                .flat_map(|(l, cv)| (0..l).map(move |_| cv))
                                .collect();

                            // Count the size of the stabilizer subgroup under dihedral permutations
                            // (rotation + flip)
                            let mut n_dihedral_perms = count_cylic_eq(&seq, &seq);
                            if seq_len >= 3 {
                                let mut seq_rev = seq.clone();
                                seq_rev.reverse();
                                n_dihedral_perms += count_cylic_eq(&seq, &seq_rev);
                            }
                            factor.mul_assign_scalar(n_dihedral_perms);
                        }
                    } else {
                        for (l, _) in rle {
                            factor.mul_assign_factorial(l);
                        }
                    }
                    dp.insert(cu, factor);
                },
            );
        },
    );

    let ans = dp.remove(&0).unwrap().factorize();
    writeln!(output, "{}", ans.len()).unwrap();
    for (f, e) in ans {
        writeln!(output, "{} {}", f, e).unwrap();
    }
}
