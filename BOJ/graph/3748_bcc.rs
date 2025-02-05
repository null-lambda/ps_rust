use std::{collections::HashSet, io::Write};

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

    pub struct BlockCutForest<'a, J> {
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
    }

    impl<'a, J: jagged::Jagged<'a, u32>> BlockCutForest<'a, J> {
        pub fn from_assoc_list(neighbors: &'a J) -> Self {
            let n = neighbors.len();

            let mut parent = vec![UNSET; n];
            let mut low = vec![0; n];
            let mut euler_in = vec![0; n];
            let mut timer = 1u32;

            let mut bct_parent = vec![UNSET; n];
            let mut bct_degree = vec![1u32; n];
            bct_parent.reserve_exact(n * 2);

            let mut current_edge = vec![0u32; n];
            let mut stack = vec![];
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
                            while let Some(c) = stack.pop() {
                                bct_parent[c as usize] = bcc_node;
                                bct_degree[bcc_node as usize] += 1;

                                if c == u {
                                    break;
                                }
                            }
                        }

                        u = p;
                        continue;
                    }

                    let v = neighbors.get(u as usize)[*iv as usize];
                    *iv += 1;
                    if v == p {
                        continue;
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
                }
            }

            Self {
                neighbors,
                parent,
                low,
                euler_in,

                bct_parent,
                bct_degree,
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

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    for _ in 0..input.value() {
        let n: usize = input.value();
        let m: usize = input.value();

        let mut edges = vec![];
        for _ in 0..m {
            let u = input.u32() - 1;
            let v = input.u32() - 1;
            edges.push((u, v));
            edges.push((v, u));
        }
        let neighbors = jagged::CSR::from_assoc_list(n, &edges);
        let bct = bcc::BlockCutForest::from_assoc_list(&neighbors);

        let mut colors = vec![HashSet::new(); n];
        let mut verts_in_bcc = vec![vec![]; bct.bcc_node_range().len()];
        for u in 0..n {
            let b = bct.bct_parent[u];
            if b != bcc::UNSET {
                colors[u].insert(b - n as u32);
                verts_in_bcc[b as usize - n].push(u as u32);
            }
        }
        for b in bct.bcc_node_range() {
            let p = bct.bct_parent[b];
            colors[p as usize].insert((b - n) as u32);
            verts_in_bcc[(b - n) as usize].push(p);
        }

        let mut edges_in_bcc = vec![vec![]; bct.bcc_node_range().len()];
        for (mut u, mut v) in edges.iter().copied() {
            if u > v {
                continue;
            }
            if colors[u as usize].len() < colors[v as usize].len() {
                std::mem::swap(&mut u, &mut v);
            }
            if let Some(&b) = colors[v as usize].intersection(&colors[u as usize]).next() {
                edges_in_bcc[b as usize].push((u, v));
            }
        }

        let mut neighbors_in_bcc = vec![vec![]; n];
        let mut parity = vec![None; n];
        let mut valid = vec![false; n];
        for b in bct.bcc_node_range() {
            if edges_in_bcc[b - n].is_empty() {
                continue;
            }

            for &(u, v) in &edges_in_bcc[b - n] {
                neighbors_in_bcc[u as usize].push(v);
                neighbors_in_bcc[v as usize].push(u);
            }

            let init = edges_in_bcc[b - n][0].0;
            parity[init as usize] = Some(true);
            let mut stack = vec![(init, 0u32, init)];
            while let Some((u, iv, p)) = stack.pop() {
                if iv < neighbors_in_bcc[u as usize].len() as u32 {
                    stack.push((u, iv + 1, p));
                    let v = neighbors_in_bcc[u as usize][iv as usize];
                    if v == p {
                        continue;
                    }
                    if parity[v as usize].is_none() {
                        parity[v as usize] = Some(!parity[u as usize].unwrap());
                        stack.push((v, 0, u));
                    } else if parity[v as usize] == parity[u as usize] {
                        for &u in &verts_in_bcc[b - n] {
                            valid[u as usize] = true;
                        }
                        break;
                    }
                }
            }

            for &u in &verts_in_bcc[b - n] {
                neighbors_in_bcc[u as usize].clear();
                parity[u as usize] = None;
            }
        }

        let ans = valid.iter().filter(|&&b| b).count();
        writeln!(output, "{}", ans).unwrap();
    }
}
