use std::io::Write;

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
        pub bct_children: Vec<Vec<u32>>,
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
            let mut bct_children = vec![vec![]; n];
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
                            bct_children.push(vec![]);
                            bct_degree.push(1);
                            while let Some(c) = stack.pop() {
                                bct_parent[c as usize] = bcc_node;
                                bct_children.last_mut().unwrap().push(c);
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
                    bct_children.push(vec![]);
                    bct_degree.push(1);
                }
            }

            Self {
                neighbors,
                parent,
                low,
                euler_in,

                bct_parent,
                bct_children,
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
    let mut bct = bcc::BlockCutForest::from_assoc_list(&neighbors);

    let n_bct_nodes = bct.bcc_node_range().end;
    let mut degree = std::mem::take(&mut bct.bct_degree);
    for root in 0..n {
        if bct.bct_parent[root] == bcc::UNSET {
            degree[root] += 2;
        }
    }

    let mut dp_cut = vec![[0u32, 1u32]; n];
    let mut dp_cycle = vec![vec![]; n_bct_nodes];
    let mut topological_order = vec![];
    for mut u in 0..n_bct_nodes {
        while degree[u] == 1 {
            let p = bct.bct_parent[u] as usize;
            degree[u] -= 1;
            degree[p] -= 1;

            if u >= n {
                let dp_cycle_u = &mut dp_cycle[u];
                *dp_cycle_u = vec![[0; 4]; bct.bct_children[u].len()];
                if let [c0, cs @ ..] = &bct.bct_children[u][..] {
                    topological_order.push(u);
                    dp_cycle_u[0][0b00] = dp_cut[*c0 as usize][0];
                    dp_cycle_u[0][0b11] = dp_cut[*c0 as usize][1];
                    for (i, c) in (1..).zip(cs) {
                        let c = *c as usize;
                        let prev = dp_cycle_u[i - 1];
                        let next = &mut dp_cycle_u[i];
                        next[0b00] = prev[0b00].max(prev[0b01]) + dp_cut[c][0];
                        next[0b01] = prev[0b00] + dp_cut[c][1];
                        next[0b10] = prev[0b10].max(prev[0b11]) + dp_cut[c][0];
                        next[0b11] = prev[0b10] + dp_cut[c][1];
                    }

                    dp_cut[p][0] += dp_cycle_u.last().unwrap().iter().max().unwrap();
                    dp_cut[p][1] += dp_cycle_u.last().unwrap()[0b00];
                }
            }

            u = p;
        }
    }

    let mut selected = vec![false; n];
    let mut max_indep = 0;
    for root in 0..n {
        if bct.bct_parent[root] == bcc::UNSET {
            max_indep += dp_cut[root][0].max(dp_cut[root][1]);
            selected[root] = dp_cut[root][0] < dp_cut[root][1];
        }
    }
    for u in topological_order.into_iter().rev() {
        let [c0, cs @ ..] = &bct.bct_children[u][..] else {
            unreachable!();
        };

        let p = bct.bct_parent[u] as usize;
        let mut tag_cycle = if selected[p] {
            0b00
        } else {
            (0b00..=0b11)
                .max_by_key(|&tag| dp_cycle[u].last().unwrap()[tag])
                .unwrap()
        };

        for (i, c) in (1..bct.bct_children[u].len()).zip(cs).rev() {
            selected[*c as usize] = tag_cycle & 1 == 1;

            let prev = dp_cycle[u][i - 1];
            tag_cycle = match tag_cycle {
                0b00 => {
                    if prev[0b00] < prev[0b01] {
                        0b01
                    } else {
                        0b00
                    }
                }
                0b10 => {
                    if prev[0b10] < prev[0b11] {
                        0b11
                    } else {
                        0b10
                    }
                }
                0b01 => 0b00,
                0b11 => 0b10,
                _ => unreachable!(),
            }
        }
        selected[*c0 as usize] = tag_cycle & 1 == 1;
    }

    writeln!(output, "{}", max_indep).unwrap();
    for u in 0..n {
        if selected[u] {
            write!(output, "{} ", u + 1).unwrap();
        }
    }
    writeln!(output).unwrap();
}
