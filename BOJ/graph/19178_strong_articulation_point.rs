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

pub mod jagged {
    use std::fmt::Debug;
    use std::mem::MaybeUninit;
    use std::ops::{Index, IndexMut};

    // Trait for painless switch between different representations of a jagged array
    pub trait Jagged<T>: IndexMut<usize, Output = [T]> {
        fn len(&self) -> usize;
    }

    impl<T, C> Jagged<T> for C
    where
        C: AsRef<[Vec<T>]> + IndexMut<usize, Output = [T]>,
    {
        fn len(&self) -> usize {
            <Self as AsRef<[Vec<T>]>>::as_ref(self).len()
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
            let v: Vec<Vec<&T>> = (0..self.len()).map(|i| self[i].iter().collect()).collect();
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
        pub fn from_pairs<I>(n: usize, pairs: I) -> Self
        where
            I: IntoIterator<Item = (u32, T)>,
            I::IntoIter: Clone,
        {
            let mut head = vec![0u32; n + 1];

            let pairs = pairs.into_iter();
            for (u, _) in pairs.clone() {
                debug_assert!(u < n as u32);
                head[u as usize] += 1;
            }
            for i in 0..n {
                head[i + 1] += head[i];
            }
            let mut data: Vec<_> = (0..head[n]).map(|_| MaybeUninit::uninit()).collect();

            for (u, v) in pairs {
                head[u as usize] -= 1;
                data[head[u as usize] as usize] = MaybeUninit::new(v.clone());
            }

            // Rustc is likely to perform in‑place iteration without new allocation.
            // [https://doc.rust-lang.org/stable/std/iter/trait.FromIterator.html#impl-FromIterator%3CT%3E-for-Vec%3CT%3E]
            let data = data
                .into_iter()
                .map(|x| unsafe { x.assume_init() })
                .collect();

            CSR { data, head }
        }
    }

    impl<T> Index<usize> for CSR<T> {
        type Output = [T];

        fn index(&self, index: usize) -> &Self::Output {
            &self.data[self.head[index] as usize..self.head[index + 1] as usize]
        }
    }

    impl<T> IndexMut<usize> for CSR<T> {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            &mut self.data[self.head[index] as usize..self.head[index + 1] as usize]
        }
    }

    impl<T> Jagged<T> for CSR<T> {
        fn len(&self) -> usize {
            self.head.len() - 1
        }
    }
}

pub mod dominators {
    // Langauer-Tarjan algorithm for computing the dominator tree
    use crate::jagged;

    pub const UNSET: u32 = !0;

    // A Union-Find structure for finding min(sdom(-)) on the DFS tree path
    struct DisjointSet {
        parent: Vec<u32>,
        label: Vec<u32>,
    }

    impl DisjointSet {
        fn new(n: usize) -> Self {
            DisjointSet {
                parent: vec![UNSET; n],
                label: (0..n as u32).collect(),
            }
        }
    }

    impl DisjointSet {
        fn link(&mut self, p: u32, u: u32) {
            self.parent[u as usize] = p;
        }

        fn eval(&mut self, v: u32, key: impl Fn(u32) -> u32) -> u32 {
            if self.parent[v as usize] == UNSET {
                return v;
            }
            self.compress(v, &key);
            self.label[v as usize]
        }

        fn compress(&mut self, v: u32, key: &impl Fn(u32) -> u32) {
            let a = self.parent[v as usize];
            debug_assert!(a != UNSET);
            if self.parent[a as usize] == UNSET {
                return;
            }

            self.compress(a, key);
            if key(self.label[a as usize]) < key(self.label[v as usize]) {
                self.label[v as usize] = self.label[a as usize];
            }
            self.parent[v as usize] = self.parent[a as usize];
        }
    }

    fn gen_dfs(
        children: &impl jagged::Jagged<u32>,
        root: u32,
        dfs: &mut Vec<u32>,
        t_in: &mut [u32],
        dfs_parent: &mut [u32],
    ) {
        let n = children.len();

        // Stackless DFS
        let mut current_edge: Vec<_> = (0..n).map(|u| children[u].len() as u32).collect();
        let mut u = root;

        dfs_parent[u as usize] = u;
        dfs.push(u);
        t_in[u as usize] = 0;

        loop {
            let p = dfs_parent[u as usize];
            let iv = &mut current_edge[u as usize];

            if *iv == 0 {
                if p == u {
                    break;
                }
                u = p;
                continue;
            }

            *iv -= 1;
            let v = children[u as usize][*iv as usize];
            if v == p || dfs_parent[v as usize] != UNSET {
                continue;
            }

            t_in[v as usize] = dfs.len() as u32;
            dfs.push(v);
            dfs_parent[v as usize] = u as u32;
            u = v;
        }
    }

    pub struct DomTree {
        // Rooted DAG structure
        pub children: jagged::CSR<u32>,
        pub parents: jagged::CSR<u32>,

        // DFS tree
        pub dfs: Vec<u32>,
        // t_in: Vec<u32>,
        pub dfs_parent: Vec<u32>,

        // Dominator tree
        pub sdom: Vec<u32>,
        pub idom: Vec<u32>,
    }

    impl DomTree {
        pub fn from_edges(
            n: usize,
            edges: impl Iterator<Item = [u32; 2]> + Clone,
            root: usize,
        ) -> DomTree {
            let edges = || edges.clone().map(|[u, v]| (u as u32, v as u32));

            let children = jagged::CSR::from_pairs(n, edges());
            let parents = jagged::CSR::from_pairs(n, edges().map(|(u, v)| (v, u)));

            let mut dfs_parent = vec![UNSET; n];
            let mut dfs = Vec::with_capacity(n);
            let mut t_in = vec![UNSET; n];
            gen_dfs(&children, root as u32, &mut dfs, &mut t_in, &mut dfs_parent);
            // assert_eq!(dfs.len(), n, "Some nodes are unreachable from the root");

            // Intermediate states
            let mut sdom = t_in;
            let mut idom = vec![UNSET; n];
            let mut bucket = vec![UNSET; n]; // Forward-star, compressed
            let mut dset = DisjointSet::new(n);

            for &w in dfs[1..].iter().rev() {
                for &v in &parents[w as usize] {
                    let u = dset.eval(v, |x| sdom[x as usize]);
                    sdom[w as usize] = sdom[w as usize].min(sdom[u as usize]);
                }

                let p = dfs_parent[w as usize];
                dset.link(p, w);

                let b = dfs[sdom[w as usize] as usize];
                bucket[w as usize] = bucket[b as usize];
                bucket[b as usize] = w;

                let mut v = std::mem::replace(&mut bucket[p as usize], UNSET);
                while v != UNSET {
                    let u = dset.eval(v, |x| sdom[x as usize]);
                    idom[v as usize] = if sdom[u as usize] < sdom[v as usize] {
                        u
                    } else {
                        p
                    };

                    v = bucket[v as usize];
                }
            }

            for &u in &dfs[1..] {
                if idom[u as usize] != dfs[sdom[u as usize] as usize] {
                    idom[u as usize] = idom[idom[u as usize] as usize];
                }
            }
            idom[root] = UNSET;

            DomTree {
                children,
                parents,

                dfs,
                // t_in,
                dfs_parent,

                idom,
                sdom,
            }
        }
    }
}

fn gen_scc(neighbors: &impl jagged::Jagged<u32>) -> (usize, Vec<u32>) {
    // Tarjan algorithm, iterative
    let n = neighbors.len();

    const UNSET: u32 = u32::MAX;
    let mut scc_index: Vec<u32> = vec![UNSET; n];
    let mut scc_count = 0;

    let mut path_stack = vec![];
    let mut dfs_stack = vec![];
    let mut order_count: u32 = 1;
    let mut order: Vec<u32> = vec![0; n];
    let mut low_link: Vec<u32> = vec![UNSET; n];

    for u in 0..n {
        if order[u] > 0 {
            continue;
        }

        const UPDATE_LOW_LINK: u32 = 1 << 31;

        dfs_stack.push((u as u32, 0));
        while let Some((u, iv)) = dfs_stack.pop() {
            if iv & UPDATE_LOW_LINK != 0 {
                let v = iv ^ UPDATE_LOW_LINK;
                low_link[u as usize] = low_link[u as usize].min(low_link[v as usize]);
                continue;
            }

            if iv == 0 {
                // Enter node
                order[u as usize] = order_count;
                low_link[u as usize] = order_count;
                order_count += 1;
                path_stack.push(u);
            }

            if iv < neighbors[u as usize].len() as u32 {
                // Iterate neighbors
                dfs_stack.push((u, iv + 1));

                let v = neighbors[u as usize][iv as usize];
                if order[v as usize] == 0 {
                    dfs_stack.push((u, v | UPDATE_LOW_LINK));
                    dfs_stack.push((v, 0));
                } else if scc_index[v as usize] == UNSET {
                    low_link[u as usize] = low_link[u as usize].min(order[v as usize]);
                }
            } else {
                // Exit node
                if low_link[u as usize] == order[u as usize] {
                    // Found a strongly connected component
                    loop {
                        let v = path_stack.pop().unwrap();
                        scc_index[v as usize] = scc_count;
                        if v == u {
                            break;
                        }
                    }
                    scc_count += 1;
                }
            }
        }
    }
    (scc_count as usize, scc_index)
}

fn yesno(b: bool) -> &'static str {
    if b {
        "YES"
    } else {
        "NO"
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let edges: Vec<[u32; 2]> = (0..m)
        .map(|_| std::array::from_fn(|_| input.u32() - 1))
        .collect();

    let children = jagged::CSR::from_pairs(n, edges.iter().map(|&[u, v]| (u, v)));
    let (n_scc, color) = gen_scc(&children);
    let sccs = jagged::CSR::from_pairs(n_scc, (0..n as u32).map(|u| (color[u as usize], u)));

    let mut index_map = vec![!0u32; n];
    for c in 0..n_scc {
        for (i, &u) in sccs[c].iter().enumerate() {
            index_map[u as usize] = i as u32;
        }
    }

    let scc_edges = jagged::CSR::from_pairs(
        n_scc,
        edges
            .iter()
            .copied()
            .filter(|&[u, v]| color[u as usize] == color[v as usize])
            .map(|[u, v]| {
                (
                    color[u as usize],
                    [index_map[u as usize], index_map[v as usize]],
                )
            }),
    );

    // Strong articulation point
    let mut sap = vec![false; n];
    for c in 0..sccs.len() {
        if sccs[c].len() <= 1 {
            continue;
        }

        let s = sccs[c].len() - 1;
        let dt = dominators::DomTree::from_edges(n, scc_edges[c].iter().copied(), s);
        let dt_rev =
            dominators::DomTree::from_edges(n, scc_edges[c].iter().map(|&[u, v]| [v, u]), s);

        for u in 0..sccs[c].len() - 1 {
            for p in [dt.idom[u], dt_rev.idom[u]] {
                sap[sccs[c][p as usize] as usize] = true;
            }
        }

        let sub_edges = jagged::CSR::from_pairs(
            sccs[c].len() - 1,
            scc_edges[c]
                .iter()
                .filter(|&&[u, v]| u != s as u32 && v != s as u32)
                .map(|&[u, v]| (u, v)),
        );
        let (n_sub_scc, _) = gen_scc(&sub_edges);

        sap[sccs[c][s] as usize] = n_sub_scc >= 2;
    }

    for a in sap {
        writeln!(output, "{}", yesno(a)).unwrap();
    }
}
