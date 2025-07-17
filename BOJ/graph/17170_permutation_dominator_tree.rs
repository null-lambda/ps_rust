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

fn lis_len<T: Clone + Ord>(xs: impl IntoIterator<Item = T>) -> Vec<u32> {
    let mut dp = vec![];
    let mut res = vec![];
    for x in xs {
        if dp.last().is_none() || dp.last().unwrap() < &x {
            dp.push(x.clone());
            res.push(dp.len() as u32);
        } else {
            let idx = dp.binary_search(&x).unwrap_or_else(|x| x);
            dp[idx] = x.clone();
            res.push((idx + 1) as u32);
        }
    }

    res
}

fn n_doms(n: usize, perm: impl Fn(u32) -> u32) -> Vec<u32> {
    let ls = lis_len((0..n as u32).map(|u| perm(u)));
    let mut inv_perm = vec![0u32; n];
    for u in 0..n as u32 {
        inv_perm[perm(u) as usize] = u;
    }

    let l_bound = ls.len() + 1;
    let layers = jagged::CSR::from_pairs(
        l_bound,
        (0..n as u32).rev().map(|u| (ls[u as usize], perm(u))),
    );
    let mut size: Vec<_> = vec![0u32; l_bound];

    let mut edges = vec![];
    for u in 0..n {
        let pu = perm(u as u32);
        let l = ls[u as usize];

        let prev = &layers[l as usize - 1][..size[l as usize - 1] as usize];
        let i = prev.partition_point(|&x| x > pu);
        if i < prev.len() {
            let pv1 = prev[i];
            edges.push([inv_perm[pv1 as usize], u as u32]);
            let pv0 = *prev.last().unwrap();
            if pv0 < pv1 {
                edges.push([inv_perm[pv0 as usize], u as u32]);
            }
        }

        size[l as usize] += 1;
    }

    let dt = dominators::DomTree::from_edges(l_bound, edges.into_iter(), 0);
    let mut n_doms = vec![0; ls.len()];
    for &u in &dt.dfs[1..] {
        let p = dt.idom[u as usize];
        n_doms[u as usize] = n_doms[p as usize] + 1;
    }

    n_doms
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let n_pad = n + 2;
    let xs: Vec<_> = std::iter::once(0)
        .chain((0..n).map(|_| input.u32()))
        .chain(std::iter::once(n as u32 + 1))
        .collect();

    let ls = &n_doms(n_pad, |u| xs[u as usize]);
    let rs = &n_doms(n_pad, |u| n_pad as u32 - 1 - xs[n_pad - 1 - u as usize]);
    let ans = ls
        .iter()
        .zip(rs.iter().rev())
        .map(|(&x, &y)| x + y - 2)
        .skip(1)
        .take(n);
    for a in ans {
        write!(output, "{} ", a).unwrap();
    }
}
