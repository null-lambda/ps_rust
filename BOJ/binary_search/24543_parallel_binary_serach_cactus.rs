use std::io::Write;

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

pub mod debug {
    pub fn with(#[allow(unused_variables)] f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
    }
}

pub mod jagged {
    use std::fmt::Debug;
    use std::iter;
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
        pub fn from_pairs(n: usize, pairs: &[(u32, T)]) -> Self {
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
        pub bct_children: Vec<Vec<u32>>,

        pub bcc_edges: Vec<(u32, u32, E)>,
    }

    impl<'a, E: 'a + Copy, J: jagged::Jagged<(u32, E)>> BlockCutForest<'a, E, J> {
        pub fn from_assoc_list(neighbors: &'a J) -> Self {
            let n = neighbors.len();

            let mut parent = vec![UNSET; n];
            let mut low = vec![0; n];
            let mut euler_in = vec![0; n];
            let mut timer = 1u32;

            let mut bct_parent = vec![UNSET; n];
            let mut bct_children = vec![vec![]; n];

            let mut current_edge = vec![0u32; n];
            let mut stack = vec![];
            for root in 0..n {
                if euler_in[root] != 0 {
                    continue;
                }

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
                    if (*iv as usize) == neighbors[u as usize].len() {
                        // On exit
                        if p == UNSET {
                            break;
                        }

                        low[p as usize] = low[p as usize].min(low[u as usize]);
                        if low[u as usize] >= euler_in[p as usize] {
                            // Found a BCC
                            let bcc_node = bct_parent.len() as u32;

                            bct_parent.push(p);
                            bct_children.push(vec![]);
                            bct_children[p as usize].push(bcc_node);

                            while let Some(c) = stack.pop() {
                                bct_parent[c as usize] = bcc_node;
                                bct_children[bcc_node as usize].push(c);

                                if c == u {
                                    break;
                                }
                            }
                        }

                        u = p;
                        continue;
                    }

                    let (v, _) = neighbors[u as usize][*iv as usize];
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
            }

            Self {
                neighbors,
                parent,
                low,
                euler_in,

                bct_parent,
                bct_children,

                bcc_edges: vec![],
            }
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let c: usize = input.value();
    let q: usize = input.value();
    let mut edges = vec![];
    for _ in 0..m {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        edges.push((u, (v, ())));
        edges.push((v, (u, ())));
    }
    let neighbors = jagged::CSR::from_pairs(n, &edges);

    let mut color_pos = vec![vec![]; c];
    for u in 0..n {
        let k = input.value::<usize>() - 1;
        color_pos[k].push(u);
    }
    let lower_bound: Vec<u64> = (0..c).map(|_| input.value()).collect();

    let bct = bcc::BlockCutForest::from_assoc_list(&neighbors);
    let parent = &bct.bct_parent;
    let children = &bct.bct_children;
    let n_nodes = parent.len();

    let mut events = vec![];
    for _ in 0..q {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        let k: u32 = input.value();

        let b = if parent[u] == parent[v] {
            parent[u]
        } else if parent[u] != bcc::UNSET && parent[parent[u] as usize] as usize == v {
            parent[u]
        } else {
            assert!(parent[v] != bcc::UNSET && parent[parent[v] as usize] as usize == u);
            parent[v]
        };
        events.push((b as u32, k));
    }

    let mut queries: Vec<_> = (0..c).map(|i| ([0, q as u32], i as u32)).collect();
    loop {
        queries.sort_unstable_by_key(|&([l, r], _)| l + r >> 1);
        let mut queries = queries.iter_mut().peekable();

        let mut finished = true;
        let mut dp = vec![0u64; n_nodes - n];
        for (t, &(b, k)) in events.iter().enumerate() {
            dp[b as usize - n] += k as u64;

            while let Some(([l, r], i)) = queries.next_if(|([l, r], _)| l + r >> 1 <= t as u32) {
                if *l == *r {
                    continue;
                }
                finished = false;

                let mid = *l + *r >> 1;

                let mut sum = 0u64;
                for &u in &color_pos[*i as usize] {
                    if parent[u as usize] != bcc::UNSET {
                        sum += dp[parent[u as usize] as usize - n];
                    }
                    for &v in &children[u as usize] {
                        sum += dp[v as usize - n];
                    }
                }

                if sum < lower_bound[*i as usize] {
                    *l = mid + 1;
                } else {
                    *r = mid;
                }
            }
        }

        if finished {
            break;
        }
    }

    let mut ans = vec![!0; c];
    queries.sort_unstable_by_key(|&([l, _], _)| l);
    for (s, ([l, _], i)) in queries.into_iter().enumerate() {
        if l as usize + c > q + s {
            writeln!(output, "-1").unwrap();
            return;
        }
        ans[i as usize] = (q - c + s) as u32;
    }

    writeln!(output, "1").unwrap();
    for a in ans {
        write!(output, "{} ", a + 1).unwrap();
    }
}
