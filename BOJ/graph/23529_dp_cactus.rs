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

#[derive(Clone, Debug)]
struct VertAgg {
    count: u32,
    sum: u64,
}

impl VertAgg {
    fn isolated() -> Self {
        Self { count: 1, sum: 0 }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut edges = vec![];
    for _ in 0..m {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        edges.push((u, (v, ())));
        edges.push((v, (u, ())));
    }

    let neighbors = jagged::CSR::from_pairs(n, &edges);
    let bct = bcc::BlockCutForest::from_assoc_list(&neighbors);

    let parent = &bct.bct_parent;
    let children = &bct.bct_children;
    let n_nodes = parent.len();

    let mut degree = vec![1u32; n_nodes];
    for &p in &parent[1..] {
        degree[p as usize] += 1;
    }
    degree[0] += 2;

    let mut ans = 0u64;
    let mut dp = vec![VertAgg::isolated(); n];
    for mut u in 0..n_nodes {
        while degree[u] == 1 {
            let p = parent[u] as usize;
            degree[u] -= 1;
            degree[p] -= 1;

            if u >= n {
                let vs = &children[u];
                let m = vs.len();

                let mut sum_to_p = 0;
                let mut sum = 0;
                let mut count = 0;
                let mut dot = 0;
                for (i, &v) in vs.iter().enumerate() {
                    let v = v as usize;
                    let d_cyclic = (i + 1).min(m - i) as u64;

                    sum_to_p += dp[v].sum + dp[v].count as u64 * d_cyclic;
                    sum += dp[v].sum;
                    count += dp[v].count;
                    dot += dp[v].sum * dp[v].count as u64;
                }
                ans += sum_to_p * dp[p].count as u64;
                ans += count as u64 * dp[p].sum;
                dp[p].sum += sum_to_p;
                dp[p].count += count;

                if m >= 2 {
                    ans += sum * count as u64 - dot;

                    let mut c0 = vec![0; m + 1];
                    let mut c1 = vec![0; m + 1];
                    for i in 0..m {
                        c0[i + 1] = c0[i] + dp[vs[i] as usize].count;
                        c1[i + 1] = c1[i] + dp[vs[i] as usize].count as u64 * i as u64;
                    }

                    for j in 0..m {
                        let mut weighted_prefix = 0;
                        let i_split = j.saturating_sub((m + 1) / 2);

                        weighted_prefix += c0[i_split] as u64 * (m + 1 - j) as u64;
                        weighted_prefix += c1[i_split];

                        weighted_prefix += (c0[j] - c0[i_split]) as u64 * j as u64;
                        weighted_prefix -= c1[j] - c1[i_split];

                        ans += dp[vs[j] as usize].count as u64 * weighted_prefix;
                    }
                }
            }

            u = p;
        }
    }

    writeln!(output, "{}", ans).unwrap();
}
