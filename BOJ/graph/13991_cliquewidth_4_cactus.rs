use std::io::Write;

use bcc::UNSET;

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

pub mod jagged {
    use std::fmt::Debug;
    use std::mem::MaybeUninit;
    use std::ops::{Index, IndexMut};

    // Compressed sparse row format, for static jagged array
    // Provides good locality for graph traversal
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CSR<T> {
        pub links: Vec<T>,
        head: Vec<u32>,
    }

    impl<T: Clone> CSR<T> {
        pub fn from_pairs(n: usize, pairs: impl Iterator<Item = (u32, T)> + Clone) -> Self {
            let mut head = vec![0u32; n + 1];

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

            CSR { links: data, head }
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
            CSR { links: data, head }
        }
    }

    impl<T> CSR<T> {
        pub fn len(&self) -> usize {
            self.head.len() - 1
        }

        pub fn edge_range(&self, index: usize) -> std::ops::Range<usize> {
            self.head[index] as usize..self.head[index as usize + 1] as usize
        }
    }

    impl<T> Index<usize> for CSR<T> {
        type Output = [T];

        fn index(&self, index: usize) -> &Self::Output {
            &self.links[self.edge_range(index)]
        }
    }

    impl<T> IndexMut<usize> for CSR<T> {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            let es = self.edge_range(index);
            &mut self.links[es]
        }
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
}

pub mod bcc {
    /// Biconnected components & 2-edge-connected components
    /// Verified with [Yosupo library checker](https://judge.yosupo.jp/problem/biconnected_components)
    use super::jagged;

    pub const UNSET: u32 = !0;

    pub struct BlockCutForest<'a, E> {
        // DFS tree structure
        pub neighbors: &'a jagged::CSR<(u32, E)>,
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

    impl<'a, E: 'a + Copy> BlockCutForest<'a, E> {
        pub fn from_assoc_list(neighbors: &'a jagged::CSR<(u32, E)>) -> Self {
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
                    if (*iv as usize) == neighbors[u as usize].len() {
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

                    let (v, w) = neighbors[u as usize][*iv as usize];
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
                if neighbors[root].is_empty() {
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

enum Cmd {
    Join(u32, u32),
    Recolor(u32, u8, u8),
    Connect(u32, u8, u8),
}
use Cmd::*;

impl std::fmt::Display for Cmd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Join(u, v) => write!(f, "j {} {}", u + 1, v + 1),
            Recolor(u, c, d) => write!(f, "r {} {} {}", u + 1, c + 1, d + 1),
            Connect(u, c, d) => write!(f, "c {} {} {}", u + 1, c + 1, d + 1),
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();

    let mut edges = vec![];
    for _ in 0..m {
        let k: usize = input.value();
        let mut p = input.value::<u32>() - 1;
        for _ in 1..k {
            let u = input.value::<u32>() - 1;
            edges.push((p, u));
            p = u;
        }
    }

    let neighbors = jagged::CSR::from_pairs(
        n,
        edges
            .iter()
            .flat_map(|&(u, v)| [(u, (v, ())), (v, (u, ()))]),
    );
    let bct = bcc::BlockCutForest::from_assoc_list(&neighbors);

    let parent = &bct.bct_parent;
    let n_bct = parent.len();
    let mut indegree = vec![0i32; n_bct];
    let mut roots = vec![];
    for u in 0..n_bct {
        let p = parent[u];
        if p != UNSET {
            indegree[p as usize] += 1;
        } else {
            roots.push(u);
        }
    }
    assert_eq!(roots.len(), 1);
    let root = roots[0];
    indegree[root as usize] += 2;

    const RIGID: u8 = 3;
    let mut cmds = vec![];
    for mut b in 0..n_bct as u32 {
        while indegree[b as usize] == 0 {
            let p = parent[b as usize];
            indegree[b as usize] -= 1;
            indegree[p as usize] -= 1;

            if b >= n as u32 {
                match &bct.bct_children[b as usize][..] {
                    &[] => {}
                    &[c0] => {
                        cmds.push(Recolor(c0, 0, 1));
                        cmds.push(Join(p, c0));
                        cmds.push(Connect(p, 0, 1));
                        cmds.push(Recolor(p, 1, RIGID));
                    }
                    [c0, c1, rest @ ..] => {
                        cmds.push(Recolor(*c0, 0, 1));

                        cmds.push(Join(*c1, *c0));
                        cmds.push(Connect(*c1, 0, 1));

                        let mut head = *c1;
                        for &c in rest.iter().chain(Some(&p)) {
                            cmds.push(Recolor(head, 0, 2));
                            cmds.push(Join(c, head));
                            cmds.push(Connect(c, 0, 2));
                            cmds.push(Recolor(c, 2, RIGID));

                            head = c;
                        }

                        assert_eq!(head, p);
                        cmds.push(Connect(head, 0, 1));
                        cmds.push(Recolor(head, 1, RIGID));
                    }
                }
            }

            b = p;
        }
    }

    writeln!(output, "{}", cmds.len()).unwrap();
    for c in cmds {
        writeln!(output, "{}", c).unwrap();
    }
}
