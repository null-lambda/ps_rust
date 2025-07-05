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
