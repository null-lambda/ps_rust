pub mod bcc {
    /// Biconnected components & 2-edge-connected components
    /// Verified with [Yosupo library checker](https://judge.yosupo.jp/problem/biconnected_components)
    use super::jagged::CSR;

    pub const UNSET: u32 = !0;

    pub struct BlockCutForest {
        // DFS tree structure
        pub neighbors: CSR<u32>,
        pub parent: Vec<u32>,
        pub t_in: Vec<u32>,
        pub low: Vec<u32>, // Lowest euler index on a subtree's back edge

        /// Block-cut tree structure,  
        /// represented as a rooted bipartite tree between  
        /// vertex nodes (indices in 0..n) and virtual BCC nodes (indices in n..).  
        /// A vertex node is a cut vertex iff its degree is >= 2,
        /// and the neighbors of a virtual BCC node represents all its belonging vertices.
        pub bct_parent: Vec<u32>,
        pub bct_degree: Vec<u32>,

        /// BCC structure
        pub bcc_edges: Vec<Vec<[u32; 2]>>,
    }

    impl BlockCutForest {
        pub fn from_edges(n_verts: usize, edges: impl Iterator<Item = [u32; 2]> + Clone) -> Self {
            let neighbors = CSR::from_pairs(n_verts, edges.flat_map(|[u, v]| [(u, v), (v, u)]));

            let mut t_in = vec![UNSET; n_verts];
            let mut parent: Vec<_> = (0..n_verts as u32).collect();
            let mut low = vec![0; n_verts];
            let mut timer = 0;

            let mut bct_parent = vec![UNSET; n_verts];
            let mut bct_degree = vec![1u32; n_verts];

            let mut bcc_edges = vec![];

            let mut current_edge: Vec<_> = (0..n_verts)
                .map(|u| neighbors.edge_range(u).start as u32)
                .collect();
            let mut stack = vec![];
            let mut edges_stack: Vec<[u32; 2]> = vec![];
            for root in 0..n_verts {
                if t_in[root] != UNSET {
                    continue;
                }
                t_in[root] = timer;
                timer += 1;

                bct_degree[root] -= 1;
                let mut u = root as u32;
                loop {
                    let p = parent[u as usize];
                    let e = current_edge[u as usize];
                    current_edge[u as usize] += 1;
                    if e == neighbors.edge_range(u as usize).start as u32 {
                        // On enter
                        t_in[u as usize] = timer;
                        low[u as usize] = timer;
                        timer += 1;
                        stack.push(u);
                    }
                    if e == neighbors.edge_range(u as usize).end as u32 {
                        // On exit
                        if p == u {
                            break;
                        }

                        if low[u as usize] >= t_in[p as usize] {
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

                            let mut es = vec![];
                            while let Some(e) = edges_stack.pop() {
                                es.push(e);
                                if e == [p, u] {
                                    break;
                                }
                            }
                            bcc_edges.push(es);
                        }

                        low[p as usize] = low[p as usize].min(low[u as usize]);

                        u = p;
                        continue;
                    }

                    let v = neighbors.links[e as usize];
                    if v == p {
                        continue;
                    }

                    // Notes: multi-edges are pushed only once

                    if t_in[v as usize] == UNSET {
                        // Front edge
                        edges_stack.push([u, v]);
                        parent[v as usize] = u;

                        u = v;
                    } else if t_in[v as usize] < t_in[u as usize] {
                        // Back edge
                        edges_stack.push([u, v]);
                        low[u as usize] = low[u as usize].min(t_in[v as usize]);
                    }
                }

                // For an isolated vertex, manually add a virtual BCC node.
                if neighbors[root].is_empty() {
                    bct_degree[root] += 1;

                    bct_parent.push(root as u32);
                    bct_degree.push(1);

                    bcc_edges.push(vec![]);
                }
            }

            Self {
                neighbors,
                parent,
                low,
                t_in,

                bct_parent,
                bct_degree,

                bcc_edges,
            }
        }

        pub fn is_cut_vert(&self, u: usize) -> bool {
            debug_assert!(u < self.neighbors.len());
            self.bct_degree[u] >= 2
        }

        pub fn is_bridge(&self, u: usize, v: usize) -> bool {
            debug_assert!(u < self.neighbors.len() && v < self.neighbors.len() && u != v);
            self.t_in[v] < self.low[u] || self.t_in[u] < self.low[v]
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
