fn gen_scc(children: &jagged::CSR<u32>) -> (usize, Vec<u32>) {
    // Tarjan algorithm, iterative
    let n = children.len();

    const UNSET: u32 = !0;
    let mut scc_index = vec![UNSET; n];
    let mut n_scc = 0;

    // Stackless DFS
    let mut parent = vec![UNSET; n];
    let mut current_edge: Vec<_> = (0..n)
        .map(|u| children.edge_range(u).start as u32)
        .collect();
    let mut t_in = vec![0u32; n];
    let mut timer = 1;

    let mut low_link = vec![UNSET; n];
    let mut path_stack = vec![];

    for mut u in 0..n as u32 {
        if t_in[u as usize] > 0 {
            continue;
        }

        parent[u as usize] = u;
        loop {
            let e = current_edge[u as usize];
            current_edge[u as usize] += 1;

            if e == children.edge_range(u as usize).start as u32 {
                // On enter
                t_in[u as usize] = timer;
                low_link[u as usize] = timer;
                timer += 1;
                path_stack.push(u);
            }

            if e < children.edge_range(u as usize).end as u32 {
                let v = children.links[e as usize];
                if t_in[v as usize] == 0 {
                    // Front edge
                    parent[v as usize] = u;

                    u = v;
                } else if scc_index[v as usize] == UNSET {
                    // Back edge or cross edge, scc not constructed yet
                    low_link[u as usize] = low_link[u as usize].min(t_in[v as usize]);
                }
            } else {
                // On exit
                if low_link[u as usize] == t_in[u as usize] {
                    // Found a scc
                    loop {
                        let v = path_stack.pop().unwrap();
                        scc_index[v as usize] = n_scc;
                        if v == u {
                            break;
                        }
                    }
                    n_scc += 1;
                }

                let p = parent[u as usize];
                if p == u {
                    break;
                }
                low_link[p as usize] = low_link[p as usize].min(low_link[u as usize]);
                u = p;
            }
        }
    }
    (n_scc as usize, scc_index)
}

pub struct TwoSat {
    n_props: usize,
    edges: Vec<(u32, u32)>,
}

impl TwoSat {
    pub fn new(n_props: usize) -> Self {
        Self {
            n_props,
            edges: vec![],
        }
    }

    pub fn add_disj(&mut self, (p, bp): (u32, bool), (q, bq): (u32, bool)) {
        self.edges
            .push((self.prop_to_node((p, !bp)), self.prop_to_node((q, bq))));
        self.edges
            .push((self.prop_to_node((q, !bq)), self.prop_to_node((p, bp))));
    }

    fn prop_to_node(&self, (p, bp): (u32, bool)) -> u32 {
        debug_assert!(p < self.n_props as u32);
        if bp {
            p
        } else {
            self.n_props as u32 + p
        }
    }

    fn node_to_prop(&self, node: u32) -> (u32, bool) {
        if node < self.n_props as u32 {
            (node, true)
        } else {
            (node - self.n_props as u32, false)
        }
    }

    pub fn solve<'a>(&'a self) -> Option<impl 'a + FnOnce() -> Vec<bool>> {
        let children = jagged::CSR::from_pairs(self.n_props * 2, self.edges.iter().copied());
        let (n_scc, scc_index) = gen_scc(&children);

        let scc = jagged::CSR::from_pairs(
            n_scc,
            (0..self.n_props * 2).map(|u| (scc_index[u], u as u32)),
        );

        for p in 0..self.n_props as u32 {
            if scc_index[self.prop_to_node((p, true)) as usize]
                == scc_index[self.prop_to_node((p, false)) as usize]
            {
                return None;
            }
        }

        Some(move || {
            let mut interpretation = vec![2u8; self.n_props];
            for u in 0..n_scc {
                for &i in &scc[u] {
                    let (p, v) = self.node_to_prop(i);
                    if interpretation[p as usize] != 2u8 {
                        break;
                    }
                    interpretation[p as usize] = v as u8;
                }
            }

            interpretation.into_iter().map(|x| x != 0).collect()
        })
    }
}
