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

    pub fn solve(&self) -> Option<Vec<bool>> {
        let (scc_count, scc_index) =
            gen_scc(&jagged::CSR::from_assoc_list(self.n_props * 2, &self.edges));

        let mut scc = vec![vec![]; scc_count];
        for (i, &scc_idx) in scc_index.iter().enumerate() {
            scc[scc_idx as usize].push(i as u32);
        }

        let satisfiable = (0..self.n_props as u32).all(|p| {
            scc_index[self.prop_to_node((p, true)) as usize]
                != scc_index[self.prop_to_node((p, false)) as usize]
        });
        if !satisfiable {
            return None;
        }

        let mut interpretation = vec![None; self.n_props];
        for component in &scc {
            for &i in component.iter() {
                let (p, p_value) = self.node_to_prop(i);
                if interpretation[p as usize].is_some() {
                    break;
                }
                interpretation[p as usize] = Some(p_value);
            }
        }
        Some(interpretation.into_iter().map(|x| x.unwrap()).collect())
    }
}
