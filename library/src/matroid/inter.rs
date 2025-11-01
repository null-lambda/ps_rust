pub mod matroid_inter {
    use std::collections::VecDeque;

    pub(crate) type BitVec = crate::bitset::BitVec;

    pub const UNSET: u32 = u32::MAX;

    // An abstract query structure for building an exchange graph.
    // Use lazy or amortized evaluation if possible.
    pub trait ExchangeOracle {
        fn len(&self) -> usize;

        fn load_indep_set(&mut self, indep_set: &BitVec);

        // Test whether I U {i} is independent.
        fn can_insert(&mut self, i: usize) -> bool;

        // Test whether I - {i} + {j} is indepdendent.
        fn can_exchange(&mut self, _i: usize, _j: usize) -> bool {
            unimplemented!()
        }

        // Assuming i in I, visit all exchangable j.
        fn left_exchange(&mut self, indep_set: &BitVec, i: usize, mut visitor: impl FnMut(usize)) {
            if !indep_set.get(i) {
                return;
            }
            for j in 0..self.len() {
                if !indep_set.get(j) && self.can_exchange(i, j) {
                    visitor(j);
                }
            }
        }

        // Assuming j not in I, visit all exchangable j.
        fn right_exchange(&mut self, indep_set: &BitVec, j: usize, mut visitor: impl FnMut(usize)) {
            if indep_set.get(j) {
                return;
            }
            for i in 0..self.len() {
                if indep_set.get(i) && self.can_exchange(i, j) {
                    visitor(i);
                }
            }
        }
    }

    fn ascend_to_root(parent: &[u32], mut u: usize, mut visitor: impl FnMut(usize)) {
        loop {
            visitor(u);

            if u == parent[u] as usize {
                break;
            }
            u = parent[u] as usize;
        }
    }

    pub fn inter(m1: &mut impl ExchangeOracle, m2: &mut impl ExchangeOracle) -> (BitVec, usize) {
        fn augment(
            m1: &mut impl ExchangeOracle,
            m2: &mut impl ExchangeOracle,
            indep_set: &mut BitVec,
        ) -> bool {
            let n = m1.len();
            m1.load_indep_set(&indep_set);
            m2.load_indep_set(&indep_set);

            let mut parent = vec![UNSET; n];
            let mut bfs = vec![];
            for i in 0..n {
                if !indep_set.get(i) && m1.can_insert(i) {
                    bfs.push(i as u32);
                    parent[i] = i as u32;
                }
            }

            let is_dest: Vec<bool> = (0..n)
                .map(|i| !indep_set.get(i) && m2.can_insert(i))
                .collect();
            let mut timer = 0;

            while let Some(u) = bfs.get(timer).map(|&u| u as usize) {
                timer += 1;

                if is_dest[u] {
                    ascend_to_root(&parent, u as usize, |u| indep_set.toggle(u));
                    return true;
                }

                let mut try_enqueue = |v| {
                    if parent[v] == UNSET {
                        parent[v] = u as u32;
                        bfs.push(v as u32);
                    }
                };

                m1.left_exchange(&indep_set, u, |v| try_enqueue(v));
                m2.right_exchange(&indep_set, u, |v| try_enqueue(v));
            }

            false
        }

        assert_eq!(m1.len(), m2.len());
        let mut set = BitVec::with_size(m1.len());
        let mut rank = 0;
        while augment(m1, m2, &mut set) {
            rank += 1;
        }
        (set, rank)
    }

    pub type W = i64;
    pub const W_INF: W = W::MAX / 3;

    pub fn inter_max_weight(
        m1: &mut impl ExchangeOracle,
        m2: &mut impl ExchangeOracle,
        weights: &[W],
        mut yield_weight: impl FnMut(W),
    ) -> (BitVec, usize) {
        fn augment_spfa(
            m1: &mut impl ExchangeOracle,
            m2: &mut impl ExchangeOracle,
            weights: &[W],
            indep_set: &mut BitVec,
        ) -> Option<W> {
            let n = m1.len();
            m1.load_indep_set(&indep_set);
            m2.load_indep_set(&indep_set);

            let mut bfs = vec![];
            let mut parent = vec![UNSET; n];

            let mut spfa_queue = VecDeque::new();
            let mut on_queue = vec![false; n];
            let mut dist = vec![(W_INF, 0u32); n];

            for u in 0..n {
                if !indep_set.get(u) && m1.can_insert(u) {
                    bfs.push(u as u32);
                    parent[u] = u as u32;
                    dist[u] = (-weights[u], 0);

                    spfa_queue.push_back(u as u32);
                    on_queue[u as usize] = true;
                }
            }

            let is_dest: Vec<bool> = (0..n)
                .map(|u| !indep_set.get(u) && m2.can_insert(u))
                .collect();

            let mut timer = 0;
            let mut adj = vec![vec![]; n];
            while let Some(u) = bfs.get(timer).map(|&u| u as usize) {
                timer += 1;

                let mut try_enqueue = |v| {
                    if parent[v] == UNSET {
                        parent[v] = u as u32;
                        bfs.push(v as u32);
                    }
                };

                m1.left_exchange(&indep_set, u, |v| {
                    try_enqueue(v);
                    assert!(!indep_set.get(v));
                    adj[u].push((v as u32, -weights[v]));
                });
                m2.right_exchange(&indep_set, u, |v| {
                    try_enqueue(v);
                    assert!(indep_set.get(v));
                    adj[u].push((v as u32, weights[v]));
                });
            }

            while let Some(u) = spfa_queue.pop_front() {
                on_queue[u as usize] = false;

                for &(v, w) in &adj[u as usize] {
                    let dv_new = (dist[u as usize].0 + w, dist[u as usize].1 + 1);
                    if dv_new < dist[v as usize] {
                        dist[v as usize] = dv_new;
                        parent[v as usize] = u;

                        if !on_queue[v as usize] {
                            spfa_queue.push_back(v);
                            on_queue[v as usize] = true;
                        }
                    }
                }
            }

            let u = (0..n)
                .filter(|&u| is_dest[u] && dist[u].0 < W_INF)
                .min_by_key(|&u| dist[u])?;
            ascend_to_root(&parent, u as usize, |u| indep_set.toggle(u));
            Some(-dist[u].0)
        }

        assert_eq!(m1.len(), m2.len());
        assert_eq!(m1.len(), weights.len());
        let mut set = BitVec::with_size(m1.len());
        let mut rank = 0;

        let mut w = 0;
        while let Some(dw) = augment_spfa(m1, m2, &weights, &mut set) {
            rank += 1;
            w += dw;
            yield_weight(w);
        }
        (set, rank)
    }
}
