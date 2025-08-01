mod rand {
    // Written in 2015 by Sebastiano Vigna (vigna@acm.org)
    // https://xoshiro.di.unimi.it/splitmix64.c
    use std::ops::Range;

    pub struct SplitMix64(u64);

    impl SplitMix64 {
        pub fn new(seed: u64) -> Self {
            assert_ne!(seed, 0);
            Self(seed)
        }

        // Available on x86-64 and target feature rdrand only.
        #[cfg(target_arch = "x86_64")]
        pub fn from_entropy() -> Self {
            let mut seed = 0;
            unsafe {
                if std::arch::x86_64::_rdrand64_step(&mut seed) == 1 {
                    Self(seed)
                } else {
                    panic!("Failed to get entropy");
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        pub fn from_entropy() -> Self {
            use std::time::{SystemTime, UNIX_EPOCH};
            let seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            Self(seed as u64)
        }

        pub fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = self.0;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        }

        pub fn range_u64(&mut self, range: Range<u64>) -> u64 {
            let Range { start, end } = range;
            debug_assert!(start < end);

            let width = end - start;
            let test = (u64::MAX - width) % width;
            loop {
                let value = self.next_u64();
                if value >= test {
                    return start + value % width;
                }
            }
        }

        pub fn shuffle<T>(&mut self, xs: &mut [T]) {
            let n = xs.len();
            for i in 0..n - 1 {
                let j = self.range_u64(i as u64..n as u64) as usize;
                xs.swap(i, j);
            }
        }
    }
}

pub mod network_flow {
    const UNSET: u32 = u32::MAX;

    // network simplex methods for dense graph MCMF
    // adapted from brunodccarvalho's c++ implementation:
    // https://gist.github.com/brunodccarvalho/fb9f2b47d7f8469d209506b336013473
    type Flow = i32;
    type Cost = i32;
    type NodeId = u32; // TOOD: use newtype instead of type alias
    type EdgeId = u32;

    struct MultiList {
        n_data: u32,
        n_list: u32,
        next: Vec<u32>,
        prev: Vec<u32>,
    }

    impl MultiList {
        pub fn new(n_data: u32, n_list: u32) -> Self {
            let next: Vec<u32> = ((0..n_data).map(|_| 0))
                .chain(n_data..n_data + n_list)
                .collect();
            let prev = next.clone();

            Self {
                n_data,
                n_list,
                next,
                prev,
            }
        }

        fn rep(&self, idx_list: u32) -> u32 {
            debug_assert!(idx_list < self.n_list);
            idx_list + self.n_data
        }

        pub fn head(&self, idx_list: u32) -> u32 {
            self.next[self.rep(idx_list) as usize]
        }

        pub fn tail(&self, idx_list: u32) -> u32 {
            self.prev[self.rep(idx_list) as usize]
        }

        pub fn push_front(&mut self, idx_list: u32, idx_elem: u32) {
            debug_assert!(idx_list < self.n_list);
            debug_assert!(idx_elem < self.n_data);
            self.link3(self.rep(idx_list), idx_elem, self.head(idx_list));
        }

        pub fn push_back(&mut self, idx_list: u32, idx_elem: u32) {
            debug_assert!(idx_list < self.n_list);
            debug_assert!(idx_elem < self.n_data);
            self.link3(self.tail(idx_list), idx_elem, self.rep(idx_list));
        }

        pub fn erase(&mut self, idx_elem: u32) {
            debug_assert!(idx_elem < self.n_data);
            self.link(self.prev[idx_elem as usize], self.next[idx_elem as usize]);
        }

        fn link(&mut self, u: u32, v: u32) {
            self.next[u as usize] = v;
            self.prev[v as usize] = u;
        }
        fn link3(&mut self, u: u32, v: u32, w: u32) {
            self.link(u, v);
            self.link(v, w);
        }
    }

    #[derive(Debug, Clone, Default)]
    struct Node {
        parent: NodeId,
        pred: EdgeId,
        supply: Flow,
        potential: Cost,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ArcState {
        Upper = -1,
        Tree = 0,
        Lower = 1,
    }

    impl ArcState {
        pub fn reverse(self) -> Self {
            match self {
                ArcState::Upper => ArcState::Lower,
                ArcState::Tree => ArcState::Tree,
                ArcState::Lower => ArcState::Upper,
            }
        }
    }

    #[derive(Debug, Clone)]
    struct Edge {
        nodes: [NodeId; 2],
        lower: Flow,
        upper: Flow,
        cost: Cost,
        flow: Flow,
        state: ArcState,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum CirculationState {
        Infeasible,
        Optimal,
    }

    pub struct NetworkSimplex {
        n_verts: u32,
        n_edges: u32,
        nodes: Vec<Node>,
        edges: Vec<Edge>,
        children: MultiList,
        next_arc: u32,
        block_size: u32,
        bfs: Vec<NodeId>,
        perm: Vec<u32>,
    }

    impl NetworkSimplex {
        pub fn with_size(n_verts: u32) -> Self {
            Self {
                n_verts,
                n_edges: 0,
                nodes: vec![Default::default(); n_verts as usize + 1],
                edges: vec![],
                children: MultiList::new(n_verts + 1, n_verts + 1),
                next_arc: 0,
                block_size: 0,
                bfs: vec![],
                perm: vec![],
            }
        }

        pub fn set_supply(&mut self, node: NodeId, supply: Flow) {
            debug_assert!(supply >= 0);
            self.nodes[node as usize].supply = supply;
        }

        pub fn set_demand(&mut self, node: NodeId, demand: Flow) {
            debug_assert!(demand >= 0);
            self.nodes[node as usize].supply = -demand;
        }

        pub fn add_edge(
            &mut self,
            nodes: [NodeId; 2],
            lower: Flow,
            upper: Flow,
            cost: Cost,
        ) -> EdgeId {
            let [u, v] = nodes;
            debug_assert!(u < self.n_verts && v < self.n_verts);

            let res = self.n_edges;
            self.edges.push(Edge {
                nodes,
                lower,
                upper,
                cost,
                flow: 0,
                state: ArcState::Lower,
            });
            self.n_edges += 1;
            res
        }

        fn reduced_cost(&self, e: EdgeId) -> Cost {
            let [u, v] = self.edges[e as usize].nodes;
            self.edges[e as usize].cost + self.nodes[u as usize].potential
                - self.nodes[v as usize].potential
        }

        pub fn excesses(&self) -> Vec<Flow> {
            let mut excess = vec![0; self.n_verts as usize];
            for e in 0..self.n_edges {
                let [u, v] = self.edges[e as usize].nodes;
                excess[u as usize] += self.edges[e as usize].upper;
                excess[v as usize] -= self.edges[e as usize].upper;
            }
            excess
        }

        pub fn circulation_cost(&self) -> Cost {
            (0..self.n_edges)
                .map(|e| self.edges[e as usize].flow * self.edges[e as usize].cost)
                .sum()
        }

        fn validate_flow(&self) {
            for e in 0..self.n_edges {
                debug_assert!(
                    (self.edges[e as usize].lower..=self.edges[e as usize].upper)
                        .contains(&self.edges[e as usize].flow)
                );
                debug_assert!(
                    self.edges[e as usize].flow == self.edges[e as usize].lower
                        || self.reduced_cost(e) <= 0
                );
                debug_assert!(
                    self.edges[e as usize].flow == self.edges[e as usize].upper
                        || self.reduced_cost(e) >= 0
                );
            }
        }

        pub fn min_cost_circulation(&mut self) -> CirculationState {
            if self.nodes.iter().map(|n| n.supply).sum::<Flow>() != 0 {
                return CirculationState::Infeasible;
            }
            self.run();
            let res = if (self.n_edges..self.n_edges + self.n_verts)
                .all(|e| self.edges[e as usize].flow == 0)
            {
                CirculationState::Optimal
            } else {
                CirculationState::Infeasible
            };
            self.edges.truncate(self.n_edges as usize);
            res
        }

        pub fn min_cost_max_flow(&mut self) -> Flow {
            self.run();
            let res = self.edges[self.n_edges as usize..(self.n_edges + self.n_verts) as usize]
                .iter()
                .filter(|&e| e.nodes[1] == self.n_verts)
                .map(|e| e.upper - e.flow)
                .sum();
            self.edges.truncate(self.n_edges as usize);
            res
        }

        fn signed_reduced_cost(&self, e: EdgeId) -> Cost {
            self.edges[e as usize].state as Flow * self.reduced_cost(e)
        }

        fn run(&mut self) {
            let mut artif_cost = 1;
            for e in 0..self.n_edges {
                let [u, v] = self.edges[e as usize].nodes;
                self.edges[e as usize].flow = 0;
                self.edges[e as usize].state = ArcState::Lower;
                self.edges[e as usize].upper -= self.edges[e as usize].lower;
                self.nodes[u as usize].supply -= self.edges[e as usize].lower;
                self.nodes[v as usize].supply += self.edges[e as usize].lower;
                artif_cost += self.edges[e as usize].cost.abs();
            }

            self.bfs.resize(self.n_verts as usize + 1, 0);
            self.children = MultiList::new(self.n_verts + 1, self.n_verts + 1);

            let root = self.n_verts;
            self.nodes[root as usize] = Node {
                parent: UNSET,
                pred: UNSET,
                supply: 0,
                potential: 0,
            };

            for u in 0..self.n_verts {
                let e = self.n_edges + u;
                self.nodes[u as usize].parent = root;
                self.nodes[u as usize].pred = e;
                self.children.push_back(root, u);
                let supply = self.nodes[u as usize].supply;

                if supply >= 0 {
                    self.nodes[u as usize].potential = -artif_cost;
                    self.edges.push(Edge {
                        nodes: [u, root],
                        lower: 0,
                        upper: supply,
                        cost: artif_cost,
                        flow: supply,
                        state: ArcState::Tree,
                    });
                } else {
                    self.nodes[u as usize].potential = artif_cost;
                    self.edges.push(Edge {
                        nodes: [root, u],
                        lower: 0,
                        upper: -supply,
                        cost: artif_cost,
                        flow: -supply,
                        state: ArcState::Tree,
                    });
                }
            }
            debug_assert_eq!(self.edges.len(), (self.n_verts + self.n_edges) as usize);

            self.block_size = (((self.n_edges + self.n_verts) as f64).sqrt().ceil() as u32)
                .min(self.n_verts + 1)
                .max(5);
            self.next_arc = 0;

            let mut rng = crate::rand::SplitMix64::from_entropy();
            self.perm = (0..self.n_edges + self.n_verts).collect();
            rng.shuffle(&mut self.perm);

            while let Some(in_arc) = self.select_pivot_edge() {
                self.pivot(in_arc);
            }

            for e in 0..self.n_edges {
                let [u, v] = self.edges[e as usize].nodes;
                self.edges[e as usize].flow += self.edges[e as usize].lower;
                self.edges[e as usize].upper += self.edges[e as usize].lower;
                self.nodes[u as usize].supply += self.edges[e as usize].lower;
                self.nodes[v as usize].supply -= self.edges[e as usize].lower;
            }
        }

        fn select_pivot_edge(&mut self) -> Option<u32> {
            let mut in_arc = UNSET;
            let mut signed_reduced_cost = 0;

            for count in 0..self.n_verts + self.n_edges {
                let x = self.perm[self.next_arc as usize];
                self.next_arc = (self.next_arc + 1) % (self.n_verts + self.n_edges);

                let c = self.signed_reduced_cost(x);
                if c < signed_reduced_cost {
                    signed_reduced_cost = c;
                    in_arc = x;
                }

                if count % self.block_size == self.block_size - 1 && signed_reduced_cost < 0 {
                    break;
                }
            }
            let res = in_arc;
            (res != UNSET).then(|| res)
        }

        fn pivot(&mut self, in_arc: u32) {
            let [u_in, v_in] = self.edges[in_arc as usize].nodes;
            let [mut a, mut b] = [u_in, v_in];

            let parent_or = |u: u32, alt| {
                let p = self.nodes[u as usize].parent;
                if p == UNSET {
                    alt
                } else {
                    p
                }
            };

            let join = loop {
                if a == b {
                    break a;
                }
                a = parent_or(a, v_in);
                b = parent_or(b, u_in);
            };

            let [src, dest] = match self.edges[in_arc as usize].state {
                ArcState::Lower => [u_in, v_in],
                _ => [v_in, u_in],
            };

            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
            enum OutArcSide {
                Same,
                Source,
                Target,
            }

            let mut flow_delta = self.edges[in_arc as usize].upper;
            let mut side = OutArcSide::Same;
            let mut u_out = UNSET;

            let mut u = src;
            while u != join && flow_delta != 0 {
                let e = self.nodes[u as usize].pred;
                let edge_down = u == self.edges[e as usize].nodes[1];
                let d = if edge_down {
                    self.edges[e as usize].upper - self.edges[e as usize].flow
                } else {
                    self.edges[e as usize].flow
                };
                if flow_delta > d {
                    flow_delta = d;
                    u_out = u;
                    side = OutArcSide::Source;
                }
                u = self.nodes[u as usize].parent;
            }

            let mut u = dest;
            while u != join && (flow_delta != 0 || side != OutArcSide::Target) {
                let e = self.nodes[u as usize].pred;
                let edge_up = u == self.edges[e as usize].nodes[0];
                let d = if edge_up {
                    self.edges[e as usize].upper - self.edges[e as usize].flow
                } else {
                    self.edges[e as usize].flow
                };
                if flow_delta >= d {
                    flow_delta = d;
                    u_out = u;
                    side = OutArcSide::Target;
                }
                u = self.nodes[u as usize].parent;
            }

            if flow_delta != 0 {
                let delta = self.edges[in_arc as usize].state as Flow * flow_delta;
                self.edges[in_arc as usize].flow += delta;

                let mut u = self.edges[in_arc as usize].nodes[0];
                while u != join {
                    let e = self.nodes[u as usize].pred;
                    self.edges[e as usize].flow += if u == self.edges[e as usize].nodes[0] {
                        -delta
                    } else {
                        delta
                    };
                    u = self.nodes[u as usize].parent;
                }

                let mut u = self.edges[in_arc as usize].nodes[1];
                while u != join {
                    let e = self.nodes[u as usize].pred;
                    self.edges[e as usize].flow += if u == self.edges[e as usize].nodes[0] {
                        delta
                    } else {
                        -delta
                    };
                    u = self.nodes[u as usize].parent;
                }
            }

            if side == OutArcSide::Same {
                self.edges[in_arc as usize].state = self.edges[in_arc as usize].state.reverse();
                return;
            }

            let out_arc = self.nodes[u_out as usize].pred;
            self.edges[in_arc as usize].state = ArcState::Tree;
            self.edges[out_arc as usize].state = if self.edges[out_arc as usize].flow != 0 {
                ArcState::Upper
            } else {
                ArcState::Lower
            };

            let [u_in, v_in] = if side == OutArcSide::Source {
                [src, dest]
            } else {
                [dest, src]
            };

            let mut u = u_in;
            let mut s = 0;
            while u != u_out {
                self.bfs[s] = u;
                s += 1;
                u = self.nodes[u as usize].parent;
            }

            for i in (0..s).rev() {
                let u = self.bfs[i];
                let p = self.nodes[u as usize].parent;
                self.children.erase(p);
                self.children.push_back(u, p);
                self.nodes[p as usize].parent = u;
                self.nodes[p as usize].pred = self.nodes[u as usize].pred;
            }
            self.children.erase(u_in);
            self.children.push_back(v_in, u_in);
            self.nodes[u_in as usize].parent = v_in;
            self.nodes[u_in as usize].pred = in_arc;

            let current_potential = self.reduced_cost(in_arc);
            let potential_delta = if u_in == self.edges[in_arc as usize].nodes[0] {
                -current_potential
            } else {
                current_potential
            };
            debug_assert_ne!(potential_delta, 0);

            self.bfs[0] = u_in;
            let mut s = 1;
            for i in 0.. {
                if i >= s {
                    break;
                }

                let u = self.bfs[i];
                self.nodes[u as usize].potential += potential_delta;
                let mut v = self.children.head(u);
                while v != self.children.rep(u) {
                    self.bfs[s] = v;
                    s += 1;
                    v = self.children.next[v as usize];
                }
            }
        }
    }
}
