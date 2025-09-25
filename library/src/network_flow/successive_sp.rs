pub mod mcmf {
    use std::{
        cmp::Reverse,
        collections::{BinaryHeap, VecDeque},
    };

    // Successive shortest path method, O((V+E) (log V) f)
    use crate::jagged;

    pub const UNSET: u32 = u32::MAX;

    type EdgeId = u32;
    type Flow = i32;
    type Cost = i64;
    pub const INF_FLOW: i32 = i32::MAX / 3;
    pub const INF_COST: i64 = i64::MAX / 3;

    pub struct SuccessiveSP {
        pub neighbors: jagged::FS<EdgeId>,
        pub twin: Vec<EdgeId>,

        pub cap: Vec<Flow>,
        pub residual: Vec<Flow>,
        pub cost: Vec<Cost>,

        max_flow: Flow,
        min_cost: Cost,

        pub dist: Vec<Cost>,
        pub phi: Vec<Cost>, // Potential
        parent_edge: Vec<EdgeId>,
    }

    impl SuccessiveSP {
        pub fn empty(n: usize) -> Self {
            Self {
                neighbors: jagged::FS::with_size(n),
                twin: vec![],
                cap: vec![],
                cost: vec![],
                residual: vec![],

                max_flow: 0,
                min_cost: 0,

                dist: vec![INF_COST; n],
                phi: vec![0; n],
                parent_edge: vec![UNSET; n],
            }
        }

        fn n_verts(&self) -> usize {
            self.dist.len()
        }

        pub fn link(&mut self, u: u32, v: u32, cap: Flow, cost: Cost) -> [EdgeId; 2] {
            let e = self.neighbors.insert(u as usize, v);
            let f = self.neighbors.insert(v as usize, u);
            self.twin.extend([f, e]);
            self.cap.extend([cap, cap]);
            self.residual.extend([cap, 0]);
            self.cost.extend([cost, -cost]);
            [e, f]
        }

        pub fn johnson_spfa(&mut self) -> bool {
            // Initialize phi for Johnson's algorithm
            let n = self.n_verts();
            self.phi.fill(0);

            let mut queue = VecDeque::new();
            let mut on_queue = vec![false; n];
            for u in 0..n as u32 {
                queue.push_back(u);
                on_queue[u as usize] = true;
            }

            while let Some(u) = queue.pop_front() {
                on_queue[u as usize] = false;

                let mut e = self.neighbors.head[u as usize];
                while e != UNSET {
                    let (e_next, v) = self.neighbors.links[e as usize];
                    if self.residual[e as usize] >= 1 {
                        let dv_new = self.phi[u as usize] + self.cost[e as usize];
                        if dv_new < self.phi[v as usize] {
                            self.phi[v as usize] = dv_new;

                            if !on_queue[v as usize] {
                                queue.push_back(v);
                                on_queue[v as usize] = true;
                            }
                        }
                    }

                    e = e_next;
                }
            }

            true
        }

        pub fn dijkstra(&mut self, src: u32) {
            self.dist.fill(INF_COST);
            self.parent_edge.fill(UNSET);

            let mut pq = BinaryHeap::new();
            self.dist[src as usize] = 0;
            pq.push(Reverse((0, src)));

            while let Some(Reverse((du, u))) = pq.pop() {
                if du > self.dist[u as usize] {
                    continue;
                }

                let mut e = self.neighbors.head[u as usize];
                while e != UNSET {
                    let (e_next, v) = self.neighbors.links[e as usize];
                    if self.residual[e as usize] >= 1 {
                        let dv_new =
                            self.dist[u as usize] + self.cost[e as usize] + self.phi[u as usize]
                                - self.phi[v as usize];
                        if dv_new < self.dist[v as usize] {
                            self.dist[v as usize] = dv_new;
                            self.parent_edge[v as usize] = e;
                            pq.push(Reverse((dv_new, v)));
                        }
                    }

                    e = e_next;
                }
            }
        }

        pub fn run(&mut self, src: u32, sink: u32) -> (Flow, Cost) {
            assert_ne!(src, sink);

            self.johnson_spfa();

            loop {
                self.dijkstra(src);

                if self.parent_edge[sink as usize] == UNSET {
                    break;
                }

                let mut delta_flow = INF_FLOW;
                let mut u = sink;
                let mut delta_cost = 0;
                while u != src {
                    let e = self.parent_edge[u as usize] as usize;
                    let f = self.twin[e] as usize;

                    delta_flow = delta_flow.min(self.residual[e as usize]);
                    delta_cost += delta_flow as Cost * self.cost[e as usize];

                    let (_, p) = self.neighbors.links[f as usize];
                    u = p;
                }

                self.max_flow += delta_flow;
                self.min_cost += delta_cost;

                for u in 0..self.n_verts() {
                    if self.cost[u] < INF_COST {
                        self.phi[u] += self.dist[u];
                    }
                }

                let mut u = sink;
                while u != src {
                    let e = self.parent_edge[u as usize] as usize;
                    let f = self.twin[e] as usize;
                    self.residual[e] -= delta_flow;
                    self.residual[f] += delta_flow;

                    let (_, p) = self.neighbors.links[f];
                    u = p;
                }
            }

            (self.max_flow, self.min_cost)
        }
    }
}
