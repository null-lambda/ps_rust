pub mod ford_fulkerson {
    use crate::jagged;

    pub const UNSET: u32 = u32::MAX;

    type EdgeId = u32;
    type Flow = i32;
    pub const INF_FLOW: i32 = i32::MAX / 3;

    pub struct MaxFlow {
        pub neighbors: jagged::FS<EdgeId>,
        pub rev: Vec<EdgeId>,
        pub cap: Vec<Flow>,
        pub residual: Vec<Flow>,

        max_flow: Flow,

        parent_edge: Vec<EdgeId>,
    }

    impl MaxFlow {
        pub fn empty(n: usize) -> Self {
            Self {
                neighbors: jagged::FS::with_size(n),
                rev: vec![],
                cap: vec![],
                residual: vec![],

                max_flow: 0,

                parent_edge: vec![UNSET; n],
            }
        }

        pub fn link(&mut self, u: u32, v: u32, cap: Flow) -> EdgeId {
            let e = self.neighbors.insert(u as usize, v);
            let f = self.neighbors.insert(v as usize, u);
            self.rev.extend([f, e]);
            self.cap.extend([cap, cap]);
            self.residual.extend([cap, 0]);
            e
        }

        pub fn increment_cap(&mut self, e: EdgeId, cap_new: Flow) {
            let delta = cap_new - self.cap[e as usize];
            assert!(delta >= 0);

            let f = self.rev[e as usize];
            self.cap[e as usize] = cap_new;
            self.cap[f as usize] = cap_new;
            self.residual[e as usize] += delta;
        }

        fn dfs(&mut self, sink: u32, u: u32) -> Flow {
            if u == sink {
                return INF_FLOW;
            }

            let mut e = self.neighbors.head[u as usize];
            while e != UNSET {
                let (e_next, v) = self.neighbors.links[e as usize];
                if self.parent_edge[v as usize] != UNSET || self.residual[e as usize] == 0 {
                    e = e_next;
                    continue;
                }

                self.parent_edge[v as usize] = e;
                let delta = self.residual[e as usize].min(self.dfs(sink, v));
                if delta == 0 {
                    e = e_next;
                    continue;
                }

                return delta;
            }

            0
        }

        pub fn run(&mut self, src: u32, sink: u32) -> Flow {
            assert_ne!(src, sink);

            loop {
                self.parent_edge.fill(UNSET);
                let delta = self.dfs(sink, src);
                if delta == 0 {
                    break;
                }

                self.max_flow += delta;

                let mut u = sink;
                while u != src {
                    let e = self.parent_edge[u as usize] as usize;
                    let f = self.rev[e] as usize;
                    self.residual[e] -= delta;
                    self.residual[f] += delta;

                    let (_, u_next) = self.neighbors.links[f];
                    u = u_next;
                }
            }

            self.max_flow
        }
    }
}
