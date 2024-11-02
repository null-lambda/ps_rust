pub mod network_flow {
    use std::collections::VecDeque;

    type Flow = i32;
    const INFINITY: Flow = i32::MAX / 3;

    struct Edge {
        to: u32,
        rev: u32,
        cap: Flow,
        cap_residual: Flow,
    }

    pub struct HLPP {
        n: usize,
        edges: Vec<Vec<Edge>>,
        pos: Vec<usize>,
        excess: Vec<Flow>,
        height: Vec<u32>,
        excess_next: Vec<u32>,
        gap_prev: Vec<u32>,
        gap_next: Vec<u32>,
        excess_highest: u32,
        gap_highest: u32,
        discharge_count: u32,
    }

    impl HLPP {
        pub fn new(n: usize) -> Self {
            Self {
                n,
                edges: (0..n).map(|_| vec![]).collect(),
                pos: vec![0; n],
                excess: vec![0; n],
                height: vec![0; n],
                excess_next: vec![0; n * 2],
                gap_prev: vec![0; n * 2],
                gap_next: vec![0; n * 2],
                excess_highest: 0,
                gap_highest: 0,
                discharge_count: 0,
            }
        }

        pub fn add_edge(&mut self, u: usize, v: usize, lower: Flow, upper: Flow) {
            debug_assert!(u < self.n);
            debug_assert!(v < self.n);
            debug_assert!(0 <= lower && lower <= upper);
            let u_rev = self.edges[u].len();
            let v_rev = self.edges[v].len();
            self.edges[u].push(Edge {
                to: v as u32,
                rev: v_rev as u32,
                cap: upper,
                cap_residual: upper,
            });
            self.edges[v].push(Edge {
                to: u as u32,
                rev: u_rev as u32,
                cap: -lower,
                cap_residual: -lower,
            });
        }

        pub fn get_flows(&self, u: usize) -> Vec<(u32, Flow)> {
            let mut res = vec![];
            for i in 0..self.edges[u].len() {
                let Edge {
                    to,
                    cap: cap_orig,
                    cap_residual: cap,
                    ..
                } = self.edges[u][i];

                res.push((to, cap_orig - cap));
            }
            res
        }

        fn excess_insert(&mut self, u: usize, h: u32) {
            self.excess_next[u] = self.excess_next[self.n + h as usize];
            self.excess_next[self.n + h as usize] = u as u32;
            self.excess_highest = self.excess_highest.max(h);
        }

        fn excess_add(&mut self, u: usize, f: Flow) {
            self.excess[u] += f;
            if self.excess[u] == f && self.height[u] != self.n as u32 + 1 {
                self.excess_insert(u, self.height[u]);
            }
        }

        fn gap_insert(&mut self, u: usize, h: u32) {
            self.gap_prev[u] = self.n as u32 + h;
            self.gap_next[u] = self.gap_next[self.n + h as usize];
            self.gap_prev[self.gap_next[u] as usize] = u as u32;
            self.gap_next[self.gap_prev[u] as usize] = u as u32;
            self.gap_highest = self.gap_highest.max(h);
        }

        fn gap_remove(&mut self, u: usize) {
            self.gap_next[self.gap_prev[u] as usize] = self.gap_next[u];
            self.gap_prev[self.gap_next[u] as usize] = self.gap_prev[u];
        }

        fn update_height(&mut self, u: usize, h: u32) {
            if self.height[u] != self.n as u32 + 1 {
                self.gap_remove(u);
            }
            self.height[u] = h;
            if h >= self.n as u32 {
                return;
            }
            self.gap_insert(u, h);
            if self.excess[u] > 0 {
                self.excess_insert(u, h);
            }
        }

        fn global_relabel(&mut self, dest: usize) {
            self.discharge_count = 0;
            for i in self.n..self.n * 2 {
                self.excess_next[i] = i as u32;
                self.gap_prev[i] = i as u32;
                self.gap_next[i] = i as u32;
            }
            self.height.fill(self.n as u32 + 1);
            self.height[dest] = 0;

            let mut queue: VecDeque<usize> = [dest].into();

            while let Some(u) = queue.pop_front() {
                for j in 0..self.edges[u].len() {
                    let Edge { to, rev, .. } = self.edges[u][j];
                    if (self.edges[to as usize][rev as usize].cap_residual == 0)
                        || (self.height[to as usize] <= self.height[u] + 1)
                    {
                        continue;
                    }

                    self.update_height(to as usize, self.height[u] + 1);
                    queue.push_back(to as usize);
                }
            }
        }

        fn discharge(&mut self, from: usize) {
            let h = self.height[from];
            let mut nh = self.n as u32;

            let i = self.edges[from].len();
            for j in 0..i {
                let p = (self.pos[from] + i - j) % i;
                let Edge {
                    to,
                    rev,
                    cap_residual: cap,
                    ..
                } = self.edges[from][p];
                if cap != 0 {
                    if h != self.height[to as usize] + 1 {
                        nh = nh.min(self.height[to as usize]);
                    } else {
                        let f = self.excess[from].min(cap);
                        self.excess[from] -= f;
                        self.excess_add(to as usize, f);
                        self.edges[from][p].cap_residual -= f;
                        self.edges[to as usize][rev as usize].cap_residual += f;
                        if self.excess[from] == 0 {
                            return;
                        }
                    }
                }
            }

            self.discharge_count += 1;

            if self.gap_next[self.n + h as usize] < self.n as u32 {
                self.update_height(from, nh + 1);
                return;
            }

            while self.gap_highest >= h {
                while self.gap_next[self.n + self.gap_highest as usize] < self.n as u32 {
                    let j = self.gap_next[self.n + self.gap_highest as usize] as usize;
                    self.height[j] = self.n as u32 + 1;
                    self.gap_remove(j);
                }
                self.gap_highest -= 1;
            }
        }

        pub fn max_flow(&mut self, src: usize, dest: usize) -> Flow {
            self.global_relabel(dest);
            self.excess_add(src, INFINITY);
            self.excess[dest] -= INFINITY;

            while self.excess_highest != 0 {
                let u = self.excess_next[self.n + self.excess_highest as usize];
                if u >= self.n as u32 {
                    self.excess_highest -= 1;
                    continue;
                }
                self.excess_next[self.n + self.excess_highest as usize] =
                    self.excess_next[u as usize];
                if self.height[u as usize] != self.excess_highest {
                    continue;
                }
                self.discharge(u as usize);
                if self.discharge_count >= 4 * self.n as u32 {
                    self.global_relabel(dest);
                }
            }

            return self.excess[dest] + INFINITY;
        }
    }
}
