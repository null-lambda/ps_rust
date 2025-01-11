use std::{cmp::Ordering, io::Write};

mod simple_io {
    pub struct InputAtOnce(std::str::SplitAsciiWhitespace<'static>);

    impl InputAtOnce {
        pub fn token(&mut self) -> &str {
            self.0.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(buf.into_boxed_str());
        InputAtOnce(buf.split_ascii_whitespace())
    }

    pub fn stdout_buf() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

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
    const UNSET: usize = i32::MAX as usize;

    // network simplex methods for dense graph MCMF
    // adapted from brunodccarvalho's c++ implementation:
    // https://gist.github.com/brunodccarvalho/fb9f2b47d7f8469d209506b336013473
    type Flow = i64;
    type Cost = i64;
    type NodeId = usize; // TOOD: use newtype instead of type alias

    struct MultiList {
        n_data: usize,
        n_list: usize,
        next: Vec<usize>,
        prev: Vec<usize>,
    }

    impl MultiList {
        pub fn new(n_data: usize, n_list: usize) -> Self {
            let next: Vec<usize> = ((0..n_data).map(|_| 0))
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

        fn rep(&self, idx_list: usize) -> usize {
            debug_assert!(idx_list < self.n_list);
            idx_list + self.n_data
        }

        pub fn head(&self, idx_list: usize) -> usize {
            self.next[self.rep(idx_list)]
        }

        pub fn tail(&self, idx_list: usize) -> usize {
            self.prev[self.rep(idx_list)]
        }

        pub fn push_front(&mut self, idx_list: usize, idx_elem: usize) {
            debug_assert!(idx_list < self.n_list);
            debug_assert!(idx_elem < self.n_data);
            self.link3(self.rep(idx_list), idx_elem, self.head(idx_list));
        }

        pub fn push_back(&mut self, idx_list: usize, idx_elem: usize) {
            debug_assert!(idx_list < self.n_list);
            debug_assert!(idx_elem < self.n_data);
            self.link3(self.tail(idx_list), idx_elem, self.rep(idx_list));
        }

        pub fn erase(&mut self, idx_elem: usize) {
            debug_assert!(idx_elem < self.n_data);
            self.link(self.prev[idx_elem], self.next[idx_elem]);
        }

        fn link(&mut self, u: usize, v: usize) {
            self.next[u] = v;
            self.prev[v] = u;
        }
        fn link3(&mut self, u: usize, v: usize, w: usize) {
            self.link(u, v);
            self.link(v, w);
        }
    }

    #[derive(Debug, Clone, Default)]
    struct Node {
        parent: NodeId,
        pred: NodeId,
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
        n_verts: usize,
        n_edges: usize,
        nodes: Vec<Node>,
        edges: Vec<Edge>,
        children: MultiList,
        next_arc: usize,
        block_size: usize,
        bfs: Vec<usize>,
        perm: Vec<usize>,
    }

    impl NetworkSimplex {
        pub fn with_size(n_verts: usize) -> Self {
            Self {
                n_verts,
                n_edges: 0,
                nodes: vec![Default::default(); n_verts + 1],
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
            self.nodes[node].supply = supply;
        }

        pub fn set_demand(&mut self, node: NodeId, demand: Flow) {
            debug_assert!(demand >= 0);
            self.nodes[node].supply = -demand;
        }

        pub fn add_edge(&mut self, nodes: [NodeId; 2], lower: Flow, upper: Flow, cost: Cost) {
            let [u, v] = nodes;
            debug_assert!(u < self.n_verts && v < self.n_verts);

            self.edges.push(Edge {
                nodes,
                lower,
                upper,
                cost,
                flow: 0,
                state: ArcState::Lower,
            });
            self.n_edges += 1;
        }

        fn reduced_cost(&self, e: usize) -> Cost {
            let [u, v] = self.edges[e].nodes;
            self.edges[e].cost + self.nodes[u].potential - self.nodes[v].potential
        }

        pub fn excesses(&self) -> Vec<Flow> {
            let mut excess = vec![0; self.n_verts];
            for e in 0..self.n_edges {
                let [u, v] = self.edges[e].nodes;
                excess[u] += self.edges[e].upper;
                excess[v] -= self.edges[e].upper;
            }
            excess
        }

        pub fn circulation_cost(&self) -> Cost {
            (0..self.n_edges)
                .map(|e| self.edges[e].flow * self.edges[e].cost)
                .sum()
        }

        fn validate_flow(&self) {
            for e in 0..self.n_edges {
                debug_assert!(
                    (self.edges[e].lower..=self.edges[e].upper).contains(&self.edges[e].flow)
                );
                debug_assert!(
                    self.edges[e].flow == self.edges[e].lower || self.reduced_cost(e) <= 0
                );
                debug_assert!(
                    self.edges[e].flow == self.edges[e].upper || self.reduced_cost(e) >= 0
                );
            }
        }

        pub fn min_cost_circulation(&mut self) -> CirculationState {
            if self.nodes.iter().map(|n| n.supply).sum::<Flow>() != 0 {
                return CirculationState::Infeasible;
            }
            self.run();
            let res =
                if (self.n_edges..self.n_edges + self.n_verts).all(|e| self.edges[e].flow == 0) {
                    CirculationState::Optimal
                } else {
                    CirculationState::Infeasible
                };
            self.edges.truncate(self.n_edges);
            res
        }

        pub fn min_cost_max_flow(&mut self) -> Flow {
            self.run();
            let res = self.edges[self.n_edges..self.n_edges + self.n_verts]
                .iter()
                .filter(|&e| e.nodes[1] == self.n_verts)
                .map(|e| e.upper - e.flow)
                .sum();
            self.edges.truncate(self.n_edges);
            res
        }

        fn signed_reduced_cost(&self, e: usize) -> Cost {
            self.edges[e].state as Flow * self.reduced_cost(e)
        }

        fn run(&mut self) {
            let mut artif_cost = 1;
            for e in 0..self.n_edges {
                let [u, v] = self.edges[e].nodes;
                self.edges[e].flow = 0;
                self.edges[e].state = ArcState::Lower;
                self.edges[e].upper -= self.edges[e].lower;
                self.nodes[u].supply -= self.edges[e].lower;
                self.nodes[v].supply += self.edges[e].lower;
                artif_cost += self.edges[e].cost.abs();
            }

            self.bfs.resize(self.n_verts + 1, 0);
            self.children = MultiList::new(self.n_verts + 1, self.n_verts + 1);

            let root = self.n_verts;
            self.nodes[root] = Node {
                parent: UNSET,
                pred: UNSET,
                supply: 0,
                potential: 0,
            };

            for u in 0..self.n_verts {
                let e = self.n_edges + u;
                self.nodes[u].parent = root;
                self.nodes[u].pred = e;
                self.children.push_back(root, u);
                let supply = self.nodes[u].supply;

                if supply >= 0 {
                    self.nodes[u].potential = -artif_cost;
                    self.edges.push(Edge {
                        nodes: [u, root],
                        lower: 0,
                        upper: supply,
                        cost: artif_cost,
                        flow: supply,
                        state: ArcState::Tree,
                    });
                } else {
                    self.nodes[u].potential = artif_cost;
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
            debug_assert_eq!(self.edges.len(), self.n_verts + self.n_edges);

            self.block_size = (((self.n_edges + self.n_verts) as f64).sqrt().ceil() as usize)
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
                let [u, v] = self.edges[e].nodes;
                self.edges[e].flow += self.edges[e].lower;
                self.edges[e].upper += self.edges[e].lower;
                self.nodes[u].supply += self.edges[e].lower;
                self.nodes[v].supply -= self.edges[e].lower;
            }
        }

        fn select_pivot_edge(&mut self) -> Option<usize> {
            let mut in_arc = UNSET;
            let mut signed_reduced_cost = 0;

            for count in 0..self.n_verts + self.n_edges {
                let x = self.perm[self.next_arc];
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

        fn pivot(&mut self, in_arc: usize) {
            let [u_in, v_in] = self.edges[in_arc].nodes;
            let [mut a, mut b] = [u_in, v_in];
            let join = loop {
                if a == b {
                    break a;
                }
                a = if self.nodes[a].parent == UNSET {
                    v_in
                } else {
                    self.nodes[a].parent
                };
                b = if self.nodes[b].parent == UNSET {
                    u_in
                } else {
                    self.nodes[b].parent
                };
            };

            let [src, dest] = match self.edges[in_arc].state {
                ArcState::Lower => [u_in, v_in],
                _ => [v_in, u_in],
            };

            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
            enum OutArcSide {
                Same,
                Source,
                Target,
            }

            let mut flow_delta = self.edges[in_arc].upper;
            let mut side = OutArcSide::Same;
            let mut u_out = UNSET;

            let mut u = src;
            while u != join && flow_delta != 0 {
                let e = self.nodes[u].pred;
                let edge_down = u == self.edges[e].nodes[1];
                let d = if edge_down {
                    self.edges[e].upper - self.edges[e].flow
                } else {
                    self.edges[e].flow
                };
                if flow_delta > d {
                    flow_delta = d;
                    u_out = u;
                    side = OutArcSide::Source;
                }
                u = self.nodes[u].parent;
            }

            let mut u = dest;
            while u != join && (flow_delta != 0 || side != OutArcSide::Target) {
                let e = self.nodes[u].pred;
                let edge_up = u == self.edges[e].nodes[0];
                let d = if edge_up {
                    self.edges[e].upper - self.edges[e].flow
                } else {
                    self.edges[e].flow
                };
                if flow_delta >= d {
                    flow_delta = d;
                    u_out = u;
                    side = OutArcSide::Target;
                }
                u = self.nodes[u].parent;
            }

            if flow_delta != 0 {
                let delta = self.edges[in_arc].state as Flow * flow_delta;
                self.edges[in_arc].flow += delta;
                let mut u = self.edges[in_arc].nodes[0];
                while u != join {
                    let e = self.nodes[u].pred;
                    self.edges[e].flow += if u == self.edges[e].nodes[0] {
                        -delta
                    } else {
                        delta
                    };
                    u = self.nodes[u].parent;
                }

                let mut u = self.edges[in_arc].nodes[1];
                while u != join {
                    let e = self.nodes[u].pred;
                    self.edges[e].flow += if u == self.edges[e].nodes[0] {
                        delta
                    } else {
                        -delta
                    };
                    u = self.nodes[u].parent;
                }
            }

            if side == OutArcSide::Same {
                self.edges[in_arc].state = self.edges[in_arc].state.reverse();
                return;
            }

            let out_arc = self.nodes[u_out].pred;
            self.edges[in_arc].state = ArcState::Tree;
            self.edges[out_arc].state = if self.edges[out_arc].flow != 0 {
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
                u = self.nodes[u].parent;
            }

            for i in (0..s).rev() {
                let u = self.bfs[i];
                let p = self.nodes[u].parent;
                self.children.erase(p);
                self.children.push_back(u, p);
                self.nodes[p].parent = u;
                self.nodes[p].pred = self.nodes[u].pred;
            }
            self.children.erase(u_in);
            self.children.push_back(v_in, u_in);
            self.nodes[u_in].parent = v_in;
            self.nodes[u_in].pred = in_arc;

            let current_potential = self.reduced_cost(in_arc);
            let potential_delta = if u_in == self.edges[in_arc].nodes[0] {
                -current_potential
            } else {
                current_potential
            };
            debug_assert_ne!(potential_delta, 0);

            self.bfs[0] = u_in;
            let mut s = 1;
            for i in 0.. {
                if !(i < s) {
                    break;
                }

                let u = self.bfs[i];
                self.nodes[u].potential += potential_delta;
                let mut v = self.children.head(u);
                while v != self.children.rep(u) {
                    self.bfs[s] = v;
                    s += 1;
                    v = self.children.next[v];
                }
            }
        }
    }
}

fn reorder(a: i64, b: i64) -> (i64, i64) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn test_interval_overlaps((a1, a2): (i64, i64), (b1, b2): (i64, i64)) -> bool {
    let (a1, a2) = reorder(a1, a2);
    let (b1, b2) = reorder(b1, b2);
    a1.max(b1) <= a2.min(b2)
}

fn signed_area(a: &[i64; 2], b: &[i64; 2], c: &[i64; 2]) -> i64 {
    let db = [b[0] - a[0], b[1] - a[1]];
    let dc = [c[0] - a[0], c[1] - a[1]];
    db[0] * dc[1] - db[1] * dc[0]
}

fn test_seg_inter(p1: [i64; 2], p2: [i64; 2], q1: [i64; 2], q2: [i64; 2]) -> bool {
    // AABB test for fast rejection
    if !test_interval_overlaps((p1[0], p2[0]), (q1[0], q2[0]))
        || !test_interval_overlaps((p1[1], p2[1]), (q1[1], q2[1]))
    {
        return false;
    }

    // Intersection = p1 + t * (p2 - p1) = q1 + s * (q2 - q1),
    // => t (p2 - p1) - s (q2 - q1) + (p1 - q1) = 0
    // => t (p2 - p1) - s (q2 - q1) = q1 - p1
    let pd = [p2[0] - p1[0], p2[1] - p1[1]];
    let qd = [q2[0] - q1[0], q2[1] - q1[1]];
    let r = [q1[0] - p1[0], q1[1] - p1[1]];

    // Solve a linear equation
    let det = -pd[0] * qd[1] + pd[1] * qd[0];
    let mul_det_t = -qd[1] * r[0] + qd[0] * r[1];
    let mul_det_s = -pd[1] * r[0] + pd[0] * r[1];

    match &det.cmp(&0) {
        Ordering::Greater => (0..=det).contains(&mul_det_t) && (0..=det).contains(&mul_det_s),
        Ordering::Less => (det..=0).contains(&mul_det_t) && (det..=0).contains(&mul_det_s),
        Ordering::Equal => signed_area(&[0, 0], &pd, &r) == 0,
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout_buf();

    let n: usize = input.value();
    let ps: Vec<[i64; 2]> = (0..n).map(|_| [input.value(), input.value()]).collect();

    let n_sq = n * n;
    let segments = (0..n_sq)
        .map(|i| (ps[i / n], ps[i % n]))
        .collect::<Vec<_>>();
    let weights: Vec<Vec<i64>> = (0..2)
        .map(|_| (0..n_sq).map(|_| input.value()).collect())
        .collect();
    let total_weight = weights.iter().map(|w| w.iter().sum::<i64>()).sum::<i64>();

    let mut network = network_flow::NetworkSimplex::with_size(n_sq * 2 + 2);
    let supersrc = n_sq * 2;
    let supersink = n_sq * 2 + 1;
    network.set_supply(supersrc, INF_CAP);
    network.set_demand(supersink, INF_CAP);

    let red = |u: usize| u;
    let blue = |u: usize| u + n_sq;

    for u in 0..n_sq {
        network.add_edge([supersrc, red(u)], 0, weights[0][u], 0);
    }
    for u in 0..n_sq {
        network.add_edge([blue(u), supersink], 0, weights[1][u], 0);
    }

    const INF_CAP: i64 = 1 << 32;
    for u in 0..n_sq {
        for v in 0..n_sq {
            let (r1, r2) = segments[u];
            let (b1, b2) = segments[v];
            if test_seg_inter(r1, r2, b1, b2) {
                network.add_edge([red(u), blue(v)], 0, INF_CAP, 0);
            }
        }
    }

    let min_cut = network.min_cost_max_flow();
    let ans = (total_weight - min_cut) / 2;
    writeln!(output, "{}", ans).unwrap();
}
