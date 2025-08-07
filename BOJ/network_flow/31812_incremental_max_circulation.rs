use std::io::Write;

mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

pub mod jagged {
    pub type EdgeId = u32;
    const UNSET: EdgeId = u32::MAX;

    // Forward-star representation (linked lists) for incremental jagged array
    #[derive(Clone, PartialEq, Eq)]
    pub struct FS<T> {
        pub head: Vec<EdgeId>,
        pub links: Vec<(EdgeId, T)>,
    }

    pub struct RowIter<'a, T> {
        owner: &'a FS<T>,
        e: u32,
    }

    impl<'a, T> Iterator for RowIter<'a, T> {
        type Item = &'a T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.e == UNSET {
                return None;
            }
            let (e_next, value) = &self.owner.links[self.e as usize];
            self.e = *e_next;
            Some(value)
        }
    }

    impl<T> FS<T> {
        pub fn with_size(n: usize) -> Self {
            Self {
                head: vec![UNSET; n],
                links: vec![],
            }
        }

        pub fn insert(&mut self, u: usize, v: T) -> EdgeId {
            let e = self.links.len() as u32;
            self.links.push((self.head[u], v));
            self.head[u] = e;
            e
        }

        pub fn iter_row<'a>(&'a self, u: usize) -> RowIter<'a, T> {
            RowIter {
                owner: &self,
                e: self.head[u],
            }
        }
    }
}

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
            assert!(cap >= 0);
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

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for _ in 0..input.value() {
        let n: usize = input.value();
        let m: usize = input.value();

        let s = (n + m) as u32;
        let t = s + 1;
        let cs = s + 2;
        let ct = s + 3;
        let mut net = ford_fulkerson::MaxFlow::empty(n + m + 4);

        let robot = |u: u32| u;
        let building = |v: u32| v + n as u32;

        let mut dynamic_edges = vec![];
        for u in 0..n as u32 {
            dynamic_edges.push(net.link(s, robot(u), 0));
        }

        for u in 0..n as u32 {
            for _ in 0..input.value() {
                let v = input.value::<u32>() - 1;
                net.link(robot(u), building(v), 1);
            }
        }

        let mut l_sum = 0;
        for v in 0..m as u32 {
            let l: i32 = input.value();
            let h: i32 = input.value();
            l_sum += l;
            net.link(building(v), t, h - l);
            net.link(building(v), ct, l);
        }

        net.link(t, s, ford_fulkerson::INF_FLOW);
        net.link(cs, t, l_sum);

        let mut feasible = false;
        for b in 1..=m as i32 {
            for &e in &dynamic_edges {
                net.increment_cap(e, b);
            }

            let mut ans = -1;
            feasible = feasible || net.run(cs, ct) >= l_sum;
            if feasible {
                ans = net.run(s, t) - l_sum;
            }

            write!(output, "{} ", ans).unwrap();
        }
        writeln!(output).unwrap();
    }
}
