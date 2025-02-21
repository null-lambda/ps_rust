use std::{cmp::Ordering, io::Write};

// 12 1 14
// 0 0
// 0 1
// 1 1
// 1 0
// -2 -2
// -2 2
// 2 2
// 2 -2
// -5 0
// 0 5
// 5 0
// 0 -5

// -3 1

// 1 2 2 3 3 4 4 1
// 5 6 6 7 7 8 8 5
// 9 10 10 11 11 12 12 9
// 5 9
// 6 9

// 7 1 5
// -4 0
// 0 4
// 4 0
// 0 -4
// -1 0
// 1 0
// -1 -1

// 4 1

// 1 2
// 2 3
// 3 4
// 4 1
// 5 6

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

pub mod debug {
    pub fn with(f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
    }
}

pub mod planar_graph {
    pub type T = i128;
    pub const UNSET: u32 = !0;

    pub fn on_lower_half(base: [T; 2], p: [T; 2]) -> bool {
        (p[1], p[0]) < (base[1], base[0])
    }

    pub fn signed_area(p: [T; 2], q: [T; 2], r: [T; 2]) -> T {
        let dq = [q[0] - p[0], q[1] - p[1]];
        let dr = [r[0] - p[0], r[1] - p[1]];
        dq[0] * dr[1] - dq[1] * dr[0]
    }

    pub fn cmp_angle(p: [T; 2], q: [T; 2], r: [T; 2]) -> std::cmp::Ordering {
        on_lower_half(p, q)
            .cmp(&on_lower_half(p, r))
            .then_with(|| 0.cmp(&signed_area(p, q, r)))
    }

    #[derive(Debug, Clone)]
    pub struct HalfEdge<E> {
        pub src: u32,
        pub cycle_next: u32,
        pub weight: E,
    }

    // Doubly connected edge list
    #[derive(Default, Debug, Clone)]
    pub struct DCEL<E> {
        n_verts: usize,
        pub edges: Vec<HalfEdge<E>>,
        pub vert_neighbors: Vec<Vec<u32>>,
        pub init: bool,
    }

    pub fn twin(e: u32) -> u32 {
        e ^ 1
    }

    impl<E> DCEL<E> {
        pub fn push_edge(&mut self, u: usize, v: usize, weight_uv: E, weight_vu: E) -> [u32; 2] {
            assert!(!self.init, "DCEL is already initialized");
            let eu = self.edges.len() as u32;
            self.edges.push(HalfEdge {
                src: u as u32,
                cycle_next: UNSET,
                weight: weight_uv,
            });
            self.edges.push(HalfEdge {
                src: v as u32,
                cycle_next: UNSET,
                weight: weight_vu,
            });
            [eu, eu + 1]
        }

        pub fn cycle_next(&self, e: u32) -> u32 {
            self.edges[e as usize].cycle_next
        }

        pub fn vertex_prev(&self, e: u32) -> u32 {
            self.cycle_next(twin(e))
        }

        pub fn build_topology(&mut self, ps: &[[T; 2]]) {
            assert!(!self.init, "DCEL is already initialized");
            self.init = true;

            self.n_verts = ps.len();
            self.vert_neighbors = vec![vec![]; self.n_verts];

            for he in 0..self.edges.len() as u32 {
                let u = self.edges[he as usize].src as usize;
                self.vert_neighbors[u].push(he);
            }

            for u in 0..self.n_verts {
                self.vert_neighbors[u].sort_unstable_by(|&e, &f| {
                    cmp_angle(
                        ps[u],
                        ps[self.edges[twin(e) as usize].src as usize],
                        ps[self.edges[twin(f) as usize].src as usize],
                    )
                });

                for (&e, &f) in self.vert_neighbors[u]
                    .iter()
                    .zip(self.vert_neighbors[u].iter().cycle().skip(1))
                {
                    self.edges[twin(f) as usize].cycle_next = e;
                }
            }
        }

        pub fn for_each_in_left_face<W>(
            &mut self,
            e_entry: u32,
            mut visitor: impl FnMut(&mut Self, u32) -> Result<(), W>,
        ) -> Result<(), W> {
            let mut e = e_entry;
            loop {
                match visitor(self, e) {
                    Ok(()) => {
                        e = self.cycle_next(e);
                        if e == e_entry {
                            return Ok(());
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Frac128(i128, i128);

fn cmp_frac((a, b): (i128, i128), (c, d): (i128, i128)) -> Ordering {
    assert!(b > 0 && d > 0);
    (a * d).cmp(&(b * c))
}

impl PartialEq for Frac128 {
    fn eq(&self, other: &Self) -> bool {
        self.0 * other.1 == self.1 * other.0
    }
}

impl Eq for Frac128 {}

impl PartialOrd for Frac128 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(cmp_frac((self.0, self.1), (other.0, other.1)))
    }
}

impl Ord for Frac128 {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_frac((self.0, self.1), (other.0, other.1))
    }
}

impl From<i128> for Frac128 {
    fn from(x: i128) -> Self {
        Self(x, 1)
    }
}

fn x_section(y: i128, mut seg: [[i128; 2]; 2]) -> Option<Frac128> {
    if seg[0][1] == seg[1][1] {
        return None;
    }
    if seg[0][1] > seg[1][1] {
        seg.swap(0, 1);
    }
    if !(seg[0][1] <= y && y <= seg[1][1]) {
        return None;
    }

    let dx = seg[1][0] - seg[0][0];
    let dy = seg[1][1] - seg[0][1];
    let numer = (dx * (y - seg[0][1]) + seg[0][0] * dy) as i128;
    let denom = dy as i128;
    let s = denom.signum();
    Some(Frac128(numer * s, denom * s))
}

mod dset {
    use std::{cell::Cell, mem};

    pub struct DisjointSet {
        // Represents parent if >= 0, size if < 0
        parent_or_size: Vec<Cell<i32>>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self {
                parent_or_size: vec![Cell::new(-1); n],
            }
        }

        fn get_parent_or_size(&self, u: usize) -> Result<usize, u32> {
            let x = self.parent_or_size[u].get();
            if x >= 0 {
                Ok(x as usize)
            } else {
                Err((-x) as u32)
            }
        }

        fn set_parent(&self, u: usize, p: usize) {
            self.parent_or_size[u].set(p as i32);
        }

        fn set_size(&self, u: usize, s: u32) {
            self.parent_or_size[u].set(-(s as i32));
        }

        pub fn find_root_with_size(&self, u: usize) -> (usize, u32) {
            match self.get_parent_or_size(u) {
                Ok(p) => {
                    let (root, size) = self.find_root_with_size(p);
                    self.set_parent(u, root);
                    (root, size)
                }
                Err(size) => (u, size),
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            self.find_root_with_size(u).0
        }

        // Returns true if two sets were previously disjoint
        pub fn merge(&mut self, u: usize, v: usize) -> bool {
            let (mut u, size_u) = self.find_root_with_size(u);
            let (mut v, size_v) = self.find_root_with_size(v);
            if u == v {
                return false;
            }

            if size_u < size_v {
                mem::swap(&mut u, &mut v);
            }
            self.set_parent(v, u);
            self.set_size(u, size_u + size_v);
            true
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    // Add a boundary to group multiple connected components
    const X_EXT: i128 = 1e9 as i128 + 10;

    let n: usize = input.value();
    let n_srcs: usize = input.value();
    let n_edges: usize = input.value();
    let ps: Vec<[i128; 2]> = (0..n)
        .map(|_| [input.value(), input.value()])
        .chain([
            [-X_EXT, -X_EXT],
            [-X_EXT, X_EXT],
            [X_EXT, X_EXT],
            [X_EXT, -X_EXT],
        ])
        .collect();
    let srcs: Vec<[i128; 2]> = (0..n_srcs)
        .map(|_| [input.value(), input.value()])
        .collect();
    let edges = (0..n_edges)
        .map(|_| [input.value::<u32>() - 1, input.value::<u32>() - 1])
        .chain(
            [[n, n + 1], [n + 1, n + 2], [n + 2, n + 3], [n + 3, n]]
                .into_iter()
                .map(|t| t.map(|x| x as u32)),
        );
    let n_edges_pad = n_edges + 4;

    let mut graph = planar_graph::DCEL::default();
    for [u, v] in edges {
        graph.push_edge(u as usize, v as usize, (), ());
    }
    graph.build_topology(&ps);

    // O(ME) solution: Do a naive raycast in +-x direction.
    let mut conn_edges = dset::DisjointSet::new(n_edges_pad * 2 + 1);
    let reachable = n_edges_pad * 2;

    let mut visited = vec![false; n_edges_pad * 2];
    for h0 in 0..n_edges_pad as u32 * 2 {
        if visited[h0 as usize] {
            continue;
        }

        // Merge all half-edges on a common face
        let mut prev = h0;
        let mut verts = vec![];
        graph
            .for_each_in_left_face::<()>(h0, |graph, h| {
                visited[h as usize] = true;
                conn_edges.merge(prev as usize, h as usize);
                verts.push(ps[graph.edges[h as usize].src as usize]);

                prev = h;
                Ok(())
            })
            .ok();

        debug::with(|| println!("verts {:?}", verts));
        let right_top = (0..verts.len()).max_by_key(|&i| verts[i]).unwrap();
        let p = verts[right_top];

        // Raycast +x direction from top right point, and merge nested faces.
        // Do it only for an outer face of a connected component.
        let m = verts.len();
        let prev = verts[(right_top + m - 1) % m];
        let next = verts[(right_top + 1) % m];
        if !(planar_graph::signed_area(prev, p, next) <= 0) {
            continue;
        }

        let mut right_end = (Frac128::from(X_EXT as i128 + 10), !0);
        for h in 0..n_edges_pad as u32 * 2 {
            let s0 = ps[graph.edges[h as usize].src as usize];
            let s1 = ps[graph.edges[planar_graph::twin(h) as usize].src as usize];
            let Some(x_frac) = x_section(p[1], [s0, s1]) else {
                continue;
            };
            if s0[1] < s1[1] && Frac128::from(p[0] as i128) < x_frac {
                right_end = right_end.min((x_frac, h));
            }
        }
        let (_, h_right) = right_end;
        if h_right != !0 {
            debug::with(|| println!("conn face {} {}", h0, h_right,));
            conn_edges.merge(h0 as usize, h_right as usize);
        }
    }

    // Raycast +x direction from src points
    for &p in &srcs {
        let mut right_end = (Frac128::from(X_EXT as i128 + 10), !0);
        for h in 0..n_edges_pad as u32 * 2 {
            let s0 = ps[graph.edges[h as usize].src as usize];
            let s1 = ps[graph.edges[planar_graph::twin(h) as usize].src as usize];
            let Some(x_frac) = x_section(p[1], [s0, s1]) else {
                continue;
            };
            if s0[1] < s1[1] && Frac128::from(p[0] as i128) < x_frac {
                right_end = right_end.min((x_frac, h));
            }
        }
        let (_x_frac, h_right) = right_end;
        debug::with(|| println!("reach face {} {:?}", h_right, _x_frac));
        conn_edges.merge(reachable, h_right as usize);
    }

    debug::with(|| {
        for u in 0..n_edges_pad * 2 + 1 {
            print!("{} ", conn_edges.find_root(u));
        }
        println!();
    });

    for e in 0..n_edges {
        let ans = conn_edges.find_root(e * 2) == conn_edges.find_root(reachable)
            || conn_edges.find_root(e * 2 + 1) == conn_edges.find_root(reachable);
        write!(output, "{}", ans as u8).unwrap();
    }
    writeln!(output).unwrap();

    // O((E + M) log E) solution: Do a slab decomposition, offline. (sweep with an ordered set of segments)
    // https://en.wikipedia.org/wiki/Point_location
    // todo!()
}
