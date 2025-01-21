use std::{cmp::Reverse, collections::BinaryHeap, io::Write};

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

fn on_lower_half(base: [i64; 2], p: [i64; 2]) -> bool {
    (p[1], p[0]) < (base[1], base[0])
}

fn signed_area(p: [i64; 2], q: [i64; 2], r: [i64; 2]) -> i64 {
    let dq = [q[0] - p[0], q[1] - p[1]];
    let dr = [r[0] - p[0], r[1] - p[1]];
    dq[0] * dr[1] - dq[1] * dr[0]
}

fn cmp_angle(p: [i64; 2], q: [i64; 2], r: [i64; 2]) -> std::cmp::Ordering {
    on_lower_half(p, q)
        .cmp(&on_lower_half(p, r))
        .then_with(|| 0.cmp(&signed_area(p, q, r)))
}

type E = i64;

const UNSET: u32 = !0;
const INF: E = 1 << 56;

#[derive(Debug, Clone)]
struct HalfEdge {
    src: u32,
    cycle_next: u32,
    left_face: u32,
    weight: E,
}

#[derive(Debug, Clone)]
struct Face {
    entry_edge: u32,
    neighbors: Vec<(u32, E)>,
}

fn twin(e: u32) -> u32 {
    e ^ 1
}

fn dijkstra<C>(n: usize, neighbors: impl Fn(u32) -> C, src: u32) -> Vec<E>
where
    C: IntoIterator<Item = (u32, E)>,
{
    let mut dist = vec![INF; n];
    let mut queue: BinaryHeap<_> = [(Reverse(0), src)].into();
    dist[src as usize] = 0;

    while let Some((Reverse(d), u)) = queue.pop() {
        // assert!(u < n as u32);
        if dist[u as usize] < d {
            continue;
        }
        for (v, w) in neighbors(u) {
            let d_new = d + w;
            if d_new < dist[v as usize] {
                dist[v as usize] = d_new;
                queue.push((Reverse(d_new), v));
            }
        }
    }

    dist
}

const X_BOUND: i64 = 10_000_000;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut ps: Vec<[i64; 2]> = (0..n).map(|_| [input.value(), input.value()]).collect();

    // Add some points at (pseudo-)infinity, to embed graph in a cylinder
    ps.push([-X_BOUND, 0]);
    ps.push([X_BOUND, 0]);
    let u_end = [n as u32, n as u32 + 1];

    // Construct planar graph
    let m: usize = input.value();
    let mut edges = vec![];
    let mut vert_neighbors = vec![vec![]; n + 2];
    let mut add_edge = |e, u, v, d| {
        edges.push(HalfEdge {
            src: u,
            cycle_next: UNSET,
            left_face: UNSET,
            weight: d,
        });
        edges.push(HalfEdge {
            src: v,
            cycle_next: UNSET,
            left_face: UNSET,
            weight: d,
        });
        vert_neighbors[u as usize].push(2 * e);
        vert_neighbors[v as usize].push(2 * e + 1);
    };
    for i in 0..m as u32 {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        let d: i64 = input.value();
        add_edge(i, u, v, d);
    }

    // Make the graph cylindrical
    add_edge(m as u32, u_end[0], 0, INF);
    add_edge(m as u32 + 1, n as u32 - 1, u_end[1], INF);

    // Connect adjacent edges in a cycle
    for u in 0..n + 2 {
        vert_neighbors[u].sort_unstable_by(|&e, &f| {
            cmp_angle(
                ps[u],
                ps[edges[twin(e) as usize].src as usize],
                ps[edges[twin(f) as usize].src as usize],
            )
        });

        for (&e, &f) in vert_neighbors[u]
            .iter()
            .zip(vert_neighbors[u].iter().cycle().skip(1))
        {
            edges[twin(f) as usize].cycle_next = e;
        }
    }
    edges[2 * m + 1].cycle_next = 2 * m as u32 + 3;
    edges[2 * m + 2].cycle_next = 2 * m as u32;

    // Convert all cycles to faces
    let mut faces = vec![];
    for mut e in 0..(2 * m + 4) as u32 {
        if edges[e as usize].left_face != UNSET {
            continue;
        }

        let face_idx = faces.len() as u32;
        let face = Face {
            entry_edge: e,
            neighbors: vec![],
        };
        loop {
            edges[e as usize].left_face = face_idx;
            e = edges[e as usize].cycle_next;
            if e == face.entry_edge {
                break;
            }
        }
        faces.push(face);
    }

    // Construct the dual graph
    let mut c_top = UNSET;
    let mut c_bot = UNSET;
    for (ic, c) in faces.iter_mut().enumerate() {
        let mut e = c.entry_edge;
        loop {
            // Detect two outer faces
            if e == 2 * m as u32 {
                c_top = ic as u32;
            } else if e == 2 * m as u32 + 1 {
                c_bot = ic as u32;
            }

            c.neighbors
                .push((edges[twin(e) as usize].left_face, edges[e as usize].weight));
            e = edges[e as usize].cycle_next;
            if e == c.entry_edge {
                break;
            }
        }
    }

    // Shortest path on the original graph
    let min_cost = {
        let d_src = dijkstra(
            n + 2,
            |u| {
                vert_neighbors[u as usize]
                    .iter()
                    .map(|&e| (edges[twin(e) as usize].src, edges[e as usize].weight))
            },
            0,
        );
        d_src[n - 1]
    };

    // Dijkstra on the dual graph
    let max_cost = {
        let d_dual_top = dijkstra(
            faces.len(),
            |u| faces[u as usize].neighbors.iter().copied(),
            c_top,
        );
        let d_dual_bot = dijkstra(
            faces.len(),
            |u| faces[u as usize].neighbors.iter().copied(),
            c_bot,
        );

        let mut sum = 0;
        let mut res = INF;
        for u in 0..faces.len() as u32 {
            for &(v, w) in &faces[u as usize].neighbors {
                if w == INF {
                    continue;
                }
                sum += w;
                res = res.min(d_dual_top[u as usize] + d_dual_bot[v as usize]);
            }
        }
        sum /= 2;
        res = sum - res;
        res
    };

    writeln!(output, "{} {}", min_cost, max_cost).unwrap();
}
