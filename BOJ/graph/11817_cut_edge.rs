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

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

fn cut_edges(n: usize, neighbors: Vec<Vec<u32>>, mut visit_edge: impl FnMut(u32, u32)) {
    static mut DFS_ORDER: Vec<u32> = vec![];
    static mut ORDER: u32 = 0;
    static mut NEIGHBORS: *const Vec<Vec<u32>> = &vec![];

    unsafe fn dfs(u: u32, parent: u32, visit_edge: &mut impl FnMut(u32, u32)) -> u32 {
        DFS_ORDER[u as usize] = ORDER;
        ORDER += 1;
        let mut low_u = ORDER;
        for &v in &NEIGHBORS.as_ref().unwrap()[u as usize] {
            if parent == v {
                continue;
            }
            if DFS_ORDER[v as usize] != 0 {
                low_u = low_u.min(DFS_ORDER[v as usize]);
            } else {
                let low_v = dfs(v, u, visit_edge);
                if low_v > DFS_ORDER[u as usize] {
                    visit_edge(u, v);
                }
                low_u = low_u.min(low_v);
            }
        }
        low_u
    }

    const UNDEFINED: u32 = i32::MAX as u32;
    unsafe {
        DFS_ORDER = vec![0; n];
        ORDER = 1;
        NEIGHBORS = &neighbors;
        dfs(0, UNDEFINED, &mut visit_edge);
    }
}

fn dijkstra(nieghbors: &Vec<Vec<(u32, u32)>>, start: u32) -> Vec<u32> {
    let mut dist = vec![i32::MAX as u32; nieghbors.len()];
    let mut pq: BinaryHeap<_> = [Reverse((0, start))].into();
    dist[start as usize] = 0;

    while let Some(Reverse((d, d_u))) = pq.pop() {
        if dist[d_u as usize] < d {
            continue;
        }
        for &(v, d_uv) in &nieghbors[d_u as usize] {
            let d_v_new = d + d_uv;
            if dist[v as usize] > d_v_new {
                dist[v as usize] = d_v_new;
                pq.push(Reverse((d_v_new, v)));
            }
        }
    }
    dist
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut neighbors = vec![vec![]; n];
    let mut neighbors_weighted = vec![vec![]; n];
    for _ in 0..m {
        let u: usize = input.value();
        let v: usize = input.value();
        let d_uv: u32 = input.value();
        neighbors[u - 1].push(v as u32 - 1);
        neighbors[v - 1].push(u as u32 - 1);
        neighbors_weighted[u - 1].push((v as u32 - 1, d_uv));
        neighbors_weighted[v - 1].push((u as u32 - 1, d_uv));
    }
    let s1 = input.value::<usize>() - 1;
    let s2 = input.value::<usize>() - 1;
    let d1 = dijkstra(&neighbors_weighted, s1 as u32);
    let d2 = dijkstra(&neighbors_weighted, s2 as u32);

    let mut ans = u32::MAX;
    cut_edges(n, neighbors, |u, v| {
        ans = ans.min(d1[u as usize].max(d2[v as usize]));
        ans = ans.min(d1[v as usize].max(d2[u as usize]));
    });
    writeln!(output, "{}", ans).unwrap();
}
