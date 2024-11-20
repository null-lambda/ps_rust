use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    io::Write,
    iter,
};

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

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let _k: usize = input.value();

    let mut parent: Vec<Vec<(u32, u64, u32)>> = vec![vec![]; n];
    let mut n_branches_per_color: Vec<HashMap<u32, u32>> = vec![HashMap::new(); n];

    for _ in 0..m {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        let d: u64 = input.value();
        for _ in 0..input.value() {
            let color = input.value::<u32>() - 1;
            parent[v as usize].push((u, d, color));
            *n_branches_per_color[u as usize].entry(color).or_default() += 1;
        }
    }

    // Dijkstra on a reversed graph
    let src = n - 1;
    let dst = 0;
    let inf = u64::MAX;
    let mut dist = vec![inf; n];
    dist[src] = 0;
    let mut dist_per_color: Vec<HashMap<u32, u64>> = vec![HashMap::new(); n];
    let mut pq: BinaryHeap<_> = [(Reverse(0), src as u32)].into();
    while let Some((Reverse(d_u), u)) = pq.pop() {
        if dist[u as usize] < d_u {
            continue;
        }
        for &(v, w, color) in &parent[u as usize] {
            let d_new = dist[u as usize] + w;
            let d = dist_per_color[v as usize].entry(color).or_default();
            *d = (*d).max(d_new);
            let n_b = n_branches_per_color[v as usize].get_mut(&color).unwrap();
            *n_b -= 1;
            if *n_b == 0 && *d < dist[v as usize] {
                dist[v as usize] = *d;
                pq.push((Reverse(*d), v));
            }
        }
    }
    let ans = dist[dst];
    if ans == inf {
        writeln!(output, "impossible").unwrap();
    } else {
        writeln!(output, "{}", ans).unwrap();
    }
}
