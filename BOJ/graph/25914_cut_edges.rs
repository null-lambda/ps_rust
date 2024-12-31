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

        pub fn try_value<T: std::str::FromStr>(&mut self) -> Option<T>
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().ok()
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

fn cut_edges(n: usize, neighbors: Vec<Vec<u32>>) -> Vec<(u32, u32)> {
    let mut dfs_order = vec![0; n];
    let mut cut_edges = vec![];
    let mut order = 1;
    fn dfs(
        u: u32,
        parent: u32,
        neighbors: &Vec<Vec<u32>>,
        dfs_order: &mut Vec<u32>,
        cut_edges: &mut Vec<(u32, u32)>,
        order: &mut u32,
    ) -> u32 {
        dfs_order[u as usize] = *order;
        *order += 1;
        let mut low_u = *order;
        for &v in &neighbors[u as usize] {
            if parent == v {
                continue;
            }
            if dfs_order[v as usize] != 0 {
                low_u = low_u.min(dfs_order[v as usize]);
            } else {
                let low_v = dfs(v, u, neighbors, dfs_order, cut_edges, order);
                if low_v > dfs_order[u as usize] {
                    cut_edges.push((u.min(v), u.max(v)));
                }
                low_u = low_u.min(low_v);
            }
        }
        low_u
    }

    const UNDEFINED: u32 = i32::MAX as u32;
    for u in 0..n {
        if dfs_order[u] == 0 {
            dfs(
                u as u32,
                UNDEFINED,
                &neighbors,
                &mut dfs_order,
                &mut cut_edges,
                &mut order,
            );
        }
    }
    cut_edges
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let k: usize = input.value();
    let mut neighbors_orig = vec![vec![]; n];
    for _ in 0..m {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        neighbors_orig[u as usize].push(v);
        neighbors_orig[v as usize].push(u);
    }

    let mut has_salmon = vec![false; n];
    for _ in 0..k {
        let i = input.value::<usize>() - 1;
        has_salmon[i] = true;
    }

    let c: u32 = input.value();

    let mut neighbors = vec![vec![]; n];
    for (u, v) in cut_edges(n, neighbors_orig) {
        neighbors[u as usize].push(v);
        neighbors[v as usize].push(u);
    }

    let mut visited = vec![false; n];
    let mut grundy_acc = 0u32;
    for u in 0..n {
        if visited[u] {
            continue;
        }
        visited[u] = true;
        let mut stack = vec![u as u32];
        let mut n_blank = 0;
        let mut n_salmon = 0;
        while let Some(u) = stack.pop() {
            n_blank += !has_salmon[u as usize] as u32;
            n_salmon += has_salmon[u as usize] as u32;
            for &v in &neighbors[u as usize] {
                if !visited[v as usize] {
                    visited[v as usize] = true;
                    stack.push(v);
                }
            }
        }

        let grundy = if n_salmon == 0 { 0 } else { n_blank };
        grundy_acc ^= grundy;
    }

    let ans = (grundy_acc > 0) ^ (c == 0);
    writeln!(output, "{}", ans as u8).unwrap();
}
