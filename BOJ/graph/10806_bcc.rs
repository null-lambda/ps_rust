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
    dfs(
        0,
        UNDEFINED,
        &neighbors,
        &mut dfs_order,
        &mut cut_edges,
        &mut order,
    );
    cut_edges
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut neighbors = vec![vec![]; n];
    for _ in 0..m {
        let u: usize = input.value();
        let v: usize = input.value();
        neighbors[u - 1].push(v as u32 - 1);
        neighbors[v - 1].push(u as u32 - 1);
    }

    let mut cut_edges = cut_edges(n, neighbors);
    writeln!(output, "{}", cut_edges.len()).unwrap();
    cut_edges.sort_unstable();
    for (u, v) in cut_edges {
        writeln!(output, "{} {}", u + 1, v + 1).unwrap();
    }
}
