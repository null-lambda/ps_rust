use std::{
    collections::{HashMap, HashSet},
    io::Write,
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

    for _ in 0..input.value() {
        let n: usize = input.value();

        #[derive(Debug, Clone)]
        struct HalfEdge {
            to: u32,
            inv: u32,
            size: u32,
        }

        impl HalfEdge {
            fn new(to: u32, inv: u32) -> Self {
                Self {
                    to,
                    inv,
                    size: u32::MAX,
                }
            }
        }

        let a: Vec<u32> = (0..n).map(|_| input.value()).collect();
        let mut neighbors = vec![vec![]; n];
        for _ in 0..n - 1 {
            let u = input.value::<u32>() - 1;
            let v = input.value::<u32>() - 1;
            let u_idx = neighbors[u as usize].len() as u32;
            let v_idx = neighbors[v as usize].len() as u32;
            neighbors[u as usize].push(HalfEdge::new(v, v_idx));
            neighbors[v as usize].push(HalfEdge::new(u, u_idx));
        }

        fn dfs_size(neighbors: &mut [Vec<HalfEdge>], u: u32, parent: u32) -> u32 {
            let mut res = 1;
            for i in 0..neighbors[u as usize].len() {
                let mut e = &mut neighbors[u as usize][i];
                if e.to == parent {
                    continue;
                }
                if e.size == u32::MAX {
                    let v = e.to;
                    let size = dfs_size(neighbors, v, u);
                    e = &mut neighbors[u as usize][i];
                    e.size = size;
                }
                res += e.size;
            }
            res
        }
        dfs_size(&mut neighbors, 0, 0);
        for u in 0..n {
            for i in 0..neighbors[u].len() {
                let e = &neighbors[u][i];
                if e.size == u32::MAX {
                    let inv_size = neighbors[e.to as usize][e.inv as usize].size;
                    neighbors[u][i].size = n as u32 - inv_size;
                }
            }
        }

        fn dfs_dp(neighbors: &[Vec<HalfEdge>], a: &[u32], u: u32, parent: u32) -> i64 {
            let mut res = 0;
            for e in &neighbors[u as usize] {
                if e.to == parent {
                    continue;
                }
                res += dfs_dp(neighbors, a, e.to, u);
                res += (a[u as usize] ^ a[e.to as usize]) as i64 * e.size as i64;
            }
            res
        }

        let mut res = vec![0; n];
        res[0] = dfs_dp(&mut neighbors, &a, 0 as u32, 0 as u32);

        let mut visited = vec![false; n];
        let mut stack = vec![(0, u32::MAX)];
        visited[0] = true;
        while let Some((u, parent)) = stack.pop() {
            for e in &neighbors[u as usize] {
                if e.to == parent {
                    continue;
                }
                visited[e.to as usize] = true;
                let size_v = e.size;
                let size_u = neighbors[e.to as usize][e.inv as usize].size;
                res[e.to as usize] = res[u as usize]
                    + (a[e.to as usize] ^ a[u as usize]) as i64 * (size_u as i64 - size_v as i64);
                stack.push((e.to, u));
            }
        }

        for u in 0..n {
            write!(output, "{} ", res[u]).unwrap();
        }
        writeln!(output).unwrap();
    }
}
