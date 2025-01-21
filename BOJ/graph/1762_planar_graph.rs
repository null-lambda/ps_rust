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

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut neighbors = vec![vec![]; n];
    let mut degree = vec![0u32; n];
    for _ in 0..m {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        neighbors[u as usize].push(v);
        neighbors[v as usize].push(u);
        degree[u as usize] += 1;
        degree[v as usize] += 1;
    }
    let mut visited = vec![false; n];
    let mut queue: BinaryHeap<_> = (0..n).map(|u| (Reverse(degree[u]), u as u32)).collect();

    let mut n_triangle = 0u32;
    while let Some((_, u)) = queue.pop() {
        if visited[u as usize] {
            continue;
        }
        visited[u as usize] = true;
        let neighbors_u: Box<[_]> = neighbors[u as usize]
            .iter()
            .copied()
            .filter(|&v| !visited[v as usize])
            .collect();

        for iv in 0..neighbors_u.len() {
            for iw in iv + 1..neighbors_u.len() {
                let v = neighbors_u[iv];
                let w = neighbors_u[iw];
                if neighbors[v as usize].contains(&w) {
                    n_triangle += 1;
                    degree[v as usize] -= 1;
                    degree[w as usize] -= 1;
                }
            }
        }

        for v in neighbors_u.iter().copied() {
            queue.push((Reverse(degree[v as usize]), v));
        }
    }
    writeln!(output, "{}", n_triangle).unwrap();
}
