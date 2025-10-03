use std::{collections::BinaryHeap, io::Write};

mod simple_io {
    pub struct InputAtOnce {
        iter: std::str::SplitAsciiWhitespace<'static>,
    }

    impl InputAtOnce {
        pub fn token(&mut self) -> &'static str {
            self.iter.next().unwrap_or_default()
        }

        pub fn try_value<T: std::str::FromStr>(&mut self) -> Option<T> {
            self.token().parse().ok()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.try_value().unwrap()
        }
    }

    pub fn stdin() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(Box::new(buf));
        let iter = buf.split_ascii_whitespace();
        InputAtOnce { iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

mod dset {
    use std::cell::Cell;

    #[derive(Clone)]
    pub struct DisjointForest {
        // Represents parent if >= 0, size if < 0
        link: Vec<Cell<i32>>,
    }

    impl DisjointForest {
        pub fn new(n: usize) -> Self {
            Self {
                link: vec![Cell::new(-1); n],
            }
        }

        pub fn find_root(&self, u: u32) -> u32 {
            let p = self.link[u as usize].get();
            if p < 0 {
                return u;
            }

            let root = self.find_root(p as u32);
            self.link[u as usize].set(root as i32);
            root
        }

        // Returns true iif two sets were previously disjoint
        pub fn merge(&mut self, mut u: u32, mut p: u32) -> bool {
            u = self.find_root(u);
            p = self.find_root(p);
            if p == u {
                return false;
            }

            self.link[u as usize].set(p as i32);
            true
        }
    }
}

fn xor_traversal(
    mut degree: Vec<u32>,
    mut xor_neighbors: Vec<u32>,
    root: u32,
) -> (Vec<u32>, Vec<u32>) {
    let n = degree.len();
    degree[root as usize] += 2;

    let mut toposort = vec![];

    for mut u in 0..n as u32 {
        while degree[u as usize] == 1 {
            let p = xor_neighbors[u as usize];
            xor_neighbors[p as usize] ^= u;
            degree[u as usize] -= 1;
            degree[p as usize] -= 1;

            toposort.push(u);

            u = p;
        }
    }
    toposort.push(root);

    let mut parent = xor_neighbors;
    parent[root as usize] = root;
    (toposort, parent)
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for _ in 0..input.value() {
        let n: usize = input.value();
        let root = 0;
        let mut weights: Vec<[i64; 2]> = std::iter::once([0, 0])
            .chain((1..n).map(|_| std::array::from_fn(|_| input.value())))
            .collect::<Vec<_>>();

        let mut degree = vec![0u32; n];
        let mut xor_neighbors = vec![0u32; n];
        for _ in 0..n - 1 {
            let u = input.value::<u32>() - 1;
            let v = input.value::<u32>() - 1;
            degree[u as usize] += 1;
            degree[v as usize] += 1;
            xor_neighbors[u as usize] ^= v;
            xor_neighbors[v as usize] ^= u;
        }
        let (_, parent) = xor_traversal(degree, xor_neighbors, root);

        const SEP: i64 = 1 << 60;
        let key = |[a, b]: [i64; 2]| {
            if a <= b { SEP - a } else { -(SEP - b) }
        };

        let mut visited = vec![false; n];
        visited[root as usize] = true;

        let mut pq: BinaryHeap<_> = (0..n as u32)
            .map(|u| (key(weights[u as usize]), u))
            .collect();
        let mut conn = dset::DisjointForest::new(n);
        while let Some((_, u)) = pq.pop() {
            if visited[u as usize] {
                continue;
            }
            visited[u as usize] = true;

            let p = parent[u as usize];
            let [a0, b0] = weights[conn.find_root(p) as usize];
            let [a1, b1] = weights[conn.find_root(u) as usize];

            conn.merge(u, p);

            let a = a0 + 0.max(a1 - b0);
            let b = a + b0 - a0 + b1 - a1;
            weights[conn.find_root(p) as usize] = [a, b];

            pq.push((key([a, b]), conn.find_root(p) as u32));
        }

        let ans = weights[conn.find_root(root) as usize][0];
        writeln!(output, "{}", ans).unwrap();
    }
}
