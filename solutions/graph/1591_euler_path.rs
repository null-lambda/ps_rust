use std::{collections::HashMap, io::Write};

mod simple_io {
    pub struct InputAtOnce(std::str::SplitAsciiWhitespace<'static>);

    impl InputAtOnce {
        pub fn token(&mut self) -> &str {
            self.0.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(buf.into_boxed_str());
        InputAtOnce(buf.split_ascii_whitespace())
    }

    pub fn stdout_buf() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout_buf();

    let n: usize = input.value();
    let m: usize = input.value();

    let mut edges = vec![];
    let mut ident_map: HashMap<_, usize> = HashMap::new();
    let mut ident_inv = vec![];
    let mut get_index = |xs: &[i32]| -> usize {
        let idx = ident_map.len();
        *ident_map.entry(xs.to_vec()).or_insert_with(|| {
            ident_inv.push(xs.to_vec());
            idx
        })
    };

    let mut seqs = vec![];
    for _ in 0..n - m + 1 {
        seqs.push((0..m).map(|_| input.value::<i32>()).collect());
        let xs: &Vec<i32> = &seqs.last().unwrap();

        let u = get_index(&xs[..m - 1]);
        let v = get_index(&xs[1..]);
        edges.push((u, v));
    }

    let n_verts = ident_map.len();
    let mut neighbors = vec![vec![]; n_verts];

    let mut indegree = vec![0; n_verts];
    let mut outdegree = vec![0; n_verts];
    for (u, v) in edges {
        neighbors[u].push(v);
        indegree[v] += 1;
        outdegree[u] += 1;
    }

    let init = (0..n_verts)
        .find(|&u| indegree[u] < outdegree[u])
        .unwrap_or(0);

    // Hierholzer's algorithm
    // Find an euler path in a connected graph (if exists)
    let mut stack = vec![init];
    let mut path = vec![];
    while let Some(&u) = stack.last() {
        if let Some(v) = neighbors[u].pop() {
            stack.push(v);
        } else {
            path.push(u);
            stack.pop();
        }
    }

    path.reverse();

    let ans = ident_inv[path[0]]
        .iter()
        .chain(path[1..].iter().map(|&u| ident_inv[u].last().unwrap()));
    for x in ans {
        write!(output, "{} ", x).unwrap();
    }
}
