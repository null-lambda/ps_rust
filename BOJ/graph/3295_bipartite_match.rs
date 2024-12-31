use std::{collections::HashSet, io::Write};

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
        let n = input.value();
        let m = input.value();

        let mut neighbors = vec![vec![]; n];
        for _ in 0..m {
            let u = input.value::<usize>();
            let v = input.value::<usize>();
            neighbors[u].push(v);
        }

        // Bipartite match
        const UNDEFINED: usize = usize::MAX;
        struct DfsState {
            visited: Vec<bool>,
            src: Vec<usize>,
        }

        fn dfs(node: usize, neighbors: &Vec<Vec<usize>>, state: &mut DfsState) -> bool {
            neighbors[node].iter().any(|&target| {
                if state.visited[target] {
                    return false;
                }
                state.visited[target] = true;
                if state.src[target] == UNDEFINED || dfs(state.src[target], neighbors, state) {
                    state.src[target] = node;
                    return true;
                }
                false
            })
        }

        let mut state = DfsState {
            visited: vec![false; n],
            src: vec![UNDEFINED; n],
        };
        let result = (0..n)
            .filter(|&u| {
                state.visited.fill(false);
                dfs(u, &neighbors, &mut state)
            })
            .count();
        writeln!(output, "{}", result).unwrap();
    }
}
