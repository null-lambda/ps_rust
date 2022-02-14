mod io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }

        #[inline]
        fn value<T>(&mut self) -> T
        where
            T: FromStr,
            T::Err: Debug,
        {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }

    #[inline]
    fn is_whitespace(c: u8) -> bool {
        c <= b' '
    }

    fn trim_newline(s: &[u8]) -> &[u8] {
        let mut s = s;
        while s
            .last()
            .map(|&c| matches! { c, b'\n' | b'\r' | 0  })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| !is_whitespace(c))
                .expect("no available tokens left");
            *self = &self[idx..];
            let idx = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(idx);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|idx| idx + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(idx);
            *self = buf_new;
            trim_newline(line)
        }
    }
}

use std::collections::VecDeque;
use std::io::{BufReader, Read, Write};
use std::iter::successors;

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

const EMPTY_NODE: usize = 0;

fn acyclic_graph_to_tree(
    n: usize,
    root_node: usize,
    neighbors: &Vec<Vec<usize>>,
    parent: &mut Vec<usize>,
    rank: &mut Vec<u32>,
) {
    {
        let mut queue: VecDeque<usize> = VecDeque::from(vec![root_node]);
        let mut visited: Vec<bool> = vec![false; n + 1];
        visited[root_node] = true;
        parent[root_node] = 0;

        let mut current_rank = 1;
        while !queue.is_empty() {
            let level_size = queue.len();
            for _ in 0..level_size {
                let u = queue.pop_front().unwrap();
                rank[u] = current_rank;
                for &v in neighbors[u].iter() {
                    if !visited[v] {
                        visited[v] = true;
                        queue.push_back(v);
                        parent[v] = u;
                    }
                }
            }
            current_rank += 1;
        }
    }
}

fn least_common_ancestor(
    parent_table: &[Vec<usize>],
    rank: &[u32],
    mut u: usize,
    mut v: usize,
) -> usize {
    if u == v {
        return u;
    }
    if rank[u] > rank[v] {
        std::mem::swap(&mut u, &mut v);
    }

    let nth_parent = |u, n| {
        (0..)
            .take_while(|j| (1 << j) <= n)
            .filter(|j| (1 << j) & n != 0)
            .fold(u, |u, j| parent_table[j][u])
    };
    if rank[u] < rank[v] {
        v = nth_parent(v, rank[v] - rank[u]);
    } else {
        u = nth_parent(u, rank[u] - rank[v]);
    }

    if u == v {
        return u;
    }
    for i in (0..parent_table.len()).rev() {
        if parent_table[i][u] != parent_table[i][v] {
            u = parent_table[i][u];
            v = parent_table[i][v];
        }
    }
    u = parent_table[0][u];
    u
}

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let n: usize = input.value();
    let mut neighbors: Vec<Vec<_>> = (0..=n).map(|_| Vec::new()).collect();
    for _ in 0..(n - 1) {
        let (u, v): (usize, usize) = (input.value(), input.value());
        neighbors[u].push(v);
        neighbors[v].push(u);
    }

    let mut parent = vec![EMPTY_NODE; n + 1];
    let mut rank: Vec<u32> = vec![0; n + 1];
    acyclic_graph_to_tree(n, 1, &neighbors, &mut parent, &mut rank);
    drop(neighbors);

    // let n_max = *rank.iter().max().unwrap();
    let rank_max: usize = 100_000;
    let log2r_bound = (rank_max as f64).log2().ceil() as usize;

    // sparse table
    // stores next 2^j-th node
    let parent_table: Vec<Vec<usize>> = successors(Some(parent.clone()), |prev_row| {
        prev_row.iter().map(|&u| Some(prev_row[u])).collect()
    })
    .take(log2r_bound + 1)
    .collect();

    let m = input.value();
    for _ in 0..m {
        let (u, v) = (input.value(), input.value());
        let result = least_common_ancestor(&parent_table, &rank, u, v);
        writeln!(output_buf, "{}", result).unwrap();
    }

    std::io::stdout().write(&output_buf[..]).unwrap();
}
