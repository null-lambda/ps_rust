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
            .map(|&c| matches!(c, b'\n' | b'\r' | 0))
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let i = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
            *self = &self[i..];
            let i = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(i);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let i = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|i| i + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(i);
            *self = buf_new;
            trim_newline(line)
        }
    }
}

use std::io::{BufReader, Read, Write};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let n: usize = input.value();
    let mut neighbors: Vec<Vec<_>> = (0..n).map(|_| vec![]).collect();
    for _ in 0..n - 1 {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        let dist: u32 = input.value();
        neighbors[u].push((v, dist));
        neighbors[v].push((u, dist));
    }

    const UNDEFINED: u32 = u32::MAX;

    #[derive(Debug)]
    struct LCA {
        parent: Vec<u32>,
        parent_table: Vec<Vec<u32>>,
        depth: Vec<u32>,
        dist: Vec<u32>,
    }

    impl LCA {
        fn build(n: usize, neighbors: &Vec<Vec<(usize, u32)>>) -> Self {
            use std::iter::successors;

            let mut lca = LCA {
                parent: vec![UNDEFINED; n],
                parent_table: vec![],
                depth: vec![0; n],
                dist: vec![0; n],
            };
            lca.parent[0] = 0;
            lca.build_dfs(&neighbors, 0);

            let max_depth = lca.depth.iter().copied().max().unwrap() + 1;
            let log2_bound = (0..).take_while(|i| 1 << i <= max_depth).last().unwrap() + 1;
            lca.parent_table = successors(Some(lca.parent.clone()), |prev_row| {
                Some(prev_row.iter().map(|&u| prev_row[u as usize]).collect())
            })
            .take(log2_bound)
            .collect();
            lca
        }

        fn build_dfs(&mut self, neighbors: &Vec<Vec<(usize, u32)>>, u: usize) {
            for &(v, d_uv) in &neighbors[u] {
                if self.parent[v] == UNDEFINED {
                    self.parent[v] = u as u32;
                    self.dist[v] = self.dist[u] + d_uv;
                    self.depth[v] = self.depth[u] + 1;
                    self.build_dfs(neighbors, v);
                }
            }
        }

        fn parent_nth(&self, mut u: usize, n: u32) -> usize {
            for (j, parent_pow2j) in self.parent_table.iter().enumerate() {
                if (1 << j) & n != 0 {
                    u = parent_pow2j[u] as usize;
                }
            }
            u
        }

        fn get(&self, mut u: usize, mut v: usize) -> usize {
            if self.depth[u] < self.depth[v] {
                v = self.parent_nth(v, self.depth[v] - self.depth[u]);
            } else {
                u = self.parent_nth(u, self.depth[u] - self.depth[v]);
            }
            if u == v {
                return u;
            }

            for parent_pow2j in self.parent_table.iter().rev() {
                if parent_pow2j[u] != parent_pow2j[v] {
                    u = parent_pow2j[u] as usize;
                    v = parent_pow2j[v] as usize;
                }
            }
            u = self.parent_table[0][u] as usize;
            u
        }

        fn get_dist(&self, u: usize, v: usize) -> u32 {
            let p = self.get(u, v);
            self.dist[u] + self.dist[v] - 2 * self.dist[p]
        }
    }

    let lca = LCA::build(n, &neighbors);

    let m = input.value();
    for _ in 0..m {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        writeln!(output_buf, "{}", lca.get_dist(u, v)).unwrap();
    }

    std::io::stdout().write(&output_buf[..]).unwrap();
}
