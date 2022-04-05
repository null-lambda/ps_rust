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

    #[derive(Copy, Clone, Debug)]
    struct MinMax {
        min: u32,
        max: u32,
    }
    impl MinMax {
        fn new(value: u32) -> Self {
            MinMax {
                min: value,
                max: value,
            }
        }
        fn id() -> Self {
            MinMax {
                min: u32::MAX,
                max: u32::MIN,
            }
        }
        fn op(self, other: Self) -> Self {
            MinMax {
                min: self.min.min(other.min),
                max: self.max.max(other.max),
            }
        }
    }

    #[derive(Debug)]
    struct LCA {
        parent: Vec<u32>,
        parent_table: Vec<Vec<(u32, MinMax)>>,
        depth: Vec<u32>,
        dist: Vec<MinMax>,
    }

    impl LCA {
        fn build(n: usize, neighbors: &Vec<Vec<(usize, u32)>>) -> Self {
            use std::iter::successors;

            let mut lca = LCA {
                parent: vec![UNDEFINED; n],
                parent_table: vec![],
                depth: vec![0; n],
                dist: vec![MinMax::id(); n],
            };
            lca.parent[0] = 0;
            lca.build_dfs(&neighbors, 0);

            let max_depth = lca.depth.iter().copied().max().unwrap() + 1;
            let log2_bound = (0..).take_while(|i| 1 << i <= max_depth).last().unwrap() + 1;
            lca.parent_table = successors(
                Some(
                    lca.parent
                        .iter()
                        .copied()
                        .zip(lca.dist.iter().cloned())
                        .collect::<Vec<_>>(),
                ),
                |prev_row| {
                    Some(
                        prev_row
                            .iter()
                            .map(|&(u, d)| {
                                let (p, dp) = prev_row[u as usize];
                                (p, dp.op(d))
                            })
                            .collect(),
                    )
                },
            )
            .take(log2_bound)
            .collect();

            lca
        }

        fn build_dfs(&mut self, neighbors: &Vec<Vec<(usize, u32)>>, u: usize) {
            for &(v, d_uv) in &neighbors[u] {
                if self.parent[v] == UNDEFINED {
                    self.parent[v] = u as u32;
                    self.dist[v] = MinMax::new(d_uv);
                    self.depth[v] = self.depth[u] + 1;
                    self.build_dfs(neighbors, v);
                }
            }
        }

        fn ascend(parent: &Vec<(u32, MinMax)>, u: &mut usize, d_acc: &mut MinMax) {
            let (p, d) = parent[*u];
            *u = p as usize;
            *d_acc = (*d_acc).op(d);
        }

        fn ascend_parent_nth(&self, u: &mut usize, d_acc: &mut MinMax, n: u32) {
            for (j, parent_pow2j) in self.parent_table.iter().enumerate() {
                if (1 << j) & n != 0 {
                    Self::ascend(parent_pow2j, u, d_acc);
                }
            }
        }

        fn get(&self, mut u: usize, mut v: usize) -> (usize, MinMax) {
            let mut d_acc = MinMax::id();
            if self.depth[u] < self.depth[v] {
                let diff = self.depth[v] - self.depth[u];
                self.ascend_parent_nth(&mut v, &mut d_acc, diff);
            } else {
                let diff = self.depth[u] - self.depth[v];
                self.ascend_parent_nth(&mut u, &mut d_acc, diff);
            }
            if u == v {
                return (u, d_acc);
            }

            for parent_pow2j in self.parent_table.iter().rev() {
                if parent_pow2j[u].0 != parent_pow2j[v].0 {
                    Self::ascend(parent_pow2j, &mut u, &mut d_acc);
                    Self::ascend(parent_pow2j, &mut v, &mut d_acc);
                }
            }
            Self::ascend(&self.parent_table[0], &mut u, &mut d_acc);
            Self::ascend(&self.parent_table[0], &mut v, &mut d_acc);
            (u, d_acc)
        }
    }

    let lca = LCA::build(n, &neighbors);

    let m = input.value();
    for _ in 0..m {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        let (_, MinMax { min, max }) = lca.get(u, v);
        writeln!(output_buf, "{} {}", min, max).unwrap();
    }

    std::io::stdout().write(&output_buf[..]).unwrap();
}
