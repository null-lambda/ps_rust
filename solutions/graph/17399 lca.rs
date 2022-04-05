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
            .map(|&c| {
                matches! {c, b'\n' | b'\r' | 0}
            })
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
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    use std::iter::successors;

    let n: usize = input.value();
    let mut neighbors: Vec<Vec<usize>> = (0..=n).map(|_| Vec::new()).collect();
    for _ in 0..n - 1 {
        let u: usize = input.value();
        let v: usize = input.value();
        neighbors[u].push(v);
        neighbors[v].push(u);
    }

    #[derive(Debug)]
    struct LCA {
        parent_table: Vec<Vec<u32>>,
        depth: Vec<u32>,
    }

    impl LCA {
        fn build(n: usize, neighbors: &[Vec<usize>]) -> Self {
            let mut parent = vec![0; n + 1];
            let mut depth = vec![0; n + 1];
            {
                fn dfs(
                    neighbors: &[Vec<usize>],
                    parent: &mut [u32],
                    depth: &mut [u32],
                    u: usize,
                ) {
                    for &v in &neighbors[u] {
                        if parent[v] == 0 {
                            parent[v] = u as u32;
                            depth[v] = depth[u] + 1;
                            dfs(neighbors, parent, depth, v);
                        }
                    }
                }
                parent[1] = 1;
                depth[1] = 1;
                dfs(&neighbors, &mut parent, &mut depth, 1);
                parent[1] = 0;
            }

            let log2n_bound = (n as f64).log2().ceil() as usize;
            let parent_table = successors(Some(parent), |prev_row| {
                prev_row.iter().map(|&u| Some(prev_row[u as usize])).collect()
            })
            .take(log2n_bound)
            .collect();

            Self {
                parent_table,
                depth,
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
            if self.depth[u] > self.depth[v] {
                std::mem::swap(&mut u, &mut v);
            }

            v = self.parent_nth(v, self.depth[v] - self.depth[u]);
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
    }

    let lca = LCA::build(n, &neighbors);

    let n_queries: usize = input.value();
    for _ in 0..n_queries {
        let a: usize = input.value();
        let b: usize = input.value();
        let c: usize = input.value();
        let uab = lca.get(a, b);
        let ubc = lca.get(b, c);
        let uac = lca.get(a, c);
        assert!(uab == ubc || uab == uac || ubc == uac);
        let (a, b, c, u, m) = if uab == ubc {
            (a, c, b, uac, uab)
        } else if uab == uac {
            (b, c, a, ubc, uab)
        } else {
            (a, b, c, uab, uac)
        };

        let dua = lca.depth[a] - lca.depth[u];
        let dub = lca.depth[b] - lca.depth[u];
        let dmu = lca.depth[u] - lca.depth[m];
        let dmc = lca.depth[c] - lca.depth[m];
        let duc = dmu + dmc;

        let result = if dua == dub && dub == duc {
            Some(u)
        } else if dua == dub && dua < duc {
            let h = duc - dua;
            (h % 2 == 0).then(|| {
                if h / 2 <= dmu {
                    lca.parent_nth(u, h / 2)
                } else {
                    lca.parent_nth(c, duc - h / 2)
                }
            })
        } else if  dua == duc && dua < dub {
            let h = dub - dua;
            (h % 2 == 0).then(|| {
                lca.parent_nth(b, dub - h / 2)
            })
        } else if dub == duc && dub < dua {
            let h=  dua - dub; 
            (h % 2 == 0).then(|| {
                lca.parent_nth(a, dua - h / 2)
            })
        }
        else {
            None
        };
        writeln!(output_buf, "{}", result.map(|x| x as i32).unwrap_or(-1)).unwrap();
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
