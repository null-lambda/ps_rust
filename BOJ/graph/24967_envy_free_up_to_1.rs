use std::io::Write;

use buffered_io::BufReadExt;

mod buffered_io {
    use std::io::{BufRead, BufReader, BufWriter, Stdin, Stdout};
    use std::str::FromStr;

    pub trait BufReadExt: BufRead {
        fn line(&mut self) -> String {
            let mut buf = String::new();
            self.read_line(&mut buf).unwrap();
            buf
        }

        fn skip_line(&mut self) {
            self.line();
        }

        fn token(&mut self) -> String {
            loop {
                let buf = self.fill_buf().unwrap();
                if buf.is_empty() {
                    return String::new();
                }

                let mut i = 0;
                while i < buf.len() && buf[i].is_ascii_whitespace() {
                    i += 1;
                }

                let should_break = i < buf.len();
                self.consume(i);
                if should_break {
                    break;
                }
            }

            let mut res = vec![];
            loop {
                let buf = self.fill_buf().unwrap();
                if buf.is_empty() {
                    break;
                }

                let mut i = 0;
                while i < buf.len() && !buf[i].is_ascii_whitespace() {
                    i += 1;
                }
                res.extend_from_slice(&buf[..i]);

                let should_break = i < buf.len();
                self.consume(i);
                if should_break {
                    break;
                }
            }

            String::from_utf8(res).unwrap()
        }

        fn try_value<T: FromStr>(&mut self) -> Option<T> {
            self.token().parse().ok()
        }

        fn value<T: FromStr>(&mut self) -> T {
            self.try_value().unwrap()
        }
    }

    impl<R: BufRead> BufReadExt for R {}

    pub fn stdin() -> BufReader<Stdin> {
        BufReader::new(std::io::stdin())
    }

    pub fn stdout() -> BufWriter<Stdout> {
        BufWriter::new(std::io::stdout())
    }
}

fn main() {
    let mut input = buffered_io::stdin();
    let mut output = buffered_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();

    // 3-Layer graph hierachy: Members({u}) - Buckets({i}) - Items({t})
    let mut weights = vec![vec![0i64; n]; m];
    for u in 0..n {
        for t in 0..m {
            weights[t][u] = input.value();
        }
    }

    if n >= m {
        for t in 0..m {
            write!(output, "{} ", t + 1).unwrap();
        }
        return;
    }
    assert!(n <= 500);

    // Bipartite matching between Members and Buckets
    let mut assignment: [Vec<u32>; 2] = [(0..n as u32).collect(), (0..n as u32).collect()];

    // Links from items to buckets
    let mut owner = vec![];

    // Valuation
    let mut bucket_sum = vec![vec![0i64; n]; n];

    for t in 0..m {
        loop {
            // Build the envy graph
            let mut adj = vec![vec![false; n]; n];
            let mut has_in = vec![false; n];
            for u in 0..n {
                for j in 0..n {
                    let i = assignment[0][u] as usize;
                    let w = bucket_sum[u][j] - bucket_sum[u][i];
                    if w > 0 {
                        adj[i][j] = true;
                        has_in[j] = true;
                    }
                }
            }

            // Push new item to zero indegree bucket, if exists.
            if let Some(i) = (0..n).find(|&i| !has_in[i]) {
                owner.push(i as u32);
                for u in 0..n {
                    bucket_sum[u][i] += weights[t][u];
                }
                break;
            }

            let mut aug_pairs = vec![];
            {
                const UNSET: u32 = !0;
                const KILLED: u32 = !0 - 1;

                let mut parent = vec![UNSET; n];
                let mut current_child = vec![0; n];

                'find_cycle: for root in 0..n as u32 {
                    if parent[root as usize] != UNSET {
                        continue;
                    }

                    parent[root as usize] = root;

                    let mut i = root;
                    loop {
                        let p = parent[i as usize];
                        let j = current_child[i as usize];
                        current_child[i as usize] += 1;

                        if j == n as u32 {
                            parent[i as usize] = KILLED;

                            if p == i {
                                break;
                            }

                            i = p;
                            continue;
                        }

                        if !adj[i as usize][j as usize] || parent[j as usize] == KILLED {
                            continue;
                        }

                        if parent[j as usize] == UNSET {
                            parent[j as usize] = i;
                            i = j;
                        } else {
                            let mut c = j;
                            let mut b = i;
                            loop {
                                let u = assignment[1][b as usize];
                                aug_pairs.push([u, c]);

                                if b == j {
                                    break;
                                }
                                c = b;
                                b = parent[b as usize];
                            }

                            let mut c = i;
                            loop {
                                let b = parent[c as usize];
                                parent[c as usize] = KILLED;
                                if c == root {
                                    break;
                                }

                                c = b;
                            }
                            break 'find_cycle;
                        }
                    }
                }
            }

            assert!(!aug_pairs.is_empty());

            for [u, c] in aug_pairs {
                assignment[0][u as usize] = c;
                assignment[1][c as usize] = u;
            }
        }
    }

    for t in 0..m {
        write!(output, "{} ", assignment[1][owner[t] as usize] + 1).unwrap();
    }
}
