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
    let energies: Vec<u32> = (0..n).map(|_| input.value()).collect();
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
    struct DfsState {
        parent: Vec<u32>,
        dist: Vec<u32>,
    }

    impl DfsState {
        fn build(n: usize, neighbors: &Vec<Vec<(usize, u32)>>) -> Self {
            let mut state = DfsState {
                parent: vec![UNDEFINED; n],
                dist: vec![0; n],
            };
            state.parent[0] = 0;
            state.build_dfs(&neighbors, 0);
            state
        }

        fn build_dfs(&mut self, neighbors: &Vec<Vec<(usize, u32)>>, u: usize) {
            for &(v, d_uv) in &neighbors[u] {
                if self.parent[v] == UNDEFINED {
                    self.parent[v] = u as u32;
                    self.dist[v] = d_uv;
                    self.build_dfs(neighbors, v);
                }
            }
        }
    }

    let DfsState { parent, dist } = DfsState::build(n, &neighbors);

    use std::iter::successors;
    let max_depth = 100_000;
    let log2_bound = (0..).take_while(|i| 1 << i <= max_depth).last().unwrap() + 1;
    let (parent_table, dist_table): (Vec<Vec<u32>>, Vec<Vec<u32>>) =
        successors(Some((parent, dist)), |(parent_nth, dist_nth)| {
            Some(
                parent_nth
                    .iter()
                    .zip(dist_nth)
                    .map(|(&p, &d)| (parent_nth[p as usize], d + dist_nth[p as usize]))
                    .unzip(),
            )
        })
        .take(log2_bound)
        .unzip();

    for (start, mut energy) in energies.into_iter().enumerate() {
        let mut u = start;
        for (parent_nth, dist_nth) in parent_table.iter().zip(&dist_table).rev() {
            if dist_nth[u] <= energy {
                energy -= dist_nth[u];
                u = parent_nth[u] as usize;
                if u == 0 {
                    break;
                }
            }
        }
        writeln!(output_buf, "{}", u + 1).unwrap();
    }

    std::io::stdout().write(&output_buf[..]).unwrap();
}
