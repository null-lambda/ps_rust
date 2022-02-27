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
            let idx = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
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

    let n = input.value();
    let n_edges = input.value();
    let mut neighbors: Vec<Vec<_>> = (0..n).map(|_| Vec::new()).collect();
    let mut neighbors_rev: Vec<Vec<_>> = (0..n).map(|_| Vec::new()).collect();
    for _ in 0..n_edges {
        let u: usize = input.value();
        let v: usize = input.value();
        let (u, v) = (u - 1, v - 1);
        neighbors[u].push(v);
        neighbors_rev[v].push(u);
    }

    assert!(n >= 1);

    // kosaraju algorithm
    let mut kosaraju_stack = vec![];
    let mut visited = vec![false; n];
    for start in 0..n {
        if !visited[start] {
            let mut dfs_stack = vec![(start, false)];
            while let Some((u, fin)) = dfs_stack.pop() {
                if fin {
                    kosaraju_stack.push(u);
                } else if !visited[u]  {
                    visited[u] = true;
                    dfs_stack.push((u, true));
                    for &v in neighbors[u].iter().rev() {
                        if !visited[v] {
                            dfs_stack.push((v, false));
                        }
                    }
                } 
            }
        }
    }
    // println!("{:?}", kosaraju_stack);

    let mut scc = Vec::new();
    let mut visited = vec![false; n];
    while let Some(start) = kosaraju_stack.pop() {
        if !visited[start] {
            visited[start] = true;
            let mut dfs_stack = vec![start];
            let mut component = vec![start];
            while let Some(u) = dfs_stack.pop() {
                for &v in &neighbors_rev[u] {
                    if !visited[v] {
                        visited[v] = true;
                        dfs_stack.push(v);
                        component.push(v);
                    }
                }
            }
            scc.push(component);
        }
    }

    for component in &mut scc {
        component.sort_unstable();
    }
    scc.sort_unstable_by_key(|component| component[0]);
    writeln!(output_buf, "{:?}", scc.len()).unwrap();
    for component in scc {
        for u in component {
            write!(output_buf, "{} ", u + 1).unwrap();
        }
        writeln!(output_buf, "-1 ").unwrap();
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
